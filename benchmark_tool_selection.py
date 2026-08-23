"""
Benchmark: Naive vs RAG vs Probe-Routed tool selection.

100 tools  ·  10 categories  ·  200 queries
160 train (16/category) / 40 test (4/category) — stratified by category.

Approaches
----------
1. Naive        – all 100 tool schemas in context → Qwen picks.
2. RAG          – sentence-transformer retrieves top-5 schemas → Qwen picks.
3. Probe-Routed – layer-8 hidden state → LogReg probe predicts top-3 categories
                  (~30 tools) → Qwen picks from that narrowed set.

Metrics per approach: accuracy, macro-F1 (category-level), tokens/query, ms/query.
"""

import warnings
warnings.filterwarnings("ignore")

import re
import time
import textwrap
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
MAIN_MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
EMBED_MODEL  = "all-MiniLM-L6-v2"
PROBE_LAYER  = 8          # layer index for hidden-state probe
TOP_K_RAG    = 5          # schemas injected by RAG
TOP_K_CATS   = 3          # categories kept by probe router (~30 tools)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKS = 20         # generation budget for tool-name output
SEED         = 42

# ─────────────────────────────────────────────────────────────────────────────
# 100 tools across 10 categories (10 per category)
# ─────────────────────────────────────────────────────────────────────────────
TOOLS = [
    # file_operations
    {"name": "read_file",           "cat": "file_operations",     "desc": "Read the contents of a file from the local filesystem.",                         "params": {"path": "str", "encoding": "str"}},
    {"name": "write_file",          "cat": "file_operations",     "desc": "Write or overwrite a file with the provided content.",                            "params": {"path": "str", "content": "str", "mode": "str"}},
    {"name": "delete_file",         "cat": "file_operations",     "desc": "Permanently delete a file or empty directory.",                                   "params": {"path": "str"}},
    {"name": "list_directory",      "cat": "file_operations",     "desc": "List files and subdirectories inside a directory path.",                          "params": {"path": "str", "recursive": "bool"}},
    {"name": "copy_file",           "cat": "file_operations",     "desc": "Copy a file from a source path to a destination path.",                           "params": {"source": "str", "destination": "str"}},
    {"name": "move_file",           "cat": "file_operations",     "desc": "Move or rename a file to a new location.",                                        "params": {"source": "str", "destination": "str"}},
    {"name": "create_directory",    "cat": "file_operations",     "desc": "Create a new directory, optionally including parent directories.",                 "params": {"path": "str", "parents": "bool"}},
    {"name": "get_file_metadata",   "cat": "file_operations",     "desc": "Return size, permissions, creation time, and modification time of a file.",       "params": {"path": "str"}},
    {"name": "compress_files",      "cat": "file_operations",     "desc": "Compress one or more files into a zip or tar archive.",                           "params": {"sources": "list[str]", "output": "str", "format": "str"}},
    {"name": "watch_file",          "cat": "file_operations",     "desc": "Monitor a file for changes and return change events.",                             "params": {"path": "str", "timeout": "int"}},

    # web_search
    {"name": "web_search",          "cat": "web_search",          "desc": "Search the internet and return ranked results with snippets.",                    "params": {"query": "str", "num_results": "int"}},
    {"name": "image_search",        "cat": "web_search",          "desc": "Search for images by keyword and return image URLs.",                             "params": {"query": "str", "safe_search": "bool"}},
    {"name": "news_search",         "cat": "web_search",          "desc": "Search recent news articles and return headlines and summaries.",                 "params": {"query": "str", "days_back": "int"}},
    {"name": "video_search",        "cat": "web_search",          "desc": "Search for videos by keyword and return links and metadata.",                     "params": {"query": "str", "platform": "str"}},
    {"name": "academic_search",     "cat": "web_search",          "desc": "Search academic databases for papers and citations.",                             "params": {"query": "str", "year_from": "int"}},
    {"name": "shopping_search",     "cat": "web_search",          "desc": "Search online stores for products, prices, and availability.",                   "params": {"query": "str", "max_price": "float"}},
    {"name": "maps_search",         "cat": "web_search",          "desc": "Search for places, addresses, and directions on a map.",                          "params": {"query": "str", "near": "str"}},
    {"name": "code_search",         "cat": "web_search",          "desc": "Search public code repositories for code snippets by keyword.",                   "params": {"query": "str", "language": "str"}},
    {"name": "social_search",       "cat": "web_search",          "desc": "Search social media platforms for posts and trending topics.",                    "params": {"query": "str", "platform": "str"}},
    {"name": "patent_search",       "cat": "web_search",          "desc": "Search patent databases for filings and claims.",                                 "params": {"query": "str", "jurisdiction": "str"}},

    # communication
    {"name": "send_email",          "cat": "communication",       "desc": "Compose and send an email to one or more recipients.",                            "params": {"to": "list[str]", "subject": "str", "body": "str", "cc": "list[str]"}},
    {"name": "send_sms",            "cat": "communication",       "desc": "Send a text message to a phone number.",                                          "params": {"to": "str", "message": "str"}},
    {"name": "post_slack",          "cat": "communication",       "desc": "Post a message to a Slack channel or direct message thread.",                     "params": {"channel": "str", "message": "str", "thread_ts": "str"}},
    {"name": "create_calendar_event","cat": "communication",      "desc": "Create a calendar event with a title, time range, and optional attendees.",       "params": {"title": "str", "start": "str", "end": "str", "attendees": "list[str]"}},
    {"name": "send_push_notification","cat": "communication",     "desc": "Send a push notification to a registered mobile device.",                         "params": {"device_token": "str", "title": "str", "body": "str"}},
    {"name": "create_ticket",       "cat": "communication",       "desc": "Create a support or task ticket in a project management tracker.",                "params": {"title": "str", "description": "str", "priority": "str", "assignee": "str"}},
    {"name": "send_webhook",        "cat": "communication",       "desc": "Send an HTTP POST payload to an external webhook URL.",                           "params": {"url": "str", "payload": "dict", "headers": "dict"}},
    {"name": "post_tweet",          "cat": "communication",       "desc": "Post a message on Twitter/X on behalf of an authenticated account.",              "params": {"text": "str", "media_ids": "list[str]"}},
    {"name": "send_discord",        "cat": "communication",       "desc": "Send a message to a Discord channel via bot or webhook.",                         "params": {"channel_id": "str", "content": "str"}},
    {"name": "create_meeting",      "cat": "communication",       "desc": "Schedule a video meeting and return the join link.",                              "params": {"topic": "str", "start_time": "str", "duration": "int", "participants": "list[str]"}},

    # code_execution
    {"name": "execute_python",      "cat": "code_execution",      "desc": "Run a Python code snippet and return stdout and stderr.",                         "params": {"code": "str", "timeout": "int"}},
    {"name": "execute_javascript",  "cat": "code_execution",      "desc": "Run JavaScript code in a Node.js runtime and return output.",                    "params": {"code": "str", "timeout": "int"}},
    {"name": "execute_bash",        "cat": "code_execution",      "desc": "Execute a shell command or bash script and return output.",                       "params": {"command": "str", "cwd": "str"}},
    {"name": "run_unit_tests",      "cat": "code_execution",      "desc": "Discover and run unit tests in a project directory and report results.",          "params": {"path": "str", "framework": "str", "verbose": "bool"}},
    {"name": "lint_code",           "cat": "code_execution",      "desc": "Run a linter over source code and return warnings and errors.",                   "params": {"code": "str", "language": "str", "rules": "list[str]"}},
    {"name": "format_code",         "cat": "code_execution",      "desc": "Auto-format source code according to language style conventions.",                "params": {"code": "str", "language": "str", "style": "str"}},
    {"name": "compile_code",        "cat": "code_execution",      "desc": "Compile source code and return the binary path or compilation errors.",           "params": {"source": "str", "language": "str", "flags": "list[str]"}},
    {"name": "profile_code",        "cat": "code_execution",      "desc": "Profile code execution and return hotspots and per-function timings.",             "params": {"code": "str", "language": "str"}},
    {"name": "execute_r",           "cat": "code_execution",      "desc": "Run R code and return console output and generated plots.",                       "params": {"code": "str", "timeout": "int"}},
    {"name": "execute_sql_script",  "cat": "code_execution",      "desc": "Execute a SQL script file against a database connection.",                        "params": {"script_path": "str", "connection_string": "str"}},

    # database
    {"name": "query_database",      "cat": "database",            "desc": "Execute a SQL SELECT query and return result rows.",                              "params": {"sql": "str", "connection_string": "str"}},
    {"name": "insert_record",       "cat": "database",            "desc": "Insert one or more rows into a database table.",                                  "params": {"table": "str", "records": "list[dict]", "connection_string": "str"}},
    {"name": "update_record",       "cat": "database",            "desc": "Update existing rows matching a condition in a table.",                           "params": {"table": "str", "updates": "dict", "where": "str", "connection_string": "str"}},
    {"name": "delete_record",       "cat": "database",            "desc": "Delete rows from a table that match a given condition.",                          "params": {"table": "str", "where": "str", "connection_string": "str"}},
    {"name": "create_table",        "cat": "database",            "desc": "Create a new database table from a column schema definition.",                    "params": {"table": "str", "columns": "list[dict]", "connection_string": "str"}},
    {"name": "backup_database",     "cat": "database",            "desc": "Create a backup dump of a database to a local file.",                             "params": {"connection_string": "str", "output_path": "str", "format": "str"}},
    {"name": "restore_database",    "cat": "database",            "desc": "Restore a database from a previously created backup file.",                       "params": {"connection_string": "str", "backup_path": "str"}},
    {"name": "migrate_schema",      "cat": "database",            "desc": "Apply pending schema migrations in order from a migrations directory.",           "params": {"connection_string": "str", "migrations_dir": "str"}},
    {"name": "explain_query",       "cat": "database",            "desc": "Return the query execution plan for a SQL statement.",                            "params": {"sql": "str", "connection_string": "str"}},
    {"name": "get_table_schema",    "cat": "database",            "desc": "Return column names, types, and constraints for a given table.",                  "params": {"table": "str", "connection_string": "str"}},

    # api_calls
    {"name": "call_rest_api",       "cat": "api_calls",           "desc": "Make an HTTP request to a REST endpoint and return the response.",               "params": {"url": "str", "method": "str", "headers": "dict", "body": "dict"}},
    {"name": "call_graphql",        "cat": "api_calls",           "desc": "Execute a GraphQL query or mutation against an endpoint.",                        "params": {"url": "str", "query": "str", "variables": "dict"}},
    {"name": "authenticate_oauth",  "cat": "api_calls",           "desc": "Perform an OAuth flow and return an access token.",                              "params": {"provider": "str", "client_id": "str", "scopes": "list[str]"}},
    {"name": "get_api_status",      "cat": "api_calls",           "desc": "Check the health and availability of an external API endpoint.",                 "params": {"url": "str", "timeout": "int"}},
    {"name": "paginate_results",    "cat": "api_calls",           "desc": "Iterate through all pages of a paginated API and collect every result.",         "params": {"url": "str", "params": "dict", "max_pages": "int"}},
    {"name": "upload_file_api",     "cat": "api_calls",           "desc": "Upload a file to an API endpoint using multipart form data.",                    "params": {"url": "str", "file_path": "str", "headers": "dict"}},
    {"name": "download_file_api",   "cat": "api_calls",           "desc": "Download a file from a remote URL and save it to a local path.",                 "params": {"url": "str", "output_path": "str", "headers": "dict"}},
    {"name": "stream_api",          "cat": "api_calls",           "desc": "Connect to a streaming API endpoint and yield server-sent events.",              "params": {"url": "str", "headers": "dict", "timeout": "int"}},
    {"name": "batch_api_request",   "cat": "api_calls",           "desc": "Send multiple API requests concurrently and return all responses.",              "params": {"requests": "list[dict]", "concurrency": "int"}},
    {"name": "mock_api",            "cat": "api_calls",           "desc": "Spin up a local mock HTTP server that returns predefined responses.",             "params": {"routes": "list[dict]", "port": "int"}},

    # data_processing
    {"name": "parse_csv",           "cat": "data_processing",     "desc": "Parse a CSV string or file into a list of row dictionaries.",                    "params": {"source": "str", "delimiter": "str", "has_header": "bool"}},
    {"name": "parse_json",          "cat": "data_processing",     "desc": "Parse a JSON string and return a structured Python object.",                     "params": {"json_string": "str", "strict": "bool"}},
    {"name": "transform_dataframe", "cat": "data_processing",     "desc": "Apply a sequence of transformations to a tabular dataset.",                      "params": {"data": "list[dict]", "operations": "list[dict]"}},
    {"name": "merge_datasets",      "cat": "data_processing",     "desc": "Join two datasets on a key column using an inner, left, or outer join.",         "params": {"left": "list[dict]", "right": "list[dict]", "on": "str", "how": "str"}},
    {"name": "filter_records",      "cat": "data_processing",     "desc": "Filter rows in a dataset based on a boolean condition expression.",              "params": {"data": "list[dict]", "condition": "str"}},
    {"name": "aggregate_data",      "cat": "data_processing",     "desc": "Group a dataset by columns and aggregate with sum, mean, or count.",             "params": {"data": "list[dict]", "group_by": "list[str]", "agg": "dict"}},
    {"name": "normalize_data",      "cat": "data_processing",     "desc": "Normalize numeric columns to 0–1 range or z-scores.",                            "params": {"data": "list[dict]", "columns": "list[str]", "method": "str"}},
    {"name": "encode_categories",   "cat": "data_processing",     "desc": "One-hot or label encode categorical columns in a dataset.",                      "params": {"data": "list[dict]", "columns": "list[str]", "method": "str"}},
    {"name": "split_dataset",       "cat": "data_processing",     "desc": "Split a dataset into train, validation, and test subsets by given ratios.",      "params": {"data": "list[dict]", "ratios": "list[float]", "seed": "int"}},
    {"name": "visualize_data",      "cat": "data_processing",     "desc": "Generate a chart from tabular data and return an image.",                        "params": {"data": "list[dict]", "chart_type": "str", "x": "str", "y": "str"}},

    # ml_operations
    {"name": "train_model",         "cat": "ml_operations",       "desc": "Train a machine learning model on a labelled dataset.",                          "params": {"model_type": "str", "train_data": "list[dict]", "config": "dict"}},
    {"name": "evaluate_model",      "cat": "ml_operations",       "desc": "Evaluate a trained model on a test set and return performance metrics.",         "params": {"model_path": "str", "test_data": "list[dict]", "metrics": "list[str]"}},
    {"name": "run_inference",       "cat": "ml_operations",       "desc": "Run single or batch inference using a loaded model.",                            "params": {"model_path": "str", "inputs": "list", "device": "str"}},
    {"name": "fine_tune_model",     "cat": "ml_operations",       "desc": "Fine-tune a pretrained model on new labelled examples.",                         "params": {"base_model": "str", "train_data": "list[dict]", "epochs": "int"}},
    {"name": "export_model",        "cat": "ml_operations",       "desc": "Export a trained model to ONNX, TorchScript, or SavedModel format.",             "params": {"model_path": "str", "format": "str", "output_path": "str"}},
    {"name": "load_checkpoint",     "cat": "ml_operations",       "desc": "Load model weights from a checkpoint file onto a device.",                       "params": {"checkpoint_path": "str", "device": "str"}},
    {"name": "compute_embeddings",  "cat": "ml_operations",       "desc": "Compute dense vector embeddings for a list of texts using a given model.",       "params": {"texts": "list[str]", "model": "str", "batch_size": "int"}},
    {"name": "hyperparameter_search","cat": "ml_operations",      "desc": "Run hyperparameter optimisation over a search space for a given metric.",        "params": {"config": "dict", "n_trials": "int", "metric": "str"}},
    {"name": "explain_prediction",  "cat": "ml_operations",       "desc": "Generate feature-importance explanations for a model prediction.",               "params": {"model_path": "str", "input": "dict", "method": "str"}},
    {"name": "monitor_drift",       "cat": "ml_operations",       "desc": "Detect distribution shift between a reference dataset and current data.",        "params": {"reference": "list[dict]", "current": "list[dict]", "threshold": "float"}},

    # system_admin
    {"name": "restart_service",     "cat": "system_admin",        "desc": "Restart a systemd or process-managed service on a host.",                        "params": {"service": "str", "host": "str"}},
    {"name": "check_system_health", "cat": "system_admin",        "desc": "Return CPU, memory, disk, and network metrics for a host.",                      "params": {"host": "str"}},
    {"name": "get_logs",            "cat": "system_admin",        "desc": "Fetch recent log lines from a service or system log with optional level filter.", "params": {"service": "str", "lines": "int", "level": "str"}},
    {"name": "set_env_variable",    "cat": "system_admin",        "desc": "Set or update an environment variable on a host or container.",                  "params": {"name": "str", "value": "str", "host": "str"}},
    {"name": "manage_processes",    "cat": "system_admin",        "desc": "List, start, stop, or kill processes by name or PID on a host.",                 "params": {"action": "str", "process": "str", "host": "str"}},
    {"name": "configure_firewall",  "cat": "system_admin",        "desc": "Add or remove firewall rules for specific ports and IP ranges.",                  "params": {"action": "str", "port": "int", "protocol": "str", "source": "str"}},
    {"name": "manage_users",        "cat": "system_admin",        "desc": "Create, delete, or modify user accounts and group memberships.",                  "params": {"action": "str", "username": "str", "groups": "list[str]"}},
    {"name": "schedule_cron",       "cat": "system_admin",        "desc": "Add or remove cron jobs on a target host.",                                       "params": {"expression": "str", "command": "str", "host": "str"}},
    {"name": "monitor_resources",   "cat": "system_admin",        "desc": "Stream real-time CPU, memory, and I/O metrics for a host or container.",         "params": {"host": "str", "interval": "int", "duration": "int"}},
    {"name": "deploy_container",    "cat": "system_admin",        "desc": "Pull and run a Docker container image on a target host.",                        "params": {"image": "str", "host": "str", "env": "dict", "ports": "dict"}},

    # document_processing
    {"name": "extract_pdf_text",    "cat": "document_processing", "desc": "Extract plain text from a PDF file, optionally from specific page ranges.",      "params": {"path": "str", "pages": "str"}},
    {"name": "convert_document",    "cat": "document_processing", "desc": "Convert a document between formats such as PDF, DOCX, HTML, or Markdown.",       "params": {"source": "str", "target_format": "str"}},
    {"name": "ocr_image",           "cat": "document_processing", "desc": "Run optical character recognition on an image and return extracted text.",        "params": {"path": "str", "language": "str"}},
    {"name": "summarize_document",  "cat": "document_processing", "desc": "Generate a concise summary of a long document.",                                  "params": {"text": "str", "max_length": "int", "style": "str"}},
    {"name": "translate_document",  "cat": "document_processing", "desc": "Translate a document into a target language.",                                    "params": {"text": "str", "target_language": "str"}},
    {"name": "sign_document",       "cat": "document_processing", "desc": "Apply a digital signature to a PDF document using a signing key.",                "params": {"path": "str", "signature_key": "str", "output": "str"}},
    {"name": "merge_pdfs",          "cat": "document_processing", "desc": "Merge multiple PDF files into a single PDF in a specified order.",                "params": {"sources": "list[str]", "output": "str"}},
    {"name": "split_pdf",           "cat": "document_processing", "desc": "Split a PDF into individual pages or named page ranges.",                         "params": {"path": "str", "ranges": "list[str]", "output_dir": "str"}},
    {"name": "watermark_document",  "cat": "document_processing", "desc": "Add a text or image watermark to every page of a PDF.",                          "params": {"path": "str", "watermark": "str", "output": "str"}},
    {"name": "extract_tables",      "cat": "document_processing", "desc": "Extract tabular data from a PDF or scanned image into structured rows.",          "params": {"path": "str", "format": "str"}},
]

assert len(TOOLS) == 100
TOOL_NAMES = [t["name"] for t in TOOLS]
TOOL_BY_NAME = {t["name"]: t for t in TOOLS}
CATEGORIES = sorted({t["cat"] for t in TOOLS})
assert len(CATEGORIES) == 10

# ─────────────────────────────────────────────────────────────────────────────
# 200 queries — 2 per tool, 20 per category
# Each entry: (query_text, correct_tool_name)
# ─────────────────────────────────────────────────────────────────────────────
QUERIES = [
    # file_operations
    ("What's inside the config.yaml file in my project root?",                        "read_file"),
    ("Show me the contents of /etc/hosts.",                                            "read_file"),
    ("Save this error summary to output.txt.",                                         "write_file"),
    ("Create a new file called report.md with the following content.",                 "write_file"),
    ("Remove the temp file at /tmp/scratch.txt.",                                      "delete_file"),
    ("Clean up the old backup archive at /backups/backup_2022.tar.",                   "delete_file"),
    ("What files are inside the /var/log directory?",                                  "list_directory"),
    ("Show me everything that's in the uploads folder.",                               "list_directory"),
    ("Make a copy of main.py before I start editing it.",                              "copy_file"),
    ("Duplicate the production config to config.backup.",                              "copy_file"),
    ("Rename old_report.pdf to 2024_annual_report.pdf.",                               "move_file"),
    ("Move the downloaded dataset file to the archive folder.",                        "move_file"),
    ("Create a new folder called experiments under the project root.",                 "create_directory"),
    ("Make the nested path /data/raw/2024 if it doesn't already exist.",               "create_directory"),
    ("When was this config file last modified and how big is it?",                     "get_file_metadata"),
    ("How large is the database dump file on disk?",                                   "get_file_metadata"),
    ("Zip up the entire logs folder for long-term archiving.",                         "compress_files"),
    ("Bundle these three config files into a tarball.",                                "compress_files"),
    ("Alert me when output.log changes.",                                              "watch_file"),
    ("Monitor access.log and tell me when new lines appear.",                          "watch_file"),

    # web_search
    ("What's the latest news about AI regulation in Europe?",                          "web_search"),
    ("Find me information on the best noise-cancelling headphones right now.",         "web_search"),
    ("Find reference photos of Aurora Borealis over Iceland.",                         "image_search"),
    ("Get me reference images of mid-century modern furniture.",                       "image_search"),
    ("What happened in the markets this morning?",                                     "news_search"),
    ("Any recent developments in the semiconductor shortage?",                         "news_search"),
    ("Find a beginner tutorial video on soldering SMD components.",                    "video_search"),
    ("Look for a TED talk about the future of nuclear energy.",                        "video_search"),
    ("Find papers on transformer attention mechanisms published after 2022.",           "academic_search"),
    ("Search for peer-reviewed research on CRISPR gene editing in crops.",             "academic_search"),
    ("What's the best standing desk under $500?",                                      "shopping_search"),
    ("Find me a mechanical keyboard with tactile switches.",                           "shopping_search"),
    ("Where is the nearest urgent care clinic to me?",                                 "maps_search"),
    ("Find coffee shops near Union Square in San Francisco.",                          "maps_search"),
    ("Search GitHub for Python async rate-limiting examples.",                         "code_search"),
    ("Find open-source LRU cache implementations in Go.",                              "code_search"),
    ("What are people tweeting about the latest OpenAI announcement?",                 "social_search"),
    ("Find Reddit threads about mechanical watch servicing.",                           "social_search"),
    ("Are there any patents on this fast-charging battery method?",                    "patent_search"),
    ("Search patents filed by Apple on flexible display technology.",                  "patent_search"),

    # communication
    ("Let my manager know I'll be out sick tomorrow.",                                  "send_email"),
    ("Send Alice the meeting notes from today's session.",                              "send_email"),
    ("Text my wife that I'm on my way home from the office.",                          "send_sms"),
    ("Send a quick SMS to the plumber confirming tomorrow's appointment.",              "send_sms"),
    ("Post the deployment summary to the #releases Slack channel.",                    "post_slack"),
    ("Send a message to the backend team channel about the ongoing outage.",           "post_slack"),
    ("Block off Friday afternoon for a team retrospective.",                            "create_calendar_event"),
    ("Schedule a 30-minute call with the client for next Tuesday at 2 PM.",            "create_calendar_event"),
    ("Push an alert to the mobile app about the upcoming maintenance window.",         "send_push_notification"),
    ("Notify users' phones that their order has shipped.",                              "send_push_notification"),
    ("File a bug report for the login-page timeout issue.",                            "create_ticket"),
    ("Create a task for updating the API documentation this sprint.",                  "create_ticket"),
    ("Trigger the CI pipeline via the GitHub webhook.",                                "send_webhook"),
    ("Notify the Zapier automation that a new form submission arrived.",               "send_webhook"),
    ("Announce our product launch on the company Twitter account.",                    "post_tweet"),
    ("Post a thread summarising the blog post we just published.",                     "post_tweet"),
    ("Alert the community server about tonight's live stream.",                        "send_discord"),
    ("Post the release notes to the #changelog Discord channel.",                      "send_discord"),
    ("Set up a Zoom call for the all-hands meeting next Monday.",                      "create_meeting"),
    ("Schedule a video interview with the job candidate for Thursday at 10 AM.",       "create_meeting"),

    # code_execution
    ("Calculate the compound interest on $10,000 at 5% for 10 years.",                "execute_python"),
    ("Parse this JSON payload and flatten all nested keys into dot notation.",         "execute_python"),
    ("Minify this JavaScript bundle for production deployment.",                       "execute_javascript"),
    ("Evaluate this regex pattern against a set of test strings.",                     "execute_javascript"),
    ("Count how many lines of Python code are in this repository.",                    "execute_bash"),
    ("Check whether port 8080 is already in use on this machine.",                     "execute_bash"),
    ("Run the test suite and tell me which tests are currently failing.",              "run_unit_tests"),
    ("Execute all tests in the auth module and show the coverage report.",             "run_unit_tests"),
    ("Check this Python file for any PEP 8 style violations.",                        "lint_code"),
    ("Lint the TypeScript code before the pull request review.",                       "lint_code"),
    ("Clean up the indentation and formatting in this Python script.",                 "format_code"),
    ("Auto-format this Go source file according to gofmt conventions.",               "format_code"),
    ("Build the C++ project and report any compiler errors.",                          "compile_code"),
    ("Compile this Rust source file and flag any warnings.",                           "compile_code"),
    ("Where is this sorting algorithm spending most of its execution time?",           "profile_code"),
    ("Profile the image preprocessing pipeline and find the bottleneck.",              "profile_code"),
    ("Run this statistical analysis script and return the p-values.",                  "execute_r"),
    ("Generate a scatter plot from this R code.",                                      "execute_r"),
    ("Run the seed data migration script against the staging database.",               "execute_sql_script"),
    ("Apply the schema changes in setup.sql to the dev environment.",                  "execute_sql_script"),

    # database
    ("How many orders were placed in the past 7 days?",                                "query_database"),
    ("What's the average age of users who churned last quarter?",                      "query_database"),
    ("Add the new customer record to the users table.",                                "insert_record"),
    ("Log this API error event to the errors table.",                                  "insert_record"),
    ("Mark this user's subscription as cancelled in the database.",                    "update_record"),
    ("Update the product price for SKU-2041 to $29.99.",                               "update_record"),
    ("Remove all expired sessions from the sessions table.",                           "delete_record"),
    ("Delete the test accounts we created during QA testing.",                         "delete_record"),
    ("Create a new table to store incoming webhook events.",                           "create_table"),
    ("Set up a table to track A/B experiment assignments.",                            "create_table"),
    ("Take a snapshot of the production database before the migration.",               "backup_database"),
    ("Export the current database state to a dump file for safekeeping.",              "backup_database"),
    ("Roll back the database to yesterday's backup after the bad deploy.",             "restore_database"),
    ("Restore the staging database from the latest nightly snapshot.",                 "restore_database"),
    ("Apply all pending migrations to the staging environment.",                       "migrate_schema"),
    ("Run the migration to add the new columns to the subscriptions table.",           "migrate_schema"),
    ("Why is this join query running so slowly on the orders table?",                  "explain_query"),
    ("Show me the execution plan for this aggregation query.",                         "explain_query"),
    ("What columns and types does the subscriptions table have?",                      "get_table_schema"),
    ("Show me the structure of the events table including all constraints.",           "get_table_schema"),

    # api_calls
    ("Fetch the current weather data for London from the weather service.",            "call_rest_api"),
    ("Get the user profile from the auth API using their account ID.",                 "call_rest_api"),
    ("Retrieve the last 10 posts with author names using GraphQL.",                    "call_graphql"),
    ("Query the full product catalog with all variants via the GraphQL API.",          "call_graphql"),
    ("Get an OAuth access token for the Salesforce integration.",                      "authenticate_oauth"),
    ("Authenticate against the Google Drive API with the required scopes.",            "authenticate_oauth"),
    ("Is the payment gateway API currently responding?",                               "get_api_status"),
    ("Check if our third-party analytics endpoint is healthy.",                        "get_api_status"),
    ("Collect every customer record from the paginated CRM API.",                      "paginate_results"),
    ("Fetch all pages of transactions from the billing API.",                          "paginate_results"),
    ("Upload the CSV export to the data warehouse ingestion API.",                     "upload_file_api"),
    ("Send the generated PDF report to the document storage API.",                     "upload_file_api"),
    ("Download the monthly invoice PDF from the billing portal.",                      "download_file_api"),
    ("Grab the latest dataset export from the analytics API.",                         "download_file_api"),
    ("Stream real-time stock price updates from the market data feed.",                "stream_api"),
    ("Connect to the live event stream from the IoT sensor API.",                      "stream_api"),
    ("Enrich all these user records through the lookup API in parallel.",              "batch_api_request"),
    ("Send all these notification requests through the API at once.",                  "batch_api_request"),
    ("Spin up a fake Stripe API for our integration test suite.",                      "mock_api"),
    ("Create a mock server to simulate the vendor's payment webhook.",                 "mock_api"),

    # data_processing
    ("Load this sales data CSV and show me the column headers.",                       "parse_csv"),
    ("Import the user export CSV and turn it into a list of records.",                 "parse_csv"),
    ("Pull the error details out of this raw API response JSON.",                      "parse_json"),
    ("Parse this nested configuration JSON into a usable Python object.",              "parse_json"),
    ("Add a calculated profit margin column to this sales dataset.",                   "transform_dataframe"),
    ("Rename these columns and drop the ones we don't need before training.",          "transform_dataframe"),
    ("Join the users table data with the orders data on user_id.",                     "merge_datasets"),
    ("Combine the product inventory records with the pricing sheet.",                  "merge_datasets"),
    ("Keep only rows where the subscription status is 'active'.",                      "filter_records"),
    ("Filter out any records that have a missing or empty email address.",             "filter_records"),
    ("Sum up revenue grouped by region for this quarter.",                             "aggregate_data"),
    ("Count the number of events each user triggered per day.",                        "aggregate_data"),
    ("Scale all the feature columns to a 0–1 range before feeding the model.",        "normalize_data"),
    ("Standardize the age and income columns using z-score normalization.",            "normalize_data"),
    ("One-hot encode the country column before training.",                             "encode_categories"),
    ("Convert the plan_type field into numeric labels for the classifier.",            "encode_categories"),
    ("Divide the labelled data into 80% training and 20% validation sets.",           "split_dataset"),
    ("Create a 70/15/15 train/val/test split with a fixed random seed.",              "split_dataset"),
    ("Plot monthly revenue as a bar chart.",                                           "visualize_data"),
    ("Create a scatter plot of user age versus average session duration.",             "visualize_data"),

    # ml_operations
    ("Train a gradient boosting classifier on this tabular churn dataset.",            "train_model"),
    ("Fit a logistic regression model to the fraud prediction training data.",         "train_model"),
    ("What's the AUC and F1 score of the classifier on the holdout set?",             "evaluate_model"),
    ("Measure the RMSE of the regression model on the test split.",                    "evaluate_model"),
    ("Run this new image through the object detection model.",                         "run_inference"),
    ("Get batch predictions for these customer records from the propensity model.",    "run_inference"),
    ("Adapt the pretrained NER model to our internal domain-specific entities.",       "fine_tune_model"),
    ("Fine-tune the sentiment model on our product review examples.",                  "fine_tune_model"),
    ("Export the trained classifier to ONNX format for the serving pipeline.",        "export_model"),
    ("Save the TensorFlow model as a SavedModel for deployment.",                      "export_model"),
    ("Resume training from the last saved checkpoint.",                                "load_checkpoint"),
    ("Load the best-performing model weights from the hyperparameter sweep.",         "load_checkpoint"),
    ("Generate embeddings for all documents in our corpus.",                           "compute_embeddings"),
    ("Encode these product descriptions as vectors for similarity search.",            "compute_embeddings"),
    ("Find the best learning rate and batch size combination for this model.",         "hyperparameter_search"),
    ("Run a Bayesian search over the regularization parameters.",                      "hyperparameter_search"),
    ("Why did the model flag this transaction as potential fraud?",                    "explain_prediction"),
    ("Show me which features drove this churn prediction score.",                      "explain_prediction"),
    ("Check whether the incoming feature distribution has shifted from training.",     "monitor_drift"),
    ("Alert me if the model's input data starts drifting from the baseline.",         "monitor_drift"),

    # system_admin
    ("The nginx server seems stuck — can you restart it?",                             "restart_service"),
    ("Bounce the Redis service on the caching host.",                                  "restart_service"),
    ("How's the CPU and memory usage on the production server?",                       "check_system_health"),
    ("Are any of the nodes running out of disk space?",                                "check_system_health"),
    ("Show me the last 100 error lines from the application service.",                 "get_logs"),
    ("Pull the recent authentication failure logs from the syslog.",                   "get_logs"),
    ("Update the DATABASE_URL environment variable on the app server.",                "set_env_variable"),
    ("Set LOG_LEVEL to DEBUG on the staging container.",                               "set_env_variable"),
    ("Kill the zombie process that's been pegging the CPU at 100%.",                  "manage_processes"),
    ("List all running processes owned by the deploy user.",                           "manage_processes"),
    ("Block incoming traffic on port 22 from all IPs except our VPN range.",          "configure_firewall"),
    ("Open port 443 for the new load balancer IP address.",                            "configure_firewall"),
    ("Create an account for the new contractor with read-only access.",               "manage_users"),
    ("Remove the departed employee from the admin and sudo groups.",                   "manage_users"),
    ("Set up a nightly cron job to purge old log files at 2 AM.",                     "schedule_cron"),
    ("Schedule the weekly backup script for every Sunday at midnight.",               "schedule_cron"),
    ("Watch memory usage on this container for the next 5 minutes.",                  "monitor_resources"),
    ("Stream live CPU metrics from the GPU training server.",                          "monitor_resources"),
    ("Launch the new version of the API container on the production host.",            "deploy_container"),
    ("Pull and start the updated worker image on the batch processing server.",       "deploy_container"),

    # document_processing
    ("Pull all the text out of this research paper PDF.",                              "extract_pdf_text"),
    ("Get the text content from pages 3 through 7 of this report.",                   "extract_pdf_text"),
    ("Convert this Word document to PDF for sending.",                                 "convert_document"),
    ("Turn this Markdown file into an HTML page.",                                     "convert_document"),
    ("What does the text on this scanned invoice say?",                               "ocr_image"),
    ("Extract the address from this photo of a business card.",                        "ocr_image"),
    ("Give me a two-paragraph summary of this 50-page contract.",                     "summarize_document"),
    ("Condense this technical specification into bullet points.",                      "summarize_document"),
    ("Translate this French press release into English.",                              "translate_document"),
    ("Convert this Japanese product manual into Spanish.",                             "translate_document"),
    ("Apply a digital signature to the final contract PDF.",                           "sign_document"),
    ("Sign this NDA with our company's signing certificate.",                          "sign_document"),
    ("Combine the cover letter and resume into a single PDF.",                         "merge_pdfs"),
    ("Merge all the monthly reports into one annual document PDF.",                    "merge_pdfs"),
    ("Extract just pages 4 through 8 from this hundred-page document.",               "split_pdf"),
    ("Split this report PDF into individual chapter files.",                           "split_pdf"),
    ("Stamp 'CONFIDENTIAL' as a watermark on every page before sharing.",             "watermark_document"),
    ("Add a 'DRAFT' watermark to all pages of this document.",                        "watermark_document"),
    ("Pull out the financial tables from this earnings report PDF.",                   "extract_tables"),
    ("Extract the data table from this scanned regulatory form.",                      "extract_tables"),
]

assert len(QUERIES) == 200

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_tool_schema(tool: dict) -> str:
    params = ", ".join(f"{k}: {v}" for k, v in tool["params"].items())
    return f"{tool['name']}({params}) — {tool['desc']}"


def format_tools_block(tool_list: list[dict]) -> str:
    return "\n".join(f"  - {format_tool_schema(t)}" for t in tool_list)


def build_selection_prompt(query: str, tool_list: list[dict]) -> str:
    block = format_tools_block(tool_list)
    return (
        "You are a tool-selection system. Given a user request and a list of tools, "
        "output ONLY the name of the single best tool. No explanation.\n\n"
        f"Tools:\n{block}\n\n"
        f"Request: {query}\n\n"
        "Tool name:"
    )


def parse_tool_name(raw: str, available: list[str]) -> str:
    """Return the first tool name found in raw; '' if none match."""
    raw_lower = raw.lower().strip()
    # exact match after stripping
    for name in available:
        if raw_lower.startswith(name.lower()):
            return name
    # substring match
    for name in available:
        if name.lower() in raw_lower:
            return name
    return ""


def tokens_in_prompt(text: str, tok) -> int:
    return tok(text, return_tensors="pt").input_ids.shape[1]


# ─────────────────────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading {MAIN_MODEL} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(MAIN_MODEL, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MAIN_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    output_hidden_states=True,
    trust_remote_code=True,
    device_map="auto" if DEVICE == "cuda" else None,
)
model.eval()
if DEVICE == "cpu":
    model = model.to(DEVICE)

print(f"Loading sentence-transformer {EMBED_MODEL} ...")
embedder = SentenceTransformer(EMBED_MODEL)

# ─────────────────────────────────────────────────────────────────────────────
# 160 / 40 stratified split (16 train + 4 test per category)
# ─────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)
train_indices, test_indices = [], []

for cat in CATEGORIES:
    cat_idx = [i for i, (_, tn) in enumerate(QUERIES) if TOOL_BY_NAME[tn]["cat"] == cat]
    assert len(cat_idx) == 20, f"{cat} has {len(cat_idx)} queries"
    perm = rng.permutation(cat_idx)
    train_indices.extend(perm[:16].tolist())
    test_indices.extend(perm[16:].tolist())

train_queries = [QUERIES[i] for i in train_indices]   # 160
test_queries  = [QUERIES[i] for i in test_indices]    #  40

print(f"\nSplit: {len(train_queries)} train / {len(test_queries)} test")

# ─────────────────────────────────────────────────────────────────────────────
# Shared: extract last-token hidden state at PROBE_LAYER
# ─────────────────────────────────────────────────────────────────────────────

def get_hidden_state(texts: list[str]) -> np.ndarray:
    """Return (N, H) float32 array of last-real-token hidden states at PROBE_LAYER."""
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=128
    ).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    hs = out.hidden_states[PROBE_LAYER + 1]          # (B, seq, H)
    seq_lens = inputs["attention_mask"].sum(dim=1) - 1  # (B,)
    idx = seq_lens.view(-1, 1, 1).expand(-1, 1, hs.size(-1))
    last = hs.gather(1, idx).squeeze(1).float().cpu().numpy()
    return last


# ─────────────────────────────────────────────────────────────────────────────
# Probe training
# ─────────────────────────────────────────────────────────────────────────────

print("\nExtracting layer-8 hidden states for 160 train queries ...")
BATCH = 16
train_texts = [q for q, _ in train_queries]
train_cats  = [TOOL_BY_NAME[tn]["cat"] for _, tn in train_queries]

X_train_parts = []
for i in range(0, len(train_texts), BATCH):
    X_train_parts.append(get_hidden_state(train_texts[i:i+BATCH]))
    print(f"  {min(i+BATCH, len(train_texts))}/{len(train_texts)}", end="\r")
X_train = np.vstack(X_train_parts)

cat_encoder = LabelEncoder().fit(CATEGORIES)
y_train = cat_encoder.transform(train_cats)

probe = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
probe.fit(X_train, y_train)
print(f"\nProbe trained. Train accuracy: {probe.score(X_train, y_train):.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# RAG: embed all 100 tool schemas
# ─────────────────────────────────────────────────────────────────────────────

print("\nBuilding RAG index over 100 tool schemas ...")
schema_texts = [f"{t['name']} {t['desc']} {' '.join(t['params'].keys())}" for t in TOOLS]
schema_embs  = embedder.encode(schema_texts, batch_size=32, show_progress_bar=False)   # (100, 384)

# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def qwen_generate(prompt: str) -> str:
    """Run greedy generation and return the generated suffix."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Run all three approaches on the 40 test queries
# ─────────────────────────────────────────────────────────────────────────────

results = {name: {"preds": [], "cats_pred": [], "tok": [], "ms": []}
           for name in ("naive", "rag", "probe")}

gt_tools = [tn for _, tn in test_queries]
gt_cats  = [TOOL_BY_NAME[tn]["cat"] for tn in gt_tools]

print(f"\nRunning inference on {len(test_queries)} test queries ...\n")

for qi, (query, correct_tool) in enumerate(test_queries):
    print(f"  [{qi+1:2d}/40] {query[:70]}")

    # ── 1. Naive ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    prompt_naive = build_selection_prompt(query, TOOLS)
    tok_naive    = tokens_in_prompt(prompt_naive, tokenizer)
    raw_naive    = qwen_generate(prompt_naive)
    ms_naive     = (time.perf_counter() - t0) * 1000

    pred_naive = parse_tool_name(raw_naive, TOOL_NAMES)
    results["naive"]["preds"].append(pred_naive)
    results["naive"]["cats_pred"].append(TOOL_BY_NAME.get(pred_naive, {}).get("cat", ""))
    results["naive"]["tok"].append(tok_naive)
    results["naive"]["ms"].append(ms_naive)

    # ── 2. RAG ────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    q_emb      = embedder.encode([query], show_progress_bar=False)
    sims       = cosine_similarity(q_emb, schema_embs)[0]
    top5_idx   = np.argsort(sims)[::-1][:TOP_K_RAG]
    rag_tools  = [TOOLS[i] for i in top5_idx]
    prompt_rag = build_selection_prompt(query, rag_tools)
    tok_rag    = tokens_in_prompt(prompt_rag, tokenizer)
    raw_rag    = qwen_generate(prompt_rag)
    ms_rag     = (time.perf_counter() - t0) * 1000

    pred_rag = parse_tool_name(raw_rag, [t["name"] for t in rag_tools])
    results["rag"]["preds"].append(pred_rag)
    results["rag"]["cats_pred"].append(TOOL_BY_NAME.get(pred_rag, {}).get("cat", ""))
    results["rag"]["tok"].append(tok_rag)
    results["rag"]["ms"].append(ms_rag)

    # ── 3. Probe-Routed ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    # Step A: hidden state → probe → top-3 categories
    hs_q      = get_hidden_state([query])                     # (1, H)
    cat_probs = probe.predict_proba(hs_q)[0]                  # (10,)
    top3_cat_idx = np.argsort(cat_probs)[::-1][:TOP_K_CATS]
    top3_cats    = cat_encoder.inverse_transform(top3_cat_idx).tolist()
    probe_tools  = [t for t in TOOLS if t["cat"] in top3_cats]  # ~30 tools
    # Step B: narrowed context → Qwen picks tool
    prompt_probe = build_selection_prompt(query, probe_tools)
    tok_probe    = tokens_in_prompt(prompt_probe, tokenizer)
    raw_probe    = qwen_generate(prompt_probe)
    ms_probe     = (time.perf_counter() - t0) * 1000

    pred_probe = parse_tool_name(raw_probe, [t["name"] for t in probe_tools])
    results["probe"]["preds"].append(pred_probe)
    results["probe"]["cats_pred"].append(TOOL_BY_NAME.get(pred_probe, {}).get("cat", ""))
    results["probe"]["tok"].append(tok_probe)
    results["probe"]["ms"].append(ms_probe)

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(r: dict, gt_tools: list, gt_cats: list) -> dict:
    preds     = r["preds"]
    cat_preds = r["cats_pred"]
    # tool-level accuracy (exact match; empty string → always wrong)
    tool_acc = accuracy_score(gt_tools, preds)
    # category-level macro F1
    all_cats = sorted(set(gt_cats + [c for c in cat_preds if c]))
    cat_f1   = f1_score(gt_cats, cat_preds, labels=all_cats, average="macro", zero_division=0)
    return {
        "accuracy":     tool_acc,
        "macro_f1_cat": cat_f1,
        "tokens_mean":  float(np.mean(r["tok"])),
        "ms_mean":      float(np.mean(r["ms"])),
    }

metrics = {name: compute_metrics(results[name], gt_tools, gt_cats)
           for name in ("naive", "rag", "probe")}

# ─────────────────────────────────────────────────────────────────────────────
# Print results table
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═" * 72)
print(f"{'Approach':<18} {'Accuracy':>10} {'MacroF1(cat)':>14} {'Tok/query':>12} {'ms/query':>10}")
print("─" * 72)
labels = {"naive": "Naive (100 tools)", "rag": "RAG (top-5)", "probe": "Probe-Routed (~30)"}
for key in ("naive", "rag", "probe"):
    m = metrics[key]
    print(f"{labels[key]:<18} {m['accuracy']:>10.3f} {m['macro_f1_cat']:>14.3f} "
          f"{m['tokens_mean']:>12.1f} {m['ms_mean']:>10.1f}")
print("═" * 72)

# Per-query breakdown (condensed)
print("\nPer-query detail (test set):")
header = f"{'#':>3}  {'Query (truncated)':<45}  {'Correct':<20}  {'Naive':>6}  {'RAG':>6}  {'Probe':>6}"
print(header)
print("-" * len(header))
for qi in range(len(test_queries)):
    q, ct = test_queries[qi]
    correct = "✓" if results["naive"]["preds"][qi] == ct else "✗"
    correct_r = "✓" if results["rag"]["preds"][qi]   == ct else "✗"
    correct_p = "✓" if results["probe"]["preds"][qi]  == ct else "✗"
    print(f"{qi+1:>3}  {q[:45]:<45}  {ct:<20}  {correct:>6}  {correct_r:>6}  {correct_p:>6}")

print("\nDone.")
