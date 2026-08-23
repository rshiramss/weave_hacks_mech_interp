#!/usr/bin/env python3
"""
Large-Scale Tool Selection Benchmark
100 tools | 10 categories | 200 queries | 3-agent swarm vs single-model baselines

Swarm: Qwen2.5-0.5B (proposer) -> Qwen2.5-1.5B (critic) -> hidden-state probe (voter)
       Final answer: majority vote (2/3); 3-way tie -> critic wins.
Baselines: naive (all 100 schemas in prompt) and RAG (top-5 retrieved schemas).
Metrics: accuracy, macro precision/recall/F1, tokens/query, ms/query, agreement rate.
"""

import sys
import time
from collections import Counter

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Configuration ──────────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SMALL  = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_LARGE  = "Qwen/Qwen2.5-1.5B-Instruct"
EMBED_MODEL  = "all-MiniLM-L6-v2"
RAG_TOP_K    = 5
N_TRAIN      = 160
N_TEST       = 40

# ── 100 Tool Schemas (10 per category) ────────────────────────────────────────
def T(name, cat, desc, params):
    return {"name": name, "cat": cat, "desc": desc, "params": params}

TOOLS = [
    # file_ops
    T("read_file",         "file_ops", "Read and return the full contents of a file from disk.", "path:str"),
    T("write_file",        "file_ops", "Write or overwrite a file on disk with the given content.", "path:str, content:str"),
    T("list_directory",    "file_ops", "List files and subdirectories inside a directory path.", "path:str, recursive:bool, pattern:str"),
    T("delete_file",       "file_ops", "Permanently remove a file or directory from the filesystem.", "path:str, recursive:bool"),
    T("copy_file",         "file_ops", "Copy a file or directory tree to a new destination path.", "src:str, dst:str, overwrite:bool"),
    T("move_file",         "file_ops", "Move or rename a file or directory to a new location.", "src:str, dst:str"),
    T("compress_file",     "file_ops", "Archive and compress files into a zip or tar.gz bundle.", "paths:list, output:str, fmt:str"),
    T("decompress_file",   "file_ops", "Extract a zip/tar/gz archive to a target directory.", "archive:str, dest:str"),
    T("watch_file",        "file_ops", "Monitor a path for filesystem changes and emit events.", "path:str, events:list, callback:str"),
    T("get_file_metadata", "file_ops", "Return size, permissions, timestamps, and owner of a path.", "path:str"),
    # web_search
    T("web_search",        "web_search", "Full-text internet search returning ranked result snippets.", "query:str, n:int, lang:str"),
    T("fetch_url",         "web_search", "Download and return the raw content of a specific URL.", "url:str, fmt:str, timeout:int"),
    T("search_news",       "web_search", "Retrieve recent news articles for a topic within a time window.", "query:str, days:int, sources:list"),
    T("image_search",      "web_search", "Find web images matching a description or keyword.", "query:str, n:int, safe:bool"),
    T("academic_search",   "web_search", "Search scholarly papers, preprints, and peer-reviewed journals.", "query:str, year_min:int, fields:list"),
    T("video_search",      "web_search", "Search for videos by title, description, or transcript keywords.", "query:str, n:int, duration:str"),
    T("social_search",     "web_search", "Search public social-media posts and threads by keyword.", "query:str, platform:str, since:str"),
    T("cached_fetch",      "web_search", "Retrieve a cached or archived version of a URL.", "url:str, date:str"),
    T("advanced_search",   "web_search", "Web search with boolean operators, site filters, and date ranges.", "query:str, site:str, after:str, before:str"),
    T("search_suggest",    "web_search", "Return autocomplete search suggestions for a partial query.", "prefix:str, n:int"),
    # calendar
    T("create_event",      "calendar", "Create a new calendar event with title, time range, and attendees.", "title:str, start:str, end:str, attendees:list"),
    T("get_events",        "calendar", "Fetch all events within a date range from a calendar.", "start:str, end:str, calendar:str"),
    T("update_event",      "calendar", "Modify an existing calendar event's details or participants.", "event_id:str, title:str, start:str, attendees:list"),
    T("delete_event",      "calendar", "Cancel and remove a calendar event by its ID.", "event_id:str, notify:bool"),
    T("check_availability","calendar", "Check whether a time slot is free across one or more calendars.", "start:str, end:str, attendees:list"),
    T("set_reminder",      "calendar", "Set a timed reminder tied to an event or deadline.", "message:str, at:str, method:str"),
    T("share_calendar",    "calendar", "Share a calendar with another user with given permissions.", "calendar:str, with_:list, perm:str"),
    T("get_recurring",     "calendar", "List all recurring event series and their recurrence rules.", "calendar:str, active_only:bool"),
    T("export_calendar",   "calendar", "Export calendar data to iCal, CSV, or JSON format.", "calendar:str, fmt:str, range:str"),
    T("sync_calendar",     "calendar", "Synchronise a local calendar with an external provider.", "calendar:str, provider:str, direction:str"),
    # email
    T("send_email",        "email", "Compose and send an email to one or more recipients.", "to:list, subject:str, body:str, cc:list, attachments:list"),
    T("read_email",        "email", "Open and return the full content of an email by its ID.", "email_id:str, mark_read:bool"),
    T("search_emails",     "email", "Search inbox by sender, subject, or body keyword.", "query:str, folder:str, limit:int, since:str"),
    T("reply_email",       "email", "Send a reply within an existing email thread.", "email_id:str, body:str, reply_all:bool"),
    T("delete_email",      "email", "Permanently delete or trash an email message.", "email_id:str, permanent:bool"),
    T("forward_email",     "email", "Forward an existing email to new recipients.", "email_id:str, to:list, note:str"),
    T("create_draft",      "email", "Save an email as a draft without sending it.", "to:list, subject:str, body:str"),
    T("get_attachments",   "email", "Download all file attachments from an email.", "email_id:str, dest:str"),
    T("label_email",       "email", "Apply or remove labels and tags on an email message.", "email_id:str, add:list, remove:list"),
    T("unsubscribe_email", "email", "Unsubscribe from a mailing list linked to an email.", "email_id:str, confirm:bool"),
    # code_exec
    T("execute_python",    "code_exec", "Run Python source code and capture stdout, stderr, return value.", "code:str, timeout:int, env:dict"),
    T("execute_bash",      "code_exec", "Execute a shell command and capture its combined output.", "cmd:str, cwd:str, env:dict, timeout:int"),
    T("run_tests",         "code_exec", "Execute a test suite with pytest or unittest and report results.", "path:str, markers:list, verbose:bool"),
    T("install_package",   "code_exec", "Install a Python library via pip or conda.", "package:str, version:str, manager:str"),
    T("format_code",       "code_exec", "Auto-format source files using black, ruff, or prettier.", "path:str, lang:str, check_only:bool"),
    T("lint_code",         "code_exec", "Run a linter and return a list of violations.", "path:str, rules:list, strict:bool"),
    T("profile_code",      "code_exec", "Profile Python execution and return hotspot statistics.", "code:str, sort_by:str, top_n:int"),
    T("debug_code",        "code_exec", "Attach a debugger and step through code to find issues.", "path:str, breakpoints:list, args:list"),
    T("execute_notebook",  "code_exec", "Execute a Jupyter notebook end-to-end and return cell outputs.", "path:str, timeout:int, params:dict"),
    T("containerize_app",  "code_exec", "Build a Docker image for an application and optionally push it.", "path:str, tag:str, push:bool, registry:str"),
    # database
    T("query_database",    "database", "Run a SQL SELECT and return rows as a list of dicts.", "sql:str, db:str, params:list"),
    T("insert_record",     "database", "Insert one or more rows into a database table.", "table:str, rows:list, db:str"),
    T("update_record",     "database", "Update table rows matching a WHERE condition.", "table:str, values:dict, where:str, db:str"),
    T("delete_record",     "database", "Delete rows matching a condition from a table.", "table:str, where:str, db:str"),
    T("list_tables",       "database", "List all tables and their column schemas in a database.", "db:str, schema:str"),
    T("create_table",      "database", "Create a new table with the specified column definitions.", "table:str, columns:list, db:str"),
    T("drop_table",        "database", "Drop a table and all its data permanently.", "table:str, db:str, cascade:bool"),
    T("create_index",      "database", "Create an index on one or more columns to speed queries.", "table:str, columns:list, unique:bool, db:str"),
    T("backup_database",   "database", "Create a full or incremental backup of a database.", "db:str, dest:str, incremental:bool"),
    T("restore_database",  "database", "Restore a database from a backup file.", "db:str, backup:str, drop_existing:bool"),
    # communication
    T("send_slack",          "communication", "Send a message to a Slack channel or direct message.", "channel:str, text:str, attachments:list"),
    T("create_slack_channel","communication", "Create a new Slack channel with given name and members.", "name:str, members:list, private:bool"),
    T("send_teams",          "communication", "Post a message to a Microsoft Teams channel or chat.", "channel:str, text:str, card:dict"),
    T("post_discord",        "communication", "Send a message or embed to a Discord channel.", "channel_id:str, content:str, embed:dict"),
    T("send_sms",            "communication", "Send an SMS text message to a phone number.", "to:str, body:str"),
    T("make_call",           "communication", "Initiate a phone or VoIP call to a contact.", "to:str, from_:str, record:bool"),
    T("schedule_video_call", "communication", "Schedule a video conference call and return a join link.", "title:str, start:str, attendees:list, platform:str"),
    T("create_zoom",         "communication", "Create a Zoom meeting and return the join URL.", "topic:str, start:str, duration:int, password:str"),
    T("send_webhook",        "communication", "POST a JSON payload to an external webhook endpoint.", "url:str, payload:dict, headers:dict"),
    T("broadcast",           "communication", "Send a notification to all members of a topic or group.", "topic:str, message:str, priority:str"),
    # monitoring
    T("get_metrics",      "monitoring", "Retrieve system or application metrics for a time range.", "service:str, metric:str, start:str, end:str"),
    T("check_health",     "monitoring", "Run a health check against a service or HTTP endpoint.", "service:str, timeout:int"),
    T("query_logs",       "monitoring", "Query structured logs with filters for time, level, and text.", "service:str, level:str, since:str, query:str, limit:int"),
    T("set_alert",        "monitoring", "Create or update a metric threshold alert rule.", "metric:str, threshold:float, op:str, notify:list"),
    T("get_error_rate",   "monitoring", "Return the error rate for a service over a rolling window.", "service:str, window:str"),
    T("monitor_latency",  "monitoring", "Retrieve p50/p95/p99 latency for a service or endpoint.", "service:str, endpoint:str, window:str"),
    T("track_resources",  "monitoring", "Track CPU, memory, disk, and network for a host over time.", "host:str, interval:int, duration:int"),
    T("get_dashboard",    "monitoring", "Fetch the current state of a Grafana or Datadog dashboard.", "dashboard:str, from_:str, to:str"),
    T("create_incident",  "monitoring", "Open a new incident for an ongoing outage or degradation.", "title:str, severity:str, services:list, description:str"),
    T("resolve_incident", "monitoring", "Mark an incident as resolved and record a post-mortem link.", "incident_id:str, resolution:str, postmortem:str"),
    # ml_ops
    T("train_model",           "ml_ops", "Launch a model training job with the given config and dataset.", "config:str, dataset:str, output:str, gpu:int"),
    T("evaluate_model",        "ml_ops", "Evaluate a model on a test dataset and return metric scores.", "model:str, dataset:str, metrics:list"),
    T("deploy_model",          "ml_ops", "Deploy a trained model to a serving endpoint.", "model:str, endpoint:str, replicas:int, framework:str"),
    T("get_model_metrics",     "ml_ops", "Fetch live serving metrics (latency, throughput) for a model.", "model:str, endpoint:str, window:str"),
    T("log_experiment",        "ml_ops", "Log parameters, metrics, and artifacts to an experiment tracker.", "run_id:str, params:dict, metrics:dict, artifacts:list"),
    T("register_model",        "ml_ops", "Register a model version in the model registry with metadata.", "model_path:str, name:str, tags:dict, stage:str"),
    T("feature_importance",    "ml_ops", "Compute feature importance scores for a trained model.", "model:str, method:str, top_n:int"),
    T("hyperparameter_search", "ml_ops", "Run a hyperparameter sweep using grid, random, or Bayesian search.", "config:str, search:dict, trials:int, objective:str"),
    T("compare_models",        "ml_ops", "Compare evaluation metrics across multiple model versions.", "models:list, dataset:str, metrics:list"),
    T("rollback_model",        "ml_ops", "Roll back a serving endpoint to a previous model version.", "endpoint:str, version:str"),
    # devops
    T("deploy_service",      "devops", "Deploy or update a service in a Kubernetes cluster or cloud.", "service:str, image:str, env:dict, replicas:int"),
    T("rollback_deployment", "devops", "Roll back a service to its previous stable deployment revision.", "service:str, revision:str"),
    T("scale_service",       "devops", "Adjust the replica count for a running service.", "service:str, replicas:int, namespace:str"),
    T("deployment_status",   "devops", "Get the current rollout status and pod health of a service.", "service:str, namespace:str"),
    T("create_pipeline",     "devops", "Create a new CI/CD pipeline from a config file.", "name:str, config:str, triggers:list"),
    T("trigger_build",       "devops", "Manually trigger a CI/CD build for a branch or commit.", "repo:str, branch:str, pipeline:str"),
    T("get_build_logs",      "devops", "Stream or return logs from a CI/CD pipeline build.", "build_id:str, stage:str, tail:int"),
    T("manage_secrets",      "devops", "Create, rotate, or delete secrets in a secrets manager.", "secret:str, action:str, value:str, scope:str"),
    T("update_config",       "devops", "Update a service's runtime configuration or environment variables.", "service:str, config:dict, restart:bool"),
    T("restart_service",     "devops", "Restart one or more instances of a running service gracefully.", "service:str, namespace:str, graceful:bool"),
]

assert len(TOOLS) == 100
TOOL_NAMES     = [t["name"] for t in TOOLS]

# ── 200 Queries ────────────────────────────────────────────────────────────────
# 20 per category: indices [0:16]=train, [16:20]=test per category block.
# Test tools (all appear in training):
#  file_ops:      read_file, write_file, compress_file, get_file_metadata
#  web_search:    web_search, fetch_url, search_news, academic_search
#  calendar:      create_event, get_events, check_availability, set_reminder
#  email:         send_email, read_email, search_emails, reply_email
#  code_exec:     execute_python, execute_bash, run_tests, install_package
#  database:      query_database, insert_record, list_tables, backup_database
#  communication: send_slack, send_teams, schedule_video_call, send_webhook
#  monitoring:    get_metrics, query_logs, set_alert, create_incident
#  ml_ops:        train_model, evaluate_model, log_experiment, deploy_model
#  devops:        deploy_service, scale_service, trigger_build, rollback_deployment

QUERIES = [
    # file_ops train (16)
    ("Open and display the nginx config at /etc/nginx/nginx.conf",             "read_file"),
    ("Persist the processed result dictionary to /tmp/output.json",            "write_file"),
    ("What is inside the models/ directory including nested folders?",         "list_directory"),
    ("Purge all __pycache__ directories from the repository tree",             "delete_file"),
    ("Duplicate production.env to staging.env for the new environment",        "copy_file"),
    ("Rename experiment_v1/ to experiment_final/ inside results/",             "move_file"),
    ("Pack the entire dataset/ folder into a tarball for the data team",       "compress_file"),
    ("Unpack the vendor.zip archive that arrived into the lib/ directory",     "decompress_file"),
    ("Alert me whenever anything under configs/ is modified on disk",          "watch_file"),
    ("How large is checkpoint.pt and when was it last written to?",            "get_file_metadata"),
    ("Print the full text of the pipeline YAML configuration file",            "read_file"),
    ("Write the aggregated metrics dictionary out to results/summary.json",    "write_file"),
    ("Show every Python file recursively under src/ matching *.py",            "list_directory"),
    ("Delete the stale lock file left behind after the process crashed",       "delete_file"),
    ("Make a copy of the baseline model weights for the ablation study",       "copy_file"),
    ("Transfer the trained model checkpoint to the shared NFS mount",          "move_file"),
    # file_ops test (4)
    ("Show me the full content of the CHANGELOG file",                         "read_file"),
    ("Dump the batch inference predictions to output/predictions.csv",         "write_file"),
    ("Create a compressed archive of last month's application logs",           "compress_file"),
    ("What is the file size and modification time of the binary checkpoint?",  "get_file_metadata"),
    # web_search train (16)
    ("Find the latest benchmarks comparing GPT-4o and Claude 3 Opus",         "web_search"),
    ("Grab the raw HTML from that documentation page at the given URL",        "fetch_url"),
    ("What has been written about the SVB collapse in the past 48 hours?",    "search_news"),
    ("Find photos of gradient descent illustrated as a loss landscape surface","image_search"),
    ("Locate peer-reviewed papers on in-context learning from 2023 onwards",   "academic_search"),
    ("Search for tutorial videos on diffusion model fine-tuning",              "video_search"),
    ("What are people on Twitter saying about the new Mistral model release?", "social_search"),
    ("Get the cached Google version of that page before it went offline",      "cached_fetch"),
    ("Search site:arxiv.org for RLHF papers published after January 2022",    "advanced_search"),
    ("What autocomplete suggestions come up for transformer architecture?",     "search_suggest"),
    ("Look up current open-LLM rankings on the LMSYS Chatbot Arena leaderboard","web_search"),
    ("Pull the content from the remote API documentation endpoint",            "fetch_url"),
    ("Summarize news coverage of the EU AI Act from the past two weeks",       "search_news"),
    ("Find images of mechanistic interpretability circuit diagrams",           "image_search"),
    ("Search scholar for Anthropic's papers on constitutional AI",             "academic_search"),
    ("Find video walkthroughs of implementing LoRA from scratch in PyTorch",   "video_search"),
    # web_search test (4)
    ("What do search results say about the latest Llama 3 model capabilities?","web_search"),
    ("Download and return the content of the remote config file at that URL",  "fetch_url"),
    ("Get recent news coverage of the NVIDIA H100 GPU supply constraints",     "search_news"),
    ("Find published academic research on speculative decoding for LLMs",      "academic_search"),
    # calendar train (16)
    ("Book a 60-minute sprint planning session with the team Monday at 10am",  "create_event"),
    ("What is on my schedule for this coming Thursday?",                       "get_events"),
    ("Move the Friday retrospective to start at 4pm instead of 3pm",          "update_event"),
    ("Cancel the one-on-one with Alex that I have tomorrow afternoon",         "delete_event"),
    ("Is the whole team free for a 90-minute deep-dive next Tuesday morning?", "check_availability"),
    ("Remind me 15 minutes before the board call so I can pull up the slides", "set_reminder"),
    ("Give my manager view-only access to my engineering calendar",            "share_calendar"),
    ("List all the weekly syncs that repeat on my calendar",                   "get_recurring"),
    ("Export my Q3 calendar as an iCal file to import into another app",       "export_calendar"),
    ("Sync my work Google Calendar with the Outlook calendar",                 "sync_calendar"),
    ("Add a two-hour design review with the product team Wednesday 2pm",       "create_event"),
    ("Pull all my meetings between the 10th and 20th of this month",          "get_events"),
    ("Update the all-hands event description with the new agenda link",        "update_event"),
    ("Remove the offsite event cancelled due to travel restrictions",          "delete_event"),
    ("Check whether Thursday at 10am works for a three-person interview panel","check_availability"),
    ("Set an alert to ping me 30 minutes before the customer demo starts",    "set_reminder"),
    # calendar test (4)
    ("Schedule a product kickoff meeting with the full team next Monday 2pm",  "create_event"),
    ("Show me everything I have scheduled for the rest of this week",          "get_events"),
    ("Find a free 45-minute window for three attendees sometime tomorrow",     "check_availability"),
    ("Set a repeating reminder to send the weekly report every Friday 5pm",   "set_reminder"),
    # email train (16)
    ("Fire off the release announcement to the entire mailing list right now", "send_email"),
    ("Open the message from the VP about the company reorg",                   "read_email"),
    ("Find all threads from vendors mentioning unpaid invoices",               "search_emails"),
    ("Reply to the client thread asking for the revised project timeline",     "reply_email"),
    ("Move the 500 promotional emails clogging my inbox straight to trash",   "delete_email"),
    ("Pass the compliance report along to the legal team address",             "forward_email"),
    ("Draft a follow-up message to the candidate but hold off on sending it", "create_draft"),
    ("Save the PDF that came attached to the contract email to my desktop",   "get_attachments"),
    ("Tag every message from the CTO as high priority",                        "label_email"),
    ("Opt me out of the marketing newsletter I keep receiving each week",      "unsubscribe_email"),
    ("Send the post-mortem summary to the on-call distribution list",          "send_email"),
    ("Read the overnight message that came in from the Singapore office",      "read_email"),
    ("Find all threads where I was CC'd during the last 30 days",              "search_emails"),
    ("Write back to the board member asking to reschedule our call",           "reply_email"),
    ("Trash the newsletter flood without opening any of them",                 "delete_email"),
    ("Forward the security audit report to the CISO team inbox",               "forward_email"),
    # email test (4)
    ("Send the incident post-mortem summary to the engineering leadership list","send_email"),
    ("What did Sarah write about the Q4 product roadmap last week?",           "read_email"),
    ("Search for all unread messages containing the phrase deployment failure", "search_emails"),
    ("Write back to the recruiter with my feedback on the interview",          "reply_email"),
    # code_exec train (16)
    ("Run this data cleaning script and show me what it outputs",              "execute_python"),
    ("Execute git log --oneline -20 in the repository root",                   "execute_bash"),
    ("Run the full test suite for the auth module and tell me what fails",     "run_tests"),
    ("Add the torch==2.3.0 dependency to this Python environment",             "install_package"),
    ("Clean up the messy import ordering across all the Python source files",  "format_code"),
    ("Check the type annotations in the inference pipeline for type errors",   "lint_code"),
    ("Profile the tokenizer encode method to find where time is being spent",  "profile_code"),
    ("Step through the gradient accumulation loop to see where NaN appears",   "debug_code"),
    ("Execute the EDA notebook end to end and capture every cell output",      "execute_notebook"),
    ("Package the inference service into a Docker image and push to ECR",      "containerize_app"),
    ("Execute the script that computes BLEU scores for the translation outputs","execute_python"),
    ("Run make build in the project root and show me the full output",         "execute_bash"),
    ("Execute only the integration tests with verbose pass/fail output",       "run_tests"),
    ("Install scikit-learn version 1.4 using conda in the ml environment",     "install_package"),
    ("Auto-format the newly generated API client code with black",             "format_code"),
    ("Scan the REST API handler files for unused imports and dead code paths",  "lint_code"),
    # code_exec test (4)
    ("Run the evaluation harness that benchmarks the new model accuracy",      "execute_python"),
    ("Execute df -h to check disk usage across partitions on the training server","execute_bash"),
    ("Run the regression tests for the newly added retrieval module",          "run_tests"),
    ("Install the latest stable version of the transformers library",          "install_package"),
    # database train (16)
    ("Fetch all users who registered in the last 30 days ordered by signup date","query_database"),
    ("Add the new experiment run results into the runs table",                 "insert_record"),
    ("Mark all open support tickets older than 90 days as stale",             "update_record"),
    ("Remove all rows from the temp_cache table older than yesterday",         "delete_record"),
    ("What tables exist in the analytics database and what columns do they have?","list_tables"),
    ("Create a new feature_store table with the schema I have here",           "create_table"),
    ("Drop the deprecated shadow_users table that we no longer need",          "drop_table"),
    ("Add an index on user_id in the events table to speed up lookups",        "create_index"),
    ("Take a full backup of the production database before the migration",     "backup_database"),
    ("Restore the database from the automated backup that ran last night",     "restore_database"),
    ("Count inference predictions grouped by output label in the results table","query_database"),
    ("Insert this batch of 500 new request records into the job queue table",  "insert_record"),
    ("Update the status field to completed for all rows where job is finished","update_record"),
    ("Purge all expired session tokens from the auth_sessions table",          "delete_record"),
    ("List all tables in the warehouse and their approximate row counts",      "list_tables"),
    ("Build a composite index on tenant_id and created_at in the events table","create_index"),
    # database test (4)
    ("Retrieve average API latency per endpoint for the past week from the db","query_database"),
    ("Write this batch of embedding vectors into the vector store table",      "insert_record"),
    ("Show me every table and its schema in the reporting database",           "list_tables"),
    ("Snapshot the production database before tonight's schema migration runs","backup_database"),
    # communication train (16)
    ("Post the deployment summary update to the engineering-updates Slack channel","send_slack"),
    ("Create a private Slack channel for the incident response task force",    "create_slack_channel"),
    ("Share the sprint review recap in the team's Microsoft Teams channel",    "send_teams"),
    ("Announce the new open-source release in the general Discord server",     "post_discord"),
    ("Text the on-call engineer that the latency alert has fired",             "send_sms"),
    ("Call the security lead to brief them on the breach before the press call","make_call"),
    ("Set up a video call for the architecture review session tomorrow 3pm",   "schedule_video_call"),
    ("Create a Zoom meeting for the external partner product demo",             "create_zoom"),
    ("Trigger the downstream CI webhook on merge to main",                     "send_webhook"),
    ("Push a high-priority alert to all subscribers of the outage topic",      "broadcast"),
    ("Message the data team on Slack with the anomaly detection report",       "send_slack"),
    ("Post the infrastructure status update in the Teams engineering channel", "send_teams"),
    ("Drop a note in the incidents Discord channel with the current status",   "post_discord"),
    ("SMS the field technicians the updated meeting location",                 "send_sms"),
    ("Kick off a video conference for the post-mortem retrospective tomorrow", "schedule_video_call"),
    ("Ping the alerting webhook every time a new error batch is detected",     "send_webhook"),
    # communication test (4)
    ("Notify the platform team in Slack about the degraded service performance","send_slack"),
    ("Post the infrastructure rollout update to the Teams engineering channel","send_teams"),
    ("Book a video conference for tomorrow's incident review session at 10am", "schedule_video_call"),
    ("Hit the integration webhook to notify the partner system of the event",  "send_webhook"),
    # monitoring train (16)
    ("Pull CPU and memory utilization for the inference cluster over the last hour","get_metrics"),
    ("Is the payment service endpoint currently returning 200 responses?",     "check_health"),
    ("Fetch all ERROR-level log entries from the API server since midnight",   "query_logs"),
    ("Create an alert that fires if queue depth stays above 1000 for 5 min",  "set_alert"),
    ("What has the 5xx error rate been for the recommendations service today?","get_error_rate"),
    ("Get p95 latency for the /predict endpoint over the past 24 hours",       "monitor_latency"),
    ("Show disk and network I/O for the training nodes over the past day",     "track_resources"),
    ("Open the production overview dashboard in Grafana",                      "get_dashboard"),
    ("File a SEV-2 incident for the checkout flow performance degradation",    "create_incident"),
    ("Close out the database connection timeout incident from last night",     "resolve_incident"),
    ("Retrieve throughput metrics for the embedding service over the last 6h", "get_metrics"),
    ("Check whether the feature-flag service is healthy and accepting requests","check_health"),
    ("Search the logs for any out-of-memory errors in today's training job",   "query_logs"),
    ("Alert me if inference latency exceeds 500ms for more than 2 minutes",   "set_alert"),
    ("What has the error rate on the search API been doing over this week?",   "get_error_rate"),
    ("Get median and tail latencies for every microservice in the cluster",    "monitor_latency"),
    # monitoring test (4)
    ("Show GPU utilization across the training fleet for the past hour",       "get_metrics"),
    ("Pull the last 500 warning and error log lines from the serving fleet",   "query_logs"),
    ("Create an alert that triggers when model accuracy drops below 90%",      "set_alert"),
    ("Open a SEV-1 incident for the payment gateway outage affecting all users","create_incident"),
    # ml_ops train (16)
    ("Kick off a fine-tuning run on the summarization dataset with this config","train_model"),
    ("Evaluate the new checkpoint against the held-out validation partition",  "evaluate_model"),
    ("Push the production-ready model to the live serving endpoint",           "deploy_model"),
    ("How is the currently-deployed classifier performing in production now?", "get_model_metrics"),
    ("Record the hyperparameters and validation loss for this training run",   "log_experiment"),
    ("Register this checkpoint as version 3 of the sentiment classifier",     "register_model"),
    ("Which input features matter most for the churn prediction model?",       "feature_importance"),
    ("Run a random search over learning rate and batch size for 50 trials",    "hyperparameter_search"),
    ("Show how the new model compares to the two previous production versions","compare_models"),
    ("The new model is broken, roll the endpoint back to the last stable version","rollback_model"),
    ("Start a training job for the retrieval model on the expanded document corpus","train_model"),
    ("Benchmark the latest candidate model on the toxicity detection test set","evaluate_model"),
    ("Ship the updated ranking model to the recommendation serving endpoint",  "deploy_model"),
    ("What are the live latency and request volume stats for the deployed NER model?","get_model_metrics"),
    ("Log the BLEU score and all config hyperparameters for this translation run","log_experiment"),
    ("Add the fine-tuned LLM to the registry as a candidate for A/B testing", "register_model"),
    # ml_ops test (4)
    ("Launch a training job for the intent classifier using the new labeled data","train_model"),
    ("Measure the F1 score and precision-recall curve on the held-out test set","evaluate_model"),
    ("Track this experiment config metrics and artifacts in the experiment tracker","log_experiment"),
    ("Promote the best candidate model to the production serving endpoint",    "deploy_model"),
    # devops train (16)
    ("Push the new container image to Kubernetes and update the serving pods", "deploy_service"),
    ("The latest release broke prod, revert to the previous working revision", "rollback_deployment"),
    ("Bump the API gateway from 3 replicas to 10 to handle the traffic spike", "scale_service"),
    ("What is the current rollout status for the recommendation service pods?","deployment_status"),
    ("Create a GitHub Actions pipeline that triggers on every push to main",   "create_pipeline"),
    ("Manually kick off the CI build for the hotfix branch right now",         "trigger_build"),
    ("Fetch the full build output for pipeline job 4821 in the data workflow", "get_build_logs"),
    ("Rotate the database password and update the entry in the secrets manager","manage_secrets"),
    ("Set FEATURE_FLAG_NEW_CHECKOUT to true in the production environment",    "update_config"),
    ("Bounce the worker pods so they pick up the updated configuration",       "restart_service"),
    ("Deploy the updated model server image into the inference Kubernetes namespace","deploy_service"),
    ("Something is broken in head, roll the API back to the last green build", "rollback_deployment"),
    ("The load balancer is saturated, scale the backend service to 20 replicas","scale_service"),
    ("Are all the pods for the payment processing service running and healthy?","deployment_status"),
    ("Wire up an automated build that runs whenever a pull request is merged",  "create_pipeline"),
    ("Kick off a build for the release candidate on the v2.1 branch right now","trigger_build"),
    # devops test (4)
    ("Roll out the new inference container image to the production cluster",   "deploy_service"),
    ("Autoscale the serving fleet upward to handle the sudden traffic surge",  "scale_service"),
    ("Manually trigger the integration test pipeline for this specific commit","trigger_build"),
    ("Something is wrong with the frontend, roll it back to the last stable tag","rollback_deployment"),
]

assert len(QUERIES) == 200

# build train / test splits: last 4 of every block of 20 = test
_train_idx, _test_idx = [], []
for _cat in range(10):
    _b = _cat * 20
    _train_idx.extend(range(_b, _b + 16))
    _test_idx.extend(range(_b + 16, _b + 20))

TRAIN_Q = [{"q": QUERIES[i][0], "tool": QUERIES[i][1]} for i in _train_idx]
TEST_Q  = [{"q": QUERIES[i][0], "tool": QUERIES[i][1]} for i in _test_idx]
assert len(TRAIN_Q) == 160 and len(TEST_Q) == 40

_train_tools = {q["tool"] for q in TRAIN_Q}
_test_tools  = {q["tool"] for q in TEST_Q}
_missing = _test_tools - _train_tools
assert not _missing, f"Test tools missing from training: {_missing}"

# ── Helpers ────────────────────────────────────────────────────────────────────

def tool_to_text(t: dict) -> str:
    return f"{t['name']}: {t['desc']} [{t['params']}]"

ALL_TOOL_TEXTS = [tool_to_text(t) for t in TOOLS]
ALL_TOOL_TEXT  = "\n".join(ALL_TOOL_TEXTS)

def count_tokens(tok, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))

def find_tool(text: str) -> str:
    low = text.lower()
    for name in TOOL_NAMES:
        if name in low:
            return name
    first = low.strip().split()[0].strip(".,;:'\"") if low.strip() else ""
    return first if first in TOOL_NAMES else "UNKNOWN"

def llm_generate(model, tok, system: str, user: str, max_new: int = 24) -> str:
    msgs   = [{"role": "system", "content": system},
              {"role": "user",   "content": user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    n_in   = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0, n_in:], skip_special_tokens=True).strip()

def last_token_vec(model, tok, text: str) -> np.ndarray:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    return out.hidden_states[-1][0, -1, :].cpu().float().numpy()

def compute_metrics(y_true, y_pred, tokens_list, lat_list):
    labels = sorted(set(y_true))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "acc":    accuracy_score(y_true, y_pred),
        "prec":   prec, "rec": rec, "f1": f1,
        "tokens": float(np.mean(tokens_list)),
        "lat_ms": float(np.mean(lat_list)) * 1000,
    }

def print_grid(headers, rows):
    widths = [max(len(str(x)) for x in [h] + [r[i] for r in rows])
              for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    fmt = lambda r: "| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |"
    print(sep); print(fmt(headers)); print(sep)
    for row in rows: print(fmt(row))
    print(sep)

# ── System prompts ─────────────────────────────────────────────────────────────
SYS_SELECT   = "You are a tool selector. Respond with ONLY the exact tool name, nothing else."
SYS_PROPOSE  = "You are a tool selector assistant. Given a user query and a list of candidate tools, output ONLY the single best tool name, nothing else."
SYS_CRITIQUE = ("You are a senior tool selector reviewing a junior model's suggestion. "
                "Output ONLY the correct tool name. Confirm the proposal if right; correct it if wrong.")

# ── Approach 1: Naive ──────────────────────────────────────────────────────────
def run_naive(model_large, tok_large) -> dict:
    schema_toks = count_tokens(tok_large, ALL_TOOL_TEXT)
    y_true, y_pred, lats = [], [], []
    for item in TEST_Q:
        user = f"Tools:\n{ALL_TOOL_TEXT}\n\nQuery: {item['q']}\n\nTool name:"
        t0   = time.perf_counter()
        raw  = llm_generate(model_large, tok_large, SYS_SELECT, user)
        lats.append(time.perf_counter() - t0)
        pred = find_tool(raw)
        y_true.append(item["tool"]); y_pred.append(pred)
        print(f"    {'OK' if pred==item['tool'] else '--'}  gt={item['tool']:<24} pred={pred}")
    m = compute_metrics(y_true, y_pred, [schema_toks]*len(lats), lats)
    m["y_true"] = y_true; m["y_pred"] = y_pred
    return m

# ── Approach 2: RAG ────────────────────────────────────────────────────────────
def run_rag(model_large, tok_large, embed_model) -> dict:
    tool_embs = embed_model.encode(ALL_TOOL_TEXTS, normalize_embeddings=True)
    y_true, y_pred, lats, toks = [], [], [], []
    for item in TEST_Q:
        t0   = time.perf_counter()
        q_e  = embed_model.encode([item["q"]], normalize_embeddings=True)
        idxs = np.argsort((q_e @ tool_embs.T).squeeze())[::-1][:RAG_TOP_K]
        ctx  = "\n".join(ALL_TOOL_TEXTS[i] for i in idxs)
        user = f"Tools:\n{ctx}\n\nQuery: {item['q']}\n\nTool name:"
        raw  = llm_generate(model_large, tok_large, SYS_SELECT, user)
        lats.append(time.perf_counter() - t0)
        pred = find_tool(raw)
        toks.append(count_tokens(tok_large, ctx))
        y_true.append(item["tool"]); y_pred.append(pred)
        retrieved = [TOOLS[i]["name"] for i in idxs]
        hit = "in" if item["tool"] in retrieved else "MISS"
        print(f"    {'OK' if pred==item['tool'] else '--'}  gt={item['tool']:<24} pred={pred:<24} ret={hit}")
    m = compute_metrics(y_true, y_pred, toks, lats)
    m["y_true"] = y_true; m["y_pred"] = y_pred
    return m

# ── Approach 3: Agent Swarm ────────────────────────────────────────────────────
def run_swarm(model_small, tok_small, model_large, tok_large, embed_model) -> dict:
    tool_embs = embed_model.encode(ALL_TOOL_TEXTS, normalize_embeddings=True)

    # train hidden-state probe (Agent 3) on 160 training queries
    print(f"    [probe] extracting hidden states for {N_TRAIN} training queries ...")
    X_train, y_train = [], []
    for i, item in enumerate(TRAIN_Q):
        X_train.append(last_token_vec(model_large, tok_large, item["q"]))
        y_train.append(item["tool"])
        if (i + 1) % 40 == 0:
            print(f"      {i+1}/{N_TRAIN}")
    X_train = np.array(X_train)
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    clf.fit(X_train, y_train)
    print(f"    [probe] ready — {len(set(y_train))} classes")

    y_true, y_pred = [], []
    a1_ok, a2_ok, a3_ok = [], [], []
    agreements, lats, toks_all = [], [], []

    for item in TEST_Q:
        t0 = time.perf_counter()

        # shared RAG retrieval
        q_e  = embed_model.encode([item["q"]], normalize_embeddings=True)
        idxs = np.argsort((q_e @ tool_embs.T).squeeze())[::-1][:RAG_TOP_K]
        ctx  = "\n".join(ALL_TOOL_TEXTS[i] for i in idxs)
        ctx_toks = count_tokens(tok_small, ctx)

        # Agent 1 — 0.5B proposer
        a1_raw  = llm_generate(model_small, tok_small, SYS_PROPOSE,
                               f"Tools:\n{ctx}\n\nQuery: {item['q']}\n\nBest tool:")
        a1_pred = find_tool(a1_raw)

        # Agent 2 — 1.5B critic (sees A1 proposal)
        a2_raw  = llm_generate(model_large, tok_large, SYS_CRITIQUE,
                               f"Tools:\n{ctx}\n\nQuery: {item['q']}\n\n"
                               f"Proposed: {a1_pred}\n\nCorrect tool name:")
        a2_pred = find_tool(a2_raw)
        a2_toks = count_tokens(tok_large, ctx + " " + a1_pred)

        # Agent 3 — probe voter (no schema in context)
        feat    = last_token_vec(model_large, tok_large, item["q"])
        a3_pred = clf.predict([feat])[0]
        a3_toks = count_tokens(tok_large, item["q"])

        lat = time.perf_counter() - t0

        # majority vote; 3-way tie -> trust critic
        votes   = [a1_pred, a2_pred, a3_pred]
        cnt     = Counter(votes)
        top_v, top_c = cnt.most_common(1)[0]
        final   = top_v if top_c >= 2 else a2_pred

        total_toks = ctx_toks + a2_toks + a3_toks
        gt = item["tool"]

        y_true.append(gt); y_pred.append(final)
        a1_ok.append(int(a1_pred == gt))
        a2_ok.append(int(a2_pred == gt))
        a3_ok.append(int(a3_pred == gt))
        agreements.append(top_c)
        lats.append(lat); toks_all.append(total_toks)

        mark = "OK" if final == gt else "--"
        print(f"    {mark}  gt={gt:<22} A1={a1_pred:<22} A2={a2_pred:<22} A3={a3_pred:<22} => {final}")

    m = compute_metrics(y_true, y_pred, toks_all, lats)
    m.update({
        "y_true": y_true, "y_pred": y_pred,
        "a1_acc":          float(np.mean(a1_ok)),
        "a2_acc":          float(np.mean(a2_ok)),
        "a3_acc":          float(np.mean(a3_ok)),
        "full_agree_rate": float(np.mean([a == 3 for a in agreements])),
        "maj_agree_rate":  float(np.mean([a >= 2 for a in agreements])),
    })
    return m

# ── Per-category breakdown ─────────────────────────────────────────────────────
CATS = ["file_ops","web_search","calendar","email","code_exec",
        "database","communication","monitoring","ml_ops","devops"]

def per_cat_acc(y_true, y_pred):
    out = []
    for i, cat in enumerate(CATS):
        yt = y_true[i*4:(i+1)*4]; yp = y_pred[i*4:(i+1)*4]
        ok = sum(a == b for a, b in zip(yt, yp))
        out.append((cat, f"{ok}/4", f"{ok/4:.0%}"))
    return out

# ── Results table ──────────────────────────────────────────────────────────────
def print_results(naive_r, rag_r, swarm_r):
    W = 76
    print("\n" + "=" * W)
    print(f"  BENCHMARK — 100 tools | 10 categories | {N_TEST}-query test set | macro-avg metrics")
    print("=" * W)

    print("\nApproach comparison")
    print_grid(
        ["Approach",           "Acc",  "Prec(M)","Rec(M)", "F1(M)", "Tokens/q","ms/q"],
        [
            ["Naive (100 schemas)", f"{naive_r['acc']:.0%}",
             f"{naive_r['prec']:.3f}", f"{naive_r['rec']:.3f}", f"{naive_r['f1']:.3f}",
             f"{naive_r['tokens']:.0f}", f"{naive_r['lat_ms']:.0f}"],
            [f"RAG (top-{RAG_TOP_K})",       f"{rag_r['acc']:.0%}",
             f"{rag_r['prec']:.3f}",   f"{rag_r['rec']:.3f}",   f"{rag_r['f1']:.3f}",
             f"{rag_r['tokens']:.0f}",  f"{rag_r['lat_ms']:.0f}"],
            ["Swarm (3-agent)",     f"{swarm_r['acc']:.0%}",
             f"{swarm_r['prec']:.3f}", f"{swarm_r['rec']:.3f}", f"{swarm_r['f1']:.3f}",
             f"{swarm_r['tokens']:.0f}", f"{swarm_r['lat_ms']:.0f}"],
        ]
    )

    print("\nSwarm agent breakdown")
    print_grid(
        ["Agent", "Model",           "Role",     "Individual acc"],
        [
            ["A1",    "Qwen2.5-0.5B",  "Proposer", f"{swarm_r['a1_acc']:.0%}"],
            ["A2",    "Qwen2.5-1.5B",  "Critic",   f"{swarm_r['a2_acc']:.0%}"],
            ["A3",    "Probe@1.5B",    "Voter",    f"{swarm_r['a3_acc']:.0%}"],
            ["Final", "Majority vote", "─",        f"{swarm_r['acc']:.0%}"],
        ]
    )
    print(f"\n  Swarm full agreement  (all 3 match): {swarm_r['full_agree_rate']:.0%}")
    print(f"  Swarm majority agreement (2+/3)   : {swarm_r['maj_agree_rate']:.0%}")

    print("\nPer-category accuracy  (4 test queries each)")
    nc = per_cat_acc(naive_r["y_true"], naive_r["y_pred"])
    rc = per_cat_acc(rag_r["y_true"],   rag_r["y_pred"])
    sc = per_cat_acc(swarm_r["y_true"], swarm_r["y_pred"])
    print_grid(
        ["Category",      "Naive",   "RAG",    "Swarm"],
        [[n[0], n[1]+" "+n[2], r[1]+" "+r[2], s[1]+" "+s[2]]
         for n, r, s in zip(nc, rc, sc)]
    )
    print(f"\n  Tokens/q: Naive={naive_r['tokens']:.0f} (all schemas)  "
          f"RAG={rag_r['tokens']:.0f} (top-{RAG_TOP_K})  "
          f"Swarm={swarm_r['tokens']:.0f} (A1+A2+A3 combined)\n")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*64}")
    print(f"  Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"{'='*64}\n")

    print(f"Loading {MODEL_SMALL} ...")
    tok_small   = AutoTokenizer.from_pretrained(MODEL_SMALL, trust_remote_code=True)
    model_small = AutoModelForCausalLM.from_pretrained(
        MODEL_SMALL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto", trust_remote_code=True,
    ).eval()
    print(f"  {sum(p.numel() for p in model_small.parameters())/1e6:.0f}M params")

    print(f"Loading {MODEL_LARGE} ...")
    tok_large   = AutoTokenizer.from_pretrained(MODEL_LARGE, trust_remote_code=True)
    model_large = AutoModelForCausalLM.from_pretrained(
        MODEL_LARGE,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto", trust_remote_code=True,
    ).eval()
    print(f"  {sum(p.numel() for p in model_large.parameters())/1e6:.0f}M params")

    print(f"Loading {EMBED_MODEL} ...")
    embed_model = SentenceTransformer(EMBED_MODEL)
    print()

    print(f"{'─'*64}")
    print("[1/3] Naive — all 100 schemas in the LLM prompt")
    print(f"{'─'*64}")
    t0 = time.perf_counter()
    naive_r = run_naive(model_large, tok_large)
    print(f"  Done {time.perf_counter()-t0:.1f}s\n")

    print(f"{'─'*64}")
    print(f"[2/3] RAG — top-{RAG_TOP_K} retrieved schemas, then LLM selects")
    print(f"{'─'*64}")
    t0 = time.perf_counter()
    rag_r = run_rag(model_large, tok_large, embed_model)
    print(f"  Done {time.perf_counter()-t0:.1f}s\n")

    print(f"{'─'*64}")
    print("[3/3] Swarm — 0.5B proposes, 1.5B critiques, probe votes, majority wins")
    print(f"{'─'*64}")
    t0 = time.perf_counter()
    swarm_r = run_swarm(model_small, tok_small, model_large, tok_large, embed_model)
    print(f"  Done {time.perf_counter()-t0:.1f}s\n")

    print_results(naive_r, rag_r, swarm_r)

if __name__ == "__main__":
    main()
