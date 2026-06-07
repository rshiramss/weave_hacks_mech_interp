import sys, os
sys.path.insert(0, '.')
from scripts.probe_ab_test import (
    load_registry, load_queries, 
    get_or_extract_hidden_states, unload_model
)
import scripts.probe_ab_test as ab

ab.MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
ab.LAYER = 24

_, _, tool_to_agent = load_registry()

for name, path in [
    ('nl', 'data/queries_nl.json'),
    ('descriptive', 'data/queries_descriptive.json')
]:
    print(f'Extracting {name}...')
    queries, _, _ = load_queries(path, tool_to_agent)
    get_or_extract_hidden_states(name, list(queries))
    unload_model()
    print(f'{name} done.')

print('All extractions complete.')
