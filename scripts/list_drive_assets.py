"""Find cloud file paths without downloading data. Example: python scripts/list_drive_assets.py best_state"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('query', nargs='?', default='')
args = parser.parse_args()
inventory = json.loads((Path(__file__).resolve().parents[1] / 'provenance/drive_inventory.json').read_text())
for item in inventory['files']:
    if args.query.lower() in item['drive_path'].lower():
        print(item['drive_path'])
        print('  ' + item['url'])
