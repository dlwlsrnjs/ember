"""Verify privately transferred inputs before preparing the allocation panel."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent


def verify():
    path = ROOT / 'PRIVATE_INPUT_MANIFEST.json'
    if not path.is_file():
        print('Missing private input manifest: extract the separately transferred input archive here.')
        return False
    manifest = json.loads(path.read_text())
    failures = []
    for entry in manifest['files']:
        path = ROOT / entry['path']
        if not path.is_file():
            failures.append((entry['path'], 'missing'))
            continue
        if path.stat().st_size != entry['bytes']:
            failures.append((entry['path'], 'size mismatch'))
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            failures.append((entry['path'], 'SHA256 mismatch'))
    for path, error in failures:
        print(f'{error}: {path}')
    if failures:
        return False
    print(f"Verified {len(manifest['files'])} exact private inputs.")
    return True


if __name__ == '__main__':
    sys.exit(0 if verify() else 1)
