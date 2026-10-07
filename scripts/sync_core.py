"""Check or synchronize the exact research primitives bundled with this app."""
from pathlib import Path
import argparse, hashlib, json, shutil
p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--write',action='store_true');a=p.parse_args()
bundle=Path(__file__).resolve().parents[1]/'vendor/seaarray_core'
manifest=json.loads((bundle/'manifest.json').read_text())
if a.write and a.source is None:p.error('--write requires --source')
for rel,expected in manifest['files'].items():
    target=bundle/rel
    if a.source:
        source=a.source.resolve()/rel
        actual=hashlib.sha256(source.read_bytes()).hexdigest()
        if a.write:shutil.copyfile(source,target);manifest['files'][rel]=actual
        elif actual!=expected:raise SystemExit('Research source differs: '+rel)
    if hashlib.sha256(target.read_bytes()).hexdigest()!=manifest['files'][rel]:raise SystemExit('Bundled source differs: '+rel)
if a.write:(bundle/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Verified',len(manifest['files']),'shared source/data files.')
