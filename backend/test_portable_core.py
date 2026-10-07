import unittest, subprocess, sys, hashlib, json
from pathlib import Path
class PortableCoreTests(unittest.TestCase):
    def test_bundle_and_example_hashes(self):
        root=Path(__file__).resolve().parents[1]
        subprocess.run([sys.executable,str(root/'scripts/sync_core.py')],check=True)
        example=root/'examples/large_circle_cross'
        manifest=json.loads((example/'provenance.json').read_text())
        for name,digest in manifest['files'].items():
            self.assertEqual(hashlib.sha256((example/name).read_bytes()).hexdigest(),digest)
    def test_default_root_is_bundled(self):
        import os
        from research_bridge import RESEARCH_ROOT
        if not os.environ.get('SEAARRAY_ROOT'):
            self.assertEqual(RESEARCH_ROOT,Path(__file__).resolve().parents[1]/'vendor/seaarray_core')
