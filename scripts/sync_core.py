"""Verify bundle bytes and numerical equivalence to the recorded upstream source.

English comments and the explicitly retained configuration exports differ from
upstream presentation. AST checks ignore docstrings only, never executable code.
Updating the upstream revision requires reviewing and regenerating the manifest.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json


def without_docs(node):
    class StripDocs(ast.NodeTransformer):
        def generic_visit(self, item):
            super().generic_visit(item)
            if isinstance(item, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.body and isinstance(item.body[0], ast.Expr):
                    value = item.body[0].value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        item.body.pop(0)
            return item
    def canonical(item):
        if isinstance(item, ast.AST):
            fields = {key: canonical(value) for key, value in ast.iter_fields(item)
                      if value is not None and value != []}
            return {'node': type(item).__name__, **fields}
        if isinstance(item, list):
            return [canonical(value) for value in item]
        return item
    return json.dumps(canonical(StripDocs().visit(node)), sort_keys=True, separators=(',', ':'))


def semantic_digest(text):
    return hashlib.sha256(without_docs(ast.parse(text)).encode()).hexdigest()


def exported_name(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--write', action='store_true', help='Verify an already reviewed source; never overwrite translated files silently.')
    args = parser.parse_args()
    if args.write and args.source is None:
        parser.error('--write requires --source')
    bundle = Path(__file__).resolve().parents[1] / 'vendor/seaarray_core'
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['files'].items():
        target = bundle / relative
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise SystemExit('Bundled source differs: ' + relative)
        if target.suffix == '.py' and semantic_digest(target.read_text(encoding='utf-8')) != manifest['python_semantics'][relative]:
            raise SystemExit('Bundled executable AST differs: ' + relative)
        if args.source:
            source = args.source.resolve() / relative
            if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['upstream_files'][relative]:
                raise SystemExit('Upstream source changed; review before refreshing the manifest: ' + relative)
            if target.suffix == '.py':
                if relative in manifest['retained_exports']:
                    tree = ast.parse(source.read_text(encoding='utf-8'))
                    for name in manifest['retained_exports'][relative]:
                        node = next(item for item in tree.body if exported_name(item) == name)
                        actual = hashlib.sha256(without_docs(node).encode()).hexdigest()
                        if actual != manifest['retained_export_semantics'][name]:
                            raise SystemExit('Retained export differs: ' + name)
                elif semantic_digest(source.read_text(encoding='utf-8')) != manifest['python_semantics'][relative]:
                    raise SystemExit('Numerical source differs: ' + relative)
    print('Verified', len(manifest['files']), 'shared source/data files and retained executable semantics.')


if __name__ == '__main__':
    main()
