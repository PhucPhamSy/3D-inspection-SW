import ast

files = [
    'inno3d/core/bumpvoid.py',
    'inno3d/core/bumpvoid_mes.py',
    'inno3d/core/bumpvoid_b2b.py',
    'inno3d/core/enhanced_volume.py',
]

for f in files:
    print(f'\n=== {f} ===')
    tree = ast.parse(open(f, encoding='utf-8').read())
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not n.name.startswith('_'):
                kind = 'class' if isinstance(n, ast.ClassDef) else 'def'
                print(f'  {kind} {n.name}')
