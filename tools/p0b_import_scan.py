"""
P0b: Mandatory import free-name scan + AST compile check.
Run after every extract PR to prevent NameError regressions.
"""
import ast
import pathlib
import sys
import subprocess

sys.stdout.reconfigure(encoding='utf-8')

TARGET_DIRS = [
    'inno3d/features',
    'inno3d/native',
    'inno3d/infra',
    'inno3d/core',
]

print("=" * 65)
print("P0b: Import scan + AST compile for inno3d/")
print("=" * 65)

ok = 0
errors = []

for d in TARGET_DIRS:
    for p in sorted(pathlib.Path(d).rglob('*.py')):
        if '__pycache__' in str(p):
            continue
        try:
            src = p.read_text(encoding='utf-8')
            ast.parse(src)
            ok += 1
        except SyntaxError as e:
            errors.append(f'SYNTAX {p}: {e}')
        except UnicodeDecodeError as e:
            errors.append(f'ENCODING {p}: {e}')

print(f"AST compile: {ok} OK, {len(errors)} errors")
for e in errors:
    print(f"  {e}")

# Count ruff fixable issues
result = subprocess.run(
    ['python', '-m', 'ruff', 'check', 'inno3d/features', 'inno3d/native', 'inno3d/infra',
     '--select', 'F', '--output-format', 'concise'],
    capture_output=True, text=True, encoding='utf-8'
)
lines = result.stdout.strip().splitlines()
f_errors = [l for l in lines if l.strip() and not l.startswith('Found')]
print(f"\nruff F-rules (pyflakes):")
if f_errors:
    for l in f_errors[:20]:
        print(f"  {l}")
    if len(f_errors) > 20:
        print(f"  ... +{len(f_errors)-20} more")
else:
    print("  ✅ No F-rule violations")

print()
print("=" * 65)
if not errors and not f_errors:
    print("✅ P0b PASS — No import or syntax issues found")
else:
    print(f"⚠️  P0b WARNINGS: {len(errors)} AST errors, {len(f_errors)} ruff F issues")
