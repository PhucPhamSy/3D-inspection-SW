"""Scan mixin files for bare name references that need imports."""
import re
text = open('inno3d/features/viewer/crosshair.py', encoding='utf-8').read()
refs = set()
for pat, name in [
    (r'\bvtk\b', 'vtk'), (r'\bQt\b', 'Qt'), (r'\bQColor\b', 'QColor'),
    (r'\bQColorDialog\b', 'QColorDialog'), (r'\bQPoint\b', 'QPoint'),
    (r'\bSemiconductorTheme\b', 'SemiconductorTheme'),
]:
    if re.search(pat, text): refs.add(name)
print('crosshair.py uses:', sorted(refs))

text2 = open('inno3d/features/viewer/mpr_nav.py', encoding='utf-8').read()
refs2 = set()
for pat, name in [
    (r'\bvtk\b', 'vtk'), (r'\bQt\b', 'Qt'), (r'\bQPoint\b', 'QPoint'),
    (r'\bQColor\b', 'QColor'),
]:
    if re.search(pat, text2): refs2.add(name)
print('mpr_nav.py uses:', sorted(refs2))
