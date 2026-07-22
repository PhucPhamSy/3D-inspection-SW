import re
lines = open('inno3d/tabs/viewer.py','r',encoding='utf-8').readlines()
targets = ['render_slice','reset_view','add_ruler_overlay','create_crosshair_actors',
           'on_crosshair_slider_changed','toggle_reverse_z','update_pixel_value',
           '_restore_interactor_style','_plane_cell_stylesheet']
METHOD_RE = re.compile(r'^    def (\w+)\(')
in_class = False
i = 0
while i < len(lines):
    line = lines[i]
    if line.startswith('class MultiPlanarView'):
        in_class = True
    if not in_class:
        i += 1; continue
    m = METHOD_RE.match(line)
    if m and m.group(1) in targets:
        name = m.group(1)
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if (nxt.startswith('    def ') or nxt.startswith('    @') or nxt.startswith('class ')) and j > i+1:
                break
            j += 1
        print(f'  {name:35s}  lines {i+1}-{j}  ({j-i} lines)')
        i = j
    else:
        i += 1
print(f'\ntotal viewer.py lines: {len(lines)}')
