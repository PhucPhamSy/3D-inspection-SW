with open('inno3d/features/teaching/seg_ui.py', encoding='utf-8') as f:
    lines = f.readlines()
terms = ['labeled_class1', 'class1_data', 'class2_data', 'seg_colors', 'bump_segmentation']
for i, line in enumerate(lines, 1):
    if any(t in line for t in terms):
        if i < 500 or 'self.' in line:
            print(f'{i}: {line.rstrip()}')
