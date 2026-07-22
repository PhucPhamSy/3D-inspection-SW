# build_spec.py
import os
import sys
from pathlib import Path

def create_spec_file():
    """Tạo file .spec cho PyInstaller"""
    
    spec_content = """
# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path

block_cipher = None

# Path to source code directory
ROOT_DIR = r'{root_dir}'

a = Analysis(
    [os.path.join(ROOT_DIR, 'main.py')],
    pathex=[ROOT_DIR],
    binaries=[],
    datas=[
        # Include logo and images if exists
        # (r'E:\\semiconductor\\DEMO\\images\\company_logo_v1.webp', 'images'),
        # Include styles.py file
        (os.path.join(ROOT_DIR, 'styles.py'), '.'),
    ],
    hiddenimports=[
        'vtk',
        'vtkmodules',
        'vtkmodules.all',
        'vtkmodules.qt.QVTKRenderWindowInteractor',
        'vtkmodules.util',
        'vtkmodules.util.numpy_support',
        'vtkmodules.numpy_interface',
        'PyQt5.QtCore',
        'PyQt5.QtGui', 
        'PyQt5.QtWidgets',
        'PyQt5.sip',
        'numpy',
        'scipy',
        'scipy.ndimage',
        'skimage',
        'skimage.io',
        'skimage.measure',
        'glob',
        'pathlib',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'pandas',
        'notebook',
        'jupyter',
        'IPython',
        'tkinter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Collect all VTK and PyQt5 binaries
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

# VTK
vtk_datas, vtk_binaries, vtk_hiddenimports = collect_all('vtk')
a.datas += vtk_datas
a.binaries += vtk_binaries
a.hiddenimports += vtk_hiddenimports

# VTK modules  
vtkmodules_datas, vtkmodules_binaries, vtkmodules_hiddenimports = collect_all('vtkmodules')
a.datas += vtkmodules_datas
a.binaries += vtkmodules_binaries
a.hiddenimports += vtkmodules_hiddenimports

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Inno3D_Inspection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Don't use UPX with VTK
    console=False,  # False = no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon if available: icon='company_logo_v1.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Inno3D_Inspection'
)
""".format(root_dir=os.getcwd())
    
    # Sửa lỗi encoding - thêm encoding='utf-8'
    with open('Inno3D_Inspection.spec', 'w', encoding='utf-8') as f:
        f.write(spec_content)
    
    print("✅ Created Inno3D_Inspection.spec")

if __name__ == '__main__':
    create_spec_file()