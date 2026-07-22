# build.py
import os
import sys
import shutil
from pathlib import Path

def clean_build():

    dirs_to_remove = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"🗑️ Removed {dir_name}")
    
    # Xóa file .pyc
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pyc'):
                os.remove(os.path.join(root, file))

def build_exe():

    print("\n🔨 Building EXE...")
    print("=" * 50)
    

    cmd_folder = 'pyinstaller --clean --noconfirm Inno3D_Inspection.spec'
    
    result = os.system(cmd_folder)
    
    if result == 0:
        print("\n✅ Build successful!")
        print(f"📁 Output: dist/Inno3D_Inspection/")
        print(f"🚀 Run: dist/Inno3D_Inspection/Inno3D_Inspection.exe")
        
   
        create_batch_file()
    else:
        print("\n❌ Build failed!")
        sys.exit(1)

def create_batch_file():
    """Tạo file .bat để chạy nhanh"""
    batch_content = """@echo off
title Inno3D Inspection
cd /d "%~dp0"
start "" "dist\\Inno3D_Inspection\\Inno3D_Inspection.exe"
"""

    with open('Run_Inno3D.bat', 'w', encoding='utf-8') as f:
        f.write(batch_content)
    print("📝 Created Run_Inno3D.bat")

def main():
    print("🚀 Inno3D Inspection - Build Tool")
    print("=" * 50)
    
    # Clean old builds
    clean_build()
    
    # Create spec file
    import build_spec
    build_spec.create_spec_file()
    
    # Build exe
    build_exe()
    
    print("\n✨ Done!")

if __name__ == '__main__':
    main()