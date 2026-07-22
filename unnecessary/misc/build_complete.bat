@echo off
echo ========================================
echo Building Inno3D with Complete VTK...
echo ========================================

REM Clean
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Build với đầy đủ VTK modules
pyinstaller --noconfirm --onefile --windowed ^
    --name "Inno3D_Inspection" ^
    --add-data "styles.py;." ^
    --hidden-import="vtk" ^
    --hidden-import="vtkmodules" ^
    --hidden-import="vtkmodules.all" ^
    --hidden-import="vtkmodules.qt.QVTKRenderWindowInteractor" ^
    --hidden-import="vtkmodules.vtkCommonCore" ^
    --hidden-import="vtkmodules.vtkCommonDataModel" ^
    --hidden-import="vtkmodules.vtkCommonExecutionModel" ^
    --hidden-import="vtkmodules.vtkFiltersCore" ^
    --hidden-import="vtkmodules.vtkFiltersGeneral" ^
    --hidden-import="vtkmodules.vtkFiltersSources" ^
    --hidden-import="vtkmodules.vtkImagingCore" ^
    --hidden-import="vtkmodules.vtkInteractionStyle" ^
    --hidden-import="vtkmodules.vtkRenderingCore" ^
    --hidden-import="vtkmodules.vtkRenderingOpenGL2" ^
    --hidden-import="vtkmodules.vtkRenderingVolume" ^
    --hidden-import="vtkmodules.vtkRenderingVolumeOpenGL2" ^
    --hidden-import="vtkmodules.util" ^
    --hidden-import="vtkmodules.util.numpy_support" ^
    --hidden-import="scipy.ndimage" ^
    --hidden-import="skimage.io" ^
    --hidden-import="skimage.measure" ^
    --collect-all="vtk" ^
    --collect-all="vtkmodules" ^
    main.py

echo.
echo Build complete! Check dist\Inno3D_Inspection.exe
pause