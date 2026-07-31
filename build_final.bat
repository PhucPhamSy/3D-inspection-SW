@echo off
setlocal
cls
echo ==========================================
echo   Inno3D Inspection - Build Tool v2
echo ==========================================
echo.

set "APP_NAME=Inno3D"
REM Native DLL source (override: set DLL_SOURCE_DIR=... before running)
if not defined DLL_SOURCE_DIR set "DLL_SOURCE_DIR=E:\semiconductor\HBM_DEV_FOR_PROD\DEV\all_v2\V6"
set "ICON_PATH=%cd%\assets\branding\company_logo_v1.ico"
set "ICON_SCRIPT=%cd%\build\tools\create_icon.py"
set "BUILD_WORKDIR=%cd%\build\pyinstaller"
set "SPEC_DIR=%cd%\build\spec"
set "DIST_DIR=%cd%\dist"

REM Read version from VERSION file
set "APP_VERSION=unknown"
if exist "%cd%\VERSION" (
    set /p APP_VERSION=<"%cd%\VERSION"
)
echo Version: %APP_VERSION%

REM PACKAGE_SLIM: set to 1 to copy only core DLLs (SEG/MES/B2B/ENH + OpenCV/CUDA).
REM Skips large TRT / onnxruntime bulk for UI-only outsource packages.
REM Usage: set PACKAGE_SLIM=1 && build_final.bat
if not defined PACKAGE_SLIM set PACKAGE_SLIM=0
if "%PACKAGE_SLIM%"=="1" (
    echo [SLIM MODE] Only core DLLs will be copied ^(no TRT bulk^)
) else (
    echo [FULL MODE] Full native DLL package will be copied
)
echo DLL source: %DLL_SOURCE_DIR%
echo.

REM Step 1: Create icon (optional, skips if script or source missing)
if exist "%ICON_SCRIPT%" (
    echo [1/4] Creating application icon...
    python "%ICON_SCRIPT%"
) else (
    echo [1/4] build\tools\create_icon.py not found, skipping icon regeneration...
)

if not exist "%ICON_PATH%" (
    echo WARNING: %ICON_PATH% not found! Build will likely fail or have default icon.
)

REM Step 2: Clean old builds
echo [2/4] Cleaning old builds...
if exist "%BUILD_WORKDIR%" rmdir /s /q "%BUILD_WORKDIR%"
if exist "%SPEC_DIR%" rmdir /s /q "%SPEC_DIR%"
if exist "%DIST_DIR%" rmdir /s /q "%DIST_DIR%"

REM Step 3: Build exe with explicit icon path
echo [3/4] Building executable...
echo.

pyinstaller --noconfirm ^
    --onedir ^
    --windowed ^
    --name="%APP_NAME%" ^
    --icon="%ICON_PATH%" ^
    --workpath="%BUILD_WORKDIR%" ^
    --specpath="%SPEC_DIR%" ^
    --distpath="%DIST_DIR%" ^
    --add-data="%cd%\assets;assets" ^
    --add-data="%cd%\config;config" ^
    --collect-submodules="inno3d" ^
    --exclude-module="PySide6" ^
    --exclude-module="PySide2" ^
    --exclude-module="PyQt6" ^
    --hidden-import="imagecodecs" ^
    --collect-all="imagecodecs" ^
    --collect-all="tifffile" ^
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

REM Step 4: Verify build + stage portable runtime files next to exe
echo.
echo [4/4] Verifying build...
if exist "dist\%APP_NAME%\%APP_NAME%.exe" (
    echo ==========================================
    echo   BUILD SUCCESSFUL!  v%APP_VERSION%
    echo ======================================
    echo.
    echo Output: dist\%APP_NAME%\%APP_NAME%.exe
    for %%I in ("dist\%APP_NAME%\%APP_NAME%.exe") do echo Size: %%~zI bytes
    echo.
    REM Write version stamp next to exe
    echo %APP_VERSION%> "dist\%APP_NAME%\VERSION.txt"
    echo Build: %DATE% %TIME%>> "dist\%APP_NAME%\VERSION.txt"
    echo Stamped VERSION.txt: %APP_VERSION%
    echo.
    REM Station config + inspection DB live next to the exe (editable / portable)
    if exist "%cd%\app_config.ini" (
        copy /Y "%cd%\app_config.ini" "dist\%APP_NAME%\app_config.ini" >nul
        echo Copied app_config.ini next to exe
    )
    if not exist "dist\%APP_NAME%\Inno3D_Data" mkdir "dist\%APP_NAME%\Inno3D_Data"
    echo Inspection DB will be created at: dist\%APP_NAME%\Inno3D_Data\inspection.db
    echo.

    REM Native SEG/MES/B2B/ENH + CUDA/OpenCV deps — MUST sit next to the exe.
    REM PyInstaller does NOT embed these; copy from shared all_v2\V6 into dist\V2.
    if exist "%DLL_SOURCE_DIR%" (
        if not exist "dist\%APP_NAME%\V2" mkdir "dist\%APP_NAME%\V2"
        if "%PACKAGE_SLIM%"=="1" (
            echo Copying SLIM native DLLs from V6 ^(core only — no TRT bulk^)...
            REM Core native binaries
            for %%F in (
                BumpVoidSeg.dll BumpVoidDLL.dll
                BumpVoidMes.dll
                BoundaryGPU.dll BumpVoidB2B.dll
                BumpVoid_ISP_ENH.dll EnhancedVolumeDLL.dll
                opencv_world4110.dll
            ) do (
                if exist "%DLL_SOURCE_DIR%\%%F" (
                    copy /Y "%DLL_SOURCE_DIR%\%%F" "dist\%APP_NAME%\V2\%%F" >nul
                    echo   Copied %%F
                )
            )
            REM CUDA runtime essentials (cublas, cudart, curand — skip cuDNN/TRT bulk)
            for %%F in (cublas64*.dll cublasLt64*.dll cudart64*.dll curand64*.dll) do (
                for %%G in ("%DLL_SOURCE_DIR%\%%F") do (
                    if exist "%%G" copy /Y "%%G" "dist\%APP_NAME%\V2\" >nul
                )
            )
            echo Slim native DLLs copied  ^(SEG+MES+B2B+ENH+OpenCV+CUDA-core^)
        ) else (
            echo Copying full native DLL package from V6 next to exe ^(this can take several minutes^)...
            robocopy "%DLL_SOURCE_DIR%" "dist\%APP_NAME%\V2" /E /XO /R:1 /W:1 /NFL /NDL /NP
            if errorlevel 8 (
                echo WARNING: robocopy reported errors while copying native DLLs
            ) else (
                echo Copied V6 -^> dist\V2  ^(required for BumpVoidSeg / OpenCV / CUDA deps^)
            )
        )
    ) else (
        echo WARNING: DLL source folder not found: %DLL_SOURCE_DIR%
        echo          Packaged app will fail to load native DLLs.
        echo          Place a complete native package at: dist\%APP_NAME%\V2
    )

    REM Optional: copy default recipe configs next to exe for portable browse defaults
    if exist "%cd%\config" (
        if not exist "dist\%APP_NAME%\config" mkdir "dist\%APP_NAME%\config"
        robocopy "%cd%\config" "dist\%APP_NAME%\config" *.txt /XO /R:1 /W:1 /NFL /NDL /NP >nul
        echo Copied config\*.txt next to exe
    )

    echo.
    echo Package layout for outsourcing team:
    echo   dist\%APP_NAME%\%APP_NAME%.exe
    echo   dist\%APP_NAME%\V2\               ^<-- native DLLs ^(BumpVoidSeg, OpenCV, CUDA^)
    echo   dist\%APP_NAME%\config\           ^<-- recipe configs
    echo   dist\%APP_NAME%\app_config.ini
    echo   dist\%APP_NAME%\Inno3D_Data\
    echo.
    echo Zip the entire dist\%APP_NAME%\ folder ^(do not send only the .exe^).
    echo Icon should now display correctly!
    echo ==========================================
    echo.
    REM Optional: post-build smoke check (no GPU required)
    if exist "%cd%\scripts\post_build_smoke.py" (
        echo Running post-build smoke check...
        python "%cd%\scripts\post_build_smoke.py" --dist-dir "dist\%APP_NAME%"
    )
) else (
    echo ==========================================
    echo   BUILD FAILED!
    echo   Expected: dist\%APP_NAME%\%APP_NAME%.exe
    echo ==========================================
)

pause
