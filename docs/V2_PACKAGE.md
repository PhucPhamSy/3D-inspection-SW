# V2 Package — Ship Layout & Native DLL Guide

> **Scope:** This document describes the `V2/` folder that must be shipped
> alongside the frozen Inno3D HBM executable (or alongside `main.py` in
> developer mode). It lists every native DLL, ONNX model, and CUDA runtime
> file required, and explains how `inno3d/infra/paths.py` locates them.

---

## 1. Directory Layout

```
<app_root>/                       ← exe folder (frozen) OR project root (dev)
├── Inno3D_HBM.exe                ← frozen binary  (or python main.py)
├── config/                       ← recipe .txt files (shipped read-only)
│   ├── config_HBM.txt
│   └── config_HBM_c2848_M.txt
├── assets/                       ← icons, fonts, backgrounds
│   ├── icons/
│   ├── fonts/
│   └── backgrounds/
├── V2/                           ← ALL native DLLs + ONNX model (required)
│   ├── BumpVoidSeg.dll           ← main segmentation DLL
│   ├── BumpVoidDLL.dll           ← legacy alias (fallback)
│   ├── BumpVoidMes.dll           ← MES integration
│   ├── BumpVoidB2B.dll           ← B2B inspection
│   ├── BumpVoid_ISP_ENH.dll      ← ISP enhancement step
│   ├── BoundaryGPU.dll           ← GPU boundary detection
│   ├── EnhancedVolumeDLL.dll     ← volume enhancement host
│   ├── restormer_nano_e9900_dynamic_ir9_fp16.onnx   ← ONNX AI model (~11 MB)
│   ├── onnxruntime.dll           ← ONNX Runtime CPU
│   ├── onnxruntime_providers_cuda.dll               ← ONNX RT CUDA provider
│   ├── onnxruntime_providers_tensorrt.dll            ← ONNX RT TRT provider
│   ├── onnxruntime_providers_shared.dll
│   ├── Microsoft.ML.OnnxRuntime.dll
│   ├── opencv_world4110.dll      ← OpenCV native
│   ├── BitMiracle.LibTiff.NET.dll
│   ├── nvinfer_10.dll            ┐
│   ├── nvinfer_lean_10.dll       │ TensorRT 10.x
│   ├── nvonnxparser_10.dll       │ (GPU path only)
│   ├── nvinfer_plugin_10.dll     │
│   └── nvinfer_*.dll             ┘
│   ├── cudart64_12.dll           ┐
│   ├── cublas64_12.dll           │ CUDA 12.x runtime
│   ├── cublasLt64_12.dll         │ (GPU path only)
│   ├── cufft64_10.dll            │
│   └── cudnn*.dll                ┘ cuDNN 9.x
└── Inno3D_Data/                  ← created at runtime (SQLite DB, logs)
```

---

## 2. How the App Finds V2/

`inno3d/infra/paths.py` → `default_dll_dir()` searches in this order:

| # | Candidate | Used when |
|---|-----------|-----------|
| 1 | `<exe_folder>/V2` | Frozen (PyInstaller) build |
| 2 | `<project_root>/V2` | Developer — `python main.py` |
| 3 | `<project_root>/../V2` | Dev monorepo (`all_v2/` layout) |

The first candidate that `is_dir()` wins. No absolute path is hard-coded in
the shipped build.

---

## 3. ENHANCED_MODEL_PATH

Set in `config/*.txt`:

```ini
ENHANCED_MODEL_PATH = "V2/restormer_nano_e9900_dynamic_ir9_fp16.onnx"
```

`teaching.py` resolves this relative path via `app_runtime_dir()` at load
time, converting it to an absolute path before `os.path.exists()` is called.

---

## 4. DLL Search Priority (SEG)

`inno3d/infra/paths.py` → `default_dll_dir()` returns the `V2/` folder.
`teaching.py` loads DLLs in this name priority:

1. `BumpVoidSeg.dll` ← preferred (current)
2. `BumpVoidDLL.dll` ← legacy fallback

---

## 5. GPU Requirements

| Component | Version |
|-----------|---------|
| CUDA Toolkit | 12.x |
| cuDNN | 9.x |
| TensorRT | 10.x |
| ONNX Runtime GPU | 1.19.2 |

CPU-only mode works without GPU DLLs — enhancement step is skipped when
`ENHANCEMENT_ENABLED = false` or no GPU is detected.

---

## 6. Build Script

`build_final.bat` in the project root handles the frozen build:

1. Runs PyInstaller with the `.spec` file
2. Copies `V2/` next to the output `.exe`
3. Copies `config/` next to the output `.exe`
4. Copies `assets/` into the bundle

---

## 7. Minimum Ship Checklist

- [ ] `V2/BumpVoidSeg.dll` (or `BumpVoidDLL.dll`)
- [ ] `V2/restormer_nano_e9900_dynamic_ir9_fp16.onnx` (if enhancement needed)
- [ ] `V2/onnxruntime.dll` + providers
- [ ] `V2/opencv_world4110.dll`
- [ ] `config/config_HBM.txt` (default recipe)
- [ ] `assets/` (icons, fonts)
- [ ] CUDA + cuDNN DLLs (GPU path only)
