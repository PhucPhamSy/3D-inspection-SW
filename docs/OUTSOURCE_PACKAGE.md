# Outsource Package — Running SEG from the Shipped ZIP

> **Audience:** External / outsource engineers who need to run the Inno3D
> HBM segmentation pipeline without the full development environment.

---

## 1. What You Receive

```
Inno3D_HBM_<version>.zip
├── Inno3D_HBM.exe          ← frozen launcher (no Python needed)
├── config/
│   └── config_HBM.txt      ← default recipe — edit INPUT_PATH & OUTPUT_DIR
├── assets/                  ← icons/fonts (read-only)
└── V2/                      ← native DLLs + ONNX model
    ├── BumpVoidSeg.dll
    ├── restormer_nano_e9900_dynamic_ir9_fp16.onnx
    └── ... (see V2_PACKAGE.md for full list)
```

---

## 2. Quick Start

### Step 1 — Set the recipe

Open `config/config_HBM.txt` in any text editor and set:

```ini
INPUT_PATH  = C:/your/data/folder        # folder with TIFF slices
OUTPUT_DIR  = C:/your/output/folder
```

All other parameters (thresholds, model path, etc.) have sensible defaults.

### Step 2 — Launch

Double-click `Inno3D_HBM.exe`.

Or from the command line:

```bat
cd Inno3D_HBM_<version>
Inno3D_HBM.exe
```

### Step 3 — Load data & run

1. **File → Open Config** → select your `config_HBM.txt`
2. **File → Open Volume** → select your TIFF slice folder
3. Click **Run Segmentation** in the Teaching tab

Results are written to `OUTPUT_DIR/`.

---

## 3. GPU Enhancement (optional)

If the `V2/restormer_*.onnx` model is present and your machine has an
NVIDIA GPU with CUDA 12.x, set in the recipe:

```ini
ENHANCEMENT_ENABLED = true
ENHANCED_MODEL_PATH  = "V2/restormer_nano_e9900_dynamic_ir9_fp16.onnx"
ENHANCED_USE_GPU     = true
```

The enhancement step runs before segmentation and improves quality on
low-SNR volumes.  Set `ENHANCEMENT_ENABLED = false` to skip it.

---

## 4. Directory Created at Runtime

The app auto-creates `Inno3D_Data/` and `Inno3D_Logs/` next to the exe
on first run.  These can be deleted safely between sessions.

---

## 5. Input Data Format

| Item | Requirement |
|------|-------------|
| File type | Single-layer TIFF (`.tif` / `.tiff`) |
| Bit depth | 8-bit or 16-bit grayscale |
| Slice naming | Alphabetical / numeric sort = Z order |
| Folder | All slices in one flat folder |

---

## 6. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `DLL not found` on launch | Ensure `V2/` is next to the `.exe` |
| Enhancement skipped | Check `ENHANCEMENT_ENABLED = true` and GPU drivers |
| Empty output | Verify `INPUT_PATH` exists and contains `.tif` files |
| App crashes on load | Check `Inno3D_Logs/` for traceback; send log to team |

---

## 7. Contact

For issues, send the log file from `Inno3D_Logs/` to the Inno3D team.
