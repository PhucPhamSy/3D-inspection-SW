# 🔬 INNO3D Inspection System — 3D Semiconductor & HBM Inspection Software

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/UI-PyQt5-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![CUDA](https://img.shields.io/badge/Acceleration-CUDA_12.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![NET8](https://img.shields.io/badge/CPU_Engine-.NET_8_NativeAOT-512BD4.svg)](https://dotnet.microsoft.com/)
[![SQLite](https://img.shields.io/badge/Database-SQLite3-003B57.svg)](https://www.sqlite.org/)

**INNO3D Inspection System** là giải pháp phần mềm kiểm tra chất lượng bán dẫn 3D cao cấp dành cho sản xuất bộ nhớ băng thông cao (**HBM - High Bandwidth Memory**), **Wafer**, **TGV (Through-Glass Via)**, và phát hiện lỗi **Bump Void / B2B**. Hệ thống kết hợp giao diện đồ họa trực quan (PyQt5), AI phân đoạn nâng cao (SAM 2, DINO) và đường ống tính toán ranh giới 3D kép (CUDA GPU Acceleration & C# NativeAOT Spatial Hashing).

---

## 🌟 Tính Năng Nổi Bật (Key Features)

### 1. 🖥️ Hiển Thị & Điều Hướng 3D (3D MPR & Viewer)
* **Multi-Planar Reconstruction (MPR)**: Điều hướng cắt lớp 3D/2D theo thời gian thực (Axial, Coronal, Sagittal).
* **Crosshair & Dynamic ROI**: Hỗ trợ tâm ngắm tương tác, quản lý vùng quan tâm (ROI), SOH Data.
* **Volume Rendering & Contrast Tuning**: Tùy chỉnh tương phản, bảng màu và rendering cấu trúc 3D siêu nét.

### 2. ⚡ Đường Ống Phân Tích Ranh Giới 3D Kép (Dual-Engine Boundary Analysis)
* **CUDA GPU Engine (`BoundaryGPU.dll`)**: Sử dụng CUDA C++ song song hóa tối đa trên GPU NVIDIA (RTX series) cho tốc độ tính toán khoảng cách 3D dưới vài miligiây.
* **C# NativeAOT CPU Engine (`BoundaryAnalysis.dll`)**: Thuật toán Spatial Hashing biên dịch mã máy gốc (NativeAOT), cung cấp phương án dự phòng (fallback) cực nhanh trên CPU đa nhân.

### 3. 🤖 Tích Hợp AI & Active Learning
* **Segment Anything 2 (SAM 2) & DINO SAM2**: Tự động gợi ý vùng lỗi (AutoPrompt) và phân đoạn chính xác các khuyết tật bán dẫn.
* **Active Learning & Pretrainer**: Học chủ động nâng cao độ chính xác nhận dạng lỗi theo từng lô sản phẩm.

### 4. 📊 Quản Lý Wafer & Batch Review (SQLite Backend)
* **Wafer Mapping & FOV Run Tracking**: Theo dõi vị trí chip trên Wafer (grid_cols × grid_rows) và đợt chạy FOV.
* **Tự Động Phân Loại Lỗi (Binning)**: Ghi nhận kết quả PASS/FAIL/BIN cho từng chip và khuyết tật.
* **Lưu Trực Tiếp Database (`inspection.db`)**: Quản lý lịch sử kiểm tra, dữ liệu đo đạc và vết kiểm toán (Audit Logs).

---

## 📁 Cấu Trúc Dự Án (Project Structure)

```text
3D-inspection-SW/
├── inno3d/                      # Mã nguồn Python chính (Modular Architecture)
│   ├── app/                     # Cấu hình ứng dụng & Settings
│   ├── core/                    # Xử lý logic 3D, DB, Spatial Analysis, ROI, Styles
│   ├── features/                # Tính năng Viewer, Crosshair, MPR Navigation
│   ├── infra/                   # Logging setup & Quản lý đường dẫn
│   ├── modes/                   # Chế độ hoạt động (Online / Offline)
│   ├── services/                # Các dịch vụ AI (SAM2, DINO, Active Learning)
│   ├── tabs/                    # Các màn hình chính (Viewer, Measurement, Analysis, Batch, AI, Help)
│   └── widgets/                 # Các UI Widget tái sử dụng
├── BoundaryGPU/                 # Mã nguồn CUDA C++ (BoundaryGPU.cu)
├── BoundaryDll/                 # Mã nguồn C# .NET 8 NativeAOT (BoundaryAnalysis)
├── config/                      # Các tệp cấu hình tham số kiểm tra (config_HBM.txt, ...)
├── assets/                      # Tài nguyên giao diện (Branding, Icons, Fonts)
├── docs/                        # Tài liệu kỹ thuật, sơ đồ CSDL, Hướng dẫn Build
│   ├── BUILD_INSTRUCTIONS.md    # Hướng dẫn build các file DLL hiệu năng cao
│   ├── database_schema_diagram.md # Sơ đồ ERD & Data Flow của SQLite
│   └── REFACTOR_PLAN_PROFESSIONAL.md
├── tests/                       # Unit tests & Integration tests
├── app_config.ini               # Tệp cấu hình khởi động
├── build_final.bat              # Script đóng gói PyInstaller tự động
├── main.py                      # Điểm khởi chạy ứng dụng (Entry Point & Splash Screen)
└── .gitignore                   # Cấu hình bỏ qua các tệp tạm / build
```

---

## 💻 Yêu Cầu Hệ Thống (System Requirements)

### Phần Cứng (Hardware)
* **CPU**: Intel Core i7/i9 hoặc AMD Ryzen 7/9 / Xeon đa nhân.
* **RAM**: Tối thiểu 16 GB (khuyến nghị 32 GB trở lên cho dữ liệu 3D thể tích lớn).
* **GPU**: NVIDIA RTX 3080 / 4080 / 5090 (Hỗ trợ CUDA 12.x+) cho tính toán Boundary GPU tốc độ cao.

### Phần Mềm (Software)
* **Hệ điều hành**: Windows 10 / 11 64-bit.
* **Python**: 3.10 trở lên.
* **C++ Compiler & CUDA Toolkit**: Visual Studio 2019/2022 + CUDA Toolkit 12.x (Nếu build lại `BoundaryGPU.dll`).
* **.NET 8 SDK**: (Nếu build lại `BoundaryAnalysis.dll`).

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Ứng Dụng

### 1. Thao Tác Chuẩn Bị
Kích hoạt môi trường ảo Python và cài đặt các thư viện cần thiết:

```bash
# Tạo và kích hoạt virtual environment (tùy chọn)
python -m venv venv
venv\Scripts\activate

# Cài đặt các phụ thuộc cơ bản
pip install -r requirements.txt
```

*(Lưu ý: Đảm bảo các thư viện `PyQt5`, `opencv-python`, `numpy`, `scipy`, `vtk`, `torch` đã được cài đặt).*

### 2. Chạy Ứng Dụng
Chạy lệnh trực tiếp từ thư mục gốc dự án:

```bash
python main.py
```

---

## 🛠️ Hướng Dẫn Biên Dịch Thư Viện Hiệu Năng Cao (DLL Build)

Hệ thống hỗ trợ 2 module DLL hiệu năng cao cho thuật toán Boundary Analysis. Để tự biên dịch lại các DLL này, tham khảo tài liệu chi tiết tại [`docs/BUILD_INSTRUCTIONS.md`](docs/BUILD_INSTRUCTIONS.md).

Tóm tắt lệnh build:

* **Biên dịch CUDA GPU DLL (`BoundaryGPU.dll`)**:
  ```cmd
  cd BoundaryGPU
  nvcc -shared -O3 -o BoundaryGPU.dll BoundaryGPU.cu
  ```

* **Biên dịch C# NativeAOT CPU DLL (`BoundaryAnalysis.dll`)**:
  ```cmd
  cd BoundaryDll\BoundaryAnalysis
  dotnet publish -c Release -r win-x64
  ```

---

## 🗄️ Cấu Trúc Cơ Sở Dữ Liệu (Database Schema)

Hệ thống lưu trữ toàn bộ dữ liệu kiểm tra vào cơ sở dữ liệu SQLite `inspection.db`. Sơ đồ CSDL gồm 7 bảng chính:
- `META`: Lưu thông tin schema version và cài đặt hệ thống.
- `WAFERS`: Quản lý thông tin Wafer, Lot ID, FOUP ID, kích thước lưới chip.
- `CHIPS`: Quản lý vị trí con chip (Row/Col) và Binning final.
- `FOV_RUNS`: Quản lý đợt quét FOV, thời gian xử lý và đánh giá tổng thể.
- `OBJECTS`: Lưu thông tin thể tích, diện tích, tọa độ 3D các khuyết tật.
- `DEFECTS`: Chi tiết lỗi và phân loại.
- `AUDIT_LOGS`: Lưu nhật ký thao tác của người dùng.

Chi tiết sơ đồ ERD xem tại [`docs/database_schema_diagram.md`](docs/database_schema_diagram.md).

---

## 📜 Giấy Phép & Tác Giả (License & Author)

* **Tác giả / Lead Developer**: Phuc Pham Sy ([GitHub @PhucPhamSy](https://github.com/PhucPhamSy))
* **Dự án**: HBM & Semiconductor 3D Inspection SW Solution.
