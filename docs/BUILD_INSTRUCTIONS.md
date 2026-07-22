# HƯỚNG DẪN BUILD DLL CHO 3D BOUNDARY ANALYSIS PIPELINE

Tài liệu này hướng dẫn cách tự biên dịch lại (recompile) hai phiên bản DLL hiệu năng cao (CUDA GPU và C# Native CPU) cho module phân tích ranh giới 3D (Boundary Analysis).

---

## 1. Yêu Cầu Hệ Thống & Công Cụ (Prerequisites)

Để build được cả 2 DLL, máy tính của bạn cần cài đặt các công cụ sau:

1. **Visual Studio 2019 hoặc 2022** (Đã tích hợp công cụ lập trình C++ Desktop Development).
2. **CUDA Toolkit** (Phiên bản phù hợp với GPU RTX 5090 của bạn, khuyến nghị CUDA 12.x trở lên).
3. **.NET 8 SDK** (Dành cho việc build DLL C# NativeAOT).

---

## 2. Hướng Dẫn Build CUDA GPU DLL (`BoundaryGPU.dll`)

Bản GPU này được viết bằng CUDA C++ để thực hiện các phép toán khoảng cách song song cực nhanh trên card đồ họa.

### Các bước thực hiện:

1. Mở menu **Start** trên Windows, tìm và khởi chạy công cụ:
   `x64 Native Tools Command Prompt for VS 2019` (hoặc bản `VS 2022` tương ứng).
   *Lưu ý: Bắt buộc phải dùng Command Prompt này của Visual Studio để trình biên dịch CUDA nhận dạng được môi trường MSVC C++.*
2. Di chuyển (cd) vào thư mục chứa code CUDA trong project của bạn:

   ```cmd
   E:
   cd E:\semiconductor\HBM_DEV_FOR_PROD\DEV\Inno3D_dev\HBM_frontend_backend_v12\BoundaryGPU
   ```
3. Chạy lệnh biên dịch sau để tạo file DLL:

   ```cmd
   nvcc -shared -O3 -o BoundaryGPU.dll BoundaryGPU.cu
   ```
4. **Triển khai (Deploy):**

   * Copy file `BoundaryGPU.dll` vừa được tạo ra trong thư mục `BoundaryGPU` dán đè vào thư mục thư viện chung của bạn:
     `E:\phuc\HBM\LIB_RELEASE\V1\`

---

## 3. Hướng Dẫn Build C# Native CPU DLL (`BoundaryAnalysis.dll`)

Bản C# này sử dụng công nghệ NativeAOT (AOT biên dịch thẳng ra mã máy không cần runtime .NET) và cấu trúc dữ liệu Spatial Hashing nâng cao để làm phương án dự phòng (fallback) siêu tốc trên CPU Xeon đa nhân của bạn.

### Các bước thực hiện:

1. Mở **Command Prompt** (cmd) thông thường hoặc PowerShell.
2. Di chuyển vào thư mục project C#:

   ```cmd
   E:
   cd E:\semiconductor\HBM_DEV_FOR_PROD\DEV\Inno3D_dev\HBM_frontend_backend_v12\BoundaryDll\BoundaryAnalysis
   ```
3. Chạy lệnh publish NativeAOT ở chế độ Release:

   ```cmd
   dotnet publish -c Release -r win-x64
   ```
4. **Triển khai (Deploy):**

   * Sau khi build xong, truy cập vào thư mục:
     `E:\semiconductor\HBM_DEV_FOR_PROD\DEV\Inno3D_dev\HBM_frontend_backend_v12\BoundaryDll\BoundaryAnalysis\bin\Release\net8.0\win-x64\publish\`
   * Copy file `BoundaryAnalysis.dll` trong đó dán đè vào thư mục thư viện chung:
     `E:\phuc\HBM\LIB_RELEASE\V1\`

---

## 4. Kiểm Tra Sau Khi Cập Nhật

Khi chạy tính toán, phần mềm sẽ hiển thị trên thanh tiến trình (Progress Bar) thông tin về bộ thư viện đang chạy:

* **CUDA GPU DLL Boundary Analysis...**: Chạy bằng card đồ họa.
* **C# Native DLL Boundary Analysis...**: Chạy bằng CPU tối ưu hóa.

Nếu gặp bất kỳ lỗi nào về định dạng cột CSV, hãy kiểm tra lại xem các file DLL trong thư mục `V1` đã được copy đè phiên bản mới nhất chưa.
