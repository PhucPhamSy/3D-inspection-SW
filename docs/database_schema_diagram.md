# Sơ đồ Cấu trúc Database `inspection.db`

Tài liệu này tổng hợp toàn bộ sơ đồ cấu trúc CSDL SQLite (`Inno3D_Data/inspection.db`) của dự án, bao gồm **Sơ đồ Thực thể Mối quan hệ (ERD)** và **Sơ đồ Luồng Dữ liệu (Data Flow)**.

---

## 1. Sơ đồ Thực thể Mối quan hệ (Entity-Relationship Diagram)

Sơ đồ dưới đây thể hiện tất cả 7 bảng trong Database cùng với các thuộc tính khóa chính (PK), khóa ngoại (FK) và mối quan hệ giữa chúng.

```mermaid
erDiagram
    META {
        text key PK "Khóa chính (vd: schema_version)"
        text value "Giá trị phiên bản / cài đặt"
    }

    WAFERS {
        text wafer_key PK "Mã định danh Wafer (date_folder|lot_foup_id|wafer_id)"
        text date_folder "Thư mục ngày kiểm tra"
        text lot_foup_id "Mã Lot / FOUP"
        text lot_id "Mã Lot riêng"
        text foup_id "Mã FOUP riêng"
        text wafer_id "Mã Wafer"
        int grid_cols "Số cột trên lưới Wafer"
        int grid_rows "Số hàng trên lưới Wafer"
        text first_seen "Thời điểm thấy lần đầu"
        text last_seen "Thời điểm cập nhật mới nhất"
        text notes "Ghi chú thêm"
    }

    CHIPS {
        text chip_key PK "Mã chip ({wafer_key}|{col}|{row})"
        text wafer_key FK "Khóa ngoại tham chiếu WAFERS"
        int chip_col "Tọa độ cột con chip"
        int chip_row "Tọa độ hàng con chip"
        text chip_folder "Thư mục lưu dữ liệu chip"
        int final_bin "Kết quả đánh giá chip (1: NG, 8: OK, 2: Pending)"
    }

    FOV_RUNS {
        text run_id PK "Mã đợt chạy FOV (UUID/Timestamp)"
        text wafer_key FK "Khóa ngoại tham chiếu WAFERS"
        text chip_key "Mã chip tương ứng"
        text date_folder "Thư mục ngày"
        text lot_foup_id "Mã Lot FOUP"
        text wafer_id "Mã Wafer"
        int chip_col "Cột của chip"
        int chip_row "Hàng của chip"
        int fov_index "Chỉ số FOV"
        text fov_folder "Thư mục FOV"
        text input_path "Đường dẫn ảnh/dữ liệu đầu vào"
        text results_dir "Thư mục kết quả"
        text judgment "Đánh giá chung (PASS/FAIL/OK/NG)"
        int final_bin "Mã BIN kết quả"
        int n_objects "Tổng số object phát hiện"
        int n_ng "Số object lỗi (NG)"
        int n_ok "Số object đạt (OK)"
        real enhance_sec "Thời gian xử lý Enhance (giây)"
        real seg_sec "Thời gian xử lý Segmentation (giây)"
        real mes_sec "Thời gian xử lý Measurement (giây)"
        real b2b_sec "Thời gian B2B (giây)"
        real total_sec "Tổng thời gian thực thi (giây)"
        text enhance_provider "Thuật toán/Provider nâng chuẩn ảnh"
        text config_path "Đường dẫn file config"
        text dll_dir "Thư mục chứa DLL C++"
        text recipe_json "Recipe chi tiết cấu hình dạng JSON"
        text host_path "Đường dẫn máy host"
        text started_at "Thời điểm bắt đầu chạy"
        text finished_at "Thời điểm hoàn thành"
        text status "Trạng thái chạy (ok/error)"
        text error_message "Thông báo lỗi nếu có"
    }

    MES_OBJECTS {
        int id PK "Khóa chính tự tăng (AUTOINCREMENT)"
        text run_id FK "Khóa ngoại tham chiếu FOV_RUNS (CASCADE)"
        int row_id "ID dòng kết quả đo"
        text layer_name "Tên lớp/tầng đo đạc"
        int grid_row "Vị trí hàng trong lưới đo"
        int grid_col "Vị trí cột trong lưới đo"
        real soh "Chỉ số SOH (Structure Height/Offset)"
        real c1_volume "Thể tích C1"
        real c2_volume "Thể tích C2"
        real ratio "Tỷ lệ đo"
        text judgment "Đánh giá chi tiết vật thể (OK/NG)"
        real pitch_x "Khoảng cách bước Pitch X"
        real pitch_y "Khoảng cách bước Pitch Y"
        int z_min "Giới hạn Z nhỏ nhất (Bounding Box 3D)"
        int z_max "Giới hạn Z lớn nhất (Bounding Box 3D)"
        int y_min "Giới hạn Y nhỏ nhất (Bounding Box 3D)"
        int y_max "Giới hạn Y lớn nhất (Bounding Box 3D)"
        int x_min "Giới hạn X nhỏ nhất (Bounding Box 3D)"
        int x_max "Giới hạn X lớn nhất (Bounding Box 3D)"
        real centroid_z "Tọa độ trọng tâm Z"
        real centroid_y "Tọa độ trọng tâm Y"
        real centroid_x "Tọa độ trọng tâm X"
        text extra_json "Dữ liệu đo đạc bổ sung dạng JSON"
    }

    ARTIFACTS {
        int id PK "Khóa chính tự tăng"
        text run_id FK "Khóa ngoại tham chiếu FOV_RUNS (CASCADE)"
        text kind "Phân loại Artifact (image_ng, mesh_3d...)"
        text path "Đường dẫn tập tin lưu trữ"
        int exists_flag "Trạng thái tồn tại file (1: Có, 0: Mất)"
    }

    TIMELINE {
        int id PK "Khóa chính tự tăng"
        text run_id FK "Khóa ngoại tham chiếu FOV_RUNS (CASCADE)"
        text ts "Thời điểm (Timestamp)"
        text step "Tên bước xử lý (Enhance, Seg, Mes...)"
        text message "Thông điệp nhật ký"
        real duration_sec "Thời gian chạy bước đó (giây)"
    }

    WAFERS ||--o{ CHIPS : "1 Wafer chứa N Chips"
    WAFERS ||--o{ FOV_RUNS : "1 Wafer chạy N lần FOV Run"
    FOV_RUNS ||--o{ MES_OBJECTS : "1 Run đo đạc N Objects (CASCADE DELETE)"
    FOV_RUNS ||--o{ ARTIFACTS : "1 Run sinh ra N Artifacts (CASCADE DELETE)"
    FOV_RUNS ||--o{ TIMELINE : "1 Run ghi nhận N Timeline Logs (CASCADE DELETE)"
```

---

## 2. Sơ đồ Luồng Dữ liệu & Mối quan hệ Phân cấp (Hierarchy Data Flow)

Dưới đây là sơ đồ phân cấp dữ liệu từ mức **Wafer** xuống **Chip**, **FOV Run** và các dữ liệu đo đạc chi tiết:

```mermaid
flowchart TD
    subgraph LEVEL_1 [Cấp Wafer]
        W[WAFERS<br/><i>wafer_key = date|lot|wafer_id</i>]
    end

    subgraph LEVEL_2 [Cấp Chip & Lượt kiểm tra FOV]
        C[CHIPS<br/><i>chip_key = wafer_key|col|row</i>]
        R[FOV_RUNS<br/><i>run_id = unique_run_id</i>]
    end

    subgraph LEVEL_3 [Cấp Kết quả Đo đạc & Artifacts]
        M[MES_OBJECTS<br/><i>Chi tiết kích thước 3D / Centroid / Volumes</i>]
        A[ARTIFACTS<br/><i>File ảnh NG / File Mesh 3D / Raw Data</i>]
        T[TIMELINE<br/><i>Lịch sử thời gian từng step xử lý</i>]
    end

    W -->|1 : N| C
    W -->|1 : N| R
    R -.->|Thuộc Chip| C
    R -->|1 : N ON DELETE CASCADE| M
    R -->|1 : N ON DELETE CASCADE| A
    R -->|1 : N ON DELETE CASCADE| T

    style W fill:#1f77b4,color:#fff,stroke:#333,stroke-width:2px
    style C fill:#2ca02c,color:#fff,stroke:#333,stroke-width:2px
    style R fill:#ff7f0e,color:#fff,stroke:#333,stroke-width:2px
    style M fill:#d62728,color:#fff,stroke:#333,stroke-width:1px
    style A fill:#9467bd,color:#fff,stroke:#333,stroke-width:1px
    style T fill:#8c564b,color:#fff,stroke:#333,stroke-width:1px
```

---

## 3. Tóm tắt các Chỉ mục (Indexes) Tối ưu hóa truy vấn SQL

| Tên Index | Bảng áp dụng | Cột tạo Index | Mục đích |
| :--- | :--- | :--- | :--- |
| `idx_runs_wafer` | `fov_runs` | `wafer_key` | Tăng tốc tìm kiếm danh sách lượt chạy theo từng Wafer |
| `idx_runs_lot` | `fov_runs` | `lot_foup_id` | Tăng tốc lọc báo cáo theo Lot / FOUP ID |
| `idx_runs_finished` | `fov_runs` | `finished_at` | Tăng tốc truy vấn danh sách lượt chạy gần đây theo thời gian |
| `idx_mes_run` | `mes_objects` | `run_id` | Tăng tốc load danh sách vật thể đo đạc khi chọn 1 FOV Run |
| `idx_art_run` | `artifacts` | `run_id` | Tăng tốc load các file ảnh / model 3D tương ứng với 1 FOV Run |
