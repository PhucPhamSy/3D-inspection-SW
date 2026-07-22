# Note: Tóm tắt email Wafer Map Draw Code

| Mục | Nội dung |
|-----|----------|
| **Nguồn** | `wafer email.pdf` (thread InnoMetry) |
| **Tiêu đề** | Re: Wafer Map Draw Code / FW:Re:Wafer Map Draw Code |
| **Người gửi chính** | Nam Yong Ho (남용호) — yongho.nam@innometry.com |
| **Người nhận / CC** | Si Phuc (syphucpham@innometry.com), đội SW / Pixel, InnoMetry… |
| **Mục đích** | Mô tả **logic wafer map**, **cấu trúc dữ liệu Wafer → Chip → Point(FOV)**, và **hình mock display** để app inspection tham chiếu khi hiển thị kết quả. |

---

## 1. Bối cảnh nghiệp vụ

Inspection CT/HBM không chỉ mở một volume lẻ. Hierarchy:

```
Wafer (thường 300 mm)
  └── Chip / Die  (ô trên map, ví dụ 25×25)
        └── Point / FOV × 9  (P1 … P9)
              └── Volume CT  (= nội dung FOV đang xem trong MPR/3D)
```

- **Volume 3D viewer** = nội dung **một FOV (một Point)**.
- Email giải thích lớp **định vị + kết quả**: FOV thuộc chip nào, chip nằm đâu trên wafer, OK/NG thế nào.

---

## 2. Mã bin (Bin code)

| Giá trị | Ý nghĩa |
|--------:|---------|
| **0** | Outside wafer — ngoài vòng wafer, không inspect |
| **1** | Bad / **NG** |
| **8** | Good / **OK** |

Áp dụng cho: ô trên wafer map, và từng Point (sau inspect).

---

## 3. Logic code tóm tắt (trong email)

```
MAP_SIZE = 25
waferMap[y][x]   // 0 / 1 / 8
```

Với mỗi ô `(y, x)`:

1. Nếu `waferMap[y,x] == 0` → **skip** (outside).
2. Lấy `chipResult[y,x]`.
3. Lặp **9 point** `p = 0..8`:
   - `InspectPoint()` → OK?
   - `PointResult[p] = 8` (OK) hoặc `1` (NG).
4. `BadCount` = số point = 1; `GoodCount` = số point = 8.
5. `ChipNG = (BadCount > 0)` → **chỉ cần ≥1 point NG là cả chip NG**.
6. Cập nhật map: `waferMap[y,x] = 1` nếu ChipNG, else `8`.

### Ví dụ 2 chip (trong email)

| Chip (X,Y) | P1–P9 (rút gọn) | Judge |
|------------|-----------------|-------|
| (10, 12) | P5 = 1, còn lại 8 | **NG** |
| (11, 12) | Tất cả 8 | **OK** |

---

## 4. Cây dữ liệu (data model)

```
Wafer
├── LotID
├── WaferID
├── Chip[25][25]
│     ├── OriginalBin   (0/1/8) — map tải về / trước inspect
│     ├── FinalBin      (1/8)   — sau inspect 9 point
│     ├── Point[9]
│     │     ├── X Offset
│     │     ├── Y Offset
│     │     ├── CT Result
│     │     ├── AI Score
│     │     └── Judge (OK/NG → 8/1)
│     ├── GoodCount
│     ├── BadCount
│     └── ChipJudge
└── Yield
```

| Tầng | Ý nghĩa cho UI / app |
|------|----------------------|
| **Wafer** | Lot/Wafer; die nào outside / good / bad |
| **Chip** | Vị trí (X,Y); tổng hợp 9 point |
| **Point / FOV** | Offset trong chip; kết quả CT/AI; **volume** tương ứng |
| **Yield** | Tỷ lệ chip good trên wafer (sau FinalBin) |

---

## 5. Hình mô phỏng trong email

### 5.1 Hình kết quả — “25×25 Wafer Map (9 Point Inspection Result)”

**Không phải** ảnh CT thô; là **mock UI display** sau khi có kết quả 9-point.

Nội dung chính:

| Khối | Nội dung |
|------|----------|
| **Wafer Information** | Lot, Wafer ID, size 300 mm, die size, map 25×25, 9-point, datetime |
| **Map tròn giữa** | Lưới 01…25; xám=0 outside; xanh=8 good; đỏ=1 bad |
| **Cluster** | Khung nét đứt = nhóm die NG gần nhau; ⊕ = centroid (chip đại diện) |
| **Cluster Summary** | ID, size (# die), centroid (X,Y), max radius |
| **Selected Chip Detail** | Chip chọn (vd 17,16): Final NG, Good/Bad count, lưới 3×3 P1…P9 |
| **Statistics** | Total / Inside / Good / Bad / Outside / Cluster count |
| **Yield** | Ví dụ **Final Yield 93.58%** |
| **Bảng số full** | Ma trận 25×25 giá trị 0/1/8 (debug / export) |

**Quy tắc chip trên map:** màu/bin chip = FinalBin sau khi gộp 9 point (có NG → đỏ).

**Ví dụ chip (17,16) trên hình:**

- Final: NG  
- 3×3 point: cột P3/P6/P9 = 1 (NG), còn lại 8  
- Bad ratio ~44% trên 9 point  

### 5.2 Hình setup tool (trang cấu hình vẽ map)

Màn hình **cấu hình hình học** (không phải kết quả inspect):

- Product Type = Wafer  
- Start position / Product size (mm)  
- Array, pitch  
- Total count **25 × 25**  
- Device size / even pitch  
- Preview wafer tròn + lưới die  

→ “Với setting này sẽ vẽ map như hình preview”.

Hai lớp hình trong email:

1. **Setup** — định nghĩa lưới die trên wafer.  
2. **Result display** — tô màu 0/1/8 + cluster + chi tiết chip + yield.

---

## 6. Ý nghĩa cho Inno3D (Inspection)

| Email | Inno3D (Online CONTEXT) |
|-------|-------------------------|
| Lưới 25×25 + bin 0/1/8 | **Wafer map** |
| Point[9] + Judge | **Chip map** P1–P9 |
| Volume CT của 1 point | **MPR + 3D** (volume = FOV; không đổi layout MPR) |
| FinalBin / Yield / results | Stats, breadcrumb, lưu **FOV/results** |

Luồng mong muốn:

```
[CT Reconstruction PC / Host DB]
        gửi path volume FOV
                ↓
[Inno3D Online – Inspection]
        CONTEXT: Wafer | Chip FOV  (parse path → select chip + P1..P9)
        MPR/3D: volume FOV đang chọn
                ↓
        Ghi kết quả → …/ChipLocation/FOVLocation/Results
```

### Path mass-production (host filesystem)

```
{yy_mm_dd}/{LotID_FoupID}/{WaferID}/
    {ChipLocationinWafer}/{FOVLocationinChip}/Input.tiff
    {ChipLocationinWafer}/{FOVLocationinChip}/Results/   ← inspection outputs
```

| Segment | Ví dụ | UI CONTEXT |
|---------|--------|------------|
| Date | `26_07_18` | breadcrumb |
| LotID_FoupID | `LOT240701_F01` | Lot/Foup |
| WaferID | `WAFER05` | Wafer map title |
| ChipLocation | `Chip_17_16` → (17,16) | **Wafer map** highlight |
| FOVLocation | `FOV_P5` → P5 | **Chip map** amber FOV |
| Input.tiff | volume CT | MPR / 3D |
| Results/ | masks, CSV, … | footer path |

Parse/link code: `inno3d/core/wafer_context.py` (`parse_host_volume_path`, `apply_host_path`).

---

## 7. Tóm một câu

Email **Wafer Map Draw Code** định nghĩa:

1. **Map 25×25** trên wafer với bin **0 / 1 / 8**.  
2. Mỗi chip inspect **9 FOV (Point)**; chip NG nếu ≥1 point NG.  
3. **Cây dữ liệu** Wafer → Chip → Point (offset, CT, AI, Judge).  
4. **Hình mock** hiển thị kết quả + cluster + yield; kèm **UI setup** vẽ map.  

Không mô tả chi tiết render volume 3D — mô tả **bản đồ inspection** để biết FOV đang xem nằm ở đâu trong chip và trên wafer.

---

## 8. Tham chiếu file dự án

| File | Vai trò |
|------|---------|
| `wafer email.pdf` | Email gốc + hình |
| `docs/NOTE_Wafer_Map_Email_Summary.md` | Note này |
| `inno3d/core/wafer_context.py` | Model Wafer/Chip/FOV trong code |
| `inno3d/widgets/context_map_panel.py` | UI CONTEXT 2 tầng (Online) |
| `after add Wafer and Chip map.png` | Mockup UI Inno3D (sidebar → CONTEXT → MPR) |

---

*Ghi chú nội bộ dự án HBM / Inno3D — tóm từ email InnoMetry, không thay thế tài liệu chính thức host/DB.*
