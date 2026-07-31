# Performance Optimization for Large Volumes (15GB+)

> [!IMPORTANT]
> This document is the **incremental optimization path** (slice copies, render throttling, selective 3D mitigations).
> For the **architectural path** toward Dragonfly-like large-volume behavior (out-of-core brick cache + pyramid + progressive MPR/3D), use:
> `docs/dragonfly_volume_engine_plan.md`.

## Problem Analysis

Khi load volume ~15GB (ví dụ uint16, ~2700×2700×2700), mọi thao tác (click mapping → ảnh, kéo crosshair, slider) đều giật lag, dù cấu hình máy rất mạnh (Xeon W7-3465X, 512GB RAM, RTX 5090).

**Root cause**: Sau khi phân tích toàn bộ code path `updatePoint → render_slice → VTK Render`, tôi xác định được **5 bottleneck chính**:

### Bottleneck 1: `render_slice` — `np.flipud` / `np.transpose` tạo full copy mỗi frame
- File: [mpr_render.py](file:///e:/semiconductor/HBM_DEV_FOR_PROD/DEV/all_v2/HBM_frontend_backend_v12/inno3d/features/viewer/mpr_render.py#L58-L80)
- Coronal: `np.flipud(self.volume_data[:, slice_idx, :])` → cắt 1 slice Z×X (~14MB cho 2700×2700 uint16), rồi flipud tạo **copy** 14MB
- Sagittal: `np.transpose(self.volume_data[:, :, slice_idx])` → cắt Y×Z (~14MB), rồi transpose tạo **copy** 14MB
- VTK data path: `np.transpose(slice_data, (1,0))` → thêm 1 copy nữa, rồi `np.ascontiguousarray(flatten('F'))` — **thêm 1 copy**
- **Mỗi render_slice gây 2-3 lần copy ~14MB** → cho 3 planes = **~126MB copy mỗi frame**

### Bottleneck 2: `numpy_to_vtk(deep=True)` — luôn copy toàn bộ slice data vào VTK
- File: [mpr_render.py](file:///e:/semiconductor/HBM_DEV_FOR_PROD/DEV/all_v2/HBM_frontend_backend_v12/inno3d/features/viewer/mpr_render.py#L237-L258)
- `deep=True` buộc VTK tạo thêm **internal copy** ~14MB mỗi slice
- Có thể dùng `deep=False` nếu ta đảm bảo numpy array sống đủ lâu

### Bottleneck 3: `render_slice` luôn rebuild ruler/overlay actors  
- File: [mpr_render.py](file:///e:/semiconductor/HBM_DEV_FOR_PROD/DEV/all_v2/HBM_frontend_backend_v12/inno3d/features/viewer/mpr_render.py#L159-L180)
- Dù dùng "fast path" (reuse image actor), vẫn **RemoveViewProp tất cả non-base actors** rồi **rebuild crosshair + ruler + overlay** mỗi frame
- Ruler có 8+ VTK actors (lines + text) → tạo mới mỗi frame = chậm

### Bottleneck 4: `update_3d_crosshair()` — Render() trên 3D widget mỗi frame
- File: [crosshair.py](file:///e:/semiconductor/HBM_DEV_FOR_PROD/DEV/all_v2/HBM_frontend_backend_v12/inno3d/features/viewer/crosshair.py#L760-L763)
- Với volume 15GB đã upload lên GPU, mỗi `GetRenderWindow().Render()` trên 3D pane = 1 full ray-cast frame
- Kéo crosshair gọi `updatePoint` → `update_3d_crosshair` → **Render() on 3D** → extremely expensive

### Bottleneck 5: `render_3d()` — `np.transpose + flatten` tạo 15GB temp copy
- File: [volume_3d.py](file:///e:/semiconductor/HBM_DEV_FOR_PROD/DEV/all_v2/HBM_frontend_backend_v12/inno3d/features/viewer/volume_3d.py#L1871-L1882)
- `np.transpose(self.volume_data, (2, 1, 0))` → view (OK, no copy)
- `np.ascontiguousarray(data_fortran.flatten('F'))` → **15GB copy**
- `numpy_to_vtk(flat_data, deep=True)` → **another 15GB copy**
- Tổng: volume 15GB cần **~45GB RAM** chỉ để build VTK data

---

## Proposed Changes

### Phase 1: Tối ưu MPR slice rendering (impact lớn nhất cho interactivity)

#### [MODIFY] [mpr_render.py](file:///e:/semiconductor/HBM_DEV_FOR_PROD/DEV/all_v2/HBM_frontend_backend_v12/inno3d/features/viewer/mpr_render.py)

**1a. Zero-copy slice extraction + VTK data update:**
- Axial slice: `self.volume_data[z, :, :]` → đây đã là contiguous view, **không cần copy**
- Coronal slice: thay `np.flipud(...)` bằng slicing `[::-1]` rồi `.copy()` chỉ khi VTK cần
- Sagittal slice: dùng `.copy()` only once thay vì transpose + flatten + copy  
- Thay `deep=True` bằng `deep=False` + giữ reference trong cache → **giảm 50% memory copy per frame**

**1b. Persistent ruler & crosshair actors:**
- Ruler actors: tạo 1 lần, chỉ **update position/text** thay vì destroy+recreate
- Khi chỉ thay đổi slice data (crosshair drag) mà overlay state không đổi → skip rebuild ruler, **chỉ update scalars**

**1c. Batch VTK renders:**
- Thay vì mỗi `render_slice` gọi `widget.GetRenderWindow().Render()` riêng, dùng `StartRender/EndRender` batch cho tất cả 3 panes

---

### Phase 2: Tối ưu 3D crosshair update

#### [MODIFY] [crosshair.py](file:///e:/semiconductor/HBM_DEV_FOR_PROD/DEV/all_v2/HBM_frontend_backend_v12/inno3d/features/viewer/crosshair.py)

**2a. Defer 3D render during crosshair drag:**
- Trong `updatePoint`, khi đang drag (rapid events), **chỉ update crosshair line geometry** mà không gọi `Render()` trên 3D widget
- Dùng debounce timer (~100ms) để gọi 3D Render() khi user dừng drag
- Điều này giải quyết bottleneck lớn nhất: mỗi pixel kéo crosshair hiện tại trigger 1 full 3D ray-cast

**2b. Skip 3D crosshair update khi 3D pane không visible:**
- Nếu 3D widget bị collapsed/hidden hoặc fullscreen 1 pane → skip `update_3d_crosshair()` entirely

---

### Phase 3: Tối ưu render_3d data pipeline

#### [MODIFY] [volume_3d.py](file:///e:/semiconductor/HBM_DEV_FOR_PROD/DEV/all_v2/HBM_frontend_backend_v12/inno3d/features/viewer/volume_3d.py)

**3a. Fortran-order storage:**
- Khi load volume, store `self.volume_data` as `np.asfortranarray()` → `np.transpose(data, (2,1,0))` trở thành view thay vì copy
- Hoặc: tạo 1 pre-transposed copy riêng cho 3D renderer

**3b. VTK ImportVoid thay vì numpy_to_vtk:**
- Dùng `vtkImageImport` + NumPy buffer pointer → zero-copy data transfer to VTK/GPU
- Giảm peak RAM từ ~45GB xuống ~30GB cho volume 15GB

---

### Phase 4: Intelligent volume size detection

#### [MODIFY] [volume_io.py](file:///e:/semiconductor/HBM_DEV_FOR_PROD/DEV/all_v2/HBM_frontend_backend_v12/inno3d/features/viewer/volume_io.py)

**4a. Auto-detect large volume và điều chỉnh quality presets:**
- Nếu `data.nbytes > 4GB` → auto-switch quality preset sang "Draft" hoặc "Standard"
- Auto-disable 3D volume render cho volumes > 8GB (user có thể bật lại manual)
- Hiển thị warning message cho user

**4b. Throttle crosshair drag interval cho large volumes:**
- Nếu volume > 4GB, tăng throttle timer từ 16ms → 33ms (~30fps thay vì 60fps)

---

## Implementation Priority

| Phase | Impact | Effort | Notes |
|-------|--------|--------|-------|
| **Phase 2a** | ⭐⭐⭐⭐⭐ | Low | Debounce 3D Render = biggest win, smallest change |
| **Phase 1a** | ⭐⭐⭐⭐ | Medium | Zero-copy slicing giảm ~60% CPU per frame |
| **Phase 4a** | ⭐⭐⭐⭐ | Low | Auto-disable 3D + auto-quality = immediate UX improvement |
| **Phase 1b** | ⭐⭐⭐ | Medium | Persistent rulers/crosshair actors |
| **Phase 4b** | ⭐⭐ | Low | Adaptive throttle |
| **Phase 3** | ⭐⭐⭐ | High | Fortran order + VTK import = advanced optimization |

> [!IMPORTANT]
> **Phase 2a (debounce 3D render) là giải pháp có impact lớn nhất với effort nhỏ nhất.** Hiện tại mỗi pixel drag crosshair gây ra 1 full 3D ray-cast render (~50-200ms cho volume 15GB trên RTX 5090). Chỉ cần debounce 100ms đã giảm 80%+ render calls.

> [!NOTE]
> Phase 3 (Fortran order + VTK import) cần thay đổi sâu hơn và có risk side-effects, nên tôi đề xuất implement sau khi Phase 1-2-4 đã verified stable.

## Open Questions

1. **Volume lớn nhất bạn cần hỗ trợ là bao nhiêu?** Nếu > 24GB (vượt VRAM RTX 5090) thì cần downsampling strategy riêng cho 3D render.

2. **Có muốn auto-disable 3D render cho volumes > 8GB không?** Hiện tại Online mode đã disable by default. Manual load có thể giữ behavior tương tự.

3. **Bạn thường dùng tab nào khi gặp lag?** (3D Viewer / 3D Teaching / cả hai) — để tôi ưu tiên đúng code path.

## Verification Plan

### Manual Verification
- Load volume ~15GB trên máy RTX 5090
- Test crosshair drag: so sánh FPS trước/sau (dùng `time.perf_counter()` profiling)
- Test click từ mapping table → ảnh: đo latency
- Test slider drag: kiểm tra smoothness
- Verify 3D render quality vẫn đúng sau optimizations

### Automated Profiling
- Thêm timing instrumentation vào `render_slice` và `updatePoint` 
- Log ra `[PERF]` messages để compare trước/sau
