# Dragonfly-like Large Volume Engine Plan

## Mục tiêu
- Tái kiến trúc Viewer theo hướng hybrid out-of-core: brick cache + multi-resolution pyramid + progressive MPR/3D.
- Loại bỏ phụ thuộc vào dense full-volume NumPy trong đường render nóng.
- Giữ chất lượng ảnh MPR cuối cùng tương đương level-0 hiện tại (voxel-equivalent sau refine).

## Vì sao cần kiến trúc mới
- Mô hình hiện tại giữ `self.volume_data` dạng dense trong RAM và nhiều đường code vẫn giả định full array.
- Nút thắt chính không chỉ là throttle input: còn do full-volume/slice copies và render VTK nặng trên main thread.
- Với dữ liệu cỡ 15 GB+, tối ưu incremental chỉ giảm phần triệu chứng; cần đổi kiến trúc dữ liệu + scheduler render.

## Kiến trúc đích (high-level)
1. Input volume đi vào background builder.
2. Builder tạo:
   - Level-0 lossless bricks.
   - Pyramid downsample 2x nhiều level.
3. `VolumeStore` cung cấp API truy xuất theo lát/ROI/brick/level.
4. RAM cache giới hạn (LRU), có prefetch và hủy request stale.
5. Progressive renderer:
   - MPR: preview mip thấp khi drag, idle thì refine level-0.
   - 3D: coarse-to-fine theo ngân sách VRAM, không ép upload toàn bộ level-0.

## Kế hoạch triển khai

### 1) Baseline + performance gates
- Đo latency/memory ở các pha: load, slice extraction, VTK upload, render.
- Đặt tiêu chí nghiệm thu theo p50/p95, RSS/VRAM, cancellation, cache hit rate trên dataset 15 GB.

### 2) Out-of-core store + multi-resolution cache
- Thêm `VolumeStore` (metadata + async `get_slice/get_roi/get_bricks/get_level`).
- Thêm cache backend có key theo file signature + schema version, lock/atomic publish.
- Thêm builder nền tạo level-0 + pyramid theo brick/chunk có benchmark.

### 3) Viewer migration từ NumPy monolith
- Chuyển ownership ở Viewer từ `self.volume_data` sang `self.volume_store`.
- Refactor load path để không emit dense 15 GB sang GUI thread.
- Các workflow cần full volume phải explicit (ROI/materialization có cảnh báo).

### 4) Progressive MPR with parity
- Generation ID + cancel stale requests.
- Drag: render mip phù hợp viewport/zoom.
- Idle (80-120 ms): refine level-0 và chỉ apply nếu generation còn hợp lệ.
- Giữ actor/buffer VTK persistent, tránh remove/rebuild toàn bộ mỗi frame.

### 5) Progressive 3D theo VRAM budget
- Bỏ đường full-volume `transpose -> flatten -> deep copy`.
- Chọn mip 3D theo VRAM khả dụng + headroom.
- Tương tác dùng mip coarse, idle mới refine nếu đủ budget.

### 6) Verification + rollout
- Unit: cache key/invalidation, pyramid dimensions, cancellation, LRU.
- Golden: parity axial/coronal/sagittal level-0 với pipeline hiện tại.
- Performance benchmark 15 GB, rollout qua feature flag `large_volume_engine`.

## Trạng thái triển khai (Implementation status)

> Cập nhật theo codebase hiện tại. Feature flag `large_volume_engine` (`INNO3D_LARGE_VOLUME_ENGINE=1` hoặc `host.large_volume_engine`) **mặc định OFF**.

### Đã có (skeleton / một phần)
- [x] **§1 Baseline:** `inno3d/core/perf_timing.py` — log `[PERF]` khi `INNO3D_PERF=1`; đã gắn một số pha load/render.
- [x] **§2 Store + pyramid:** `VolumeStore` (abstract + `MemoryVolumeStore`, `MemmapVolumeStore`), `volume_pyramid.PyramidBuilder`, `infra/volume_cache.py`; unit tests `test_volume_store`, `test_volume_pyramid`, `test_perf_timing`.
- [x] **§3 Viewer (RAW):** `LoadVolumeThread` mở RAW qua memmap/VolumeStore khi flag bật — không copy dense 15 GB sang GUI thread.
- [x] **§4 Progressive MPR (khi flag ON):** generation ID, preview mip khi drag, idle refine level-0 (~100 ms) trong `mpr_render.py`.
- [x] **§6 Rollout cơ bản:** flag env/host; chưa bật mặc định.

### Band-aid tạm (không phải end-state Dragonfly)
- [x] Volume **> 8 GB** (kể cả ~10.5 GB): log `[PERF] … quality=Draft …` và **tự tắt 3D volume render** (`_apply_large_volume_perf_settings` trong `volume_io.py`). User có thể bật lại bằng nút 3D toggle.
- **Lưu ý:** Hành vi auto-disable 3D này là giải pháp PERF ngắn hạn trên pipeline dense cũ — **không** phải mục tiêu §5 (coarse-to-fine 3D theo VRAM).

### Đang làm / tiếp theo
- [ ] **§5 3D coarse mip + VRAM budget** — thay auto-disable bằng upload mip thô + refine idle (đang implement song song).
- [ ] **§2–3 TIFF/folder out-of-core** — hiện vẫn wrap dense NumPy (`MemoryVolumeStore`); chưa brick cache thật cho stack TIFF/folder.
- [ ] **§3 Migration đầy đủ** — nhiều workflow vẫn giả định `self.volume_data` dense.
- [ ] **§1 + §6 Gates** — benchmark 15 GB, golden parity level-0, p50/p95 RSS/VRAM; sau đó mới **bật flag mặc định** cho Viewer.

## Thứ tự giao hàng đề xuất
1. Instrumentation + tầng `VolumeStore`/cache builder.
2. Viewer load progressive + MPR parity level-0.
3. MPR preview/refine + cancellation + persistent buffers.
4. 3D mip/VRAM budgeting.
5. Benchmark/regression fix + bật mặc định cho Viewer.
6. Sau khi Viewer ổn định mới mở rộng Teaching/Online.

## Ghi chú chuyển tiếp từ kế hoạch cũ
- Kế hoạch trong `implementation_plan_reduce_lag_lagre_input_volume.md` vẫn hữu ích cho tối ưu incremental ngắn hạn.
- Tuy nhiên, với mục tiêu Dragonfly-like ở data volume rất lớn, tài liệu này là hướng kiến trúc chính để đạt độ mượt ổn định và scale dài hạn.
