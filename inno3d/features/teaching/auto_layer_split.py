# inno3d/features/teaching/auto_layer_split.py
# -----------------------------------------------------------------------
# Automatic HBM layer boundary detection via Z-profile analysis.
#
# Algorithm (designed for C++/C# portability — no scipy/sklearn):
#   1. Per-slice std_dev across full XY → 1D signal
#   2. Box-smooth + baseline normalization (detrend)
#   3. FFT → estimate dominant period (layer spacing)
#   4. Local minima → prominence scoring → greedy pick with
#      min-distance from FFT period
#   5. Fill gaps using period-guided local-minimum search
#   6. Build layer boundaries with small inward margins
#
# C++ porting notes:
#   - _box_smooth: cumulative-sum → O(N)
#   - std_dev: E[X²]-E[X]² with SIMD (or Welford single-pass)
#   - FFT: FFTW / MathNet.Numerics
#   - prominence + greedy pick: simple loops + partial_sort
# -----------------------------------------------------------------------

import numpy as np


def _box_smooth(signal, kernel_size):
    """Vectorised 1D box smooth using cumulative sum. O(N).

    C++ port: keep a running sum; add right element, subtract left.
    """
    n = len(signal)
    half = kernel_size // 2
    cumsum = np.zeros(n + 1, dtype=np.float64)
    cumsum[1:] = np.cumsum(signal)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = (cumsum[hi] - cumsum[lo]) / (hi - lo)
    return out


def auto_detect_layers(volume_data, num_layers,
                       skip_air_top=10, skip_air_bottom=10):
    """Detect HBM layer Z-boundaries from a 3D volume.

    Uses per-slice std_dev with baseline normalization and FFT-guided
    minimum-distance constraints for robust valley detection.

    Parameters
    ----------
    volume_data : ndarray, shape (Z, Y, X)
        Raw CT volume (uint8, uint16, or float).
    num_layers : int
        Desired number of layers (typically 8, 12, or 16).
    skip_air_top, skip_air_bottom : int
        Slices to skip at the top/bottom (air gaps / substrate).

    Returns
    -------
    list of dict
        Each dict: {'id', 'name', 'z_start', 'z_end', 'selected'}.
    """
    z_total = volume_data.shape[0]

    if z_total < 20 or num_layers < 1:
        step = max(1, z_total // max(1, num_layers))
        return [
            {'id': i, 'name': f'Layer {i+1}',
             'z_start': i * step,
             'z_end': min(z_total - 1, (i + 1) * step),
             'selected': True}
            for i in range(num_layers)
        ]

    # ── 1. Per-slice std dev (full XY for best signal quality) ──
    # Var = E[X²] - E[X]²  →  single-pass, SIMD-friendly in C++
    std_profile = np.empty(z_total, dtype=np.float64)
    for z in range(z_total):
        sl = volume_data[z].ravel().astype(np.float64)
        n = sl.size
        s1 = sl.sum()
        s2 = (sl * sl).sum()
        var = s2 / n - (s1 / n) ** 2
        std_profile[z] = np.sqrt(max(0.0, var))

    # ── 2. Denoise smooth (small kernel) ──
    ks_small = max(5, min(z_total // 80, 15))
    ks_small |= 1
    smoothed = _box_smooth(std_profile, ks_small)

    # ── 3. Baseline normalization ──
    # Large kernel captures global trend (std typically decreases top→bottom).
    # Dividing removes the trend so valleys at all depths are equally prominent.
    ks_large = max(51, z_total // 4)
    ks_large |= 1
    baseline = _box_smooth(std_profile, ks_large)
    baseline = np.maximum(baseline, 1.0)
    normalized = smoothed / baseline

    # ── 4. FFT → dominant period ──
    z_lo = max(0, skip_air_top)
    z_hi = min(z_total, z_total - skip_air_bottom)
    z_valid = std_profile[z_lo:z_hi]
    n_valid = len(z_valid)

    trend = np.linspace(z_valid[0], z_valid[-1], n_valid)
    detrended = z_valid - trend
    fft_mag = np.abs(np.fft.rfft(detrended))

    min_period = max(20, z_total // (num_layers * 2))
    max_period = z_total // max(2, num_layers // 2)
    freq_lo = max(1, n_valid // max_period)
    freq_hi = min(len(fft_mag) - 1, n_valid // min_period)

    if freq_hi > freq_lo:
        peak_freq = np.argmax(fft_mag[freq_lo:freq_hi + 1]) + freq_lo
        period = n_valid / peak_freq if peak_freq > 0 else z_total / num_layers
    else:
        period = z_total / num_layers

    # ── 5. Find local minima ──
    minima = []
    for z in range(z_lo + 1, z_hi - 1):
        if normalized[z] < normalized[z - 1] and normalized[z] < normalized[z + 1]:
            minima.append((z, normalized[z]))

    # ── 6. Prominence scoring ──
    scored = []
    for z_min, val_min in minima:
        left_max = val_min
        for z in range(z_min - 1, z_lo - 1, -1):
            left_max = max(left_max, normalized[z])
            if (z_lo < z < z_hi - 1
                    and normalized[z] < normalized[z - 1]
                    and normalized[z] < normalized[z + 1]
                    and normalized[z] <= val_min):
                break
        right_max = val_min
        for z in range(z_min + 1, z_hi):
            right_max = max(right_max, normalized[z])
            if (z_lo < z < z_hi - 1
                    and normalized[z] < normalized[z - 1]
                    and normalized[z] < normalized[z + 1]
                    and normalized[z] <= val_min):
                break
        prom = min(left_max - val_min, right_max - val_min)
        scored.append((z_min, val_min, prom))

    # ── 7. Greedy pick with min-distance ──
    min_dist = max(25, int(period * 0.50))
    scored.sort(key=lambda x: -x[2])

    selected = []
    for z_min, val, prom in scored:
        if len(selected) >= num_layers - 1:
            break
        too_close = any(abs(z_min - s) < min_dist for s in selected)
        if not too_close:
            selected.append(z_min)

    valleys = sorted(selected)

    # ── 8. Fill gaps using period-guided local-minimum search ──
    while len(valleys) < num_layers - 1:
        all_edges = [z_lo] + valleys + [z_hi]
        max_gap, gap_idx = 0, 0
        for i in range(len(all_edges) - 1):
            g = all_edges[i + 1] - all_edges[i]
            if g > max_gap:
                max_gap, gap_idx = g, i

        gap_start = all_edges[gap_idx]
        gap_end = all_edges[gap_idx + 1]
        n_fill = max(1, round(max_gap / period))
        fill_step = max_gap / (n_fill + 1)

        added = False
        for j in range(1, n_fill + 1):
            if len(valleys) >= num_layers - 1:
                break
            target = int(gap_start + j * fill_step)
            search_lo = max(z_lo, target - int(period / 3))
            search_hi = min(z_hi, target + int(period / 3))

            best_z, best_val = target, float('inf')
            for z in range(search_lo, search_hi):
                if normalized[z] < best_val:
                    best_val = normalized[z]
                    best_z = z

            too_close = any(abs(best_z - s) < min_dist for s in valleys)
            if not too_close:
                valleys.append(best_z)
                added = True

        if not added:
            valleys.append((gap_start + gap_end) // 2)

        valleys.sort()

    valleys = valleys[:num_layers - 1]

    # ── 9. Build layer definitions ──
    med_norm = float(np.median(normalized[z_lo:z_hi]))
    data_start = z_lo
    for z in range(z_lo, z_hi):
        if normalized[z] > med_norm * 0.7:
            data_start = max(z_lo, z - 2)
            break
    data_end = z_hi
    for z in range(z_hi - 1, z_lo, -1):
        if normalized[z] > med_norm * 0.7:
            data_end = min(z_hi, z + 2)
            break

    all_bounds = [data_start] + valleys + [data_end]

    layers = []
    for i in range(len(all_bounds) - 1):
        z_s = all_bounds[i]
        z_e = all_bounds[i + 1]

        # Inward margin from gap center (~12% of half-gap width)
        if i > 0:
            half_gap = (all_bounds[i] - all_bounds[i - 1]) // 2
            margin = max(2, int(half_gap * 0.12))
            z_s = all_bounds[i] + margin
        if i < len(all_bounds) - 2:
            half_gap = (all_bounds[i + 2] - all_bounds[i + 1]) // 2
            margin = max(2, int(half_gap * 0.12))
            z_e = all_bounds[i + 1] - margin

        layers.append({
            'id': i,
            'name': f'Layer {i + 1}',
            'z_start': max(0, z_s),
            'z_end': min(z_total - 1, z_e),
            'selected': True,
        })

    return layers
