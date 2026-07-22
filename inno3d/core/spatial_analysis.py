"""
3D Structural Integrity Map (3D-SIM) — Core Analysis Engine

Novel spatial analysis for HBM micro-bump inspection:
- Moran's I spatial autocorrelation (global + LISA local)
- Cross-layer vertical defect propagation
- Structural Integrity Index (SII)
- Weak zone auto-detection
- Process fingerprint computation

Author: Inno3D Team
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, norm
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ─────────────────────── Data Structures ───────────────────────

@dataclass
class BumpNode:
    """Single bump measurement."""
    idx: int
    layer: str
    cx: float  # centroid X
    cy: float  # centroid Y
    cz: float  # centroid Z
    soh: float  # state of height (µm)
    void_ratio: float
    bump_volume: float
    judgment: str  # 'OK' or 'NG'


@dataclass
class MoranResult:
    """Moran's I result for one layer."""
    layer: str
    I_global: float  # -1 (dispersed) to +1 (clustered)
    z_score: float
    p_value: float
    pattern: str  # 'Clustered', 'Dispersed', 'Random'
    local_I: np.ndarray  # per-bump LISA values
    hotspot_labels: np.ndarray  # 'HH', 'LL', 'HL', 'LH', 'NS'


@dataclass
class PropagationResult:
    """Cross-layer propagation analysis."""
    layer_names: List[str]
    correlation_matrix: np.ndarray  # N_layers × N_layers
    propagation_pairs: List[dict]  # [{from_layer, to_layer, r, p, direction}]
    dominant_direction: str  # 'bottom-up', 'top-down', 'none'
    overall_score: float  # 0-1, higher = more propagation


@dataclass
class SIIResult:
    """Structural Integrity Index results."""
    bump_indices: np.ndarray
    sii_values: np.ndarray  # 0.0 (critical) to 1.0 (perfect)
    cx: np.ndarray
    cy: np.ndarray
    layer: str
    mean_sii: float
    std_sii: float
    risk_count: int  # bumps with SII < threshold


@dataclass
class WeakZone:
    """Detected weak zone on die."""
    zone_id: int
    bbox: Tuple[float, float, float, float]  # x_min, y_min, x_max, y_max
    centroid: Tuple[float, float]
    area: float
    n_bumps: int
    mean_sii: float
    severity: str  # 'Critical', 'Warning', 'Watch'
    zone_type: str  # 'corner', 'edge', 'center'


@dataclass
class ProcessFingerprint:
    """Process fingerprint for a sample."""
    sample_name: str
    moran_per_layer: List[float]
    vert_corr_per_pair: List[float]
    mean_sii: float
    std_sii: float
    defect_density_per_layer: List[float]
    dominant_direction: str
    vector: np.ndarray  # flattened fingerprint for comparison


# ─────────────────────── Helper: Parse raw_rows → BumpNodes ───────────────────────

def _safe_float(val, default=0.0):
    if not val or val == 'N/A':
        return default
    val = str(val).replace(',', '').replace('%', '').strip()
    try:
        return float(val)
    except ValueError:
        return default


def parse_bumps_from_rows(raw_rows: List[dict]) -> Dict[str, List[BumpNode]]:
    """Convert raw CSV rows into BumpNode objects grouped by layer."""
    layers: Dict[str, List[BumpNode]] = {}
    for i, row in enumerate(raw_rows):
        layer = row.get('_layer_resolved', row.get('Layer', 'Default')).strip()
        cx = _safe_float(row.get('Centroid_X', row.get('centroid_x', '0')))
        cy = _safe_float(row.get('Centroid_Y', row.get('centroid_y', '0')))
        cz = _safe_float(row.get('Centroid_Z', row.get('centroid_z', '0')))
        soh = _safe_float(row.get('SOH (um)', row.get('SOH(µm)',
              row.get('Z Height (um)', row.get('Z Height(µm)', '0')))))
        if soh == 0.0:
            z_min = _safe_float(row.get('Z_min', '0'))
            z_max = _safe_float(row.get('Z_max', '0'))
            soh = abs(z_max - z_min)
        void_ratio = _safe_float(row.get('Ratio (Void/(TGV+Void))',
                     row.get('Ratio', '0')))
        if void_ratio > 1.0:
            void_ratio /= 100.0
        bump_vol = _safe_float(row.get('Bump Volume (um3)',
                   row.get('Bump Vol(µm³)', row.get('Bump Vol(um3)', '0'))))
        judgment = row.get('Judgment', row.get('judgment', 'OK')).strip()

        node = BumpNode(idx=i, layer=layer, cx=cx, cy=cy, cz=cz,
                        soh=soh, void_ratio=void_ratio,
                        bump_volume=bump_vol, judgment=judgment)
        layers.setdefault(layer, []).append(node)
    return layers


# ─────────────────────── 1. Moran's I Spatial Autocorrelation ───────────────────────

def compute_moran_i(bumps: List[BumpNode], metric: str = 'void_ratio',
                    k_neighbors: int = 6) -> Optional[MoranResult]:
    """
    Compute Global and Local Moran's I for a single layer.

    Global I: measures overall spatial autocorrelation
      I > 0 → clustered, I ≈ 0 → random, I < 0 → dispersed

    Local I (LISA): identifies per-bump hot/cold spots
      HH = high surrounded by high (hotspot)
      LL = low surrounded by low (coldspot)
    """
    n = len(bumps)
    if n < 4:
        return None

    coords = np.array([[b.cx, b.cy] for b in bumps])
    values = np.array([getattr(b, metric, 0.0) for b in bumps])

    # Handle constant values
    val_std = np.std(values)
    if val_std < 1e-12:
        return MoranResult(
            layer=bumps[0].layer, I_global=0.0, z_score=0.0, p_value=1.0,
            pattern='Random', local_I=np.zeros(n),
            hotspot_labels=np.array(['NS'] * n))

    # Build k-NN spatial weight matrix
    k = min(k_neighbors, n - 1)
    tree = cKDTree(coords)
    dists, indices = tree.query(coords, k=k + 1)  # +1 because includes self

    # Row-standardized weight matrix (sparse via dict)
    z = (values - np.mean(values)) / val_std
    mean_val = np.mean(values)

    # Global Moran's I
    W_total = 0.0
    numerator = 0.0
    denominator = np.sum((values - mean_val) ** 2)

    # Local I array
    local_I = np.zeros(n)

    for i in range(n):
        neighbors = indices[i, 1:]  # exclude self
        neighbor_dists = dists[i, 1:]
        # Inverse distance weights, row-standardized
        valid = neighbor_dists > 0
        if not np.any(valid):
            continue
        weights = np.zeros(k)
        weights[valid] = 1.0 / neighbor_dists[valid]
        w_sum = np.sum(weights)
        if w_sum > 0:
            weights /= w_sum

        W_total += w_sum
        xi = values[i] - mean_val
        for j_idx in range(k):
            j = neighbors[j_idx]
            xj = values[j] - mean_val
            numerator += weights[j_idx] * xi * xj

        # Local I (LISA)
        lag_i = np.sum(weights * z[neighbors])
        local_I[i] = z[i] * lag_i

    if denominator > 0 and W_total > 0:
        I_global = (n / W_total) * (numerator / denominator)
    else:
        I_global = 0.0

    # Expected value and variance for significance test
    E_I = -1.0 / (n - 1)
    # Simplified variance (normal approximation)
    var_I = max(1.0 / (n * n), 1e-12)  # approximate
    z_score = (I_global - E_I) / np.sqrt(var_I) if var_I > 0 else 0.0
    p_value = 2.0 * (1.0 - norm.cdf(abs(z_score)))

    # Pattern classification
    if p_value < 0.05:
        pattern = 'Clustered' if I_global > E_I else 'Dispersed'
    else:
        pattern = 'Random'

    # LISA hotspot labels
    hotspot_labels = np.array(['NS'] * n)
    sig_threshold = 1.96  # 95% confidence
    for i in range(n):
        neighbors = indices[i, 1:]
        lag_mean = np.mean(z[neighbors])
        if abs(z[i]) > sig_threshold * 0.5:  # relaxed for small samples
            if z[i] > 0 and lag_mean > 0:
                hotspot_labels[i] = 'HH'  # hot spot
            elif z[i] < 0 and lag_mean < 0:
                hotspot_labels[i] = 'LL'  # cold spot
            elif z[i] > 0 and lag_mean < 0:
                hotspot_labels[i] = 'HL'  # spatial outlier
            elif z[i] < 0 and lag_mean > 0:
                hotspot_labels[i] = 'LH'  # spatial outlier

    return MoranResult(
        layer=bumps[0].layer, I_global=I_global, z_score=z_score,
        p_value=p_value, pattern=pattern, local_I=local_I,
        hotspot_labels=hotspot_labels)


# ─────────────────────── 2. Cross-Layer Vertical Propagation ───────────────────────

def compute_vertical_propagation(
    layer_bumps: Dict[str, List[BumpNode]],
    match_radius: float = 50.0,
    metric: str = 'void_ratio'
) -> Optional[PropagationResult]:
    """
    Compute correlation of defect metrics between bumps at corresponding
    XY positions across different layers.
    """
    layer_names = sorted(layer_bumps.keys())
    n_layers = len(layer_names)
    if n_layers < 2:
        return None

    corr_matrix = np.eye(n_layers)
    pairs = []

    for i in range(n_layers):
        bumps_i = layer_bumps[layer_names[i]]
        coords_i = np.array([[b.cx, b.cy] for b in bumps_i])
        vals_i = np.array([getattr(b, metric, 0.0) for b in bumps_i])
        tree_i = cKDTree(coords_i)

        for j in range(i + 1, n_layers):
            bumps_j = layer_bumps[layer_names[j]]
            coords_j = np.array([[b.cx, b.cy] for b in bumps_j])
            vals_j = np.array([getattr(b, metric, 0.0) for b in bumps_j])

            # Match bumps by nearest XY
            matched_i_vals = []
            matched_j_vals = []
            for idx_j, coord_j in enumerate(coords_j):
                dist, idx_i = tree_i.query(coord_j)
                if dist <= match_radius:
                    matched_i_vals.append(vals_i[idx_i])
                    matched_j_vals.append(vals_j[idx_j])

            r_val, p_val = 0.0, 1.0
            if len(matched_i_vals) >= 3:
                mi = np.array(matched_i_vals)
                mj = np.array(matched_j_vals)
                if np.std(mi) > 1e-12 and np.std(mj) > 1e-12:
                    r_val, p_val = pearsonr(mi, mj)

            corr_matrix[i, j] = r_val
            corr_matrix[j, i] = r_val

            direction = 'none'
            if abs(r_val) > 0.3 and p_val < 0.1:
                direction = 'bottom-up' if i < j else 'top-down'

            pairs.append({
                'from_layer': layer_names[i],
                'to_layer': layer_names[j],
                'r': r_val, 'p': p_val,
                'n_matched': len(matched_i_vals),
                'direction': direction
            })

    # Determine dominant direction
    up_scores = [p['r'] for p in pairs if p['direction'] == 'bottom-up']
    down_scores = [p['r'] for p in pairs if p['direction'] == 'top-down']
    if up_scores and np.mean(up_scores) > 0.3:
        dominant = 'bottom-up'
    elif down_scores and np.mean(down_scores) > 0.3:
        dominant = 'top-down'
    else:
        dominant = 'none'

    # Overall propagation score (mean of significant correlations)
    sig_corrs = [abs(p['r']) for p in pairs if p['p'] < 0.1]
    overall = float(np.mean(sig_corrs)) if sig_corrs else 0.0

    return PropagationResult(
        layer_names=layer_names, correlation_matrix=corr_matrix,
        propagation_pairs=pairs, dominant_direction=dominant,
        overall_score=overall)


# ─────────────────────── 3. Structural Integrity Index (SII) ───────────────────────

def compute_sii(
    bumps: List[BumpNode],
    vertical_scores: Optional[Dict[int, float]] = None,
    k_neighbors: int = 6,
    weights: Tuple[float, float, float] = (0.40, 0.35, 0.25),
    sii_risk_threshold: float = 0.4
) -> Optional[SIIResult]:
    """
    Compute Structural Integrity Index for each bump.

    SII = w1 × SelfScore + w2 × NeighborScore + w3 × VerticalScore

    SelfScore: combines SOH deviation and void ratio health
    NeighborScore: mean health of k-nearest neighbors
    VerticalScore: vertical alignment quality (from propagation analysis)
    """
    n = len(bumps)
    if n < 2:
        return None

    w1, w2, w3 = weights

    # Self Score: normalized health metric
    soh_vals = np.array([b.soh for b in bumps])
    vr_vals = np.array([b.void_ratio for b in bumps])

    # Normalize SOH: closer to mean = better
    soh_mean = np.mean(soh_vals) if len(soh_vals) > 0 else 1.0
    soh_std = np.std(soh_vals) if len(soh_vals) > 0 else 1.0
    if soh_std < 1e-12:
        soh_scores = np.ones(n)
    else:
        z_soh = np.abs(soh_vals - soh_mean) / soh_std
        soh_scores = np.clip(1.0 - z_soh / 3.0, 0.0, 1.0)  # 3σ → 0

    # Void ratio: lower = better
    vr_scores = np.clip(1.0 - vr_vals * 5.0, 0.0, 1.0)  # 20% ratio → 0

    self_scores = 0.5 * soh_scores + 0.5 * vr_scores

    # Neighbor Score (iterative smoothing, 2 passes)
    coords = np.array([[b.cx, b.cy] for b in bumps])
    k = min(k_neighbors, n - 1)
    tree = cKDTree(coords)
    _, nn_indices = tree.query(coords, k=k + 1)

    neighbor_scores = self_scores.copy()
    for _ in range(2):
        new_scores = np.zeros(n)
        for i in range(n):
            neighbors = nn_indices[i, 1:]
            new_scores[i] = np.mean(neighbor_scores[neighbors])
        neighbor_scores = new_scores

    # Vertical Score
    if vertical_scores:
        vert_scores = np.array([vertical_scores.get(b.idx, 1.0) for b in bumps])
    else:
        vert_scores = np.ones(n)

    # Combine
    sii = w1 * self_scores + w2 * neighbor_scores + w3 * vert_scores
    sii = np.clip(sii, 0.0, 1.0)

    return SIIResult(
        bump_indices=np.array([b.idx for b in bumps]),
        sii_values=sii,
        cx=np.array([b.cx for b in bumps]),
        cy=np.array([b.cy for b in bumps]),
        layer=bumps[0].layer,
        mean_sii=float(np.mean(sii)),
        std_sii=float(np.std(sii)),
        risk_count=int(np.sum(sii < sii_risk_threshold)))


# ─────────────────────── 4. Weak Zone Detection ───────────────────────

def detect_weak_zones(
    sii_result: SIIResult,
    threshold: float = 0.4,
    min_bumps: int = 2,
    merge_radius: float = 30.0
) -> List[WeakZone]:
    """
    Find contiguous regions of low-SII bumps.
    Uses distance-based clustering on bump centroids.
    """
    weak_mask = sii_result.sii_values < threshold
    if not np.any(weak_mask):
        return []

    weak_cx = sii_result.cx[weak_mask]
    weak_cy = sii_result.cy[weak_mask]
    weak_sii = sii_result.sii_values[weak_mask]
    n_weak = len(weak_cx)

    if n_weak < min_bumps:
        return []

    # Simple distance-based clustering
    coords = np.column_stack([weak_cx, weak_cy])
    labels = np.full(n_weak, -1, dtype=int)
    cluster_id = 0

    for i in range(n_weak):
        if labels[i] >= 0:
            continue
        # BFS from this point
        queue = [i]
        labels[i] = cluster_id
        while queue:
            current = queue.pop(0)
            for j in range(n_weak):
                if labels[j] >= 0:
                    continue
                dist = np.sqrt((coords[current, 0] - coords[j, 0]) ** 2 +
                               (coords[current, 1] - coords[j, 1]) ** 2)
                if dist <= merge_radius:
                    labels[j] = cluster_id
                    queue.append(j)
        cluster_id += 1

    # Build weak zones
    all_cx = sii_result.cx
    all_cy = sii_result.cy
    x_range = np.ptp(all_cx) if len(all_cx) > 1 else 1.0
    y_range = np.ptp(all_cy) if len(all_cy) > 1 else 1.0
    x_center = np.mean(all_cx)
    y_center = np.mean(all_cy)

    zones = []
    for cid in range(cluster_id):
        mask = labels == cid
        if np.sum(mask) < min_bumps:
            continue

        zx = weak_cx[mask]
        zy = weak_cy[mask]
        zsii = weak_sii[mask]
        mean_sii = float(np.mean(zsii))

        bbox = (float(np.min(zx)), float(np.min(zy)),
                float(np.max(zx)), float(np.max(zy)))
        centroid = (float(np.mean(zx)), float(np.mean(zy)))
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        # Severity
        if mean_sii < 0.2:
            severity = 'Critical'
        elif mean_sii < 0.35:
            severity = 'Warning'
        else:
            severity = 'Watch'

        # Zone type (position on die)
        rx = (centroid[0] - x_center) / (x_range / 2) if x_range > 0 else 0
        ry = (centroid[1] - y_center) / (y_range / 2) if y_range > 0 else 0
        if abs(rx) > 0.6 and abs(ry) > 0.6:
            zone_type = 'corner'
        elif abs(rx) > 0.6 or abs(ry) > 0.6:
            zone_type = 'edge'
        else:
            zone_type = 'center'

        zones.append(WeakZone(
            zone_id=cid, bbox=bbox, centroid=centroid,
            area=area, n_bumps=int(np.sum(mask)),
            mean_sii=mean_sii, severity=severity, zone_type=zone_type))

    zones.sort(key=lambda z: z.mean_sii)
    return zones


# ─────────────────────── 5. Process Fingerprint ───────────────────────

def compute_fingerprint(
    sample_name: str,
    layer_bumps: Dict[str, List[BumpNode]],
    moran_results: Dict[str, MoranResult],
    prop_result: Optional[PropagationResult],
    sii_results: Dict[str, SIIResult]
) -> ProcessFingerprint:
    """Build a process fingerprint vector for cross-sample comparison."""
    layer_names = sorted(layer_bumps.keys())

    moran_vals = [moran_results[ln].I_global if ln in moran_results else 0.0
                  for ln in layer_names]

    vert_vals = []
    if prop_result:
        for pair in prop_result.propagation_pairs:
            vert_vals.append(pair['r'])
    if not vert_vals:
        vert_vals = [0.0]

    sii_means = [sii_results[ln].mean_sii if ln in sii_results else 1.0
                 for ln in layer_names]
    mean_sii = float(np.mean(sii_means)) if sii_means else 1.0
    std_sii = float(np.std(sii_means)) if sii_means else 0.0

    densities = []
    for ln in layer_names:
        bumps = layer_bumps[ln]
        ng = sum(1 for b in bumps if b.judgment.upper() == 'NG')
        densities.append(ng / max(len(bumps), 1))

    direction = prop_result.dominant_direction if prop_result else 'none'
    dir_code = {'bottom-up': 1.0, 'top-down': -1.0, 'none': 0.0}.get(direction, 0.0)

    vector = np.array(moran_vals + vert_vals + [mean_sii, std_sii] +
                      densities + [dir_code])

    return ProcessFingerprint(
        sample_name=sample_name, moran_per_layer=moran_vals,
        vert_corr_per_pair=vert_vals, mean_sii=mean_sii, std_sii=std_sii,
        defect_density_per_layer=densities, dominant_direction=direction,
        vector=vector)


def fingerprint_similarity(fp1: ProcessFingerprint, fp2: ProcessFingerprint) -> float:
    """Cosine similarity between two process fingerprints."""
    v1, v2 = fp1.vector, fp2.vector
    min_len = min(len(v1), len(v2))
    v1, v2 = v1[:min_len], v2[:min_len]
    dot = np.dot(v1, v2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return float(dot / (n1 * n2))


# ─────────────────────── 6. Full Pipeline ───────────────────────

def run_full_analysis(
    raw_rows: List[dict],
    sample_name: str = "Sample",
    k_neighbors: int = 6,
    match_radius: float = 50.0,
    sii_weights: Tuple[float, float, float] = (0.40, 0.35, 0.25),
    sii_risk_threshold: float = 0.4,
    weak_zone_threshold: float = 0.4,
    metric: str = 'void_ratio'
) -> dict:
    """
    Run the complete 3D-SIM analysis pipeline.

    Returns dict with:
      'layer_bumps', 'moran', 'propagation', 'sii', 'weak_zones', 'fingerprint'
    """
    layer_bumps = parse_bumps_from_rows(raw_rows)

    # 1. Moran's I per layer
    moran_results = {}
    for ln, bumps in layer_bumps.items():
        result = compute_moran_i(bumps, metric=metric, k_neighbors=k_neighbors)
        if result:
            moran_results[ln] = result

    # 2. Cross-layer propagation
    prop_result = compute_vertical_propagation(
        layer_bumps, match_radius=match_radius, metric=metric)

    # 3. SII per layer
    sii_results = {}
    for ln, bumps in layer_bumps.items():
        sii = compute_sii(bumps, k_neighbors=k_neighbors, weights=sii_weights,
                          sii_risk_threshold=sii_risk_threshold)
        if sii:
            sii_results[ln] = sii

    # 4. Weak zones per layer
    weak_zones = {}
    for ln, sii in sii_results.items():
        zones = detect_weak_zones(sii, threshold=weak_zone_threshold)
        if zones:
            weak_zones[ln] = zones

    # 5. Process fingerprint
    fingerprint = compute_fingerprint(
        sample_name, layer_bumps, moran_results, prop_result, sii_results)

    return {
        'layer_bumps': layer_bumps,
        'moran': moran_results,
        'propagation': prop_result,
        'sii': sii_results,
        'weak_zones': weak_zones,
        'fingerprint': fingerprint
    }


# ─────────────────────── 7. LOO Variance Decomposition ───────────────────────

@dataclass
class LOOBumpResult:
    """LOO result for a single bump."""
    bump_idx: int
    grid_label: str  # e.g. "R2C3"
    cx: float
    cy: float
    value: float  # raw metric value
    deviation: float  # value - mean
    sigma_full: float  # σ with all bumps
    sigma_without: float  # σ without this bump
    impact_pct: float  # (σ_full - σ_without) / σ_full × 100
    contribution_pct: float  # (xi - mean)² / Σ(xj - mean)² × 100


@dataclass
class LOOLayerResult:
    """LOO analysis for one layer of one sample."""
    layer: str
    sample_name: str
    n_bumps: int
    sigma_full: float
    mean_full: float
    pct_tol: float  # %Tol = σ/μ × 100
    bumps: List[LOOBumpResult]
    # What-if: removing top-K outliers
    whatif_1: float  # σ after removing top-1 outlier
    whatif_2: float  # σ after removing top-2 outliers
    whatif_3: float  # σ after removing top-3 outliers


def compute_loo_variance(
    raw_rows: List[dict],
    metric_key: str = 'soh',
    grid_row_col: bool = True
) -> Dict[str, LOOLayerResult]:
    """
    Leave-One-Out Variance Decomposition.

    For each bump in each layer, compute:
    1. Variance contribution: how much this bump contributes to total variance
    2. LOO impact: how σ changes when this bump is removed
    3. What-if: σ after removing top 1/2/3 outliers

    Parameters
    ----------
    raw_rows : raw CSV rows from sample
    metric_key : 'soh', 'void_ratio', or 'bump_volume'
    """
    layer_bumps = parse_bumps_from_rows(raw_rows)
    results = {}

    for layer_name, bumps in layer_bumps.items():
        n = len(bumps)
        if n < 3:
            continue

        # Extract metric values
        values = np.array([getattr(b, metric_key, 0.0) for b in bumps])
        mean_val = np.mean(values)
        sigma_full = float(np.std(values, ddof=1)) if n > 1 else 0.0
        pct_tol = (sigma_full / abs(mean_val) * 100.0) if abs(mean_val) > 1e-12 else 0.0

        # Variance contributions
        deviations = values - mean_val
        sq_devs = deviations ** 2
        total_sq_dev = np.sum(sq_devs)
        contributions = (sq_devs / total_sq_dev * 100.0) if total_sq_dev > 0 else np.zeros(n)

        # LOO impact for each bump
        loo_bumps = []
        for i in range(n):
            remaining = np.delete(values, i)
            sigma_without = float(np.std(remaining, ddof=1)) if len(remaining) > 1 else 0.0
            impact = ((sigma_full - sigma_without) / sigma_full * 100.0) if sigma_full > 1e-12 else 0.0

            # Grid label (R,C) based on sorted position
            grid_label = f"#{i+1}"

            loo_bumps.append(LOOBumpResult(
                bump_idx=bumps[i].idx,
                grid_label=grid_label,
                cx=bumps[i].cx,
                cy=bumps[i].cy,
                value=float(values[i]),
                deviation=float(deviations[i]),
                sigma_full=sigma_full,
                sigma_without=sigma_without,
                impact_pct=float(impact),
                contribution_pct=float(contributions[i])
            ))

        # Sort by impact (descending) for ranking
        loo_bumps.sort(key=lambda b: b.impact_pct, reverse=True)

        # Assign grid labels by spatial position (row, col)
        if grid_row_col:
            _assign_grid_labels(bumps, loo_bumps)

        # What-if: remove top-K outliers
        sorted_by_contrib = sorted(range(n), key=lambda i: contributions[i], reverse=True)
        whatif_sigmas = []
        for k in [1, 2, 3]:
            if k >= n:
                whatif_sigmas.append(0.0)
                continue
            remove_indices = set(sorted_by_contrib[:k])
            remaining = [values[i] for i in range(n) if i not in remove_indices]
            whatif_sigmas.append(float(np.std(remaining, ddof=1)) if len(remaining) > 1 else 0.0)

        results[layer_name] = LOOLayerResult(
            layer=layer_name,
            sample_name="",
            n_bumps=n,
            sigma_full=sigma_full,
            mean_full=float(mean_val),
            pct_tol=pct_tol,
            bumps=loo_bumps,
            whatif_1=whatif_sigmas[0],
            whatif_2=whatif_sigmas[1],
            whatif_3=whatif_sigmas[2]
        )

    return results


def _assign_grid_labels(bumps: List[BumpNode], loo_bumps: List[LOOBumpResult]):
    """Assign R(row)C(col) grid labels based on spatial Y/X sorting."""
    # Build idx→loo_bump map
    idx_map = {lb.bump_idx: lb for lb in loo_bumps}

    # Sort bumps by Y then X to determine grid position
    sorted_bumps = sorted(bumps, key=lambda b: (b.cy, b.cx))
    if not sorted_bumps:
        return

    # Group into rows using adaptive Y tolerance
    rows_list = []
    current_row = [sorted_bumps[0]]
    h_mean = np.mean([b.soh for b in bumps]) if bumps else 10.0
    tolerance = max(h_mean * 0.7, 5.0)

    for b in sorted_bumps[1:]:
        if abs(b.cy - current_row[-1].cy) < tolerance:
            current_row.append(b)
        else:
            rows_list.append(current_row)
            current_row = [b]
    rows_list.append(current_row)

    # Assign R,C labels
    for r_idx, row_bumps in enumerate(rows_list):
        row_bumps.sort(key=lambda b: b.cx)
        for c_idx, b in enumerate(row_bumps):
            if b.idx in idx_map:
                idx_map[b.idx].grid_label = f"R{r_idx+1}C{c_idx+1}"


def compute_cross_sample_loo(
    samples_raw_rows: List[Tuple[str, List[dict]]],
    metric_key: str = 'soh'
) -> Dict[str, dict]:
    """
    Cross-sample LOO: for each layer, find which SAMPLE is the outlier.

    For each layer across all samples, compute mean metric per sample,
    then LOO on sample-level means to find which sample drives cross-sample σ.
    """
    # Collect per-layer, per-sample means
    layer_sample_means: Dict[str, List[Tuple[str, float]]] = {}

    for sample_name, raw_rows in samples_raw_rows:
        layer_bumps = parse_bumps_from_rows(raw_rows)
        for ln, bumps in layer_bumps.items():
            vals = np.array([getattr(b, metric_key, 0.0) for b in bumps])
            if len(vals) > 0:
                layer_sample_means.setdefault(ln, []).append(
                    (sample_name, float(np.mean(vals))))

    results = {}
    for ln, sample_means in layer_sample_means.items():
        if len(sample_means) < 3:
            continue
        names = [sm[0] for sm in sample_means]
        vals = np.array([sm[1] for sm in sample_means])
        sigma_full = float(np.std(vals, ddof=1))
        mean_full = float(np.mean(vals))

        sample_impacts = []
        for i in range(len(vals)):
            remaining = np.delete(vals, i)
            sigma_without = float(np.std(remaining, ddof=1)) if len(remaining) > 1 else 0.0
            impact = ((sigma_full - sigma_without) / sigma_full * 100.0
                      ) if sigma_full > 1e-12 else 0.0
            sample_impacts.append({
                'sample': names[i],
                'value': float(vals[i]),
                'deviation': float(vals[i] - mean_full),
                'sigma_without': sigma_without,
                'impact_pct': impact
            })

        sample_impacts.sort(key=lambda x: x['impact_pct'], reverse=True)
        results[ln] = {
            'sigma_full': sigma_full,
            'mean_full': mean_full,
            'pct_tol': (sigma_full / abs(mean_full) * 100.0) if abs(mean_full) > 1e-12 else 0.0,
            'sample_impacts': sample_impacts
        }

    return results
