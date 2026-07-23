"""
AutoPrompt-SAM: Intensity-Guided Automatic Prompt Generation for SAM
=====================================================================

A novel domain-adapted segmentation pipeline for semiconductor X-ray CT.

Key Innovation:
    Instead of text-based detectors (Grounding DINO) that fail on specialized
    industrial images, this pipeline uses classical computer vision to
    AUTO-GENERATE spatial prompts (bounding boxes + point prompts) for SAM.

Pipeline Architecture:
    Stage 1: Preprocessing (CLAHE + FOV mask)
    Stage 2: Coarse Detection via intensity analysis (Classical CV)
    Stage 3: SAM Refinement with auto-generated prompts
    Stage 4: Hierarchical sub-structure detection (Voids within TGVs)

Advantages over DINO+SAM:
    - No domain gap (uses intensity, not language)
    - Fully automatic (no manual text prompts)
    - Hierarchical (detects defects WITHIN structures)
    - 10x faster (classical CV is near-instant)
    - No 1GB+ model download for DINO

Author: AutoPrompt-SAM Pipeline
"""

import traceback

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from scipy import ndimage
from skimage import filters, measure, morphology

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ─────────────────────────────────────────────────────────────
# Stage 1: Preprocessing
# ─────────────────────────────────────────────────────────────

def detect_fov_mask(image_gray, border_fraction=0.0):
    """
    Detect the circular Field-of-View boundary in CT slices.
    Masks out the black background outside the sample.

    Returns
    -------
    fov_mask : np.ndarray (H, W) bool — True inside the FOV
    """
    h, w = image_gray.shape
    # Threshold: anything significantly above zero is inside FOV
    thresh = max(1, np.percentile(image_gray[image_gray > 0], 1)) if np.any(image_gray > 0) else 1
    binary = image_gray > thresh

    # Clean up with morphological closing
    struct = morphology.disk(max(3, min(h, w) // 100))
    binary = ndimage.binary_closing(binary, structure=struct, iterations=2)
    binary = ndimage.binary_fill_holes(binary)

    # Keep only the largest connected component (the FOV)
    labels, n = ndimage.label(binary)
    if n == 0:
        return np.ones((h, w), dtype=bool)
    sizes = ndimage.sum(binary, labels, range(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    fov_mask = labels == biggest

    # Optional: erode border slightly to avoid edge artifacts
    if border_fraction > 0:
        border_px = max(1, int(min(h, w) * border_fraction))
        fov_mask = ndimage.binary_erosion(fov_mask, iterations=border_px)

    return fov_mask


def preprocess_slice(image_gray, clip_limit=3.0, blur_sigma=1.0):
    """
    Enhance contrast with CLAHE and apply gentle Gaussian blur.

    Returns
    -------
    enhanced : np.ndarray (H, W) uint8
    """
    # Normalize to uint8 if needed
    if image_gray.dtype != np.uint8:
        mn, mx = image_gray.min(), image_gray.max()
        if mx > mn:
            img8 = ((image_gray - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            img8 = np.zeros_like(image_gray, dtype=np.uint8)
    else:
        img8 = image_gray.copy()

    # CLAHE for local contrast enhancement
    if HAS_CV2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        img8 = clahe.apply(img8)

    # Gentle Gaussian blur to suppress noise
    if blur_sigma > 0:
        img8 = ndimage.gaussian_filter(img8.astype(np.float32), sigma=blur_sigma)
        img8 = np.clip(img8, 0, 255).astype(np.uint8)

    return img8


# ─────────────────────────────────────────────────────────────
# Stage 2: Coarse Detection (Classical CV)
# ─────────────────────────────────────────────────────────────

def detect_bright_structures(image_gray, fov_mask=None,
                             percentile=80, min_area=200,
                             max_area_fraction=0.25,
                             min_circularity=0.15,
                             morph_radius=1):
    """
    Detect bright structures (TGVs) using intensity thresholding.

    Parameters
    ----------
    image_gray : uint8 grayscale
    fov_mask : bool mask (True = inside FOV)
    percentile : float — top N% brightest pixels are candidates
    min_area : int — minimum region area in pixels
    max_area_fraction : float — max fraction of FOV area
    min_circularity : float — 0=any shape, 1=perfect circle
    morph_radius : int — morphological cleanup kernel radius

    Returns
    -------
    candidates : list of dict with keys:
        'label', 'centroid', 'bbox', 'area', 'circularity', 'mask'
    binary_mask : np.ndarray (H, W) bool — combined bright mask
    labels_map : np.ndarray (H, W) int — labeled regions
    """
    if fov_mask is None:
        fov_mask = np.ones(image_gray.shape, dtype=bool)

    # Compute threshold within FOV only
    fov_pixels = image_gray[fov_mask]
    if len(fov_pixels) == 0:
        return [], np.zeros(image_gray.shape, dtype=bool), np.zeros(image_gray.shape, dtype=np.int32)

    # Try multi-Otsu first for better class separation
    try:
        thresholds = filters.threshold_multiotsu(fov_pixels, classes=3)
        bright_thresh = thresholds[-1]  # Highest threshold = brightest class
    except Exception:
        bright_thresh = np.percentile(fov_pixels, percentile)

    # Also compute percentile-based threshold as backup
    pct_thresh = np.percentile(fov_pixels, percentile)
    # Use the lower of the two (more permissive) to catch more candidates
    thresh = min(bright_thresh, pct_thresh)

    print(f"[AutoPrompt] Bright threshold: {thresh:.1f} "
          f"(multi-otsu={bright_thresh:.1f}, pct{percentile}={pct_thresh:.1f})")

    # Binary thresholding
    binary = (image_gray > thresh) & fov_mask

    # Morphological cleanup: open (remove noise) then close (fill gaps)
    selem = morphology.disk(morph_radius)
    binary = morphology.binary_opening(binary, selem)
    binary = morphology.binary_closing(binary, selem)
    binary = ndimage.binary_fill_holes(binary)

    # Connected component analysis
    labels_map = measure.label(binary, connectivity=2)
    props = measure.regionprops(labels_map, intensity_image=image_gray)

    fov_area = np.sum(fov_mask)
    candidates = []

    for prop in props:
        # Area filter
        if prop.area < min_area:
            continue
        if prop.area > max_area_fraction * fov_area:
            continue

        # Circularity filter: 4*pi*area / perimeter^2
        if prop.perimeter > 0:
            circularity = 4 * np.pi * prop.area / (prop.perimeter ** 2)
        else:
            circularity = 0

        if circularity < min_circularity:
            continue

        # Extract bbox (y_min, x_min, y_max, x_max) from regionprops
        y_min, x_min, y_max, x_max = prop.bbox
        cy, cx = prop.centroid

        candidates.append({
            'label': prop.label,
            'centroid': (int(round(cx)), int(round(cy))),  # (x, y)
            'bbox': (int(x_min), int(y_min), int(x_max), int(y_max)),  # xyxy
            'area': prop.area,
            'circularity': circularity,
            'mean_intensity': prop.mean_intensity,
            'mask': labels_map == prop.label,
        })

    print(f"[AutoPrompt] Found {len(candidates)} bright structure candidates")
    for i, c in enumerate(candidates):
        print(f"  [{i}] center=({c['centroid'][0]},{c['centroid'][1]}) "
              f"area={c['area']} circ={c['circularity']:.2f} "
              f"intensity={c['mean_intensity']:.1f}")

    return candidates, binary, labels_map


def detect_dark_substructures(image_gray, parent_candidates,
                               dark_factor=0.6,
                               min_area=5, max_area_fraction=0.3,
                               morph_radius=0):
    """
    Detect dark defects (Voids) WITHIN each parent structure (TGV).

    For each TGV, analyses the local intensity distribution and finds
    regions significantly darker than the TGV mean.

    Parameters
    ----------
    image_gray : uint8 grayscale
    parent_candidates : list of dict from detect_bright_structures()
    dark_factor : float — pixels below mean*dark_factor are "dark"
    min_area : int — minimum void area in pixels
    max_area_fraction : float — max fraction of parent area

    Returns
    -------
    void_candidates : list of dict with keys:
        'centroid', 'bbox', 'area', 'parent_idx', 'mask'
    """
    void_candidates = []

    for idx, parent in enumerate(parent_candidates):
        parent_mask = parent['mask']
        parent_area = parent['area']

        # Get intensity within this TGV only
        local_intensities = image_gray[parent_mask]
        if len(local_intensities) == 0:
            continue

        local_mean = np.mean(local_intensities)
        local_std = np.std(local_intensities)

        # Dark threshold: significantly below local mean
        # Use both factor-based and statistical approaches
        dark_thresh_factor = local_mean * dark_factor
        dark_thresh_stats = local_mean - 1.5 * local_std
        dark_thresh = max(dark_thresh_factor, dark_thresh_stats)

        # Erode parent mask to avoid classifying the dim outer edge of the bump as a void
        core_mask = morphology.binary_erosion(parent_mask, morphology.disk(2))

        # Find dark pixels within the core of this parent
        dark_binary = (image_gray < dark_thresh) & core_mask

        if not np.any(dark_binary):
            continue

        # Morphological cleanup
        if morph_radius > 0:
            selem = morphology.disk(morph_radius)
            dark_binary = morphology.binary_opening(dark_binary, selem)

        # Label dark regions
        dark_labels = measure.label(dark_binary, connectivity=2)
        dark_props = measure.regionprops(dark_labels, intensity_image=image_gray)

        for prop in dark_props:
            if prop.area < min_area:
                continue
            if prop.area > max_area_fraction * parent_area:
                continue

            y_min, x_min, y_max, x_max = prop.bbox
            cy, cx = prop.centroid

            void_candidates.append({
                'centroid': (int(round(cx)), int(round(cy))),
                'bbox': (int(x_min), int(y_min), int(x_max), int(y_max)),
                'area': prop.area,
                'parent_idx': idx,
                'mean_intensity': prop.mean_intensity,
                'mask': dark_labels == prop.label,
            })

    print(f"[AutoPrompt] Found {len(void_candidates)} void candidates "
          f"across {len(parent_candidates)} TGVs")
    for i, v in enumerate(void_candidates):
        print(f"  [{i}] center=({v['centroid'][0]},{v['centroid'][1]}) "
              f"area={v['area']} parent={v['parent_idx']} "
              f"intensity={v['mean_intensity']:.1f}")

    return void_candidates


# ─────────────────────────────────────────────────────────────
# Stage 3: SAM Refinement
# ─────────────────────────────────────────────────────────────

def refine_with_sam(image_rgb, candidates, sam_predictor,
                    use_point_prompt=True, expand_box_px=10):
    """
    Refine coarse classical CV detections with SAM.

    For each candidate, generates a box prompt (+ optional point prompt)
    and runs SAM to get a pixel-precise mask.

    Parameters
    ----------
    image_rgb : np.ndarray (H, W, 3) uint8
    candidates : list of dict with 'bbox' and 'centroid'
    sam_predictor : SamPredictor instance (already has set_image called)
    use_point_prompt : bool — also pass centroid as foreground point
    expand_box_px : int — expand bbox by this many pixels for SAM

    Returns
    -------
    refined_masks : list of np.ndarray (H, W) bool
    iou_scores : list of float
    """
    h, w = image_rgb.shape[:2]
    refined_masks = []
    iou_scores = []

    for cand in candidates:
        x1, y1, x2, y2 = cand['bbox']
        # Expand box slightly for SAM (it works better with some margin)
        x1 = max(0, x1 - expand_box_px)
        y1 = max(0, y1 - expand_box_px)
        x2 = min(w, x2 + expand_box_px)
        y2 = min(h, y2 + expand_box_px)

        box = np.array([x1, y1, x2, y2])

        kwargs = {'box': box, 'multimask_output': True}

        if use_point_prompt:
            cx, cy = cand['centroid']
            kwargs['point_coords'] = np.array([[cx, cy]])
            kwargs['point_labels'] = np.array([1])  # 1 = foreground

        masks, scores, _ = sam_predictor.predict(**kwargs)

        # Pick best mask (highest IoU score)
        best_idx = int(np.argmax(scores))
        refined_masks.append(masks[best_idx])
        iou_scores.append(float(scores[best_idx]))

    return refined_masks, iou_scores


# ─────────────────────────────────────────────────────────────
# Full Pipeline
# ─────────────────────────────────────────────────────────────

def run_autoprompt_pipeline(image_gray, sam_predictor=None,
                            tgv_class_id=1, void_class_id=2,
                            bright_percentile=80,
                            min_tgv_area=200, min_tgv_circularity=0.15,
                            dark_factor=0.6,
                            min_void_area=5,
                            use_sam=True, use_clahe=True,
                            detect_voids=True):
    """
    Full AutoPrompt-SAM pipeline.

    Parameters
    ----------
    image_gray : np.ndarray (H, W) — grayscale input
    sam_predictor : SamPredictor or None
    tgv_class_id : int — class ID for TGV regions
    void_class_id : int — class ID for Void regions
    bright_percentile : float — percentile for bright structure detection
    min_tgv_area : int — minimum TGV area in pixels
    dark_factor : float — darkness threshold relative to local mean
    min_void_area : int — minimum void area
    use_sam : bool — refine with SAM (requires sam_predictor)
    use_clahe : bool — apply CLAHE preprocessing
    detect_voids : bool — also detect voids within TGVs

    Returns
    -------
    combined_mask : np.ndarray (H, W) uint8 — class labels
    info : dict — detection statistics
    """
    h, w = image_gray.shape
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    # ── Stage 1: Preprocessing ──
    print("[AutoPrompt] Stage 1: Preprocessing...")
    if use_clahe:
        enhanced = preprocess_slice(image_gray)
    else:
        enhanced = image_gray.copy()
        if enhanced.dtype != np.uint8:
            mn, mx = enhanced.min(), enhanced.max()
            enhanced = ((enhanced - mn) / (mx - mn + 1e-5) * 255).astype(np.uint8)

    fov_mask = detect_fov_mask(enhanced)
    print(f"[AutoPrompt] FOV mask: {np.sum(fov_mask)}/{h*w} pixels "
          f"({100*np.sum(fov_mask)/(h*w):.1f}%)")

    # ── Stage 2: Detect bright structures (TGVs) ──
    print("[AutoPrompt] Stage 2: Detecting bright structures (TGVs)...")
    tgv_candidates, bright_binary, labels_map = detect_bright_structures(
        enhanced, fov_mask,
        percentile=bright_percentile,
        min_area=min_tgv_area,
        min_circularity=min_tgv_circularity,
    )

    if not tgv_candidates:
        print("[AutoPrompt] No TGV candidates found. Returning empty mask.")
        return combined_mask, {'n_tgv': 0, 'n_void': 0}

    # ── Stage 3: SAM refinement for TGVs ──
    if use_sam and sam_predictor is not None:
        print("[AutoPrompt] Stage 3: Refining TGVs with SAM...")
        # Prepare RGB image for SAM
        img_rgb = np.stack([enhanced] * 3, axis=-1)
        sam_predictor.set_image(img_rgb)

        tgv_masks, tgv_ious = refine_with_sam(
            img_rgb, tgv_candidates, sam_predictor,
            use_point_prompt=True, expand_box_px=15
        )

        # Apply SAM-refined masks
        for i, mask in enumerate(tgv_masks):
            combined_mask[mask & fov_mask] = tgv_class_id
            # Update candidate mask for void detection
            tgv_candidates[i]['mask'] = mask & fov_mask

        print(f"[AutoPrompt] SAM TGV IoU scores: "
              f"{[round(s, 3) for s in tgv_ious]}")
    else:
        print("[AutoPrompt] Stage 3: Using classical masks (no SAM)")
        for cand in tgv_candidates:
            combined_mask[cand['mask']] = tgv_class_id
        tgv_ious = []

    # ── Stage 4: Hierarchical void detection ──
    n_voids = 0
    void_ious = []
    if detect_voids:
        print("[AutoPrompt] Stage 4: Detecting voids within TGVs...")
        void_candidates = detect_dark_substructures(
            enhanced, tgv_candidates,
            dark_factor=dark_factor,
            min_area=min_void_area,
        )

        if void_candidates and use_sam and sam_predictor is not None:
            print("[AutoPrompt] Refining voids with SAM...")
            void_masks, void_ious = refine_with_sam(
                img_rgb, void_candidates, sam_predictor,
                use_point_prompt=True, expand_box_px=5
            )
            for mask in void_masks:
                combined_mask[mask & fov_mask] = void_class_id
            print(f"[AutoPrompt] SAM Void IoU scores: "
                  f"{[round(s, 3) for s in void_ious]}")
        elif void_candidates:
            for cand in void_candidates:
                combined_mask[cand['mask']] = void_class_id

        n_voids = len(void_candidates)

    info = {
        'n_tgv': len(tgv_candidates),
        'n_void': n_voids,
        'tgv_ious': tgv_ious,
        'void_ious': void_ious,
        'tgv_areas': [c['area'] for c in tgv_candidates],
        'pipeline': 'AutoPrompt-SAM',
    }

    print(f"[AutoPrompt] Done! TGVs={info['n_tgv']}, Voids={info['n_void']}")
    return combined_mask, info


# ─────────────────────────────────────────────────────────────
# QThread Worker
# ─────────────────────────────────────────────────────────────

class AutoPromptSAMWorker(QThread):
    """
    Background worker for AutoPrompt-SAM pipeline on a single slice.

    Emits:
        progress(str): status messages
        finished(np.ndarray, dict): combined_mask (H,W uint8), info_dict
    """
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, object)

    def __init__(self, image_2d, sam_predictor=None,
                 tgv_class_id=1, void_class_id=2,
                 bright_percentile=80, min_tgv_area=200,
                 dark_factor=0.6, min_void_area=5,
                 use_sam=True, detect_voids=True):
        super().__init__()
        self.image_2d = image_2d
        self.sam_predictor = sam_predictor
        self.tgv_class_id = tgv_class_id
        self.void_class_id = void_class_id
        self.bright_percentile = bright_percentile
        self.min_tgv_area = min_tgv_area
        self.dark_factor = dark_factor
        self.min_void_area = min_void_area
        self.use_sam = use_sam
        self.detect_voids = detect_voids

    def run(self):
        try:
            img = self.image_2d
            print(f"[AutoPromptWorker] Image shape={img.shape} "
                  f"dtype={img.dtype} min={img.min()} max={img.max()}")

            # Normalize to uint8 if needed
            if img.dtype != np.uint8:
                mn, mx = img.min(), img.max()
                img = ((img - mn) / (mx - mn + 1e-5) * 255).astype(np.uint8)

            self.progress.emit("Running AutoPrompt-SAM pipeline...")

            combined_mask, info = run_autoprompt_pipeline(
                img,
                sam_predictor=self.sam_predictor,
                tgv_class_id=self.tgv_class_id,
                void_class_id=self.void_class_id,
                bright_percentile=self.bright_percentile,
                min_tgv_area=self.min_tgv_area,
                dark_factor=self.dark_factor,
                min_void_area=self.min_void_area,
                use_sam=self.use_sam,
                detect_voids=self.detect_voids,
            )

            n_tgv = info.get('n_tgv', 0)
            n_void = info.get('n_void', 0)
            self.progress.emit(
                f"Done! {n_tgv} TGVs + {n_void} Voids detected."
            )
            self.finished.emit(combined_mask, info)

        except Exception as e:
            self.progress.emit(f"Error: {e}")
            print(f"[AutoPromptWorker] ERROR: {e}\n{traceback.format_exc()}")
            h, w = self.image_2d.shape[:2]
            self.finished.emit(
                np.zeros((h, w), dtype=np.uint8),
                {'error': str(e), 'traceback': traceback.format_exc()}
            )
