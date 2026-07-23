"""
Active Learning Engine — Uncertainty-Guided Smart Slice Selection.

Patent: "Active Learning System for 3D Semiconductor Image Segmentation
with Uncertainty-Guided Slice Selection and Human-in-the-Loop Annotation
Optimization"

Key innovations:
    1. MC-Dropout uncertainty quantification per pixel → per slice
    2. CoreSet diversity selection to avoid redundant labeling
    3. Annotation efficiency tracking (accuracy vs. #labels curve)
    4. Integrated "label-these-next" recommendation engine

No external dependencies beyond PyTorch + NumPy + SciPy.

Author: INNO3D Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from PyQt5.QtCore import QThread, pyqtSignal

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SliceUncertainty:
    """Uncertainty information for a single slice."""
    slice_index: int
    mean_entropy: float             # Mean per-pixel entropy (higher = more uncertain)
    max_entropy: float              # Max per-pixel entropy
    fg_entropy: float               # Entropy only in foreground regions
    prediction_variance: float      # Variance across MC passes
    diversity_score: float = 0.0    # CoreSet distance to already-labeled set
    combined_score: float = 0.0     # Final selection score
    is_labeled: bool = False        # Already has ground truth


@dataclass
class ActiveLearningState:
    """Persistent state for the active learning session."""
    labeled_slices: list[int] = field(default_factory=list)
    recommended_slices: list[int] = field(default_factory=list)
    all_uncertainties: list[SliceUncertainty] = field(default_factory=list)
    entropy_map: np.ndarray | None = None  # (D, H, W) per-pixel entropy
    accuracy_history: list[tuple[int, float]] = field(default_factory=list)
    total_slices: int = 0
    mc_passes: int = 0
    status: str = "idle"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MC-Dropout Uncertainty Quantification
# ═══════════════════════════════════════════════════════════════════════════════

class MCDropoutEstimator:
    """
    Monte Carlo Dropout for per-pixel uncertainty estimation.
    
    Runs T forward passes with dropout enabled, then computes:
    - Predictive entropy: H[p(y|x)] = -Σ p log p
    - Mutual information: epistemic uncertainty
    - Prediction variance: how much predictions change across passes
    """

    def __init__(self, n_passes: int = 5):
        self.n_passes = n_passes

    def _enable_mc_dropout(self, model):
        """Enable dropout during inference for MC sampling."""
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()

    def estimate_volume(
        self,
        model: 'nn.Module',
        volume: np.ndarray,
        device: 'torch.device',
        in_channels: int = 1,
        adj: int = 3,
        dim_mode: str = '2D',
        z_from: int = 0,
        z_to: int = -1,
        progress_callback=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run MC-Dropout estimation on a volume.
        
        Returns:
            entropy_map: (D, H, W) per-pixel entropy
            variance_map: (D, H, W) per-pixel prediction variance
            mean_prediction: (D, H, W) mean predicted class
        """
        D, H, W = volume.shape[:3]
        if z_to < 0:
            z_to = D - 1

        entropy_map = np.zeros((D, H, W), dtype=np.float32)
        variance_map = np.zeros((D, H, W), dtype=np.float32)
        mean_prediction = np.zeros((D, H, W), dtype=np.uint8)

        model.eval()
        self._enable_mc_dropout(model)

        num_slices = z_to - z_from + 1

        with torch.no_grad():
            for idx, z in enumerate(range(z_from, z_to + 1)):
                # Prepare input
                if dim_mode == '2.5D' and in_channels > 1:
                    pad = in_channels // 2
                    indices = np.clip(
                        np.arange(z - pad, z + pad + 1), 0, D - 1
                    )
                    img = volume[indices].astype(np.float32)
                    img = (img - img.min()) / (img.max() - img.min() + 1e-5)
                    img_t = torch.from_numpy(img).unsqueeze(0).to(device)
                else:
                    img = volume[z].astype(np.float32)
                    img = (img - img.min()) / (img.max() - img.min() + 1e-5)
                    img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

                # MC forward passes
                all_probs = []
                for _ in range(self.n_passes):
                    logits = model(img_t)
                    probs = F.softmax(logits, dim=1)  # (1, C, H, W)
                    all_probs.append(probs.cpu().numpy())

                # Stack: (T, C, H, W)
                stacked = np.concatenate(all_probs, axis=0)
                T, C = stacked.shape[0], stacked.shape[1]

                # Mean prediction
                mean_probs = stacked.mean(axis=0)  # (C, H, W)
                pred = mean_probs.argmax(axis=0)    # (H, W)
                mean_prediction[z] = pred.astype(np.uint8)

                # Predictive entropy: H = -Σ p̄ log p̄
                entropy = -np.sum(
                    mean_probs * np.log(mean_probs + 1e-10), axis=0
                )
                entropy_map[z] = entropy

                # Prediction variance: mean variance of class probabilities
                var = stacked.var(axis=0).mean(axis=0)  # (H, W)
                variance_map[z] = var

                if progress_callback:
                    progress_callback(idx + 1, num_slices)

        # Restore eval mode
        model.eval()

        return entropy_map, variance_map, mean_prediction


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CoreSet Diversity Selection
# ═══════════════════════════════════════════════════════════════════════════════

class CoreSetSelector:
    """
    Diversity-based slice selection using CoreSet approach.
    
    Ensures selected slices are maximally different from each other
    and from already-labeled slices, avoiding redundant annotations.
    """

    @staticmethod
    def compute_slice_features(
        entropy_map: np.ndarray,
        volume: np.ndarray,
        variance_map: np.ndarray = None,
    ) -> np.ndarray:
        """
        Compute a feature vector per slice for diversity selection.
        
        Features per slice (11D):
            - mean, std, max entropy
            - entropy histogram (4 bins)
            - mean, std pixel intensity
            - foreground ratio (entropy > threshold)
            - mean variance (if available)
        """
        D = entropy_map.shape[0]
        features = np.zeros((D, 11), dtype=np.float32)

        for z in range(D):
            e = entropy_map[z]
            v = volume[z].astype(np.float32)

            features[z, 0] = e.mean()
            features[z, 1] = e.std()
            features[z, 2] = e.max()

            # Entropy histogram (4 bins)
            hist, _ = np.histogram(e.ravel(), bins=4, range=(0, e.max() + 1e-8))
            hist = hist.astype(np.float32)
            hist /= max(1, hist.sum())
            features[z, 3:7] = hist

            features[z, 7] = v.mean() / 255.0
            features[z, 8] = v.std() / 255.0

            # Foreground ratio: pixels where entropy > 0.1
            features[z, 9] = (e > 0.1).sum() / max(1, e.size)

            if variance_map is not None:
                features[z, 10] = variance_map[z].mean()

        # Normalize
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-8
        features = (features - mean) / std

        return features

    @staticmethod
    def select_diverse(
        features: np.ndarray,
        n_select: int,
        labeled_indices: list[int] = None,
        candidate_indices: list[int] = None,
    ) -> list[int]:
        """
        Greedy CoreSet selection: pick slices that maximize minimum
        distance to the already-selected set.
        
        Args:
            features: (D, F) feature matrix
            n_select: number of slices to select
            labeled_indices: already-labeled slice indices
            candidate_indices: allowed candidate indices (None = all)
            
        Returns:
            List of selected slice indices
        """
        from scipy.spatial.distance import cdist

        D = features.shape[0]
        if candidate_indices is None:
            candidate_indices = list(range(D))

        # Remove already-labeled from candidates
        if labeled_indices:
            candidate_set = set(candidate_indices) - set(labeled_indices)
            candidate_indices = sorted(candidate_set)

        if len(candidate_indices) == 0:
            return []

        n_select = min(n_select, len(candidate_indices))

        # Initialize: if we have labeled data, use it as the "selected" set
        if labeled_indices and len(labeled_indices) > 0:
            selected_features = features[labeled_indices]
        else:
            # Pick the most uncertain slice as seed
            candidate_features = features[candidate_indices]
            seed_local = np.argmax(np.linalg.norm(candidate_features, axis=1))
            seed = candidate_indices[seed_local]
            selected_features = features[seed:seed+1]
            candidate_indices.remove(seed)
            n_select -= 1
            selected = [seed]

        selected = labeled_indices.copy() if labeled_indices else (
            [seed] if 'seed' in dir() else []
        )

        for _ in range(n_select):
            if not candidate_indices:
                break

            cand_feats = features[candidate_indices]
            dists = cdist(cand_feats, selected_features, metric='euclidean')
            min_dists = dists.min(axis=1)

            best_local = np.argmax(min_dists)
            best_idx = candidate_indices[best_local]

            selected.append(best_idx)
            selected_features = np.vstack([
                selected_features, features[best_idx:best_idx+1]
            ])
            candidate_indices.remove(best_idx)

        # Return only the newly selected (not the pre-labeled ones)
        if labeled_indices:
            return [s for s in selected if s not in labeled_indices]
        return selected


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Active Learning Controller (QThread worker)
# ═══════════════════════════════════════════════════════════════════════════════

class ActiveLearningWorker(QThread):
    """
    Background worker that runs MC-Dropout uncertainty estimation
    and CoreSet selection.
    """

    progress = pyqtSignal(int, int, str)      # current, total, message
    finished = pyqtSignal(object)              # ActiveLearningState
    error = pyqtSignal(str)

    def __init__(self, volume, config, labeled_slices=None):
        """
        Args:
            volume: (D, H, W) uint8 volume
            config: dict with keys:
                - ckpt: path to model checkpoint
                - arch: model architecture name
                - classes: number of output classes
                - dim: '2D' or '2.5D'
                - adj: adjacency for 2.5D
                - mc_passes: number of MC-Dropout forward passes
                - n_recommend: number of slices to recommend
                - z_from, z_to: slice range (optional)
            labeled_slices: list of already-labeled slice indices
        """
        super().__init__()
        self.volume = volume
        self.config = config
        self.labeled_slices = labeled_slices or []

    def run(self):
        try:
            if not HAS_TORCH:
                self.error.emit("PyTorch not available")
                return

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Load model
            self.progress.emit(0, 100, "Loading model...")
            arch = self.config.get('arch', 'UNet')
            out_c = self.config.get('classes', 3)
            dim_str = self.config.get('dim', '2D')
            adj = self.config.get('adj', 3)
            in_c = int(adj) if dim_str == '2.5D' else 1
            mc_passes = self.config.get('mc_passes', 5)
            n_recommend = self.config.get('n_recommend', 10)

            from inno3d.models import create_model
            model = create_model(arch, in_c, out_c).to(device)
            ckpt = self.config.get('ckpt', '')
            if ckpt and os.path.exists(ckpt):
                model.load_state_dict(
                    torch.load(ckpt, map_location=device, weights_only=True)
                )
            else:
                self.error.emit(
                    "No model checkpoint found. Train a model first, "
                    "then use Active Learning to find the best slices to label next."
                )
                return

            # Run MC-Dropout estimation
            self.progress.emit(5, 100, f"MC-Dropout ({mc_passes} passes)...")
            estimator = MCDropoutEstimator(n_passes=mc_passes)

            z_from = int(self.config.get('z_from', 0))
            z_to = int(self.config.get('z_to', self.volume.shape[0] - 1))

            def on_progress(current, total):
                pct = 5 + int(current / total * 70)
                self.progress.emit(pct, 100,
                    f"Uncertainty scan: slice {current}/{total}")

            entropy_map, variance_map, mean_pred = estimator.estimate_volume(
                model=model,
                volume=self.volume,
                device=device,
                in_channels=in_c,
                adj=adj,
                dim_mode=dim_str,
                z_from=z_from,
                z_to=z_to,
                progress_callback=on_progress,
            )

            # Compute per-slice uncertainties
            self.progress.emit(78, 100, "Computing per-slice scores...")
            D = self.volume.shape[0]
            uncertainties = []
            for z in range(z_from, z_to + 1):
                e = entropy_map[z]
                v = variance_map[z]
                fg_mask = e > 0.05
                fg_entropy = e[fg_mask].mean() if fg_mask.any() else 0.0

                su = SliceUncertainty(
                    slice_index=z,
                    mean_entropy=float(e.mean()),
                    max_entropy=float(e.max()),
                    fg_entropy=float(fg_entropy),
                    prediction_variance=float(v.mean()),
                    is_labeled=(z in self.labeled_slices),
                )
                uncertainties.append(su)

            # CoreSet diversity selection
            self.progress.emit(85, 100, "Diversity selection (CoreSet)...")
            selector = CoreSetSelector()
            features = selector.compute_slice_features(
                entropy_map, self.volume, variance_map
            )

            # Candidates: only slices with meaningful content
            candidate_indices = [
                u.slice_index for u in uncertainties
                if u.mean_entropy > 0.01 and not u.is_labeled
            ]

            # Select diverse slices
            diverse_selected = selector.select_diverse(
                features=features,
                n_select=n_recommend,
                labeled_indices=self.labeled_slices,
                candidate_indices=candidate_indices,
            )

            # Compute combined scores
            self.progress.emit(92, 100, "Ranking recommendations...")
            for u in uncertainties:
                # Combined score: entropy + diversity
                u.combined_score = (
                    0.4 * u.mean_entropy +
                    0.3 * u.fg_entropy +
                    0.2 * u.prediction_variance +
                    0.1 * (1.0 if u.slice_index in diverse_selected else 0.0)
                )

            # Final recommendation: merge uncertainty ranking + diversity
            unlabeled = [u for u in uncertainties if not u.is_labeled]
            unlabeled.sort(key=lambda u: u.combined_score, reverse=True)
            top_uncertain = [u.slice_index for u in unlabeled[:n_recommend * 2]]

            # Combine: prefer slices that are both uncertain AND diverse
            recommended = []
            for z in diverse_selected:
                if z in top_uncertain:
                    recommended.append(z)
            for z in diverse_selected:
                if z not in recommended:
                    recommended.append(z)
            for z in top_uncertain:
                if z not in recommended and len(recommended) < n_recommend:
                    recommended.append(z)
            recommended = recommended[:n_recommend]

            # Build state
            self.progress.emit(100, 100, "Done!")
            state = ActiveLearningState(
                labeled_slices=self.labeled_slices.copy(),
                recommended_slices=recommended,
                all_uncertainties=uncertainties,
                entropy_map=entropy_map,
                total_slices=D,
                mc_passes=mc_passes,
                status="complete",
            )

            self.finished.emit(state)

        except Exception as e:
            import traceback
            self.error.emit(f"Active Learning error: {e}\n{traceback.format_exc()}")
