"""
Self-Supervised Foundation Pre-Training Engine (BYOL).

Patent: "Self-Supervised Foundation Model for 3D Semiconductor
Microstructure Understanding via Contrastive Volume Representation Learning"

Approach: BYOL (Bootstrap Your Own Latent) — learns useful encoder
representations from UNLABELED volume data by predicting one
augmented view from another.

After pre-training, the encoder weights are loaded into segmentation
models (UNet, ResUNet++, etc.) for few-shot fine-tuning, dramatically
reducing the number of labeled slices needed.

Author: INNO3D Team
Version: 1.0.0
"""

import os
import copy
import math
import numpy as np
from typing import Optional, Tuple, Callable

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from PyQt5.QtCore import QThread, pyqtSignal


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Semiconductor-Specific Augmentations
# ═══════════════════════════════════════════════════════════════════════════════

class SemiconductorAugmentations:
    """
    Domain-specific augmentations for semiconductor X-ray/SEM volumes.
    
    Unlike natural images, semiconductor images have unique properties:
    - Gray-scale with subtle intensity gradients
    - Rotational symmetry (TGV arrays are symmetric)
    - Intensity varies with depth (Z-dependent)
    - Noise characteristics from X-ray acquisition
    """

    @staticmethod
    def random_crop(img: torch.Tensor, crop_size: int = 96) -> torch.Tensor:
        """Random spatial crop from a slice."""
        _, H, W = img.shape
        if H <= crop_size or W <= crop_size:
            return F.interpolate(
                img.unsqueeze(0), size=(crop_size, crop_size),
                mode='bilinear', align_corners=False
            ).squeeze(0)
        y = torch.randint(0, H - crop_size, (1,)).item()
        x = torch.randint(0, W - crop_size, (1,)).item()
        return img[:, y:y + crop_size, x:x + crop_size]

    @staticmethod
    def random_rotation(img: torch.Tensor) -> torch.Tensor:
        """Random 90° rotation (0, 90, 180, 270)."""
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            img = torch.rot90(img, k, [1, 2])
        return img

    @staticmethod
    def random_flip(img: torch.Tensor) -> torch.Tensor:
        """Random horizontal and vertical flip."""
        if torch.rand(1).item() > 0.5:
            img = torch.flip(img, [1])
        if torch.rand(1).item() > 0.5:
            img = torch.flip(img, [2])
        return img

    @staticmethod
    def random_intensity_jitter(img: torch.Tensor) -> torch.Tensor:
        """Random brightness and contrast perturbation."""
        brightness = torch.empty(1).uniform_(-0.15, 0.15).item()
        contrast = torch.empty(1).uniform_(0.7, 1.3).item()
        img = img * contrast + brightness
        return img.clamp(0, 1)

    @staticmethod
    def random_gaussian_blur(img: torch.Tensor) -> torch.Tensor:
        """Random Gaussian blur (simulates focus variation)."""
        if torch.rand(1).item() > 0.5:
            ksize = 3
            sigma = torch.empty(1).uniform_(0.1, 1.5).item()
            # Create 1D Gaussian kernel
            x = torch.arange(ksize, dtype=torch.float32) - ksize // 2
            kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)
            kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
            C = img.shape[0]
            kernel_2d = kernel_2d.expand(C, 1, -1, -1)
            img = F.conv2d(
                img.unsqueeze(0), kernel_2d, padding=ksize // 2, groups=C
            ).squeeze(0)
        return img

    @staticmethod
    def random_noise(img: torch.Tensor) -> torch.Tensor:
        """Add random Gaussian noise (simulates X-ray acquisition noise)."""
        if torch.rand(1).item() > 0.5:
            std = torch.empty(1).uniform_(0.01, 0.05).item()
            noise = torch.randn_like(img) * std
            img = (img + noise).clamp(0, 1)
        return img

    @staticmethod
    def random_cutout(img: torch.Tensor, n_holes: int = 2,
                      max_size: int = 16) -> torch.Tensor:
        """Random rectangular cutout (forces model to learn from context)."""
        _, H, W = img.shape
        for _ in range(n_holes):
            if torch.rand(1).item() > 0.5:
                h = torch.randint(4, max_size + 1, (1,)).item()
                w = torch.randint(4, max_size + 1, (1,)).item()
                y = torch.randint(0, max(1, H - h), (1,)).item()
                x = torch.randint(0, max(1, W - w), (1,)).item()
                img[:, y:y + h, x:x + w] = 0
        return img

    @classmethod
    def apply_view(cls, img: torch.Tensor, crop_size: int = 96) -> torch.Tensor:
        """Apply full augmentation pipeline to create one 'view'."""
        img = cls.random_crop(img, crop_size)
        img = cls.random_rotation(img)
        img = cls.random_flip(img)
        img = cls.random_intensity_jitter(img)
        img = cls.random_gaussian_blur(img)
        img = cls.random_noise(img)
        img = cls.random_cutout(img)
        return img


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Unlabeled Volume Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class UnlabeledVolumeDataset(Dataset):
    """
    Dataset that yields pairs of augmented views from unlabeled volume slices.
    No masks or labels needed — uses only raw pixel data.
    """

    def __init__(self, volume: np.ndarray, crop_size: int = 96,
                 in_channels: int = 1, adj: int = 3,
                 dim_mode: str = '2D'):
        """
        Args:
            volume: (D, H, W) uint8 raw volume
            crop_size: spatial crop size for augmented views
            in_channels: 1 for 2D, adj for 2.5D
            adj: adjacency for 2.5D
            dim_mode: '2D' or '2.5D'
        """
        self.volume = volume
        self.crop_size = crop_size
        self.in_channels = in_channels
        self.adj = adj
        self.dim_mode = dim_mode
        self.D, self.H, self.W = volume.shape[:3]

    def __len__(self):
        return self.D

    def __getitem__(self, idx):
        # Extract slice (or multi-slice for 2.5D)
        if self.dim_mode == '2.5D' and self.in_channels > 1:
            pad = self.in_channels // 2
            indices = np.clip(
                np.arange(idx - pad, idx + pad + 1), 0, self.D - 1
            )
            img = self.volume[indices].astype(np.float32)
        else:
            img = self.volume[idx:idx + 1].astype(np.float32)

        # Normalize to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        img_t = torch.from_numpy(img)

        # Create two augmented views
        view1 = SemiconductorAugmentations.apply_view(img_t, self.crop_size)
        view2 = SemiconductorAugmentations.apply_view(img_t, self.crop_size)

        return view1, view2


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BYOL Components
# ═══════════════════════════════════════════════════════════════════════════════

class ProjectionMLP(nn.Module):
    """MLP projection head: encoder output → compact representation."""

    def __init__(self, in_dim: int, hidden_dim: int = 256,
                 out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class PredictionMLP(nn.Module):
    """MLP predictor: online projection → predict target projection."""

    def __init__(self, in_dim: int = 128, hidden_dim: int = 256,
                 out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class EncoderWrapper(nn.Module):
    """
    Wraps a segmentation model (UNet, ResUNet++, etc.) and extracts
    only the encoder features + global average pooling.
    
    This allows pre-training the encoder part of any supported model
    architecture, then transferring weights to the full segmentation model.
    """

    def __init__(self, model: nn.Module, arch: str):
        super().__init__()
        self.model = model
        self.arch = arch.lower().replace(' ', '')
        self._feature_dim = self._detect_feature_dim()

    def _detect_feature_dim(self) -> int:
        """Detect the encoder's output feature dimension."""
        if self.arch in ('unet', 'tinyunet'):
            return 32  # enc2 output channels
        elif self.arch in ('resunet++', 'resunetplusplus', 'resunet'):
            return 256  # enc4 output channels (last encoder block)
        elif self.arch in ('mask2former',):
            return 256  # backbone stage4 output
        return 64  # fallback

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x) -> torch.Tensor:
        """Extract encoder features → global average pool → (B, D)."""
        if self.arch in ('unet', 'tinyunet'):
            e1 = self.model.enc1(x)
            e2 = self.model.enc2(self.model.pool(e1))
            features = F.adaptive_avg_pool2d(e2, 1).flatten(1)

        elif self.arch in ('resunet++', 'resunetplusplus', 'resunet'):
            e1 = self.model.enc1(x)
            e2 = self.model.enc2(e1)
            e3 = self.model.enc3(e2)
            e4 = self.model.enc4(e3)
            b = self.model.bridge(e4)
            features = F.adaptive_avg_pool2d(b, 1).flatten(1)

        elif self.arch in ('mask2former',):
            feats = []
            out = x
            for stage in [self.model.backbone.stage1,
                          self.model.backbone.stage2,
                          self.model.backbone.stage3,
                          self.model.backbone.stage4]:
                out = stage(out)
                feats.append(out)
            features = F.adaptive_avg_pool2d(feats[-1], 1).flatten(1)

        else:
            # Generic fallback: run full model, pool the last conv output
            out = self.model(x)
            features = F.adaptive_avg_pool2d(out, 1).flatten(1)

        return features


class BYOL(nn.Module):
    """
    BYOL (Bootstrap Your Own Latent) for self-supervised pre-training.
    
    No negative pairs needed — uses an exponential moving average (EMA)
    target network to provide stable learning targets.
    
    Architecture:
        Online:  Encoder → Projector → Predictor → loss
        Target:  Encoder → Projector (EMA copy, no gradients)
    """

    def __init__(self, encoder: EncoderWrapper, feature_dim: int,
                 proj_hidden: int = 256, proj_out: int = 128,
                 ema_decay: float = 0.996):
        super().__init__()

        # Online network
        self.online_encoder = encoder
        self.online_projector = ProjectionMLP(feature_dim, proj_hidden, proj_out)
        self.online_predictor = PredictionMLP(proj_out, proj_hidden, proj_out)

        # Target network (EMA copy — no gradients)
        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

        self.ema_decay = ema_decay

    @torch.no_grad()
    def update_target(self):
        """Update target network via exponential moving average."""
        for op, tp in zip(self.online_encoder.parameters(),
                          self.target_encoder.parameters()):
            tp.data = self.ema_decay * tp.data + (1 - self.ema_decay) * op.data
        for op, tp in zip(self.online_projector.parameters(),
                          self.target_projector.parameters()):
            tp.data = self.ema_decay * tp.data + (1 - self.ema_decay) * op.data

    @staticmethod
    def regression_loss(x, y):
        """Normalized MSE loss (cosine distance)."""
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1).mean()

    def forward(self, view1, view2):
        """
        Forward pass with symmetric loss.
        
        Args:
            view1, view2: (B, C, H, W) two augmented views
            
        Returns:
            loss: scalar BYOL loss
        """
        # Online branch for view1
        z1_online = self.online_projector(self.online_encoder(view1))
        p1 = self.online_predictor(z1_online)

        # Online branch for view2
        z2_online = self.online_projector(self.online_encoder(view2))
        p2 = self.online_predictor(z2_online)

        # Target branch (no gradients)
        with torch.no_grad():
            z1_target = self.target_projector(self.target_encoder(view1))
            z2_target = self.target_projector(self.target_encoder(view2))

        # Symmetric loss
        loss = self.regression_loss(p1, z2_target.detach())
        loss += self.regression_loss(p2, z1_target.detach())
        return loss / 2


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Pre-Training Worker (QThread)
# ═══════════════════════════════════════════════════════════════════════════════

class FoundationPretrainWorker(QThread):
    """
    Background worker that runs BYOL self-supervised pre-training
    on unlabeled volume data.
    """

    progress = pyqtSignal(int, int, float, str)  # epoch, total, loss, msg
    finished = pyqtSignal(str, str)               # status_msg, saved_path
    error = pyqtSignal(str)

    def __init__(self, volume: np.ndarray, config: dict):
        """
        Args:
            volume: (D, H, W) uint8 raw volume — NO labels needed!
            config: dict with keys:
                - arch: model architecture ('UNet', 'ResUNet++', 'Mask2Former')
                - epochs: pre-training epochs (default 50)
                - batch_size: (default 16)
                - lr: learning rate (default 3e-4)
                - crop_size: spatial crop size (default 96)
                - save_dir: directory to save pre-trained weights
                - dim: '2D' or '2.5D'
                - adj: adjacency for 2.5D
                - classes: number of output classes
        """
        super().__init__()
        self.volume = volume
        self.config = config
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        try:
            if not HAS_TORCH:
                self.error.emit("PyTorch not available")
                return

            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )

            arch = self.config.get('arch', 'UNet')
            epochs = self.config.get('epochs', 50)
            batch_size = self.config.get('batch_size', 16)
            lr = self.config.get('lr', 3e-4)
            crop_size = self.config.get('crop_size', 96)
            save_dir = self.config.get('save_dir', '')
            dim_str = self.config.get('dim', '2D')
            adj = self.config.get('adj', 3)
            out_c = self.config.get('classes', 3)
            in_c = int(adj) if dim_str == '2.5D' else 1

            # Create segmentation model (we'll pre-train its encoder)
            from inno3d.models import create_model
            seg_model = create_model(arch, in_c, out_c).to(device)

            # Wrap encoder
            encoder = EncoderWrapper(seg_model, arch).to(device)
            feature_dim = encoder.feature_dim

            # Build BYOL
            byol = BYOL(encoder, feature_dim).to(device)

            # Dataset
            dataset = UnlabeledVolumeDataset(
                self.volume, crop_size=crop_size,
                in_channels=in_c, adj=adj, dim_mode=dim_str
            )
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=(device.type == 'cuda'),
                drop_last=True
            )

            # Optimizer (only online network parameters)
            online_params = list(byol.online_encoder.parameters()) + \
                            list(byol.online_projector.parameters()) + \
                            list(byol.online_predictor.parameters())
            optimizer = torch.optim.AdamW(online_params, lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )

            # CuDNN optimization
            if device.type == 'cuda':
                torch.backends.cudnn.benchmark = True

            scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

            self.progress.emit(0, epochs, 0.0, "Starting pre-training...")

            best_loss = float('inf')

            for epoch in range(1, epochs + 1):
                if self._stopped:
                    break

                byol.train()
                epoch_loss = 0.0
                n_batches = 0

                for view1, view2 in loader:
                    if self._stopped:
                        break

                    view1 = view1.to(device, non_blocking=True)
                    view2 = view2.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)

                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        loss = byol(view1, view2)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    # Update target network (EMA)
                    byol.update_target()

                    epoch_loss += loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / max(1, n_batches)
                scheduler.step()

                # Progress
                self.progress.emit(
                    epoch, epochs, avg_loss,
                    f"Epoch {epoch}/{epochs} | BYOL Loss: {avg_loss:.4f}"
                )

                # Save best
                if avg_loss < best_loss and save_dir and os.path.isdir(save_dir):
                    best_loss = avg_loss
                    # Save the FULL segmentation model state_dict
                    # (encoder is a wrapper around seg_model)
                    pretrained_path = os.path.join(
                        save_dir, 'foundation_pretrained.pth'
                    )
                    torch.save(
                        seg_model.state_dict(), pretrained_path
                    )

            # Final save
            saved_path = ""
            if save_dir and os.path.isdir(save_dir):
                saved_path = os.path.join(
                    save_dir, 'foundation_pretrained.pth'
                )
                torch.save(seg_model.state_dict(), saved_path)

            self.finished.emit(
                f"Pre-training complete! {epochs} epochs, "
                f"final loss: {avg_loss:.4f}\n"
                f"Saved: {saved_path}",
                saved_path
            )

        except Exception as e:
            import traceback
            self.error.emit(
                f"Foundation pre-training error: {e}\n"
                f"{traceback.format_exc()}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Weight Transfer Utility
# ═══════════════════════════════════════════════════════════════════════════════

def load_pretrained_encoder(
    model: nn.Module,
    pretrained_path: str,
    device: torch.device = None,
    strict: bool = False,
) -> Tuple[int, int]:
    """
    Load pre-trained foundation encoder weights into a segmentation model.
    
    Uses strict=False by default so that:
    - Matching encoder weights are loaded (pre-trained)
    - Non-matching decoder weights are left randomly initialized
    - Different in_channels/out_classes are handled gracefully
    
    Args:
        model: Segmentation model (UNet, ResUNet++, etc.)
        pretrained_path: Path to foundation_pretrained.pth
        device: Target device
        strict: If False, skip mismatched keys
        
    Returns:
        (n_loaded, n_skipped) tuple
    """
    if device is None:
        device = torch.device('cpu')

    state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
    model_dict = model.state_dict()

    # Filter: only load keys that match in both name and shape
    loaded = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            loaded[k] = v
        else:
            skipped.append(k)

    model_dict.update(loaded)
    model.load_state_dict(model_dict)

    return len(loaded), len(skipped)
