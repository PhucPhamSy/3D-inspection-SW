import numpy as np
import os
import glob
from pathlib import Path
from skimage import io, draw, filters, measure
from skimage.morphology import disk
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Assuming styles and resources are available
from inno3d.core.styles import SemiconductorTheme
try:
    from segment_anything import sam_model_registry, SamPredictor
    import torch; SAM_AVAIL = True
except ImportError:
    SAM_AVAIL = False

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class NoScrollEventFilter(QObject):
    """Blocks mouse wheel on SpinBoxes/ComboBoxes/Sliders — scroll is for sidebar only."""
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            if isinstance(obj, (QAbstractSpinBox, QComboBox, QSlider)):
                # Always block wheel on these widgets, even when focused
                event.ignore()
                # Forward to the nearest scrollable parent
                p = obj.parent()
                while p:
                    if isinstance(p, QScrollArea):
                        QApplication.sendEvent(p.viewport(), event)
                        return True
                    p = p.parent()
                return True
        return super().eventFilter(obj, event)

class TinyChartWidget(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(3, 2.5), dpi=80, facecolor='#0B0F19')
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#0B0F19')
        self.ax.tick_params(colors='#8B949E', labelsize=8)
        
        # Hide top and right borders for a modern, clean look
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_color('#30363D')
        self.ax.spines['left'].set_color('#30363D')
        
        self.ax.grid(True, linestyle=':', alpha=0.5, color='#8B949E')
        self.ax.set_xlabel('Epoch', color='#8B949E', fontsize=9, fontweight='bold')
        self.ax.set_ylabel('Loss', color='#8B949E', fontsize=9, fontweight='bold')
        
        self.history_tr, self.history_val = [], []
        # Neon colors with slightly thicker lines for better visibility
        self.line_tr, = self.ax.plot([], [], color='#00FF96', label='Train Loss', linewidth=2.5)
        self.line_val, = self.ax.plot([], [], color='#FF6400', label='Val Loss', linewidth=2.5)
        
        # Clean legend without border
        legend = self.ax.legend(facecolor='#0B0F19', labelcolor='white', frameon=False, loc='upper right')
        
        # Explicit margins to prevent clipping on the right and reduce empty space on the left
        self.fig.subplots_adjust(left=0.15, right=0.96, top=0.95, bottom=0.15)

    def add_data(self, tr, val):
        self.history_tr.append(tr); self.history_val.append(val); self.update_plot()
        
    def clear(self):
        self.history_tr.clear(); self.history_val.clear(); self.update_plot()

    def update_plot(self):
        if not self.history_tr: return
        epochs = list(range(1, len(self.history_tr) + 1))
        # Smooth lines safely
        try:
            from scipy.ndimage import gaussian_filter1d
            if len(epochs) >= 3:
                tr_sm = gaussian_filter1d(self.history_tr, sigma=1.0)
                val_sm = gaussian_filter1d(self.history_val, sigma=1.0)
            else:
                tr_sm, val_sm = self.history_tr, self.history_val
        except:
            tr_sm, val_sm = self.history_tr, self.history_val
            
        self.line_tr.set_data(epochs, tr_sm); self.line_val.set_data(epochs, val_sm)
        self.ax.relim(); self.ax.autoscale_view(); self.draw()



def get_tiny_unet():
    import torch
    class TinyUNet(torch.nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.enc1 = torch.nn.Sequential(torch.nn.Conv2d(in_c, 16, 3, padding=1), torch.nn.ReLU(inplace=True))
            self.pool = torch.nn.MaxPool2d(2)
            self.enc2 = torch.nn.Sequential(torch.nn.Conv2d(16, 32, 3, padding=1), torch.nn.ReLU(inplace=True))
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # Decoder takes concatenated features: 32 (from up_e2) + 16 (from e1) = 48
            self.dec1 = torch.nn.Sequential(torch.nn.Conv2d(48, 16, 3, padding=1), torch.nn.ReLU(inplace=True))
            self.final = torch.nn.Conv2d(16, out_c, 1)
        def forward(self, x):
            import torch.nn.functional as F
            import torch
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            up_e2 = self.up(e2)
            up_e2 = F.interpolate(up_e2, size=e1.shape[2:], mode='bilinear', align_corners=True)
            # Concatenate along channel dimension (dim=1)
            d1 = self.dec1(torch.cat([up_e2, e1], dim=1))
            return self.final(d1)
    return TinyUNet

class DiceLoss:
    """Soft Dice Loss for segmentation — foreground-only by default.
    
    Excludes background (class 0) from the mean so that minority foreground
    classes (e.g. void) receive equal gradient pressure regardless of how
    many pixels they occupy relative to the background.
    
    When all foreground classes are absent from a patch the loss gracefully
    falls back to including background to avoid NaN gradients.
    """
    def __init__(self, num_classes, smooth=1e-5, ignore_bg=True):
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_bg = ignore_bg and num_classes > 1

    def __call__(self, logits, targets):
        import torch
        import torch.nn.functional as F
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets.long(), self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * targets_oh).sum(dims)
        card  = probs.sum(dims) + targets_oh.sum(dims)
        dice  = (2.0 * inter + self.smooth) / (card + self.smooth)   # (num_classes,)

        if self.ignore_bg:
            fg_dice = dice[1:]                       # exclude background
            # Weight: inverse-frequency so rare classes matter more
            fg_card = card[1:]                        # per-class pixel counts
            weights = 1.0 / (fg_card + self.smooth)
            weights = weights / weights.sum()         # normalize to 1
            return 1.0 - (fg_dice * weights).sum()
        return 1.0 - dice.mean()


class FocalLoss:
    """Focal Loss (Lin et al., 2017) for class-imbalanced segmentation.
    
    Down-weights easy (well-classified) examples so that the model focuses
    training on hard, minority-class pixels like void regions.
    """
    def __init__(self, num_classes, gamma=2.0, alpha=None):
        self.num_classes = num_classes
        self.gamma = gamma
        # alpha: per-class weight tensor or None for uniform
        self.alpha = alpha

    def __call__(self, logits, targets):
        import torch
        import torch.nn.functional as F
        ce = F.cross_entropy(logits, targets.long(), reduction='none')
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets.long()]
            focal = alpha_t * focal

        return focal.mean()


class AITrainWorker(QThread):
    epoch_done = pyqtSignal(int, float, float, object, object, object) # epoch, try_loss, val_loss, in_img, gt_img, pr_img
    batch_progress = pyqtSignal(int, int, int, int) # ep, tot_ep, batch, tot_batch
    finished_train = pyqtSignal(str)
    error = pyqtSignal(str)


    def __init__(self, sources, frames, config):
        super().__init__()
        self.sources = sources
        self.frames = frames
        self.config = config
        self.active_source_id = config.get('active_source_id')
        self._stopped = False

    def stop(self):
        self._stopped = True

    def _resolve_source(self, sid):
        src = next((s for s in self.sources if s['id'] == sid), None)
        if src is not None:
            return src
        if self.active_source_id is not None:
            src = next((s for s in self.sources if s['id'] == self.active_source_id), None)
            if src is not None:
                return src
        if self.sources and 'volume' in self.sources[0]:
            return self.sources[0]
        return None

    def _build_samples(self, frames):
        samples = []
        # Path A: Frame-based samples
        for f in frames:
            sid = f.get('source_id', 1)
            src = self._resolve_source(sid)
            if not src: continue
            vol = src.get('volume')
            msk = src.get('mask')
            if vol is None: continue
            if msk is None: msk = np.zeros_like(vol, dtype=np.uint8)
            z = int(f.get('slice_index', 0))
            if z < 0 or z >= vol.shape[0]: continue
            x1, y1, x2, y2 = f.get('rect', (0, 0, 0, 0))
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h, w = vol.shape[1], vol.shape[2]
            x1, x2 = max(0, min(x1, w-1)), max(1, min(x2, w))
            y1, y2 = max(0, min(y1, h-1)), max(1, min(y2, h))
            if x2 <= x1 or y2 <= y1: continue
            sp = f.get('split', 'train')
            fg_pixels = int(np.count_nonzero(msk[z, y1:y2, x1:x2]))
            samples.append({'sid': sid, 'source': src, 'z': z, 'rect': (x1, y1, x2, y2), 'fg_pixels': fg_pixels, 'split': sp})

        # Path B: Auto-tile labeled slices if no frame-based samples
        if not samples:
            patch_size = self.config.get('patch_size', 256)
            slice_splits = self.config.get('slice_splits', {})
            for src in self.sources:
                vol = src.get('volume')
                msk = src.get('mask')
                if vol is None or msk is None: continue
                labeled_z = np.where(np.any(msk > 0, axis=(1, 2)))[0]
                for z in labeled_z:
                    H, W = vol.shape[1], vol.shape[2]
                    # Use user-assigned split if available, else mark as 'auto'
                    z_split = slice_splits.get(int(z), 'auto')
                    for y0 in range(0, H, patch_size):
                        for x0 in range(0, W, patch_size):
                            y1, x1 = min(y0 + patch_size, H), min(x0 + patch_size, W)
                            if np.any(msk[z, y0:y1, x0:x1] > 0):
                                samples.append({'sid': src.get('id', 1), 'source': src, 'z': int(z),
                                    'rect': (x0, y0, x1, y1), 'fg_pixels': int(np.count_nonzero(msk[z, y0:y1, x0:x1])), 'split': z_split})
            # Auto-assign splits ONLY for samples that were not manually assigned
            auto = [s for s in samples if s.get('split') == 'auto']
            np.random.shuffle(auto)
            n = len(auto)
            for i, s in enumerate(auto):
                if i < int(n * 0.7): s['split'] = 'train'
                elif i < int(n * 0.85): s['split'] = 'val'
                else: s['split'] = 'test'
        return samples

    def run(self):
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import Dataset, DataLoader
            from torch.optim.lr_scheduler import CosineAnnealingLR
            import torch.nn.functional as F

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 1. Prepare data — _build_samples handles both frames and auto-tiling
            all_samples = self._build_samples(self.frames)
            train_samples = [s for s in all_samples if s.get('split') == 'train']
            val_samples = [s for s in all_samples if s.get('split') == 'val']

            if not train_samples:
                self.error.emit("No training samples found! Add ROI frames or label some slices.")
                return

            out_c = int(self.config['num_classes'])
            dim_str = self.config.get('dim', '2D')
            adj = self.config.get('adj', 3)
            in_c = int(adj) if dim_str == '2.5D' else 1

            class ROIDataset(Dataset):
                def __init__(self, samples, out_classes, in_c, augment=False):
                    self.samples = samples
                    self.out_classes = out_classes
                    self.in_c = in_c
                    self.augment = augment
                def __len__(self): return len(self.samples)
                def __getitem__(self, idx):
                    s = self.samples[idx]
                    vol = s['source']['volume']
                    msk = s['source']['mask'] if 'mask' in s['source'] and s['source']['mask'] is not None else np.zeros((1, 1, 1), dtype=np.uint8)
                    z = s['z']; x1, y1, x2, y2 = s['rect']
                    
                    if self.in_c == 1:
                        img = vol[z, y1:y2, x1:x2].astype(np.float32)
                        img_t = torch.from_numpy(img).unsqueeze(0)
                        mc_ref = img
                    else:
                        pad = self.in_c // 2
                        indices = np.clip(np.arange(z - pad, z + pad + 1), 0, vol.shape[0] - 1)
                        img = vol[indices, y1:y2, x1:x2].astype(np.float32)
                        img_t = torch.from_numpy(img)
                        mc_ref = img[0]
                        
                    mc = msk[z, y1:y2, x1:x2].astype(np.int64) if msk.shape[0] > 1 else np.zeros_like(mc_ref, dtype=np.int64)
                    mc = np.clip(mc, 0, self.out_classes - 1)
                    # NOTE: Augmentation and normalization are moved to GPU for massive speedup
                    return img_t, torch.from_numpy(mc), s

            def collate_and_pad(batch):
                if not batch: return None
                imgs, masks, metas = zip(*batch)
                
                # Fast path: all patches have exactly the same shape
                if all(t.shape == imgs[0].shape for t in imgs):
                    return torch.stack(imgs), torch.stack(masks), list(metas)
                
                # Slow path: variable sizes (e.g., edge patches)
                mh = max(int(t.shape[-2]) for t in imgs); mw = max(int(t.shape[-1]) for t in imgs)
                ib = torch.zeros((len(imgs), in_c, mh, mw), dtype=torch.float32)
                mb = torch.zeros((len(imgs), mh, mw), dtype=torch.long)
                for i, (it, mt) in enumerate(zip(imgs, masks)):
                    h, w = int(it.shape[-2]), int(it.shape[-1])
                    ib[i, :, :h, :w] = it; mb[i, :h, :w] = mt.long()
                return ib, mb, list(metas)

            bs = max(1, int(self.config.get('batch_size', 1)))
            pin = (device.type == 'cuda')
            train_loader = DataLoader(ROIDataset(train_samples, out_c, in_c, augment=True), batch_size=bs, shuffle=True, num_workers=0, pin_memory=pin, collate_fn=collate_and_pad)
            val_loader = DataLoader(ROIDataset(val_samples, out_c, in_c, augment=False), batch_size=bs, shuffle=False, num_workers=0, pin_memory=pin, collate_fn=collate_and_pad) if val_samples else None

            # Model
            from inno3d.models import create_model
            model = create_model(self.config.get('arch', 'UNet'), in_c, out_c).to(device)

            # CuDNN optimization for fixed input sizes (massive speedup on RTX 4000 series)
            if device.type == 'cuda':
                torch.backends.cudnn.benchmark = True

            # Loss — compute class-frequency weights for balanced training
            loss_name = self.config.get('loss', 'CrossEntropy')
            manual_weights = self.config.get('class_weights', None)  # None = auto
            
            if manual_weights is not None and len(manual_weights) == out_c:
                # Use user-specified manual weights
                ce_weights = torch.tensor(manual_weights, dtype=torch.float32)
            else:
                # Auto: compute per-class pixel counts from training data for weighting
                class_pixel_counts = torch.zeros(out_c, dtype=torch.float32)
                for s in train_samples:
                    msk = s['source'].get('mask')
                    if msk is None: continue
                    z = s['z']; x1, y1, x2, y2 = s['rect']
                    patch_mask = msk[z, y1:y2, x1:x2]
                    for c in range(out_c):
                        class_pixel_counts[c] += float((patch_mask == c).sum())
                
                # Inverse-frequency weights (clamped to avoid inf for absent classes)
                total_pixels = class_pixel_counts.sum()
                if total_pixels > 0:
                    ce_weights = total_pixels / (out_c * class_pixel_counts.clamp(min=1.0))
                    ce_weights = ce_weights / ce_weights.sum() * out_c  # normalize so mean = 1
                else:
                    ce_weights = torch.ones(out_c)
            ce_weights = ce_weights.to(device)
            
            if loss_name == 'DiceLoss':
                criterion = DiceLoss(out_c)
            elif loss_name == 'Dice+CE':
                ce_fn = nn.CrossEntropyLoss(weight=ce_weights)
                dice_fn = DiceLoss(out_c)
                criterion = lambda logits, tgt: ce_fn(logits, tgt) + dice_fn(logits, tgt)
            elif loss_name == 'Focal+Dice':
                focal_fn = FocalLoss(out_c, gamma=2.0, alpha=ce_weights)
                dice_fn = DiceLoss(out_c)
                criterion = lambda logits, tgt: focal_fn(logits, tgt) + dice_fn(logits, tgt)
            elif loss_name == 'FocalLoss':
                criterion = FocalLoss(out_c, gamma=2.0, alpha=ce_weights)
            else:
                criterion = nn.CrossEntropyLoss(weight=ce_weights)

            # Optimizer
            lr = self.config.get('lr', 1e-3)
            opt_name = self.config.get('optimizer', 'Adam')
            if opt_name == 'AdamW': optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            elif opt_name == 'SGD': optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
            else: optimizer = optim.Adam(model.parameters(), lr=lr)

            epochs = self.config['epochs']
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

            # Mixed Precision setup
            scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

            for epoch in range(epochs):
                if self._stopped: break
                model.train(); tr_loss = 0.0; tr_n = 0
                for batch in train_loader:
                    if self._stopped: break
                    ib, mb, _ = batch
                    ib, mb = ib.to(device, non_blocking=pin), mb.to(device, non_blocking=pin)
                    
                    # GPU Augmentation (configurable from AI Data tab)
                    aug = self.config.get('augmentation', {})
                    if aug.get('hflip', True) and torch.rand(1).item() > 0.5:
                        ib = torch.flip(ib, [2]); mb = torch.flip(mb, [1])
                    if aug.get('vflip', True) and torch.rand(1).item() > 0.5:
                        ib = torch.flip(ib, [3]); mb = torch.flip(mb, [2])
                    if aug.get('rot90', True):
                        k = torch.randint(0, 4, (1,)).item()
                        if k > 0: ib = torch.rot90(ib, k, [2, 3]); mb = torch.rot90(mb, k, [1, 2])
                    
                    # GPU Normalization (per-batch/per-image)
                    ib_min = ib.amin(dim=(2, 3), keepdim=True)
                    ib_max = ib.amax(dim=(2, 3), keepdim=True)
                    ib = (ib - ib_min) / (ib_max - ib_min + 1e-5)
                    
                    # Intensity augmentations (applied after normalization to [0,1])
                    if aug.get('brightness', False):
                        shift = (torch.rand(1, device=device) - 0.5) * 0.2  # ±10%
                        ib = torch.clamp(ib + shift, 0.0, 1.0)
                    if aug.get('contrast', False):
                        factor = 0.8 + torch.rand(1, device=device) * 0.4  # [0.8, 1.2]
                        mean = ib.mean(dim=(2, 3), keepdim=True)
                        ib = torch.clamp((ib - mean) * factor + mean, 0.0, 1.0)
                    if aug.get('noise', False):
                        ib = ib + torch.randn_like(ib) * 0.02
                        ib = torch.clamp(ib, 0.0, 1.0)
                    if aug.get('cutout', False) and torch.rand(1).item() > 0.5:
                        _, _, ch, cw = ib.shape
                        cut_h, cut_w = ch // 4, cw // 4
                        cy = torch.randint(0, ch, (1,)).item()
                        cx = torch.randint(0, cw, (1,)).item()
                        y1 = max(0, cy - cut_h // 2); y2 = min(ch, cy + cut_h // 2)
                        x1 = max(0, cx - cut_w // 2); x2 = min(cw, cx + cut_w // 2)
                        ib[:, :, y1:y2, x1:x2] = 0.0

                    optimizer.zero_grad(set_to_none=True)
                    
                    # Mixed precision forward pass
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        out = model(ib)
                        loss = criterion(out, mb)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    bsz = int(ib.shape[0]); tr_loss += float(loss.item()) * bsz; tr_n += bsz
                    b_idx = tr_n // bsz
                    if b_idx % 5 == 0:
                        self.batch_progress.emit(epoch + 1, epochs, b_idx, len(train_loader))
                tr_loss /= max(1, tr_n)

                # Validation
                model.eval(); vl = 0.0; vn = 0
                with torch.no_grad():
                    if val_loader:
                        for batch in val_loader:
                            if self._stopped: break
                            ib, mb, _ = batch
                            ib, mb = ib.to(device, non_blocking=pin), mb.to(device, non_blocking=pin)
                            
                            # GPU Normalization (per-batch/per-image)
                            ib_min = ib.amin(dim=(2, 3), keepdim=True)
                            ib_max = ib.amax(dim=(2, 3), keepdim=True)
                            ib = (ib - ib_min) / (ib_max - ib_min + 1e-5)

                            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                                out = model(ib)
                                loss = criterion(out, mb)
                            
                            bsz = int(ib.shape[0]); vl += float(loss.item()) * bsz; vn += bsz
                val_loss = vl / max(1, vn)
                scheduler.step()

                # Preview
                best_in, best_gt, best_pr = None, None, None
                pool = val_samples if val_samples else train_samples
                ps = max(pool, key=lambda s: int(s.get('fg_pixels', 0))) if pool else None
                if ps:
                    pv = ps['source']['volume']
                    pm = ps['source']['mask'] if 'mask' in ps['source'] and ps['source']['mask'] is not None else None
                    z = ps['z']; x1, y1, x2, y2 = ps['rect']
                    
                    if in_c == 1:
                        pi = pv[z, y1:y2, x1:x2].astype(np.float32)
                    else:
                        pad = in_c // 2
                        indices = np.clip(np.arange(z - pad, z + pad + 1), 0, pv.shape[0] - 1)
                        pi = pv[indices, y1:y2, x1:x2].astype(np.float32)
                        
                    pg = pm[z, y1:y2, x1:x2].astype(np.int64) if pm is not None else np.zeros((pi.shape[-2], pi.shape[-1]), dtype=np.int64)
                    pi = (pi - pi.min()) / (pi.max() - pi.min() + 1e-5)
                    with torch.no_grad():
                        pr = torch.argmax(model(torch.from_numpy(pi).unsqueeze(0).to(device)), dim=1).squeeze(0).cpu().numpy()
                    
                    # Extract the center slice from the normalized 'pi' for visualization
                    vis_pi = pi if in_c == 1 else pi[in_c // 2]
                    best_in = (vis_pi * 255).astype(np.uint8)
                    best_gt = pg.astype(np.uint8)
                    best_pr = pr.astype(np.uint8)

                # Save best checkpoint
                bvl = getattr(self, 'best_val_loss', 1e9)
                sd = self.config.get('save_dir')
                if val_loss < bvl:
                    self.best_val_loss = val_loss
                    if sd and os.path.isdir(sd):
                        torch.save(model.state_dict(), os.path.join(sd, 'best_model.pth'))
                self.epoch_done.emit(epoch + 1, tr_loss, val_loss, best_in, best_gt, best_pr)
                self.msleep(50)

            # ONNX export
            final_msg = 'Training Complete!'
            if sd and os.path.isdir(sd):
                try:
                    onnx_path = os.path.join(sd, 'model.onnx')
                    model.eval()
                    dummy = torch.randn(1, in_c, 256, 256).to(device)
                    torch.onnx.export(model, dummy, onnx_path, input_names=['input'], output_names=['output'],
                        dynamic_axes={'input': {0:'batch',2:'h',3:'w'}, 'output': {0:'batch',2:'h',3:'w'}}, opset_version=17)
                    final_msg = f'Training Complete!\nSaved: best_model.pth + model.onnx in {sd}'
                except Exception as onnx_err:
                    final_msg = f'Training Complete! (ONNX export failed: {onnx_err})'
            self.finished_train.emit(final_msg)

        except Exception as e:
            import traceback
            self.error.emit(f'Error during training: {str(e)}\n{traceback.format_exc()}')


class AIInferWorker(QThread):
    finished_infer = pyqtSignal(object, object)  # mask volume, confidence_per_slice
    progress = pyqtSignal(int, int)  # current_slice, total_slices
    error = pyqtSignal(str)

    def __init__(self, volume, config):
        super().__init__()
        self.volume = volume
        self.config = config

    def run(self):
        try:
            import torch
            import numpy as np
            import torch.nn.functional as F
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            out_c = self.config['classes']
            arch = self.config.get('arch', 'UNet')
            dim_str = self.config.get('dim', '2D')
            adj = self.config.get('adj', 3)
            in_c = int(adj) if dim_str == '2.5D' else 1
            
            from inno3d.models import create_model
            model = create_model(arch, in_c, out_c).to(device)
            ckpt = self.config.get('ckpt', "")
            if ckpt and os.path.exists(ckpt):
                model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            model.eval()
            
            mode = self.config['mode']
            z_idx = self.config['slice']
            stride = max(1, int(self.config.get('stride', 1)))
            z_from = int(self.config.get('z_from', 0))
            z_to = int(self.config.get('z_to', self.volume.shape[0] - 1))
            
            result_mask = np.zeros_like(self.volume, dtype=np.uint8)
            confidence = np.zeros(self.volume.shape[0], dtype=np.float32)
            
            def infer_slice(z):
                """Infer a single slice, return (prediction, mean_confidence)."""
                if in_c == 1:
                    img = self.volume[z].astype(np.float32)
                    img = (img - img.min()) / (img.max() - img.min() + 1e-5)
                    img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
                else:
                    pad = in_c // 2
                    indices = np.clip(np.arange(z - pad, z + pad + 1), 0, self.volume.shape[0] - 1)
                    img = self.volume[indices].astype(np.float32)
                    img = (img - img.min()) / (img.max() - img.min() + 1e-5)
                    img_t = torch.from_numpy(img).unsqueeze(0).to(device)
                    
                logits = model(img_t)
                probs = F.softmax(logits, dim=1)
                pr = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
                # Confidence = mean of max-class probability (higher = more certain)
                max_prob = probs.max(dim=1)[0].squeeze(0).cpu().numpy()
                conf = float(np.mean(max_prob))
                return pr, conf
            
            with torch.no_grad():
                if mode == "Single Slice (XY)":
                    pr, conf = infer_slice(z_idx)
                    result_mask[z_idx] = pr
                    confidence[z_idx] = conf

                elif mode == "Slice Range":
                    z_from = max(0, z_from)
                    z_to = min(self.volume.shape[0] - 1, z_to)
                    total = z_to - z_from + 1
                    
                    if stride <= 1:
                        # Normal: infer every slice in range
                        for i, z in enumerate(range(z_from, z_to + 1)):
                            pr, conf = infer_slice(z)
                            result_mask[z] = pr
                            confidence[z] = conf
                            self.progress.emit(i + 1, total)
                    else:
                        # Stride mode: infer keyframes, copy to neighbors
                        keyframes = list(range(z_from, z_to + 1, stride))
                        if keyframes[-1] != z_to:
                            keyframes.append(z_to)
                        
                        for i, kz in enumerate(keyframes):
                            pr, conf = infer_slice(kz)
                            result_mask[kz] = pr
                            confidence[kz] = conf
                            
                            # Fill neighbors: copy to slices between this and next keyframe
                            next_kz = keyframes[i + 1] if i + 1 < len(keyframes) else z_to + 1
                            for fill_z in range(kz + 1, min(next_kz, z_to + 1)):
                                result_mask[fill_z] = pr
                                # Neighbors get reduced confidence (further = less confident)
                                dist = abs(fill_z - kz)
                                confidence[fill_z] = conf * max(0.5, 1.0 - dist * 0.05)
                            
                            self.progress.emit(i + 1, len(keyframes))

                else:  # Full Volume
                    total = self.volume.shape[0]
                    
                    if stride <= 1:
                        # Normal: infer every slice
                        for z in range(total):
                            pr, conf = infer_slice(z)
                            result_mask[z] = pr
                            confidence[z] = conf
                            self.progress.emit(z + 1, total)
                    else:
                        # Stride mode: infer keyframes, copy to neighbors
                        keyframes = list(range(0, total, stride))
                        if keyframes[-1] != total - 1:
                            keyframes.append(total - 1)
                        
                        for i, kz in enumerate(keyframes):
                            pr, conf = infer_slice(kz)
                            result_mask[kz] = pr
                            confidence[kz] = conf
                            
                            # Fill neighbors
                            next_kz = keyframes[i + 1] if i + 1 < len(keyframes) else total
                            for fill_z in range(kz + 1, min(next_kz, total)):
                                result_mask[fill_z] = pr
                                dist = abs(fill_z - kz)
                                confidence[fill_z] = conf * max(0.5, 1.0 - dist * 0.05)
                            
                            self.progress.emit(i + 1, len(keyframes))
                        
            self.finished_infer.emit(result_mask, confidence)
        except Exception as e:
            import traceback
            self.error.emit(f"Error during inference: {str(e)}\n{traceback.format_exc()}")
            
class HistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.hist_data = None
        self.setMinimumHeight(120)
        self.setStyleSheet("background-color: #050510; border: 1px solid #444; border-radius: 4px;")
    def set_data(self, data):
        if data is None: self.hist_data = None
        else:
            mn, mx = data.min(), data.max()
            hist, _ = np.histogram(data.flatten(), bins=256, range=(mn, mx+1e-5))
            self.hist_data = hist / (hist.max() + 1e-5)
        self.update()
    def paintEvent(self, event):
        if self.hist_data is None: return
        p = QPainter(self); w, h = self.width(), self.height()
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(0, 255, 150, 150), 1))
        for i, val in enumerate(self.hist_data):
            x = int(i / 256 * w); p.drawLine(x, h, x, h - int((h-5) * val))

class AILoadVolumeThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, object)
    
    def __init__(self, file_path, raw_shape=None, raw_dtype=None):
        super().__init__()
        self.file_path = file_path
        self.raw_shape = raw_shape
        self.raw_dtype = raw_dtype
        
    def run(self):
        try:
            path = Path(self.file_path)
            
            if path.suffix.lower() in ['.raw', '.bin']:
                self.progress.emit(10, f"Loading RAW file {path.name}...")
                if not self.raw_shape or not self.raw_dtype:
                    raise ValueError("Raw shape and dtype must be provided for .raw files")
                
                dtype = np.dtype(self.raw_dtype)
                data_1d = np.fromfile(str(path), dtype=dtype)
                expected_size = self.raw_shape[0] * self.raw_shape[1] * self.raw_shape[2]
                
                if data_1d.size != expected_size:
                    raise ValueError(f"RAW file size mismatch! Expected {expected_size} pixels, got {data_1d.size}")
                    
                data = data_1d.reshape(self.raw_shape)
                self.progress.emit(100, "Complete!")
                self.finished.emit(data, None)
                return
                
            self.progress.emit(5, "Searching for TIFF files...")
            
            if path.is_dir():
                all_files = sorted(glob.glob(str(path / "*.tif*")))
                if not all_files:
                    raise ValueError(f"No .tif files found in {self.file_path}")
                
                n = len(all_files)
                # Load first to get shape
                first = io.imread(all_files[0])
                h, w = first.shape[:2]
                
                # Preallocate for speed and better memory safety
                data = np.zeros((n, h, w), dtype=first.dtype)
                data[0] = first
                
                for i in range(1, n):
                    data[i] = io.imread(all_files[i])
                    if i % 10 == 0:
                        prog = 10 + int(i / n * 85)
                        self.progress.emit(prog, f"Loading slice {i}/{n}...")
            else:
                directory = path.parent
                self.progress.emit(10, "Loading TIFF stack...")
                data = io.imread(self.file_path)
                
                if data.ndim == 2:
                    all_files = sorted(glob.glob(str(directory / "*.tif*")))
                    if len(all_files) > 1:
                        n = len(all_files)
                        # Switch to multi-file mode
                        h, w = data.shape
                        new_data = np.zeros((n, h, w), dtype=data.dtype)
                        idx_found = 0
                        for i, f in enumerate(all_files):
                            slc = io.imread(f)
                            if slc.shape == (h, w):
                                new_data[idx_found] = slc
                                idx_found += 1
                                if idx_found % 10 == 0:
                                    self.progress.emit(10 + int(idx_found/n * 85), f"Loading multi-file slice {idx_found}/{n}...")
                        data = new_data[:idx_found]
                    else:
                        data = data[np.newaxis, :, :]
            
            self.progress.emit(100, "Complete!")
            self.finished.emit(data, None)
            
        except Exception as e:
            import traceback
            self.finished.emit(None, f"{str(e)}\n{traceback.format_exc()}")

class SAMLoadThread(QThread):
    finished = pyqtSignal(object, str) # Predictor, Error
    def __init__(self, model_path):
        super().__init__(); self.model_path = model_path
    def run(self):
        try:
            mt = "vit_h"
            if "vit_l" in self.model_path.lower(): mt = "vit_l"
            elif "vit_b" in self.model_path.lower(): mt = "vit_b"
            sam = sam_model_registry[mt](checkpoint=self.model_path)
            if torch.cuda.is_available(): sam.to(device="cuda")
            self.finished.emit(SamPredictor(sam), None)
        except Exception as e: self.finished.emit(None, str(e))

class SliceViewer(QWidget):
    """Interactive slice viewer for XZ, YZ, XY slices with drawing capabilities"""
    sliceChanged = pyqtSignal(int)
    crosshairChanged = pyqtSignal(int, int, int)  # (z, y, x) world coords
    pixelClicked = pyqtSignal(int, int, int)       # (z, y, x) world coords — right-click to pick region
    
    def __init__(self, title, orientation='axial'):
        super().__init__()
        self.title = title
        self.orientation = orientation
        self.volume_data = None
        self.segmentation_data = None
        self.display_segmentation_data = None
        self.current_slice = 0
        self.max_slice = 0
        
        # Tools state
        self.active_tool = None
        self.active_class = 1
        self.class_colors = {1: (255, 255, 0), 2: (255, 0, 0)} # Default
        self.visible_classes = {1, 2} # Set of visible class IDs
        self.brush_size = 5
        self.mask_opacity = 0.4 # Default 40%
        self.zoom_factor = 1.0 # Default 1.0 (fit)
        self.threshold_mode = "Above" # "Above" or "Below"
        self.use_sam = False; self.sam_predictor = None
        self.pan_offset = QPoint(0, 0) # For panning when zoomed
        self.display_min = 0.0
        self.display_max = 255.0
        self.is_drawing = False
        self.update_callback = None
        self.img_x, self.img_y, self.img_w, self.img_h = 0, 0, 0, 0 # Track actual image in viewport
        self.sam_emb_slice = -1 # Cache for SAM embeddings
        self.view_roi = None
        self.frame_boxes = []
        self.active_frame_id = None
        self.frame_create_callback = None
        self.region_boxes = []
        self.view_origin_x, self.view_origin_y = 0, 0
        
        self.data_h, self.data_w = 0, 0
        self.mouse_pos = None
        self.last_draw_pos = None
        self.tool_points = []
        self.pending_eraser_points = []
        self.pending_brush_points = []  # Brush paint tool stroke buffer
        self.sam_positive_points = []   # SAM point-click: positive (left-click)
        self.sam_negative_points = []   # SAM point-click: negative (right-click)
        self.flood_fill_tolerance = 20  # Flood fill intensity tolerance
        self._mask_version = 0
        self._cache_key = None
        self._cached_pixmap = None
        self._scaled_cache_key = None
        self._cached_scaled_pixmap = None
        self._last_interactive_refresh = 0.0
        
        # Crosshair state (pixel coords in data space, None = hidden)
        self.show_crosshair = False
        self.crosshair_data_pos = None  # (data_x, data_y) in this view's 2D data space
        self.highlight_region_key = None  # Key of region to highlight mask
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        self.title_label = QLabel(self.title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet(f"background-color: {SemiconductorTheme.BG_MEDIUM}; color: {SemiconductorTheme.TEXT_PRIMARY}; padding: 4px; font-weight: 700; border-bottom: 2px solid {SemiconductorTheme.ACCENT_PRIMARY}; border-radius: 4px;")
        self.layout.addWidget(self.title_label)
        
        self.image_label = QLabel("No Data")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(f"background-color: {SemiconductorTheme.BG_DARK}; color: {SemiconductorTheme.TEXT_SECONDARY};")
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setMouseTracking(True)
        self.image_label.installEventFilter(self)
        self.layout.addWidget(self.image_label, 1)
        
        # SLIDER Integration
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{ height: 4px; background: {SemiconductorTheme.BORDER_DEFAULT}; }}
            QSlider::handle:horizontal {{ background: {SemiconductorTheme.ACCENT_PRIMARY}; width: 14px; margin: -5px 0; border-radius: 7px; }}
        """)
        self.slice_slider.valueChanged.connect(self.set_current_slice)
        self.layout.addWidget(self.slice_slider)
        
    def refresh_theme(self):
        from inno3d.core.styles import SemiconductorTheme
        self.title_label.setStyleSheet(f"background-color: {SemiconductorTheme.BG_MEDIUM}; color: {SemiconductorTheme.TEXT_PRIMARY}; padding: 4px; font-weight: 700; border-bottom: 2px solid {SemiconductorTheme.ACCENT_PRIMARY}; border-radius: 4px;")
        if self.volume_data is None:
            self.image_label.setStyleSheet(f"background-color: {SemiconductorTheme.BG_DARK}; color: {SemiconductorTheme.TEXT_DISABLED};")
        self.slice_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{ height: 4px; background: {SemiconductorTheme.BORDER_DEFAULT}; }}
            QSlider::handle:horizontal {{ background: {SemiconductorTheme.ACCENT_PRIMARY}; width: 14px; margin: -5px 0; border-radius: 7px; }}
        """)
        
    def set_current_slice(self, value):
        if self.current_slice != value:
            self.current_slice = value
            self.slice_slider.setValue(value)
            self.update_view()
            self.sliceChanged.emit(value)
            
    def set_volume(self, data, render_immediately=True):
        self.volume_data = data
        if data is None:
            self.image_label.setText("No Data")
            self.image_label.setPixmap(QPixmap())
            return
            
        if self.orientation == 'axial':
            self.max_slice = self.volume_data.shape[0] - 1
            self.data_h, self.data_w = self.volume_data.shape[1], self.volume_data.shape[2]
        elif self.orientation == 'coronal':
            self.max_slice = self.volume_data.shape[1] - 1
            self.data_h, self.data_w = self.volume_data.shape[0], self.volume_data.shape[2]
        elif self.orientation == 'sagittal':
            self.max_slice = self.volume_data.shape[2] - 1
            self.data_h, self.data_w = self.volume_data.shape[0], self.volume_data.shape[1]
            
        self.current_slice = self.max_slice // 2
        self.slice_slider.setRange(0, self.max_slice)
        self.slice_slider.setValue(self.current_slice)
        if render_immediately:
            self.update_view()

    def set_view_roi(self, roi):
        if self.view_roi != roi:
            self._cache_key = None  # invalidate base pixmap cache on ROI change
        self.view_roi = roi

    def set_frame_boxes(self, frame_boxes, active_frame_id=None):
        self.frame_boxes = frame_boxes or []
        self.active_frame_id = active_frame_id

    def set_region_boxes(self, region_boxes):
        self.region_boxes = region_boxes or []

    def _invalidate_scaled_cache(self):
        self._scaled_cache_key = None
        self._cached_scaled_pixmap = None

    def _request_interactive_refresh(self, force=False, min_interval=1.0 / 30.0):
        now = QDateTime.currentMSecsSinceEpoch() / 1000.0
        if force or (now - self._last_interactive_refresh) >= min_interval:
            self._last_interactive_refresh = now
            self.update_view()

    def _map_screen_pos_to_orig(self, pos):
        x_off, y_off = self.img_x, self.img_y
        p_w, p_h = self.img_w, self.img_h
        if p_w <= 0 or p_h <= 0:
            return None
        curr_w = (self.view_roi[2] - self.view_roi[0]) if self.view_roi else self.data_w
        curr_h = (self.view_roi[3] - self.view_roi[1]) if self.view_roi else self.data_h
        ox = int((pos.x() - x_off) / p_w * curr_w) + self.view_origin_x
        oy = int((pos.y() - y_off) / p_h * curr_h) + self.view_origin_y
        ox = max(0, min(self.data_w - 1, ox))
        oy = max(0, min(self.data_h - 1, oy))
        return ox, oy

    def set_zoom_factor(self, zoom_factor, anchor_pos=None):
        zoom_factor = max(0.5, min(10.0, float(zoom_factor)))
        if abs(zoom_factor - self.zoom_factor) < 1e-6:
            return
        self.zoom_factor = zoom_factor
        if self.zoom_factor <= 1.01:
            self.zoom_factor = 1.0
            self.pan_offset = QPoint(0, 0)
        self._invalidate_scaled_cache()
        self.update_view()

    def zoom_in(self):
        self.set_zoom_factor(self.zoom_factor * 1.2)

    def zoom_out(self):
        self.set_zoom_factor(self.zoom_factor / 1.2)

    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self._invalidate_scaled_cache()
        self.update_view()
        
    def update_view(self, msg=""):
        if self.volume_data is None: return
        roi = None
        
        # === LAYER 0: Base grayscale (rebuild only on slice/contrast/ROI change) ===
        base_key = (self.current_slice, id(self.volume_data), self.view_roi, self.display_min, self.display_max)
        
        if getattr(self, '_base_key', None) != base_key:
            if self.orientation == 'axial':
                slice_data = self.volume_data[self.current_slice, :, :]
            elif self.orientation == 'coronal':
                slice_data = np.ascontiguousarray(self.volume_data[:, self.current_slice, :])
            elif self.orientation == 'sagittal':
                slice_data = np.ascontiguousarray(np.transpose(self.volume_data[:, :, self.current_slice]))
            self.data_h, self.data_w = slice_data.shape
            if self.view_roi is not None:
                x1, y1, x2, y2 = self.view_roi
                x1, x2 = max(0, min(int(x1), slice_data.shape[1]-1)), max(1, min(int(x2), slice_data.shape[1]))
                y1, y2 = max(0, min(int(y1), slice_data.shape[0]-1)), max(1, min(int(y2), slice_data.shape[0]))
                roi = (x1, y1, x2, y2)
                slice_data = slice_data[y1:y2, x1:x2]
                self.view_origin_x, self.view_origin_y = x1, y1
            else:
                self.view_origin_x, self.view_origin_y = 0, 0
            d_min, d_max = self.display_min, self.display_max
            if d_max > d_min:
                s8 = np.clip((slice_data - d_min) / (d_max - d_min) * 255.0, 0, 255).astype(np.uint8)
            else:
                s8 = np.zeros_like(slice_data, dtype=np.uint8)
            h, w = s8.shape
            rgb = np.ascontiguousarray(np.stack([s8]*3, axis=-1))
            self._base_qimg = QImage(rgb.data, w, h, w*3, QImage.Format_RGB888).copy()
            self._base_key = base_key
            self._mask_cache_key = None  # force mask rebuild when base changes
        else:
            if self.view_roi is not None:
                roi = self.view_roi
                self.view_origin_x, self.view_origin_y = int(self.view_roi[0]), int(self.view_roi[1])
            else:
                self.view_origin_x, self.view_origin_y = 0, 0
        
        w, h = self._base_qimg.width(), self._base_qimg.height()
        curr_w = (self.view_roi[2] - self.view_roi[0]) if self.view_roi else self.data_w
        curr_h = (self.view_roi[3] - self.view_roi[1]) if self.view_roi else self.data_h
        
        # === LAYER 1: Mask RGBA overlay (rebuild only on mask/visibility change) ===
        m_version = getattr(self, '_mask_version', 0)
        source_mask = self.display_segmentation_data if self.display_segmentation_data is not None else self.segmentation_data
        mask_key = (self.current_slice, id(source_mask), m_version, self.mask_opacity,
                    tuple(sorted(self.visible_classes)), getattr(self, 'highlight_region_key', None), self.view_roi)
        
        if getattr(self, '_mask_cache_key', None) != mask_key:
            self._mask_overlay_img = None
            if source_mask is not None:
                if self.orientation == 'axial':
                    md = source_mask[self.current_slice, :, :]
                elif self.orientation == 'coronal':
                    md = np.ascontiguousarray(source_mask[:, self.current_slice, :])
                elif self.orientation == 'sagittal':
                    md = np.ascontiguousarray(np.transpose(source_mask[:, :, self.current_slice]))
                if self.view_roi is not None:
                    vx1, vy1 = max(0, int(self.view_roi[0])), max(0, int(self.view_roi[1]))
                    vx2, vy2 = min(md.shape[1], int(self.view_roi[2])), min(md.shape[0], int(self.view_roi[3]))
                    md = md[vy1:vy2, vx1:vx2]
                alpha_byte = int(self.mask_opacity * 255)
                ov = np.zeros((md.shape[0], md.shape[1], 4), dtype=np.uint8)
                has_vis = False
                for cls_id, color in self.class_colors.items():
                    if cls_id not in self.visible_classes: continue
                    idx = (md == cls_id)
                    if np.any(idx):
                        has_vis = True
                        ov[idx, 0] = color[0]; ov[idx, 1] = color[1]; ov[idx, 2] = color[2]; ov[idx, 3] = alpha_byte
                # Region highlighting
                if getattr(self, 'highlight_region_key', None) and has_vis:
                    rmap = getattr(self, '_highlight_labels', None)
                    rid = getattr(self, '_highlight_region_id', None)
                    rcls = getattr(self, '_highlight_class_id', None)
                    offset = getattr(self, '_highlight_offset', (0, 0, 0))
                    if rmap is not None and rid is not None and rcls is not None:
                        try:
                            if self.orientation == 'axial':
                                cz = self.current_slice - offset[0]
                                lf = np.zeros((self.volume_data.shape[1], self.volume_data.shape[2]), dtype=rmap.dtype)
                                if 0 <= cz < rmap.shape[0]:
                                    lf[offset[1]:offset[1]+rmap.shape[1], offset[2]:offset[2]+rmap.shape[2]] = rmap[cz]
                            elif self.orientation == 'coronal':
                                cy = self.current_slice - offset[1]
                                lf = np.zeros((self.volume_data.shape[0], self.volume_data.shape[2]), dtype=rmap.dtype)
                                if 0 <= cy < rmap.shape[1]:
                                    lf[offset[0]:offset[0]+rmap.shape[0], offset[2]:offset[2]+rmap.shape[2]] = rmap[:, cy, :]
                            elif self.orientation == 'sagittal':
                                cx = self.current_slice - offset[2]
                                lf = np.zeros((self.volume_data.shape[0], self.volume_data.shape[1]), dtype=rmap.dtype)
                                if 0 <= cx < rmap.shape[2]:
                                    lf[offset[0]:offset[0]+rmap.shape[0], offset[1]:offset[1]+rmap.shape[1]] = rmap[:, :, cx]
                            ls = lf[self.view_origin_y:self.view_origin_y+md.shape[0], self.view_origin_x:self.view_origin_x+md.shape[1]]
                            sel = (md == rcls) & (ls == rid)
                            other = (md > 0) & ~sel
                            ov[other, 3] = int(alpha_byte * 0.3)
                            if np.any(sel):
                                c = self.class_colors.get(rcls, (255, 200, 50))
                                ov[sel, 0] = min(255, int(c[0]*1.2)); ov[sel, 1] = min(255, int(c[1]*1.2))
                                ov[sel, 2] = min(255, int(c[2]*1.2)); ov[sel, 3] = min(255, int(alpha_byte*1.5))
                        except Exception:
                            pass
                if has_vis:
                    ov = np.ascontiguousarray(ov)
                    self._mask_overlay_img = QImage(ov.data, ov.shape[1], ov.shape[0], ov.shape[1]*4, QImage.Format_RGBA8888).copy()
            self._mask_cache_key = mask_key
            self._scaled_cache_key = None
        
        # === COMPOSITE: base + mask overlay -> pixmap ===
        pixmap = QPixmap.fromImage(self._base_qimg)
        if getattr(self, '_mask_overlay_img', None) is not None:
            p = QPainter(pixmap)
            p.setCompositionMode(QPainter.CompositionMode_SourceOver)
            p.drawImage(0, 0, self._mask_overlay_img)
            p.end()
        lbl_size = self.image_label.size()
        fit_scale = min(lbl_size.width() / max(1, w), lbl_size.height() / max(1, h))
        final_scale = fit_scale * self.zoom_factor
        
        scaled_cache_key = (
            self._base_key,
            self._mask_cache_key,
            int(w * final_scale),
            int(h * final_scale),
            round(final_scale, 4),
        )
        if self._scaled_cache_key == scaled_cache_key and self._cached_scaled_pixmap is not None:
            scaled_pixmap = self._cached_scaled_pixmap
        else:
            transform_mode = Qt.FastTransformation if (self.is_drawing or self.zoom_factor > 1.0 or self.active_tool is not None) else Qt.SmoothTransformation
            scaled_pixmap = pixmap.scaled(
                int(w * final_scale),
                int(h * final_scale),
                Qt.KeepAspectRatio,
                transform_mode
            )
            self._scaled_cache_key = scaled_cache_key
            self._cached_scaled_pixmap = scaled_pixmap
        self.img_w, self.img_h = scaled_pixmap.width(), scaled_pixmap.height()
        self.img_x = (lbl_size.width()-self.img_w)/2 + self.pan_offset.x()
        self.img_y = (lbl_size.height()-self.img_h)/2 + self.pan_offset.y()

        canvas = QPixmap(lbl_size); canvas.fill(QColor(5, 5, 10))
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing); painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(int(self.img_x), int(self.img_y), scaled_pixmap)

        curr_w = (self.view_roi[2] - self.view_roi[0]) if self.view_roi else self.data_w
        curr_h = (self.view_roi[3] - self.view_roi[1]) if self.view_roi else self.data_h
        
        if self.orientation == 'axial' and self.frame_boxes:
            for frame in self.frame_boxes:
                fx1, fy1, fx2, fy2 = frame['rect']
                color = QColor(255, 196, 0, 230) if frame['id'] == self.active_frame_id else QColor(0, 210, 255, 210)
                rx1 = self.img_x + ((fx1 - self.view_origin_x) / max(1, curr_w)) * self.img_w
                rx2 = self.img_x + ((fx2 - self.view_origin_x) / max(1, curr_w)) * self.img_w
                ry1 = self.img_y + ((fy1 - self.view_origin_y) / max(1, curr_h)) * self.img_h
                ry2 = self.img_y + ((fy2 - self.view_origin_y) / max(1, curr_h)) * self.img_h
                rect = QRectF(QPointF(rx1, ry1), QPointF(rx2, ry2)).normalized()
                if rect.width() > 2 and rect.height() > 2:
                    painter.setPen(QPen(color, 2))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRect(rect)
                    painter.setPen(QPen(color, 1))
                    painter.drawText(rect.adjusted(4, 4, -4, -4), Qt.AlignTop | Qt.AlignLeft, frame['name'])


        if self.region_boxes:
            for region in self.region_boxes:
                rx1, ry1, rx2, ry2 = region['rect']
                color = region.get('color', QColor(255, 80, 80, 220))
                if not isinstance(color, QColor):
                    color = QColor(*color)
                px1 = self.img_x + ((rx1 - self.view_origin_x) / max(1, curr_w)) * self.img_w
                px2 = self.img_x + ((rx2 - self.view_origin_x) / max(1, curr_w)) * self.img_w
                py1 = self.img_y + ((ry1 - self.view_origin_y) / max(1, curr_h)) * self.img_h
                py2 = self.img_y + ((ry2 - self.view_origin_y) / max(1, curr_h)) * self.img_h

                rect = QRectF(QPointF(px1, py1), QPointF(px2, py2)).normalized()
                if rect.width() > 2 and rect.height() > 2:
                    painter.setPen(QPen(color, 2, Qt.DashLine))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRect(rect)
                    painter.setPen(QPen(color, 1))
                    painter.drawText(rect.adjusted(4, 4, -4, -4), Qt.AlignBottom | Qt.AlignRight, region.get('label', 'REGION'))

        if self.active_tool and self.mouse_pos:
            m_px = self.mouse_pos
            if self.active_tool in ['eraser', 'brush']:
                r_px = self.brush_size * (self.img_w/max(1, self.data_w))
                color = QColor(255,100,100) if self.active_tool == 'eraser' else QColor(0,255,150)
                painter.setPen(QPen(color, 1, Qt.DashLine))
                painter.drawEllipse(m_px, r_px, r_px)
            elif self.active_tool == 'circle' and self.tool_points:
                s_px = self.tool_points[0]; radius = ((m_px.x()-s_px.x())**2 + (m_px.y()-s_px.y())**2)**0.5
                painter.setPen(QPen(QColor(0,255,150), 2)); painter.setBrush(QBrush(QColor(0,255,150,60)))
                painter.drawEllipse(s_px, radius, radius)
            elif self.active_tool == 'rect_fill' and self.tool_points:
                rect = QRectF(self.tool_points[0], m_px).normalized()
                painter.setPen(QPen(QColor(0,255,150), 1, Qt.DashLine))
                painter.setBrush(QBrush(QColor(0,255,150,40)))
                painter.drawRect(rect)
            elif self.active_tool in ['auto_roi', 'contrast_roi', 'frame_crop'] and self.tool_points:
                rect = QRectF(self.tool_points[0], m_px).normalized()
                if self.active_tool == 'contrast_roi':
                    pen_color = QColor(255,255,0); brush_color = QColor(255,255,0,30)
                elif self.active_tool == 'frame_crop':
                    pen_color = QColor(0,210,255); brush_color = QColor(0,210,255,45)
                else:
                    pen_color = QColor(255,255,255); brush_color = QColor(0,120,255,40)
                painter.setPen(QPen(pen_color, 1, Qt.DashLine))
                painter.setBrush(QBrush(brush_color))
                painter.drawRect(rect)
            elif self.active_tool == 'poly' and self.tool_points:
                path = QPainterPath(); path.moveTo(self.tool_points[0])
                for p in self.tool_points[1:]: path.lineTo(p)
                path.lineTo(m_px)
                dist = ((m_px.x()-self.tool_points[0].x())**2 + (m_px.y()-self.tool_points[0].y())**2)**0.5
                if dist < 15 and len(self.tool_points) > 2: path.closeSubpath(); painter.setBrush(QBrush(QColor(255,255,0,60)))
                else: painter.setBrush(QBrush(QColor(0,255,150,60)))
                painter.setPen(QPen(QColor(255,255,0) if dist < 15 else QColor(0,255,150), 2))
                painter.drawPath(path)
            elif self.active_tool == 'flood_fill':
                painter.setPen(QPen(QColor(0,200,255), 1, Qt.DashLine))
                painter.drawLine(int(m_px.x())-8, int(m_px.y()), int(m_px.x())+8, int(m_px.y()))
                painter.drawLine(int(m_px.x()), int(m_px.y())-8, int(m_px.x()), int(m_px.y())+8)
        # SAM point-click overlay: draw positive (green) and negative (red) points
        if self.active_tool == 'sam_click' and (self.sam_positive_points or self.sam_negative_points):
            for dx, dy in self.sam_positive_points:
                sx = self.img_x + ((dx - self.view_origin_x) / max(1, curr_w)) * self.img_w
                sy = self.img_y + ((dy - self.view_origin_y) / max(1, curr_h)) * self.img_h
                painter.setPen(QPen(QColor(0,255,0), 2)); painter.setBrush(QBrush(QColor(0,255,0,180)))
                painter.drawEllipse(QPointF(sx, sy), 5, 5)
            for dx, dy in self.sam_negative_points:
                sx = self.img_x + ((dx - self.view_origin_x) / max(1, curr_w)) * self.img_w
                sy = self.img_y + ((dy - self.view_origin_y) / max(1, curr_h)) * self.img_h
                painter.setPen(QPen(QColor(255,0,0), 2)); painter.setBrush(QBrush(QColor(255,0,0,180)))
                painter.drawEllipse(QPointF(sx, sy), 5, 5)
        # Active brush/eraser stroke preview
        if self.is_drawing and self.active_tool in ['eraser', 'brush']:
            pending = self.pending_eraser_points if self.active_tool == 'eraser' else self.pending_brush_points
            if pending:
                stroke_points = []
                for data_x, data_y in pending:
                    sx = self.img_x + ((data_x - self.view_origin_x) / max(1, curr_w)) * self.img_w
                    sy = self.img_y + ((data_y - self.view_origin_y) / max(1, curr_h)) * self.img_h
                    stroke_points.append(QPointF(sx, sy))
                if stroke_points:
                    stroke_width = max(2.0, (2 * self.brush_size + 1) * (self.img_w / max(1, curr_w)))
                    stroke_color = QColor(255, 110, 110, 180) if self.active_tool == 'eraser' else QColor(0, 255, 150, 180)
                    painter.setPen(QPen(stroke_color, stroke_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                    if len(stroke_points) == 1:
                        painter.drawPoint(stroke_points[0])
                    else:
                        painter.drawPolyline(QPolygonF(stroke_points))
        if getattr(self, 'show_crosshair', False) and getattr(self, 'crosshair_data_pos', None) is not None:
            cx_data, cy_data = self.crosshair_data_pos
            # Map data coords to screen
            cx_s = int(self.img_x + ((cx_data - self.view_origin_x) / max(1, curr_w)) * self.img_w)
            cy_s = int(self.img_y + ((cy_data - self.view_origin_y) / max(1, curr_h)) * self.img_h)
            lbl_w, lbl_h = lbl_size.width(), lbl_size.height()
            gap = 12  # pixel gap at center, matching viewer.py
            
            # Color per axis exactly like viewer.py:
            # Orientation axial:    h_axis=X(red),   v_axis=Y(green)
            # Orientation coronal:  h_axis=X(red),   v_axis=Z(blue)
            # Orientation sagittal: h_axis=Z(blue),  v_axis=Y(green)
            RED    = QColor(255, 49, 49, 220)
            GREEN  = QColor(57, 255, 20, 220)
            BLUE   = QColor(31,  81, 255, 220)
            if self.orientation == 'axial':
                h_color, v_color = RED, GREEN
                h_label, v_label = '+X', '+Y'
            elif self.orientation == 'coronal':
                h_color, v_color = RED, BLUE
                h_label, v_label = '+X', '+Z'
            else:  # sagittal
                h_color, v_color = BLUE, GREEN
                h_label, v_label = '+Z', '+Y'
            
            # Horizontal line with gap
            if cx_s > gap:
                painter.setPen(QPen(h_color, 1))
                painter.drawLine(0, cy_s, cx_s - gap, cy_s)
            painter.setPen(QPen(h_color, 1))
            painter.drawLine(cx_s + gap, cy_s, lbl_w, cy_s)
            
            # Vertical line with gap
            if cy_s > gap:
                painter.setPen(QPen(v_color, 1))
                painter.drawLine(cx_s, 0, cx_s, cy_s - gap)
            if cy_s < lbl_h - gap:
                painter.setPen(QPen(v_color, 1))
                painter.drawLine(cx_s, cy_s + gap, cx_s, lbl_h)
            
            # Axis labels at edges (matching viewer.py)
            font = painter.font(); font.setPointSize(9); font.setBold(True); painter.setFont(font)
            painter.setPen(QPen(h_color, 1))
            painter.drawText(lbl_w - 28, cy_s - 4, h_label)
            painter.setPen(QPen(v_color, 1))
            painter.drawText(cx_s + 5, 18, v_label)
        
        painter.end()
        self.image_label.setPixmap(canvas)
        text = f"{self.title} - Slice: {self.current_slice}/{self.max_slice} | Zoom: {self.zoom_factor:.1f}x | Brush: {self.brush_size}"
        if roi is not None: text += f" | ROI: {roi[0]}:{roi[2]}, {roi[1]}:{roi[3]}"
        if msg: text += f" | {msg}"
        self.title_label.setText(text)
        
    def _handle_wheel_event(self, event):
        if self.volume_data is None: return
        mods = event.modifiers()
        if mods & Qt.AltModifier:
            delta = event.angleDelta().y()
            self.brush_size = max(1, min(100, self.brush_size + (1 if delta > 0 else -1)))
            self.update_view()
            if self.update_callback: self.update_callback(brush_only=True)
            return True
        elif (mods & Qt.ControlModifier) or (mods & Qt.ShiftModifier):
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            return True
        delta = event.angleDelta().y()
        self.set_current_slice(max(0, min(self.max_slice, self.current_slice + (1 if delta > 0 else -1))))
        return True

    def wheelEvent(self, event):
        if self._handle_wheel_event(event):
            event.accept()
            return
        super().wheelEvent(event)

    def resizeEvent(self, event):
        self._invalidate_scaled_cache()
        if self.volume_data is not None: self.update_view()
        super().resizeEvent(event)

    def eventFilter(self, obj, event):
        if obj == self.image_label and self.volume_data is not None:
            if event.type() == QEvent.Wheel:
                handled = self._handle_wheel_event(event)
                if handled:
                    event.accept()
                    return True
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                if getattr(self, 'show_crosshair', False):
                    self._set_crosshair_from_pos(event.pos())
                    return True
                if self.active_tool == 'eraser':
                    self.is_drawing = True
                    self.pending_eraser_points = []
                    self.last_draw_pos = event.pos()
                    if self.update_callback: self.update_callback(save_state=True)
                    self._append_eraser_point(event.pos())
                    self._request_interactive_refresh(force=True)
                    return True
                elif self.active_tool == 'brush':
                    self.is_drawing = True
                    self.pending_brush_points = []
                    self.last_draw_pos = event.pos()
                    if self.update_callback: self.update_callback(save_state=True)
                    self._append_brush_point(event.pos())
                    self._request_interactive_refresh(force=True)
                    return True
                elif self.active_tool == 'sam_click':
                    mapped = self._map_screen_pos_to_orig(event.pos())
                    if mapped:
                        self.sam_positive_points.append(mapped)
                        self._run_sam_point_predict()
                    return True
                elif self.active_tool == 'flood_fill':
                    if self.update_callback: self.update_callback(save_state=True)
                    self._perform_flood_fill(event.pos())
                    if self.update_callback: self.update_callback(brush_only=True)
                    return True
                elif self.active_tool in ['circle', 'rect_fill', 'auto_roi', 'contrast_roi', 'frame_crop']:
                    self.tool_points = [event.pos()]; return True
                elif self.active_tool == 'poly':
                    if self.tool_points:
                        x_off, y_off = self.img_x, self.img_y
                        p_w, p_h = self.img_w, self.img_h
                        s_px = QPointF(self.tool_points[0].x()-x_off, self.tool_points[0].y()-y_off)
                        m_px = QPointF(event.pos().x()-x_off, event.pos().y()-y_off)
                        if ((m_px.x()-s_px.x())**2 + (m_px.y()-s_px.y())**2)**0.5 < 15 and len(self.tool_points) > 2:
                            self.finish_draw('poly'); return True
                    self.tool_points.append(event.pos()); self.update_view(); return True

            elif event.type() == QEvent.MouseButtonPress and (event.button() == Qt.RightButton or event.button() == Qt.MiddleButton):
                # SAM point-click: right-click adds a NEGATIVE point
                if event.button() == Qt.RightButton and self.active_tool == 'sam_click':
                    mapped = self._map_screen_pos_to_orig(event.pos())
                    if mapped:
                        self.sam_negative_points.append(mapped)
                        self._run_sam_point_predict()
                    return True
                # Right-click with no tool → pick region at pixel
                if event.button() == Qt.RightButton and self.active_tool not in ['eraser', 'brush', 'circle', 'auto_roi', 'contrast_roi', 'frame_crop', 'poly', 'rect_fill', 'flood_fill']:
                    self._emit_pixel_clicked(event.pos())
                self.last_mouse_pos = event.pos()
                return True
            
            elif event.type() == QEvent.MouseMove:
                self.mouse_pos = event.pos()
                if getattr(self, 'show_crosshair', False) and (event.buttons() & Qt.LeftButton):
                    self._set_crosshair_from_pos(event.pos())
                    return True
                if self.is_drawing and self.active_tool == 'eraser': 
                    self._append_eraser_point(event.pos()); self.last_draw_pos = event.pos(); self._request_interactive_refresh(min_interval=1.0/20.0)
                elif self.is_drawing and self.active_tool == 'brush':
                    self._append_brush_point(event.pos()); self.last_draw_pos = event.pos(); self._request_interactive_refresh(min_interval=1.0/20.0)
                elif event.buttons() & (Qt.RightButton | Qt.MiddleButton):
                    if self.active_tool != 'sam_click':
                        delta = event.pos() - self.last_mouse_pos
                        self.pan_offset += delta
                        self.last_mouse_pos = event.pos()
                        self._invalidate_scaled_cache()
                        self.update_view()
                else:
                    if self.active_tool in ['eraser', 'brush', 'circle', 'auto_roi', 'contrast_roi', 'frame_crop', 'poly', 'rect_fill', 'flood_fill', 'sam_click']:
                        self._request_interactive_refresh(min_interval=1.0/20.0)
                return True
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                if self.active_tool == 'eraser':
                    self._append_eraser_point(event.pos())
                    self._commit_eraser_stroke()
                    self.is_drawing = False
                    self.last_draw_pos = None
                    if self.update_callback:
                        self.update_callback(brush_only=True)
                    self.update_view()
                elif self.active_tool == 'brush':
                    self._append_brush_point(event.pos())
                    self._commit_brush_stroke()
                    self.is_drawing = False
                    self.last_draw_pos = None
                    if self.update_callback:
                        self.update_callback(brush_only=True)
                    self.update_view()
                elif self.active_tool == 'rect_fill' and self.tool_points:
                    if self.update_callback: self.update_callback(save_state=True)
                    self._perform_rect_fill(event.pos()); self.tool_points = []
                    if self.update_callback: self.update_callback(brush_only=True)
                elif self.active_tool in ['circle', 'auto_roi', 'contrast_roi', 'frame_crop'] and self.tool_points:
                    if self.active_tool not in ['contrast_roi', 'frame_crop'] and self.update_callback: self.update_callback(save_state=True)
                    self.perform_draw(event.pos()); self.tool_points = []
                    if self.active_tool not in ['contrast_roi', 'frame_crop'] and self.update_callback:
                        self.update_callback(brush_only=True)
                return True
            
            elif event.type() == QEvent.MouseButtonDblClick and event.button() == Qt.RightButton:
                self.reset_zoom(); return True
            elif event.type() == QEvent.Leave:
                self.mouse_pos = None; self.update_view(); return True
        return super().eventFilter(obj, event)

    def auto_contrast(self, roi=None):
        if self.volume_data is None: return
        if self.orientation == 'axial': slice_img = self.volume_data[self.current_slice]
        elif self.orientation == 'coronal': slice_img = np.ascontiguousarray(self.volume_data[:, self.current_slice, :])
        else: slice_img = np.ascontiguousarray(self.volume_data[:, :, self.current_slice])
        
        if roi:
            x1, y1, x2, y2 = roi
            crop = slice_img[y1:y2, x1:x2]
            if crop.size > 0: self.display_min, self.display_max = float(crop.min()), float(crop.max())
        else:
            self.display_min, self.display_max = float(slice_img.min()), float(slice_img.max())
            
        if self.display_max <= self.display_min: self.display_max = self.display_min + 1.0
        self.update_view()
        if self.update_callback: self.update_callback(from_viewer=True, min_max=(self.display_min, self.display_max))

    def _set_crosshair_from_pos(self, pos):
        """Convert screen pos to data coords, store in crosshair_data_pos and emit signal.
        Coronal: no flipud (raw Z increases downward in Qt).
        Sagittal: transpose means col=Z, row=Y.
        """
        if self.volume_data is None: return
        x_off, y_off = self.img_x, self.img_y
        p_w, p_h = self.img_w, self.img_h
        if p_w <= 0 or p_h <= 0: return
        curr_w = self.data_w
        curr_h = self.data_h
        ox = max(0, min(curr_w - 1, int((pos.x() - x_off) / p_w * curr_w)))
        oy = max(0, min(curr_h - 1, int((pos.y() - y_off) / p_h * curr_h)))
        
        if self.orientation == 'axial':
            # col=world_x, row=world_y
            self.crosshair_data_pos = (ox, oy)
            self.crosshairChanged.emit(self.current_slice, oy, ox)
            
        elif self.orientation == 'coronal':
            # No flipud: col=world_x, row=world_z (Z increases downward)
            world_x, world_z = ox, oy
            self.crosshair_data_pos = (world_x, world_z)
            self.crosshairChanged.emit(world_z, self.current_slice, world_x)
            
        elif self.orientation == 'sagittal':
            # Transpose: col=world_z, row=world_y
            world_z, world_y = ox, oy
            self.crosshair_data_pos = (world_z, world_y)
            self.crosshairChanged.emit(world_z, world_y, self.current_slice)
        
        self.update_view()


    def _emit_pixel_clicked(self, pos):
        """Convert screen pos to world (z, y, x) and emit pixelClicked signal."""
        if self.volume_data is None: return
        p_w, p_h = self.img_w, self.img_h
        if p_w <= 0 or p_h <= 0: return
        x_off, y_off = self.img_x, self.img_y
        ox = max(0, min(self.data_w - 1, int((pos.x() - x_off) / p_w * self.data_w)))
        oy = max(0, min(self.data_h - 1, int((pos.y() - y_off) / p_h * self.data_h)))
        # Convert display pixel → world (z, y, x)
        if self.orientation == 'axial':
            world_z, world_y, world_x = self.current_slice, oy, ox
        elif self.orientation == 'coronal':
            world_z, world_y, world_x = oy, self.current_slice, ox
        elif self.orientation == 'sagittal':
            # transpose: col=world_z, row=world_y
            world_z, world_y, world_x = ox, oy, self.current_slice
        self.pixelClicked.emit(world_z, world_y, world_x)

    def _apply_stroke_mask(self, x0, y0, x1, y1, radius, value):
        radius = max(1, int(radius))
        data_w = int(self.data_w)
        data_h = int(self.data_h)
        if data_w <= 0 or data_h <= 0:
            return

        x_min = max(0, min(x0, x1) - radius - 1)
        x_max = min(data_w, max(x0, x1) + radius + 2)
        y_min = max(0, min(y0, y1) - radius - 1)
        y_max = min(data_h, max(y0, y1) + radius + 2)
        if x_max <= x_min or y_max <= y_min:
            return

        local_h = y_max - y_min
        local_w = x_max - x_min
        try:
            import cv2
            local_mask_uint8 = np.zeros((local_h, local_w), dtype=np.uint8)
            pt1 = (x0 - x_min, y0 - y_min)
            pt2 = (x1 - x_min, y1 - y_min)
            cv2.line(local_mask_uint8, pt1, pt2, 1, thickness=radius * 2)
            cv2.circle(local_mask_uint8, pt1, radius, 1, thickness=-1)
            cv2.circle(local_mask_uint8, pt2, radius, 1, thickness=-1)
            local_mask = local_mask_uint8.astype(bool)
        except ImportError:
            local_mask = np.zeros((local_h, local_w), dtype=bool)
            rr, cc = draw.line(y0 - y_min, x0 - x_min, y1 - y_min, x1 - x_min)
            valid = (rr >= 0) & (rr < local_h) & (cc >= 0) & (cc < local_w)
            if not np.any(valid):
                return
            local_mask[rr[valid], cc[valid]] = True
            try:
                import scipy.ndimage as ndi
                footprint = disk(radius).astype(bool)
                local_mask = ndi.binary_dilation(local_mask, structure=footprint)
            except Exception:
                yy, xx = np.ogrid[:local_h, :local_w]
                for py, px in zip(rr[valid], cc[valid]):
                    local_mask |= ((yy - py) ** 2 + (xx - px) ** 2) <= (radius ** 2)

        self._apply_mask(y_min, y_max, x_min, x_max, local_mask, value)

    def _append_eraser_point(self, pos):
        mapped = self._map_screen_pos_to_orig(pos)
        if mapped is None:
            return
        if self.pending_eraser_points:
            px, py = self.pending_eraser_points[-1]
            if abs(mapped[0] - px) + abs(mapped[1] - py) < 1:
                return
        self.pending_eraser_points.append(mapped)

    def _commit_eraser_stroke(self):
        if not self.pending_eraser_points:
            return
        if len(self.pending_eraser_points) == 1:
            x0, y0 = self.pending_eraser_points[0]
            self._apply_stroke_mask(x0, y0, x0, y0, self.brush_size, 0)
            self.pending_eraser_points = []
            return

        radius = max(1, int(self.brush_size))
        xs = [pt[0] for pt in self.pending_eraser_points]
        ys = [pt[1] for pt in self.pending_eraser_points]
        x_min = max(0, min(xs) - radius - 1)
        x_max = min(int(self.data_w), max(xs) + radius + 2)
        y_min = max(0, min(ys) - radius - 1)
        y_max = min(int(self.data_h), max(ys) + radius + 2)
        if x_max <= x_min or y_max <= y_min:
            self.pending_eraser_points = []
            return

        local_h = y_max - y_min
        local_w = x_max - x_min
        try:
            import cv2
            local_mask_uint8 = np.zeros((local_h, local_w), dtype=np.uint8)
            pts = [(x - x_min, y - y_min) for x, y in zip(xs, ys)]
            # Draw thick lines for segments
            for pt1, pt2 in zip(pts[:-1], pts[1:]):
                cv2.line(local_mask_uint8, pt1, pt2, 1, thickness=radius * 2)
            # Draw rounded joints/caps
            for pt in pts:
                cv2.circle(local_mask_uint8, pt, radius, 1, thickness=-1)
            local_mask = local_mask_uint8.astype(bool)
        except ImportError:
            local_mask = np.zeros((local_h, local_w), dtype=bool)
            for (x0, y0), (x1, y1) in zip(self.pending_eraser_points[:-1], self.pending_eraser_points[1:]):
                rr, cc = draw.line(y0 - y_min, x0 - x_min, y1 - y_min, x1 - x_min)
                valid = (rr >= 0) & (rr < local_h) & (cc >= 0) & (cc < local_w)
                if np.any(valid):
                    local_mask[rr[valid], cc[valid]] = True

            try:
                import scipy.ndimage as ndi
                footprint = disk(radius).astype(bool)
                local_mask = ndi.binary_dilation(local_mask, structure=footprint)
            except Exception:
                yy, xx = np.ogrid[:local_h, :local_w]
                for x0, y0 in self.pending_eraser_points:
                    px = x0 - x_min
                    py = y0 - y_min
                    local_mask |= ((yy - py) ** 2 + (xx - px) ** 2) <= (radius ** 2)

        self._apply_mask(y_min, y_max, x_min, x_max, local_mask, 0)
        self.pending_eraser_points = []

    # ── Brush Paint Tool (inspired by DigitalSreeni annotation tool) ──
    def _append_brush_point(self, pos):
        """Append a point to the brush paint stroke buffer (same logic as eraser)."""
        mapped = self._map_screen_pos_to_orig(pos)
        if mapped is None:
            return
        if self.pending_brush_points:
            px, py = self.pending_brush_points[-1]
            if abs(mapped[0] - px) + abs(mapped[1] - py) < 1:
                return
        self.pending_brush_points.append(mapped)

    def _commit_brush_stroke(self):
        """Commit brush paint stroke — same as eraser but paints active_class instead of 0."""
        if not self.pending_brush_points:
            return
        if len(self.pending_brush_points) == 1:
            x0, y0 = self.pending_brush_points[0]
            self._apply_stroke_mask(x0, y0, x0, y0, self.brush_size, self.active_class)
            self.pending_brush_points = []
            return

        radius = max(1, int(self.brush_size))
        xs = [pt[0] for pt in self.pending_brush_points]
        ys = [pt[1] for pt in self.pending_brush_points]
        x_min = max(0, min(xs) - radius - 1)
        x_max = min(int(self.data_w), max(xs) + radius + 2)
        y_min = max(0, min(ys) - radius - 1)
        y_max = min(int(self.data_h), max(ys) + radius + 2)
        if x_max <= x_min or y_max <= y_min:
            self.pending_brush_points = []
            return

        local_h = y_max - y_min
        local_w = x_max - x_min
        try:
            import cv2
            local_mask_uint8 = np.zeros((local_h, local_w), dtype=np.uint8)
            pts = [(x - x_min, y - y_min) for x, y in zip(xs, ys)]
            for pt1, pt2 in zip(pts[:-1], pts[1:]):
                cv2.line(local_mask_uint8, pt1, pt2, 1, thickness=radius * 2)
            for pt in pts:
                cv2.circle(local_mask_uint8, pt, radius, 1, thickness=-1)
            local_mask = local_mask_uint8.astype(bool)
        except ImportError:
            local_mask = np.zeros((local_h, local_w), dtype=bool)
            for (x0, y0), (x1, y1) in zip(self.pending_brush_points[:-1], self.pending_brush_points[1:]):
                rr, cc = draw.line(y0 - y_min, x0 - x_min, y1 - y_min, x1 - x_min)
                valid = (rr >= 0) & (rr < local_h) & (cc >= 0) & (cc < local_w)
                if np.any(valid):
                    local_mask[rr[valid], cc[valid]] = True
            try:
                import scipy.ndimage as ndi
                footprint = disk(radius).astype(bool)
                local_mask = ndi.binary_dilation(local_mask, structure=footprint)
            except Exception:
                yy, xx = np.ogrid[:local_h, :local_w]
                for x0, y0 in self.pending_brush_points:
                    px = x0 - x_min
                    py = y0 - y_min
                    local_mask |= ((yy - py) ** 2 + (xx - px) ** 2) <= (radius ** 2)

        self._apply_mask(y_min, y_max, x_min, x_max, local_mask, self.active_class)
        self.pending_brush_points = []

    # ── SAM Point-Click Refinement (inspired by DigitalSreeni annotation tool) ──
    def _run_sam_point_predict(self):
        """Run SAM prediction using collected positive/negative point prompts.
        Left-click = positive (include), Right-click = negative (exclude).
        """
        if self.sam_predictor is None or self.segmentation_data is None:
            return
        if not self.sam_positive_points and not self.sam_negative_points:
            return
        try:
            # Ensure embedding is set for current slice
            if self.orientation == 'axial':
                slice_img = self.volume_data[self.current_slice]
            elif self.orientation == 'coronal':
                slice_img = np.ascontiguousarray(self.volume_data[:, self.current_slice, :])
            else:
                slice_img = np.ascontiguousarray(np.transpose(self.volume_data[:, :, self.current_slice]))

            if self.sam_emb_slice != self.current_slice:
                mn, mx = slice_img.min(), slice_img.max()
                img_8 = ((slice_img - mn) / (mx - mn + 1e-5) * 255).astype(np.uint8)
                self.sam_predictor.set_image(np.stack([img_8] * 3, axis=-1))
                self.sam_emb_slice = self.current_slice

            # Build point coords and labels (1=positive, 0=negative)
            all_points = []
            all_labels = []
            for (px, py) in self.sam_positive_points:
                all_points.append([px, py])
                all_labels.append(1)
            for (px, py) in self.sam_negative_points:
                all_points.append([px, py])
                all_labels.append(0)

            point_coords = np.array(all_points, dtype=np.float32)
            point_labels = np.array(all_labels, dtype=np.int32)

            masks, scores, _ = self.sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            # Pick the best mask (highest score)
            best_idx = int(np.argmax(scores))
            mask = masks[best_idx]

            # Show as temporary preview via display_segmentation_data
            if self.display_segmentation_data is None:
                self.display_segmentation_data = self.segmentation_data.copy()
            # Apply preview: restore base first, then overlay SAM mask
            if self.orientation == 'axial':
                self.display_segmentation_data[self.current_slice] = self.segmentation_data[self.current_slice].copy()
                self.display_segmentation_data[self.current_slice][mask] = self.active_class
            elif self.orientation == 'coronal':
                self.display_segmentation_data[:, self.current_slice, :] = self.segmentation_data[:, self.current_slice, :].copy()
                self.display_segmentation_data[:, self.current_slice, :][mask] = self.active_class
            else:
                self.display_segmentation_data[:, :, self.current_slice] = self.segmentation_data[:, :, self.current_slice].copy()
                self.display_segmentation_data[:, :, self.current_slice][mask.T] = self.active_class
            self._mask_version += 1
            self._mask_cache_key = None
            self._scaled_cache_key = None
            self.update_view("SAM Preview (Enter to accept, Esc to cancel)")
        except Exception as e:
            print(f"SAM Point-Click Error: {e}")
            self.update_view(f"SAM Error: {e}")

    def sam_click_accept(self):
        """Accept current SAM point-click preview → commit to segmentation."""
        if self.display_segmentation_data is not None:
            # Save local reference because update_callback (sync) will overwrite self.display_segmentation_data to None
            preview_data = self.display_segmentation_data
            if self.update_callback: self.update_callback(save_state=True)
            if self.orientation == 'axial':
                self.segmentation_data[self.current_slice] = preview_data[self.current_slice].copy()
            elif self.orientation == 'coronal':
                self.segmentation_data[:, self.current_slice, :] = preview_data[:, self.current_slice, :].copy()
            else:
                self.segmentation_data[:, :, self.current_slice] = preview_data[:, :, self.current_slice].copy()
            self.display_segmentation_data = None
        self.sam_positive_points = []
        self.sam_negative_points = []
        self._mask_version += 1
        self._mask_cache_key = None
        self.update_view("SAM: Accepted")
        if self.update_callback: self.update_callback(brush_only=True)

    def sam_click_cancel(self):
        """Cancel SAM point-click preview → discard."""
        self.display_segmentation_data = None
        self.sam_positive_points = []
        self.sam_negative_points = []
        self._mask_version += 1
        self._mask_cache_key = None
        self.update_view("SAM: Cancelled")

    # ── Flood Fill Tool (Magic Wand — inspired by DigitalSreeni) ──
    def _perform_flood_fill(self, pos):
        """Flood fill connected region with similar intensity starting from clicked pixel."""
        if self.volume_data is None or self.segmentation_data is None:
            return
        mapped = self._map_screen_pos_to_orig(pos)
        if mapped is None:
            return
        ox, oy = mapped

        if self.orientation == 'axial':
            slice_img = self.volume_data[self.current_slice]
        elif self.orientation == 'coronal':
            slice_img = np.ascontiguousarray(self.volume_data[:, self.current_slice, :])
        else:
            slice_img = np.ascontiguousarray(np.transpose(self.volume_data[:, :, self.current_slice]))

        seed_val = float(slice_img[oy, ox])
        tolerance = self.flood_fill_tolerance

        try:
            import cv2
            # OpenCV floodFill requires padded mask
            h, w = slice_img.shape[:2]
            fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            img_f = slice_img.astype(np.float32)
            # Normalize to 0-255 for flood fill
            mn, mx = img_f.min(), img_f.max()
            if mx > mn:
                img_norm = ((img_f - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                img_norm = np.zeros_like(img_f, dtype=np.uint8)
            tol = int(tolerance)
            cv2.floodFill(img_norm, fill_mask, (ox, oy), 255, (tol,) * (1 if img_norm.ndim == 2 else 3), (tol,) * (1 if img_norm.ndim == 2 else 3), cv2.FLOODFILL_MASK_ONLY)
            region = fill_mask[1:-1, 1:-1].astype(bool)
        except ImportError:
            # Fallback: simple threshold-based region
            low = seed_val - tolerance
            high = seed_val + tolerance
            region = (slice_img >= low) & (slice_img <= high)
            # Connected component: keep only the component containing the seed
            try:
                import scipy.ndimage as ndi
                labeled, n = ndi.label(region)
                seed_label = labeled[oy, ox]
                if seed_label > 0:
                    region = (labeled == seed_label)
                else:
                    return
            except ImportError:
                pass  # Use raw threshold region as fallback

        if np.any(region):
            y_idx, x_idx = np.where(region)
            y1, y2 = int(y_idx.min()), int(y_idx.max()) + 1
            x1, x2 = int(x_idx.min()), int(x_idx.max()) + 1
            self._apply_mask(y1, y2, x1, x2, region[y1:y2, x1:x2], self.active_class)
            self.update_view()

    # ── Rectangle Fill Tool ──
    def _perform_rect_fill(self, pos):
        """Fill a solid rectangle with the active class."""
        if self.volume_data is None or self.segmentation_data is None or not self.tool_points:
            return
        x_off, y_off = self.img_x, self.img_y
        p_w, p_h = self.img_w, self.img_h
        if p_w <= 0:
            return
        curr_w = (self.view_roi[2] - self.view_roi[0]) if self.view_roi else self.data_w
        curr_h = (self.view_roi[3] - self.view_roi[1]) if self.view_roi else self.data_h

        def map_to_orig(p):
            return int((p.x() - x_off) / p_w * curr_w) + self.view_origin_x, int((p.y() - y_off) / p_h * curr_h) + self.view_origin_y

        s_ox, s_oy = map_to_orig(self.tool_points[0])
        ox, oy = map_to_orig(pos)
        x1, x2 = sorted([s_ox, ox])
        y1, y2 = sorted([s_oy, oy])
        x1, x2 = max(0, x1), min(self.data_w, x2)
        y1, y2 = max(0, y1), min(self.data_h, y2)
        if x2 > x1 and y2 > y1:
            mask = np.ones((y2 - y1, x2 - x1), dtype=bool)
            self._apply_mask(y1, y2, x1, x2, mask, self.active_class)
            self.update_view()

    def perform_draw(self, pos):
        if self.volume_data is None: return
        x_off, y_off = self.img_x, self.img_y; p_w, p_h = self.img_w, self.img_h
        if p_w <= 0: return

        curr_w = (self.view_roi[2] - self.view_roi[0]) if self.view_roi else self.data_w
        curr_h = (self.view_roi[3] - self.view_roi[1]) if self.view_roi else self.data_h
        def map_to_orig(p): return int((p.x()-x_off)/p_w*curr_w) + self.view_origin_x, int((p.y()-y_off)/p_h*curr_h) + self.view_origin_y

        ox, oy = map_to_orig(pos); val = self.active_class if self.active_tool != 'eraser' else 0
        
        if self.active_tool == 'circle' and self.tool_points:
            s_ox, s_oy = map_to_orig(self.tool_points[0]); radius = int(((ox-s_ox)**2+(oy-s_oy)**2)**0.5)
            y_min, y_max, x_min, x_max = max(0, s_oy-radius), min(self.data_h, s_oy+radius+1), max(0, s_ox-radius), min(self.data_w, s_ox+radius+1)
            YY, XX = np.ogrid[y_min-s_oy:y_max-s_oy, x_min-s_ox:x_max-s_ox]
            self._apply_mask(y_min, y_max, x_min, x_max, XX**2+YY**2 <= radius**2, val)
        elif self.active_tool in ['auto_roi', 'contrast_roi', 'frame_crop'] and self.tool_points:
            s_ox, s_oy = map_to_orig(self.tool_points[0]); x1, x2 = sorted([s_ox, ox]); y1, y2 = sorted([s_oy, oy])
            x1, x2, y1, y2 = max(0, x1), min(self.data_w, x2), max(0, y1), min(self.data_h, y2)
            if x2-x1 > 2 and y2-y1 > 2:
                if self.active_tool == 'contrast_roi':
                    self.auto_contrast(roi=(x1, y1, x2, y2))
                    return
                if self.active_tool == 'frame_crop':
                    if self.orientation == 'axial' and self.frame_create_callback is not None:
                        self.frame_create_callback((x1, y1, x2, y2), self.current_slice)
                        self.update_view("FRAME ADDED")
                    return
                if self.orientation == 'axial': slice_img = self.volume_data[self.current_slice]
                elif self.orientation == 'coronal': slice_img = np.ascontiguousarray(self.volume_data[:, self.current_slice, :])
                else: slice_img = np.ascontiguousarray(np.transpose(self.volume_data[:, :, self.current_slice]))
                
                if self.use_sam and self.sam_predictor is not None:
                    self.update_view("AI: PROCESSING..."); QApplication.setOverrideCursor(Qt.WaitCursor); QApplication.processEvents()
                    try:
                        if self.sam_emb_slice != self.current_slice:
                            self.update_view("AI: EMBEDDING..."); QApplication.processEvents()
                            mn, mx = slice_img.min(), slice_img.max(); img_8 = ((slice_img-mn)/(mx-mn+1e-5)*255).astype(np.uint8)
                            self.sam_predictor.set_image(np.stack([img_8]*3, axis=-1)); self.sam_emb_slice = self.current_slice
                        masks, _, _ = self.sam_predictor.predict(box=np.array([x1, y1, x2, y2]), multimask_output=False); mask_f = masks[0]
                        m_roi = np.zeros_like(mask_f, dtype=bool); m_roi[y1:y2, x1:x2] = mask_f[y1:y2, x1:x2]
                        if self.orientation == 'axial': self.segmentation_data[self.current_slice][m_roi] = self.active_class
                        elif self.orientation == 'coronal': self.segmentation_data[:, self.current_slice, :][m_roi] = self.active_class
                        else: self.segmentation_data[:, :, self.current_slice][m_roi.T] = self.active_class
                    except Exception as e: print(f"SAM Error: {e}")
                    finally: QApplication.restoreOverrideCursor(); self.update_view("AI: DONE")
                else:
                    try:
                        crop = slice_img[y1:y2, x1:x2]; t = filters.threshold_otsu(crop)
                        mask = (crop > t) if self.threshold_mode == "Above" else (crop < t)
                        self._apply_mask(y1, y2, x1, x2, mask, self.active_class)
                    except: pass
        self.update_view()

    def finish_draw(self, t):
        self._mask_version += 1
        if t == 'poly' and len(self.tool_points) >= 3:
            if self.update_callback: self.update_callback(save_state=True)
            x_off, y_off = self.img_x, self.img_y
            p_w, p_h = self.img_w, self.img_h
            curr_w = (self.view_roi[2] - self.view_roi[0]) if self.view_roi else self.data_w
            curr_h = (self.view_roi[3] - self.view_roi[1]) if self.view_roi else self.data_h
            coords = [(int((p.y()-y_off)/p_h*curr_h) + self.view_origin_y, int((p.x()-x_off)/p_w*curr_w) + self.view_origin_x) for p in self.tool_points]

            rr, cc = draw.polygon([c[0] for c in coords], [c[1] for c in coords], shape=(self.data_h, self.data_w))
            if len(rr) == 0:
                self.tool_points = []
                self.update_view()
                return
            if self.orientation == 'axial': 
                self.segmentation_data[self.current_slice, rr, cc] = self.active_class
                if self.display_segmentation_data is not None: self.display_segmentation_data[self.current_slice, rr, cc] = self.active_class
            elif self.orientation == 'coronal': 
                self.segmentation_data[rr, self.current_slice, cc] = self.active_class
                if self.display_segmentation_data is not None: self.display_segmentation_data[rr, self.current_slice, cc] = self.active_class
            elif self.orientation == 'sagittal': 
                self.segmentation_data[rr, cc, self.current_slice] = self.active_class
                if self.display_segmentation_data is not None: self.display_segmentation_data[rr, cc, self.current_slice] = self.active_class
            self.tool_points = []
            # Force mask cache invalidation so the new polygon pixels are visible
            self._mask_cache_key = None
            self._scaled_cache_key = None
            self.update_view()
            if self.update_callback: self.update_callback(brush_only=True)

    def _apply_stroke_mask(self, x0, y0, x1, y1, radius, val):
        """Apply a single brush circle at (x0, y0) with given radius and value."""
        r = max(1, int(radius))
        xmin = max(0, x0 - r)
        xmax = min(int(self.data_w), x0 + r + 1)
        ymin = max(0, y0 - r)
        ymax = min(int(self.data_h), y0 + r + 1)
        if xmax <= xmin or ymax <= ymin:
            return
        YY, XX = np.ogrid[ymin - y0:ymax - y0, xmin - x0:xmax - x0]
        circle_mask = (XX ** 2 + YY ** 2) <= (r ** 2)
        self._apply_mask(ymin, ymax, xmin, xmax, circle_mask, val)

    def _apply_mask(self, y1, y2, x1, x2, mask, val):
        self._mask_version += 1
        self._mask_cache_key = None  # force mask overlay rebuild
        self._scaled_cache_key = None
        if self.orientation == 'axial': 
            self.segmentation_data[self.current_slice, y1:y2, x1:x2][mask] = val
            if self.display_segmentation_data is not None: 
                self.display_segmentation_data[self.current_slice, y1:y2, x1:x2][mask] = val
                
        elif self.orientation == 'coronal': 
            self.segmentation_data[y1:y2, self.current_slice, x1:x2][mask] = val
            if self.display_segmentation_data is not None: 
                self.display_segmentation_data[y1:y2, self.current_slice, x1:x2][mask] = val
                
        elif self.orientation == 'sagittal': 
            self.segmentation_data[x1:x2, y1:y2, self.current_slice][mask.T] = val
            if self.display_segmentation_data is not None: 
                self.display_segmentation_data[x1:x2, y1:y2, self.current_slice][mask.T] = val

class AI3DTab(QWidget):
    """Tab 3D AI with Toggle Visibility, Slice Sliders and working Auto Labelling"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Prevent accidental wheel scrolling on all spinboxes/comboboxes globally
        self._wheel_filter = NoScrollEventFilter()
        app = QApplication.instance()
        if app:
            app.installEventFilter(self._wheel_filter)
            
        self.volume_data = None
        self.segmentation_data = None
        self.classes = {1: {"name": "Class 1", "color": (255, 255, 0), "visible": True}, 2: {"name": "Class 2", "color": (255, 0, 0), "visible": True}}
        self.active_tool, self.current_class_id, self.brush_size = None, 1, 5
        self.mask_opacity = 0.4
        self.threshold_mode = "Above"
        self.auto_mode = "Otsu" # "Otsu" or "SAM"
        self.sam_predictor = None
        self.display_min, self.display_max = 0.0, 255.0
        self.undo_stack, self.redo_stack = [], []
        self.crop_frames = []
        self.next_frame_id = 1
        self.active_frame_id = None
        self.frame_split = "train"
        self.current_source_path = ""
        self.display_segmentation_data = None
        self.sources = []
        self.active_source_id = None
        self.region_index = []
        self.region_label_maps = {}
        self.hidden_regions = set()
        self.active_region_key = None
        # For very large volumes, disable connected-component region indexing/highlights.
        self.enable_connected_components = False
        self.region_refresh_timer = QTimer(self)
        self.region_refresh_timer.setSingleShot(True)
        self.region_refresh_timer.timeout.connect(self.refresh_region_index)
        self.init_ui()
        self.setAcceptDrops(True)
        # Initial build of per-class weight spinboxes
        self._rebuild_class_weight_spins()

    def set_current_slice(self, value):
        """Navigate XY viewer to a specific slice (used by ActiveLearning goto_slice)."""
        if hasattr(self, 'xy_view') and self.xy_view is not None:
            self.xy_view.set_current_slice(value)
            self.sync(brush_only=True, source_viewer=self.xy_view)

    def _load_ui_icon(self, icon_name, fallback_standard_icon=None):
        icon_path = Path(__file__).resolve().parents[1] / "assets" / "icons" / icon_name
        if icon_path.exists():
            return QIcon(str(icon_path))
        if fallback_standard_icon is not None:
            return self.style().standardIcon(fallback_standard_icon)
        return QIcon()
        
    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # === QSplitter: resizable view + sidebar ===
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(5)
        self.main_splitter.setStyleSheet(
            "QSplitter::handle { background: #2a3a4a; }"
            "QSplitter::handle:hover { background: #00e5ff; }"
        )

        # === LEFT: View Area ===
        view_container = QWidget()
        view_layout = QVBoxLayout(view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(0)

        # Hidden proxy buttons for sidebar cross-reference
        self.btn_xy_only = QPushButton("XY Only")
        self.btn_xy_only.setCheckable(True); self.btn_xy_only.setChecked(True)
        self.btn_xy_only.setVisible(False)
        self.btn_xy_only.clicked.connect(lambda: self.set_view_mode(0))

        self.btn_multi_view = QPushButton("Multi-Planar")
        self.btn_multi_view.setCheckable(True); self.btn_multi_view.setVisible(False)
        self.btn_multi_view.clicked.connect(lambda: self.set_view_mode(1))

        self.btn_crosshair = QPushButton("Crosshair")
        self.btn_crosshair.setCheckable(True); self.btn_crosshair.setVisible(False)
        self.btn_crosshair.clicked.connect(self.toggle_crosshair)

        # btn_show_frames: controlled from sidebar SHOW FRAMES toggle (no visual in view area)
        self.btn_show_frames = QPushButton()
        self.btn_show_frames.setCheckable(True); self.btn_show_frames.setChecked(True)
        self.btn_show_frames.setVisible(False)
        self.btn_show_frames.clicked.connect(self.sync)

        # MPR refresh button: other views only render when user clicks this
        self.btn_refresh_mpr = QPushButton("\U0001f504 Sync 3D Views")
        self.btn_refresh_mpr.setVisible(False)  # controlled from sidebar
        self.btn_refresh_mpr.clicked.connect(self.refresh_mpr_views)
        self._mpr_dirty = True  # flag: XZ/YZ need refresh


        self.v_stack = QStackedWidget()
        self.v_stack.setContentsMargins(0, 0, 0, 0)
        
        # Mode 0: XY Only
        self.single_view_container = QWidget()
        single_layout = QVBoxLayout(self.single_view_container)
        single_layout.setContentsMargins(0, 0, 0, 0)
        self.xy_view = SliceViewer("XY (Axial)", "axial")
        single_layout.addWidget(self.xy_view)
        self.v_stack.addWidget(self.single_view_container)

        # Mode 1: Multi-Planar
        self.multi_view_container = QWidget()
        self.multi_layout = QGridLayout(self.multi_view_container)
        self.multi_layout.setSpacing(2)
        self.multi_layout.setContentsMargins(0, 0, 0, 0)
        
        self.yz_view = SliceViewer("YZ (Sagittal)", "sagittal")
        self.xz_view = SliceViewer("XZ (Coronal)", "coronal")
        self.region_panel = self._create_region_index_panel()
        if not self.enable_connected_components:
            self.region_panel.setVisible(False)
        
        self.multi_layout.addWidget(self.yz_view, 0, 1)
        self.multi_layout.addWidget(self.xz_view, 1, 0)
        self.multi_layout.addWidget(self.region_panel, 1, 1)
        self.v_stack.addWidget(self.multi_view_container)

        # Mode 2: XZ Only (Coronal)
        self.xz_single_container = QWidget()
        xz_single_layout = QVBoxLayout(self.xz_single_container)
        xz_single_layout.setContentsMargins(0, 0, 0, 0)
        self.v_stack.addWidget(self.xz_single_container)

        # Mode 3: YZ Only (Sagittal)
        self.yz_single_container = QWidget()
        yz_single_layout = QVBoxLayout(self.yz_single_container)
        yz_single_layout.setContentsMargins(0, 0, 0, 0)
        self.v_stack.addWidget(self.yz_single_container)

        view_layout.addWidget(self.v_stack)
        self.main_splitter.addWidget(view_container)

        for v in [self.xy_view, self.yz_view, self.xz_view]:
            v.sliceChanged.connect(self.on_slice_synced)
            v.crosshairChanged.connect(self.on_crosshair_synced)
            v.pixelClicked.connect(self.on_pixel_clicked)

        self.all_viewers = [self.xy_view, self.yz_view, self.xz_view]
        self.isolate_roi = False

        # === RIGHT: Control Tabs (resizable sidebar) ===
        self.ctrl = QTabWidget()
        self.ctrl.setMinimumWidth(320)
        self.ctrl.setMaximumWidth(800)
        self.main_splitter.addWidget(self.ctrl)
        self.ctrl.addTab(self._create_preprocessing_tab(), "Pre-processing")
        self.ctrl.addTab(self._create_labelling_tab(), "Labelling")
        self.ctrl.addTab(self._create_data_tab(), "AI Data")
        self.ctrl.addTab(self._create_model_tab(), "Model")

        # Set initial splitter ratio (65% view : 35% sidebar)
        self.main_splitter.setStretchFactor(0, 65)
        self.main_splitter.setStretchFactor(1, 35)
        main_layout.addWidget(self.main_splitter)

    def refresh_theme(self):
        for v in getattr(self, 'all_viewers', []):
            if hasattr(v, 'refresh_theme'):
                v.refresh_theme()
        if hasattr(self, 'region_table'):
            self._style_dark_table(self.region_table)
        if hasattr(self, 'mother_table'):
            self._style_dark_table(self.mother_table)
        if hasattr(self, 'frame_table'):
            self._style_dark_table(self.frame_table)

    def get_visible_viewers(self):
        mode = self.v_stack.currentIndex()
        if mode == 0:
            return [self.xy_view]
        elif mode == 1:
            return [self.xy_view, self.xz_view, self.yz_view]
        elif mode == 2:
            return [self.xz_view]
        elif mode == 3:
            return [self.yz_view]
        return [self.xy_view]

    def zoom_in_view(self):
        for viewer in self.get_visible_viewers():
            viewer.zoom_in()

    def zoom_out_view(self):
        for viewer in self.get_visible_viewers():
            viewer.zoom_out()

    def reset_view_zoom(self):
        for viewer in self.get_visible_viewers():
            viewer.reset_zoom()

    def set_view_mode(self, mode_index):
        """Switch view: 0=XY only, 1=Multi-Planar, 2=XZ only, 3=YZ only."""
        # Return viewers to their home containers before switching
        if mode_index == 0:
            self.single_view_container.layout().addWidget(self.xy_view)
        elif mode_index == 1:
            self.multi_layout.addWidget(self.xy_view, 0, 0)
            self.multi_layout.addWidget(self.yz_view, 0, 1)
            self.multi_layout.addWidget(self.xz_view, 1, 0)
        elif mode_index == 2:
            self.xz_single_container.layout().addWidget(self.xz_view)
        elif mode_index == 3:
            self.yz_single_container.layout().addWidget(self.yz_view)

        self.v_stack.setCurrentIndex(mode_index)
        self.sync()

        # Auto-refresh the focused non-XY viewer
        if mode_index in (1, 2, 3):
            self.refresh_mpr_views()

    def toggle_crosshair(self, state):
        for v in self.all_viewers:
            v.show_crosshair = state
        for v in self.get_visible_viewers():
            v.update_view()

    def refresh_mpr_views(self):
        """Manually refresh XZ/YZ multi-planar views. Called by MPR Refresh button."""
        for v in [self.xz_view, self.yz_view]:
            if v.volume_data is not None:
                v._base_key = None  # force full rebuild
                v._mask_cache_key = None
                v.update_view()
        self._mpr_dirty = False
        self._set_mpr_btn_dirty(False)

    def _set_mpr_btn_dirty(self, dirty):
        """Update the sidebar MPR Refresh button style to indicate dirty state."""
        p = self.parent()
        for _ in range(10):
            if p is None: break
            if hasattr(p, 'ai_btn_mpr_refresh'):
                if dirty:
                    p.ai_btn_mpr_refresh.setStyleSheet("background-color: #FF6600; color: white; font-weight: bold; border-radius: 4px; padding: 4px 8px;")
                else:
                    p.ai_btn_mpr_refresh.setStyleSheet("")  # Reset to default theme
                break
            p = p.parent()

    def on_pixel_clicked(self, world_z, world_y, world_x):
        """Right-click on any view -> find the region at that world coord and select it in the table."""
        if not self.enable_connected_components:
            return
        if self.segmentation_data is None: return
        Z, Y, X = self.segmentation_data.shape
        if not (0 <= world_z < Z and 0 <= world_y < Y and 0 <= world_x < X): return
        
        class_id = int(self.segmentation_data[world_z, world_y, world_x])
        if class_id == 0:
            self.active_region_key = None
            self._clear_region_highlight()
            if hasattr(self, 'region_table'):
                self.region_table.clearSelection()
            self.sync(brush_only=True)
            return
        
        labels = self.region_label_maps.get(class_id)
        if labels is None: return
        region_id = int(labels[world_z, world_y, world_x])
        if region_id == 0: return
        key = (class_id, region_id)
        self.active_region_key = key
        
        # Find the region's centroid Z from bbox so slice-filter shows it
        target_region = next((r for r in self.region_index if r['key'] == key), None)
        if target_region is not None:
            z1, y1, x1, z2, y2, x2 = target_region['bbox']
            cz = min(max(0, (z1 + z2) // 2), self.xy_view.max_slice)
            if self.xy_view.current_slice != cz:
                self.xy_view.set_current_slice(cz)
        
        # Refresh filtered table at new slice
        self.refresh_region_table()
        
        if not hasattr(self, 'region_table'): return
        # Find and select the matching row
        self.region_table.blockSignals(True)
        found = False
        for row in range(self.region_table.rowCount()):
            item = self.region_table.item(row, 0)
            if item and item.data(Qt.UserRole) == key:
                self.region_table.selectRow(row)
                self.region_table.scrollToItem(item)
                found = True
                break
        self.region_table.blockSignals(False)
        if found:
            self.on_region_selection_changed()




    def on_crosshair_synced(self, z, y, x):
        """Called when any viewer's crosshair is dragged/clicked. Sync all 3 views' slice and crosshair."""
        if self.volume_data is None: return

        # Update slice positions
        self.xy_view.blockSignals(True)
        self.xz_view.blockSignals(True)
        self.yz_view.blockSignals(True)
        
        if self.xy_view.current_slice != z: self.xy_view.set_current_slice(z)
        if self.xz_view.current_slice != y: self.xz_view.set_current_slice(y)
        if self.yz_view.current_slice != x: self.yz_view.set_current_slice(x)
        
        self.xy_view.blockSignals(False)
        self.xz_view.blockSignals(False)
        self.yz_view.blockSignals(False)

        # Update each viewer's crosshair_data_pos in DISPLAY space:
        # xy_view (axial):    col=world_x, row=world_y
        self.xy_view.crosshair_data_pos = (x, y)
        # xz_view (coronal):  col=world_x, row=world_z  (no flipud: Z increases downward)
        self.xz_view.crosshair_data_pos = (x, z)
        # yz_view (sagittal): col=world_z, row=world_y  (transpose: Z is horizontal)
        self.yz_view.crosshair_data_pos = (z, y)
        for v in self.all_viewers:
            v.update_view()




    def _style_dark_table(self, table):
        from inno3d.core.styles import SemiconductorTheme
        table.setAlternatingRowColors(True)
        table.setStyleSheet(SemiconductorTheme.table_stylesheet())

    def _create_region_index_panel(self):
        panel = QGroupBox("Label Region Index")
        layout = QVBoxLayout(panel)
        self.lbl_region_info = QLabel("Regions are extracted per class. Select one region to map it on XY, YZ, XZ.")
        self.lbl_region_info.setWordWrap(True)
        self.lbl_region_info.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;")
        layout.addWidget(self.lbl_region_info)

        self.region_table = QTableWidget(0, 7)
        self.region_table.setHorizontalHeaderLabels(["Hide\n/Unhide", "Region", "Class", "Voxels", "XY", "YZ", "XZ"])
        self.region_table.verticalHeader().setVisible(False)
        self.region_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.region_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.region_table.itemSelectionChanged.connect(self.on_region_selection_changed)
        self.region_table.itemChanged.connect(self.on_region_item_changed)
        self._style_dark_table(self.region_table)
        layout.addWidget(self.region_table, 1)

        btn_row = QHBoxLayout()
        self.btn_region_refresh = QPushButton("Refresh Index")
        self.btn_region_refresh.clicked.connect(self.refresh_region_index)
        self.btn_region_delete = QPushButton("Delete Region")
        self.btn_region_delete.clicked.connect(self.delete_selected_region)
        self.btn_region_show_all = QPushButton("Show All")
        self.btn_region_show_all.clicked.connect(self.show_all_regions)
        btn_row.addWidget(self.btn_region_refresh)
        btn_row.addWidget(self.btn_region_delete)
        btn_row.addWidget(self.btn_region_show_all)
        layout.addLayout(btn_row)
        return panel
        
    def _create_labelling_tab(self):
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        w = QWidget()
        l = QVBoxLayout(w)
        l.setSpacing(12)
        l.setContentsMargins(4, 8, 4, 8)

        gm = QGroupBox("Source Volume Info")
        lm = QVBoxLayout()
        self.mother_table = QTableWidget(0, 6)
        self.mother_table.setHorizontalHeaderLabels(["Idx", "Source", "Shape", "ROIs", "Split", "Active Slice"])
        self.mother_table.verticalHeader().setVisible(False)
        self.mother_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.mother_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.mother_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.mother_table.setMaximumHeight(160)
        
        self.mother_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.mother_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.mother_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.mother_table.itemSelectionChanged.connect(self.on_mother_table_selection)
        
        self._style_dark_table(self.mother_table)
        lm.addWidget(self.mother_table)
        
        h_src_act = QHBoxLayout()
        h_src_act.addWidget(QLabel("Set Split for Selected Sources:"))
        self.combo_src_split = QComboBox()
        self.combo_src_split.addItems(["none", "train", "val", "test", "mix", "monitoring"])
        h_src_act.addWidget(self.combo_src_split)
        self.btn_src_split = QPushButton("Apply Split")
        self.btn_src_split.clicked.connect(self.apply_source_split)
        h_src_act.addWidget(self.btn_src_split)
        
        self.btn_src_delete = QPushButton("Delete Selected")
        self.btn_src_delete.setIcon(self._load_ui_icon("trash.svg"))
        self.btn_src_delete.setStyleSheet("color: #ff5252; font-weight: bold;")
        self.btn_src_delete.setToolTip("Remove selected source volumes from the list")
        self.btn_src_delete.clicked.connect(self.delete_selected_sources)
        h_src_act.addWidget(self.btn_src_delete)
        h_src_act.addStretch()
        lm.addLayout(h_src_act)
        
        gm.setLayout(lm)
        # gm added later in ordered assembly

        # Workspace Management
        gw = QGroupBox("Workspace")
        lw = QHBoxLayout()
        self.btn_load_workspace = QPushButton("Load Workspace")
        self.btn_save_workspace = QPushButton("Save Workspace")
        self.btn_load_workspace.clicked.connect(self.load_workspace)
        self.btn_save_workspace.clicked.connect(self.save_workspace)
        self.btn_save_workspace.setStyleSheet("background-color: #0277bd; font-weight: bold;")
        lw.addWidget(self.btn_load_workspace)
        lw.addWidget(self.btn_save_workspace)
        gw.setLayout(lw)
        # gw added later in ordered assembly

        g0 = QGroupBox("ROI Annotation Frames")
        l0 = QVBoxLayout()
        grid_f = QGridLayout()
        self.btn_frame_crop = QPushButton("New ROI")
        self.btn_frame_crop.setCheckable(True)
        self.btn_frame_crop.clicked.connect(lambda: self.on_tool("frame_crop"))
        self.btn_frame_focus = QPushButton("Focus Selected")
        self.btn_frame_focus.clicked.connect(self.focus_selected_frame)
        self.btn_frame_delete = QPushButton("Delete ROI")
        self.btn_frame_delete.clicked.connect(self.delete_selected_frame)
        self.btn_frame_clear = QPushButton("Clear All")
        self.btn_frame_clear.clicked.connect(self.clear_all_frames)
        grid_f.addWidget(self.btn_frame_crop, 0, 0)
        grid_f.addWidget(self.btn_frame_focus, 0, 1)
        grid_f.addWidget(self.btn_frame_delete, 1, 0)
        grid_f.addWidget(self.btn_frame_clear, 1, 1)
        
        self.btn_frame_full_range = QPushButton("Generate Full-XY Range")
        self.btn_frame_full_range.clicked.connect(self.generate_full_xy_frames)
        self.btn_frame_full_range.setStyleSheet("color: #00d2ff;")
        grid_f.addWidget(self.btn_frame_full_range, 2, 0, 1, 2)
        
        l0.addLayout(grid_f)


        self.chk_isolate_roi = QCheckBox("Isolate ROI (Zoom to selected)")
        self.chk_isolate_roi.stateChanged.connect(self.toggle_isolate_roi)
        l0.addWidget(self.chk_isolate_roi)

        split_grid = QGridLayout()
        split_grid.addWidget(QLabel("Default Split"), 0, 0)
        self.combo_frame_split = QComboBox()
        self.combo_frame_split.addItems(["train", "val", "test", "mix", "monitoring"])
        self.combo_frame_split.currentTextChanged.connect(self.on_default_frame_split_changed)
        split_grid.addWidget(self.combo_frame_split, 0, 1)
        
        split_grid.addWidget(QLabel("Selected ROI"), 1, 0)
        self.combo_frame_assign = QComboBox()
        self.combo_frame_assign.addItems(["train", "val", "test", "mix", "monitoring"])
        split_grid.addWidget(self.combo_frame_assign, 1, 1)
        
        self.btn_apply_frame_split = QPushButton("Apply Split")
        self.btn_apply_frame_split.clicked.connect(self.apply_selected_frame_split)
        split_grid.addWidget(self.btn_apply_frame_split, 2, 0)
        
        self.lbl_frame_summary = QLabel("No ROIs")
        split_grid.addWidget(self.lbl_frame_summary, 2, 1)
        l0.addLayout(split_grid)


        self.frame_table = QTableWidget(0, 7)
        self.frame_table.setHorizontalHeaderLabels(["[X]", "ROI", "Slice", "Split", "X1,Y1", "X2,Y2", "Size"])
        self.frame_table.verticalHeader().setVisible(False)
        self.frame_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.frame_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.frame_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.frame_table.setSortingEnabled(True)
        self.frame_table.setMaximumHeight(200)
        self._style_dark_table(self.frame_table)
        self.frame_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.frame_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.frame_table.itemSelectionChanged.connect(self.on_frame_selection_changed)
        self.frame_table.cellDoubleClicked.connect(lambda *_: self.focus_selected_frame())
        l0.addWidget(self.frame_table)
        frame_tip = QLabel("Draw a rectangle on the XY image to create small ROI annotation frames.")
        frame_tip.setWordWrap(True)
        frame_tip.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;")
        l0.addWidget(frame_tip)
        g0.setLayout(l0)
        # g0 added later in ordered assembly

        g1 = QGroupBox("Labels & Colors")
        l1 = QVBoxLayout()
        self.list_ui = QListWidget()
        self.list_ui.setStyleSheet("""
            QListWidget {
                background-color: #0d1b2a; border: 1px solid #2d3b4e;
                border-radius: 4px; padding: 2px;
                outline: none;
            }
            QListWidget::item {
                padding: 4px 6px; border-radius: 3px;
                color: #b0c4de; margin: 1px 2px;
            }
            QListWidget::item:selected {
                background-color: #0277bd; color: #ffffff;
                font-weight: bold;
                border-left: 3px solid #29b6f6;
            }
            QListWidget::item:hover:!selected {
                background-color: #1a3050;
            }
        """)
        self.list_ui.itemChanged.connect(self.on_item_changed)
        self.list_ui.itemSelectionChanged.connect(self.on_selection_changed)
        self.list_ui.itemDoubleClicked.connect(self.on_rename)
        l1.addWidget(self.list_ui)
        b_l = QHBoxLayout()
        b1 = QPushButton("Add")
        b1.clicked.connect(self.on_add)
        b2 = QPushButton("Del")
        b2.clicked.connect(self.on_del)
        b3 = QPushButton("Color")
        b3.clicked.connect(self.on_color)
        b_l.addWidget(b1)
        b_l.addWidget(b2)
        b_l.addWidget(b3)
        l1.addLayout(b_l)
        g1.setLayout(l1)
        # g1 added later in ordered assembly
        self._refresh()

        g_batch = QGroupBox("Batch Operations (Refinement)")
        l_batch = QVBoxLayout()
        
        # 1. Clear Slices
        l_clear = QHBoxLayout()
        l_clear.addWidget(QLabel("Clear Slices From:"))
        self.spin_clear_start = QSpinBox(); self.spin_clear_start.setRange(0, 9999)
        l_clear.addWidget(self.spin_clear_start)
        l_clear.addWidget(QLabel("To:"))
        self.spin_clear_end = QSpinBox(); self.spin_clear_end.setRange(0, 9999)
        l_clear.addWidget(self.spin_clear_end)
        self.btn_batch_clear = QPushButton("Clear Objects")
        self.btn_batch_clear.setStyleSheet("color: #ff5252; font-weight: bold;")
        self.btn_batch_clear.clicked.connect(self.clear_regions_in_range)
        l_clear.addWidget(self.btn_batch_clear)
        l_batch.addLayout(l_clear)
        
        # 2. Fill Holes
        l_fill = QHBoxLayout()
        l_fill.addWidget(QLabel("Find holes in:"))
        self.combo_fill_source = QComboBox()
        l_fill.addWidget(self.combo_fill_source)
        l_fill.addWidget(QLabel("Fill with:"))
        self.combo_fill_target = QComboBox()
        l_fill.addWidget(self.combo_fill_target)
        self.btn_batch_fill = QPushButton("Fill Holes (2D XY)")
        self.btn_batch_fill.setStyleSheet("color: #00e676; font-weight: bold;")
        self.btn_batch_fill.clicked.connect(self.fill_holes_2d)
        l_fill.addWidget(self.btn_batch_fill)
        l_batch.addLayout(l_fill)
        
        # 3. Propagate Labels
        self.btn_propagate = QPushButton("Propagate Z")
        self.btn_propagate.setStyleSheet("color: #40c4ff; font-weight: bold;")
        self.btn_propagate.setToolTip("Copy/morph labels from current slice to adjacent slices")
        self.btn_propagate.clicked.connect(self.propagate_labels)
        l_batch.addWidget(self.btn_propagate)
        
        # 4. Smart Frame Generation
        l_smart = QHBoxLayout()
        self.btn_smart_frames = QPushButton("Smart Frames")
        self.btn_smart_frames.setStyleSheet("color: #b388ff; font-weight: bold;")
        self.btn_smart_frames.setToolTip("Auto-create training frames only on labeled slices")
        self.btn_smart_frames.clicked.connect(self.generate_smart_frames)
        l_smart.addWidget(self.btn_smart_frames)
        
        self.btn_gen_full_xy = QPushButton("Full-XY Frames")
        self.btn_gen_full_xy.setToolTip("Generate full-slice frames for a Z range")
        self.btn_gen_full_xy.clicked.connect(self.generate_full_xy_frames)
        l_smart.addWidget(self.btn_gen_full_xy)
        l_batch.addLayout(l_smart)
        
        # Undo/Redo row for Batch Ops
        h_batch_ur = QHBoxLayout()
        self.btn_batch_undo = QPushButton("Undo")
        self.btn_batch_undo.setIcon(self._load_ui_icon("arrow-back-up.svg"))
        self.btn_batch_undo.setMinimumHeight(30)
        self.btn_batch_undo.clicked.connect(self.undo)
        self.btn_batch_redo = QPushButton("Redo")
        self.btn_batch_redo.setIcon(self._load_ui_icon("arrow-forward-up.svg"))
        self.btn_batch_redo.setMinimumHeight(30)
        self.btn_batch_redo.clicked.connect(self.redo)
        h_batch_ur.addWidget(self.btn_batch_undo)
        h_batch_ur.addWidget(self.btn_batch_redo)
        l_batch.addLayout(h_batch_ur)
        
        g_batch.setLayout(l_batch)
        # g_batch added later in ordered assembly

        g2 = QGroupBox("Manual Tools")
        l2 = QVBoxLayout()
        # Row 1: Original tools
        r = QHBoxLayout()
        self.bt_poly = QPushButton("Poly")
        self.bt_circ = QPushButton("Circle")
        self.bt_auto = QPushButton("Auto Brush")
        self.bt_erase = QPushButton("Eraser")
        _tool_btn_style = """
            QPushButton {
                background-color: #1e2a3a; color: #8899aa;
                font-weight: 600; font-size: 8pt;
                border: 1px solid #2d3b4e; border-radius: 4px;
                padding: 5px 4px; min-height: 22px;
            }
            QPushButton:hover {
                background-color: #263545; color: #b0c4de;
                border-color: #3d5a80;
            }
            QPushButton:checked {
                background-color: #0277bd; color: #ffffff;
                border: 2px solid #29b6f6; font-weight: 800;
            }
        """
        for b in [self.bt_poly, self.bt_circ, self.bt_auto, self.bt_erase]: 
            b.setCheckable(True); b.setStyleSheet(_tool_btn_style); b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _, t=b.text().lower().replace(" ","_"): self.on_tool(t)); r.addWidget(b)
        l2.addLayout(r)

        # Row 2: New tools inspired by DigitalSreeni annotation tool
        r2 = QHBoxLayout()
        self.bt_brush = QPushButton("Brush")
        self.bt_brush.setToolTip("Freehand paint brush — draw with active class")
        self.bt_rect_fill = QPushButton("Rect Fill")
        self.bt_rect_fill.setToolTip("Drag to fill a solid rectangle with active class")
        self.bt_flood = QPushButton("Flood Fill")
        self.bt_flood.setToolTip("Click to fill connected region with similar intensity\n(Magic Wand — inspired by DigitalSreeni annotation tool)")
        self.bt_sam_click = QPushButton("SAM Click")
        self.bt_sam_click.setToolTip(
            "SAM Point-Click Refinement (DigitalSreeni style)\n"
            "Left-click = Positive point (include in mask)\n"
            "Right-click = Negative point (exclude from mask)\n"
            "Requires SAM model loaded.\n"
            "Press 'Accept' or Enter to commit, 'Cancel' or Esc to discard."
        )
        for b in [self.bt_brush, self.bt_rect_fill, self.bt_flood, self.bt_sam_click]:
            b.setCheckable(True); b.setStyleSheet(_tool_btn_style); b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _, t=b.text().lower().replace(" ","_"): self.on_tool(t)); r2.addWidget(b)
        l2.addLayout(r2)

        self.lbl_active_tool = QLabel("Active: None")
        self.lbl_active_tool.setStyleSheet("color: #29b6f6; font-weight: 700; font-size: 8pt; padding: 2px 0;")
        l2.addWidget(self.lbl_active_tool)

        # SAM point-click Accept/Cancel row
        h_sam_btns = QHBoxLayout()
        self.btn_sam_accept = QPushButton("✓ Accept SAM")
        self.btn_sam_accept.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; font-size: 8pt;")
        self.btn_sam_accept.setMinimumHeight(28)
        self.btn_sam_accept.setToolTip("Accept SAM point-click prediction (or press Enter)")
        self.btn_sam_accept.clicked.connect(self._sam_click_accept)
        self.btn_sam_cancel = QPushButton("✗ Cancel SAM")
        self.btn_sam_cancel.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; font-size: 8pt;")
        self.btn_sam_cancel.setMinimumHeight(28)
        self.btn_sam_cancel.setToolTip("Cancel SAM point-click prediction (or press Esc)")
        self.btn_sam_cancel.clicked.connect(self._sam_click_cancel)
        h_sam_btns.addWidget(self.btn_sam_accept)
        h_sam_btns.addWidget(self.btn_sam_cancel)
        l2.addLayout(h_sam_btns)

        # Flood fill tolerance slider
        h_flood = QHBoxLayout()
        h_flood.addWidget(QLabel("Flood Tol:"))
        self.slider_flood_tol = QSlider(Qt.Horizontal)
        self.slider_flood_tol.setRange(1, 100)
        self.slider_flood_tol.setValue(20)
        self.slider_flood_tol.setToolTip("Flood fill intensity tolerance (higher = more permissive)")
        self.lbl_flood_tol = QLabel("20")
        self.lbl_flood_tol.setMinimumWidth(25)
        self.lbl_flood_tol.setStyleSheet("color: #29b6f6; font-weight: bold;")
        self.slider_flood_tol.valueChanged.connect(lambda v: (self.lbl_flood_tol.setText(str(v)), self._set_flood_tolerance(v)))
        h_flood.addWidget(self.slider_flood_tol)
        h_flood.addWidget(self.lbl_flood_tol)
        l2.addLayout(h_flood)

        h_l = QHBoxLayout(); self.bt_undo = QPushButton("Undo"); self.bt_undo.setIcon(self._load_ui_icon("arrow-back-up.svg")); self.bt_undo.setMinimumHeight(30); self.bt_undo.clicked.connect(self.undo)
        self.bt_redo = QPushButton("Redo"); self.bt_redo.setIcon(self._load_ui_icon("arrow-forward-up.svg")); self.bt_redo.setMinimumHeight(30); self.bt_redo.clicked.connect(self.redo); h_l.addWidget(self.bt_undo); h_l.addWidget(self.bt_redo)
        l2.addLayout(h_l)
        l2.addWidget(QLabel("Brush Size"))
        self.slider_brush = QSlider(Qt.Horizontal)
        self.slider_brush.setRange(1, 100); self.slider_brush.setValue(self.brush_size)
        self.slider_brush.valueChanged.connect(self.on_brush_slider_changed)
        l2.addWidget(self.slider_brush)
        
        l2.addWidget(QLabel("Mask Opacity"))
        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(0, 100); self.slider_opacity.setValue(int(self.mask_opacity * 100))
        self.slider_opacity.valueChanged.connect(self.on_opacity_changed)
        l2.addWidget(self.slider_opacity)
        
        g2.setLayout(l2)  # g2 added later in ordered assembly
        
        g3 = QGroupBox("Auto Labelling Mode (for Auto Brush)"); l3 = QVBoxLayout()
        self.rb_otsu = QRadioButton("Otsu Thresholding"); self.rb_otsu.setChecked(True)
        self.rb_sam = QRadioButton("SAM (AI Model)"); self.rb_sam.setEnabled(False)
        self.rb_otsu.toggled.connect(lambda: self.on_auto_mode_changed("Otsu"))
        self.rb_sam.toggled.connect(lambda: self.on_auto_mode_changed("SAM"))
        l3.addWidget(self.rb_otsu); l3.addWidget(self.rb_sam)
        
        l3.addWidget(QLabel("Threshold Direction (Otsu)"))
        self.combo_mode = QComboBox(); self.combo_mode.addItems(["Above", "Below"])
        self.combo_mode.currentTextChanged.connect(self.on_mode_changed); l3.addWidget(self.combo_mode)

        # ── Manual Threshold Control ──
        h_thresh_mode = QHBoxLayout()
        self.rb_thresh_auto = QRadioButton("Auto (Otsu)")
        self.rb_thresh_auto.setChecked(True)
        self.rb_thresh_manual = QRadioButton("Manual")
        h_thresh_mode.addWidget(self.rb_thresh_auto)
        h_thresh_mode.addWidget(self.rb_thresh_manual)
        l3.addLayout(h_thresh_mode)

        # Manual threshold slider + value label
        h_thresh_slider = QHBoxLayout()
        h_thresh_slider.addWidget(QLabel("Threshold:"))
        self.slider_manual_thresh = QSlider(Qt.Horizontal)
        self.slider_manual_thresh.setRange(0, 255)
        self.slider_manual_thresh.setValue(128)
        self.slider_manual_thresh.setEnabled(False)
        self.lbl_manual_thresh = QLabel("128")
        self.lbl_manual_thresh.setMinimumWidth(30)
        self.lbl_manual_thresh.setStyleSheet("color: #29b6f6; font-weight: bold;")
        self.slider_manual_thresh.valueChanged.connect(
            lambda v: self.lbl_manual_thresh.setText(str(v))
        )
        h_thresh_slider.addWidget(self.slider_manual_thresh)
        h_thresh_slider.addWidget(self.lbl_manual_thresh)
        l3.addLayout(h_thresh_slider)

        # Enable/disable manual slider based on mode
        self.rb_thresh_manual.toggled.connect(self.slider_manual_thresh.setEnabled)

        # Apply threshold buttons
        h_thresh_apply = QHBoxLayout()
        self.btn_thresh_preview = QPushButton("Preview Slice")
        self.btn_thresh_preview.setToolTip("Preview thresholding on current slice (temporary overlay)")
        self.btn_thresh_preview.clicked.connect(self._preview_threshold)
        h_thresh_apply.addWidget(self.btn_thresh_preview)

        self.btn_thresh_apply = QPushButton("Apply to Slice")
        self.btn_thresh_apply.setToolTip("Apply thresholding to current slice segmentation")
        self.btn_thresh_apply.clicked.connect(lambda: self._apply_threshold_to("slice"))
        h_thresh_apply.addWidget(self.btn_thresh_apply)

        self.btn_thresh_apply_vol = QPushButton("Apply to Volume")
        self.btn_thresh_apply_vol.setToolTip("Apply thresholding to all slices in the volume")
        self.btn_thresh_apply_vol.clicked.connect(lambda: self._apply_threshold_to("volume"))
        h_thresh_apply.addWidget(self.btn_thresh_apply_vol)
        l3.addLayout(h_thresh_apply)
        
        self.btn_load_sam = QPushButton("Load SAM2/3 Model"); self.btn_load_sam.clicked.connect(self.on_load_sam)
        l3.addWidget(self.btn_load_sam); g3.setLayout(l3)  # g3 added later

        # ── Text-Prompted Auto-Segmentation (Grounding DINO + SAM) ──
        g_dino = QGroupBox("Text-Prompted Segmentation (DINO + SAM)")
        l_dino = QVBoxLayout()

        dino_tip = QLabel(
            "Pipeline: Text → Grounding DINO (detect) → SAM (segment)\n"
            "Type object descriptions → AI auto-generates pixel masks.\n"
            "Separate multiple prompts with periods: e.g. 'void. crack. defect'"
        )
        dino_tip.setWordWrap(True)
        dino_tip.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 7.5pt;")
        l_dino.addWidget(dino_tip)

        # Domain-shift warning (Section 8 of notebook)
        dino_warn = QLabel(
            "⚠ Domain Note: DINO is trained on natural images. For specialized\n"
            "images (X-ray CT, EM, SEM), use descriptive prompts like\n"
            "'dark hole. bright region. dark spot' instead of domain terms.\n"
            "If DINO fails, use SAM Point-Click (Auto Brush + SAM mode above)."
        )
        dino_warn.setWordWrap(True)
        dino_warn.setStyleSheet(
            f"color: {SemiconductorTheme.ACCENT_WARNING}; font-size: 7pt; "
            f"padding: 4px; border: 1px solid {SemiconductorTheme.ACCENT_WARNING}; "
            f"border-radius: 3px; margin: 2px 0;"
        )
        l_dino.addWidget(dino_warn)

        h_prompt = QHBoxLayout()
        h_prompt.addWidget(QLabel("Prompt:"))
        self.txt_dino_prompt = QLineEdit()
        self.txt_dino_prompt.setPlaceholderText("dark spot. bright region. hole...")
        self.txt_dino_prompt.setToolTip(
            "Text prompts for Grounding DINO object detection.\n"
            "Separate multiple objects with periods.\n"
            "For semiconductor: 'dark hole. bright column. dark spot'\n"
            "For natural images: 'cat. dog. person'"
        )
        h_prompt.addWidget(self.txt_dino_prompt)
        l_dino.addLayout(h_prompt)

        # Parameters
        f_dino = QFormLayout()
        self.spin_dino_conf = QDoubleSpinBox()
        self.spin_dino_conf.setRange(0.01, 1.0)
        self.spin_dino_conf.setSingleStep(0.05)
        self.spin_dino_conf.setValue(0.15)
        self.spin_dino_conf.setToolTip(
            "Detection confidence threshold.\n"
            "Lower = more detections but more noise.\n"
            "For semiconductor images: try 0.05-0.15\n"
            "For natural images: 0.25-0.35"
        )
        f_dino.addRow("Confidence:", self.spin_dino_conf)

        self.spin_dino_iou = QDoubleSpinBox()
        self.spin_dino_iou.setRange(0.0, 1.0)
        self.spin_dino_iou.setSingleStep(0.05)
        self.spin_dino_iou.setValue(0.50)
        self.spin_dino_iou.setToolTip("NMS IoU threshold. Higher = less duplicate suppression.")
        f_dino.addRow("NMS IoU:", self.spin_dino_iou)

        self.combo_dino_target = QComboBox()
        self.combo_dino_target.addItems(["Current Slice", "Slice Range", "Full Volume"])
        f_dino.addRow("Apply to:", self.combo_dino_target)

        self.spin_dino_z_from = QSpinBox(); self.spin_dino_z_from.setRange(0, 9999)
        self.spin_dino_z_to = QSpinBox(); self.spin_dino_z_to.setRange(0, 9999); self.spin_dino_z_to.setValue(999)
        h_dino_range = QHBoxLayout()
        h_dino_range.addWidget(QLabel("Z from:")); h_dino_range.addWidget(self.spin_dino_z_from)
        h_dino_range.addWidget(QLabel("to:")); h_dino_range.addWidget(self.spin_dino_z_to)
        self.dino_range_widget = QWidget()
        self.dino_range_widget.setLayout(h_dino_range)
        self.dino_range_widget.setVisible(False)
        self.combo_dino_target.currentTextChanged.connect(
            lambda t: self.dino_range_widget.setVisible(t == "Slice Range")
        )
        l_dino.addLayout(f_dino)
        l_dino.addWidget(self.dino_range_widget)

        # Class mapping
        self.chk_dino_auto_map = QCheckBox("Auto-map prompt words → class names")
        self.chk_dino_auto_map.setChecked(True)
        self.chk_dino_auto_map.setToolTip(
            "When checked, detected labels are matched to your class names.\n"
            "Unmatched detections default to the currently selected class."
        )
        l_dino.addWidget(self.chk_dino_auto_map)

        # Action buttons
        h_dino_btns = QHBoxLayout()
        self.btn_dino_load = QPushButton("Load DINO+SAM")
        self.btn_dino_load.setIcon(self._load_ui_icon("download.svg"))
        self.btn_dino_load.setMinimumHeight(30)
        self.btn_dino_load.setStyleSheet("color: #4fc3f7; font-weight: bold;")
        self.btn_dino_load.setToolTip("Pre-load Grounding DINO + SAM models (downloads ~1GB on first use)")
        self.btn_dino_load.clicked.connect(self._load_dino_sam2_models)
        h_dino_btns.addWidget(self.btn_dino_load)

        self.btn_dino_run = QPushButton("Run Segmentation")
        self.btn_dino_run.setIcon(self._load_ui_icon("search-ai.svg"))
        self.btn_dino_run.setMinimumHeight(30)
        self.btn_dino_run.setStyleSheet("background-color: #00897b; font-weight: bold;")
        self.btn_dino_run.setToolTip("Run text-prompted detection + segmentation on the selected slices")
        self.btn_dino_run.clicked.connect(self._run_dino_sam2)
        h_dino_btns.addWidget(self.btn_dino_run)
        l_dino.addLayout(h_dino_btns)

        self.lbl_dino_status = QLabel("Models not loaded. Click 'Load DINO+SAM' first.")
        self.lbl_dino_status.setWordWrap(True)
        self.lbl_dino_status.setStyleSheet("color: #aaa; font-size: 8pt;")
        l_dino.addWidget(self.lbl_dino_status)

        # SAM-only fallback tip (Section 8 recommendation)
        sam_tip = QLabel(
            "💡 Tip: If text detection fails, use the SAM Point-Click workflow:\n"
            "  1. Load SAM model above (Auto Labelling → SAM)\n"
            "  2. Select 'Auto Brush' tool\n"
            "  3. Drag a box around the object on the slice viewer\n"
            "  → SAM segments the object boundary precisely"
        )
        sam_tip.setWordWrap(True)
        sam_tip.setStyleSheet(
            f"color: {SemiconductorTheme.ACCENT_SUCCESS}; font-size: 7pt; "
            f"padding: 4px; border: 1px solid {SemiconductorTheme.ACCENT_SUCCESS}; "
            f"border-radius: 3px; margin: 4px 0 0 0;"
        )
        l_dino.addWidget(sam_tip)

        g_dino.setLayout(l_dino)  # g_dino added later in ordered assembly

        # ── AutoPrompt-SAM: Intensity-Guided Auto-Segmentation ──
        g_auto = QGroupBox("AutoPrompt-SAM (Intensity-Guided)")
        g_auto.setStyleSheet(
            f"QGroupBox {{ border: 1px solid #00e5ff; border-radius: 4px; "
            f"margin-top: 8px; padding-top: 14px; font-weight: bold; "
            f"color: #00e5ff; }}"
        )
        l_auto = QVBoxLayout()

        auto_desc = QLabel(
            "Novel pipeline: Classical CV detects structures by intensity,\n"
            "then auto-generates SAM prompts for pixel-precise masks.\n"
            "No text prompts needed. Designed for industrial X-ray CT."
        )
        auto_desc.setWordWrap(True)
        auto_desc.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 7.5pt;")
        l_auto.addWidget(auto_desc)

        # Parameters
        f_auto = QFormLayout()

        self.spin_auto_bright = QDoubleSpinBox()
        self.spin_auto_bright.setRange(50, 99)
        self.spin_auto_bright.setSingleStep(5)
        self.spin_auto_bright.setValue(80)
        self.spin_auto_bright.setDecimals(0)
        self.spin_auto_bright.setToolTip(
            "Brightness percentile for TGV detection.\n"
            "Top N% brightest pixels become candidates.\n"
            "Lower = more sensitive, Higher = stricter."
        )
        f_auto.addRow("Bright Percentile:", self.spin_auto_bright)

        self.spin_auto_min_area = QSpinBox()
        self.spin_auto_min_area.setRange(10, 50000)
        self.spin_auto_min_area.setSingleStep(100)
        self.spin_auto_min_area.setValue(500)
        self.spin_auto_min_area.setToolTip(
            "Minimum area (pixels) for TGV structures.\n"
            "Smaller values catch more structures but also noise."
        )
        f_auto.addRow("Min TGV Area:", self.spin_auto_min_area)

        self.spin_auto_dark = QDoubleSpinBox()
        self.spin_auto_dark.setRange(0.1, 0.95)
        self.spin_auto_dark.setSingleStep(0.05)
        self.spin_auto_dark.setValue(0.6)
        self.spin_auto_dark.setToolTip(
            "Dark factor for void detection within TGVs.\n"
            "Pixels below (local_mean * factor) are voids.\n"
            "Lower = only very dark holes, Higher = more sensitive."
        )
        f_auto.addRow("Void Dark Factor:", self.spin_auto_dark)

        self.spin_auto_min_void = QSpinBox()
        self.spin_auto_min_void.setRange(1, 5000)
        self.spin_auto_min_void.setSingleStep(5)
        self.spin_auto_min_void.setValue(10)
        self.spin_auto_min_void.setToolTip("Minimum area (pixels) for void defects.")
        f_auto.addRow("Min Void Area:", self.spin_auto_min_void)

        l_auto.addLayout(f_auto)

        # Options
        self.chk_auto_sam = QCheckBox("Refine with SAM")
        self.chk_auto_sam.setChecked(True)
        self.chk_auto_sam.setToolTip(
            "When checked, classical detections are refined by SAM\n"
            "for pixel-precise boundaries. Uncheck to use raw masks."
        )
        l_auto.addWidget(self.chk_auto_sam)

        self.chk_auto_voids = QCheckBox("Detect Voids (hierarchical)")
        self.chk_auto_voids.setChecked(True)
        self.chk_auto_voids.setToolTip(
            "Detect dark defects inside each bright structure.\n"
            "Uses local intensity analysis within each TGV region."
        )
        l_auto.addWidget(self.chk_auto_voids)

        # Apply target
        h_auto_target = QHBoxLayout()
        h_auto_target.addWidget(QLabel("Apply to:"))
        self.combo_auto_target = QComboBox()
        self.combo_auto_target.addItems(["Current Slice", "Slice Range", "Full Volume"])
        h_auto_target.addWidget(self.combo_auto_target)
        l_auto.addLayout(h_auto_target)

        self.spin_auto_z_from = QSpinBox(); self.spin_auto_z_from.setRange(0, 9999)
        self.spin_auto_z_to = QSpinBox(); self.spin_auto_z_to.setRange(0, 9999); self.spin_auto_z_to.setValue(999)
        h_auto_range = QHBoxLayout()
        h_auto_range.addWidget(QLabel("Z from:")); h_auto_range.addWidget(self.spin_auto_z_from)
        h_auto_range.addWidget(QLabel("to:")); h_auto_range.addWidget(self.spin_auto_z_to)
        self.auto_range_widget = QWidget()
        self.auto_range_widget.setLayout(h_auto_range)
        self.auto_range_widget.setVisible(False)
        self.combo_auto_target.currentTextChanged.connect(
            lambda t: self.auto_range_widget.setVisible(t == "Slice Range")
        )
        l_auto.addWidget(self.auto_range_widget)

        # Run button
        self.btn_auto_run = QPushButton("Run AutoPrompt-SAM")
        self.btn_auto_run.setMinimumHeight(34)
        self.btn_auto_run.setStyleSheet(
            "background-color: #00b8d4; color: #000; font-weight: bold; "
            "font-size: 10pt; border-radius: 4px;"
        )
        self.btn_auto_run.setToolTip(
            "Run the intensity-guided auto-segmentation pipeline.\n"
            "Detects TGVs and Voids automatically without text prompts."
        )
        self.btn_auto_run.clicked.connect(self._run_autoprompt_sam)
        l_auto.addWidget(self.btn_auto_run)

        self.lbl_auto_status = QLabel("Ready. Load SAM model above for best results.")
        self.lbl_auto_status.setWordWrap(True)
        self.lbl_auto_status.setStyleSheet("color: #aaa; font-size: 8pt;")
        l_auto.addWidget(self.lbl_auto_status)

        g_auto.setLayout(l_auto)  # g_auto added later in ordered assembly

        # ══════════════════════════════════════════════════════
        # ORDERED ASSEMBLY — workflow-based section layout
        # ══════════════════════════════════════════════════════
        def _section_hdr(text, color="#00e5ff"):
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f"color: {color}; font-size: 9pt; font-weight: bold; "
                f"padding: 6px 0 2px 4px; border-bottom: 1px solid {color};"
            )
            return lbl

        # ── A. Data Management ──
        l.addWidget(_section_hdr("A. Data Management"))
        l.addWidget(gm)   # Source Volume Info
        l.addWidget(gw)   # Workspace Save/Load

        # ── B. Label Configuration ──
        l.addWidget(_section_hdr("B. Label Configuration"))
        l.addWidget(g1)   # Classes (Labels & Colors)
        l.addWidget(g0)   # ROI Annotation Frames

        # ── C. Manual Tools ──
        l.addWidget(_section_hdr("C. Manual Tools"))
        l.addWidget(g2)   # Drawing tools, Undo/Redo, Brush

        # ── D. Semi-Automatic Tools ──
        l.addWidget(_section_hdr("D. Semi-Automatic Tools"))
        l.addWidget(g3)       # Threshold (Otsu / Manual)
        l.addWidget(g_batch)  # Batch Refinement (Fill, Clear, Propagate)

        # ── E. AI-Powered Tools ──
        l.addWidget(_section_hdr("E. AI-Powered Tools", "#00e5ff"))
        l.addWidget(g_auto)   # AutoPrompt-SAM (recommended)
        l.addWidget(g_dino)   # DINO + SAM (text-prompted, experimental)

        l.addStretch()
        scroll.setWidget(w)
        outer_layout.addWidget(scroll)
        return outer

    def _create_data_tab(self):
        """AI Data tab — data management, augmentation, and statistics for training."""
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        l = QVBoxLayout(content)
        l.setSpacing(12)
        l.setContentsMargins(8, 8, 8, 8)

        # ── 1. Train / Val / Test Split Assignment ──
        g_split = QGroupBox("1. Train / Val / Test Split")
        l_split = QVBoxLayout()

        h_split_top = QHBoxLayout()
        self.mdl_btn_refresh_slices = QPushButton("Scan Labeled Slices")
        self.mdl_btn_refresh_slices.setIcon(self._load_ui_icon("scan.svg"))
        self.mdl_btn_refresh_slices.setMinimumHeight(28)
        self.mdl_btn_refresh_slices.setStyleSheet("color: #4fc3f7; font-weight: bold;")
        self.mdl_btn_refresh_slices.setToolTip("Scan segmentation data to find all slices with labels")
        self.mdl_btn_refresh_slices.clicked.connect(self._refresh_slice_split_table)
        h_split_top.addWidget(self.mdl_btn_refresh_slices)

        self.mdl_lbl_split_summary = QLabel("Click 'Scan' to detect labeled slices")
        self.mdl_lbl_split_summary.setStyleSheet("color: #aaa; font-size: 8pt;")
        h_split_top.addWidget(self.mdl_lbl_split_summary)
        h_split_top.addStretch()
        l_split.addLayout(h_split_top)

        self.mdl_slice_table = QTableWidget(0, 3)
        self.mdl_slice_table.setHorizontalHeaderLabels(["Z Slice", "Labeled Px", "Split"])
        self.mdl_slice_table.verticalHeader().setVisible(False)
        self.mdl_slice_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.mdl_slice_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.mdl_slice_table.setMaximumHeight(180)
        self.mdl_slice_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.mdl_slice_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.mdl_slice_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._style_dark_table(self.mdl_slice_table)
        l_split.addWidget(self.mdl_slice_table)

        h_assign = QHBoxLayout()
        h_assign.addWidget(QLabel("Set selected slices to:"))
        self.mdl_combo_slice_split = QComboBox()
        self.mdl_combo_slice_split.addItems(["train", "val", "test", "monitoring"])
        h_assign.addWidget(self.mdl_combo_slice_split)
        self.mdl_btn_assign_split = QPushButton("Apply")
        self.mdl_btn_assign_split.setStyleSheet("background-color: #0277bd; font-weight: bold;")
        self.mdl_btn_assign_split.clicked.connect(self._apply_slice_split)
        h_assign.addWidget(self.mdl_btn_assign_split)
        h_assign.addStretch()
        l_split.addLayout(h_assign)

        h_range = QHBoxLayout()
        h_range.addWidget(QLabel("Range Z:"))
        self.mdl_spin_split_from = QSpinBox(); self.mdl_spin_split_from.setRange(0, 9999)
        h_range.addWidget(self.mdl_spin_split_from)
        h_range.addWidget(QLabel("→"))
        self.mdl_spin_split_to = QSpinBox(); self.mdl_spin_split_to.setRange(0, 9999); self.mdl_spin_split_to.setValue(999)
        h_range.addWidget(self.mdl_spin_split_to)
        self.mdl_combo_range_split = QComboBox()
        self.mdl_combo_range_split.addItems(["train", "val", "test", "monitoring"])
        h_range.addWidget(self.mdl_combo_range_split)
        self.mdl_btn_range_split = QPushButton("Apply Range")
        self.mdl_btn_range_split.setStyleSheet("color: #ffb300; font-weight: bold;")
        self.mdl_btn_range_split.clicked.connect(self._apply_range_split)
        h_range.addWidget(self.mdl_btn_range_split)
        l_split.addLayout(h_range)

        self.mdl_btn_auto_split = QPushButton("Auto Split (70% Train / 15% Val / 15% Test)")
        self.mdl_btn_auto_split.setStyleSheet("color: #80cbc4;")
        self.mdl_btn_auto_split.clicked.connect(self._auto_split_slices)
        l_split.addWidget(self.mdl_btn_auto_split)

        g_split.setLayout(l_split)
        l.addWidget(g_split)

        if not hasattr(self, '_slice_splits'):
            self._slice_splits = {}

        # ── 2. Label Expansion ──
        g_expand = QGroupBox("2. Label Expansion (Sparse → Dense)")
        l_expand = QVBoxLayout()
        self.mdl_btn_sparse_dense = QPushButton("Sparse → Dense Labels (Expand to nearby Z)")
        self.mdl_btn_sparse_dense.setIcon(self._load_ui_icon("arrows-left-right.svg"))
        self.mdl_btn_sparse_dense.setMinimumHeight(30)
        self.mdl_btn_sparse_dense.setStyleSheet("color: #b388ff; font-weight: bold;")
        self.mdl_btn_sparse_dense.setToolTip(
            "Automatically copy labels from labeled slices to their neighbors.\n"
            "Since 3D adjacent slices are nearly identical,\n"
            "this creates more training data from sparse labels.\n\n"
            "Example: If you labeled slices 100, 200, 300,\n"
            "this expands each ±N slices = more training patches."
        )
        self.mdl_btn_sparse_dense.clicked.connect(self.sparse_to_dense_expand)
        l_expand.addWidget(self.mdl_btn_sparse_dense)
        g_expand.setLayout(l_expand)
        l.addWidget(g_expand)

        # ── 3. Data Augmentation ──
        g_aug = QGroupBox("3. Data Augmentation (GPU)")
        l_aug = QVBoxLayout()

        aug_tip = QLabel("Enable/disable augmentations applied on-the-fly during training (GPU-accelerated).")
        aug_tip.setWordWrap(True)
        aug_tip.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;")
        l_aug.addWidget(aug_tip)

        # Geometric augmentations
        l_aug.addSpacing(4)
        geo_label = QLabel("Geometric")
        geo_label.setStyleSheet("color: #4fc3f7; font-weight: bold; font-size: 8pt; border-bottom: 1px solid #2d3b4e; padding-bottom: 2px;")
        l_aug.addWidget(geo_label)

        self.chk_aug_hflip = QCheckBox("Horizontal Flip (50%)")
        self.chk_aug_hflip.setChecked(True)
        self.chk_aug_hflip.setToolTip("Randomly flip image left-right with 50% probability")
        l_aug.addWidget(self.chk_aug_hflip)

        self.chk_aug_vflip = QCheckBox("Vertical Flip (50%)")
        self.chk_aug_vflip.setChecked(True)
        self.chk_aug_vflip.setToolTip("Randomly flip image top-bottom with 50% probability")
        l_aug.addWidget(self.chk_aug_vflip)

        self.chk_aug_rot90 = QCheckBox("Random 90° Rotation")
        self.chk_aug_rot90.setChecked(True)
        self.chk_aug_rot90.setToolTip("Randomly rotate image by 0°, 90°, 180°, or 270°")
        l_aug.addWidget(self.chk_aug_rot90)

        # Intensity augmentations
        l_aug.addSpacing(4)
        int_label = QLabel("Intensity")
        int_label.setStyleSheet("color: #ffb74d; font-weight: bold; font-size: 8pt; border-bottom: 1px solid #2d3b4e; padding-bottom: 2px;")
        l_aug.addWidget(int_label)

        self.chk_aug_brightness = QCheckBox("Random Brightness (±10%)")
        self.chk_aug_brightness.setChecked(False)
        self.chk_aug_brightness.setToolTip("Randomly shift pixel brightness by ±10%")
        l_aug.addWidget(self.chk_aug_brightness)

        self.chk_aug_contrast = QCheckBox("Random Contrast (0.8 - 1.2x)")
        self.chk_aug_contrast.setChecked(False)
        self.chk_aug_contrast.setToolTip("Randomly scale contrast between 0.8x and 1.2x")
        l_aug.addWidget(self.chk_aug_contrast)

        self.chk_aug_noise = QCheckBox("Gaussian Noise (σ=0.02)")
        self.chk_aug_noise.setChecked(False)
        self.chk_aug_noise.setToolTip("Add random Gaussian noise to the input image")
        l_aug.addWidget(self.chk_aug_noise)

        # Spatial augmentations
        l_aug.addSpacing(4)
        sp_label = QLabel("Spatial")
        sp_label.setStyleSheet("color: #81c784; font-weight: bold; font-size: 8pt; border-bottom: 1px solid #2d3b4e; padding-bottom: 2px;")
        l_aug.addWidget(sp_label)

        self.chk_aug_cutout = QCheckBox("Random Cutout (Erasing)")
        self.chk_aug_cutout.setChecked(False)
        self.chk_aug_cutout.setToolTip("Randomly erase a rectangular region (forces model to learn from partial context)")
        l_aug.addWidget(self.chk_aug_cutout)

        self.chk_aug_elastic = QCheckBox("Elastic Deformation")
        self.chk_aug_elastic.setChecked(False)
        self.chk_aug_elastic.setToolTip("Apply slight elastic warping to simulate structural variation")
        l_aug.addWidget(self.chk_aug_elastic)

        # Quick preset buttons
        l_aug.addSpacing(6)
        h_presets = QHBoxLayout()
        btn_aug_minimal = QPushButton("Minimal")
        btn_aug_minimal.setToolTip("Geometric only (flip + rotate)")
        btn_aug_minimal.setStyleSheet("font-size: 7.5pt;")
        btn_aug_minimal.clicked.connect(lambda: self._set_aug_preset('minimal'))
        h_presets.addWidget(btn_aug_minimal)

        btn_aug_standard = QPushButton("Standard")
        btn_aug_standard.setToolTip("Geometric + Brightness + Contrast")
        btn_aug_standard.setStyleSheet("font-size: 7.5pt;")
        btn_aug_standard.clicked.connect(lambda: self._set_aug_preset('standard'))
        h_presets.addWidget(btn_aug_standard)

        btn_aug_heavy = QPushButton("Heavy")
        btn_aug_heavy.setToolTip("All augmentations enabled")
        btn_aug_heavy.setStyleSheet("font-size: 7.5pt;")
        btn_aug_heavy.clicked.connect(lambda: self._set_aug_preset('heavy'))
        h_presets.addWidget(btn_aug_heavy)

        btn_aug_none = QPushButton("None")
        btn_aug_none.setToolTip("Disable all augmentations")
        btn_aug_none.setStyleSheet("font-size: 7.5pt;")
        btn_aug_none.clicked.connect(lambda: self._set_aug_preset('none'))
        h_presets.addWidget(btn_aug_none)
        l_aug.addLayout(h_presets)

        g_aug.setLayout(l_aug)
        l.addWidget(g_aug)

        # ── 4. Data Statistics ──
        g_stats = QGroupBox("4. Data Statistics")
        l_stats = QVBoxLayout()

        self.btn_scan_stats = QPushButton("Scan Data Statistics")
        self.btn_scan_stats.setIcon(self._load_ui_icon("chart-bar.svg"))
        self.btn_scan_stats.setMinimumHeight(28)
        self.btn_scan_stats.setStyleSheet("color: #4fc3f7; font-weight: bold;")
        self.btn_scan_stats.setToolTip("Compute label distribution, class balance, and coverage statistics")
        self.btn_scan_stats.clicked.connect(self._compute_data_stats)
        l_stats.addWidget(self.btn_scan_stats)

        self.lbl_data_stats = QLabel("Click 'Scan' to compute statistics.")
        self.lbl_data_stats.setWordWrap(True)
        self.lbl_data_stats.setStyleSheet("color: #b0c4de; font-size: 8pt; padding: 4px;")
        self.lbl_data_stats.setTextFormat(Qt.RichText)
        l_stats.addWidget(self.lbl_data_stats)

        g_stats.setLayout(l_stats)
        l.addWidget(g_stats)

        # ── 5. Active Learning (Smart Labeling) ──
        g_al = QGroupBox("5. Active Learning (Smart Labeling)")
        l_al = QVBoxLayout()
        try:
            from inno3d.widgets.active_learning_widget import ActiveLearningWidget
            self.al_widget = ActiveLearningWidget()
            self.al_widget.run_requested.connect(self.run_active_learning)
            self.al_widget.goto_slice.connect(self.set_current_slice)
            l_al.addWidget(self.al_widget)
        except ImportError as e:
            l_al.addWidget(QLabel(f"Active Learning UI not available: {e}"))
        g_al.setLayout(l_al)
        l.addWidget(g_al)

        l.addStretch()
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)
        return outer

    def _set_aug_preset(self, preset):
        """Apply a data augmentation preset."""
        configs = {
            'none':     (False, False, False, False, False, False, False, False),
            'minimal':  (True,  True,  True,  False, False, False, False, False),
            'standard': (True,  True,  True,  True,  True,  False, False, False),
            'heavy':    (True,  True,  True,  True,  True,  True,  True,  True),
        }
        vals = configs.get(preset, configs['minimal'])
        widgets = [
            self.chk_aug_hflip, self.chk_aug_vflip, self.chk_aug_rot90,
            self.chk_aug_brightness, self.chk_aug_contrast, self.chk_aug_noise,
            self.chk_aug_cutout, self.chk_aug_elastic,
        ]
        for w, v in zip(widgets, vals):
            w.setChecked(v)

    def _get_augmentation_config(self):
        """Collect augmentation settings into a dict for the training worker."""
        return {
            'hflip': getattr(self, 'chk_aug_hflip', None) and self.chk_aug_hflip.isChecked(),
            'vflip': getattr(self, 'chk_aug_vflip', None) and self.chk_aug_vflip.isChecked(),
            'rot90': getattr(self, 'chk_aug_rot90', None) and self.chk_aug_rot90.isChecked(),
            'brightness': getattr(self, 'chk_aug_brightness', None) and self.chk_aug_brightness.isChecked(),
            'contrast': getattr(self, 'chk_aug_contrast', None) and self.chk_aug_contrast.isChecked(),
            'noise': getattr(self, 'chk_aug_noise', None) and self.chk_aug_noise.isChecked(),
            'cutout': getattr(self, 'chk_aug_cutout', None) and self.chk_aug_cutout.isChecked(),
            'elastic': getattr(self, 'chk_aug_elastic', None) and self.chk_aug_elastic.isChecked(),
        }

    def _compute_data_stats(self):
        """Compute and display training data statistics."""
        if self.segmentation_data is None:
            self.lbl_data_stats.setText("No segmentation data loaded.")
            return

        seg = self.segmentation_data
        total_voxels = seg.size
        labeled_voxels = int(np.count_nonzero(seg))
        labeled_pct = labeled_voxels / total_voxels * 100 if total_voxels > 0 else 0

        # Per-class counts
        unique, counts = np.unique(seg, return_counts=True)
        class_lines = []
        for cls_id, cnt in zip(unique, counts):
            if cls_id == 0:
                name = "Background"
                color = "#666"
            else:
                info = self.classes.get(int(cls_id), {})
                name = info.get('name', f'Class {cls_id}')
                rgb = info.get('color', (200, 200, 200))
                color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
            pct = cnt / total_voxels * 100
            class_lines.append(
                f"<span style='color:{color};'>■</span> {name}: "
                f"<b>{cnt:,}</b> px ({pct:.2f}%)"
            )

        # Labeled slices
        labeled_slices = int(np.any(seg > 0, axis=(1, 2)).sum()) if seg.ndim == 3 else 0
        total_slices = seg.shape[0] if seg.ndim == 3 else 1

        # Split stats
        splits = getattr(self, '_slice_splits', {})
        split_counts = {}
        for z, sp in splits.items():
            split_counts[sp] = split_counts.get(sp, 0) + 1

        # ROI count
        n_rois = len(getattr(self, 'crop_frames', []))
        train_rois = len([f for f in getattr(self, 'crop_frames', []) if f.get('split') == 'train'])
        val_rois = len([f for f in getattr(self, 'crop_frames', []) if f.get('split') == 'val'])

        html = (
            f"<b>Volume:</b> {seg.shape}<br>"
            f"<b>Labeled Voxels:</b> {labeled_voxels:,} / {total_voxels:,} ({labeled_pct:.1f}%)<br>"
            f"<b>Labeled Slices:</b> {labeled_slices} / {total_slices}<br>"
            f"<b>ROIs:</b> {n_rois} total (train: {train_rois}, val: {val_rois})<br>"
            f"<hr>"
            f"<b>Class Distribution:</b><br>"
            + "<br>".join(class_lines)
        )
        if split_counts:
            split_str = " | ".join(f"{k}: {v}" for k, v in sorted(split_counts.items()))
            html += f"<br><hr><b>Split Assignment:</b> {split_str}"

        self.lbl_data_stats.setText(html)

    def _create_model_tab(self):
        outer = QWidget(); outer_layout = QVBoxLayout(outer); outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget(); l = QVBoxLayout(content); l.setSpacing(12); l.setContentsMargins(8, 8, 8, 8)
        
        # 1. Architecture
        g_arch = QGroupBox("1. Model Architecture")
        l_arch = QFormLayout()
        self.mdl_combo_arch = QComboBox(); self.mdl_combo_arch.addItems(["UNet", "ResUNet++", "Mask2Former"])
        self.mdl_combo_dim = QComboBox(); self.mdl_combo_dim.addItems(["2D", "2.5D", "3D"])
        self.mdl_spin_adj = QSpinBox(); self.mdl_spin_adj.setRange(1, 10); self.mdl_spin_adj.setValue(5)
        self.mdl_spin_adj.setEnabled(False)
        self.mdl_combo_dim.currentTextChanged.connect(lambda t: self.mdl_spin_adj.setEnabled(t == "2.5D"))
        
        l_arch.addRow("Architecture:", self.mdl_combo_arch)
        l_arch.addRow("Dimensionality:", self.mdl_combo_dim)
        l_arch.addRow("Adjacent Slices (2.5D):", self.mdl_spin_adj)
        
        self.mdl_spin_classes = QSpinBox()
        self.mdl_spin_classes.setRange(1, 20)
        self.mdl_spin_classes.setValue(max(1, len(self.classes)))
        self.mdl_spin_classes.setToolTip("Number of foreground classes you want to train on (Background is automatically added +1)")
        self.mdl_spin_classes.valueChanged.connect(lambda _: self._rebuild_class_weight_spins())
        l_arch.addRow("Foreground classes:", self.mdl_spin_classes)
        
        g_arch.setLayout(l_arch)
        l.addWidget(g_arch)

        # 1b. Foundation Pre-Training (Self-Supervised)
        g_foundation = QGroupBox("2. Foundation Pre-Training (Self-Supervised)")
        l_foundation = QVBoxLayout()
        try:
            from inno3d.widgets.foundation_widget import FoundationWidget
            self.foundation_widget = FoundationWidget()
            self.foundation_widget.run_pretrain.connect(self.run_foundation_pretrain)
            self.foundation_widget.stop_pretrain.connect(self.stop_foundation_pretrain)
            self.foundation_widget.load_weights.connect(self.load_foundation_weights)
            l_foundation.addWidget(self.foundation_widget)
        except ImportError as e:
            l_foundation.addWidget(QLabel(f"Foundation widget not available: {e}"))
        g_foundation.setLayout(l_foundation)
        l.addWidget(g_foundation)

        # 2. Config
        g_train = QGroupBox("3. Training Configuration")
        l_train = QVBoxLayout()
        form_train = QFormLayout()
        self.mdl_spin_ep = QSpinBox(); self.mdl_spin_ep.setRange(1, 1000); self.mdl_spin_ep.setValue(100)
        self.mdl_spin_bs = QSpinBox(); self.mdl_spin_bs.setRange(1, 128); self.mdl_spin_bs.setValue(4)
        self.mdl_spin_lr = QDoubleSpinBox(); self.mdl_spin_lr.setRange(1e-6, 1.0); self.mdl_spin_lr.setDecimals(5); self.mdl_spin_lr.setValue(0.001)
        self.mdl_combo_loss = QComboBox(); self.mdl_combo_loss.addItems(["DiceLoss", "CrossEntropy", "Dice+CE", "Focal+Dice", "FocalLoss"])
        self.mdl_combo_opt = QComboBox(); self.mdl_combo_opt.addItems(["Adam", "AdamW", "SGD"])
        form_train.addRow("Epochs:", self.mdl_spin_ep)
        form_train.addRow("Batch Size:", self.mdl_spin_bs)
        form_train.addRow("Learning Rate:", self.mdl_spin_lr)
        form_train.addRow("Loss Function:", self.mdl_combo_loss)

        # --- Per-class loss weight editor ---
        self.loss_weight_container = QWidget()
        loss_weight_outer = QVBoxLayout(self.loss_weight_container)
        loss_weight_outer.setContentsMargins(0, 0, 0, 0)
        loss_weight_outer.setSpacing(2)
        
        self.mdl_chk_auto_weight = QCheckBox("Auto (inverse-frequency)")
        self.mdl_chk_auto_weight.setChecked(True)
        self.mdl_chk_auto_weight.setToolTip(
            "When checked, class weights are automatically computed from\n"
            "training data pixel frequencies (rare classes get higher weight).\n"
            "Uncheck to use manual weights below."
        )
        loss_weight_outer.addWidget(self.mdl_chk_auto_weight)
        
        self.loss_weight_row = QHBoxLayout()
        self.loss_weight_row.setSpacing(4)
        self.loss_weight_spins = []  # list of QDoubleSpinBox, index = class id
        loss_weight_outer.addLayout(self.loss_weight_row)
        
        form_train.addRow("Class Weights:", self.loss_weight_container)
        
        # Toggle manual row enabled/disabled
        self.mdl_chk_auto_weight.toggled.connect(lambda auto: self._set_weight_spins_enabled(not auto))
        
        form_train.addRow("Optimizer:", self.mdl_combo_opt)
        l_train.addLayout(form_train)
        
        chkpt_row = QHBoxLayout()
        self.mdl_txt_save_dir = QLineEdit(); self.mdl_txt_save_dir.setPlaceholderText("Save Checkpoints Dir...")
        btn_browse_sd = QPushButton(); btn_browse_sd.setFixedSize(30, 30); btn_browse_sd.setProperty("class", "icon-button"); btn_browse_sd.setIcon(self._load_ui_icon("folder-open.svg", QStyle.SP_DirOpenIcon)); btn_browse_sd.setIconSize(QSize(14, 14)); btn_browse_sd.setToolTip("Browse save directory")
        btn_browse_sd.clicked.connect(lambda: self.mdl_txt_save_dir.setText(QFileDialog.getExistingDirectory(self, "Dir")))
        chkpt_row.addWidget(QLabel("Save Dir:"))
        chkpt_row.addWidget(self.mdl_txt_save_dir); chkpt_row.addWidget(btn_browse_sd)
        l_train.addLayout(chkpt_row)
        
        h_ctrl = QHBoxLayout()
        self.mdl_btn_start = QPushButton("Start Training"); self.mdl_btn_start.setIcon(self._load_ui_icon("player-play.svg")); self.mdl_btn_start.setMinimumHeight(32); self.mdl_btn_start.setStyleSheet("background-color: #2e7d32; font-weight: bold;")
        self.mdl_btn_stop = QPushButton("Stop"); self.mdl_btn_stop.setIcon(self._load_ui_icon("player-stop.svg")); self.mdl_btn_stop.setMinimumHeight(32); self.mdl_btn_stop.setEnabled(False)
        self.mdl_btn_start.clicked.connect(self.start_training)
        self.mdl_btn_stop.clicked.connect(self.stop_training)
        h_ctrl.addWidget(self.mdl_btn_start); h_ctrl.addWidget(self.mdl_btn_stop)
        l_train.addLayout(h_ctrl)
        g_train.setLayout(l_train)
        l.addWidget(g_train)

        # 3. Monitor
        g_mon = QGroupBox("4. Training Monitor")
        l_mon = QVBoxLayout()
        self.mdl_lbl_status = QLabel("Ready to train. Output classes: auto-detected.")
        self.mdl_lbl_status.setStyleSheet("color: #4fc3f7;")
        l_mon.addWidget(self.mdl_lbl_status)
        
        self.mdl_chart = TinyChartWidget()
        self.mdl_chart.setMinimumHeight(280)
        self.mdl_chart.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        l_mon.addWidget(self.mdl_chart, 1)

        
        self.mdl_prev_layout = QHBoxLayout()
        self.mdl_prev_in = QLabel("Input"); self.mdl_prev_in.setAlignment(Qt.AlignCenter); self.mdl_prev_in.setStyleSheet("border: 1px solid #444;")
        self.mdl_prev_gt = QLabel("Ground Truth"); self.mdl_prev_gt.setAlignment(Qt.AlignCenter); self.mdl_prev_gt.setStyleSheet("border: 1px solid #444;")
        self.mdl_prev_out = QLabel("Prediction"); self.mdl_prev_out.setAlignment(Qt.AlignCenter); self.mdl_prev_out.setStyleSheet("border: 1px solid #444;")
        for w in [self.mdl_prev_in, self.mdl_prev_gt, self.mdl_prev_out]: 
            w.setMinimumSize(140, 140); w.setScaledContents(True); self.mdl_prev_layout.addWidget(w)
        l_mon.addLayout(self.mdl_prev_layout)
        g_mon.setLayout(l_mon)
        l.addWidget(g_mon)

        # 5. Inference & Refine (Optimized for 3D)
        g_inf = QGroupBox("5. Inference & Refine")
        l_inf = QVBoxLayout()
        inf_row = QHBoxLayout()
        self.mdl_txt_ckpt = QLineEdit(); self.mdl_txt_ckpt.setPlaceholderText("Select Checkpoint (.pth)...")
        btn_ld = QPushButton(); btn_ld.setFixedSize(30, 30); btn_ld.setProperty("class", "icon-button"); btn_ld.setIcon(self._load_ui_icon("file-search.svg", QStyle.SP_FileDialogContentsView)); btn_ld.setIconSize(QSize(14, 14)); btn_ld.setToolTip("Browse checkpoint")
        btn_ld.clicked.connect(lambda: self.mdl_txt_ckpt.setText(QFileDialog.getOpenFileName(self, "Checkpoint", "", "*.pth")[0]))
        inf_row.addWidget(self.mdl_txt_ckpt); inf_row.addWidget(btn_ld)
        l_inf.addLayout(inf_row)

        f_inf = QFormLayout()
        self.mdl_inf_input = QComboBox()
        self.mdl_inf_input.addItems(["Single Slice (XY)", "Slice Range", "Full Volume"])
        self.mdl_inf_input.currentTextChanged.connect(self._on_inf_mode_changed)
        f_inf.addRow("Input Source:", self.mdl_inf_input)
        
        # Slice range controls (shown for Slice Range mode)
        self.mdl_inf_range_widget = QWidget()
        range_layout = QHBoxLayout(self.mdl_inf_range_widget)
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_layout.addWidget(QLabel("From Z:"))
        self.mdl_inf_z_from = QSpinBox(); self.mdl_inf_z_from.setRange(0, 9999); self.mdl_inf_z_from.setValue(0)
        range_layout.addWidget(self.mdl_inf_z_from)
        range_layout.addWidget(QLabel("To Z:"))
        self.mdl_inf_z_to = QSpinBox(); self.mdl_inf_z_to.setRange(0, 9999); self.mdl_inf_z_to.setValue(999)
        range_layout.addWidget(self.mdl_inf_z_to)
        self.mdl_inf_range_widget.setVisible(False)
        l_inf.addWidget(self.mdl_inf_range_widget)
        
        # Stride control — key 3D optimization
        self.mdl_inf_stride = QSpinBox(); self.mdl_inf_stride.setRange(1, 50); self.mdl_inf_stride.setValue(1)
        self.mdl_inf_stride.setToolTip(
            "Stride = 1: infer every slice (slow but precise)\n"
            "Stride = 5: infer every 5th slice, copy to neighbors (~5x faster)\n"
            "Stride = 10: ~10x faster, good for initial coarse pass\n\n"
            "Since adjacent 3D slices are nearly identical,\n"
            "stride inference gives big speedup with minimal quality loss."
        )
        f_inf.addRow("Z Stride (3D Speed):", self.mdl_inf_stride)
        l_inf.addLayout(f_inf)
        
        # Speed estimate label
        self.mdl_lbl_speed_est = QLabel("")
        self.mdl_lbl_speed_est.setStyleSheet("color: #80cbc4; font-size: 7.5pt; font-style: italic;")
        self.mdl_inf_stride.valueChanged.connect(self._update_speed_estimate)
        self.mdl_inf_input.currentTextChanged.connect(lambda _: self._update_speed_estimate())
        l_inf.addWidget(self.mdl_lbl_speed_est)
        
        self.mdl_btn_infer = QPushButton("Run Inference"); self.mdl_btn_infer.setIcon(self._load_ui_icon("player-play.svg")); self.mdl_btn_infer.setMinimumHeight(32); self.mdl_btn_infer.setStyleSheet("background-color: #0277bd; font-weight: bold;")
        self.mdl_btn_infer.clicked.connect(self.run_inference)
        self.mdl_btn_map = QPushButton("Map Result -> Labelling"); self.mdl_btn_map.setIcon(self._load_ui_icon("arrows-left-right.svg")); self.mdl_btn_map.setMinimumHeight(32); self.mdl_btn_map.setStyleSheet("color: #ffb300;")
        self.mdl_btn_map.clicked.connect(self.map_inference_to_labelling)
        self.mdl_btn_map.setEnabled(False) # Enable when inference is done
        h_inf = QHBoxLayout()
        h_inf.addWidget(self.mdl_btn_infer); h_inf.addWidget(self.mdl_btn_map)
        l_inf.addLayout(h_inf)
        
        # Confidence summary (shown after inference)
        self.mdl_lbl_confidence = QLabel("")
        self.mdl_lbl_confidence.setStyleSheet("color: #ffab40; font-size: 8pt;")
        self.mdl_lbl_confidence.setWordWrap(True)
        l_inf.addWidget(self.mdl_lbl_confidence)
        
        g_inf.setLayout(l_inf)
        l.addWidget(g_inf)

        # 6. Export & Optimization
        g_exp = QGroupBox("6. Export \u0026 Optimize")
        l_exp = QVBoxLayout()
        
        # ONNX precision
        f_exp = QFormLayout()
        self.mdl_combo_precision = QComboBox()
        self.mdl_combo_precision.addItems(["FP32 (Default)", "FP16 (Half — 2x smaller)"])
        self.mdl_combo_precision.setToolTip("FP16 halves model size with minimal accuracy loss.\nIdeal for GPU inference.")
        f_exp.addRow("ONNX Precision:", self.mdl_combo_precision)
        
        # Channel pruning ratio
        self.mdl_spin_prune = QDoubleSpinBox()
        self.mdl_spin_prune.setRange(0.0, 0.9)
        self.mdl_spin_prune.setSingleStep(0.1)
        self.mdl_spin_prune.setValue(0.0)
        self.mdl_spin_prune.setToolTip("0 = no pruning. 0.3 = remove 30% smallest channels.\nReduces model size and speeds up inference.")
        f_exp.addRow("Channel Pruning:", self.mdl_spin_prune)
        l_exp.addLayout(f_exp)
        
        # Export buttons
        h_exp = QHBoxLayout()
        self.mdl_btn_export_onnx = QPushButton("Export ONNX")
        self.mdl_btn_export_onnx.setMinimumHeight(30)
        self.mdl_btn_export_onnx.setStyleSheet("background-color: #00897b; font-weight: bold;")
        self.mdl_btn_export_onnx.setToolTip("Export trained model to ONNX format")
        self.mdl_btn_export_onnx.clicked.connect(self.export_onnx_standalone)
        h_exp.addWidget(self.mdl_btn_export_onnx)
        
        self.mdl_btn_export_trt = QPushButton("Build TensorRT")
        self.mdl_btn_export_trt.setMinimumHeight(30)
        self.mdl_btn_export_trt.setStyleSheet("background-color: #e65100; font-weight: bold;")
        self.mdl_btn_export_trt.setToolTip("Convert ONNX → TensorRT engine for max GPU speed.\nRequires: pip install tensorrt")
        self.mdl_btn_export_trt.clicked.connect(self.build_tensorrt_engine)
        h_exp.addWidget(self.mdl_btn_export_trt)
        l_exp.addLayout(h_exp)
        
        self.mdl_lbl_export = QLabel("Ready to export.")
        self.mdl_lbl_export.setStyleSheet("color: #80cbc4; font-size: 8pt;")
        self.mdl_lbl_export.setWordWrap(True)
        l_exp.addWidget(self.mdl_lbl_export)
        
        g_exp.setLayout(l_exp)
        l.addWidget(g_exp)

        l.addStretch()
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)
        return outer


    def _create_preprocessing_tab(self):
        w = QWidget(); l = QVBoxLayout(w)
        g1 = QGroupBox("Intensity Histogram")
        l1 = QVBoxLayout(); self.hist_view = HistogramWidget(); l1.addWidget(self.hist_view); g1.setLayout(l1); l.addWidget(g1)
        
        g2 = QGroupBox("Contrast & Brightness")
        l2 = QVBoxLayout()
        l2.addWidget(QLabel("Black Level (Display Min)"))
        self.sld_min = QSlider(Qt.Horizontal); self.sld_min.setRange(0, 5000)
        self.sld_min.valueChanged.connect(self.on_contrast_changed); l2.addWidget(self.sld_min)
        
        l2.addWidget(QLabel("White Level (Display Max)"))
        self.sld_max = QSlider(Qt.Horizontal); self.sld_max.setRange(0, 5000); self.sld_max.setValue(255)
        self.sld_max.valueChanged.connect(self.on_contrast_changed); l2.addWidget(self.sld_max)
        
        h_l = QHBoxLayout()
        self.btn_auto_all = QPushButton("Auto (Overall)"); self.btn_auto_all.clicked.connect(self.on_auto_contrast)
        self.btn_auto_roi = QPushButton("Auto (ROI)"); self.btn_auto_roi.setCheckable(True)
        self.btn_auto_roi.clicked.connect(lambda: self.on_tool("contrast_roi"))
        self.btn_reset_contrast = QPushButton("Reset"); self.btn_reset_contrast.clicked.connect(self.on_reset_contrast)
        h_l.addWidget(self.btn_auto_all); h_l.addWidget(self.btn_auto_roi); h_l.addWidget(self.btn_reset_contrast)
        l2.addLayout(h_l); g2.setLayout(l2); l.addWidget(g2)
        l.addStretch(); return w

    def on_contrast_changed(self):
        self.display_min = float(self.sld_min.value())
        self.display_max = float(self.sld_max.value())
        self.sync(light=True)

    def on_auto_contrast(self):
        if self.xy_view.volume_data is not None:
            self.xy_view.auto_contrast()
            self.display_min, self.display_max = self.xy_view.display_min, self.xy_view.display_max
            self.sld_min.blockSignals(True); self.sld_min.setValue(int(self.display_min)); self.sld_min.blockSignals(False)
            self.sld_max.blockSignals(True); self.sld_max.setValue(int(self.display_max)); self.sld_max.blockSignals(False)
            self.sync()

    def on_reset_contrast(self):
        self.display_min, self.display_max = 0.0, 255.0
        self.sld_min.setValue(0); self.sld_max.setValue(255); self.sync()

    def _refresh(self):
        self.list_ui.blockSignals(True)
        self.list_ui.clear()
        
        if hasattr(self, 'combo_fill_source'):
            self.combo_fill_source.blockSignals(True)
            self.combo_fill_target.blockSignals(True)
            self.combo_fill_source.clear()
            self.combo_fill_target.clear()
            self.combo_fill_source.addItem("All Foreground (1 & 2...)", -1)
            
        for cid, info in self.classes.items():
            name = f"ID {cid}: {info['name']}"
            it = QListWidgetItem(name); it.setData(Qt.UserRole, cid)
            it.setCheckState(Qt.Checked if info['visible'] else Qt.Unchecked)
            pix = QPixmap(16,16); pix.fill(QColor(*info['color'])); it.setIcon(QIcon(pix)); self.list_ui.addItem(it)
            if cid == self.current_class_id: it.setSelected(True); self.list_ui.setCurrentItem(it)
            
            if hasattr(self, 'combo_fill_source'):
                self.combo_fill_source.addItem(name, cid)
                self.combo_fill_target.addItem(name, cid)
                
        if hasattr(self, 'combo_fill_source'):
            self.combo_fill_source.blockSignals(False)
            self.combo_fill_target.blockSignals(False)
            
        self.list_ui.blockSignals(False)

    def _default_color_for_class(self, class_id):
        palette = [
            (255, 255, 0),
            (255, 0, 0),
            (0, 220, 140),
            (0, 170, 255),
            (255, 140, 0),
            (190, 90, 255),
            (255, 90, 150),
            (120, 240, 80),
        ]
        if class_id > 0 and class_id <= len(palette):
            return palette[class_id - 1]

        rng = np.random.default_rng(1000 + int(class_id))
        return tuple(int(v) for v in rng.integers(60, 256, size=3))

    def _ensure_mask_classes_exist(self, mask):
        if mask is None:
            return []

        added = []
        for class_id in sorted(int(v) for v in np.unique(mask) if int(v) > 0):
            if class_id in self.classes:
                continue
            self.classes[class_id] = {
                "name": f"Class {class_id}",
                "color": self._default_color_for_class(class_id),
                "visible": True,
            }
            added.append(class_id)

        if added:
            self._refresh()
            if hasattr(self, 'mdl_spin_classes'):
                self.mdl_spin_classes.setValue(max(self.mdl_spin_classes.value(), len(self.classes)))
        return added

    def _sync_active_source_state(self):
        if not getattr(self, 'sources', None) or self.active_source_id is None:
            return

        active_src = next((s for s in self.sources if s.get('id') == self.active_source_id), None)
        if active_src is None:
            return

        active_src['mask'] = self.segmentation_data
        active_src['region_index'] = self.region_index
        active_src['region_label_maps'] = self.region_label_maps
        active_src['hidden_regions'] = set(self.hidden_regions)

    def _load_mask_volume_from_path(self, mask_path):
        path = Path(mask_path)
        suffix = path.suffix.lower()
        source_desc = str(path)

        if suffix in ('.tif', '.tiff'):
            try:
                import tifffile
                mask = tifffile.imread(str(path))  # Fast: reads 3D TIFF stack natively
            except ImportError:
                mask = io.imread(str(path))
            if mask.ndim == 2:
                tiff_files = sorted(glob.glob(str(path.parent / "*.tif*")))
                if len(tiff_files) > 1:
                    try:
                        import tifffile
                        mask = np.stack([tifffile.imread(p) for p in tiff_files], axis=0)
                    except ImportError:
                        mask = np.stack([io.imread(p) for p in tiff_files], axis=0)
                    source_desc = f"{path.parent} ({len(tiff_files)} TIFF slices)"
                else:
                    mask = mask[np.newaxis, :, :]
        elif suffix == '.npy':
            mask = np.load(str(path), allow_pickle=False)
        elif suffix == '.npz':
            with np.load(str(path), allow_pickle=False) as npz_data:
                if 'mask' in npz_data.files:
                    mask = npz_data['mask']
                elif len(npz_data.files) == 1:
                    mask = npz_data[npz_data.files[0]]
                else:
                    raise ValueError("NPZ file must contain a 'mask' array or only one array entry.")
        elif suffix == '.png':
            png_files = sorted(glob.glob(str(path.parent / "*.png")))
            if len(png_files) > 1:
                mask = np.stack([io.imread(p) for p in png_files], axis=0)
                source_desc = f"{path.parent} ({len(png_files)} PNG slices)"
            else:
                mask = io.imread(str(path))[np.newaxis, :, :]
        else:
            raise ValueError(f"Unsupported mask format: {suffix}")

        mask = np.asarray(mask)
        if mask.ndim == 4:
            if mask.shape[-1] == 1:
                mask = mask[..., 0]
            elif mask.shape[-1] in (3, 4):
                first_channel = mask[..., 0]
                if all(np.array_equal(mask[..., idx], first_channel) for idx in range(1, 3)):
                    mask = first_channel
                else:
                    raise ValueError("Mask must be single-channel. RGB color masks are not supported.")
            else:
                raise ValueError(f"Unsupported mask array shape: {mask.shape}")

        if mask.ndim == 2:
            mask = mask[np.newaxis, :, :]
        if mask.ndim != 3:
            raise ValueError(f"Mask must be a 3D array (Z, Y, X). Received shape: {mask.shape}")

        if np.issubdtype(mask.dtype, np.bool_):
            mask = mask.astype(np.uint8)
        elif np.issubdtype(mask.dtype, np.floating):
            if not np.all(np.isfinite(mask)):
                raise ValueError("Mask contains NaN or infinite values.")
            rounded = np.rint(mask)
            if not np.allclose(mask, rounded, atol=1e-6):
                raise ValueError("Float mask must contain integer-like class IDs only.")
            mask = rounded.astype(np.int64)
        elif not np.issubdtype(mask.dtype, np.integer):
            mask = mask.astype(np.int64)

        if mask.size:
            min_val = int(mask.min())
            max_val = int(mask.max())
            if min_val < 0:
                raise ValueError("Mask contains negative class IDs, which are not supported.")
            if max_val > 255:
                raise ValueError("Mask contains class IDs greater than 255, which exceed the editable mask limit.")

        if self.volume_data is None:
            raise ValueError("Please load a volume before importing a mask.")
        if tuple(mask.shape) != tuple(self.volume_data.shape):
            raise ValueError(
                f"Mask shape {tuple(mask.shape)} does not match current volume shape {tuple(self.volume_data.shape)}."
            )

        return np.ascontiguousarray(mask.astype(np.uint8)), source_desc

    def _extract_foreground_for_class_import(self, imported_mask, class_id):
        # Fast path: check if ANY nonzero exists first
        if not np.any(imported_mask > 0):
            return np.zeros_like(imported_mask, dtype=bool), "empty mask"

        # Fast path: if mask contains the target class_id, use it directly
        has_class = np.any(imported_mask == class_id)
        if has_class:
            return imported_mask == class_id, f"class value {class_id}"

        # Binary mask: treat all nonzero as foreground
        return imported_mask > 0, "binary foreground"

    def import_class_mask(self, class_id):
        if self.volume_data is None:
            QMessageBox.warning(self, "No Volume", "Load a volume first.")
            return

        class_name = self.classes.get(class_id, {}).get('name', f"Class {class_id}")
        mask_path, _ = QFileDialog.getOpenFileName(
            self, f"Import {class_name} Mask", "",
            "Mask Files (*.tif *.tiff *.npy *.npz *.png);;All Files (*)"
        )
        if not mask_path:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            imported_mask, source_desc = self._load_mask_volume_from_path(mask_path)
            foreground_mask, import_mode = self._extract_foreground_for_class_import(imported_mask, class_id)
            del imported_mask  # free memory immediately

            if self.segmentation_data is None:
                self.segmentation_data = np.zeros(self.volume_data.shape, dtype=np.uint8)

            # Direct in-place modification (no copy, no history for import)
            self.segmentation_data[self.segmentation_data == class_id] = 0
            assigned = int(np.count_nonzero(foreground_mask))
            self.segmentation_data[foreground_mask] = class_id
            del foreground_mask  # free memory

            self.segmentation_data = np.ascontiguousarray(self.segmentation_data)
            self.display_segmentation_data = None
            self.hidden_regions = set()
            self.active_region_key = None
            self.undo_stack = []
            self.redo_stack = []
            self._ensure_mask_classes_exist(self.segmentation_data)

            # Fast viewer sync - just set data, no heavy computation
            for v in getattr(self, 'all_viewers', []):
                v.segmentation_data = self.segmentation_data
                v.display_segmentation_data = None
                v._mask_cache_key = None  # invalidate mask cache
                v._base_key = None  # force full rebuild

            # Only update XY viewer immediately
            if self.xy_view.volume_data is not None:
                self.xy_view.update_view()
            self._mpr_dirty = True

            # Schedule region analysis in background (non-blocking)
            self.schedule_region_refresh()

        except Exception as e:
            QMessageBox.critical(self, "Import Error", str(e))
        finally:
            QApplication.restoreOverrideCursor()

        QMessageBox.information(self, "Done",
            f"{class_name} imported from {Path(mask_path).name}\n"
            f"Voxels assigned: {assigned:,}")

    def import_class1_mask(self):
        self.import_class_mask(1)

    def import_class2_mask(self):
        self.import_class_mask(2)

    def on_selection_changed(self):
        it = self.list_ui.currentItem()
        if it: self.current_class_id = it.data(Qt.UserRole); self.sync(light=True)
    def on_item_changed(self, item):
        cid = item.data(Qt.UserRole); self.classes[cid]['visible'] = (item.checkState() == Qt.Checked); self.sync()
    def on_add(self):
        nid = max(self.classes.keys())+1 if self.classes else 1
        self.classes[nid] = {"name":f"New Class","color":(np.random.randint(50,255),30,200),"visible":True}; self._refresh(); self.sync()
    def on_del(self):
        if self.list_ui.currentItem(): del self.classes[self.list_ui.currentItem().data(Qt.UserRole)]; self._refresh(); self.sync()
    def on_rename(self, it):
        v, ok = QInputDialog.getText(self,"Rename","Name:",text=self.classes[it.data(Qt.UserRole)]['name'])
        if ok: self.classes[it.data(Qt.UserRole)]['name']=v; self._refresh()  # No sync needed - just a name
    def on_color(self):
        it = self.list_ui.currentItem(); c = QColorDialog.getColor()
        if c.isValid(): self.classes[it.data(Qt.UserRole)]['color'] = (c.red(),c.green(),c.blue()); self._refresh(); self.sync()
    def on_tool(self, t):
        tools = {
            "poly": self.bt_poly, "circle": self.bt_circ, "auto_brush": self.bt_auto,
            "eraser": self.bt_erase, "contrast_roi": self.btn_auto_roi,
            "frame_crop": self.btn_frame_crop,
            "brush": self.bt_brush, "rect_fill": self.bt_rect_fill,
            "flood_fill": self.bt_flood, "sam_click": self.bt_sam_click,
        }
        btn = tools.get(t)
        if btn:
            self.active_tool = t if btn.isChecked() else None
            for name, b in tools.items():
                if b and name != t: b.setChecked(False)
        if t == "auto_brush" and self.bt_auto.isChecked(): self.active_tool = "auto_roi"
        # SAM click requires SAM model
        if t == "sam_click" and self.bt_sam_click.isChecked() and self.sam_predictor is None:
            QMessageBox.warning(self, "No SAM Model", "Please load a SAM model first!\n(Auto Labelling → Load SAM)")
            self.bt_sam_click.setChecked(False)
            self.active_tool = None
        # Clear SAM points when switching away from sam_click
        if self.active_tool != 'sam_click':
            for v in getattr(self, 'all_viewers', []):
                if v.sam_positive_points or v.sam_negative_points:
                    v.sam_click_cancel()
        # Update status label to show current tool
        tool_name = self.active_tool if self.active_tool else "None"
        if hasattr(self, 'lbl_active_tool'):
            self.lbl_active_tool.setText(f"Active: {tool_name.replace('_',' ').title()}")
        self.sync(light=True)

    def _sam_click_accept(self):
        """Accept SAM point-click on all viewers."""
        for v in getattr(self, 'all_viewers', []):
            v.sam_click_accept()
        self.sync()

    def _sam_click_cancel(self):
        """Cancel SAM point-click on all viewers."""
        for v in getattr(self, 'all_viewers', []):
            v.sam_click_cancel()
        self.sync(light=True)

    def _set_flood_tolerance(self, v):
        """Update flood fill tolerance on all viewers."""
        for viewer in getattr(self, 'all_viewers', []):
            viewer.flood_fill_tolerance = v

    def on_auto_mode_changed(self, mode):
        if mode == "SAM" and self.sam_predictor is None:
            QMessageBox.warning(self, "No SAM Model", "Please load a SAM model first!"); self.rb_otsu.setChecked(True); return
        self.auto_mode = mode
        self.sync()

    def on_brush_slider_changed(self, v):
        self.brush_size = v; self.sync(brush_only=True)

    def on_opacity_changed(self, v):
        self.mask_opacity = v / 100.0; self.sync(light=True)

    def on_mode_changed(self, v):
        self.threshold_mode = v; self.sync(light=True)

    def on_load_sam(self):
        if not SAM_AVAIL: QMessageBox.warning(self,"SAM Not Available","Install 'segment-anything' and 'torch' first!"); return
        p, _ = QFileDialog.getOpenFileName(self, "Select SAM Model", "", "Model (*.pth *.pt)")
        if p:
            self.pg_sam = QProgressDialog("Loading SAM Model...", "Cancel", 0, 0, self); self.pg_sam.show()
            self.sam_thread = SAMLoadThread(p); self.sam_thread.finished.connect(self._sam_loaded); self.sam_thread.start()
    def _sam_loaded(self, pred, err):
        self.pg_sam.close()
        if err: QMessageBox.critical(self,"Error",err); return
        self.sam_predictor = pred; self.rb_sam.setEnabled(True); self.rb_sam.setChecked(True)
        QMessageBox.information(self,"Success","SAM Model Loaded! Auto Brush now uses SAM.")
        self.btn_load_sam.setText("SAM Model: Loaded"); self.sync()

    # ── Grounding DINO + SAM handlers ──
    def _load_dino_sam2_models(self):
        """Pre-load Grounding DINO and SAM models in background thread."""
        self.btn_dino_load.setEnabled(False)
        self.lbl_dino_status.setText("Loading models... (downloads ~700MB on first use)")
        self.lbl_dino_status.setStyleSheet("color: #ffb74d; font-size: 8pt;")
        
        from inno3d.services.dino_sam2 import DinoSam2LoadThread
        # If SAM already loaded via the Load SAM button, inject it
        if self.sam_predictor is not None:
            from inno3d.services import dino_sam2
            dino_sam2._sam_predictor = self.sam_predictor
        self._dino_loader = DinoSam2LoadThread()
        self._dino_loader.progress.connect(lambda msg: self.lbl_dino_status.setText(msg))
        self._dino_loader.finished.connect(self._on_dino_sam2_loaded)
        self._dino_loader.start()

    def _on_dino_sam2_loaded(self, success, error):
        self.btn_dino_load.setEnabled(True)
        if success:
            self.lbl_dino_status.setText("✓ Grounding DINO + SAM 2 models loaded and ready.")
            self.lbl_dino_status.setStyleSheet("color: #81c784; font-size: 8pt;")
            self.btn_dino_load.setText("Models Loaded ✓")
            self.btn_dino_load.setStyleSheet("color: #81c784; font-weight: bold;")
        else:
            self.lbl_dino_status.setText(f"Error loading models: {error[:200]}")
            self.lbl_dino_status.setStyleSheet("color: #ff5252; font-size: 8pt;")

    def _run_dino_sam2(self):
        """Run text-prompted segmentation on the current slice or range."""
        if self.volume_data is None or self.segmentation_data is None:
            QMessageBox.warning(self, "No Data", "Load a volume and create segmentation first.")
            return
        
        prompt_text = self.txt_dino_prompt.text().strip()
        if not prompt_text:
            QMessageBox.warning(self, "No Prompt", 
                "Enter a text prompt describing objects to find.\n"
                "Examples: 'void' or 'solder ball. crack. void'")
            return
        
        # Parse prompts
        prompts = [p.strip() for p in prompt_text.split('.') if p.strip()]
        
        # Build class mapping: match prompt words to existing class names
        class_mapping = {}
        if self.chk_dino_auto_map.isChecked():
            for prompt in prompts:
                prompt_lower = prompt.lower()
                for cid, info in self.classes.items():
                    name_lower = info.get('name', '').lower()
                    if prompt_lower in name_lower or name_lower in prompt_lower:
                        class_mapping[prompt_lower] = cid
                        break
                if prompt_lower not in class_mapping:
                    class_mapping[prompt_lower] = self.current_class_id
        else:
            for prompt in prompts:
                class_mapping[prompt.lower()] = self.current_class_id
        
        # Determine which slices to process
        mode = self.combo_dino_target.currentText()
        z_current = getattr(self.xy_view, 'current_slice', 0)
        
        if mode == "Current Slice":
            z_list = [z_current]
        elif mode == "Slice Range":
            z_from = self.spin_dino_z_from.value()
            z_to = min(self.spin_dino_z_to.value(), self.volume_data.shape[0] - 1)
            z_list = list(range(z_from, z_to + 1))
        else:  # Full Volume
            z_list = list(range(self.volume_data.shape[0]))
        
        # Process
        self.save_history_full(z_list=z_list)
        self.btn_dino_run.setEnabled(False)
        self.lbl_dino_status.setText(f"Processing {len(z_list)} slice(s)...")
        self.lbl_dino_status.setStyleSheet("color: #ffb74d; font-size: 8pt;")
        
        from inno3d.services.dino_sam2 import DinoSam2Worker
        
        # For multi-slice, run sequentially via a helper
        self._dino_z_queue = list(z_list)
        self._dino_prompts = prompts
        self._dino_class_mapping = class_mapping
        self._dino_total = len(z_list)
        self._dino_processed = 0
        self._dino_total_objects = 0
        self._process_next_dino_slice()

    def _process_next_dino_slice(self):
        """Process the next slice in the DINO+SAM2 queue."""
        if not self._dino_z_queue:
            # All done
            self.btn_dino_run.setEnabled(True)
            self.lbl_dino_status.setText(
                f"✓ Done! Processed {self._dino_total} slices, "
                f"{self._dino_total_objects} objects segmented."
            )
            self.lbl_dino_status.setStyleSheet("color: #81c784; font-size: 8pt;")
            self.sync()
            return
        
        z = self._dino_z_queue.pop(0)
        self._dino_processed += 1
        self.lbl_dino_status.setText(
            f"Slice {self._dino_processed}/{self._dino_total} (Z={z})..."
        )
        
        from inno3d.services.dino_sam2 import DinoSam2Worker
        
        # Get slice image
        img = self.volume_data[z]
        if img.dtype != np.uint8:
            img = ((img - img.min()) / max(1, img.max() - img.min()) * 255).astype(np.uint8)
        
        self._dino_worker = DinoSam2Worker(
            img, self._dino_prompts,
            class_mapping=self._dino_class_mapping,
            box_threshold=self.spin_dino_conf.value(),
            nms_iou=self.spin_dino_iou.value(),
        )
        self._dino_current_z = z
        self._dino_worker.finished.connect(self._on_dino_slice_done)
        self._dino_worker.start()

    def _on_dino_slice_done(self, mask_2d, info):
        """Handle result from one slice of DINO+SAM2 processing."""
        z = self._dino_current_z
        n = info.get('n_accepted', info.get('n_detected', 0))
        self._dino_total_objects += n
        
        # Merge mask into segmentation volume (only overwrite where mask > 0)
        if mask_2d is not None and np.any(mask_2d > 0):
            self.segmentation_data[z][mask_2d > 0] = mask_2d[mask_2d > 0]
            for v in self.all_viewers:
                v._mask_version += 1
        
        # Continue with next slice
        self._process_next_dino_slice()


    # ── AutoPrompt-SAM handlers ──

    def _run_autoprompt_sam(self):
        """Run intensity-guided auto-segmentation pipeline."""
        if self.volume_data is None or self.segmentation_data is None:
            QMessageBox.warning(self, "No Data", "Load a volume and create segmentation first.")
            return

        # Determine class IDs
        # Instead of searching for 'tgv', use the currently selected class in combo_fill_target
        if hasattr(self, 'combo_fill_target') and self.combo_fill_target.count() > 0:
            tgv_id = self.combo_fill_target.currentData()
            if tgv_id is None:
                tgv_id = 1
        else:
            tgv_id = 1

        # For voids, use tgv_id + 1 by default, or search for 'void' if it exists
        void_id = tgv_id + 1
        for cid, info in self.classes.items():
            name = info.get('name', '').lower()
            if 'void' in name:
                void_id = cid

        # Determine slices
        mode = self.combo_auto_target.currentText()
        z_current = getattr(self.xy_view, 'current_slice', 0)
        if mode == "Current Slice":
            z_list = [z_current]
        elif mode == "Slice Range":
            z_from = self.spin_auto_z_from.value()
            z_to = min(self.spin_auto_z_to.value(), self.volume_data.shape[0] - 1)
            z_list = list(range(z_from, z_to + 1))
        else:
            z_list = list(range(self.volume_data.shape[0]))

        # Save undo
        self.save_history_full(z_list=z_list)

        # Pipeline parameters
        self._auto_z_queue = list(z_list)
        self._auto_total = len(z_list)
        self._auto_processed = 0
        self._auto_total_tgv = 0
        self._auto_total_void = 0
        self._auto_tgv_id = tgv_id
        self._auto_void_id = void_id

        self.btn_auto_run.setEnabled(False)
        self.lbl_auto_status.setText(f"Processing {len(z_list)} slice(s)...")
        self.lbl_auto_status.setStyleSheet("color: #ffb74d; font-size: 8pt;")

        self._process_next_auto_slice()

    def _process_next_auto_slice(self):
        """Process the next slice in the AutoPrompt-SAM queue."""
        if not self._auto_z_queue:
            self.btn_auto_run.setEnabled(True)
            self.lbl_auto_status.setText(
                f"Done! {self._auto_total} slices: "
                f"{self._auto_total_tgv} TGVs + {self._auto_total_void} Voids"
            )
            self.lbl_auto_status.setStyleSheet("color: #81c784; font-size: 8pt;")
            self.sync()
            return

        z = self._auto_z_queue.pop(0)
        self._auto_processed += 1
        self.lbl_auto_status.setText(
            f"Slice {self._auto_processed}/{self._auto_total} (Z={z})..."
        )

        from inno3d.services.autoprompt_sam import AutoPromptSAMWorker

        img = self.volume_data[z]
        if img.dtype != np.uint8:
            mn, mx = img.min(), img.max()
            img = ((img - mn) / max(1, mx - mn) * 255).astype(np.uint8)

        # Use SAM predictor if available and checkbox is checked
        sam_pred = self.sam_predictor if self.chk_auto_sam.isChecked() else None

        self._auto_worker = AutoPromptSAMWorker(
            img, sam_predictor=sam_pred,
            tgv_class_id=self._auto_tgv_id,
            void_class_id=self._auto_void_id,
            bright_percentile=int(self.spin_auto_bright.value()),
            min_tgv_area=self.spin_auto_min_area.value(),
            dark_factor=self.spin_auto_dark.value(),
            min_void_area=self.spin_auto_min_void.value(),
            use_sam=(sam_pred is not None),
            detect_voids=self.chk_auto_voids.isChecked(),
        )
        self._auto_current_z = z
        self._auto_worker.finished.connect(self._on_auto_slice_done)
        self._auto_worker.start()

    def _on_auto_slice_done(self, mask_2d, info):
        """Handle result from one slice of AutoPrompt-SAM processing."""
        z = self._auto_current_z
        self._auto_total_tgv += info.get('n_tgv', 0)
        self._auto_total_void += info.get('n_void', 0)

        if mask_2d is not None and np.any(mask_2d > 0):
            self.segmentation_data[z][mask_2d > 0] = mask_2d[mask_2d > 0]
            for v in self.all_viewers:
                v._mask_version += 1

        self._process_next_auto_slice()

    def on_slice_synced(self, val):
        """Called when any viewer's slice slider changes. If crosshair is on, sync crosshair positions."""
        sender = self.sender()
        any_crosshair = any(v.show_crosshair for v in self.all_viewers)
        if any_crosshair and sender is not None and self.volume_data is not None:
            # Display-space positions (all views, after transforms):
            # xy_view: (world_x, world_y)
            # xz_view: (world_x, world_z)  [no flipud: Z increases downward]
            # yz_view: (world_z, world_y)   [transpose: col=Z, row=Y]
            xy_pos = self.xy_view.crosshair_data_pos or (self.volume_data.shape[2]//2, self.volume_data.shape[1]//2)
            xz_pos = self.xz_view.crosshair_data_pos or (self.volume_data.shape[2]//2, self.xy_view.current_slice)
            yz_pos = self.yz_view.crosshair_data_pos or (self.xy_view.current_slice, self.volume_data.shape[1]//2)

            world_x  = xy_pos[0]
            world_y  = xy_pos[1]
            world_z  = xz_pos[1]      # xz: row=world_z (no flip)
            world_z2 = yz_pos[0]      # yz: col=world_z (transpose)

            if sender is self.xy_view:
                # Z changed: update Z-row in XZ, Z-col in YZ
                new_z = val
                self.xz_view.crosshair_data_pos = (world_x, new_z)   # xz: row=world_z downward
                self.yz_view.crosshair_data_pos = (new_z,   world_y)  # yz: col=world_z rightward
            elif sender is self.xz_view:
                # Y changed: update Y-row in XY and Y-row in YZ
                new_y = val
                self.xy_view.crosshair_data_pos = (world_x, new_y)
                self.yz_view.crosshair_data_pos = (world_z2, new_y)
            elif sender is self.yz_view:
                # X changed: update X-col in XY and X-col in XZ
                new_x = val
                self.xy_view.crosshair_data_pos = (new_x, world_y)
                self.xz_view.crosshair_data_pos = (new_x, world_z)

            for v in self.get_visible_viewers():
                if v.show_crosshair:
                    v.update_view()
        self.sync()

    def on_otsu(self):
        if self.volume_data is None: return
        self.save_history_full()
        # Full 3D Otsu
        thresh = filters.threshold_otsu(self.volume_data)
        mask = (self.volume_data > thresh) if self.threshold_mode == "Above" else (self.volume_data < thresh)
        self.segmentation_data[mask] = self.current_class_id
        QMessageBox.information(self,"Success",f"Otsu 3D ({self.threshold_mode}) applied. Threshold: {thresh}")
        self.sync()

    def _get_threshold_value(self, slice_data=None):
        """Get threshold value — either from Otsu auto-calc or manual slider."""
        if self.rb_thresh_manual.isChecked():
            return self.slider_manual_thresh.value()
        else:
            if slice_data is not None:
                return filters.threshold_otsu(slice_data)
            elif self.volume_data is not None:
                z = getattr(self.xy_view, 'current_slice', 0)
                return filters.threshold_otsu(self.volume_data[z])
            return 128

    def _preview_threshold(self):
        """Preview thresholding on the current slice as a temporary display overlay."""
        if self.volume_data is None or self.segmentation_data is None:
            return
        z = getattr(self.xy_view, 'current_slice', 0)
        slice_data = self.volume_data[z]
        thresh = self._get_threshold_value(slice_data)
        direction = self.combo_mode.currentText()

        if direction == "Above":
            mask = slice_data > thresh
        else:
            mask = slice_data < thresh

        # Temporarily show preview on display_segmentation_data
        if self.display_segmentation_data is None:
            self.display_segmentation_data = self.segmentation_data.copy()
        preview = self.display_segmentation_data[z].copy()
        preview[mask] = self.current_class_id
        self.display_segmentation_data[z] = preview

        for v in self.all_viewers:
            v.display_segmentation_data = self.display_segmentation_data
            v._mask_version += 1
        self.xy_view.update_view()

        n_px = int(np.sum(mask))
        pct = 100 * n_px / mask.size
        mode_str = "Manual" if self.rb_thresh_manual.isChecked() else "Otsu"
        QMessageBox.information(
            self, "Threshold Preview",
            f"Mode: {mode_str} | Direction: {direction}\n"
            f"Threshold: {thresh:.1f}\n"
            f"Pixels selected: {n_px:,} ({pct:.1f}%)\n\n"
            f"This is a preview. Click 'Apply to Slice' to commit."
        )
        # Restore display data
        self.display_segmentation_data[z] = self.segmentation_data[z].copy()
        for v in self.all_viewers:
            v._mask_version += 1
        self.sync()

    def _apply_threshold_to(self, target="slice"):
        """Apply thresholding to current slice or full volume."""
        if self.volume_data is None or self.segmentation_data is None:
            return
        direction = self.combo_mode.currentText()
        cls_id = self.current_class_id

        if target == "slice":
            z = getattr(self.xy_view, 'current_slice', 0)
            self.save_history(changed_slices=[z])
            slice_data = self.volume_data[z]
            thresh = self._get_threshold_value(slice_data)
            mask = (slice_data > thresh) if direction == "Above" else (slice_data < thresh)
            self.segmentation_data[z][mask] = cls_id
            for v in self.all_viewers:
                v._mask_version += 1
            self.sync()
        else:
            reply = QMessageBox.question(
                self, "Confirm",
                f"Apply threshold ({direction}) to ALL {self.volume_data.shape[0]} slices?\n"
                f"Class: {cls_id}",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
            self.save_history_full()
            if self.rb_thresh_manual.isChecked():
                thresh = self.slider_manual_thresh.value()
                mask = (self.volume_data > thresh) if direction == "Above" else (self.volume_data < thresh)
                self.segmentation_data[mask] = cls_id
            else:
                thresh = filters.threshold_otsu(self.volume_data)
                mask = (self.volume_data > thresh) if direction == "Above" else (self.volume_data < thresh)
                self.segmentation_data[mask] = cls_id
            for v in self.all_viewers:
                v._mask_version += 1
            QMessageBox.information(
                self, "Success",
                f"Threshold ({direction}) applied to full volume.\n"
                f"Value: {thresh:.1f} | Class: {cls_id}"
            )
            self.sync()

    def sync(self, brush_only=False, save_state=False, from_viewer=False, min_max=None, light=False, source_viewer=None):
        if save_state:
            self.save_history()
        vs = {cid for cid, info in self.classes.items() if info['visible']}

        if from_viewer and min_max:
            self.display_min, self.display_max = min_max
        elif from_viewer:
            self.display_min, self.display_max = self.xy_view.display_min, self.xy_view.display_max

        # === FAST PATH: brush_only refreshes ONLY the drawing viewer ===
        if brush_only:
            color_dict = {cid: val['color'] for cid, val in self.classes.items()}
            for v in self.all_viewers:
                v.active_class = self.current_class_id
                v.brush_size = self.brush_size
                v.visible_classes = vs
                v.mask_opacity = self.mask_opacity
                v.class_colors = color_dict
                v.segmentation_data = self.segmentation_data
                v.display_segmentation_data = self.display_segmentation_data
            # Only re-render the viewer that triggered the draw (avoid lag)
            draw_v = source_viewer or self.xy_view
            if draw_v.volume_data is not None:
                draw_v.update_view()
            self._mpr_dirty = True
            return

        active_frame = self.get_active_frame()
        if not light:
            self.sld_min.blockSignals(True); self.sld_min.setValue(int(self.display_min)); self.sld_min.blockSignals(False)
            self.sld_max.blockSignals(True); self.sld_max.setValue(int(self.display_max)); self.sld_max.blockSignals(False)
            self.slider_brush.blockSignals(True); self.slider_brush.setValue(self.brush_size); self.slider_brush.blockSignals(False)
            self.slider_opacity.blockSignals(True); self.slider_opacity.setValue(int(self.mask_opacity * 100)); self.slider_opacity.blockSignals(False)
            self.schedule_region_refresh()

        # Pre-compute shared state ONCE
        color_dict = {cid: val['color'] for cid, val in self.classes.items()}
        if self.enable_connected_components:
            region_box_map = self.build_active_region_box_map()
        else:
            region_box_map = {'axial': [], 'coronal': [], 'sagittal': []}
        # Only render viewers that are actually visible on screen
        visible_set = set(self.get_visible_viewers()) if hasattr(self, 'get_visible_viewers') else set(self.all_viewers)
        
        for v in self.all_viewers:
            v.active_tool = self.active_tool  # Allow drawing on all 3 planes
            v.active_class, v.brush_size, v.visible_classes = self.current_class_id, self.brush_size, vs
            v.mask_opacity, v.threshold_mode = self.mask_opacity, self.threshold_mode
            v.use_sam = (self.auto_mode == "SAM")
            v.sam_predictor = self.sam_predictor
            v.display_min, v.display_max = self.display_min, self.display_max
            v.class_colors = color_dict
            v.segmentation_data = self.segmentation_data
            v.display_segmentation_data = self.display_segmentation_data
            
            if v is self.xy_view:
                v.frame_create_callback = self.add_crop_frame
                if getattr(self, 'isolate_roi', False) and active_frame is not None:
                    v.set_view_roi(active_frame['rect'])
                else:
                    v.set_view_roi(None)
                show_frames = getattr(self.btn_show_frames, 'isChecked', lambda: True)()
                if show_frames:
                    v.set_frame_boxes(self.get_frames_for_slice(v.current_slice), self.active_frame_id)
                else:
                    v.set_frame_boxes([], None)
            else:
                v.frame_create_callback = None
                v.set_view_roi(None)
                v.set_frame_boxes([], None)
            if self.enable_connected_components:
                if not getattr(v, 'highlight_region_key', None):
                    v.set_region_boxes(region_box_map.get(v.orientation, []))
                else:
                    v.set_region_boxes([])
            else:
                v.set_region_boxes([])

        # === KEY OPTIMIZATION: Only update XY viewer during normal sync ===
        # XZ/YZ only update when user explicitly clicks "Refresh MPR" button
        if self.xy_view.volume_data is not None:
            self.xy_view.update_view()
        
        # Mark MPR as dirty (XZ/YZ need refresh)
        self._mpr_dirty = True
        self._set_mpr_btn_dirty(True)
        # Skip histogram during brush-only or light sync
        if not brush_only and not light and self.xy_view.volume_data is not None:
            self.hist_view.set_data(self.xy_view.volume_data[self.xy_view.current_slice])
        if brush_only or light:
            return
        self.refresh_frame_table()
        self.refresh_mother_table()
        if self.enable_connected_components:
            self.refresh_region_table()

    def save_history(self, changed_slices=None):
        if self.segmentation_data is None:
            return
        if changed_slices is None:
            z = getattr(self.xy_view, 'current_slice', 0)
            z_max = self.segmentation_data.shape[0] - 1
            changed_slices = list(range(max(0, z - 1), min(z_max, z + 1) + 1))
        snapshot = {}
        for z in changed_slices:
            if 0 <= z < self.segmentation_data.shape[0]:
                snapshot[z] = self.segmentation_data[z].copy()
        self.undo_stack.append(snapshot)
        self.redo_stack = []
        if len(self.undo_stack) > 30:
            self.undo_stack.pop(0)

    def save_history_full(self, z_list=None):
        """Save undo snapshot — either specific slices or sampled subset to avoid OOM."""
        if self.segmentation_data is None:
            return
        try:
            if z_list is None:
                # Sample at most 50 evenly-spaced slices to cap memory usage
                total_z = self.segmentation_data.shape[0]
                if total_z <= 50:
                    z_list = list(range(total_z))
                else:
                    step = max(1, total_z // 50)
                    z_list = list(range(0, total_z, step))
            snapshot = {}
            for z in z_list:
                if 0 <= z < self.segmentation_data.shape[0]:
                    snapshot[z] = self.segmentation_data[z].copy()
            self.undo_stack.append(snapshot)
            self.redo_stack = []
            if len(self.undo_stack) > 10:
                self.undo_stack.pop(0)
        except (MemoryError, np.core._exceptions._ArrayMemoryError):
            print("[Warning] Not enough memory for full undo snapshot — skipping")

    def undo(self):
        if not self.undo_stack or self.segmentation_data is None:
            return
        old_snapshot = self.undo_stack[-1]
        redo_snapshot = {}
        for z in old_snapshot:
            if 0 <= z < self.segmentation_data.shape[0]:
                redo_snapshot[z] = self.segmentation_data[z].copy()
        self.redo_stack.append(redo_snapshot)
        snapshot = self.undo_stack.pop()
        for z, data in snapshot.items():
            if 0 <= z < self.segmentation_data.shape[0]:
                self.segmentation_data[z] = data
        if self.display_segmentation_data is not None:
            for z in snapshot:
                if 0 <= z < self.display_segmentation_data.shape[0]:
                    self.display_segmentation_data[z] = self.segmentation_data[z].copy()
        for v in self.all_viewers:
            v._mask_version += 1
        self.sync()

    def redo(self):
        if not self.redo_stack or self.segmentation_data is None:
            return
        old_snapshot = self.redo_stack[-1]
        undo_snapshot = {}
        for z in old_snapshot:
            if 0 <= z < self.segmentation_data.shape[0]:
                undo_snapshot[z] = self.segmentation_data[z].copy()
        self.undo_stack.append(undo_snapshot)
        snapshot = self.redo_stack.pop()
        for z, data in snapshot.items():
            if 0 <= z < self.segmentation_data.shape[0]:
                self.segmentation_data[z] = data
        if self.display_segmentation_data is not None:
            for z in snapshot:
                if 0 <= z < self.display_segmentation_data.shape[0]:
                    self.display_segmentation_data[z] = self.segmentation_data[z].copy()
        for v in self.all_viewers:
            v._mask_version += 1
        self.sync()

    def toggle_isolate_roi(self, state):
        self.isolate_roi = bool(state)
        self.sync()

    def fill_holes_2d(self):
        if self.segmentation_data is None: return
        source_idx = self.combo_fill_source.currentData()
        target_idx = self.combo_fill_target.currentData()
        
        if source_idx is None or target_idx is None: return
            
        source_name = "All Foreground Classes" if source_idx == -1 else f"Class {source_idx}"
        reply = QMessageBox.question(self, "Confirm Fill", f"Find all 2D holes enclosed by {source_name} on XY planes and fill them with Class {target_idx}?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            import scipy.ndimage as ndi
            self.save_history_full()
            
            if source_idx == -1:
                mask = (self.segmentation_data > 0)
            else:
                mask = (self.segmentation_data == source_idx)
                
            holes_mask = np.zeros_like(mask, dtype=bool)
            
            # Slice-by-slice 2D hole filling on XY planes
            for z in range(mask.shape[0]):
                slice_mask = mask[z]
                holes_mask[z] = ndi.binary_fill_holes(slice_mask) & ~slice_mask
            
            # Count them
            holes_count = np.sum(holes_mask)
            if holes_count == 0:
                QMessageBox.information(self, "No Holes Found", f"No enclosed 2D holes were detected inside {source_name}.")
                return
                
            # Fill the holes with target_idx
            self.segmentation_data[holes_mask] = target_idx
            if self.display_segmentation_data is not None:
                self.display_segmentation_data[holes_mask] = target_idx
                
            self.refresh_region_index()
            self.sync()
            QMessageBox.information(self, "Success", f"Filled {holes_count} voxels successfully. Check the visual results.")

    def clear_regions_in_range(self):
        if self.segmentation_data is None: return
        start = self.spin_clear_start.value()
        end = self.spin_clear_end.value()
        if start > end: start, end = end, start
            
        max_slice = self.segmentation_data.shape[0] - 1
        start = max(0, min(start, max_slice))
        end = max(0, min(end, max_slice))
        
        reply = QMessageBox.question(self, "Confirm Clear", f"Are you sure you want to completely erase ALL objects from slice {start} to {end}?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.save_history(changed_slices=list(range(start, end + 1)))
            self.segmentation_data[start:end+1] = 0
            if self.display_segmentation_data is not None:
                self.display_segmentation_data[start:end+1] = 0
            
            self.refresh_region_index()
            self.sync()
            QMessageBox.information(self, "Success", f"Cleared regions from slice {start} to {end}.")

    def propagate_labels(self):
        """Copy/propagate labels from current slice to adjacent slices."""
        if self.segmentation_data is None:
            QMessageBox.warning(self, "No Data", "Load a volume first.")
            return
        
        z_curr = self.xy_view.current_slice
        z_max = self.segmentation_data.shape[0] - 1
        
        # Check if current slice has labels
        curr_fg = np.count_nonzero(self.segmentation_data[z_curr])
        if curr_fg == 0:
            QMessageBox.warning(self, "Empty Slice", 
                f"Slice {z_curr} has no labels. Navigate to a labeled slice first.")
            return
        
        dlg = QDialog(self)
        dlg.setWindowTitle("Propagate Labels Across Slices")
        dlg.setMinimumWidth(420)
        lay = QVBoxLayout(dlg)
        
        # Source info
        info = QLabel(f"Source: Slice {z_curr}  |  {curr_fg:,} labeled pixels")
        info.setStyleSheet("font-weight: bold; font-size: 10pt;")
        lay.addWidget(info)
        lay.addSpacing(8)
        
        # Direction & Range
        form = QFormLayout()
        
        combo_dir = QComboBox()
        combo_dir.addItems(["Forward (+Z)", "Backward (-Z)", "Both directions"])
        form.addRow("Direction:", combo_dir)
        
        spin_count = QSpinBox()
        spin_count.setRange(1, z_max)
        spin_count.setValue(min(10, z_max))
        form.addRow("Number of slices:", spin_count)
        
        # Mode
        combo_mode = QComboBox()
        combo_mode.addItems([
            "Copy Exact — duplicate labels identically",
            "Dilate Shrink — gradual erosion per slice",
            "Interpolate — morph between two labeled slices",
        ])
        form.addRow("Mode:", combo_mode)
        
        # Erosion rate
        spin_erosion = QSpinBox()
        spin_erosion.setRange(1, 20)
        spin_erosion.setValue(2)
        spin_erosion.setEnabled(False)
        spin_erosion.setToolTip("Pixels to erode per Z step (Dilate Shrink mode)")
        combo_mode.currentIndexChanged.connect(lambda i: spin_erosion.setEnabled(i == 1))
        form.addRow("Erosion per slice (px):", spin_erosion)
        
        # Interpolation target
        spin_target = QSpinBox()
        spin_target.setRange(0, z_max)
        spin_target.setValue(min(z_curr + 10, z_max))
        spin_target.setEnabled(False)
        combo_mode.currentIndexChanged.connect(lambda i: spin_target.setEnabled(i == 2))
        form.addRow("Interp. target slice:", spin_target)
        
        # Class filter
        chk_all_cls = QCheckBox("All classes")
        chk_all_cls.setChecked(True)
        form.addRow("Classes:", chk_all_cls)
        
        # Overwrite mode
        chk_overwrite = QCheckBox("Overwrite existing labels")
        chk_overwrite.setChecked(False)
        form.addRow("", chk_overwrite)
        
        lay.addLayout(form)
        
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        lay.addWidget(btn_box)
        
        if dlg.exec() != QDialog.Accepted:
            return
        
        mode_idx = combo_mode.currentIndex()
        direction = combo_dir.currentIndex()  # 0=fwd, 1=bwd, 2=both
        n_slices = spin_count.value()
        erosion_px = spin_erosion.value()
        overwrite = chk_overwrite.isChecked()
        
        # Build target slice list
        targets = []
        if direction in (0, 2):  # forward
            targets += list(range(z_curr + 1, min(z_curr + n_slices + 1, z_max + 1)))
        if direction in (1, 2):  # backward
            targets += list(range(z_curr - 1, max(z_curr - n_slices - 1, -1), -1))
        targets = sorted(set(targets))
        
        if not targets:
            QMessageBox.warning(self, "No Targets", "No valid target slices in range.")
            return
        
        # Save undo
        self.save_history(changed_slices=targets)
        
        source_mask = self.segmentation_data[z_curr].copy()
        applied = 0
        
        if mode_idx == 0:
            # COPY EXACT
            for z in targets:
                if overwrite:
                    self.segmentation_data[z] = source_mask.copy()
                else:
                    empty = self.segmentation_data[z] == 0
                    self.segmentation_data[z][empty] = source_mask[empty]
                applied += 1
                
        elif mode_idx == 1:
            # DILATE SHRINK — erode progressively
            import scipy.ndimage as ndi
            for z in sorted(targets):
                dist = abs(z - z_curr)
                erode_iter = dist * erosion_px
                if erode_iter <= 0:
                    eroded = source_mask.copy()
                else:
                    eroded = np.zeros_like(source_mask)
                    for cls_id in np.unique(source_mask):
                        if cls_id == 0:
                            continue
                        cls_mask = (source_mask == cls_id)
                        cls_eroded = ndi.binary_erosion(cls_mask, iterations=erode_iter)
                        eroded[cls_eroded] = cls_id
                
                if np.count_nonzero(eroded) == 0:
                    continue  # fully eroded, skip
                
                if overwrite:
                    self.segmentation_data[z] = eroded
                else:
                    empty = self.segmentation_data[z] == 0
                    self.segmentation_data[z][empty] = eroded[empty]
                applied += 1
                
        elif mode_idx == 2:
            # INTERPOLATE between source and target
            import scipy.ndimage as ndi
            z_target = spin_target.value()
            target_mask = self.segmentation_data[z_target].copy()
            
            if np.count_nonzero(target_mask) == 0:
                QMessageBox.warning(self, "Empty Target", 
                    f"Target slice {z_target} has no labels. Label it first for interpolation.")
                return
            
            z_lo, z_hi = min(z_curr, z_target), max(z_curr, z_target)
            if z_lo == z_hi:
                return
            
            mask_lo = self.segmentation_data[z_lo].copy()
            mask_hi = self.segmentation_data[z_hi].copy()
            
            # Distance-field interpolation per class
            for z in range(z_lo + 1, z_hi):
                t = (z - z_lo) / (z_hi - z_lo)  # 0..1
                interp_mask = np.zeros_like(mask_lo)
                
                all_cls = set(np.unique(mask_lo)) | set(np.unique(mask_hi))
                for cls_id in all_cls:
                    if cls_id == 0:
                        continue
                    m_lo = (mask_lo == cls_id).astype(np.float32)
                    m_hi = (mask_hi == cls_id).astype(np.float32)
                    
                    # Linear blend of distance transforms
                    if np.any(m_lo > 0):
                        d_lo = ndi.distance_transform_edt(m_lo)
                        d_lo = d_lo / (d_lo.max() + 1e-8)
                    else:
                        d_lo = np.zeros_like(m_lo)
                    if np.any(m_hi > 0):
                        d_hi = ndi.distance_transform_edt(m_hi)
                        d_hi = d_hi / (d_hi.max() + 1e-8)
                    else:
                        d_hi = np.zeros_like(m_hi)
                    
                    blended = d_lo * (1 - t) + d_hi * t
                    interp_mask[blended > 0.3] = cls_id
                
                if overwrite:
                    self.segmentation_data[z] = interp_mask
                else:
                    empty = self.segmentation_data[z] == 0
                    self.segmentation_data[z][empty] = interp_mask[empty]
                applied += 1
        
        # Sync display
        if self.display_segmentation_data is not None:
            for z in targets:
                if 0 <= z < self.display_segmentation_data.shape[0]:
                    self.display_segmentation_data[z] = self.segmentation_data[z].copy()
        
        self.refresh_region_index()
        self.sync()
        QMessageBox.information(self, "Propagated", 
            f"Propagated labels to {applied} slices from slice {z_curr}.")

    def sparse_to_dense_expand(self):
        """Auto-expand sparse labels to adjacent Z slices for better training.
        
        Since adjacent 3D slices in semiconductor data are nearly identical,
        copying labels ±N slices from each labeled slice creates significantly
        more training data with minimal annotation effort.
        """
        if self.segmentation_data is None or not np.any(self.segmentation_data > 0):
            QMessageBox.warning(self, "No Labels", "Label some slices first.")
            return
        
        # Find currently labeled slices
        labeled_z = np.where(np.any(self.segmentation_data > 0, axis=(1, 2)))[0]
        z_max = self.segmentation_data.shape[0] - 1
        
        if len(labeled_z) == 0:
            QMessageBox.warning(self, "No Labels", "No labeled slices found.")
            return
        
        # Dialog for expansion settings
        dlg = QDialog(self)
        dlg.setWindowTitle("Sparse → Dense Label Expansion")
        dlg.setMinimumWidth(450)
        lay = QVBoxLayout(dlg)
        
        info = QLabel(
            f"Found {len(labeled_z)} labeled slices (out of {z_max + 1} total)\n"
            f"Labeled at Z: {', '.join(str(z) for z in labeled_z[:15])}"
            f"{'...' if len(labeled_z) > 15 else ''}"
        )
        info.setStyleSheet("font-weight: bold; font-size: 9pt; color: #4fc3f7;")
        info.setWordWrap(True)
        lay.addWidget(info)
        
        lay.addSpacing(6)
        explain = QLabel(
            "Since adjacent 3D slices are nearly identical,\n"
            "copying labels to neighbors creates more training data\n"
            "from your sparse annotations — improving model accuracy."
        )
        explain.setStyleSheet("color: #aaa; font-size: 8pt;")
        explain.setWordWrap(True)
        lay.addWidget(explain)
        
        lay.addSpacing(8)
        form = QFormLayout()
        
        spin_expand = QSpinBox()
        spin_expand.setRange(1, 50)
        spin_expand.setValue(3)
        spin_expand.setToolTip("Number of slices to expand in each direction (±N)")
        form.addRow("Expand ± slices:", spin_expand)
        
        chk_overwrite = QCheckBox("Overwrite existing labels")
        chk_overwrite.setChecked(False)
        chk_overwrite.setToolTip("If unchecked, only fills empty (background) pixels")
        form.addRow("", chk_overwrite)
        
        lay.addLayout(form)
        
        # Preview calculation
        lbl_preview = QLabel()
        lbl_preview.setStyleSheet("color: #80cbc4; font-size: 8pt;")
        
        def update_preview():
            n = spin_expand.value()
            expanded_set = set()
            for z in labeled_z:
                for dz in range(-n, n + 1):
                    zz = z + dz
                    if 0 <= zz <= z_max:
                        expanded_set.add(zz)
            new_count = len(expanded_set) - len(labeled_z)
            lbl_preview.setText(
                f"Result: {len(expanded_set)} slices will have labels "
                f"(+{new_count} new from expansion)"
            )
        
        spin_expand.valueChanged.connect(lambda _: update_preview())
        update_preview()
        lay.addWidget(lbl_preview)
        
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        lay.addWidget(btn_box)
        
        if dlg.exec() != QDialog.Accepted:
            return
        
        expand_n = spin_expand.value()
        overwrite = chk_overwrite.isChecked()
        
        # Build expansion targets
        expansion_map = {}  # target_z -> source_z (closest labeled slice)
        for z in labeled_z:
            for dz in range(-expand_n, expand_n + 1):
                target_z = z + dz
                if target_z < 0 or target_z > z_max:
                    continue
                if target_z in set(labeled_z) and dz != 0:
                    continue  # Don't overwrite original labels
                dist = abs(dz)
                if target_z not in expansion_map or dist < expansion_map[target_z][1]:
                    expansion_map[target_z] = (z, dist)
        
        # Remove original labeled slices from targets
        for z in labeled_z:
            expansion_map.pop(z, None)
        
        if not expansion_map:
            QMessageBox.information(self, "No Expansion", "All nearby slices already have labels.")
            return
        
        # Save undo
        target_slices = sorted(expansion_map.keys())
        self.save_history(changed_slices=target_slices)
        
        # Apply expansion
        applied = 0
        for target_z, (source_z, dist) in sorted(expansion_map.items()):
            source_mask = self.segmentation_data[source_z]
            if overwrite:
                self.segmentation_data[target_z] = source_mask.copy()
            else:
                empty = self.segmentation_data[target_z] == 0
                self.segmentation_data[target_z][empty] = source_mask[empty]
            applied += 1
        
        # Sync display
        if self.display_segmentation_data is not None:
            for z in target_slices:
                if 0 <= z < self.display_segmentation_data.shape[0]:
                    self.display_segmentation_data[z] = self.segmentation_data[z].copy()
        
        self.refresh_region_index()
        self.sync()
        
        total_labeled = len(np.where(np.any(self.segmentation_data > 0, axis=(1, 2)))[0])
        QMessageBox.information(self, "Expansion Complete", 
            f"Expanded labels to {applied} new slices (±{expand_n} from each labeled slice).\n"
            f"Total labeled slices now: {total_labeled}/{z_max + 1}\n\n"
            f"You can now start training with {total_labeled}x more data!")

    def generate_smart_frames(self):
        """Auto-detect slices that have labeled content and create frames only there."""
        if self.volume_data is None or self.segmentation_data is None:
            QMessageBox.warning(self, "No Data", "Load a volume and create some labels first.")
            return
        
        # Scan for labeled slices
        labeled_slices = []
        for z in range(self.segmentation_data.shape[0]):
            if np.count_nonzero(self.segmentation_data[z]) > 0:
                labeled_slices.append(z)
        
        if not labeled_slices:
            QMessageBox.warning(self, "No Labels", "No labeled slices found. Label some data first.")
            return
        
        dlg = QDialog(self)
        dlg.setWindowTitle("Smart Frame Generation")
        dlg.setMinimumWidth(400)
        lay = QVBoxLayout(dlg)
        
        info = QLabel(f"Found {len(labeled_slices)} labeled slices "
                      f"(Z: {labeled_slices[0]}..{labeled_slices[-1]})")
        info.setStyleSheet("font-weight: bold; font-size: 10pt;")
        lay.addWidget(info)
        lay.addSpacing(6)
        
        form = QFormLayout()
        
        # Frame type
        combo_type = QComboBox()
        combo_type.addItems([
            "Full-XY (entire slice)",
            "Tight Bounding Box (crop to labeled region)",
            "Padded Bounding Box (crop + margin)",
        ])
        form.addRow("Frame type:", combo_type)
        
        spin_pad = QSpinBox()
        spin_pad.setRange(0, 200)
        spin_pad.setValue(32)
        spin_pad.setEnabled(False)
        combo_type.currentIndexChanged.connect(lambda i: spin_pad.setEnabled(i == 2))
        form.addRow("Padding (px):", spin_pad)
        
        # Split assignment
        combo_split = QComboBox()
        combo_split.addItems(["train", "val", "test", "mix"])
        form.addRow("Default split:", combo_split)
        
        # Auto train/val ratio
        chk_auto_split = QCheckBox("Auto-split train/val (80/20)")
        chk_auto_split.setChecked(True)
        form.addRow("", chk_auto_split)
        
        # Min FG pixels
        spin_min_fg = QSpinBox()
        spin_min_fg.setRange(0, 10000)
        spin_min_fg.setValue(10)
        spin_min_fg.setToolTip("Skip slices with fewer foreground pixels than this")
        form.addRow("Min FG pixels:", spin_min_fg)
        
        lay.addLayout(form)
        
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        lay.addWidget(btn_box)
        
        if dlg.exec() != QDialog.Accepted:
            return
        
        frame_type = combo_type.currentIndex()
        pad = spin_pad.value()
        default_split = combo_split.currentText()
        auto_split = chk_auto_split.isChecked()
        min_fg = spin_min_fg.value()
        
        H, W = self.volume_data.shape[1], self.volume_data.shape[2]
        added = 0
        
        # Filter by min FG
        valid_slices = [z for z in labeled_slices 
                        if np.count_nonzero(self.segmentation_data[z]) >= min_fg]
        
        if not valid_slices:
            QMessageBox.warning(self, "No Valid Slices", 
                f"No slices have >= {min_fg} foreground pixels.")
            return
        
        # Auto-split assignment
        import random
        shuffled = valid_slices.copy()
        random.shuffle(shuffled)
        n_train = int(len(shuffled) * 0.8)
        train_set = set(shuffled[:n_train])
        val_set = set(shuffled[n_train:])
        
        for z in valid_slices:
            if auto_split:
                split = "train" if z in train_set else "val"
            else:
                split = default_split
            
            # Determine rect
            if frame_type == 0:
                rect = (0, 0, W, H)
            else:
                ys, xs = np.where(self.segmentation_data[z] > 0)
                if len(ys) == 0:
                    continue
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
                if frame_type == 2:
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(W, x2 + pad)
                    y2 = min(H, y2 + pad)
                rect = (x1, y1, x2, y2)
            
            # Avoid duplicates
            existing = [f for f in self.crop_frames 
                        if f['slice_index'] == z and f['rect'] == rect 
                        and f.get('source_id') == self.active_source_id]
            if existing:
                continue
            
            frame = {
                'id': self.next_frame_id,
                'name': f'Smart-{self.next_frame_id}',
                'source_id': self.active_source_id,
                'slice_index': z,
                'split': split,
                'rect': rect,
            }
            self.crop_frames.append(frame)
            self.next_frame_id += 1
            added += 1
        
        if added > 0:
            self.sync()
        
        n_train_final = sum(1 for f in self.crop_frames if f.get('source_id') == self.active_source_id and f['split'] == 'train')
        n_val_final = sum(1 for f in self.crop_frames if f.get('source_id') == self.active_source_id and f['split'] == 'val')
        QMessageBox.information(self, "Smart Frames Generated", 
            f"Added {added} frames from {len(valid_slices)} labeled slices.\n"
            f"Train: {n_train_final}  |  Val: {n_val_final}")

    def generate_full_xy_frames(self):
        if self.volume_data is None: return
        max_slices = self.volume_data.shape[0] - 1
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Generate Full-XY Frames")
        layout = QFormLayout(dialog)
        
        spin_start = QSpinBox()
        spin_start.setRange(0, max_slices)
        spin_start.setValue(0)
        
        spin_end = QSpinBox()
        spin_end.setRange(0, max_slices)
        spin_end.setValue(max_slices)
        
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        
        layout.addRow("Start Slice:", spin_start)
        layout.addRow("End Slice:", spin_end)
        layout.addRow(btn_box)
        
        if dialog.exec() == QDialog.Accepted:
            s1 = spin_start.value()
            s2 = spin_end.value()
            if s1 > s2: s1, s2 = s2, s1
            
            H, W = self.volume_data.shape[1], self.volume_data.shape[2]
            added = 0
            for z in range(s1, s2 + 1):
                # Check for existing identical frames
                existing = [f for f in self.crop_frames if f['slice_index'] == z and f['rect'] == (0, 0, W, H)]
                if not existing:
                    frame = {
                        'id': self.next_frame_id,
                        'name': f'Full-XY {self.next_frame_id}',
                        'source_id': self.active_source_id,
                        'slice_index': z,
                        'split': self.frame_split,
                        'rect': (0, 0, W, H)
                    }
                    self.crop_frames.append(frame)
                    self.next_frame_id += 1
                    added += 1
            if added > 0:
                self.sync()
            QMessageBox.information(self, "Generated", f"Added {added} Full-XY frames from slice {s1} to {s2}.")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self._handle_volume_load(file_path)
            
    def _handle_volume_load(self, file_path):
        from pathlib import Path
        path = Path(file_path)
        
        raw_shape = None
        raw_dtype = None
        if path.suffix.lower() in ['.raw', '.bin']:
            from inno3d.core.view_support import RawImportDialog
            dlg = RawImportDialog(path.name, self)
            if dlg.exec_():
                raw_shape, raw_dtype = dlg.get_data()
            else:
                return # user cancelled
                
        self.pending_load_path = file_path
        self.pg = QProgressDialog(f"Loading {path.name}...", "Cancel", 0, 100, self)
        self.pg.setWindowModality(Qt.WindowModal)
        self.pg.setAutoClose(True)
        self.pg.show()
        
        self.t = AILoadVolumeThread(str(path), raw_shape, raw_dtype)
        self.t.progress.connect(lambda v, m: (self.pg.setValue(v), self.pg.setLabelText(m)))
        self.t.finished.connect(self._done)
        self.t.start()
        
    def load_volume(self, from_folder=False):
        if from_folder:
            p = QFileDialog.getExistingDirectory(self, "Dir")
        else:
            p, _ = QFileDialog.getOpenFileName(self, "File", "", "Image Files (*.tif *.tiff *.raw *.bin)")
            
        if p:
            self._handle_volume_load(p)
            
    def load_folder(self, **kwargs): self.load_volume(from_folder=True)
    def _done(self, d, e):
        self.pg.close()
        if e: QMessageBox.critical(self,"Err",str(e)); return
        
        src_id = getattr(self, 'sources', [])
        src_id = len(src_id) + 1
        pending_path = getattr(self, 'pending_load_path', f"Volume_{src_id}")
        name = Path(pending_path).name if pending_path else f"Volume {src_id}"
        
        vol_data = d if d.ndim==3 else d[:,:,:,0]
        
        new_src = {
            'id': src_id,
            'name': name,
            'path': pending_path,
            'volume': vol_data,
            'mask': np.zeros_like(vol_data, dtype=np.uint8),
            'region_index': [],
            'region_label_maps': {},
            'hidden_regions': set()
        }
        if not hasattr(self, 'sources'): self.sources = []
        self.sources.append(new_src)
        self.switch_to_source(src_id)

    def switch_to_source(self, src_id):
        src = next((s for s in self.sources if s['id'] == src_id), None)
        if not src: return
        
        # Pull back the active modifications into its own dictionary if needed
        # (References to numpy arrays are auto-maintained, but region lists might need saving if detached)
        
        self.active_source_id = src_id
        self.current_source_path = src['path']
        self.volume_data = src['volume']
        self.segmentation_data = src['mask']
        self.region_index = src['region_index']
        self.region_label_maps = src['region_label_maps']
        self.hidden_regions = src.get('hidden_regions', set())
        self.active_region_key = None
        self.display_segmentation_data = None
        
        v_max = int(self.volume_data.max())
        self.sld_min.setRange(0, v_max); self.sld_max.setRange(0, v_max)
        self.display_min, self.display_max = 0.0, float(v_max)
        self.sld_min.setValue(0); self.sld_max.setValue(v_max)

        for v in self.all_viewers:
            v.segmentation_data = self.segmentation_data
            v.display_segmentation_data = None
            v.update_callback = lambda save_state=False, brush_only=False, from_viewer=False, min_max=None, _v=v: self.sync(brush_only, save_state, from_viewer, min_max, source_viewer=_v)
            should_render_now = (v is self.xy_view)
            v.set_volume(self.volume_data, render_immediately=should_render_now)
            
        self.undo_stack, self.redo_stack = [], []
        # Support multiple sources natively: keep crop_frames accumulated in memory
        self.active_frame_id = None
        
        if hasattr(self, 'btn_focus_mode'):
            self.btn_focus_mode.setChecked(False)
        self.v_stack.setCurrentIndex(0)

        self.on_auto_contrast()
        self.refresh_region_index()
        self.sync()

    def get_frames_for_slice(self, slice_index):
        return [frame for frame in self.crop_frames if frame.get('source_id', 1) == self.active_source_id and frame['slice_index'] == slice_index]

    def get_active_frame(self):
        for frame in self.crop_frames:
            if frame['id'] == self.active_frame_id:
                return frame
        return None

    def on_default_frame_split_changed(self, value):
        self.frame_split = value

    def add_crop_frame(self, rect, slice_index):
        frame = {
            'id': self.next_frame_id,
            'source_id': self.active_source_id,
            'name': f'Frame {self.next_frame_id}',
            'slice_index': int(slice_index),
            'split': self.frame_split,
            'rect': tuple(int(v) for v in rect),
        }
        self.crop_frames.append(frame)
        self.active_frame_id = frame['id']
        self.next_frame_id += 1
        if hasattr(self, 'btn_frame_crop'):
            self.btn_frame_crop.setChecked(False)
        if self.active_tool == 'frame_crop':
            self.active_tool = None
        self.sync()

    def refresh_frame_table(self):
        if not hasattr(self, 'frame_table'):
            return
        self.frame_table.blockSignals(True)
        self.frame_table.setSortingEnabled(False)
        self.frame_table.setRowCount(0)
        
        # Only show frames for the current active source
        source_frames = [f for f in self.crop_frames if f.get('source_id', 1) == self.active_source_id]
        
        self.frame_table.setRowCount(len(source_frames))
        selected_row = None
        split_counts = {name: 0 for name in ['train', 'val', 'test', 'mix', 'monitoring']}
        for row, frame in enumerate(source_frames):
            x1, y1, x2, y2 = frame['rect']
            split_counts[frame['split']] = split_counts.get(frame['split'], 0) + 1
            
            chk = QTableWidgetItem(); chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable); chk.setCheckState(Qt.Unchecked); chk.setData(Qt.UserRole, frame['id'])
            item_name = QTableWidgetItem(); item_name.setData(Qt.EditRole, frame['name'])
            item_idx = QTableWidgetItem(); item_idx.setData(Qt.DisplayRole, frame['slice_index'])
            item_split = QTableWidgetItem(); item_split.setData(Qt.EditRole, frame['split'])
            item_p1 = QTableWidgetItem(f'{x1},{y1}')
            item_p2 = QTableWidgetItem(f'{x2},{y2}')
            item_sz = QTableWidgetItem(f'{max(0, x2-x1)}x{max(0, y2-y1)}')
            
            for c, itm in enumerate([chk, item_name, item_idx, item_split, item_p1, item_p2, item_sz]):
                self.frame_table.setItem(row, c, itm)
            
            if frame['id'] == self.active_frame_id:
                selected_row = row
                
        self.frame_table.clearSelection()
        if selected_row is not None:
            self.frame_table.selectRow(selected_row)
            frame = self.get_active_frame()
            if frame is not None and hasattr(self, 'combo_frame_assign'):
                self.combo_frame_assign.blockSignals(True)
                self.combo_frame_assign.setCurrentText(frame['split'])
                self.combo_frame_assign.blockSignals(False)
        self.frame_table.setSortingEnabled(True)
        self.frame_table.blockSignals(False)
        summary_parts = [f"{name}:{count}" for name, count in split_counts.items() if count]
        self.lbl_frame_summary.setText(' | '.join(summary_parts) if summary_parts else 'No frames')

    def apply_source_split(self):
        sel_rows = self.mother_table.selectionModel().selectedRows()
        if not sel_rows:
            QMessageBox.warning(self, "No Selection", "Please select at least one source in the table.")
            return
            
        new_split = self.combo_src_split.currentText()
        count = 0
        for sr in sel_rows:
            src_id = self.mother_table.item(sr.row(), 0).data(Qt.UserRole)
            src = next((s for s in self.sources if s['id'] == src_id), None)
            if src:
                src['split'] = new_split
                count += 1
                
        self.refresh_mother_table()
        QMessageBox.information(self, "Success", f"Updated {count} sources to '{new_split}' split.")

    def delete_selected_sources(self):
        """Delete selected source volumes from the list."""
        sel_rows = self.mother_table.selectionModel().selectedRows()
        if not sel_rows:
            QMessageBox.warning(self, "No Selection", "Please select at least one source in the table.")
            return
        
        # Collect IDs to delete
        ids_to_delete = set()
        for sr in sel_rows:
            item = self.mother_table.item(sr.row(), 0)
            if item:
                ids_to_delete.add(item.data(Qt.UserRole))
        
        if not ids_to_delete:
            return
        
        # Don't allow deleting ALL sources if only one remains
        remaining = [s for s in self.sources if s['id'] not in ids_to_delete]
        
        names = [s['name'] for s in self.sources if s['id'] in ids_to_delete]
        reply = QMessageBox.question(
            self, "Delete Sources",
            f"Delete {len(ids_to_delete)} source(s)?\n" + "\n".join(f"  • {n}" for n in names[:5]) +
            ("\n  ..." if len(names) > 5 else "") +
            "\n\nAssociated ROI frames will also be removed.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        
        # Remove associated frames
        self.crop_frames = [f for f in self.crop_frames if f.get('source_id', 1) not in ids_to_delete]
        
        # Remove sources
        self.sources = remaining
        
        # Fix active state
        if self.active_source_id in ids_to_delete:
            if self.sources:
                self.switch_to_source(self.sources[0]['id'])
            else:
                self.active_source_id = None
                self.volume_data = None
                self.segmentation_data = None
                for v in self.all_viewers:
                    v.set_volume(None)
        
        self.refresh_mother_table()
        self.sync()

    def on_frame_selection_changed(self):
        rows = self.frame_table.selectionModel().selectedRows() if hasattr(self, 'frame_table') else []
        if not rows:
            self.active_frame_id = None
            self.sync()
            return
        row = rows[0].row()
        item = self.frame_table.item(row, 0)
        if item is not None:
            self.active_frame_id = item.data(Qt.UserRole)
        frame = self.get_active_frame()
        if frame is not None and getattr(self.xy_view, 'current_slice', -1) != frame['slice_index']:
            self.xy_view.set_current_slice(frame['slice_index'])
        self.sync()

    def focus_selected_frame(self):
        frame = self.get_active_frame()
        if frame is None:
            QMessageBox.information(self, 'No ROI', 'Please create or select an ROI first.')
            return
        if hasattr(self, 'chk_isolate_roi'):
            self.chk_isolate_roi.setChecked(True)
        self.xy_view.set_current_slice(frame['slice_index'])
        self.set_view_mode(0)
        self.sync()

    def apply_selected_frame_split(self):
        checked_ids = []
        for r in range(self.frame_table.rowCount()):
            item = self.frame_table.item(r, 0)
            if item and item.checkState() == Qt.Checked:
                checked_ids.append(item.data(Qt.UserRole))
                
        if not checked_ids:
            sel_rows = self.frame_table.selectionModel().selectedRows()
            for sr in sel_rows:
                itm = self.frame_table.item(sr.row(), 0)
                if itm: checked_ids.append(itm.data(Qt.UserRole))
                
        if not checked_ids:
            QMessageBox.information(self, 'No Frames', 'Please check [X] or select frames to update split.')
            return
            
        new_split = self.combo_frame_assign.currentText()
        count = 0
        for frame in self.crop_frames:
            if frame['id'] in checked_ids:
                frame['split'] = new_split
                count += 1
                
        self.sync()
        if count > 1:
            QMessageBox.information(self, 'Updated', f'Updated split to {new_split} for {count} frames.')

    def delete_selected_frame(self):
        """Delete selected ROI frames (supports multi-select via checkbox or row selection)."""
        # Collect IDs from checked [X] checkboxes first
        ids_to_delete = set()
        for r in range(self.frame_table.rowCount()):
            item = self.frame_table.item(r, 0)
            if item and item.checkState() == Qt.Checked:
                ids_to_delete.add(item.data(Qt.UserRole))
        
        # If no checkboxes checked, fall back to selected rows
        if not ids_to_delete:
            sel_rows = self.frame_table.selectionModel().selectedRows()
            for sr in sel_rows:
                itm = self.frame_table.item(sr.row(), 0)
                if itm:
                    ids_to_delete.add(itm.data(Qt.UserRole))
        
        # Final fallback: active frame
        if not ids_to_delete and self.active_frame_id is not None:
            ids_to_delete.add(self.active_frame_id)
        
        if not ids_to_delete:
            return
        
        self.crop_frames = [f for f in self.crop_frames if f['id'] not in ids_to_delete]
        if self.active_frame_id in ids_to_delete:
            self.active_frame_id = self.crop_frames[0]['id'] if self.crop_frames else None
        self.sync()

    def clear_all_frames(self):
        if not self.crop_frames:
            return
        self.crop_frames = []
        self.active_frame_id = None
        self.sync()

    def refresh_mother_table(self):
        if not hasattr(self, 'mother_table'):
            return
        self.mother_table.setRowCount(0)
        if not self.sources:
            return
        
        self.mother_table.setRowCount(len(self.sources))
        for row, src in enumerate(self.sources):
            # Frames for this source
            s_frames = [f for f in self.crop_frames if f.get('source_id', 1) == src['id']]
            active_slice = getattr(self.xy_view, 'current_slice', '-') if src['id'] == self.active_source_id else "-"
            
            shape_text = ' x '.join(str(v) for v in src['volume'].shape)
            values = [str(src['id']), src['name'], shape_text, str(len(s_frames)), src.get('split', 'none'), str(active_slice)]
            
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setData(Qt.UserRole, src['id'])
                if src['id'] == self.active_source_id:
                    item.setBackground(QColor(0, 100, 50, 80)) # Highlight active
                self.mother_table.setItem(row, col, item)

    def on_mother_table_selection(self):
        rows = self.mother_table.selectionModel().selectedRows() if hasattr(self, 'mother_table') else []
        if not rows: return
        item = self.mother_table.item(rows[0].row(), 0)
        if item:
            src_id = item.data(Qt.UserRole)
            if src_id != getattr(self, 'active_source_id', None):
                self.switch_to_source(src_id)

    def schedule_region_refresh(self):
        if self.segmentation_data is None or not self.enable_connected_components:
            return
        self.region_refresh_timer.start(250)

    def refresh_region_index(self):
        if not self.enable_connected_components:
            self.region_index = []
            self.region_label_maps = {}
            self.hidden_regions = set()
            self.active_region_key = None
            self.display_segmentation_data = None
            if hasattr(self, 'region_table'):
                self.region_table.blockSignals(True)
                self.region_table.setRowCount(0)
                self.region_table.blockSignals(False)
            self._clear_region_highlight()
            return
        if self.segmentation_data is None or not hasattr(self, 'region_table'):
            return
        self.region_index = []
        self.region_label_maps = {}
        for class_id in sorted(self.classes.keys()):
            mask = self.segmentation_data == class_id
            if not np.any(mask):
                continue
            
            nz = np.any(mask, axis=(1, 2))
            z_min, z_max = np.where(nz)[0][[0, -1]]
            z_max += 1
            ny = np.any(mask, axis=(0, 2))
            y_min, y_max = np.where(ny)[0][[0, -1]]
            y_max += 1
            nx = np.any(mask, axis=(0, 1))
            x_min, x_max = np.where(nx)[0][[0, -1]]
            x_max += 1
            
            cropped_mask = mask[z_min:z_max, y_min:y_max, x_min:x_max]
            try:
                labels = measure.label(cropped_mask, connectivity=1)
            except MemoryError:
                import scipy.ndimage as ndimage
                out = np.zeros(cropped_mask.shape, dtype=np.uint16)
                ndimage.label(cropped_mask, output=out)
                labels = out
                
            self.region_label_maps[class_id] = (labels, (z_min, y_min, x_min))
            
            for region in measure.regionprops(labels):
                cz1, cy1, cx1, cz2, cy2, cx2 = region.bbox
                key = f"{class_id}:{region.label}"
                self.region_index.append({
                    'key': key,
                    'class_id': class_id,
                    'region_id': region.label,
                    'name': f"C{class_id}-R{region.label}",
                    'voxels': int(region.area),
                    'bbox': (int(cz1 + z_min), int(cy1 + y_min), int(cx1 + x_min), 
                             int(cz2 + z_min), int(cy2 + y_min), int(cx2 + x_min)),
                })
        valid_keys = {region['key'] for region in self.region_index}
        
        self.hidden_regions = {key for key in self.hidden_regions if key in valid_keys}
            
        if self.active_region_key not in valid_keys:
            self.active_region_key = self.region_index[0]['key'] if self.region_index else None
        self._sync_active_source_state()
        self.rebuild_display_segmentation()
        self.refresh_region_table()
        self.sync(brush_only=True)

    def rebuild_display_segmentation(self):
        if self.segmentation_data is None:
            self.display_segmentation_data = None
            return
        if not self.hidden_regions:
            self.display_segmentation_data = None
            return
        display = self.segmentation_data.copy()
        for region in self.region_index:
            if region['key'] not in self.hidden_regions:
                continue
            labels_info = self.region_label_maps.get(region['class_id'])
            if labels_info is None:
                continue
            
            if isinstance(labels_info, tuple):
                labels, offset = labels_info
                z0, y0, x0 = offset
                z1, y1, x1, z2, y2, x2 = region['bbox']
                
                cz1, cy1, cx1 = z1 - z0, y1 - y0, x1 - x0
                cz2, cy2, cx2 = z2 - z0, y2 - y0, x2 - x0
                
                sub_display = display[z1:z2, y1:y2, x1:x2]
                sub_labels = labels[cz1:cz2, cy1:cy2, cx1:cx2]
                
                sub_display[sub_labels == region['region_id']] = 0
            else:
                labels = labels_info
                display[(self.segmentation_data == region['class_id']) & (labels == region['region_id'])] = 0
        self.display_segmentation_data = display

    def refresh_region_table(self):
        if not hasattr(self, 'region_table'):
            return
        self.region_table.blockSignals(True)
        
        current_slice = self.xy_view.current_slice
        # Filter regions that intersect with the current slice
        filtered_regions = [
            r for r in self.region_index 
            if r['bbox'][0] <= current_slice < r['bbox'][3]
        ]
        
        self.region_table.setRowCount(len(filtered_regions))
        selected_row = None
        for row, region in enumerate(filtered_regions):
            z1, y1, x1, z2, y2, x2 = region['bbox']
            visible_item = QTableWidgetItem()
            visible_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
            visible_item.setCheckState(Qt.Unchecked if region['key'] in self.hidden_regions else Qt.Checked)
            visible_item.setData(Qt.UserRole, region['key'])
            self.region_table.setItem(row, 0, visible_item)
            values = [
                region['name'],
                self.classes.get(region['class_id'], {}).get('name', f"Class {region['class_id']}"),
                str(region['voxels']),
                f'X:{x1}-{x2} Y:{y1}-{y2}',
                f'Y:{y1}-{y2} Z:{z1}-{z2}',
                f'X:{x1}-{x2} Z:{z1}-{z2}',
            ]
            for col, value in enumerate(values, start=1):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, region['key'])
                self.region_table.setItem(row, col, item)
            if region['key'] == self.active_region_key:
                selected_row = row
        self.region_table.clearSelection()
        if selected_row is not None:
            self.region_table.selectRow(selected_row)
        self.region_table.blockSignals(False)
        self.lbl_region_info.setText(f"Slice {current_slice}: {len(filtered_regions)} region(s) visible (Total: {len(self.region_index)} index).")


    def get_active_region(self):
        for region in self.region_index:
            if region['key'] == self.active_region_key:
                return region
        return None

    def on_region_selection_changed(self):
        rows = self.region_table.selectionModel().selectedRows() if hasattr(self, 'region_table') else []
        if not rows:
            self.active_region_key = None
            self._clear_region_highlight()
            self.sync(brush_only=True)
            return
        item = self.region_table.item(rows[0].row(), 1)
        if item is not None:
            self.active_region_key = item.data(Qt.UserRole)
            
            # Jump viewers to region centroid
            region = self.get_active_region()
            if region and self.volume_data is not None:
                z1, y1, x1, z2, y2, x2 = region['bbox']
                cz, cy, cx = (z1+z2)//2, (y1+y2)//2, (x1+x2)//2
                z_max, y_max, x_max = self.volume_data.shape[0]-1, self.volume_data.shape[1]-1, self.volume_data.shape[2]-1
                cz = min(max(0, int(cz)), z_max)
                cy = min(max(0, int(cy)), y_max)
                cx = min(max(0, int(cx)), x_max)
                if self.xy_view.current_slice != cz: self.xy_view.set_current_slice(cz)
                if self.xz_view.current_slice != cy: self.xz_view.set_current_slice(cy)
                if self.yz_view.current_slice != cx: self.yz_view.set_current_slice(cx)
                # Sync crosshair to centroid  
                for v in self.all_viewers:
                    if v.show_crosshair:
                        if v.orientation == 'axial':    v.crosshair_data_pos = (cx, cy)
                        elif v.orientation == 'coronal': v.crosshair_data_pos = (cx, cz)
                        elif v.orientation == 'sagittal': v.crosshair_data_pos = (cy, cz)
                
                # Set highlight data
                self._set_region_highlight(region)
                
        self.sync(brush_only=True)

    def _set_region_highlight(self, region):
        labels_info = self.region_label_maps.get(region['class_id'])
        if labels_info is None:
            self._clear_region_highlight()
            return
            
        if isinstance(labels_info, tuple):
            labels_3d, offset = labels_info
        else:
            labels_3d = labels_info
            offset = (0, 0, 0)
            
        for v in self.all_viewers:
            v.highlight_region_key = region['key']
            v._highlight_labels = labels_3d
            v._highlight_offset = offset
            v._highlight_region_id = region['region_id']
            v._highlight_class_id = region['class_id']

    def _clear_region_highlight(self):
        for v in self.all_viewers:
            v.highlight_region_key = None
            v._highlight_labels = None
            v._highlight_offset = (0, 0, 0)
            v._highlight_region_id = None
            v._highlight_class_id = None

    def on_region_item_changed(self, item):
        if item.column() != 0:
            return
        key = item.data(Qt.UserRole)
        if key is None:
            return
        if item.checkState() == Qt.Checked:
            self.hidden_regions.discard(key)
        else:
            self.hidden_regions.add(key)
        self.rebuild_display_segmentation()
        self.sync(brush_only=True)

    def show_all_regions(self):
        self.hidden_regions.clear()
        self.rebuild_display_segmentation()
        self.refresh_region_table()
        self.sync(brush_only=True)

    def delete_selected_region(self):
        region = self.get_active_region()
        if region is None or self.segmentation_data is None:
            return
        labels = self.region_label_maps.get(region['class_id'])
        if labels is None:
            return
        self.save_history()
        mask = (self.segmentation_data == region['class_id']) & (labels == region['region_id'])
        self.segmentation_data[mask] = 0
        self.refresh_region_index()

    def build_active_region_box_map(self):
        region = self.get_active_region()
        box_map = {'axial': [], 'coronal': [], 'sagittal': []}
        if region is None:
            return box_map
        z1, y1, x1, z2, y2, x2 = region['bbox']
        color = self.classes.get(region['class_id'], {}).get('color', (255, 90, 90))
        axial_slice = self.xy_view.current_slice
        if z1 <= axial_slice < z2:
            box_map['axial'].append({'rect': (x1, y1, x2, y2), 'label': region['name'], 'color': color})
        coronal_slice = self.xz_view.current_slice
        if y1 <= coronal_slice < y2:
            box_map['coronal'].append({'rect': (x1, z1, x2, z2), 'label': region['name'], 'color': color})
        sagittal_slice = self.yz_view.current_slice
        if x1 <= sagittal_slice < x2:
            box_map['sagittal'].append({'rect': (y1, z1, y2, z2), 'label': region['name'], 'color': color})
        return box_map

    def run_inference(self):
        if self.volume_data is None:
            QMessageBox.warning(self, "No Data", "Please load a volume first.")
            return
        ckpt = self.mdl_txt_ckpt.text()
        if not ckpt or not os.path.exists(ckpt):
            if QMessageBox.question(self, "No Checkpoint", "No valid checkpoint selected. Continue with random initialized weights?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.No:
                return
        
        mode = self.mdl_inf_input.currentText()
        stride = self.mdl_inf_stride.value()
        z_from = self.mdl_inf_z_from.value() if mode == "Slice Range" else 0
        z_to = self.mdl_inf_z_to.value() if mode == "Slice Range" else (self.volume_data.shape[0] - 1)
        
        config = {
            'classes': self.mdl_spin_classes.value() + 1,
            'arch': self.mdl_combo_arch.currentText(),
            'ckpt': ckpt,
            'mode': mode,
            'slice': self.xy_view.current_slice,
            'stride': stride,
            'z_from': z_from,
            'z_to': z_to,
            'dim': self.mdl_combo_dim.currentText(),
            'adj': self.mdl_spin_adj.value()
        }
        
        self.mdl_btn_infer.setEnabled(False)
        self.mdl_btn_infer.setText("Inferring...")
        self.mdl_lbl_confidence.setText("")
        
        self.infer_worker = AIInferWorker(self.volume_data, config)
        self.infer_worker.finished_infer.connect(self.on_inference_done)
        self.infer_worker.error.connect(self.on_inference_error)
        
        # Show stride info in progress
        stride_info = f" (stride={stride})" if stride > 1 else ""
        self.infer_worker.progress.connect(
            lambda cur, tot: self.mdl_lbl_status.setText(
                f"Inferring{stride_info}: {cur}/{tot} slices..."
            )
        )
        self.infer_worker.start()

    def on_inference_done(self, mask, confidence):
        self.inference_result_mask = mask
        self.inference_confidence = confidence
        self.mdl_btn_map.setEnabled(True)
        self.mdl_btn_infer.setEnabled(True)
        self.mdl_btn_infer.setText("Run Inference")
        
        # Confidence analysis
        fg_slices = np.where(np.any(mask > 0, axis=(1, 2)))[0]
        if len(fg_slices) > 0 and confidence is not None:
            mean_conf = float(np.mean(confidence[fg_slices]))
            min_conf = float(np.min(confidence[fg_slices]))
            
            # Find lowest-confidence slices for user to review
            n_suggest = min(5, len(fg_slices))
            sorted_by_conf = fg_slices[np.argsort(confidence[fg_slices])]
            low_conf_slices = sorted_by_conf[:n_suggest]
            
            conf_msg = (
                f"✓ Inference complete | {len(fg_slices)} slices with predictions\n"
                f"Confidence: mean={mean_conf:.1%}  min={min_conf:.1%}\n"
                f"🔍 Refine these low-confidence slices first: {', '.join(str(z) for z in low_conf_slices)}"
            )
            self.mdl_lbl_confidence.setText(conf_msg)
            self.mdl_lbl_status.setText(f"Inference done. Mean confidence: {mean_conf:.1%}")
        else:
            self.mdl_lbl_confidence.setText("✓ Inference complete.")
            self.mdl_lbl_status.setText("Inference complete. You can now map the result.")
        
        QMessageBox.information(self, "Inference Done", 
            f"Inference completed on {len(fg_slices)} slices.\n"
            f"Press 'Map Result -> Labelling' to see the predictions.")

    def on_inference_error(self, err_msg):
        self.mdl_btn_infer.setEnabled(True)
        self.mdl_btn_infer.setText("Run Inference")
        QMessageBox.critical(self, "Inference Error", err_msg)

    # ═══════════════════════════════════════════════════════════════════════
    # ACTIVE LEARNING (SMART LABELING)
    # ═══════════════════════════════════════════════════════════════════════

    def run_active_learning(self):
        if self.volume_data is None:
            QMessageBox.warning(self, "Active Learning", "Please load a volume first.")
            return

        config = self._build_train_config()  # Reuse train config for model architecture/path
        if not config.get('ckpt') or not os.path.exists(config.get('ckpt', '')):
            QMessageBox.warning(self, "Active Learning", 
                "No trained model found! Please train a model first so the system can compute uncertainty.")
            return

        # Gather labeled slices
        labeled_slices = []
        if self.segmentation_data is not None:
            for z in range(self.segmentation_data.shape[0]):
                if np.count_nonzero(self.segmentation_data[z]) > 0:
                    labeled_slices.append(z)

        # Add AL specific config
        config['mc_passes'] = self.al_widget.spin_mc_passes.value()
        config['n_recommend'] = self.al_widget.spin_n_recommend.value()

        try:
            from inno3d.services.active_learning import ActiveLearningWorker
            self.al_worker = ActiveLearningWorker(
                volume=self.volume_data,
                config=config,
                labeled_slices=labeled_slices
            )
            self.al_worker.progress.connect(self.al_widget.set_progress)
            self.al_worker.finished.connect(self._on_al_finished)
            self.al_worker.error.connect(self._on_al_error)
            
            self.al_widget.set_running(True)
            self.al_worker.start()
        except Exception as e:
            QMessageBox.critical(self, "Active Learning Error", f"Failed to start: {e}")

    def _on_al_finished(self, state):
        self.al_widget.update_state(state)
        self.al_widget.set_running(False)

    def _on_al_error(self, err_msg):
        self.al_widget.set_running(False)
        self.al_widget._lbl_status.setText("Error occurred.")
        QMessageBox.critical(self, "Active Learning Error", err_msg)

    # ═══════════════════════════════════════════════════════════════════════
    # FOUNDATION PRE-TRAINING (SELF-SUPERVISED)
    # ═══════════════════════════════════════════════════════════════════════

    def run_foundation_pretrain(self):
        if self.volume_data is None:
            QMessageBox.warning(self, "Foundation Pre-Training",
                "Please load a volume first. No labels needed — just raw data!")
            return

        save_dir = self.mdl_txt_save_dir.text()
        if not save_dir or not os.path.isdir(save_dir):
            QMessageBox.warning(self, "Foundation Pre-Training",
                "Please set a valid Save Directory in the Training Configuration section first.")
            return

        fw = self.foundation_widget
        dim_str = self.mdl_combo_dim.currentText()
        adj = self.mdl_spin_adj.value()

        config = {
            'arch': self.mdl_combo_arch.currentText(),
            'epochs': fw.spin_epochs.value(),
            'batch_size': fw.spin_bs.value(),
            'lr': fw.spin_lr.value(),
            'crop_size': fw.spin_crop.value(),
            'save_dir': save_dir,
            'dim': dim_str,
            'adj': adj,
            'classes': self.mdl_spin_classes.value() + 1,
        }

        try:
            from inno3d.services.foundation_pretrainer import FoundationPretrainWorker
            self.foundation_worker = FoundationPretrainWorker(
                volume=self.volume_data, config=config
            )
            self.foundation_worker.progress.connect(fw.set_progress)
            self.foundation_worker.finished.connect(self._on_foundation_finished)
            self.foundation_worker.error.connect(self._on_foundation_error)

            fw.set_running(True)
            self.foundation_worker.start()
        except Exception as e:
            QMessageBox.critical(self, "Foundation Error", f"Failed to start: {e}")

    def stop_foundation_pretrain(self):
        if hasattr(self, 'foundation_worker') and self.foundation_worker.isRunning():
            self.foundation_worker.stop()
            self.foundation_widget._lbl_status.setText("Stopping...")

    def _on_foundation_finished(self, msg, saved_path):
        self.foundation_widget.set_running(False)
        self.foundation_widget.set_finished(msg, saved_path)

    def _on_foundation_error(self, err_msg):
        self.foundation_widget.set_running(False)
        QMessageBox.critical(self, "Foundation Pre-Training Error", err_msg)

    def load_foundation_weights(self, path):
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Load Weights", "Invalid checkpoint path.")
            return
        try:
            from inno3d.services.foundation_pretrainer import load_pretrained_encoder
            from inno3d.models import create_model
            import torch

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            arch = self.mdl_combo_arch.currentText()
            dim_str = self.mdl_combo_dim.currentText()
            adj = self.mdl_spin_adj.value()
            in_c = int(adj) if dim_str == '2.5D' else 1
            out_c = self.mdl_spin_classes.value() + 1

            model = create_model(arch, in_c, out_c).to(device)
            n_loaded, n_skipped = load_pretrained_encoder(model, path, device)

            # Save the model with pre-trained weights for training
            save_dir = self.mdl_txt_save_dir.text()
            if save_dir and os.path.isdir(save_dir):
                merged_path = os.path.join(save_dir, 'pretrained_ready.pth')
                torch.save(model.state_dict(), merged_path)
                self.mdl_txt_ckpt.setText(merged_path)
                self.foundation_widget.set_load_status(
                    f"✓ Loaded {n_loaded} weight tensors, skipped {n_skipped}.\n"
                    f"Checkpoint saved → {os.path.basename(merged_path)}\n"
                    f"Now click 'Train' — encoder is pre-initialized!",
                    success=True
                )
            else:
                self.foundation_widget.set_load_status(
                    f"✓ Loaded {n_loaded} weight tensors, skipped {n_skipped}.\n"
                    f"Set a Save Directory to persist the merged weights.",
                    success=True
                )
        except Exception as e:
            self.foundation_widget.set_load_status(f"Error: {e}", success=False)

    def _on_inf_mode_changed(self, mode_text):
        """Show/hide Slice Range controls based on inference mode."""
        is_range = (mode_text == "Slice Range")
        self.mdl_inf_range_widget.setVisible(is_range)
        if is_range and self.volume_data is not None:
            self.mdl_inf_z_from.setRange(0, self.volume_data.shape[0] - 1)
            self.mdl_inf_z_to.setRange(0, self.volume_data.shape[0] - 1)
            self.mdl_inf_z_to.setValue(self.volume_data.shape[0] - 1)

    def _update_speed_estimate(self):
        """Update the speed estimate label based on stride and mode."""
        stride = self.mdl_inf_stride.value()
        mode = self.mdl_inf_input.currentText()
        if mode == "Single Slice (XY)" or stride <= 1:
            self.mdl_lbl_speed_est.setText("")
            return
        
        total = 0
        if self.volume_data is not None:
            if mode == "Full Volume":
                total = self.volume_data.shape[0]
            elif mode == "Slice Range":
                total = self.mdl_inf_z_to.value() - self.mdl_inf_z_from.value() + 1
        
        if total > 0:
            actual_infer = (total + stride - 1) // stride
            speedup = total / max(1, actual_infer)
            self.mdl_lbl_speed_est.setText(
                f"⚡ Stride={stride}: infer {actual_infer}/{total} slices → ~{speedup:.1f}x faster"
            )
        else:
            self.mdl_lbl_speed_est.setText(f"⚡ Stride={stride}: ~{stride}x faster (load volume first)")

    def map_inference_to_labelling(self):
        if not hasattr(self, 'inference_result_mask') or self.inference_result_mask is None:
            return
        
        self.save_history() # Preserve for UNDO
        
        # Merge inference mask into current segmentation data ONLY where inference > 0
        mask_fg = self.inference_result_mask > 0
        self.segmentation_data[mask_fg] = self.inference_result_mask[mask_fg]
        self.refresh_region_index() # Rebuild display and map colors
        
        # Switch to Labelling Tab
        self.ctrl.setCurrentIndex(0)
        self.sync()
        QMessageBox.information(self, "Mapped", "Inference results mapped to Labelling Data successfully!")

    # ═══════════════════════════════════════════════════════════════
    # Labeled Slice Split Assignment — backend methods
    # ═══════════════════════════════════════════════════════════════

    def _refresh_slice_split_table(self):
        """Scan segmentation data and populate the slice split table."""
        if self.segmentation_data is None:
            QMessageBox.warning(self, "No Data", "Load a volume and label some slices first.")
            return
        
        labeled_z = np.where(np.any(self.segmentation_data > 0, axis=(1, 2)))[0]
        if len(labeled_z) == 0:
            self.mdl_lbl_split_summary.setText("No labeled slices found")
            self.mdl_slice_table.setRowCount(0)
            return
        
        # Initialize _slice_splits if not exists
        if not hasattr(self, '_slice_splits'):
            self._slice_splits = {}
        
        # Auto-assign 'train' to any new slices not yet in _slice_splits
        for z in labeled_z:
            if int(z) not in self._slice_splits:
                self._slice_splits[int(z)] = 'train'
        
        # Remove stale entries (slices that no longer have labels)
        stale = [z for z in self._slice_splits if z not in labeled_z]
        for z in stale:
            del self._slice_splits[z]
        
        # Populate table
        self.mdl_slice_table.setRowCount(len(labeled_z))
        for row, z in enumerate(labeled_z):
            z_int = int(z)
            px_count = int(np.count_nonzero(self.segmentation_data[z]))
            split = self._slice_splits.get(z_int, 'train')
            
            item_z = QTableWidgetItem(str(z_int))
            item_z.setData(Qt.UserRole, z_int)
            item_z.setFlags(item_z.flags() & ~Qt.ItemIsEditable)
            
            item_px = QTableWidgetItem(f"{px_count:,}")
            item_px.setFlags(item_px.flags() & ~Qt.ItemIsEditable)
            item_px.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            
            # Split column: editable combo via QTableWidgetItem
            item_split = QTableWidgetItem(split)
            item_split.setData(Qt.UserRole, z_int)
            # Color code
            split_colors = {'train': '#4caf50', 'val': '#ff9800', 'test': '#f44336', 'monitoring': '#9c27b0'}
            item_split.setForeground(QColor(split_colors.get(split, '#aaa')))
            item_split.setFlags(item_split.flags() & ~Qt.ItemIsEditable)
            
            self.mdl_slice_table.setItem(row, 0, item_z)
            self.mdl_slice_table.setItem(row, 1, item_px)
            self.mdl_slice_table.setItem(row, 2, item_split)
        
        # Summary
        counts = {}
        for sp in self._slice_splits.values():
            counts[sp] = counts.get(sp, 0) + 1
        summary_parts = [f"{sp}: {c}" for sp, c in sorted(counts.items())]
        self.mdl_lbl_split_summary.setText(
            f"{len(labeled_z)} labeled slices | " + " | ".join(summary_parts)
        )
    
    def _apply_slice_split(self):
        """Apply the selected split to all selected rows in the table."""
        split = self.mdl_combo_slice_split.currentText()
        selected_rows = set(idx.row() for idx in self.mdl_slice_table.selectedIndexes())
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Select slices in the table first.")
            return
        
        for row in selected_rows:
            item = self.mdl_slice_table.item(row, 0)
            if item:
                z = item.data(Qt.UserRole)
                self._slice_splits[z] = split
        
        self._refresh_slice_split_table()
    
    def _apply_range_split(self):
        """Apply split to all labeled slices in a Z range."""
        z_from = self.mdl_spin_split_from.value()
        z_to = self.mdl_spin_split_to.value()
        split = self.mdl_combo_range_split.currentText()
        
        if z_from > z_to:
            z_from, z_to = z_to, z_from
        
        applied = 0
        for z in list(self._slice_splits.keys()):
            if z_from <= z <= z_to:
                self._slice_splits[z] = split
                applied += 1
        
        if applied == 0:
            QMessageBox.information(self, "No Match", 
                f"No labeled slices found in range Z {z_from}→{z_to}.\n"
                f"Click 'Scan Labeled Slices' first.")
        else:
            self._refresh_slice_split_table()
    
    def _auto_split_slices(self):
        """Auto-assign 70% train / 15% val / 15% test to all labeled slices."""
        if not self._slice_splits:
            QMessageBox.warning(self, "No Slices", "Scan labeled slices first.")
            return
        
        z_list = sorted(self._slice_splits.keys())
        np.random.shuffle(z_list)
        n = len(z_list)
        
        for i, z in enumerate(z_list):
            if i < int(n * 0.70):
                self._slice_splits[z] = 'train'
            elif i < int(n * 0.85):
                self._slice_splits[z] = 'val'
            else:
                self._slice_splits[z] = 'test'
        
        self._refresh_slice_split_table()

    # ── Per-class weight spin box helpers ──────────────────────────────────
    def _rebuild_class_weight_spins(self):
        """Rebuild the row of per-class weight QDoubleSpinBoxes to match current class count."""
        # Clear old widgets
        while self.loss_weight_row.count():
            item = self.loss_weight_row.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.loss_weight_spins.clear()

        num_fg = self.mdl_spin_classes.value()
        total_classes = num_fg + 1  # +1 for background

        for c in range(total_classes):
            if c == 0:
                label_text = "BG:"
            else:
                cls_info = self.classes.get(c, {})
                cls_name = cls_info.get('name', f'C{c}')
                # Shorten to max 6 chars for compact display
                label_text = f"{cls_name[:6]}:"

            lbl = QLabel(label_text)
            lbl.setStyleSheet("font-size: 8pt; color: #aaa; padding: 0; margin: 0;")
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 100.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.1)
            spin.setValue(1.0)
            spin.setFixedWidth(60)
            spin.setToolTip(f"Loss weight for class {c} ({label_text.rstrip(':')})")
            spin.setEnabled(not self.mdl_chk_auto_weight.isChecked())

            self.loss_weight_row.addWidget(lbl)
            self.loss_weight_row.addWidget(spin)
            self.loss_weight_spins.append(spin)

        self.loss_weight_row.addStretch()

    def _set_weight_spins_enabled(self, enabled):
        """Enable or disable manual weight spinboxes."""
        for spin in self.loss_weight_spins:
            spin.setEnabled(enabled)

    def _get_manual_class_weights(self):
        """Return list of class weights from spinboxes, or None if auto mode."""
        if self.mdl_chk_auto_weight.isChecked():
            return None
        return [spin.value() for spin in self.loss_weight_spins]

    def _build_train_config(self):
        active_sid = getattr(self, 'active_source_id', 1)
        if active_sid is None: active_sid = 1
        
        return {
            'epochs': self.mdl_spin_ep.value(),
            'batch_size': self.mdl_spin_bs.value(),
            'lr': self.mdl_spin_lr.value(),
            'loss': self.mdl_combo_loss.currentText(),
            'optimizer': self.mdl_combo_opt.currentText(),
            'arch': self.mdl_combo_arch.currentText(),
            'num_classes': self.mdl_spin_classes.value() + 1, # Include background
            'active_source_id': active_sid,
            'patch_size': 256,
            'slice_splits': dict(getattr(self, '_slice_splits', {})),
            'dim': self.mdl_combo_dim.currentText(),
            'adj': self.mdl_spin_adj.value(),
            'class_weights': self._get_manual_class_weights(),
            'augmentation': self._get_augmentation_config(),
            'save_dir': self.mdl_txt_save_dir.text() if self.mdl_txt_save_dir.text() else None,
            'ckpt': self.mdl_txt_ckpt.text()
        }

    def start_training(self):
        if self.volume_data is None or self.segmentation_data is None:
            QMessageBox.warning(self, "No Data", "Please load a volume and create labels first.")
            return
        # Keep active source mask in sync with the current editable segmentation volume.
        if getattr(self, 'sources', None) and self.active_source_id is not None:
            active_src = next((s for s in self.sources if s.get('id') == self.active_source_id), None)
            if active_src is not None:
                active_src['mask'] = self.segmentation_data
            
        active_sid = self.active_source_id if self.active_source_id is not None else 1
        train_frames = [f for f in self.crop_frames if f.get('source_id', active_sid) == active_sid and f['split'] == 'train']
        has_labels = np.any(self.segmentation_data > 0) if self.segmentation_data is not None else False
        if not train_frames and not has_labels:
            QMessageBox.warning(self, "No Data", "No ROI frames and no labeled slices found.\nLabel some voxels or create ROI frames first.")
            return

        config = self._build_train_config()
        
        self.mdl_btn_start.setEnabled(False)
        self.mdl_btn_stop.setEnabled(True)
        self.mdl_lbl_status.setText("Training started...")
        
        # Initialize Chart Data
        self.mdl_chart.clear()
        
        srcs = getattr(self, 'sources', [])
        if not srcs:
            srcs = [{'id': 1, 'volume': self.volume_data, 'mask': self.segmentation_data}]
        elif active_sid is not None:
            active_src = next((s for s in srcs if s.get('id') == active_sid), None)
            if active_src is not None:
                srcs = [active_src] + [s for s in srcs if s is not active_src]

        frames_for_training = [f for f in self.crop_frames if f.get('source_id', active_sid) == active_sid]
        self.train_worker = AITrainWorker(srcs, frames_for_training, config)

        self.train_worker.batch_progress.connect(self.on_batch_progress)
        self.train_worker.epoch_done.connect(self.on_epoch_done)
        self.train_worker.finished_train.connect(self.on_train_complete)
        self.train_worker.error.connect(self.on_train_error)
        self.train_worker.start()

    def on_batch_progress(self, ep, tot_ep, b, tot_b):
        self.mdl_lbl_status.setText(f"Training Epoch {ep}/{tot_ep} | Batch {b}/{tot_b}...")

    def on_epoch_done(self, epoch, tr_loss, val_loss, in_img, gt_img, pr_img):
        msg = f"Epoch: {epoch} | TR: {tr_loss:.4f} | VAL: {val_loss:.4f}"
        self.mdl_lbl_status.setText(msg)
        self.mdl_chart.add_data(tr_loss, val_loss)
        
        # Display preview images
        def to_color_pixmap(arr_2d, is_mask=True):
            if arr_2d is None: return QPixmap()
            h, w = arr_2d.shape
            if not is_mask:
                qimg = QImage(arr_2d.copy().data, w, h, w, QImage.Format_Grayscale8).copy()
                return QPixmap.fromImage(qimg)
            
            colored = np.zeros((h, w, 4), dtype=np.uint8)
            colored[..., 3] = 255
            for cid, info in self.classes.items():
                maskc = (arr_2d == cid)
                if np.any(maskc):
                    colored[maskc] = list(info['color']) + [255]

            # Ensure monitor still visualizes labels/predictions for class IDs that
            # are not currently defined in the class list.
            unknown_ids = np.unique(arr_2d)
            for cid in unknown_ids:
                if cid == 0 or int(cid) in self.classes:
                    continue
                # Deterministic fallback color per class id.
                r = (37 * int(cid) + 53) % 256
                g = (97 * int(cid) + 29) % 256
                b = (17 * int(cid) + 191) % 256
                maskc = (arr_2d == cid)
                if np.any(maskc):
                    colored[maskc] = [r, g, b, 255]
            qimg = QImage(colored.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
            return QPixmap.fromImage(qimg)
            
        self.mdl_prev_in.setPixmap(to_color_pixmap(in_img, False))
        self.mdl_prev_gt.setPixmap(to_color_pixmap(gt_img, True))
        self.mdl_prev_out.setPixmap(to_color_pixmap(pr_img, True))


    def stop_training(self):
        if hasattr(self, 'train_worker') and self.train_worker.isRunning():
            self.train_worker.stop()
            self.mdl_lbl_status.setText("Stopping...")

    def on_train_complete(self, msg):
        self.mdl_lbl_status.setText(msg)
        self.mdl_btn_start.setEnabled(True)
        self.mdl_btn_stop.setEnabled(False)
        
        # Auto-populate checkpoint path for immediate inference
        save_dir = self.mdl_txt_save_dir.text()
        if save_dir:
            best_pth = os.path.join(save_dir, 'best_model.pth')
            if os.path.exists(best_pth):
                self.mdl_txt_ckpt.setText(best_pth)
                self.mdl_lbl_status.setText(
                    msg + f"\n✓ Checkpoint auto-loaded: {os.path.basename(best_pth)}  →  Ready to Infer!"
                )

    def on_train_error(self, err_msg):
        QMessageBox.critical(self, "Training Error", err_msg)
        self.mdl_lbl_status.setText("Training failed.")
        self.mdl_btn_start.setEnabled(True)
        self.mdl_btn_stop.setEnabled(False)

    def export_onnx_standalone(self):
        """Export a trained .pth checkpoint to ONNX with FP16/FP32 + optional pruning."""
        ckpt = self.mdl_txt_ckpt.text()
        if not ckpt or not os.path.exists(ckpt):
            QMessageBox.warning(self, "No Checkpoint", "Select a .pth checkpoint first in section 4.")
            return
        out_dir = os.path.dirname(ckpt)
        
        try:
            import torch
            import torch.nn as nn
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.mdl_lbl_export.setText("Loading model...")
            QApplication.processEvents()
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            state_dict = torch.load(ckpt, map_location=device, weights_only=True)
            out_c = self.mdl_spin_classes.value() + 1
            
            # Auto-detect in_channels from checkpoint weights
            in_c = 1
            for key in ['enc1.0.weight', 'encoder.0.weight', 'conv1.weight']:
                if key in state_dict:
                    in_c = state_dict[key].shape[1]
                    break
            
            from inno3d.models import create_model
            model = create_model(self.mdl_combo_arch.currentText(), in_c, out_c).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            
            # Channel pruning
            prune_ratio = self.mdl_spin_prune.value()
            if prune_ratio > 0:
                self.mdl_lbl_export.setText(f"Pruning {prune_ratio*100:.0f}% channels...")
                QApplication.processEvents()
                import torch.nn.utils.prune as prune
                for name, module in model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        prune.ln_structured(module, name='weight', amount=prune_ratio, n=1, dim=0)
                        prune.remove(module, 'weight')
            
            use_fp16 = "FP16" in self.mdl_combo_precision.currentText()
            if use_fp16:
                model = model.half()
                dummy = torch.randn(1, in_c, 256, 256, dtype=torch.float16).to(device)
                suffix = "_fp16"
            else:
                dummy = torch.randn(1, in_c, 256, 256).to(device)
                suffix = "_fp32"
            
            if prune_ratio > 0:
                suffix += f"_pruned{int(prune_ratio*100)}"
            
            onnx_path = os.path.join(out_dir, f"model{suffix}.onnx")
            self.mdl_lbl_export.setText(f"Exporting ONNX ({suffix.strip('_')})...")
            QApplication.processEvents()
            
            torch.onnx.export(
                model, dummy, onnx_path,
                input_names=['input'], output_names=['output'],
                dynamic_axes={'input': {0:'batch',2:'h',3:'w'}, 'output': {0:'batch',2:'h',3:'w'}},
                opset_version=17
            )
            
            # Report file size
            size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            self.mdl_lbl_export.setText(f"Exported: {os.path.basename(onnx_path)} ({size_mb:.1f} MB)")
            QApplication.restoreOverrideCursor()
            QMessageBox.information(self, "ONNX Export", f"Saved: {onnx_path}\nSize: {size_mb:.1f} MB\nPrecision: {'FP16' if use_fp16 else 'FP32'}")
            
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self.mdl_lbl_export.setText(f"Export failed: {e}")
            QMessageBox.critical(self, "Export Error", str(e))

    def build_tensorrt_engine(self):
        """Convert ONNX model to TensorRT engine for maximum GPU inference speed."""
        # Find ONNX file
        ckpt = self.mdl_txt_ckpt.text()
        if ckpt and os.path.exists(ckpt):
            onnx_dir = os.path.dirname(ckpt)
        else:
            onnx_dir = self.mdl_txt_save_dir.text()
        
        if not onnx_dir:
            QMessageBox.warning(self, "No Directory", "Set a Save Dir or select a checkpoint first.")
            return
        
        # Find latest ONNX in directory
        onnx_files = [f for f in os.listdir(onnx_dir) if f.endswith('.onnx')]
        if not onnx_files:
            QMessageBox.warning(self, "No ONNX", f"No .onnx files found in:\n{onnx_dir}\n\nExport ONNX first.")
            return
        
        onnx_path = os.path.join(onnx_dir, sorted(onnx_files)[-1])
        
        try:
            import tensorrt as trt
        except ImportError:
            QMessageBox.critical(self, "TensorRT Not Found",
                "TensorRT is not installed.\n\n"
                "Install with:\n"
                "  pip install tensorrt\n\n"
                "Or download from NVIDIA:\n"
                "  https://developer.nvidia.com/tensorrt")
            return
        
        use_fp16 = "FP16" in self.mdl_combo_precision.currentText()
        engine_path = onnx_path.replace('.onnx', '.engine')
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.mdl_lbl_export.setText(f"Building TensorRT engine ({'FP16' if use_fp16 else 'FP32'})... This may take minutes.")
        QApplication.processEvents()
        
        try:
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                    raise RuntimeError(f"ONNX parse errors:\n" + "\n".join(errors))
            
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB
            
            if use_fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            
            # Dynamic shape: detect channels from ONNX input, batch 1-8, H/W 64-2048
            inp_tensor = network.get_input(0)
            onnx_in_c = inp_tensor.shape[1] if len(inp_tensor.shape) >= 2 else 1
            profile = builder.create_optimization_profile()
            profile.set_shape("input",
                (1, onnx_in_c, 64, 64),
                (1, onnx_in_c, 512, 512),
                (8, onnx_in_c, 2048, 2048))
            config.add_optimization_profile(profile)
            
            engine_bytes = builder.build_serialized_network(network, config)
            if engine_bytes is None:
                raise RuntimeError("TensorRT engine build failed.")
            
            with open(engine_path, 'wb') as f:
                f.write(engine_bytes)
            
            size_mb = os.path.getsize(engine_path) / (1024 * 1024)
            self.mdl_lbl_export.setText(f"TensorRT: {os.path.basename(engine_path)} ({size_mb:.1f} MB)")
            QApplication.restoreOverrideCursor()
            QMessageBox.information(self, "TensorRT", f"Engine saved: {engine_path}\nSize: {size_mb:.1f} MB\nPrecision: {'FP16' if use_fp16 else 'FP32'}")
            
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self.mdl_lbl_export.setText(f"TensorRT failed: {e}")
            QMessageBox.critical(self, "TensorRT Error", str(e))

    def save_workspace(self):
        if self.volume_data is None:
            QMessageBox.warning(self, "No Data", "No volume loaded to save.")
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save Workspace", "", "Inno3D Workspace (*.inno3d)")
        if not p: return
        import json
        info = {
            'source_path': self.current_source_path,
            'classes': self.classes,
            'crop_frames': self.crop_frames,
            'next_frame_id': self.next_frame_id
        }
        try:
            np.savez_compressed(p, segmentation_data=self.segmentation_data, info=json.dumps(info))
            QMessageBox.information(self, "Saved", "Workspace saved successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")

    def _export_mask_array(self, mask, title_prefix):
        dlg = QDialog(self)
        dlg.setWindowTitle(title_prefix)
        dlg.setMinimumWidth(380)
        lay = QVBoxLayout(dlg)

        info_lbl = QLabel(
            f"Mask shape: {mask.shape}  |  "
            f"Foreground voxels: {np.count_nonzero(mask):,}"
        )
        info_lbl.setWordWrap(True)
        lay.addWidget(info_lbl)

        # Slice range option (only for 3D masks)
        if mask.ndim == 3 and mask.shape[0] > 1:
            lay.addSpacing(4)
            chk_range = QCheckBox("Export specific slice range only")
            lay.addWidget(chk_range)
            range_row = QHBoxLayout()
            range_row.addWidget(QLabel("From Z:"))
            spin_from = QSpinBox(); spin_from.setRange(0, mask.shape[0] - 1); spin_from.setValue(0)
            range_row.addWidget(spin_from)
            range_row.addWidget(QLabel("To Z:"))
            spin_to = QSpinBox(); spin_to.setRange(0, mask.shape[0] - 1); spin_to.setValue(mask.shape[0] - 1)
            range_row.addWidget(spin_to)
            range_widget = QWidget()
            range_widget.setLayout(range_row)
            range_widget.setEnabled(False)
            chk_range.toggled.connect(range_widget.setEnabled)
            lay.addWidget(range_widget)
        else:
            chk_range = None

        lay.addSpacing(8)
        lbl_fmt = QLabel("Export Format:")
        lay.addWidget(lbl_fmt)

        rb_tiff = QRadioButton("TIFF Stack (.tif) - single multi-page file")
        rb_slices = QRadioButton("PNG Slices - one file per Z slice (folder)")
        rb_npz = QRadioButton("NumPy Compressed (.npz)")
        rb_npy = QRadioButton("NumPy Raw (.npy)")
        rb_tiff.setChecked(True)
        for rb in [rb_tiff, rb_slices, rb_npz, rb_npy]:
            lay.addWidget(rb)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        lay.addWidget(btn_box)

        if dlg.exec() != QDialog.Accepted:
            return

        # Apply slice range if selected
        if chk_range is not None and chk_range.isChecked():
            z_from = spin_from.value()
            z_to = spin_to.value()
            if z_from > z_to:
                z_from, z_to = z_to, z_from
            mask = mask[z_from:z_to + 1]
            title_prefix += f" (Z {z_from}→{z_to})"

        try:
            if rb_tiff.isChecked():
                p, _ = QFileDialog.getSaveFileName(self, f"Save {title_prefix}", "", "TIFF Files (*.tif *.tiff)")
                if not p:
                    return
                io.imsave(p, mask.astype(np.uint8), check_contrast=False)
                QMessageBox.information(self, "Exported", f"TIFF stack saved: {p}\nShape: {mask.shape}")

            elif rb_slices.isChecked():
                folder = QFileDialog.getExistingDirectory(self, "Select Output Folder for Slices")
                if not folder:
                    return
                from PIL import Image
                for z in range(mask.shape[0]):
                    fname = os.path.join(folder, f"mask_{z:05d}.png")
                    Image.fromarray(mask[z].astype(np.uint8)).save(fname)
                QMessageBox.information(self, "Exported", f"Saved {mask.shape[0]} slices to: {folder}")

            elif rb_npz.isChecked():
                p, _ = QFileDialog.getSaveFileName(self, f"Save {title_prefix}", "", "NumPy Compressed (*.npz)")
                if not p:
                    return
                np.savez_compressed(p, mask=mask.astype(np.uint8))
                QMessageBox.information(self, "Exported", f"NPZ saved: {p}\nShape: {mask.shape}")

            elif rb_npy.isChecked():
                p, _ = QFileDialog.getSaveFileName(self, f"Save {title_prefix}", "", "NumPy Array (*.npy)")
                if not p:
                    return
                np.save(p, mask.astype(np.uint8))
                QMessageBox.information(self, "Exported", f"NPY saved: {p}\nShape: {mask.shape}")

        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Export Error", f"{str(e)}\n{traceback.format_exc()}")

    def export_class_mask(self, class_id):
        if self.segmentation_data is None or not np.any(self.segmentation_data == class_id):
            class_name = self.classes.get(class_id, {}).get('name', f"Class {class_id}")
            QMessageBox.warning(self, "No Mask", f"No voxels found for {class_name}.")
            return

        mask = (self.segmentation_data == class_id).astype(np.uint8) * 255
        class_name = self.classes.get(class_id, {}).get('name', f"Class {class_id}")
        self._export_mask_array(mask, f"Export {class_name} Mask")

    def export_class1_mask(self):
        self.export_class_mask(1)

    def export_class2_mask(self):
        self.export_class_mask(2)

    def export_combined_mask(self):
        """Export all classes in a single 8-bit mask: C1=128, C2=255, background=0."""
        if self.segmentation_data is None or not np.any(self.segmentation_data > 0):
            QMessageBox.warning(self, "No Mask", "No labeled voxels found.")
            return

        # Build combined mask: map each class to a unique gray value
        combined = np.zeros(self.segmentation_data.shape, dtype=np.uint8)
        class_ids = sorted(self.classes.keys())
        
        if len(class_ids) <= 2:
            # Default: C1=128, C2=255
            gray_map = {class_ids[0]: 128}
            if len(class_ids) > 1:
                gray_map[class_ids[1]] = 255
        else:
            # Auto-distribute: evenly spaced from 64 to 255
            step = max(1, 255 // len(class_ids))
            gray_map = {cid: min(255, (i + 1) * step) for i, cid in enumerate(class_ids)}

        info_lines = []
        for cid, gv in gray_map.items():
            mask_idx = (self.segmentation_data == cid)
            combined[mask_idx] = gv
            count = int(np.count_nonzero(mask_idx))
            name = self.classes.get(cid, {}).get('name', f'Class {cid}')
            info_lines.append(f"  {name} (ID {cid}) -> gray {gv}  ({count:,} voxels)")

        detail = "\n".join(info_lines)
        reply = QMessageBox.question(
            self, "Export Combined Mask",
            f"Gray value mapping:\n{detail}\n\nProceed?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        self._export_mask_array(combined, "Export Combined Mask")


    def export_label_range(self):
        """Export labels from a specific Z slice range with format selection dialog."""
        if self.segmentation_data is None or not np.any(self.segmentation_data > 0):
            QMessageBox.warning(self, "No Labels", "No labeled voxels found.")
            return
        self._export_mask_array(self.segmentation_data.copy(), "Export Label Range")

    def export_training_dataset(self):
        """Export labeled slices (full original shape) to train/val/test folders."""
        if self.segmentation_data is None or not np.any(self.segmentation_data > 0):
            QMessageBox.warning(self, "No Mask", "No labeled voxels found.")
            return

        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder for Training Dataset")
        if not folder:
            return

        # Build combined mask: C1=128, C2=255
        combined = np.zeros(self.segmentation_data.shape, dtype=np.uint8)
        class_ids = sorted(self.classes.keys())
        gray_map = {}
        if len(class_ids) <= 2:
            gray_map = {class_ids[0]: 128}
            if len(class_ids) > 1:
                gray_map[class_ids[1]] = 255
        else:
            step = max(1, 255 // len(class_ids))
            gray_map = {cid: min(255, (i + 1) * step) for i, cid in enumerate(class_ids)}
        
        class_mapping_info = {}
        for cid, gv in gray_map.items():
            mask_idx = (self.segmentation_data == cid)
            combined[mask_idx] = gv
            class_mapping_info[str(gv)] = self.classes.get(cid, {}).get('name', f'Class {cid}')

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            import json
            from PIL import Image
            import tifffile
            
            # Setup folders
            splits = ['train', 'val', 'test', 'mix', 'monitoring']
            for sp in splits:
                os.makedirs(os.path.join(folder, sp, 'images'), exist_ok=True)
                os.makedirs(os.path.join(folder, sp, 'masks'), exist_ok=True)

            # Find slices to export
            # 1. Slices with labels
            labeled_slices = set(np.where(np.any(self.segmentation_data > 0, axis=(1, 2)))[0])
            
            # 2. Assign split per slice
            slice_splits = {}
            # Frame assignments take priority
            for frame in self.frame_boxes:
                z = frame['slice']
                sp = frame.get('split', 'train')
                slice_splits[z] = sp
            
            # For unlabeled but framed slices (just in case they are negative samples)
            all_export_slices = sorted(list(labeled_slices.union(set(slice_splits.keys()))))
            
            # Assign auto split for slices without frames
            auto_slices = [z for z in all_export_slices if z not in slice_splits]
            np.random.shuffle(auto_slices)
            n_auto = len(auto_slices)
            for i, z in enumerate(auto_slices):
                if i < int(n_auto * 0.7):
                    slice_splits[z] = 'train'
                elif i < int(n_auto * 0.85):
                    slice_splits[z] = 'val'
                else:
                    slice_splits[z] = 'test'
            
            # Export
            exported_count = {sp: 0 for sp in splits}
            for z in all_export_slices:
                sp = slice_splits[z]
                
                # Image: tifffile to preserve exact 8/16bit values (full 1000x1000)
                img_path = os.path.join(folder, sp, 'images', f"z{z:04d}.tif")
                tifffile.imwrite(img_path, self.volume_data[z])
                
                # Mask: standard 8-bit png
                mask_path = os.path.join(folder, sp, 'masks', f"z{z:04d}.png")
                Image.fromarray(combined[z]).save(mask_path)
                
                exported_count[sp] += 1

            # Save metadata
            meta = {
                "class_mapping": class_mapping_info,
                "original_shape": list(self.volume_data.shape),
                "splits_count": exported_count
            }
            with open(os.path.join(folder, "metadata.json"), "w") as jf:
                json.dump(meta, jf, indent=4)
                
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"Failed to export dataset:\n{str(e)}")
            return
            
        QApplication.restoreOverrideCursor()
        
        msg = f"Exported {len(all_export_slices)} full slices (1000x1000) to:\n{folder}\n\n"
        for sp, count in exported_count.items():
            if count > 0:
                msg += f"- {sp}: {count} slices\n"
        QMessageBox.information(self, "Export Complete", msg)

    def load_workspace(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load Workspace", "", "Inno3D Workspace (*.inno3d)")
        if not p: return
        try:
            import json
            data = np.load(p, allow_pickle=True)
            info = json.loads(str(data['info']))
            
            if self.volume_data is None or self.current_source_path != info['source_path']:
                reply = QMessageBox.question(self, "Load Source Volume", 
                    f"This workspace requires the source volume:\n{info['source_path']}\n\nDo you want to load it now? (App will load the segmentation data afterwards)",
                    QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    self.pending_workspace_data = data
                    self.pending_workspace_info = info
                    self.pending_load_path = info['source_path']
                    self.pg = QProgressDialog("Loading Source Volume...","Cancel",0,100,self); self.pg.show()
                    self.t = AILoadVolumeThread(self.pending_load_path); self.t.progress.connect(lambda v,m: (self.pg.setValue(v),self.pg.setLabelText(m)))
                    self.t.finished.connect(self._done_workspace_load)
                    self.t.start()
                    return
                else:
                    return
            else:
                self._apply_workspace(info, data)
        except Exception as e:
             QMessageBox.critical(self, "Error", f"Failed to load: {str(e)}")

    def _done_workspace_load(self, d, e):
        self._done(d, e)
        if not e and hasattr(self, 'pending_workspace_info'):
            self._apply_workspace(self.pending_workspace_info, self.pending_workspace_data)
            del self.pending_workspace_info
            del self.pending_workspace_data

    def _apply_workspace(self, info, data):
        self.segmentation_data = data['segmentation_data']
        # Important: workspace load replaces segmentation array object.
        # We must also update the active source entry to keep training/inference consistent.
        if getattr(self, 'sources', None):
            active_src = next((s for s in self.sources if s.get('id') == self.active_source_id), None)
            if active_src is not None:
                active_src['mask'] = self.segmentation_data

        # Refresh viewer references to point to the newly loaded mask array.
        for v in getattr(self, 'all_viewers', []):
            v.segmentation_data = self.segmentation_data
            v.display_segmentation_data = None

        self.classes = {int(k): v for k, v in info['classes'].items()} 
        self.crop_frames = info['crop_frames']
        self.next_frame_id = info['next_frame_id']
        self.undo_stack, self.redo_stack = [], []
        self._refresh()
        self.refresh_region_index()
        self.sync()
        QMessageBox.information(self, "Loaded", "Workspace loaded successfully!")
