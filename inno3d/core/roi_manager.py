"""
ROI Manager - Manages Region of Interest state for 3D volume data.
Separates ROI logic (undo/redo, cropping, memory estimation) from the UI.
"""
import numpy as np
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal


class ROIManager:
    """Manages ROI state with undo/redo history (max 10 steps)."""
    
    MAX_HISTORY = 10
    
    def __init__(self):
        self.current_roi = None  # dict: {x1, y1, x2, y2, z_start, z_end}
        self._undo_stack = []    # list of roi dicts
        self._redo_stack = []    # list of roi dicts
    
    def set_roi(self, x1, y1, x2, y2, z_start, z_end):
        """Set a new ROI, pushing the previous one onto undo stack."""
        # Store as-is (normalization will happen at point of use)
        
        # Push current to undo stack (if it exists)
        if self.current_roi is not None:
            # Check if fundamentally different (ignore sub-pixel/float noise if any)
            curr = self.current_roi
            if (curr['x1'] == x1 and curr['y1'] == y1 and 
                curr['x2'] == x2 and curr['y2'] == y2 and
                curr['z_start'] == z_start and curr['z_end'] == z_end):
                return curr

            self._undo_stack.append(self.current_roi.copy())
            # Trim undo stack to MAX_HISTORY
            if len(self._undo_stack) > self.MAX_HISTORY:
                self._undo_stack = self._undo_stack[-self.MAX_HISTORY:]
        
        # Clear redo stack on new action
        self._redo_stack.clear()
        
        self.current_roi = {
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'z_start': z_start, 'z_end': z_end
        }
        return self.current_roi
    
    def update_z_range(self, z_start, z_end):
        """Update only the Z range of the current ROI (pushes to undo)."""
        if self.current_roi is None:
            return None
        return self.set_roi(
            self.current_roi['x1'], self.current_roi['y1'],
            self.current_roi['x2'], self.current_roi['y2'],
            z_start, z_end
        )
    
    def undo(self):
        """Undo to previous ROI. Returns the restored ROI or None."""
        if not self._undo_stack:
            return None
        
        # Push current to redo
        if self.current_roi is not None:
            self._redo_stack.append(self.current_roi.copy())
        
        self.current_roi = self._undo_stack.pop()
        return self.current_roi
    
    def redo(self):
        """Redo to next ROI. Returns the restored ROI or None."""
        if not self._redo_stack:
            return None
        
        # Push current to undo
        if self.current_roi is not None:
            self._undo_stack.append(self.current_roi.copy())
        
        self.current_roi = self._redo_stack.pop()
        return self.current_roi
    
    @property
    def can_undo(self):
        return len(self._undo_stack) > 0
    
    @property
    def can_redo(self):
        return len(self._redo_stack) > 0
    
    @property
    def has_valid_roi(self):
        if self.current_roi is None:
            return False
        roi = self.current_roi
        # Valid means some area in all 3 axes
        return (abs(roi['x2'] - roi['x1']) > 0 and 
                abs(roi['y2'] - roi['y1']) > 0 and 
                abs(roi['z_end'] - roi['z_start']) > 0)
    
    def get_roi_size(self):
        """Returns (dx, dy, dz) tuple or None."""
        if not self.has_valid_roi:
            return None
        roi = self.current_roi
        return (
            abs(roi['x2'] - roi['x1']),
            abs(roi['y2'] - roi['y1']),
            abs(roi['z_end'] - roi['z_start'])
        )
    
    def estimate_memory_mb(self, volume_dtype=np.uint16):
        """Estimate memory in MB for the cropped region."""
        size = self.get_roi_size()
        if size is None:
            return 0.0
        dx, dy, dz = size
        itemsize = np.dtype(volume_dtype).itemsize
        return (dx * dy * dz * itemsize) / (1024 * 1024)
    
    def get_crop(self, volume):
        """Extract the cropped sub-volume from the full volume array.
        
        Args:
            volume: numpy array with shape (Z, Y, X)
        
        Returns:
            Cropped numpy array copy
        """
        if not self.has_valid_roi or volume is None:
            return None
        roi = self.current_roi
        # Universal min/max normalization here
        z1, z2 = sorted([roi['z_start'], roi['z_end']])
        y1, y2 = sorted([roi['y1'], roi['y2']])
        x1, x2 = sorted([roi['x1'], roi['x2']])
        
        z1 = max(0, int(z1))
        z2 = min(volume.shape[0], int(z2))
        y1 = max(0, int(y1))
        y2 = min(volume.shape[1], int(y2))
        x1 = max(0, int(x1))
        x2 = min(volume.shape[2], int(x2))
        
        if (z2 - z1 <= 0) or (y2 - y1 <= 0) or (x2 - x1 <= 0):
            return None
            
        return volume[z1:z2, y1:y2, x1:x2].copy()
    
    def clear(self):
        """Reset all ROI state."""
        self.current_roi = None
        self._undo_stack.clear()
        self._redo_stack.clear()


class ExportCropThread(QThread):
    """Background thread for exporting large cropped volumes (>500MB)."""
    
    finished = pyqtSignal(bool, str)  # (success, message)
    progress = pyqtSignal(int, str)   # (percent, status_text)
    
    def __init__(self, volume, roi_dict, save_path, save_format='npy'):
        """
        Args:
            volume: Full volume numpy array (Z, Y, X)
            roi_dict: dict with x1, y1, x2, y2, z_start, z_end
            save_path: Output file path string
            save_format: 'npy' or 'tiff'
        """
        super().__init__()
        self.volume = volume
        self.roi_dict = roi_dict
        self.save_path = save_path
        self.save_format = save_format
    
    def run(self):
        try:
            self.progress.emit(10, "Cropping volume...")
            
            roi = self.roi_dict
            z1 = max(0, roi['z_start'])
            z2 = min(self.volume.shape[0], roi['z_end'])
            y1 = max(0, roi['y1'])
            y2 = min(self.volume.shape[1], roi['y2'])
            x1 = max(0, roi['x1'])
            x2 = min(self.volume.shape[2], roi['x2'])
            
            sub_vol = self.volume[z1:z2, y1:y2, x1:x2].copy()
            
            self.progress.emit(50, "Saving file...")
            
            if self.save_format == 'npy':
                np.save(self.save_path, sub_vol)
            elif self.save_format == 'tiff':
                from skimage import io as skio
                skio.imsave(self.save_path, sub_vol)
            else:
                self.finished.emit(False, f"Unsupported format: {self.save_format}")
                return
            
            size_mb = sub_vol.nbytes / (1024 * 1024)
            self.progress.emit(100, "Export complete")
            self.finished.emit(True, f"Exported successfully ({size_mb:.1f} MB) → {self.save_path}")
            
        except Exception as e:
            import traceback
            self.finished.emit(False, f"Export failed: {str(e)}\n{traceback.format_exc()}")
