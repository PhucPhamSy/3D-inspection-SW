# inno3d/features/viewer/volume_io.py
# -----------------------------------------------------------------------
# VolumeIOMixin  --  extracted from inno3d/tabs/viewer.py (Phase 3.8)
#
# load_volume, load_volume_from_path, load_segmentation, on_volume_loaded,
# batch review / MES parse, HBM alignment, save_aligned_volume,
# camera state save/restore, reset_3d_view, update_info_label.
# -----------------------------------------------------------------------

import glob
import json
import math
import os
import re
import traceback
from pathlib import Path
from pathlib import Path as _Path
from typing import Dict, List, Optional

import numpy as np
import vtk
from scipy import ndimage
from skimage import io

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QFileDialog, QMessageBox, QWidget, QProgressDialog, QInputDialog,
    QApplication,
)

from inno3d.core.styles import SemiconductorTheme
from inno3d.core.view_support import (
    LoadVolumeThread,
    apply_dragonfly_volume_zoom,
    push_camera_outside_aabb,
    volume_world_aabb,
)
from inno3d.core.perf_timing import perf_phase
from inno3d.core.online_mpr_walltime import online_walltime_phase
from inno3d.infra.textio import read_text_auto


class VolumeIOMixin:
    """Mixin: Volume load/IO, batch review, HBM alignment, camera utilities.

    Extracted from inno3d/tabs/viewer.py Phase 3.8.
    """

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self._handle_volume_load(file_path)

    def load_volume(self, from_folder=False):
        """Load 16-bit volume data.

        Opens a unified file dialog where the user can select either
        a single 3D file (.tif/.tiff/.raw/.bin) **or** a folder that
        contains a stack of images.  The selection type is determined
        at the time of user interaction — no separate buttons needed.

        The *from_folder* parameter is kept for backward-compatibility
        with callers that still pass it, but it is now ignored when
        the dialog is shown interactively.
        """
        dlg = QFileDialog(self, "Open File or Folder")
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter("Image Files (*.tif *.tiff *.raw *.bin)")
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        # Allow the user to also select directories in the same dialog
        dlg.setOption(QFileDialog.ShowDirsOnly, False)
        # Permit clicking on a directory to select it (instead of
        # only entering it).  We achieve this by accepting both files
        # and directories via a small proxy-model trick on the dialog's
        # internal tree/list views.
        from PyQt5.QtWidgets import QTreeView, QListView, QAbstractItemView
        for view in dlg.findChildren((QTreeView, QListView)):
            if isinstance(view, (QTreeView, QListView)):
                view.setSelectionMode(QAbstractItemView.SingleSelection)

        # Override the accept logic so that selecting a directory also
        # closes the dialog successfully (by default QFileDialog in
        # ExistingFile mode only accepts files).
        _orig_accept = dlg.accept

        def _custom_accept():
            selected = dlg.selectedFiles()
            if selected and os.path.isdir(selected[0]):
                # Directory selected — accept immediately
                dlg.done(QFileDialog.Accepted)
                return
            _orig_accept()

        dlg.accept = _custom_accept

        if dlg.exec_() == QFileDialog.Accepted:
            selected = dlg.selectedFiles()
            if selected:
                self._handle_volume_load(selected[0])

    def load_volume_from_path(self, file_path, mask_bump_path=None, mask_void_path=None):
        """Programmatic load (Batch Review / Online deep-link).

        Optionally load combined bump/void masks after volume finishes.
        """
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(
                self,
                "Load volume",
                f"Volume path not found:\n{file_path or '(empty)'}",
            )
            return False
        # Stash masks to apply in on_volume_loaded
        self._pending_mask_bump = mask_bump_path or ""
        self._pending_mask_void = mask_void_path or ""
        self._handle_volume_load(file_path)
        return True

    # ------------------------------------------------------------------
    # Batch Review → Full Viewer parity (volume + masks + MES + B2B + map)
    # ------------------------------------------------------------------
    @staticmethod
    def resolve_batch_run_paths(run: dict):
        """Resolve volume / bump / void paths from a FOV run dict (DB + artifacts)."""
        run = run or {}
        results = run.get("results_dir") or ""
        arts = {
            a.get("kind"): a.get("path")
            for a in (run.get("artifacts") or [])
            if isinstance(a, dict) and a.get("path")
        }

        vol = ""
        for cand in (
            arts.get("enhanced_volume"),
            os.path.join(results, "enhanced_volume", "Enhanced_Volume.tif") if results else "",
            os.path.join(results, "enhanced_volume", "Enhanced_Volume.tiff") if results else "",
            run.get("input_path") or "",
            run.get("host_path") or "",
        ):
            if cand and os.path.isfile(cand):
                vol = cand
                break

        bump = arts.get("mask_bump_far") or arts.get("mask_bump") or ""
        void = arts.get("mask_void_far") or arts.get("mask_void") or ""
        if results:
            if not (bump and os.path.isfile(bump)):
                for cand in (
                    os.path.join(results, "online_combined_bump3D_FAR.tif"),
                    os.path.join(results, "online_combined_bump3D.tif"),
                ):
                    if os.path.isfile(cand):
                        bump = cand
                        break
            if not (void and os.path.isfile(void)):
                for cand in (
                    os.path.join(results, "online_combined_voidsOnly_FAR.tif"),
                    os.path.join(results, "online_combined_voidsOnly.tif"),
                ):
                    if os.path.isfile(cand):
                        void = cand
                        break
        return vol, bump, void

    @staticmethod
    def _normalize_mes_stat_row(s: dict, row_fallback: int = 0) -> dict:
        """Normalize DB / CSV MES object row → Viewer object_stats schema."""
        if not isinstance(s, dict):
            return {}

        def _f(key, default=0.0):
            try:
                v = s.get(key, default)
                if v is None or v == "":
                    return default
                return float(v)
            except (TypeError, ValueError):
                return default

        def _i(key, default=0):
            try:
                v = s.get(key, default)
                if v is None or v == "":
                    return default
                return int(float(v))
            except (TypeError, ValueError):
                return default

        ratio = _f("ratio", 0.0)
        # CSV sometimes stores percent 0..100; DB/online use fraction 0..1
        if ratio > 1.0 and ratio <= 100.0 and "ratio_pct" not in s:
            # Heuristic: values like 12.5 mean 12.5% if no explicit fraction
            # Prefer fraction if clearly already fraction
            pass
        if ratio > 1.5:  # almost certainly percent
            ratio = ratio / 100.0

        jud = s.get("judgment")
        if jud in (1, 8):
            jud = "NG" if jud == 1 else "OK"
        elif jud is None:
            jud = ""
        else:
            jud = str(jud)

        gr = s.get("grid_row", s.get("Grid_row", -1))
        gc = s.get("grid_col", s.get("Grid_col", -1))
        try:
            gr = int(float(gr)) if gr not in (None, "", "?") else -1
        except (TypeError, ValueError):
            gr = -1
        try:
            gc = int(float(gc)) if gc not in (None, "", "?") else -1
        except (TypeError, ValueError):
            gc = -1

        label = _i("label", 0)
        return {
            "row_id": _i("row_id", row_fallback) or row_fallback,
            "label": label,
            "layer_name": str(s.get("layer_name") or s.get("Layer") or ""),
            "grid_row": gr,
            "grid_col": gc,
            "soh": _f("soh", 0.0),
            "c1_volume": _f("c1_volume", 0.0),
            "c2_volume": _f("c2_volume", 0.0),
            "ratio": ratio,
            "judgment": jud,
            "pitch_x": _f("pitch_x", 0.0),
            "pitch_y": _f("pitch_y", 0.0),
            "z_min": _i("z_min", 0),
            "z_max": _i("z_max", 0),
            "y_min": _i("y_min", 0),
            "y_max": _i("y_max", 0),
            "x_min": _i("x_min", 0),
            "x_max": _i("x_max", 0),
            "centroid_z": _f("centroid_z", 0.0),
            "centroid_y": _f("centroid_y", 0.0),
            "centroid_x": _f("centroid_x", 0.0),
        }

    @classmethod
    def _parse_mes_object_statistics_csv(cls, path: str):
        """Parse Online/Viewer object_statistics.csv into object_stats list."""
        import csv
        import re

        if not path or not os.path.isfile(path):
            return []
        stats = []
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return []
            # Normalize header keys
            for i, raw in enumerate(reader):
                # Map flexible headers → internal keys
                lower = {(k or "").strip().lower(): (k or "") for k in raw.keys()}

                def pick(*names, default=""):
                    for n in names:
                        k = lower.get(n.lower())
                        if k is not None and raw.get(k) not in (None, ""):
                            return raw.get(k)
                    return default

                bump = str(pick("Bump ID", "bump id", "bump_id", default=""))
                gr, gc = -1, -1
                m = re.search(r"\(\s*([-\d]+)\s*,\s*([-\d]+)\s*\)", bump)
                if m:
                    gr, gc = int(m.group(1)), int(m.group(2))

                ratio_raw = pick(
                    "Ratio (C2/(C1+C2))", "Ratio (%)", "ratio", "Ratio", default="0"
                )
                ratio_s = str(ratio_raw).replace("%", "").strip()
                try:
                    ratio_v = float(ratio_s) if ratio_s and ratio_s.upper() != "N/A" else 0.0
                except ValueError:
                    ratio_v = 0.0
                # Viewer export writes percent; Online write_python_format_csv writes percent too
                if ratio_v > 1.5:
                    ratio_v = ratio_v / 100.0

                row = {
                    "row_id": pick("#", "row_id", default=i + 1),
                    "layer_name": pick("Layer", "layer_name", default=""),
                    "grid_row": gr,
                    "grid_col": gc,
                    "soh": pick("B. H", "soh", default=0),
                    "c1_volume": pick("B. V", "c1_volume", default=0),
                    "c2_volume": pick("V. V", "c2_volume", default=0),
                    "ratio": ratio_v,
                    "judgment": pick("Judgment", "judgment", default=""),
                    "pitch_x": pick("Gap X (um)", "Gap X", "pitch_x", default=0),
                    "pitch_y": pick("Gap Y (um)", "Gap Y", "pitch_y", default=0),
                    "z_min": pick("Z_min", "z_min", default=0),
                    "z_max": pick("Z_max", "z_max", default=0),
                    "y_min": pick("Y_min", "y_min", default=0),
                    "y_max": pick("Y_max", "y_max", default=0),
                    "x_min": pick("X_min", "x_min", default=0),
                    "x_max": pick("X_max", "x_max", default=0),
                    "centroid_z": pick("Centroid_Z", "centroid_z", default=0),
                    "centroid_y": pick("Centroid_Y", "centroid_y", default=0),
                    "centroid_x": pick("Centroid_X", "centroid_x", default=0),
                    "label": pick("label", "Label", default=0),
                }
                stats.append(cls._normalize_mes_stat_row(row, row_fallback=i + 1))
        return stats

    @classmethod
    def extract_mes_stats_from_run(cls, run: dict):
        """MES objects: prefer DB rows, fallback object_statistics.csv on disk."""
        run = run or {}
        objs = run.get("mes_objects") or []
        stats = []
        if objs:
            for i, s in enumerate(objs):
                if isinstance(s, dict):
                    stats.append(cls._normalize_mes_stat_row(s, row_fallback=i + 1))
            if stats:
                return stats

        results = run.get("results_dir") or ""
        arts = {
            a.get("kind"): a.get("path")
            for a in (run.get("artifacts") or [])
            if isinstance(a, dict) and a.get("path")
        }
        for cand in (
            arts.get("csv_mes"),
            arts.get("mes_csv"),
            os.path.join(results, "object_statistics.csv") if results else "",
        ):
            if cand and os.path.isfile(cand):
                parsed = cls._parse_mes_object_statistics_csv(cand)
                if parsed:
                    return parsed
        return []

    @staticmethod
    def extract_b2b_rows_from_run(run: dict):
        """B2B gap/summary rows from artifacts or Results folder CSV."""
        run = run or {}
        results = run.get("results_dir") or ""
        arts = {
            a.get("kind"): a.get("path")
            for a in (run.get("artifacts") or [])
            if isinstance(a, dict) and a.get("path")
        }
        candidates = [
            arts.get("csv_b2b"),
            arts.get("b2b_csv"),
            os.path.join(results, "boundary_gap_all_layers.csv") if results else "",
            os.path.join(results, "boundary_summary_all_layers.csv") if results else "",
            os.path.join(results, "boundary_gap.csv") if results else "",
            os.path.join(results, "boundary_summary.csv") if results else "",
        ]
        try:
            from inno3d.core.bumpvoid_b2b import read_boundary_csv
        except Exception as e:
            print(f"[Viewer] B2B import failed: {e}")
            return []

        for path in candidates:
            if path and os.path.isfile(path):
                try:
                    rows = read_boundary_csv(path)
                    if rows:
                        return rows
                except Exception as e:
                    print(f"[Viewer] B2B CSV read failed ({path}): {e}")
        return []

    def show_stats_panel_for_review(self, visible: bool = True, collapse_tools: bool = True):
        """Show STATISTICS (MES/B2B) panel — same chrome Online uses.

        Offline/Batch Review default keeps ``stats_panel`` hidden; Open Full Viewer
        must call this or the tables fill but stay invisible.
        """
        visible = bool(visible)
        if hasattr(self, "stats_panel") and self.stats_panel is not None:
            self.stats_panel.setVisible(visible)
        # Online log stays Offline-style (no live TCP log for batch replay)
        if hasattr(self, "set_online_log_visible"):
            try:
                self.set_online_log_visible(False)
            except Exception:
                pass
        if visible and collapse_tools and self.volume_data is not None:
            if hasattr(self, "align_tools_host"):
                self.align_tools_host.setVisible(True)
            # More room for MPR + stats (user can reopen tools via rail)
            if hasattr(self, "set_align_tools_expanded"):
                try:
                    self.set_align_tools_expanded(False, remember=False)
                except TypeError:
                    self.set_align_tools_expanded(False)
        if visible and hasattr(self, "_rebalance_main_splitter_for_tools"):
            expanded = bool(getattr(self, "_align_tools_expanded", True))
            try:
                self._rebalance_main_splitter_for_tools(expanded)
            except Exception:
                pass
            # Force a readable stats width even if splitter had 0
            try:
                sp = getattr(self, "_main_splitter", None)
                if sp is not None and self.stats_panel is not None and self.stats_panel.isVisible():
                    sizes = sp.sizes()
                    if len(sizes) >= 3 and sizes[2] < 260:
                        total = sum(sizes) or max(sp.width(), 1600)
                        tools_w = sizes[1] if sizes[1] > 0 else 28
                        stats_w = max(320, min(420, total // 4))
                        grid_w = max(480, total - tools_w - stats_w)
                        sp.setSizes([grid_w, tools_w, stats_w])
            except Exception:
                pass
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(50, getattr(self, "_reposition_visible_overlays", lambda: None))

    def load_batch_review_run(self, run: dict):
        """Full parity deep-link from Batch Review → volume + masks + MES + B2B.

        Volume load is async; MES/B2B tables and label mapping are applied after
        volume + masks land (``_apply_pending_batch_replay_if_any``).
        """
        run = run or {}
        vol, bump, void = self.resolve_batch_run_paths(run)
        if not vol:
            QMessageBox.warning(
                self,
                "Open full Viewer",
                "No volume file found on disk for this run.\n\n"
                f"input_path: {run.get('input_path') or '—'}\n"
                f"results: {run.get('results_dir') or '—'}\n\n"
                "DB only stores paths — file may be missing or on another PC.",
            )
            return False

        mes_stats = self.extract_mes_stats_from_run(run)
        b2b_rows = self.extract_b2b_rows_from_run(run)

        # Show STATISTICS panel immediately (was hidden offline — looked "no table")
        self.show_stats_panel_for_review(True, collapse_tools=False)
        if hasattr(self, "set_stats_info"):
            self.set_stats_info(
                f"Loading Batch FOV · Chip({run.get('chip_col')},{run.get('chip_row')}) "
                f"P{run.get('fov_index') or '?'} · MES {len(mes_stats)} · B2B {len(b2b_rows)}…"
            )

        # Stash for post-load application (after masks)
        self._pending_batch_replay = {
            "run_id": run.get("run_id") or "",
            "mes_stats": mes_stats,
            "b2b_rows": b2b_rows,
            "source_label": (
                f"Batch · Chip({run.get('chip_col')},{run.get('chip_row')}) "
                f"P{run.get('fov_index') or '?'}"
            ),
            "chip_col": run.get("chip_col"),
            "chip_row": run.get("chip_row"),
            "fov_index": run.get("fov_index"),
        }

        ok = self.load_volume_from_path(
            vol,
            mask_bump_path=bump if bump and os.path.isfile(bump) else None,
            mask_void_path=void if void and os.path.isfile(void) else None,
        )
        if not ok:
            self._pending_batch_replay = None
        return ok

    def _assign_mes_labels_from_mask(self, stats):
        """Rebuild labeled_class1 from bump mask and map each MES row → label.

        Production MES CSV/DB often omit instance ``label``; without it table→
        image highlight cannot work. Match by centroid (then bbox center).
        """
        import numpy as np

        stats = list(stats or [])
        c1 = getattr(self, "class1_data", None)
        if c1 is None or self.volume_data is None or not stats:
            return stats

        try:
            from skimage.measure import label as sk_label
        except Exception:
            try:
                from scipy.ndimage import label as _scipy_label

                def sk_label(mask):
                    lab, _ = _scipy_label(mask)
                    return lab
            except Exception as e:
                print(f"[Viewer] cannot label mask for MES map: {e}")
                return stats

        try:
            binary = (np.asarray(c1) > 0).astype(np.uint8)
            labeled = sk_label(binary).astype(np.uint32)
            self.labeled_class1_data = labeled
            Z, Y, X = labeled.shape

            def _lookup(z, y, x):
                z = int(round(z))
                y = int(round(y))
                x = int(round(x))
                if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
                    return int(labeled[z, y, x])
                return 0

            matched = 0
            for s in stats:
                lab = int(s.get("label") or 0)
                if lab > 0:
                    matched += 1
                    continue
                lab = _lookup(
                    s.get("centroid_z", 0),
                    s.get("centroid_y", 0),
                    s.get("centroid_x", 0),
                )
                if lab <= 0:
                    # Bbox center fallback
                    lab = _lookup(
                        (float(s.get("z_min", 0)) + float(s.get("z_max", 0))) * 0.5,
                        (float(s.get("y_min", 0)) + float(s.get("y_max", 0))) * 0.5,
                        (float(s.get("x_min", 0)) + float(s.get("x_max", 0))) * 0.5,
                    )
                if lab <= 0:
                    # Small search around centroid in XY on that Z
                    z0 = int(round(float(s.get("centroid_z", 0))))
                    y0 = int(round(float(s.get("centroid_y", 0))))
                    x0 = int(round(float(s.get("centroid_x", 0))))
                    if 0 <= z0 < Z:
                        for rad in (2, 5, 10):
                            y1, y2 = max(0, y0 - rad), min(Y, y0 + rad + 1)
                            x1, x2 = max(0, x0 - rad), min(X, x0 + rad + 1)
                            patch = labeled[z0, y1:y2, x1:x2]
                            vals = patch[patch > 0]
                            if vals.size:
                                # mode
                                lab = int(np.bincount(vals.ravel()).argmax())
                                break
                if lab > 0:
                    s["label"] = lab
                    matched += 1
            print(f"[Viewer] MES label map: {matched}/{len(stats)} objects matched")
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[Viewer] MES label assign failed: {e}")
        return stats

    def _apply_pending_batch_replay_if_any(self):
        """After volume+masks: fill MES + B2B tables and enable table→image map."""
        pending = getattr(self, "_pending_batch_replay", None)
        self._pending_batch_replay = None
        if not pending:
            return

        try:
            # CRITICAL: panel is hidden offline — must show or user sees no tables
            self.show_stats_panel_for_review(True, collapse_tools=True)

            # Clear prior picks / overlays
            try:
                self.clear_object_selection()
            except Exception:
                pass
            self._measurement_start_slice = 0

            mes_stats = list(pending.get("mes_stats") or [])
            b2b_rows = list(pending.get("b2b_rows") or [])
            src = pending.get("source_label") or "Batch Review"

            if mes_stats:
                mes_stats = self._assign_mes_labels_from_mask(mes_stats)
                self.populate_object_stats_table(
                    mes_stats, source_label=src
                )
            else:
                # Keep empty table honest
                self.object_stats = []
                if getattr(self, "object_stats_table", None) is not None:
                    self.object_stats_table.setRowCount(0)
                if hasattr(self, "set_stats_info"):
                    self.set_stats_info(f"No MES objects in DB/CSV · {src}")

            if b2b_rows:
                self.populate_b2b_table(b2b_rows, source_label=src)
            else:
                self.b2b_stats = []
                if getattr(self, "b2b_table", None) is not None:
                    self.b2b_table.setRowCount(0)
                if getattr(self, "b2b_info_label", None) is not None:
                    self.b2b_info_label.setText(
                        f"No B2B CSV on disk · {src}"
                    )

            # Refresh MPR so grid labels / overlays use new stats
            if self.volume_data is not None:
                for ori in ("axial", "coronal", "sagittal"):
                    try:
                        self.render_slice(ori, preserve_camera=True)
                    except Exception:
                        pass

            n_mes = len(mes_stats)
            n_b2b = len(b2b_rows)
            print(
                f"[Viewer] Batch replay ready · MES={n_mes} B2B={n_b2b} · {src}"
            )
            # Focus STATISTICS panel on MES
            if hasattr(self, "stats_tabs"):
                try:
                    self.stats_tabs.setCurrentIndex(0)
                except Exception:
                    pass
            # Second rebalance after rows populate (table needs width)
            self.show_stats_panel_for_review(True, collapse_tools=True)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[Viewer] batch replay apply failed: {e}")

    def _apply_pending_masks_if_any(self):
        """Load offline mask TIFFs queued by load_volume_from_path."""
        bump_p = getattr(self, "_pending_mask_bump", None) or ""
        void_p = getattr(self, "_pending_mask_void", None) or ""
        self._pending_mask_bump = ""
        self._pending_mask_void = ""
        if not bump_p and not void_p:
            # Still apply MES/B2B even without masks (navigate-by-centroid works)
            try:
                self._apply_pending_batch_replay_if_any()
            except Exception as e:
                print(f"[Viewer] batch replay (no mask): {e}")
            return
        if self.volume_data is None:
            return
        try:
            import tifffile
            import numpy as np

            def _load_mask(path):
                if not path or not os.path.isfile(path):
                    return None
                m = np.asarray(tifffile.imread(path))
                if m.ndim == 2:
                    m = m[np.newaxis]
                if m.ndim == 4:
                    m = m[..., 0] if m.shape[-1] <= 4 else m[:, 0]
                # Align to volume
                out = np.zeros(self.volume_data.shape[:3], dtype=np.uint8)
                z = min(m.shape[0], out.shape[0])
                y = min(m.shape[1], out.shape[1])
                x = min(m.shape[2], out.shape[2])
                out[:z, :y, :x] = (m[:z, :y, :x] > 0).astype(np.uint8)
                return out

            c1 = _load_mask(bump_p)
            c2 = _load_mask(void_p)
            if c1 is not None:
                self.class1_data = c1
            if c2 is not None:
                self.class2_data = c2
            if c1 is not None or c2 is not None:
                if hasattr(self, "merge_class_masks"):
                    self.merge_class_masks()
                # Enable overlays
                for orientation in ("axial", "coronal", "sagittal"):
                    c1_check = getattr(self, f"{orientation}_overlay_c1", None)
                    if c1_check is not None and c1 is not None:
                        c1_check.blockSignals(True)
                        c1_check.setChecked(True)
                        c1_check.blockSignals(False)
                    c2_check = getattr(self, f"{orientation}_overlay_c2", None)
                    if c2_check is not None and c2 is not None:
                        c2_check.blockSignals(True)
                        c2_check.setChecked(True)
                        c2_check.blockSignals(False)
                if hasattr(self, "update_all_views"):
                    self.update_all_views()
                elif hasattr(self, "render_slice"):
                    for o in ("axial", "coronal", "sagittal"):
                        try:
                            self.render_slice(o)
                        except Exception:
                            pass
        except Exception as e:
            print(f"[Viewer] pending mask load failed: {e}")
        finally:
            # Full parity: MES + B2B after masks (label map needs class1)
            try:
                self._apply_pending_batch_replay_if_any()
            except Exception as e:
                print(f"[Viewer] batch replay after mask: {e}")
            
    def _handle_volume_load(self, file_path):
        from pathlib import Path
        path = Path(file_path)
        raw_shape = None
        raw_dtype = None
        raw_offset = 0
        
        if path.suffix.lower() in ['.raw', '.bin']:
            from inno3d.core.view_support import RawImportDialog
            dlg = RawImportDialog(path.name, self)
            if dlg.exec_():
                raw_shape, raw_dtype, raw_offset = dlg.get_data()
            else:
                return # user cancelled
                
        from inno3d.core.view_support import LoadVolumeThread
        from inno3d.core.volume_store import large_volume_engine_enabled

        self.pending_load_path = file_path
        use_lve = large_volume_engine_enabled(self)
        self.load_thread = LoadVolumeThread(
            file_path,
            downsample_factor=1,
            raw_shape=raw_shape,
            raw_dtype=raw_dtype,
            raw_offset=raw_offset,
            use_large_volume_engine=use_lve,
        )
        
        self.progress = QProgressDialog("Loading volume...", "Cancel", 0, 100, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.setAutoClose(True)
        self.progress.setMinimumWidth(400)
        self.progress.setStyleSheet(f"""
            QProgressDialog {{
                background: {SemiconductorTheme.BG_PANEL};
                color: {SemiconductorTheme.TEXT_PRIMARY};
            }}
            QProgressBar {{
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 5px;
                background: {SemiconductorTheme.BG_DARK};
                height: 18px;
                text-align: center;
                color: {SemiconductorTheme.TEXT_PRIMARY};
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {SemiconductorTheme.ACCENT_PRIMARY},
                    stop:1 #00b4d8);
                border-radius: 4px;
            }}
            QLabel {{ color: {SemiconductorTheme.TEXT_PRIMARY}; }}
            QPushButton {{
                background: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 4px;
                padding: 4px 16px;
            }}
            QPushButton:hover {{ background: {SemiconductorTheme.ACCENT_PRIMARY}; }}
        """)
        self.progress.show()
        
        # Smooth progress animation: interpolates between target values
        self._progress_target = 0
        self._progress_current = 0.0
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(30)  # ~33 fps smooth animation
        self._progress_timer.timeout.connect(self._animate_progress)
        self._progress_timer.start()
        
        self.load_thread.progress.connect(self._update_progress_target)
        self.load_thread.finished.connect(self.on_volume_loaded)
        self.load_thread.start()
    
    def _update_progress_target(self, value, message):
        """Receive target progress from loading thread — animation will smoothly catch up."""
        self._progress_target = value
        if hasattr(self, 'progress') and self.progress is not None:
            self.progress.setLabelText(message)
            if self.progress.wasCanceled():
                if hasattr(self, 'load_thread') and self.load_thread is not None:
                    self.load_thread.terminate()
    
    def _animate_progress(self):
        """Smooth progress bar animation — runs at 33fps, interpolates toward target."""
        if not hasattr(self, 'progress') or self.progress is None:
            if hasattr(self, '_progress_timer'):
                self._progress_timer.stop()
            return
        
        target = self._progress_target
        current = self._progress_current
        
        # Ease toward target: move 20% of remaining distance per frame
        # This creates a smooth deceleration effect
        if abs(target - current) < 0.5:
            self._progress_current = float(target)
        else:
            self._progress_current = current + (target - current) * 0.2
        
        self.progress.setValue(int(self._progress_current))

    def on_volume_loaded(self, data, error):
        wt = getattr(self, "_online_mpr_walltime", None)
        with perf_phase("viewer.on_volume_loaded.total"):
            with online_walltime_phase(wt, "on_volume_loaded.total"):
                self._on_volume_loaded_body(data, error)

        # Batch Review deep-link: apply mask TIFFs after volume is ready
        try:
            self._apply_pending_masks_if_any()
        except Exception as e:
            print(f"[Viewer] apply pending masks: {e}")

    def _on_volume_loaded_body(self, data, error):
        if hasattr(self, '_progress_timer'):
            self._progress_timer.stop()
        if hasattr(self, 'progress') and self.progress is not None:
            self.progress.setValue(100)  # Ensure bar reaches 100%
            self.progress.close()
        if error:
            # Drop pending Batch Review replay so next load is clean
            self._pending_batch_replay = None
            self._pending_mask_bump = ""
            self._pending_mask_void = ""
            QMessageBox.critical(self, "Error", f"Failed to load volume: {error}")
            return

        # Reset alignment backups and camera init state
        self.original_volume_data = None
        self._camera_initialized = False
        self.original_class1_data = None
        self.original_class2_data = None
        self.original_labeled_class1_data = None
        self.original_labeled_class2_data = None

        if hasattr(self, 'btn_reset_align'):
            self.btn_reset_align.setEnabled(False)
        if hasattr(self, 'reset_align_btn'):
            self.reset_align_btn.setEnabled(False)

        # Invalidate caches from previous volume
        self._cached_slice_actors = {}
        if hasattr(self, '_persistent_crosshair'):
            self._persistent_crosshair = {'axial': {}, 'coronal': {}, 'sagittal': {}}

        # Clear all segmentation/class overlays from the previous volume.
        # A new input volume must always start clean — no stale masks from a
        # different-shaped result should survive into the new session.
        self.segmentation_data = None
        self.class1_data = None
        self.class2_data = None
        self.labeled_class1_data = None
        self.labeled_class2_data = None

        from inno3d.core.volume_store import (
            VolumeArrayProxy,
            VolumeStore,
            large_volume_engine_enabled,
            store_from_dense_array,
            wrap_volume_for_viewer,
        )

        prev_store = getattr(self, "volume_store", None)
        if prev_store is not None and hasattr(prev_store, "close"):
            try:
                prev_store.close()
            except Exception:
                pass
        self.volume_store = None

        wt = getattr(self, "_online_mpr_walltime", None)
        with online_walltime_phase(wt, "store_wrap"):
            if isinstance(data, VolumeStore):
                store, proxy = wrap_volume_for_viewer(data)
                self.volume_store = store
                self.volume_data = proxy
                data = proxy
                print(
                    f"[LVE] VolumeStore attached: {store.shape} {store.dtype} "
                    f"(large_volume_engine on)"
                )
            else:
                # Handle multi-channel images (e.g., RGBA): convert to single channel
                if getattr(data, "ndim", 0) == 4:
                    data = data[:, :, :, 0]
                self.volume_data = data
                # Interim TIFF/folder: still in RAM; expose store API when flag on
                if large_volume_engine_enabled(self) and getattr(data, "ndim", 0) == 3:
                    try:
                        sp = getattr(self, "spacing", None) or (1.0, 1.0, 1.0)
                        self.volume_store = store_from_dense_array(
                            data,
                            spacing=(float(sp[0]), float(sp[1]), float(sp[2])),
                            # Online first-paint latency: avoid forced contiguous copy
                            # and avoid full-volume min/max scan in store wrapping.
                            prefer_no_copy=True,
                            sampled_value_range=True,
                        )
                        print(
                            f"[LVE] Dense VolumeStore wrapper for TIFF/folder "
                            f"{data.shape} (still in RAM; 3D can use coarse mip)"
                        )
                    except Exception as e:
                        print(f"[LVE] dense store wrap skipped: {e}")

        z, y, x = data.shape

        # Initialize crosshair at center
        self.crosshair_position = [x // 2, y // 2, z // 2]

        # Update crosshair position sliders ranges
        for axis, dim, ctr in [('x', x-1, x//2), ('y', y-1, y//2), ('z', z-1, z//2)]:
            sl = getattr(self, f'crosshair_slider_{axis}', None)
            if sl:
                sl.blockSignals(True)
                sl.setMaximum(dim)
                sl.setValue(ctr)
                sl.blockSignals(False)

        # Fast min/max: avoid scanning the entire 32GB array.
        total_voxels = int(getattr(data, "size", z * y * x))
        with perf_phase("viewer.on_volume_loaded.minmax"):
            with online_walltime_phase(wt, "minmax"):
                store = getattr(self, "volume_store", None)
                if store is not None:
                    data_min, data_max = store.metadata.value_range
                elif isinstance(data, VolumeArrayProxy):
                    data_min, data_max = float(data.min()), float(data.max())
                elif total_voxels > 10_000_000:  # > 10M voxels: use subsampling
                    step = max(1, total_voxels // 2_000_000)
                    flat_view = data.ravel()
                    sampled = flat_view[::step]
                    data_min = float(sampled.min())
                    data_max = float(sampled.max())
                else:
                    data_min = float(data.min())
                    data_max = float(data.max())
        window = data_max - data_min
        level = (data_max + data_min) / 2.0

        engine_tag = " [LVE]" if getattr(self, "volume_store", None) is not None else ""
        self.info_label.setText(
            f"Volume: ({z},{y},{x}) [{data_min}-{data_max}]{engine_tag}"
        )

        with online_walltime_phase(wt, "wl_setup"):
            # Update 3D W/L spinboxes + plotted/data range (Dragonfly Window Leveling)
            self._set_data_range_ui(data_min, data_max)

            self.view_3d_min_spin.blockSignals(True)
            self.view_3d_max_spin.blockSignals(True)
            self.view_3d_min_spin.setValue(int(data_min))
            self.view_3d_max_spin.setValue(int(data_max))
            self.view_3d_min_spin.blockSignals(False)
            self.view_3d_max_spin.blockSignals(False)

            # Histogram bars normalised 0..1 → full data range
            if hasattr(self, 'tf_widget'):
                self.tf_widget.set_range(0.0, 1.0)
            self._update_wl_readout()

            # MPR W/L (right sidebar): linked to all 3 planes
            for orientation in ['axial', 'coronal', 'sagittal']:
                self.window_level[orientation] = (window, level)
            self._sync_mpr_wl_ui(int(data_min), int(data_max), data_min, data_max)

        # Reset custom spacing when new volume is loaded
        self.custom_spacing = None
        if hasattr(self, 'spacing_x_spin'):
            self.spacing_x_spin.setValue(self.spacing[0])
            self.spacing_y_spin.setValue(self.spacing[1])
            self.spacing_z_spin.setValue(self.spacing[2])

        for orientation in ['axial', 'coronal', 'sagittal']:
            self.camera_state[orientation] = None

        # Progressive MPR generation counters (large_volume_engine)
        self._mpr_gen = {'axial': 0, 'coronal': 0, 'sagittal': 0}
        self._mpr_applied_gen = {'axial': 0, 'coronal': 0, 'sagittal': 0}
        self._mpr_interacting = False
        self._mpr_refine_level = {'axial': 0, 'coronal': 0, 'sagittal': 0}

        # ── Phase 4a: Adaptive performance tuning for large volumes ──────
        data_gb = float(getattr(data, "nbytes", z * y * x * 2)) / (1024 ** 3)
        if data_gb >= 4.0:
            self._apply_large_volume_perf_settings(data_gb)

        with perf_phase("viewer.on_volume_loaded.update_all_views"):
            with online_walltime_phase(wt, "on_volume_loaded.update_all_views_total"):
                self.update_all_views()

    def _apply_large_volume_perf_settings(self, data_gb):
        """Auto-tune rendering settings for large volumes (≥ 4 GB).

        Called from ``on_volume_loaded`` before ``update_all_views`` so the
        first render already uses the optimised parameters.

        Thresholds (user-requested 4 GB / 8 GB bands):
            4–8 GB:  quality → Standard, 3D render stays ON
            > 8 GB:  quality → Draft; legacy path auto-disables 3D;
                     LVE + volume_store keeps 3D ON for coarse mip upload
                     (plan §5) instead of blank/disabled pane.
        """
        from PyQt5.QtCore import QTimer

        # ── 1. Quality preset auto-switch ────────────────────────────────
        if data_gb >= 8.0:
            target_quality = "Draft"
        else:
            target_quality = "Standard"

        prev_quality = getattr(self, '_current_quality', 'High')
        if hasattr(self, 'quality_combo'):
            idx = self.quality_combo.findText(target_quality)
            if idx >= 0:
                self.quality_combo.blockSignals(True)
                self.quality_combo.setCurrentIndex(idx)
                self.quality_combo.blockSignals(False)
        self._current_quality = target_quality

        # ── 2. 3D volume render: auto-disable only when coarse upload is unavailable ───
        lve_coarse_3d = False
        budgeted_coarse_3d = False
        try:
            from inno3d.core.volume_store import (
                large_volume_engine_enabled,
                default_3d_upload_budget_bytes,
                level_nbytes,
            )

            lve_coarse_3d = (
                large_volume_engine_enabled(self)
                and getattr(self, "volume_store", None) is not None
            )
            budget = int(default_3d_upload_budget_bytes())
            store = getattr(self, "volume_store", None)
            if store is not None:
                full_bytes = int(level_nbytes(store.shape, store.dtype, 0))
                budgeted_coarse_3d = full_bytes > budget
            else:
                vol = getattr(self, "volume_data", None)
                if (
                    vol is not None
                    and getattr(vol, "ndim", 0) == 3
                    and all(int(s) > 0 for s in getattr(vol, "shape", ()))
                ):
                    full_bytes = int(getattr(vol, "nbytes", 0) or 0)
                    budgeted_coarse_3d = full_bytes > budget
        except Exception:
            lve_coarse_3d = False
            budgeted_coarse_3d = False

        if data_gb >= 8.0:
            if lve_coarse_3d or budgeted_coarse_3d:
                # Prefer coarse/budgeted upload over disabling 3D entirely.
                print(
                    f"[PERF] Volume {data_gb:.1f} GB → 3D kept enabled "
                    f"(budgeted coarse upload; not auto-disabled)"
                )
            elif hasattr(self, 'set_3d_volume_render_enabled'):
                self.set_3d_volume_render_enabled(False)
                print(f"[PERF] Volume {data_gb:.1f} GB → 3D volume render "
                      f"auto-disabled (re-enable via 3D toggle button)")

        # ── 3. Slice throttle interval (adaptive) ────────────────────────
        if data_gb >= 8.0:
            throttle_ms = 33   # ~30 fps
        elif data_gb >= 4.0:
            throttle_ms = 25   # ~40 fps
        else:
            throttle_ms = 16   # ~60 fps (default)

        if hasattr(self, '_slice_throttle_timer'):
            self._slice_throttle_timer.setInterval(throttle_ms)

        print(f"[PERF] Volume {data_gb:.1f} GB → quality={target_quality} "
              f"(was {prev_quality}), slice throttle={throttle_ms}ms, "
              f"3D crosshair debounce={'100ms' if data_gb >= 8 else '33ms'}"
              f"{' [LVE coarse 3D]' if lve_coarse_3d else ''}")

    def update_pixel_value(self, orientation, pos):
        """Show volume XYZ + intensity under cursor (match Teaching MPR readout).

        Previous Viewer text only showed 2D image (px,py) — missing Z / full
        voxel address. Coordinates follow ``render_slice`` orientation mapping.
        """
        try:
            if self.volume_data is None:
                return
            widget = getattr(self, f"{orientation}_widget")
            renderer = getattr(self, f"{orientation}_renderer")
            label = getattr(self, f"{orientation}_pixel_label", None)
            if label is None:
                return

            x, y = pos.x(), pos.y()
            size = widget.GetRenderWindow().GetSize()
            vtk_y = size[1] - y

            picker = vtk.vtkWorldPointPicker()
            picker.Pick(x, vtk_y, 0, renderer)
            world_pos = picker.GetPickPosition()

            slice_idx = self.current_slices[orientation]
            vol_z, vol_y, vol_x = self.volume_data.shape
            value = None
            coord_str = ""

            if orientation == "axial":
                # XY flipud: display X=X, VTK Y runs bottom→top along flipped Y index (Dragonfly/Fiji parity)
                px = int(round(world_pos[0]))
                py_disp = int(round(world_pos[1]))
                if 0 <= px < vol_x and 0 <= py_disp < vol_y:
                    py = (vol_y - 1) - py_disp
                    actual_z = (
                        (vol_z - 1 - slice_idx)
                        if getattr(self, "reverse_z", False)
                        else slice_idx
                    )
                    value = self.volume_data[actual_z, py, px]
                    coord_str = f"X:{px} Y:{py} Z:{actual_z}"
            elif orientation == "coronal":
                # XZ flipud: VTK Y runs bottom→top along flipped Z index
                # (un-flip to match Teaching readout and crosshair Z picking)
                px = int(round(world_pos[0]))
                pz_disp = int(round(world_pos[1]))
                if 0 <= px < vol_x and 0 <= pz_disp < vol_z:
                    pz = (vol_z - 1) - pz_disp
                    value = self.volume_data[pz, slice_idx, px]
                    coord_str = f"X:{px} Y:{slice_idx} Z:{pz}"
            else:
                # Sagittal: transpose → VTK X=Z, VTK Y=Y
                pz = int(round(world_pos[0]))
                py = int(round(world_pos[1]))
                if 0 <= pz < vol_z and 0 <= py < vol_y:
                    value = self.volume_data[pz, py, slice_idx]
                    coord_str = f"X:{slice_idx} Y:{py} Z:{pz}"

            if value is not None:
                label.setText(f"{coord_str} | Pixel: {value}")
            else:
                label.setText("Pixel: --")
        except Exception:
            pass
    
    def load_segmentation(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Segmentation TIFF", "", "TIFF Files (*.tif *.tiff)"
        )
        
        if file_path:
            try:
                data = io.imread(file_path)
                if data.ndim == 2:
                    directory = Path(file_path).parent
                    all_files = sorted(glob.glob(str(directory / "*.tif*")))
                    if len(all_files) > 1:
                        data = np.array([io.imread(f) for f in all_files])
                    else:
                        data = data[np.newaxis, :, :]
                
                self.segmentation_data = data
                
                if self.volume_data is not None:
                    for orientation in ['axial', 'coronal', 'sagittal']:
                        self.render_slice(orientation)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
    
    def update_all_views(self):
        wt = getattr(self, "_online_mpr_walltime", None)
        with perf_phase("viewer.update_all_views.total"):
            with online_walltime_phase(wt, "update_all_views.total"):
                if self.volume_data is None:
                    return

                z, y, x = self.volume_data.shape

                self.axial_slice_slider.setMaximum(z - 1)
                self.axial_slice_slider.setValue(z // 2)
                self.current_slices['axial'] = z // 2

                # Coronal(Y): slices along Y axis, so max = y-1
                self.coronal_slice_slider.setMaximum(y - 1)
                self.coronal_slice_slider.setValue(y // 2)
                self.current_slices['coronal'] = y // 2

                # Sagittal(X): slices along X axis, so max = x-1
                self.sagittal_slice_slider.setMaximum(x - 1)
                self.sagittal_slice_slider.setValue(x // 2)
                self.current_slices['sagittal'] = x // 2

                # Init crosshair at center
                self.crosshair_position = [x // 2, y // 2, z // 2]
                for orientation in ['axial', 'coronal', 'sagittal']:
                    with online_walltime_phase(wt, f"render_slice.{orientation}"):
                        self.render_slice(orientation)

                # Online / 3D-off: skip GPU volume path so MPR paints first.
                if hasattr(self, "is_3d_volume_render_enabled") and not self.is_3d_volume_render_enabled():
                    with perf_phase("viewer.update_all_views.3d_placeholder"):
                        try:
                            self._teardown_3d_volume_actors()
                        except Exception:
                            pass
                        try:
                            self._show_3d_disabled_placeholder()
                        except Exception:
                            pass
                        if getattr(self, "crosshair_enabled", False) and hasattr(self, "update_3d_crosshair"):
                            try:
                                self.update_3d_crosshair()
                            except Exception:
                                pass
                else:
                    with perf_phase("viewer.update_all_views.render_3d"):
                        self.render_3d()
                    with perf_phase("viewer.update_all_views.crosshair_3d"):
                        self.update_3d_crosshair()
        
        # Show VIEW TOOLS host when volume is loaded
        # Online → collapsed rail by default (or user session pref); Manual → full sidebar
        if hasattr(self, "align_tools_host"):
            self.align_tools_host.setVisible(True)
            if getattr(self, "_online_viewer_mode", False):
                self.set_align_tools_expanded(
                    getattr(self, "_align_tools_user_expanded", False),
                    remember=False,
                )
            else:
                self.set_align_tools_expanded(True, remember=False)
        elif hasattr(self, "align_sidebar"):
            self.align_sidebar.setVisible(True)

        # Let Online MPR panes paint before the rest of the Online pipeline continues.
        if getattr(self, "_online_viewer_mode", False):
            try:
                from PyQt5.QtWidgets import QApplication
                QApplication.processEvents()
            except Exception:
                pass

        # Force MPR tool strips (C1/C2/FS/Reset) back after Online dialogs hid them
        QTimer.singleShot(50, self._reposition_visible_overlays)
        QTimer.singleShot(200, self._reposition_visible_overlays)
    
    def update_slice(self, orientation, value):
        """Slider moved → update crosshair axis for that plane (full linked sync).

        axial    slider → Z
        coronal  slider → Y
        sagittal slider → X

        Coalesces ultra-fast drag events (~60 fps) but never leaves crosshair
        and sliders on different indices.
        """
        if self.volume_data is None:
            return

        # Progressive MPR: treat slider/wheel as interaction even when
        # sliderPressed is not available (e.g. mouse wheel on slice bar).
        if hasattr(self, "mark_mpr_interaction_start"):
            self.mark_mpr_interaction_start()
            end_timer = getattr(self, "_mpr_slider_end_timer", None)
            if end_timer is None:
                end_timer = QTimer(self)
                end_timer.setSingleShot(True)
                end_timer.setInterval(140)
                end_timer.timeout.connect(self._end_mpr_slider_interaction)
                self._mpr_slider_end_timer = end_timer
            end_timer.start()

        if not hasattr(self, '_slice_update_pending'):
            self._slice_update_pending = {}
            self._slice_throttle_timer = QTimer(self)
            self._slice_throttle_timer.setSingleShot(True)
            self._slice_throttle_timer.setInterval(16)
            self._slice_throttle_timer.timeout.connect(self._flush_slice_update)

        self._slice_update_pending[orientation] = int(value)

        # First event: apply immediately so UI feels locked to the thumb
        if not self._slice_throttle_timer.isActive():
            self._flush_slice_update()
            # Coalesce any events that arrive in the next frame
            self._slice_throttle_timer.start()

    def _end_mpr_slider_interaction(self):
        """Idle after last slider/wheel event → refine MPR to level-0."""
        # Still holding the thumb: keep preview until sliderReleased.
        for ori in ("axial", "coronal", "sagittal"):
            sl = getattr(self, f"{ori}_slice_slider", None)
            if sl is not None and hasattr(sl, "isSliderDown") and sl.isSliderDown():
                end_timer = getattr(self, "_mpr_slider_end_timer", None)
                if end_timer is not None:
                    end_timer.start()
                return
        if hasattr(self, "mark_mpr_interaction_end"):
            self.mark_mpr_interaction_end()
    
    def _flush_slice_update(self):
        """Apply latest pending slider values through updatePoint (single source of truth)."""
        if not hasattr(self, '_slice_update_pending') or not self._slice_update_pending:
            return

        pending = self._slice_update_pending.copy()
        self._slice_update_pending.clear()

        newX, newY, newZ = self.crosshair_position
        for orientation, value in pending.items():
            if orientation == 'axial':
                newZ = value
            elif orientation == 'coronal':
                newY = value
            elif orientation == 'sagittal':
                newX = value

        self.updatePoint(newX, newY, newZ)

        # If more slider events arrived during render, flush again next tick
        if self._slice_update_pending:
            self._slice_throttle_timer.start()
    
    def save_camera_state(self, orientation):
        renderer = getattr(self, f'{orientation}_renderer')
        camera = renderer.GetActiveCamera()
        
        self.camera_state[orientation] = {
            'position': camera.GetPosition(),
            'focal_point': camera.GetFocalPoint(),
            'view_up': camera.GetViewUp(),
            'parallel_scale': camera.GetParallelScale()
        }
    
    def restore_camera_state(self, orientation):
        if self.camera_state[orientation] is not None:
            renderer = getattr(self, f'{orientation}_renderer')
            camera = renderer.GetActiveCamera()
            
            state = self.camera_state[orientation]
            camera.SetPosition(state['position'])
            camera.SetFocalPoint(state['focal_point'])
            camera.SetViewUp(state['view_up'])
            camera.SetParallelScale(state['parallel_scale'])
            
            renderer.ResetCameraClippingRange()
    
    def _volume_world_center(self):
        """Center of the loaded volume in VTK world coordinates (spacing-aware)."""
        if self.volume_data is None:
            return (0.0, 0.0, 0.0)
        z, y, x = self.volume_data.shape
        sp = self.custom_spacing if getattr(self, 'custom_spacing', None) is not None else self.spacing
        return (
            (x - 1) * float(sp[0]) * 0.5,
            (y - 1) * float(sp[1]) * 0.5,
            (z - 1) * float(sp[2]) * 0.5,
        )

    def _volume_world_radius(self):
        """Half-diagonal of the volume AABB — used for zoom distance clamps."""
        if self.volume_data is None:
            return 100.0
        z, y, x = self.volume_data.shape
        sp = self.custom_spacing if getattr(self, 'custom_spacing', None) is not None else self.spacing
        hx = (x - 1) * float(sp[0]) * 0.5
        hy = (y - 1) * float(sp[1]) * 0.5
        hz = (z - 1) * float(sp[2]) * 0.5
        return max((hx * hx + hy * hy + hz * hz) ** 0.5, 1.0)

    def _volume_world_bounds(self):
        """World AABB of the loaded volume (origin 0 + spacing)."""
        if self.volume_data is None:
            return None
        sp = self.custom_spacing if getattr(self, 'custom_spacing', None) is not None else self.spacing
        return volume_world_aabb(self.volume_data.shape, sp)

    def _sync_3d_orbit_pivot(self):
        """Set Track pivot = volume center (default). Does not re-anchor camera.

        Track rotates around this pivot without snapping focal point on
        left-drag. Later: Pivot Point tool can call SetOrbitPivot with a
        user-picked world point.
        """
        style = getattr(self, '_3d_interactor_style', None)
        if style is None or not hasattr(style, 'SetOrbitPivot'):
            return
        cx, cy, cz = self._volume_world_center()
        style.SetOrbitPivot(
            cx, cy, cz,
            radius=self._volume_world_radius(),
            bounds=self._volume_world_bounds(),
        )

    def _force_3d_render_window_size(self):
        """Push Qt widget pixel size into VTK (needed after fullscreen / sidebar).

        If VTK keeps a stale size, arcball center/radius and pick coords drift
        so Track orbit feels off-axis only in the large pane.
        """
        widget = getattr(self, "view_3d_widget", None)
        if widget is None:
            return
        try:
            rw = widget.GetRenderWindow()
            if rw is None:
                return
            # Prefer device-pixel size when available (HiDPI)
            try:
                dpr = float(widget.devicePixelRatioF())
            except Exception:
                try:
                    dpr = float(widget.devicePixelRatio())
                except Exception:
                    dpr = 1.0
            dpr = max(dpr, 1.0)
            w = max(int(round(widget.width() * dpr)), 1)
            h = max(int(round(widget.height() * dpr)), 1)
            cur = rw.GetSize()
            if int(cur[0]) != w or int(cur[1]) != h:
                rw.SetSize(w, h)
        except Exception:
            pass

    def _reframe_3d_after_viewport_change(self):
        """Keep orbit direction + zoom; re-center volume after layout change.

        Called on enter/exit 3D in-grid fullscreen (sidebar shows/hides and
        aspect ratio jumps). Without this, FP/volume center is no longer at
        the visual center of the new viewport and Track pivot feels “wrong”.
        """
        ren = getattr(self, "view_3d_renderer", None)
        widget = getattr(self, "view_3d_widget", None)
        if ren is None or widget is None or self.volume_data is None:
            return
        try:
            self._force_3d_render_window_size()
            cam = ren.GetActiveCamera()
            if cam is None:
                return
            cx, cy, cz = self._volume_world_center()
            pos = list(cam.GetPosition())
            fp = list(cam.GetFocalPoint())
            dx, dy, dz = pos[0] - fp[0], pos[1] - fp[1], pos[2] - fp[2]
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            if dist < 1e-9:
                # Degenerate — place camera along +Z from volume center
                dist = max(float(self._volume_world_radius()) * 3.0, 1.0)
                dx, dy, dz = 0.0, 0.0, 1.0
            else:
                inv = 1.0 / dist
                dx, dy, dz = dx * inv, dy * inv, dz * inv
            # Re-anchor look-at on volume center (orbit pivot); keep distance
            cam.SetFocalPoint(cx, cy, cz)
            cam.SetPosition(cx + dx * dist, cy + dy * dist, cz + dz * dist)
            self._sync_3d_orbit_pivot()
            ren.ResetCameraClippingRange()
            rw = widget.GetRenderWindow()
            if rw is not None:
                rw.Render()
        except Exception as e:
            print(f"[VIEWER] 3D viewport reframe skipped: {e}")

    def _dragonfly_3d_zoom(self, zoom_in=True, strength=1.0, display_xy=None):
        """ORS Dragonfly object zoom — dolly about orbit pivot (pre-P1 behaviour).

        Scroll wheel and right-drag share this path (host for
        ``Dragonfly3DInteractorStyle``).

        ``display_xy`` is intentionally ignored: volume-render world-under-cursor
        picks are unstable and were shifting FP/position, which made Track / pan
        / zoom feel broken after the P1 experiment.
        """
        ren = getattr(self, 'view_3d_renderer', None)
        widget = getattr(self, 'view_3d_widget', None)
        if ren is None or widget is None or self.volume_data is None:
            return

        try:
            if hasattr(self, '_start_lod_interaction'):
                self._start_lod_interaction()

            apply_dragonfly_volume_zoom(
                ren,
                zoom_in=zoom_in,
                strength=strength,
                volume_center=self._volume_world_center(),
                volume_radius=self._volume_world_radius(),
                volume_bounds=self._volume_world_bounds(),
                display_xy=None,  # restore: no cursor pan on 3D volume
            )

            rw = widget.GetRenderWindow()
            if rw is not None:
                rw.Render()
        except Exception as e:
            print(f">>> 3D zoom error (swallowed): {e}")
        finally:
            if hasattr(self, '_end_lod_interaction'):
                QTimer.singleShot(150, self._end_lod_interaction)

    def reset_3d_view(self):
        """Reset 3D zoom/pan to a tight fit of the volume (keep orbit direction).

        Avoids bare ``ResetCamera()`` which often frames the scene too loose
        (volume becomes a tiny speck) especially after layout changes or when
        only the focal point was re-centered without adjusting distance.
        """
        import math

        if not getattr(self, "volume_actor", None) or self.volume_data is None:
            return
        ren = getattr(self, "view_3d_renderer", None)
        widget = getattr(self, "view_3d_widget", None)
        if ren is None or widget is None:
            return

        cam = ren.GetActiveCamera()
        cx, cy, cz = self._volume_world_center()
        R = max(float(self._volume_world_radius()), 1.0)

        # Keep current viewing direction (reset = zoom/pan, not re-orient)
        pos = cam.GetPosition()
        fp = cam.GetFocalPoint()
        dx, dy, dz = pos[0] - fp[0], pos[1] - fp[1], pos[2] - fp[2]
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        if dist < 1e-9:
            # Fallback: Top view direction
            dx, dy, dz, dist = 0.0, 0.0, 1.0, 1.0
        inv = 1.0 / dist
        dx, dy, dz = dx * inv, dy * inv, dz * inv

        # Volume should fill ~88% of the FOV — readable, not a speck
        fill = 0.88

        if cam.GetParallelProjection():
            # ParallelScale = half-height of the orthographic view in world units
            cam.SetParallelScale(max(R / fill, 1e-6))
            d = max(R * 3.0, 1.0)
        else:
            va = float(cam.GetViewAngle())
            if va < 10.0 or va > 100.0:
                # Match set_projection default when angle is invalid
                va = 40.0
                cam.SetViewAngle(va)
            half_rad = math.radians(va * 0.5)
            # Distance so a sphere of radius R fills `fill` of the vertical FOV
            d = (R / fill) / max(math.tan(half_rad), 1e-6)
            d = max(d, R * 1.15)

        cam.SetFocalPoint(cx, cy, cz)
        cam.SetPosition(cx + dx * d, cy + dy * d, cz + dz * d)
        bounds = self._volume_world_bounds()
        if bounds is not None:
            push_camera_outside_aabb(cam, bounds, pad_frac=0.02)

        # If view-up is nearly collinear with view direction, pick a stable up
        ux, uy, uz = cam.GetViewUp()
        # |view · up| close to 1 → degenerate
        if abs(dx * ux + dy * uy + dz * uz) > 0.95:
            # Prefer Z-up, else Y-up
            if abs(dz) < 0.9:
                cam.SetViewUp(0.0, 0.0, 1.0)
            else:
                cam.SetViewUp(0.0, 1.0, 0.0)

        self._sync_3d_orbit_pivot()
        ren.ResetCameraClippingRange()
        widget.GetRenderWindow().Render()

    def toggle_rotate_mode(self, checked):
        self.rotate_mode = checked
        if not checked:
            self.last_mouse_pos = None

        # Keep button checked state + CSS dynamic property in sync
        btn = getattr(self, 'rotate2d_btn', None)
        if btn is not None:
            if btn.isChecked() != bool(checked):
                btn.blockSignals(True)
                btn.setChecked(bool(checked))
                btn.blockSignals(False)
            btn.setProperty("checked", "true" if checked else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        
        # Auto-disable crosshair when rotate mode is turned on
        if checked and self.crosshair_enabled:
            self.crosshair_enabled = False
            # Update the external crosshair button if accessible
            parent = self.parent()
            while parent is not None:
                if hasattr(parent, 'cross_btn'):
                    parent.cross_btn.setChecked(False)
                    parent.cross_btn.setProperty("checked", "false")
                    parent.cross_btn.style().unpolish(parent.cross_btn)
                    parent.cross_btn.style().polish(parent.cross_btn)
                    break
                parent = parent.parent()
            if self.volume_data is not None:
                for ori in ['axial', 'coronal', 'sagittal']:
                    self.render_slice(ori, preserve_camera=True)
                self.update_3d_crosshair()

    def reset_2d_rotation(self):
        for orientation in ['axial', 'coronal', 'sagittal']:
            renderer = getattr(self, f'{orientation}_renderer')
            camera = renderer.GetActiveCamera()
            
            # Save state
            self.save_camera_state(orientation)
            state = self.camera_state[orientation]
            
            # Reset view-up
            camera.SetViewUp(0, 1, 0)
            
            # Re-save state
            self.save_camera_state(orientation)
            
            widget = getattr(self, f'{orientation}_widget')
            widget.GetRenderWindow().Render()

    def handle_rotation_drag(self, orientation, pos):
        try:
            if not hasattr(self, 'last_mouse_pos') or self.last_mouse_pos is None:
                self.last_mouse_pos = pos
                return
            
            widget = getattr(self, f'{orientation}_widget')
            renderer = getattr(self, f'{orientation}_renderer')
            camera = renderer.GetActiveCamera()
            
            # Viewport center
            size = widget.GetRenderWindow().GetSize()
            cx = size[0] / 2.0
            cy = size[1] / 2.0
            
            # Current and previous mouse positions
            p1 = pos
            p0 = self.last_mouse_pos
            
            # Vectors from center (Qt Y is downwards, so invert y)
            v0_x = p0.x() - cx
            v0_y = cy - p0.y()
            
            v1_x = p1.x() - cx
            v1_y = cy - p1.y()
            
            import math
            # Calculate angles
            a0 = math.atan2(v0_y, v0_x)
            a1 = math.atan2(v1_y, v1_x)
            
            da = a1 - a0
            # Normalize da to [-pi, pi]
            if da > math.pi:
                da -= 2.0 * math.pi
            elif da < -math.pi:
                da += 2.0 * math.pi
                
            da_deg = math.degrees(da)
            
            # Apply Roll to camera
            camera.Roll(da_deg)
            
            # Save camera state so it is preserved
            self.save_camera_state(orientation)
            
            # Render
            widget.GetRenderWindow().Render()
            
            # Save current pos for next drag event
            self.last_mouse_pos = pos
        except Exception as e:
            print("Error in handle_rotation_drag:", e)
    
    def _window_level_to_uint8(self, slice_data, window, level):
        """Map scalar slice through W/L to display uint8 (H,W)."""
        arr = np.asarray(slice_data, dtype=np.float32)
        half = max(float(window), 1e-6) * 0.5
        lo = float(level) - half
        hi = float(level) + half
        if hi <= lo:
            hi = lo + 1.0
        out = (arr - lo) * (255.0 / (hi - lo))
        return np.clip(out, 0, 255).astype(np.uint8)

    def _compose_mpr_overlay_rgb(
        self,
        slice_data,
        window,
        level,
        c1_slice,
        c2_slice,
        seg_slice,
        c1_on,
        c2_on,
        opacity,
        orientation="",
    ):
        """Alpha-blend C1/C2 masks onto W/L-mapped CT → RGB uint8 (H,W,3).

        Background mask pixels keep pure CT gray (no black film).
        C2 overwrites C1 on shared voxels (void-in-bump).
        """
        gray = self._window_level_to_uint8(slice_data, window, level)
        h, w = gray.shape[:2]
        rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
        a = float(np.clip(opacity, 0.0, 1.0))

        c1_color = np.array(
            [float(v * 255) for v in self.seg_colors.get(128, [1.0, 1.0, 0.0])],
            dtype=np.float32,
        )
        c2_color = np.array(
            [float(v * 255) for v in self.seg_colors.get(255, [1.0, 0.0, 0.0])],
            dtype=np.float32,
        )

        def _valid_mask(src, exact=None):
            if src is None:
                return None
            arr = np.asarray(src)
            if arr.ndim > 2:
                arr = arr[..., 0]
            if arr.shape[:2] != (h, w):
                print(
                    f"[VIEWER OVERLAY] mask shape {arr.shape} != {(h, w)} ({orientation}) — skipped"
                )
                return None
            return (arr == exact) if exact is not None else (arr > 0)

        if c1_on:
            if c1_slice is not None:
                m1 = _valid_mask(c1_slice)
            else:
                m1 = _valid_mask(seg_slice, exact=128)
            if m1 is not None and np.any(m1):
                rgb[m1] = (1.0 - a) * rgb[m1] + a * c1_color

        if c2_on:
            if c2_slice is not None:
                m2 = _valid_mask(c2_slice)
            else:
                m2 = _valid_mask(seg_slice, exact=255)
            if m2 is not None and np.any(m2):
                rgb[m2] = (1.0 - a) * rgb[m2] + a * c2_color

        return np.clip(rgb, 0, 255).astype(np.uint8)

    def update_overlay_visibility(self, orientation):
        """Refresh C1/C2 visibility and overlay opacity for one pane (or 3D)."""
        if orientation == '3d':
            if not self.is_3d_volume_render_enabled():
                return
            from inno3d.features.viewer.seg_mask_3d import set_mask_overlay_opacity

            c1_show = self.view_3d_overlay_group.c1_btn.isChecked() if hasattr(self.view_3d_overlay_group, 'c1_btn') else False
            c2_show = self.view_3d_overlay_group.c2_btn.isChecked() if hasattr(self.view_3d_overlay_group, 'c2_btn') else False
            op_val = self.view_3d_overlay_group.opacity_slider.value() / 100.0 if hasattr(self.view_3d_overlay_group, 'opacity_slider') else 0.5

            if getattr(self, 'c1_actor_3d', None):
                self.c1_actor_3d.SetVisibility(c1_show)
                set_mask_overlay_opacity(self.c1_actor_3d, op_val)

            if getattr(self, 'c2_actor_3d', None):
                self.c2_actor_3d.SetVisibility(c2_show)
                # Void slightly stronger so it reads inside bump shells
                set_mask_overlay_opacity(self.c2_actor_3d, min(1.0, float(op_val) * 1.15))

            self.view_3d_widget.GetRenderWindow().Render()
            return

        # Invalidate cached actors so toggle/opacity cannot leave a stale overlay film
        if hasattr(self, '_cached_slice_actors') and orientation in self._cached_slice_actors:
            try:
                del self._cached_slice_actors[orientation]
            except Exception:
                self._cached_slice_actors[orientation] = None
        self.render_slice(orientation, preserve_camera=True)
    
    def _refresh_all_index_views(self):
        """Refresh all three views when Show Index checkbox changes."""
        for ori in ['axial', 'coronal', 'sagittal']:
            self.render_slice(ori, preserve_camera=True)

    def draw_object_labels(self, renderer, actual_z, orientation='axial', slice_idx=0):
        """Draw bump index labels on MPR views (Index checkbox).

        UX rules (avoid left-edge pile-up of R# + Bump (R,C)):
          · **One identity per bump**: ``(R,C)`` matching MES Bump ID — no extra R#/C#
            edge headers that collide with left/top dies.
          · Labels sit on the **centroid** with a dark plate + cyan text.
          · Selected MES rows highlight that label in amber.
          · Dense FOVs use smaller scale automatically.
        """
        if not self.object_stats:
            return

        # Determine z-offset strategy
        is_multi_layer = getattr(self, 'input_mode', 'single') == 'multi_layer'
        ml_z_starts = getattr(self, '_ml_layer_z_starts', {})
        single_z_offset = getattr(self, '_measurement_start_slice', 0) if not is_multi_layer else 0

        # Collect visible stats for this orientation/slice
        visible_stats = []
        for stat in self.object_stats:
            # Multi-layer filter: only filter when a specific layer is selected
            if is_multi_layer and self.current_display_layer is not None:
                if stat.get('layer_name') != self.current_display_layer:
                    continue

            if orientation == 'axial':
                # Compute the per-stat z offset
                if is_multi_layer and self.current_display_layer is None:
                    # "All Layers" mode: volume is combined stack, use per-layer z_start
                    layer_z_start = ml_z_starts.get(stat.get('layer_name', ''), 0)
                else:
                    # Specific layer or single mode
                    layer_z_start = 0
                local_z = actual_z - single_z_offset - layer_z_start
                if stat['z_min'] <= local_z < stat['z_max']:
                    visible_stats.append((stat, layer_z_start))
            elif orientation == 'coronal':
                # Coronal slices along Y axis
                if stat['y_min'] <= slice_idx < stat['y_max']:
                    if is_multi_layer and self.current_display_layer is None:
                        layer_z_start = ml_z_starts.get(stat.get('layer_name', ''), 0)
                    else:
                        layer_z_start = single_z_offset
                    visible_stats.append((stat, layer_z_start))
            elif orientation == 'sagittal':
                # Sagittal slices along X axis
                if stat['x_min'] <= slice_idx < stat['x_max']:
                    if is_multi_layer and self.current_display_layer is None:
                        layer_z_start = ml_z_starts.get(stat.get('layer_name', ''), 0)
                    else:
                        layer_z_start = single_z_offset
                    visible_stats.append((stat, layer_z_start))

        if not visible_stats:
            return

        # Adaptive scale by density
        n_bumps = len(visible_stats)
        if n_bumps > 500:
            label_scale, font_size = 0.16, 13
        elif n_bumps > 200:
            label_scale, font_size = 0.20, 14
        elif n_bumps > 50:
            label_scale, font_size = 0.24, 15
        else:
            label_scale, font_size = 0.28, 17

        stats_only = [s for s, _ in visible_stats]
        has_grid = all(
            s.get("grid_row") is not None and s.get("grid_col") is not None
            for s in stats_only
        )
        selected_labels = {
            int(lab)
            for _c, lab in (getattr(self, "selected_highlight_objects", None) or [])
            if lab
        }

        if orientation == 'axial':
            self._draw_axial_labels(
                renderer, visible_stats, has_grid, label_scale, font_size, selected_labels
            )
        elif orientation == 'coronal':
            self._draw_coronal_labels(
                renderer, visible_stats, has_grid, label_scale, font_size,
                slice_idx, selected_labels,
            )
        elif orientation == 'sagittal':
            self._draw_sagittal_labels(
                renderer, visible_stats, has_grid, label_scale, font_size,
                slice_idx, selected_labels,
            )

    @staticmethod
    def _style_index_text_prop(tprop, font_size, color, selected=False):
        """Readable index plate: bold + dark background (no extra R# clutter)."""
        tprop.SetFontSize(int(font_size))
        tprop.SetBold(True)
        tprop.ShadowOn()
        tprop.SetShadowOffset(1, -1)
        if selected:
            tprop.SetColor(1.0, 0.92, 0.25)  # amber when MES-selected
            tprop.SetBackgroundColor(0.05, 0.05, 0.0)
            tprop.SetBackgroundOpacity(0.72)
            frame = (0.95, 0.75, 0.15)
        else:
            tprop.SetColor(float(color[0]), float(color[1]), float(color[2]))
            tprop.SetBackgroundColor(0.02, 0.04, 0.08)
            tprop.SetBackgroundOpacity(0.62)
            frame = (0.15, 0.55, 0.65)
        try:
            tprop.SetJustificationToCentered()
            tprop.SetVerticalJustificationToCentered()
        except Exception:
            pass
        try:
            tprop.FrameOn()
            tprop.SetFrameColor(*frame)
            tprop.SetFrameWidth(1)
        except Exception:
            pass

    def _index_label_text(self, stat, has_grid):
        """Single identity string: Bump (R,C) or MES # — never both + R# header."""
        if has_grid:
            return f"{stat['grid_row']},{stat['grid_col']}"
        return str(stat.get("row_id", stat.get("label", "")))

    def _draw_axial_labels(
        self, renderer, visible_stats, has_grid, label_scale, font_size, selected_labels=None
    ):
        """XY: one (R,C) plate per bump at centroid — no R#/C# edge rails."""
        selected_labels = selected_labels or set()
        seen = set()
        for stat, _z_off in visible_stats:
            if has_grid:
                key = (stat["grid_row"], stat["grid_col"])
                if key in seen:
                    continue
                seen.add(key)
            try:
                lab = int(stat.get("label", 0))
            except (TypeError, ValueError):
                lab = 0
            is_sel = lab in selected_labels
            text = self._index_label_text(stat, has_grid)
            if not text:
                continue
            cx = float(stat["centroid_x"])
            cy = float(stat["centroid_y"])
            # Slight inset from geometric edge so labels on rim bumps stay on the pad
            caption = vtk.vtkTextActor3D()
            caption.SetInput(text)
            caption.SetPosition(cx, cy, 0.55)
            sc = label_scale * (1.15 if is_sel else 1.0)
            caption.SetScale(sc, sc, sc)
            self._style_index_text_prop(
                caption.GetTextProperty(),
                font_size + (2 if is_sel else 0),
                (0.25, 0.95, 1.0),
                selected=is_sel,
            )
            renderer.AddActor(caption)

    def _draw_coronal_labels(
        self, renderer, visible_stats, has_grid, label_scale, font_size, slice_y,
        selected_labels=None,
    ):
        """XZ: same (R,C) identity — no duplicate C# header row."""
        selected_labels = selected_labels or set()
        vol_z = self.volume_data.shape[0] if self.volume_data is not None else 1
        seen = set()
        for stat, z_off in visible_stats:
            if has_grid:
                key = (stat["grid_row"], stat["grid_col"])
                if key in seen:
                    continue
                seen.add(key)
            try:
                lab = int(stat.get("label", 0))
            except (TypeError, ValueError):
                lab = 0
            is_sel = lab in selected_labels
            text = self._index_label_text(stat, has_grid)
            if not text:
                continue
            cx = float(stat["centroid_x"])
            global_z = float(stat["centroid_z"]) + z_off
            cz_display = (vol_z - 1) - global_z
            caption = vtk.vtkTextActor3D()
            caption.SetInput(text)
            caption.SetPosition(cx, cz_display, 0.55)
            sc = label_scale * (1.15 if is_sel else 1.0)
            caption.SetScale(sc, sc, sc)
            self._style_index_text_prop(
                caption.GetTextProperty(),
                font_size + (2 if is_sel else 0),
                (0.4, 1.0, 0.55),
                selected=is_sel,
            )
            renderer.AddActor(caption)

    def _draw_sagittal_labels(
        self, renderer, visible_stats, has_grid, label_scale, font_size, slice_x,
        selected_labels=None,
    ):
        """YZ: same (R,C) identity — no duplicate R# rail."""
        selected_labels = selected_labels or set()
        seen = set()
        for stat, z_off in visible_stats:
            if has_grid:
                key = (stat["grid_row"], stat["grid_col"])
                if key in seen:
                    continue
                seen.add(key)
            try:
                lab = int(stat.get("label", 0))
            except (TypeError, ValueError):
                lab = 0
            is_sel = lab in selected_labels
            text = self._index_label_text(stat, has_grid)
            if not text:
                continue
            # Sagittal: horizontal = Z, vertical = Y (after view transform)
            cz = float(stat["centroid_z"]) + z_off
            cy = float(stat["centroid_y"])
            caption = vtk.vtkTextActor3D()
            caption.SetInput(text)
            caption.SetPosition(cz, cy, 0.55)
            sc = label_scale * (1.15 if is_sel else 1.0)
            caption.SetScale(sc, sc, sc)
            self._style_index_text_prop(
                caption.GetTextProperty(),
                font_size + (2 if is_sel else 0),
                (1.0, 0.6, 0.35),
                selected=is_sel,
            )
            renderer.AddActor(caption)

    def _apply_3d_mask_overlay_colors(self):
        """Push current ``seg_colors`` into existing 3D C1/C2 mask actors.

        Works for medical surface shells and legacy volume actors.
        Avoids a full ``render_3d()`` rebuild when only the colour changes.
        """
        if not hasattr(self, "is_3d_volume_render_enabled"):
            return
        if not self.is_3d_volume_render_enabled():
            return

        from inno3d.features.viewer.seg_mask_3d import set_mask_overlay_color

        updated = False
        for key, actor_name in ((128, "c1_actor_3d"), (255, "c2_actor_3d")):
            actor = getattr(self, actor_name, None)
            if actor is None:
                continue
            try:
                rgb = self.seg_colors.get(
                    key, [0.0, 1.0, 0.0] if key == 128 else [1.0, 0.0, 0.0]
                )
                set_mask_overlay_color(actor, rgb)
                updated = True
            except Exception as e:
                print(f"[VIEWER] 3D mask color update failed ({actor_name}): {e}")

        if updated:
            widget = getattr(self, "view_3d_widget", None)
            if widget is not None:
                try:
                    widget.GetRenderWindow().Render()
                except Exception:
                    pass

    def choose_overlay_color(self, class_num):
        """Open color dialog to pick class overlay color (MPR + 3D volume)."""
        from PyQt5.QtWidgets import QColorDialog

        key = 128 if class_num == 1 else 255
        current = self.seg_colors.get(key, [0.0, 1.0, 0.0])
        init_color = QColor(int(current[0]*255), int(current[1]*255), int(current[2]*255))
        color = QColorDialog.getColor(init_color, self, f"Choose Class {class_num} Color")
        if not color.isValid():
            return

        self.seg_colors[key] = [color.redF(), color.greenF(), color.blueF()]

        # Preview on sidebar color button (optional — may be icon-only)
        btn = getattr(self, "color_c1_btn" if class_num == 1 else "color_c2_btn", None)
        if btn is not None:
            try:
                btn.setStyleSheet(
                    f"background-color: {color.name()}; "
                    f"color: {'black' if color.lightness() > 128 else 'white'};"
                )
            except Exception:
                pass

        # Invalidate MPR actor cache so baked RGB overlays are rebuilt
        if hasattr(self, "_cached_slice_actors") and self._cached_slice_actors:
            try:
                self._cached_slice_actors.clear()
            except Exception:
                self._cached_slice_actors = {}

        # Re-render MPR (uses updated seg_colors in _compose_mpr_overlay_rgb)
        if self.volume_data is not None:
            for ori in ("axial", "coronal", "sagittal"):
                try:
                    self.render_slice(ori, preserve_camera=True)
                except Exception as e:
                    print(f"[VIEWER] render_slice after color change ({ori}): {e}")

        # Live-update 3D mask volumes if present
        try:
            self._apply_3d_mask_overlay_colors()
        except Exception as e:
            print(f"[VIEWER] 3D color apply after pick: {e}")

    def choose_ruler_color(self):
        """Pick ruler color"""
        from PyQt5.QtWidgets import QColorDialog

        init_c = QColor(int(self.ruler_color[0]*255), int(self.ruler_color[1]*255), int(self.ruler_color[2]*255))
        color = QColorDialog.getColor(init_c, self, "Ruler Color")
        if color.isValid():
            self.ruler_color = [color.redF(), color.greenF(), color.blueF()]
            self.ruler_color_btn.setStyleSheet(f"background-color: {SemiconductorTheme.BG_LIGHT}; color: {color.name()}; font-weight: bold; border: 1px solid {SemiconductorTheme.BORDER_DEFAULT}; border-radius: 3px;")
            for ori in ['axial', 'coronal', 'sagittal']:
                self.render_slice(ori, preserve_camera=True)

    def load_class_mask(self, class_num):
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select Class {class_num} Binary Mask", "", "TIFF Files (*.tif *.tiff)"
        )
        if file_path:
            try:
                data = io.imread(file_path)
                if data.ndim == 2:
                    directory = Path(file_path).parent
                    all_files = sorted(glob.glob(str(directory / "*.tif*")))
                    if len(all_files) > 1:
                        data = np.array([io.imread(f) for f in all_files])
                    else:
                        data = data[np.newaxis, :, :]
                
                # --- Handle shape mismatch (per-layer sub-volume) ---
                if self.volume_data is not None and data.shape != self.volume_data.shape:
                    vol_z, vol_y, vol_x = self.volume_data.shape
                    mask_z, mask_y, mask_x = data.shape

                    # XY must match
                    if mask_y != vol_y or mask_x != vol_x:
                        QMessageBox.critical(self, "Shape Error",
                            f"Mask XY dimensions ({mask_y}Ã—{mask_x}) do not match "
                            f"volume XY ({vol_y}Ã—{vol_x}).\n\nCannot load this mask.")
                        return

                    # Z is smaller â†’ need layer offset from config
                    if mask_z < vol_z:
                        reply = QMessageBox.question(self, "Layer Mask Detected",
                            f"Mask Z={mask_z} < Volume Z={vol_z}.\n\n"
                            "This appears to be a per-layer mask.\n"
                            "Load a config file to get the layer Z offset?\n\n"
                            "(The config must contain LAYER definitions with z_start values.)",
                            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                        
                        if reply == QMessageBox.Yes:
                            z_start = self._get_layer_offset_from_config(mask_z)
                            if z_start is None:
                                return  # User cancelled or error
                        else:
                            # Fallback: place at start
                            z_start = 0
                        
                        # Create full-size mask and place sub-volume at correct Z
                        full_mask = np.zeros(self.volume_data.shape, dtype=data.dtype)
                        z_end = min(z_start + mask_z, vol_z)
                        actual_len = z_end - z_start
                        full_mask[z_start:z_end] = data[:actual_len]
                        data = full_mask
                        
                        QMessageBox.information(self, "Mask Placed",
                            f"Layer mask placed at Z=[{z_start}:{z_end}] "
                            f"within full volume Z=[0:{vol_z}].")

                    elif mask_z > vol_z:
                        # Mask is larger â€” crop to volume size
                        data = data[:vol_z]
                
                if class_num == 1:
                    self.class1_data = data
                else:
                    self.class2_data = data
                
                self.merge_class_masks()
                
                if self.volume_data is not None:
                    for orientation in ['axial', 'coronal', 'sagittal']:
                        self.render_slice(orientation)

                # Rebuild 3D overlay so glass shells appear in the 3D pane
                if hasattr(self, 'render_3d'):
                    try:
                        self.render_3d()
                    except Exception as e:
                        print(f"[MaskC{class_num}] 3D overlay rebuild failed: {e}")

                self.update_info_label()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load mask: {str(e)}")

    def load_mask_folder(self):
        """Load all per-layer masks from a sample output folder.
        
        Expects structure:
            <folder>/Layer_1/*_bump3D.tif   â†’ Class 1 (Bump)
            <folder>/Layer_1/*_voidsOnly.tif â†’ Class 2 (Void)
            ...
        Also requires a config file with LAYER_N definitions for Z offsets.
        """
        if self.volume_data is None:
            QMessageBox.warning(self, "No Volume", "Please load a volume first.")
            return
        
        # 1. Select mask folder
        folder = QFileDialog.getExistingDirectory(
            self, "Select Sample Output Folder (contains Layer_N subfolders)",
            "", QFileDialog.ShowDirsOnly)
        if not folder:
            return
        
        # 2. Discover layer subfolders
        import re
        layer_dirs = []
        for entry in os.listdir(folder):
            sub = os.path.join(folder, entry)
            if os.path.isdir(sub) and re.match(r'Layer_\d+', entry):
                layer_dirs.append((entry, sub))
        
        if not layer_dirs:
            QMessageBox.warning(self, "No Layers",
                f"No Layer_N subfolders found in:\n{folder}")
            return
        
        # Natural sort
        layer_dirs.sort(key=lambda x: [int(c) if c.isdigit() else c.lower() 
                                        for c in re.split(r'(\d+)', x[0])])
        
        # 3. Get layer definitions from 3D Teaching tab config (if loaded),
        #    otherwise fall back to manual config file selection.
        layer_config = {}  # "Layer 1" → (z_start, z_end)
        config_source = None  # for the report message

        # Try to find layer_definitions from SegmentationTab via MainWindow
        seg_tab = None
        try:
            from inno3d.app.main_window import MainWindow
            w = self.window()
            if isinstance(w, MainWindow):
                seg_tab = getattr(w, 'segmentation_tab', None)
        except Exception:
            pass

        seg_layers = getattr(seg_tab, 'layer_definitions', None) if seg_tab else None
        seg_config_path = getattr(seg_tab, 'config_path', None) if seg_tab else None

        if seg_layers and seg_config_path:
            # Use layer definitions already loaded in 3D Teaching tab
            config_source = seg_config_path
            for ld in seg_layers:
                name = ld.get('name', '')
                z_start = ld.get('z_start', 0)
                z_end = ld.get('z_end', 0)
                layer_config[name] = (z_start, z_end)
                layer_config[name.replace(' ', '_')] = (z_start, z_end)
            print(f"[MaskFolder] Using {len(seg_layers)} layer definitions "
                  f"from 3D Teaching config: {config_source}")
        else:
            # Fallback: ask user to select a config file manually
            config_path, _ = QFileDialog.getOpenFileName(
                self, "Select Config File (for layer Z offsets)",
                os.path.dirname(folder),
                "Config Files (*.txt *.cfg *.ini);;All Files (*.*)")
            if not config_path:
                return
            config_source = config_path

            # 4. Parse layer definitions from config (UTF-8 first — not locale cp949)
            try:
                for line in read_text_auto(config_path).splitlines():
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    match = re.match(r'LAYER_\d+\s*=\s*(.+)', line)
                    if match:
                        parts = match.group(1).rsplit(',', 3)
                        if len(parts) >= 3:
                            try:
                                name = parts[0].strip()
                                z_start = int(parts[1].strip())
                                z_end = int(parts[2].strip())
                                # Also store with underscore variant: "Layer 1" → "Layer_1"
                                layer_config[name] = (z_start, z_end)
                                layer_config[name.replace(' ', '_')] = (z_start, z_end)
                            except ValueError:
                                continue
            except Exception as e:
                QMessageBox.critical(self, "Config Error", f"Failed to parse config:\n{str(e)}")
                return
        
        if not layer_config:
            QMessageBox.warning(self, "No Layers in Config",
                "No LAYER_N definitions found.\n\n"
                "Please load a config file in the 3D Teaching tab first,\n"
                "or select a config file manually.")
            return
        
        # 5. Build full-volume masks
        vol_z, vol_y, vol_x = self.volume_data.shape
        full_bump = np.zeros((vol_z, vol_y, vol_x), dtype=np.uint8)
        full_void = np.zeros((vol_z, vol_y, vol_x), dtype=np.uint8)
        
        progress = QProgressDialog("Loading per-layer masks...", "Cancel", 0, len(layer_dirs), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        
        loaded_count = 0
        errors = []
        
        for idx, (dir_name, dir_path) in enumerate(layer_dirs):
            if progress.wasCanceled():
                break
            progress.setValue(idx)
            progress.setLabelText(f"Loading {dir_name}...")
            QApplication.processEvents()
            
            # Match dir name to config: "Layer_1" â†’ try "Layer_1" and "Layer 1"
            z_info = layer_config.get(dir_name) or layer_config.get(dir_name.replace('_', ' '))
            if z_info is None:
                errors.append(f"{dir_name}: no matching layer in config")
                continue
            
            z_start, z_end = z_info
            
            # Find bump3D and voidsOnly TIFs
            bump_file = None
            void_file = None
            for fname in os.listdir(dir_path):
                fl = fname.lower()
                if fl.endswith('.tif') or fl.endswith('.tiff'):
                    if 'bump3d' in fl:
                        bump_file = os.path.join(dir_path, fname)
                    elif 'voidsonly' in fl and 'flatten' not in fl:
                        void_file = os.path.join(dir_path, fname)
            
            try:
                if bump_file:
                    bump_data = io.imread(bump_file)
                    mask_z = bump_data.shape[0] if bump_data.ndim == 3 else 1
                    layer_len = min(mask_z, z_end - z_start, vol_z - z_start)
                    if bump_data.ndim == 3:
                        full_bump[z_start:z_start + layer_len] = np.maximum(
                            full_bump[z_start:z_start + layer_len],
                            bump_data[:layer_len])
                    
                if void_file:
                    void_data = io.imread(void_file)
                    mask_z = void_data.shape[0] if void_data.ndim == 3 else 1
                    layer_len = min(mask_z, z_end - z_start, vol_z - z_start)
                    if void_data.ndim == 3:
                        full_void[z_start:z_start + layer_len] = np.maximum(
                            full_void[z_start:z_start + layer_len],
                            void_data[:layer_len])
                
                loaded_count += 1
                
            except Exception as e:
                errors.append(f"{dir_name}: {str(e)}")
        
        progress.setValue(len(layer_dirs))
        progress.close()
        
        # 6. Assign to viewer
        self.class1_data = full_bump
        self.class2_data = full_void
        self.merge_class_masks()
        
        for orientation in ['axial', 'coronal', 'sagittal']:
            self.render_slice(orientation)

        # Rebuild 3D overlay so bump/void glass shells appear in the 3D pane
        if hasattr(self, 'render_3d'):
            try:
                self.render_3d()
            except Exception as e:
                print(f"[MaskFolder] 3D overlay rebuild failed: {e}")

        self.update_info_label()
        
        # Report
        cfg_basename = os.path.basename(config_source) if config_source else "unknown"
        msg = f"Loaded {loaded_count}/{len(layer_dirs)} layers successfully.\n"
        msg += f"Config: {cfg_basename}"
        if errors:
            msg += f"\n\nWarnings:\n" + "\n".join(errors[:10])
        QMessageBox.information(self, "Mask Folder Loaded", msg)

    def _get_layer_offset_from_config(self, mask_z: int):
        """Parse a config file and let user pick which layer matches the mask Z size.
        
        Config format: LAYER_N = LayerName,z_start,z_end,selected
        Example:       LAYER_0 = Layer 1,666,831,true
        
        Returns the z_start offset for the selected layer, or None if cancelled.
        """
        import re
        
        config_path, _ = QFileDialog.getOpenFileName(
            self, "Select Config File with Layer Definitions", "",
            "Config Files (*.txt *.cfg *.ini);;All Files (*.*)")
        if not config_path:
            return None
        
        try:
            layers = []
            for line in read_text_auto(config_path).splitlines():
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                # Match: LAYER_0 = Layer 1,666,831,true
                match = re.match(r'LAYER_\d+\s*=\s*(.+)', line)
                if match:
                    parts = match.group(1).rsplit(',', 3)  # split from right: name may contain commas
                    if len(parts) >= 3:
                        # parts: ['Layer 1', '666', '831', 'true'] or ['Layer 1', '666', '831']
                        selected_str = parts[3].strip().lower() if len(parts) >= 4 else 'true'
                        try:
                            name = parts[0].strip()
                            z_start = int(parts[1].strip())
                            z_end = int(parts[2].strip())
                            z_size = z_end - z_start
                            layers.append((name, z_start, z_end, z_size))
                        except ValueError:
                            continue
            
            if not layers:
                QMessageBox.warning(self, "No Layers",
                    "No LAYER definitions found in config file.\n"
                    "Expected format: LAYER_N = LayerName,z_start,z_end,selected")
                return None
            
            # Filter layers that match the mask Z size
            matching = [(name, z_start, z_end, z_size) 
                       for name, z_start, z_end, z_size in layers if z_size == mask_z]
            
            if len(matching) == 1:
                # Exact single match â€” use it directly
                return matching[0][1]
            
            # Multiple matches or no exact match â€” let user pick
            if not matching:
                matching = layers  # Show all layers if none match exactly
            
            items = [f"{name} (Z={z_start}â†’{z_end}, size={z_size})" 
                    for name, z_start, z_end, z_size in matching]
            
            chosen, ok = QInputDialog.getItem(
                self, "Select Layer",
                f"Mask Z size = {mask_z}. Which layer does this mask belong to?",
                items, 0, False)
            
            if ok and chosen:
                idx = items.index(chosen)
                return matching[idx][1]
            
            return None
            
        except Exception as e:
            QMessageBox.critical(self, "Config Error",
                f"Failed to parse config file:\n{str(e)}")
            return None

    def merge_class_masks(self):
        if self.class1_data is None and self.class2_data is None:
            self.segmentation_data = None
            return

        shape = self.class1_data.shape if self.class1_data is not None else self.class2_data.shape
        self.segmentation_data = np.zeros(shape, dtype=np.uint8)
        
        if self.class1_data is not None:
            self.segmentation_data[self.class1_data > 0] = 128
            
        if self.class2_data is not None:
            self.segmentation_data[self.class2_data > 0] = 255
            
    def clear_masks(self):
        """Clear C1/C2 segmentation masks from MPR + 3D volume overlays."""
        self.class1_data = None
        self.class2_data = None
        self.segmentation_data = None
        self.labeled_class1_data = None
        if hasattr(self, 'labeled_class2_data'):
            self.labeled_class2_data = None
        self.object_stats = []
        if hasattr(self, 'selected_highlight_objects'):
            self.selected_highlight_objects = []
        if hasattr(self, '_refresh_stats_table'):
            self._refresh_stats_table()

        # Remove mask shells / volumes from the 3D pane (MPR re-render alone leaves them).
        from inno3d.features.viewer.seg_mask_3d import remove_mask_overlay_from_renderer
        ren = getattr(self, 'view_3d_renderer', None)
        for name in ('c1_actor_3d', 'c2_actor_3d'):
            act = getattr(self, name, None)
            if act is not None and ren is not None:
                remove_mask_overlay_from_renderer(ren, act)
            setattr(self, name, None)

        # Mask-dependent 3D overlays (MES pick surfaces / B2B gaps)
        if hasattr(self, '_clear_mes_3d_highlight'):
            try:
                self._clear_mes_3d_highlight(render=False)
            except Exception:
                pass
        if hasattr(self, '_clear_b2b_gap_actors'):
            try:
                self._clear_b2b_gap_actors()
            except Exception:
                pass

        if self.volume_data is not None:
            for orientation in ['axial', 'coronal', 'sagittal']:
                self.render_slice(orientation)

        widget = getattr(self, 'view_3d_widget', None)
        if widget is not None:
            try:
                if ren is not None:
                    ren.ResetCameraClippingRange()
                widget.GetRenderWindow().Render()
            except Exception:
                pass

        self.update_info_label()

    def clear_all_views(self):
        """Completely reset all 2D and 3D views to a blank state."""
        self.volume_data = None
        self.clear_masks()
        
        # Clear 3D actors
        if getattr(self, 'volume_actor', None):
            self.view_3d_renderer.RemoveVolume(self.volume_actor)
            self.volume_actor = None
        from inno3d.features.viewer.seg_mask_3d import remove_mask_overlay_from_renderer
        if getattr(self, 'c1_actor_3d', None):
            remove_mask_overlay_from_renderer(self.view_3d_renderer, self.c1_actor_3d)
            self.c1_actor_3d = None
        if getattr(self, 'c2_actor_3d', None):
            remove_mask_overlay_from_renderer(self.view_3d_renderer, self.c2_actor_3d)
            self.c2_actor_3d = None
            
        # Clear 2D actors
        for ori in ['axial', 'coronal', 'sagittal']:
            renderer = getattr(self, f'{ori}_renderer')
            renderer.RemoveAllViewProps()
            widget = getattr(self, f'{ori}_widget')
            widget.GetRenderWindow().Render()
            
        if hasattr(self, '_cached_slice_actors'):
            self._cached_slice_actors.clear()
            
        if hasattr(self, '_persistent_crosshair'):
            self._persistent_crosshair.clear()
            
        # Also clear object stats table
        if hasattr(self, 'object_stats_table'):
            self.object_stats_table.blockSignals(True)
            self.object_stats_table.setRowCount(0)
            self.object_stats_table.blockSignals(False)
            
        self.view_3d_widget.GetRenderWindow().Render()
        self.update_info_label()

    def align_hbm_volume(self):
        """Align the HBM volume data using PCA alignment (with LTIC principles)"""
        # Threads still defined on tabs.viewer (host module) — lazy to avoid circular import
        from inno3d.tabs.viewer import AlignVolumeThread

        if self.volume_data is None:
            QMessageBox.warning(self, "No Data", "Please load a volume file first.")
            return
            
        # Show progress dialog
        self.align_progress = QProgressDialog("Aligning HBM volume...", "Cancel", 0, 100, self)
        self.align_progress.setWindowModality(Qt.WindowModal)
        self.align_progress.setMinimumDuration(0)
        self.align_progress.show()
        
        # Start worker thread
        self.align_thread = AlignVolumeThread(self.volume_data)
        
        def on_progress(value, msg):
            self.align_progress.setValue(value)
            self.align_progress.setLabelText(msg)
            
        def on_finished(aligned_vol, best_R, error):
            self.align_progress.close()
            if error:
                QMessageBox.critical(self, "Alignment Error", f"Failed to align volume: {error}")
                return
                
            # Create backup of unaligned data first time alignment is run
            if getattr(self, 'original_volume_data', None) is None:
                self.original_volume_data = self.volume_data.copy()
                self.original_class1_data = self.class1_data.copy() if self.class1_data is not None else None
                self.original_class2_data = self.class2_data.copy() if self.class2_data is not None else None
                self.original_labeled_class1_data = self.labeled_class1_data.copy() if getattr(self, 'labeled_class1_data', None) is not None else None
                self.original_labeled_class2_data = self.labeled_class2_data.copy() if getattr(self, 'labeled_class2_data', None) is not None else None

            # Update volume data
            self.volume_data = aligned_vol
            
            # Apply same rotation to labels and masks if they exist
            import math
            c_old = (np.array(aligned_vol.shape) - 1.0) / 2.0
            offset = c_old - np.dot(best_R, c_old)
            
            # Rotate Class 1 mask
            if self.class1_data is not None:
                self.class1_data = ndimage.affine_transform(
                    self.class1_data,
                    matrix=best_R,
                    offset=offset,
                    order=0,
                    mode='constant',
                    cval=0
                )
                
            # Rotate Class 2 mask
            if self.class2_data is not None:
                self.class2_data = ndimage.affine_transform(
                    self.class2_data,
                    matrix=best_R,
                    offset=offset,
                    order=0,
                    mode='constant',
                    cval=0
                )
                
            # Re-merge masks if either exists
            if self.class1_data is not None or self.class2_data is not None:
                self.merge_class_masks()
                
            # Rotate labeled arrays if present
            if getattr(self, 'labeled_class1_data', None) is not None:
                self.labeled_class1_data = ndimage.affine_transform(
                    self.labeled_class1_data,
                    matrix=best_R,
                    offset=offset,
                    order=0,
                    mode='constant',
                    cval=0
                )
            if getattr(self, 'labeled_class2_data', None) is not None:
                self.labeled_class2_data = ndimage.affine_transform(
                    self.labeled_class2_data,
                    matrix=best_R,
                    offset=offset,
                    order=0,
                    mode='constant',
                    cval=0
                )
                
            # Reset oblique angles to 0
            self.oblique_angles = {'axial': 0.0, 'coronal': 0.0, 'sagittal': 0.0}
            
            # Recalculate stats if needed
            if hasattr(self, 'update_object_measurements'):
                self.update_object_measurements()
                
            # Re-initialize crosshair position at new center
            z, y, x = aligned_vol.shape
            self.crosshair_position = [x // 2, y // 2, z // 2]
            
            # Update sliders for crosshair
            for axis, dim, ctr in [('x', x-1, x//2), ('y', y-1, y//2), ('z', z-1, z//2)]:
                sl = getattr(self, f'crosshair_slider_{axis}', None)
                if sl:
                    sl.blockSignals(True)
                    sl.setMaximum(dim)
                    sl.setValue(ctr)
                    sl.blockSignals(False)
            
            # Redraw all views
            self.update_all_views()
            
            # Compute angles in degrees for user feedback
            try:
                pitch = math.degrees(math.asin(-best_R[0, 2]))
                cos_pitch = math.cos(math.radians(pitch))
                if abs(cos_pitch) > 1e-4:
                    roll = math.degrees(math.atan2(best_R[1, 2], best_R[2, 2]))
                    yaw = math.degrees(math.atan2(best_R[0, 1], best_R[0, 0]))
                else:
                    roll = 0.0
                    yaw = math.degrees(math.atan2(-best_R[1, 0], best_R[1, 1]))
            except Exception:
                roll, pitch, yaw = 0.0, 0.0, 0.0
                
            # Enable reset buttons
            if hasattr(self, 'btn_reset_align'):
                self.btn_reset_align.setEnabled(True)
            if hasattr(self, 'reset_align_btn'):
                self.reset_align_btn.setEnabled(True)
                
            QMessageBox.information(
                self, 
                "Alignment Success", 
                f"HBM volume aligned successfully!\n\n"
                f"Computed misalignment corrections:\n"
                f"• Roll (X-axis): {roll:+.2f}°\n"
                f"• Pitch (Y-axis): {pitch:+.2f}°\n"
                f"• Yaw (Z-axis): {yaw:+.2f}°"
            )
            
        self.align_thread.progress.connect(on_progress)
        self.align_thread.finished.connect(on_finished)
        self.align_thread.start()

    def reset_hbm_alignment(self):
        """Restore the volume and masks to their original unaligned states"""
        if getattr(self, 'original_volume_data', None) is None:
            QMessageBox.warning(self, "No Backup", "No alignment has been performed yet, or backup is not available.")
            return
            
        self.volume_data = self.original_volume_data.copy()
        
        self.class1_data = self.original_class1_data.copy() if self.original_class1_data is not None else None
        self.class2_data = self.original_class2_data.copy() if self.original_class2_data is not None else None
        
        if self.class1_data is not None or self.class2_data is not None:
            self.merge_class_masks()
        else:
            self.segmentation_data = None
            
        if self.original_labeled_class1_data is not None:
            self.labeled_class1_data = self.original_labeled_class1_data.copy()
        else:
            self.labeled_class1_data = None
            
        if getattr(self, 'original_labeled_class2_data', None) is not None:
            self.labeled_class2_data = self.original_labeled_class2_data.copy()
        else:
            self.labeled_class2_data = None
            
        # Reset oblique angles to 0
        self.oblique_angles = {'axial': 0.0, 'coronal': 0.0, 'sagittal': 0.0}
        
        # Recalculate stats if needed
        if hasattr(self, 'update_object_measurements'):
            self.update_object_measurements()
            
        # Re-initialize crosshair position at original center
        z, y, x = self.volume_data.shape
        self.crosshair_position = [x // 2, y // 2, z // 2]
        
        # Update sliders for crosshair
        for axis, dim, ctr in [('x', x-1, x//2), ('y', y-1, y//2), ('z', z-1, z//2)]:
            sl = getattr(self, f'crosshair_slider_{axis}', None)
            if sl:
                sl.blockSignals(True)
                sl.setMaximum(dim)
                sl.setValue(ctr)
                sl.blockSignals(False)
        
        # Redraw all views
        self.update_all_views()
        
        # Disable reset buttons
        if hasattr(self, 'btn_reset_align'):
            self.btn_reset_align.setEnabled(False)
        if hasattr(self, 'reset_align_btn'):
            self.reset_align_btn.setEnabled(False)
            
        QMessageBox.information(self, "Reset Success", "Volume and masks restored to original unaligned states.")

    def _toggle_oblique_handles(self, checked):
        """Toggle Dragonfly-style oblique rotation handles on/off.
        
        When ON: shows curved arrow handles on crosshair lines, auto-enables
        crosshair if not already active.
        When OFF: hides handles and resets hover state.
        """
        self.oblique_handles_enabled = checked
        
        if checked:
            # Update button visual
            if hasattr(self, 'btn_oblique_handles'):
                self.btn_oblique_handles.setStyleSheet("background: rgba(0, 200, 100, 0.3); font-weight: bold;")
            
            # Auto-enable crosshair (handles require it)
            if not self.crosshair_enabled:
                self.crosshair_enabled = True
                parent = self.parent()
                while parent is not None:
                    if hasattr(parent, 'cross_btn'):
                        parent.cross_btn.setChecked(True)
                        break
                    parent = parent.parent()
        else:
            # Clear hover/drag state
            self._oblique_hover_handle = None
            self._oblique_dragging = None
            if hasattr(self, 'btn_oblique_handles'):
                self.btn_oblique_handles.setStyleSheet("")
            # Reset cursors
            for ori in ['axial', 'coronal', 'sagittal']:
                w = getattr(self, f'{ori}_widget', None)
                if w:
                    w.unsetCursor()
        
        # Re-render all views to show/hide handles
        if self.volume_data is not None:
            for ori in ['axial', 'coronal', 'sagittal']:
                self.render_slice(ori, preserve_camera=True)
    
    def _toggle_manual_align_mode(self, checked):
        """Toggle manual alignment mode on/off.
        
        When ON: enables crosshair + oblique rotate mode so user can drag
        crosshair handles on 2D views to preview rotation in real-time.
        When OFF: disables the mode and leaves the preview rotation in place.
        """
        self._manual_align_active = checked
        
        if checked:
            # Turn ON: enable crosshair and oblique dragging
            if hasattr(self, 'btn_manual_align_mode'):
                self.btn_manual_align_mode.setText("Manual Align: ON")
                self.btn_manual_align_mode.setStyleSheet("background: rgba(0, 200, 100, 0.3); font-weight: bold;")
            if hasattr(self, 'btn_commit_manual_align'):
                self.btn_commit_manual_align.setEnabled(True)
            if hasattr(self, 'btn_cancel_manual_align'):
                self.btn_cancel_manual_align.setEnabled(True)
            
            # Auto-enable rotation handles for manual align
            if not self.oblique_handles_enabled:
                self.oblique_handles_enabled = True
                if hasattr(self, 'btn_oblique_handles'):
                    self.btn_oblique_handles.setChecked(True)
                    self.btn_oblique_handles.setStyleSheet("background: rgba(0, 200, 100, 0.3); font-weight: bold;")
            
            # Ensure crosshair is visible (needed for handle-based rotation)
            if not self.crosshair_enabled:
                self.crosshair_enabled = True
                parent = self.parent()
                while parent is not None:
                    if hasattr(parent, 'cross_btn'):
                        parent.cross_btn.setChecked(True)
                        break
                    parent = parent.parent()
            
            # Reset oblique angles for a fresh manual align session
            self.oblique_angles = {'axial': 0.0, 'coronal': 0.0, 'sagittal': 0.0}
            self._update_manual_angle_labels()
            
            if self.volume_data is not None:
                for ori in ['axial', 'coronal', 'sagittal']:
                    self.render_slice(ori, preserve_camera=True)
                self.update_3d_crosshair()
        else:
            # Turn OFF: just update button text, keep preview angles
            if hasattr(self, 'btn_manual_align_mode'):
                self.btn_manual_align_mode.setText("Manual Align: OFF")
                self.btn_manual_align_mode.setStyleSheet("")
        
        # Sync the left sidebar button in MainWindow
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, 'manual_align_btn'):
                parent.manual_align_btn.blockSignals(True)
                parent.manual_align_btn.setChecked(checked)
                if checked:
                    parent.manual_align_btn.setText("MANUAL ALIGN: ON")
                    parent.manual_align_btn.setStyleSheet("background: rgba(0, 200, 100, 0.3); font-weight: bold;")
                else:
                    parent.manual_align_btn.setText("MANUAL ALIGN")
                    parent.manual_align_btn.setStyleSheet("")
                parent.manual_align_btn.blockSignals(False)
                break
            parent = parent.parent()
    
    def _update_manual_angle_labels(self):
        """Sync the angle display labels with current oblique_angles.
        
        Called after every oblique rotation drag so the user sees the live angles.
        """
        if not hasattr(self, 'manual_angle_x_label'):
            return
        roll = self.oblique_angles.get('sagittal', 0.0)
        pitch = self.oblique_angles.get('coronal', 0.0)
        yaw = self.oblique_angles.get('axial', 0.0)
        self.manual_angle_x_label.setText(f"{roll:+.2f}°")
        self.manual_angle_y_label.setText(f"{pitch:+.2f}°")
        self.manual_angle_z_label.setText(f"{yaw:+.2f}°")
    
    def commit_manual_alignment(self):
        """Apply the current oblique preview rotation permanently to the volume data.
        
        Reads the current oblique_angles (set interactively by dragging crosshair
        handles on 2D views), computes the rotation matrix, applies affine_transform
        to volume + masks, then resets oblique_angles to 0.
        """
        if self.volume_data is None:
            QMessageBox.warning(self, "No Data", "Please load a volume file first.")
            return
        
        roll = self.oblique_angles.get('sagittal', 0.0)
        pitch = self.oblique_angles.get('coronal', 0.0)
        yaw = self.oblique_angles.get('axial', 0.0)
        
        if abs(roll) < 0.01 and abs(pitch) < 0.01 and abs(yaw) < 0.01:
            QMessageBox.information(self, "No Rotation", "No rotation to commit. Drag crosshair handles first.")
            return
        
        # Create backup if this is the first alignment operation
        if getattr(self, 'original_volume_data', None) is None:
            self.original_volume_data = self.volume_data.copy()
            self.original_class1_data = self.class1_data.copy() if self.class1_data is not None else None
            self.original_class2_data = self.class2_data.copy() if self.class2_data is not None else None
            self.original_labeled_class1_data = self.labeled_class1_data.copy() if getattr(self, 'labeled_class1_data', None) is not None else None
            self.original_labeled_class2_data = self.labeled_class2_data.copy() if getattr(self, 'labeled_class2_data', None) is not None else None
        
        # Show progress dialog
        from inno3d.tabs.viewer import ManualAlignThread

        self.manual_align_progress = QProgressDialog("Committing manual alignment...", "Cancel", 0, 100, self)
        self.manual_align_progress.setWindowModality(Qt.WindowModal)
        self.manual_align_progress.setMinimumDuration(0)
        self.manual_align_progress.show()
        
        # Use the oblique_angles as rotation angles
        self.manual_align_thread = ManualAlignThread(self.volume_data, roll, pitch, yaw)
        
        def on_progress(value, msg):
            self.manual_align_progress.setValue(value)
            self.manual_align_progress.setLabelText(msg)
            
        def on_finished(rotated_vol, R_matrix, error):
            self.manual_align_progress.close()
            if error:
                QMessageBox.critical(self, "Manual Alignment Error", f"Failed to apply rotation: {error}")
                return
                
            # Update volume data
            self.volume_data = rotated_vol
            
            # Apply same rotation to masks
            c_old = (np.array(rotated_vol.shape) - 1.0) / 2.0
            offset = c_old - np.dot(R_matrix, c_old)
            
            if self.class1_data is not None:
                self.class1_data = ndimage.affine_transform(
                    self.class1_data, matrix=R_matrix, offset=offset, order=0, mode='constant', cval=0)
            if self.class2_data is not None:
                self.class2_data = ndimage.affine_transform(
                    self.class2_data, matrix=R_matrix, offset=offset, order=0, mode='constant', cval=0)
            if self.class1_data is not None or self.class2_data is not None:
                self.merge_class_masks()
            if getattr(self, 'labeled_class1_data', None) is not None:
                self.labeled_class1_data = ndimage.affine_transform(
                    self.labeled_class1_data, matrix=R_matrix, offset=offset, order=0, mode='constant', cval=0)
            if getattr(self, 'labeled_class2_data', None) is not None:
                self.labeled_class2_data = ndimage.affine_transform(
                    self.labeled_class2_data, matrix=R_matrix, offset=offset, order=0, mode='constant', cval=0)
            
            # Reset oblique angles to 0 (rotation is now baked into the data)
            self.oblique_angles = {'axial': 0.0, 'coronal': 0.0, 'sagittal': 0.0}
            self._update_manual_angle_labels()
            
            # Recalculate stats
            if hasattr(self, 'update_object_measurements'):
                self.update_object_measurements()
            
            # Update crosshair
            z, y, x = rotated_vol.shape
            self.crosshair_position = [x // 2, y // 2, z // 2]
            for axis, dim, ctr in [('x', x-1, x//2), ('y', y-1, y//2), ('z', z-1, z//2)]:
                sl = getattr(self, f'crosshair_slider_{axis}', None)
                if sl:
                    sl.blockSignals(True)
                    sl.setMaximum(dim)
                    sl.setValue(ctr)
                    sl.blockSignals(False)
            
            self.update_all_views()
            
            # Enable reset buttons
            if hasattr(self, 'btn_reset_align'):
                self.btn_reset_align.setEnabled(True)
            if hasattr(self, 'reset_align_btn'):
                self.reset_align_btn.setEnabled(True)
            
            # Turn off manual align mode
            self.btn_manual_align_mode.setChecked(False)
            self._toggle_manual_align_mode(False)
            self.btn_commit_manual_align.setEnabled(False)
            self.btn_cancel_manual_align.setEnabled(False)
            
            QMessageBox.information(
                self, "Manual Alignment Committed",
                f"Volume rotation applied permanently!\n\n"
                f"Applied rotation angles:\n"
                f"• Roll (X-axis): {roll:+.2f}°\n"
                f"• Pitch (Y-axis): {pitch:+.2f}°\n"
                f"• Yaw (Z-axis): {yaw:+.2f}°"
            )
            
        self.manual_align_thread.progress.connect(on_progress)
        self.manual_align_thread.finished.connect(on_finished)
        self.manual_align_thread.start()
    
    def _cancel_manual_align(self):
        """Cancel the manual alignment preview and reset oblique angles to 0."""
        self.oblique_angles = {'axial': 0.0, 'coronal': 0.0, 'sagittal': 0.0}
        self._update_manual_angle_labels()
        
        # Turn off manual align mode
        self.btn_manual_align_mode.setChecked(False)
        self._toggle_manual_align_mode(False)
        self.btn_commit_manual_align.setEnabled(False)
        self.btn_cancel_manual_align.setEnabled(False)
        
        # Re-render with no oblique angles (reset to straight views)
        if self.volume_data is not None:
            for ori in ['axial', 'coronal', 'sagittal']:
                self.render_slice(ori, preserve_camera=True)
            self.update_3d_crosshair()
    
    def save_aligned_volume(self):
        """Save the current volume data to a TIFF stack or RAW binary file."""
        if self.volume_data is None:
            QMessageBox.warning(self, "No Data", "No volume data to save.")
            return
        
        from pathlib import Path
        
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Volume",
            "",
            "TIFF Stack (*.tif *.tiff);;RAW Binary (*.raw);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            ext = Path(file_path).suffix.lower()
            
            progress = QProgressDialog("Saving volume...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()
            
            if ext in ['.tif', '.tiff']:
                # Save as multi-page TIFF using tifffile or skimage
                progress.setLabelText("Writing TIFF stack...")
                progress.setValue(10)
                
                try:
                    import tifffile
                    tifffile.imwrite(file_path, self.volume_data)
                except ImportError:
                    from skimage import io as skio
                    skio.imsave(file_path, self.volume_data)
                    
                progress.setValue(90)
                
            elif ext == '.raw':
                # Save as flat binary
                progress.setLabelText("Writing RAW binary...")
                progress.setValue(10)
                
                self.volume_data.tofile(file_path)
                progress.setValue(80)
                
                # Write metadata sidecar file
                meta_path = file_path + '.meta.txt'
                z, y, x = self.volume_data.shape
                with open(meta_path, 'w') as f:
                    f.write(f"# Volume Metadata\n")
                    f.write(f"shape_z={z}\n")
                    f.write(f"shape_y={y}\n")
                    f.write(f"shape_x={x}\n")
                    f.write(f"dtype={self.volume_data.dtype}\n")
                    f.write(f"byte_order=little_endian\n")
                    f.write(f"spacing_x={self.spacing[0]}\n")
                    f.write(f"spacing_y={self.spacing[1]}\n")
                    f.write(f"spacing_z={self.spacing[2]}\n")
                progress.setValue(90)
            else:
                # Fallback: try TIFF
                progress.setLabelText("Writing TIFF stack (default)...")
                progress.setValue(10)
                try:
                    import tifffile
                    tifffile.imwrite(file_path, self.volume_data)
                except ImportError:
                    from skimage import io as skio
                    skio.imsave(file_path, self.volume_data)
                progress.setValue(90)
            
            progress.setValue(100)
            progress.close()
            
            z, y, x = self.volume_data.shape
            file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            
            QMessageBox.information(
                self, "Save Success",
                f"Volume saved successfully!\n\n"
                f"File: {file_path}\n"
                f"Shape: ({z}, {y}, {x})\n"
                f"Dtype: {self.volume_data.dtype}\n"
                f"Size: {file_size_mb:.1f} MB"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save volume:\n{str(e)}")

    def update_info_label(self):
        """Update info label vÃ¡Â»â€ºi status cÃ¡Â»Â§a masks"""
        info_parts = []
        
        if self.volume_data is not None:
            z, y, x = self.volume_data.shape
            info_parts.append(f"Vol:({z},{y},{x})")
        
        if self.class1_data is not None:
            c1_voxels = np.sum(self.class1_data > 127)
            info_parts.append(f"C1:{c1_voxels:,}")
        
        if self.class2_data is not None:
            c2_voxels = np.sum(self.class2_data > 127)
            info_parts.append(f"C2:{c2_voxels:,}")
        
        if info_parts:
            self.info_label.setText(" | ".join(info_parts))
        else:
            self.info_label.setText("No data loaded")

# ==================== INSPECTION RESULT TAB ====================

# ==================== INSPECTION RESULT TAB ====================


