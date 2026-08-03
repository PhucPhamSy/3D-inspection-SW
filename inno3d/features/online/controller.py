# inno3d/features/online/controller.py
# -----------------------------------------------------------------------
# OnlineModeMixin — extracted from inno3d/modes/online.py (Phase 5)
#
# TCP orchestration, FOV pipeline, MES/B2B, DB record, wafer context.
# This is the main mixin applied to MainWindow for Online mode.
# -----------------------------------------------------------------------

"""
Online mode support: TCP server, auto-load pipeline, and MainWindow mixin.

Protocol matches Server.cpp / ClientBumpVoid.cpp:
  Port: 8000
  RECV: int32 counter + char Value1[512] + 4 bools = 520 bytes
  SEND: int32 counter + char Value1[512] + 3 int32 values = 528 bytes
"""

import concurrent.futures
import gc
import os
import re
import shutil
import socket
import struct
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import *
from skimage import io

from inno3d.core.styles import SemiconductorTheme
from inno3d.core.online_mpr_walltime import (
    OnlineMprWalltimeReport,
    online_walltime_phase,
)
from inno3d.core.ui_system import OnlinePipelineProgress
from inno3d.core.view_support import LoadVolumeThread
from inno3d.core.wafer_context import (
    BIN_NG,
    BIN_OK,
    BIN_PENDING,
    HostVolumePathInfo,
    parse_host_volume_path,
)
from inno3d.features.online.server import OnlineLoadThread, OnlineServerThread
from inno3d.features.teaching.workers import EnhancementThread
# ==================== PACKET STRUCTURES ====================
# Server.cpp / ClientBumpVoid.cpp use char Value1[512] (NOT wchar_t!)
# _EXAMPLE_RECV_PACKET: int32(4) + char[512](512) + 4 bools(4) = 520 bytes
RECV_PACKET_SIZE = 520

# _EXAMPLE_SEND_PACKET: int32(4) + char[512](512) + 3Ã—int32(12) = 528 bytes
SEND_PACKET_SIZE = 528

# Fixed port â€” must match Server.cpp #define PORT 8000
FIXED_PORT = 8000


def _log(msg):
    """Print timestamped log message to console"""
    ts = time.strftime("%H:%M:%S")
    print(f"[ONLINE {ts}] {msg}")


class OnlineModeMixin:
    def _online_walltime_begin(self, received_path):
        """Start wall-time tracking for Online receive → first MPR paint."""
        self._online_mpr_walltime = OnlineMprWalltimeReport(host_path=str(received_path))
        self._online_mpr_walltime.mark("pipeline_start")
        mpv = getattr(self, "multiplanar_tab", None)
        if mpv is not None:
            mpv._online_mpr_walltime = self._online_mpr_walltime

    def _online_walltime_phase(self, phase_name, elapsed_sec=None, note=""):
        """Append one phase timing entry (best-effort, UI-thread only)."""
        wt = getattr(self, "_online_mpr_walltime", None)
        if wt is None:
            return
        if elapsed_sec is not None:
            wt.record(str(phase_name), float(elapsed_sec))
        else:
            wt.mark(str(phase_name))
        if note:
            wt.set_meta(f"note_{phase_name}", str(note))

    def _online_walltime_finalize_first_mpr(self):
        """Seal receive → first MPR timing and write final report."""
        wt = getattr(self, "_online_mpr_walltime", None)
        if wt is None:
            return
        mpv = getattr(self, "multiplanar_tab", None)
        if mpv is not None and getattr(mpv, "volume_data", None) is not None:
            vd = mpv.volume_data
            try:
                wt.set_meta("volume_shape", tuple(vd.shape))
                wt.set_meta("volume_dtype", str(getattr(vd, "dtype", "")))
                wt.set_meta("volume_nbytes", int(getattr(vd, "nbytes", 0)))
                wt.set_meta(
                    "lve_used_store",
                    bool(getattr(mpv, "volume_store", None) is not None),
                )
            except Exception:
                pass
        wt.mark("first_mpr_ready")
        report_path = wt.write_report(output_dir=Path(__file__).resolve().parents[3] / "output")
        if report_path is not None:
            total_s = wt.elapsed_total()
            msg = f"[WALLTIME] first MPR ready in {total_s:.2f}s → {report_path}"
            print(f"[ONLINE] {msg}")
            if hasattr(self, "_online_append_log") and self._online_append_log:
                self._online_append_log(msg, SemiconductorTheme.TEXT_SECONDARY)
        if mpv is not None:
            mpv._online_mpr_walltime = None
        self._online_mpr_walltime = None

    def _online_walltime_abort(self, error_text=""):
        """Write partial report on pipeline failure (never raises)."""
        wt = getattr(self, "_online_mpr_walltime", None)
        if wt is None:
            return
        if error_text:
            wt.set_meta("error", str(error_text))
        try:
            wt.write_report(output_dir=Path(__file__).resolve().parents[3] / "output")
        except Exception as e:
            if hasattr(self, "_online_append_log") and self._online_append_log:
                self._online_append_log(
                    f"[WALLTIME WARN] failed to write report: {e}",
                    SemiconductorTheme.ACCENT_WARNING,
                )
        mpv = getattr(self, "multiplanar_tab", None)
        if mpv is not None:
            mpv._online_mpr_walltime = None
        self._online_mpr_walltime = None

    def _online_preflight_cleanup_for_new_volume(self):
        """Release prior Online runtime volume memory before next pipeline."""
        mpv = getattr(self, "multiplanar_tab", None)
        seg_tab = getattr(self, "segmentation_tab", None)

        # Stop/replace old viewer load thread if still around.
        try:
            old_thread = getattr(mpv, "load_thread", None) if mpv is not None else None
            if old_thread is not None and hasattr(old_thread, "isRunning") and old_thread.isRunning():
                old_thread.terminate()
                old_thread.wait(200)
        except Exception:
            pass

        # Release viewer-side heavy objects (stores, 3D pins, actors, dense arrays).
        if mpv is not None:
            try:
                prev_store = getattr(mpv, "volume_store", None)
                if prev_store is not None and hasattr(prev_store, "close"):
                    prev_store.close()
            except Exception:
                pass
            try:
                mpv.volume_store = None
            except Exception:
                pass
            try:
                mpv._vtk_3d_numpy_pin = None
            except Exception:
                pass
            try:
                if hasattr(mpv, "_teardown_3d_volume_actors"):
                    mpv._teardown_3d_volume_actors()
            except Exception:
                pass
            try:
                mpv.clear_all_views()
            except Exception:
                pass

        # Release teaching-side runtime data while keeping config/DLL state.
        if seg_tab is not None:
            try:
                prev_store = getattr(seg_tab, "volume_store", None)
                if prev_store is not None and hasattr(prev_store, "close"):
                    prev_store.close()
            except Exception:
                pass
            try:
                seg_tab.volume_store = None
            except Exception:
                pass
            for attr in (
                "volume_data",
                "bump_segmentation",
                "void_segmentation",
                "labeled_class1_data",
                "labeled_class2_data",
                "_teaching_vtk_numpy_pin",
            ):
                try:
                    setattr(seg_tab, attr, None)
                except Exception:
                    pass
            try:
                if hasattr(seg_tab, "_3d_renderer") and getattr(seg_tab, "_3d_volume_actor", None) is not None:
                    seg_tab._3d_renderer.RemoveVolume(seg_tab._3d_volume_actor)
                    seg_tab._3d_volume_actor = None
                    seg_tab._3d_volume_mapper = None
                if hasattr(seg_tab, "_3d_renderer") and getattr(seg_tab, "_3d_seg_actor", None) is not None:
                    seg_tab._3d_renderer.RemoveActor(seg_tab._3d_seg_actor)
                    seg_tab._3d_seg_actor = None
            except Exception:
                pass

        gc.collect()
        self._online_append_log(
            "[MEM] Preflight cleanup done (released prior volume RAM/VRAM state)",
            SemiconductorTheme.TEXT_SECONDARY,
        )

    # ── Online progress (stage-only, no algorithm detail) ─────────────
    def _online_has_enhance(self) -> bool:
        return bool(
            getattr(self, "_online_enhancement_enabled", False)
            and getattr(self, "_online_enhancement_model", None)
        )

    def _open_online_progress(self, stage="load"):
        """Show compact stage progress card for the current FOV.

        Never raises into the online pipeline — progress UI is best-effort only.
        """
        try:
            has_enh = self._online_has_enhance()
            dlg = getattr(self, "_online_progress", None)
            if dlg is None or not isinstance(dlg, OnlinePipelineProgress):
                dlg = OnlinePipelineProgress(self, has_enhance=has_enh)
                self._online_progress = dlg
            else:
                dlg.configure(has_enhance=has_enh)
            dlg.set_stage(stage, 0)
            dlg.show()
            return dlg
        except Exception as e:
            print(f"[ONLINE] progress UI failed (pipeline continues): {e}")
            self._online_progress = None
            return None

    def _set_online_stage(self, stage, progress=None):
        dlg = getattr(self, "_online_progress", None)
        if dlg is None:
            return
        if isinstance(dlg, OnlinePipelineProgress):
            dlg.set_stage(stage, progress)
        else:
            # Legacy QProgressDialog fallback
            titles = OnlinePipelineProgress.STAGE_META
            title = titles.get(stage, ("Working", ""))[0]
            try:
                dlg.setLabelText(title)
                if progress is not None:
                    dlg.setValue(int(progress))
            except Exception:
                pass
        # Keep status sink short & non-secret
        titles = OnlinePipelineProgress.STAGE_META
        label = titles.get(stage, ("Working", ""))[0]
        try:
            self.status_label.setText(f"ONLINE: {label}")
        except Exception:
            pass

    def _tick_online_progress(self, value):
        """Update percent within current stage; ignore detailed callback strings."""
        dlg = getattr(self, "_online_progress", None)
        if not dlg:
            return
        try:
            if isinstance(dlg, OnlinePipelineProgress):
                dlg.set_stage_progress(value)
            else:
                dlg.setValue(int(value))
        except Exception:
            pass

    def _close_online_progress(self, ok=None):
        dlg = getattr(self, "_online_progress", None)
        if not dlg:
            return
        try:
            if isinstance(dlg, OnlinePipelineProgress):
                if ok is True:
                    dlg.finish_ok()
                elif ok is False:
                    dlg.finish_error()
                else:
                    dlg.close()
            else:
                dlg.close()
        except Exception:
            pass

    def _on_apply_online_roi(self, roi, config, config_path):
        """Handler for 'Apply for Online Mode' button in SegmentationTab.

        ``roi`` may contain XY/Z crop keys and/or ``layers`` (INPUT_MODE=single).
        """
        self.online_roi = roi if isinstance(roi, dict) else {}
        self.online_roi_active = True
        self.online_config_path = config_path

        # Refresh recipe if Online is already ON
        if getattr(self, 'online_toggle', None) is not None and self.online_toggle.isChecked():
            self._capture_online_recipe()
        
        status = "ONLINE RECIPE LOCKED: "
        if 'x2' in self.online_roi and 'x1' in self.online_roi:
            status += (
                f"{int(self.online_roi['x2'])-int(self.online_roi['x1'])}x"
                f"{int(self.online_roi['y2'])-int(self.online_roi['y1'])}x"
                f"{int(self.online_roi['z_end'])-int(self.online_roi['z_start'])}"
            )
        else:
            status += "no XY crop"
        
        if self.online_roi.get('layers'):
            status += f" · {len(self.online_roi['layers'])} layers"
            
        self.status_label.setText(status)
        
    def create_nav_button(self, text, index):
        """Create styled sidebar navigation button"""
        btn = QPushButton(text)
        btn.setProperty("class", "sidebar-btn")
        btn.setCheckable(True)
        btn.clicked.connect(lambda: self.switch_tab(index))
        return btn
        
    def switch_tab(self, index):
        """Switch content stack and update styling"""
        
        # Handle VTK visibility logic 
        self.on_tab_changed(index)
        
        # UI Updates
        self.stack.setCurrentIndex(index)
        self.sidebar_stack.setCurrentIndex(index)
        
        # Update navigation buttons state (Title Bar)
        for i, btn in enumerate(self.title_bar.nav_btns):
            btn.setChecked(i == index)
    
    
    def toggle_online(self):
        """Toggle Online mode ON/OFF"""
        import time as _time
        is_online = self.online_toggle.isChecked()
        seg_tab = self.segmentation_tab
        
        if is_online:
            # --- Turn ON ---
            # No longer asking for config folder (using pre-loaded config from Segmentation tab)
            
            # Save current config to restore later when Online mode is turned OFF
            self._old_manual_config = seg_tab.config_path
            self._manual_online_state = {
                "input_path_text": seg_tab.input_path_input.text().strip() if hasattr(seg_tab, "input_path_input") else "",
                "output_path_text": seg_tab.output_path_input.text().strip() if hasattr(seg_tab, "output_path_input") else "",
                "config_input_path": bytes(seg_tab.config.inputPath) if seg_tab.config and getattr(seg_tab.config, "inputPath", None) else None,
                "config_output_dir": bytes(seg_tab.config.outputDir) if seg_tab.config and getattr(seg_tab.config, "outputDir", None) else None,
            }
            
            self.online_config_folder = None
            
            # Check that DLL is already loaded from Teaching (V2 / V3 / custom)
            if seg_tab.dll_path is None:
                QMessageBox.warning(self, "Online Mode",
                    "Please load the DLL folder in the 3D Teaching tab first\n"
                    "(Browse → all_v2\\V2 or all_v2\\V3)\n"
                    "before turning on Online mode.")
                self.online_toggle.setChecked(False)
                self.online_config_folder = None
                return

            # Force Online to use the exact Teaching DLL package (not a stale default V2)
            try:
                ver = seg_tab.apply_dll_folder(
                    seg_tab.dll_path, persist=True, force=True
                )
                self._online_dll_bind_note = (
                    f"Teaching DLL bound for Online: {seg_tab.dll_path} (SEG v{ver})"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Online Mode",
                    f"Failed to bind Teaching DLL folder for Online:\n{seg_tab.dll_path}\n\n{e}",
                )
                self.online_toggle.setChecked(False)
                self.online_config_folder = None
                return
            
            # Check that config is loaded (or use default)
            if seg_tab.config is None:
                QMessageBox.warning(self, "Online Mode",
                    "Please load a Config file in the 3D Segmentation tab first\n"
                    "before turning on Online mode.\n\n"
                    "The loaded config will be used for ALL incoming files.\n"
                    "To change config: turn OFF Online -> edit/reload config -> turn ON again.")
                self.online_toggle.setChecked(False)
                return
            
            # Soft warning only when neither locked ROI nor layer bands exist
            has_layers_preview = False
            if getattr(seg_tab, 'layers_active', False):
                has_layers_preview = any(
                    L.get('selected') for L in getattr(seg_tab, 'layer_definitions', []) or []
                )
            if (
                self.online_roi_active
                and isinstance(self.online_roi, dict)
                and self.online_roi.get('layers')
            ):
                has_layers_preview = True
            has_xy_roi = (
                self.online_roi_active
                and isinstance(self.online_roi, dict)
                and 'x1' in self.online_roi
            )
            if not has_xy_roi and not has_layers_preview:
                reply = QMessageBox.warning(
                    self, "Online Mode",
                    "No Online ROI crop and no LAYER bands locked.\n"
                    "Will run full-volume (test_1layer-style) on each host FOV.\n\n"
                    "For HBM multi-die stacks: load config with LAYER_N or click\n"
                    "'Apply Layers for Online Mode' in Teaching first.\n\n"
                    "Continue?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                )
                if reply == QMessageBox.No:
                    self.online_toggle.setChecked(False)
                    return

            # Lock to tab 0 (3D Viewer) or tab 3 (3D Analysis)
            # If current tab is 3, keep it. Otherwise, switch to 0.
            if getattr(self, "current_tab_index", 0) != 3:
                self.switch_tab(0)
            for i, btn in enumerate(self.title_bar.nav_btns):
                btn.setEnabled(i in (0, 3))
            
            # Track online processing state
            self._online_processing = False
            
            # Show stats panel
            self.multiplanar_tab.stats_panel.setVisible(True)

            # Wafer → Chip → FOV context strip (left of MPR; does not alter MPR layout)
            if hasattr(self.multiplanar_tab, "set_online_context_visible"):
                self.multiplanar_tab.set_online_context_visible(True)

            # Collapse VIEW TOOLS (Window Leveling sidebar) → more room for MPR + stats
            # User reopens via thin TOOLS rail; choice remembered for this Online session.
            if hasattr(self.multiplanar_tab, "set_online_viewer_layout"):
                self.multiplanar_tab.set_online_viewer_layout(True)

            # Configure left app menu for Online mode (keep expanded by default, enable manual collapse button)
            if hasattr(self, "set_online_app_sidebar_layout"):
                self.set_online_app_sidebar_layout(True)
            
            # --- Online chrome + lock manual input (#3, #8) ---
            if hasattr(self, "_apply_online_chrome"):
                self._apply_online_chrome(True)
            if hasattr(self, "_set_online_input_locked"):
                self._set_online_input_locked(True)

            # --- Show Embedded Online Log ---
            self.multiplanar_tab.set_online_log_visible(True)
            self.multiplanar_tab.online_log_text.clear()
            self.repaint() # Force refresh to clear any ghosting artifacts
            
            def _append_log(msg, color=SemiconductorTheme.TEXT_PRIMARY):
                self.multiplanar_tab.append_online_log(msg, color)
            
            self._online_append_log = _append_log

            note = getattr(self, "_online_dll_bind_note", None)
            if note:
                _append_log(note, SemiconductorTheme.ACCENT_PRIMARY)
                self._online_dll_bind_note = None
            
            # Start TCP server
            self.online_server = OnlineServerThread(port=8000)
            self.online_server.folder_received.connect(self.on_online_folder_received)
            self.online_server.folder_received.connect(
                lambda path: _append_log(f"[FILE RECEIVED] {path}", SemiconductorTheme.ACCENT_SUCCESS))
            self.online_server.status_update.connect(lambda msg: self.status_label.setText(msg))
            self.online_server.status_update.connect(
                lambda msg: _append_log(msg, SemiconductorTheme.ACCENT_PRIMARY))
            self.online_server.server_error.connect(lambda msg: self.status_label.setText(msg))
            self.online_server.server_error.connect(
                lambda msg: _append_log(f"[ERROR] {msg}", SemiconductorTheme.ACCENT_ERROR))
            self.online_server.start()
            
            # Capture enhancement settings from SegmentationTab / Enhance UI
            if hasattr(seg_tab, '_sync_enhancement_state_from_ui'):
                try:
                    seg_tab._sync_enhancement_state_from_ui()
                except Exception:
                    pass
            self._online_enhancement_enabled = getattr(seg_tab, 'enhancement_enabled', False)
            self._online_enhancement_model = getattr(seg_tab, 'enhancement_model_path', '') or ''
            if hasattr(seg_tab, 'enh_model_input'):
                _m = seg_tab.enh_model_input.text().strip()
                if _m:
                    self._online_enhancement_model = _m
            self._online_enhancement_trt_cache = getattr(seg_tab, 'enh_trt_input', None)
            self._online_enhancement_trt_cache = self._online_enhancement_trt_cache.text().strip() if self._online_enhancement_trt_cache else './TRT_Cache'
            self._online_enhancement_use_gpu = getattr(seg_tab, 'enh_gpu_check', None)
            self._online_enhancement_use_gpu = self._online_enhancement_use_gpu.isChecked() if self._online_enhancement_use_gpu else True
            self._online_enhancement_gpu_id = getattr(seg_tab, 'enh_gpuid_spin', None)
            self._online_enhancement_gpu_id = self._online_enhancement_gpu_id.value() if self._online_enhancement_gpu_id else 0
            # full | layers  (from ENHANCEMENT_MODE / Enhance-tab radios)
            self._online_enhancement_mode = getattr(seg_tab, 'enhancement_mode', 'full') or 'full'
            if hasattr(seg_tab, 'get_enhancement_slice_range'):
                self._online_enhancement_start, self._online_enhancement_end = (
                    seg_tab.get_enhancement_slice_range()
                )
            else:
                self._online_enhancement_start, self._online_enhancement_end = -1, -1

            # Capture production recipe (input mode / layers / ROI) from Teaching
            self._capture_online_recipe()
            
            _append_log(f"Server started on port 8000", SemiconductorTheme.ACCENT_PRIMARY)
            recipe = getattr(self, '_online_recipe', {}) or {}
            mode = recipe.get('input_mode', 'test_1layer')
            layers = recipe.get('layers') or []
            _append_log(f"INPUT_MODE: {mode}", SemiconductorTheme.ACCENT_PRIMARY)
            if layers:
                _append_log(
                    f"LAYERS: {len(layers)} selected "
                    f"(Z bands → DLL split, stitch overlay)",
                    "#00BFA5",
                )
                for L in layers[:6]:
                    _append_log(
                        f"  · {L.get('name','?')} Z[{L.get('z_start')}:{L.get('z_end')}]",
                        SemiconductorTheme.TEXT_SECONDARY,
                    )
                if len(layers) > 6:
                    _append_log(f"  · … +{len(layers)-6} more", SemiconductorTheme.TEXT_SECONDARY)
            elif mode == 'single':
                _append_log(
                    "LAYERS: none selected → will run as full-volume (like test_1layer)",
                    SemiconductorTheme.ACCENT_WARNING,
                )
            if mode == 'multi_layer':
                _append_log(
                    "NOTE: multi_layer (sep files) is Teaching offline mode; "
                    "Online host path is per-FOV volume → use single/test_1layer recipe",
                    SemiconductorTheme.ACCENT_WARNING,
                )

            if self.online_roi_active and self.online_roi and 'x1' in self.online_roi:
                roi = self.online_roi
                _append_log(
                    f"ROI ACTIVE: X[{roi['x1']}:{roi['x2']}] "
                    f"Y[{roi['y1']}:{roi['y2']}] Z[{roi['z_start']}:{roi['z_end']}]",
                    "#00BFA5",
                )
            else:
                _append_log("ROI: FULL VOLUME (No XY/Z crop)", SemiconductorTheme.TEXT_SECONDARY)

            # CONTEXT hierarchy note
            ctx = None
            if hasattr(self.multiplanar_tab, "get_wafer_context"):
                ctx = self.multiplanar_tab.get_wafer_context()
            if ctx is not None:
                _append_log(
                    f"CONTEXT: {ctx.breadcrumb()}  ·  results → "
                    f"{self.multiplanar_tab.get_active_fov_results_dir() or 'FOV/results'}",
                    SemiconductorTheme.ACCENT_PRIMARY,
                )
            else:
                _append_log(
                    "CONTEXT: Wafer/Chip map ready (awaiting host DB payload)",
                    SemiconductorTheme.TEXT_SECONDARY,
                )
                
            # ISP DLL identity (name + version) for production audit
            try:
                from inno3d.core import bumpvoid as _seg_mod
                seg_info = _seg_mod.get_product_info()
                seg_ver = seg_info.get("version") or "?"
                seg_name = seg_info.get("name") or "BumpVoidSeg.dll"
                seg_path = seg_info.get("path") or (seg_tab.dll_path or "")
                _append_log(
                    f"[SEG] {seg_name}  v{seg_ver}  ·  {seg_path}",
                    SemiconductorTheme.ACCENT_PRIMARY,
                )
            except Exception as e:
                _append_log(
                    f"[SEG] folder={seg_tab.dll_path} (version unavailable: {e})",
                    SemiconductorTheme.TEXT_SECONDARY,
                )
            try:
                from inno3d.core import bumpvoid_mes as _mes_mod
                try:
                    # Prefer FAR-capable MES (skips locked pre-FAR all_v2\\V2 binary)
                    self._ensure_mes_loaded()
                except Exception as le:
                    _append_log(f"[MES] load defer: {le}", SemiconductorTheme.TEXT_SECONDARY)
                if _mes_mod.is_loaded():
                    mes_info = _mes_mod.get_product_info()
                    _append_log(
                        f"[MES] {mes_info.get('name')}  v{mes_info.get('version') or '?'}  ·  "
                        f"{mes_info.get('path') or ''}",
                        "#00BFA5",
                    )
                else:
                    _append_log(
                        "[MES] BumpVoidMes.dll not loaded yet (will load on first measure)",
                        SemiconductorTheme.TEXT_SECONDARY,
                    )
            except Exception as e:
                _append_log(f"[MES] unavailable: {e}", SemiconductorTheme.ACCENT_WARNING)
            try:
                from inno3d.core import bumpvoid_b2b as _b2b_mod
                if not _b2b_mod.is_loaded() and seg_tab.dll_path:
                    try:
                        _b2b_mod.load_dll(seg_tab.dll_path)
                    except Exception:
                        pass
                if _b2b_mod.is_loaded():
                    b2b_info = _b2b_mod.get_product_info()
                    _append_log(
                        f"[B2B] {b2b_info.get('name')}  v{b2b_info.get('version') or '?'}  ·  "
                        f"{b2b_info.get('path') or ''}",
                        "#FFB74D",
                    )
                else:
                    _append_log(
                        "[B2B] BoundaryGPU.dll not loaded yet (will load after MES)",
                        SemiconductorTheme.TEXT_SECONDARY,
                    )
            except Exception as e:
                _append_log(f"[B2B] unavailable: {e}", SemiconductorTheme.ACCENT_WARNING)

            # ENH DLL (same folder as SEG package — BumpVoid_ISP_ENH.dll)
            try:
                from inno3d.core import enhanced_volume as _enh_mod
                enh_path = ""
                enh_ver = "?"
                if seg_tab.dll_path:
                    for _name in ("BumpVoid_ISP_ENH.dll", "EnhancedVolumeDLL.dll"):
                        _cand = os.path.join(seg_tab.dll_path, _name)
                        if os.path.isfile(_cand):
                            enh_path = _cand
                            break
                try:
                    if seg_tab.dll_path:
                        _enh_mod.load_dll(seg_tab.dll_path)
                    enh_ver = _enh_mod.get_version() or "?"
                    if not enh_path:
                        enh_path = getattr(_enh_mod, "_dll_path", None) or seg_tab.dll_path or ""
                except Exception as le:
                    _append_log(
                        f"[ENH] load defer: {le}",
                        SemiconductorTheme.TEXT_SECONDARY,
                    )
                if enh_path or enh_ver != "?":
                    _buf = ""
                    try:
                        if hasattr(_enh_mod, "has_process_volume_buffer") and _enh_mod.has_process_volume_buffer():
                            _buf = "  ·  ProcessVolumeBuffer"
                    except Exception:
                        pass
                    _append_log(
                        f"[ENH] {os.path.basename(enh_path) or 'BumpVoid_ISP_ENH.dll'}  "
                        f"v{enh_ver}  ·  {enh_path or seg_tab.dll_path}{_buf}",
                        "#CE93D8",
                    )
                else:
                    _append_log(
                        "[ENH] BumpVoid_ISP_ENH.dll not found in DLL package",
                        SemiconductorTheme.TEXT_SECONDARY,
                    )
            except Exception as e:
                _append_log(f"[ENH] unavailable: {e}", SemiconductorTheme.ACCENT_WARNING)

            # Re-apply after Online DLL loads (reload clears SetProfiling flag)
            try:
                self._online_apply_dll_profiling()
            except Exception:
                pass

            _append_log(f"Config: {seg_tab.config_path or 'default'}", SemiconductorTheme.TEXT_SECONDARY)
            if self._online_enhancement_enabled:
                _mode = getattr(self, '_online_enhancement_mode', 'full')
                _s = getattr(self, '_online_enhancement_start', -1)
                _e = getattr(self, '_online_enhancement_end', -1)
                _range = f" slices[{_s}:{_e})" if _mode == 'layers' and _s >= 0 else " full volume"
                _append_log(
                    f"Enhancement: ENABLED — {os.path.basename(self._online_enhancement_model)} "
                    f"(mode={_mode}{_range})",
                    "#CE93D8",
                )
            else:
                _append_log(f"Enhancement: Disabled (raw volume → segmentation)", SemiconductorTheme.TEXT_SECONDARY)
            _append_log("Waiting for Server.cpp connection...", SemiconductorTheme.ACCENT_WARNING)

            try:
                from inno3d.core import bumpvoid as _seg_mod
                _sv = _seg_mod.get_version()
            except Exception:
                _sv = "?"
            self.status_label.setText(
                f"ONLINE: Ready [{mode}] - SEG v{_sv} | "
                f"Config: {os.path.basename(seg_tab.config_path or 'default')}"
            )
            
        else:
            # --- Turn OFF ---
            # Log shutdown
            if hasattr(self, '_online_append_log') and self._online_append_log:
                self._online_append_log("Server stopping...", SemiconductorTheme.ACCENT_WARNING)
            
            # Stop TCP server
            if self.online_server:
                self.online_server.stop()
                self.online_server.wait(2000)
                self.online_server = None
            
            # Log stopped
            if hasattr(self, '_online_append_log') and self._online_append_log:
                self._online_append_log("Server stopped - Online mode OFF", SemiconductorTheme.ACCENT_ERROR)
            
            # Unlock input + clear chrome
            if hasattr(self, "_set_online_input_locked"):
                self._set_online_input_locked(False)
            if hasattr(self, "_apply_online_chrome"):
                self._apply_online_chrome(False)

            # Hide stats panel + Wafer/Chip CONTEXT strip
            self.multiplanar_tab.set_online_log_visible(False)
            self.multiplanar_tab.stats_panel.setVisible(False)
            if hasattr(self.multiplanar_tab, "set_online_context_visible"):
                self.multiplanar_tab.set_online_context_visible(False)

            # Restore full VIEW TOOLS sidebar (Window Leveling) in manual mode
            if hasattr(self.multiplanar_tab, "set_online_viewer_layout"):
                self.multiplanar_tab.set_online_viewer_layout(False)

            # Restore full left app menu
            if hasattr(self, "set_online_app_sidebar_layout"):
                self.set_online_app_sidebar_layout(False)

            # Force viewer overlays and VTK widgets to repaint cleanly after Online mode UI changes.
            for orientation in ['axial', 'coronal', 'sagittal']:
                overlay = getattr(self.multiplanar_tab, f'{orientation}_overlay_group', None)
                if overlay is not None:
                    if getattr(overlay, 'slider_bar', None) is not None:
                        overlay.slider_bar.hide()
                    overlay.adjustSize()
                    overlay.update_position()
                try:
                    self.multiplanar_tab.render_slice(orientation, preserve_camera=True)
                except Exception:
                    pass
            try:
                self.multiplanar_tab.render_3d()
            except Exception:
                pass
            self.multiplanar_tab.update()
            self.multiplanar_tab.repaint()
            
            # Re-enable navigation
            for btn in self.title_bar.nav_btns:
                btn.setEnabled(True)
            
            self._online_processing = False
            self.status_label.setText("ONLINE: Disconnected - Manual mode")
            
            # --- Restore previous config ---
            if hasattr(self, '_old_manual_config') and self._old_manual_config:
                if os.path.exists(self._old_manual_config):
                    seg_tab.load_config_file(self._old_manual_config)
                self._old_manual_config = None

            if hasattr(self, '_manual_online_state') and self._manual_online_state:
                manual_state = self._manual_online_state
                online_source = getattr(self, '_online_source_file', '')
                restored_input = online_source or manual_state.get('input_path_text', '')
                restored_output = manual_state.get('output_path_text', '') or seg_tab.output_path_input.text().strip()
                if hasattr(seg_tab, 'input_path_input'):
                    seg_tab.input_path_input.setText(restored_input)
                if hasattr(seg_tab, 'output_path_input'):
                    seg_tab.output_path_input.setText(restored_output)
                if getattr(seg_tab, 'config', None) is not None:
                    if restored_input:
                        seg_tab.config.inputPath = restored_input.encode('utf-8')
                    elif manual_state.get('config_input_path') is not None:
                        seg_tab.config.inputPath = manual_state['config_input_path']
                    if restored_output:
                        seg_tab.config.outputDir = restored_output.encode('utf-8')
                    elif manual_state.get('config_output_dir') is not None:
                        seg_tab.config.outputDir = manual_state['config_output_dir']
                else:
                    seg_tab._pending_input_path = restored_input or None
                self._manual_online_state = None
            
            self.online_config_folder = None
    
    def load_online_wafer_context(self, payload):
        """Push Wafer/Chip/FOV map from CT host / DB into the 3D Viewer CONTEXT panel.

        ``payload`` is a dict matching ``WaferContext.to_dict()`` (or host-equivalent
        keys). Results for the active FOV go under ``.../FOV_Pn/results``.
        """
        mpv = getattr(self, "multiplanar_tab", None)
        if mpv is None or not hasattr(mpv, "load_wafer_context"):
            return None
        ctx = mpv.load_wafer_context(payload)
        if hasattr(mpv, "set_online_context_visible"):
            mpv.set_online_context_visible(True)
        if ctx is not None and hasattr(self, "_online_append_log") and self._online_append_log:
            self._online_append_log(
                f"CONTEXT loaded: {ctx.breadcrumb()}",
                SemiconductorTheme.ACCENT_SUCCESS,
            )
        return ctx

    # ------------------------------------------------------------------
    # Online production recipe (mirrors Teaching INPUT_MODE semantics)
    # ------------------------------------------------------------------
    def _capture_online_recipe(self):
        """Snapshot Teaching input mode + layers + ROI for mass-production Online runs.

        Priority for layers:
          1. ``online_roi['layers']`` from Apply Online (locked recipe)
          2. Teaching ``layer_definitions`` when INPUT_MODE=single and layers_active
        """
        seg = self.segmentation_tab
        mode = getattr(seg, 'input_mode', 'test_1layer') or 'test_1layer'

        layers = None
        # Prefer explicitly locked Online ROI layers
        if self.online_roi_active and isinstance(self.online_roi, dict):
            locked = self.online_roi.get('layers')
            if locked:
                layers = [dict(L) for L in locked]
                # Layers imply single-volume multi-layer pipeline
                if mode == 'test_1layer':
                    mode = 'single'

        # Fallback: Teaching single-mode layer table
        if layers is None and mode == 'single' and getattr(seg, 'layers_active', False):
            selected = [dict(L) for L in seg.layer_definitions if L.get('selected')]
            if selected:
                layers = selected

        # multi_layer sep-files is offline Teaching; Online host sends one FOV volume
        if mode == 'multi_layer':
            if layers:
                mode = 'single'
            else:
                mode = 'test_1layer'

        roi_box = None
        if self.online_roi_active and isinstance(self.online_roi, dict) and 'x1' in self.online_roi:
            roi_box = {
                k: self.online_roi[k]
                for k in ('x1', 'x2', 'y1', 'y2', 'z_start', 'z_end')
                if k in self.online_roi
            }

        self._online_recipe = {
            'input_mode': mode,
            'layers_original': layers,   # Z in original / host volume coords
            'layers': layers,            # will be remapped after crop
            'roi_box': roi_box,
            'config_path': getattr(seg, 'config_path', None),
        }
        self._online_crop_meta = None
        return self._online_recipe

    @staticmethod
    def _remap_layers_for_crop(layers, z_offset, z_len):
        """Map layer Z bands from original volume into cropped runtime volume.

        After ROI Z-crop starting at ``z_offset``, layer [20,70) becomes [0,50)
        so stitch/overlay lands on the correct slices of the displayed volume.
        """
        if not layers:
            return None
        remapped = []
        for L in layers:
            try:
                zs = int(L['z_start']) - int(z_offset)
                ze = int(L['z_end']) - int(z_offset)
            except (KeyError, TypeError, ValueError):
                continue
            zs = max(0, zs)
            ze = min(int(z_len), ze)
            if ze > zs:
                entry = dict(L)
                entry['z_start'] = zs
                entry['z_end'] = ze
                remapped.append(entry)
        return remapped or None

    def _resolve_online_results_dir(self):
        """Mass-production Results path (never under ephemeral crop temp).

        Canonical host layout::

            .../ChipLocation/FOVLocation/Input.tiff
            .../ChipLocation/FOVLocation/Results/

        Order:
          1. Parsed host path ``Results`` (from last RECV)
          2. Active FOV ``results_dir`` on Wafer CONTEXT
          3. Sibling ``Results`` next to Input.tiff / FOV folder
          4. Fallback under runtime folder
        """
        # 1) Explicit host parse from this job
        host_info = getattr(self, '_online_host_path_info', None)
        if isinstance(host_info, HostVolumePathInfo) and host_info.results_dir:
            os.makedirs(host_info.results_dir, exist_ok=True)
            return host_info.results_dir

        # 2) CONTEXT FOV selection
        mpv = getattr(self, 'multiplanar_tab', None)
        if mpv is not None and hasattr(mpv, 'get_active_fov_results_dir'):
            fov_dir = mpv.get_active_fov_results_dir()
            if fov_dir:
                os.makedirs(fov_dir, exist_ok=True)
                return fov_dir

        # 3) Sibling Results next to volume (Input.tiff parent = FOV folder)
        src = getattr(self, '_online_source_file', None)
        if src:
            p = Path(src)
            base = p if p.is_dir() else p.parent
            base_s = str(base).replace('\\', '/').lower()
            if 'online_crop_' not in base_s:
                results = base / "Results"
                os.makedirs(results, exist_ok=True)
                return str(results)

        runtime = getattr(self, '_online_folder', None) or tempfile.gettempdir()
        results = os.path.join(runtime, "Results")
        os.makedirs(results, exist_ok=True)
        return results

    def _apply_host_path_to_context_ui(self, file_path: str) -> Optional[HostVolumePathInfo]:
        """Parse host Input.tiff path → select Chip + FOV on Wafer/Chip maps."""
        info = parse_host_volume_path(file_path)
        self._online_host_path_info = info

        mpv = getattr(self, 'multiplanar_tab', None)
        if mpv is None:
            return info

        # Ensure CONTEXT panel is visible during Online
        if hasattr(mpv, 'set_online_context_visible'):
            mpv.set_online_context_visible(True)

        panel = getattr(mpv, 'context_map_panel', None)
        if panel is not None and hasattr(panel, 'apply_host_path_info'):
            ctx = panel.apply_host_path_info(info)
            mpv._wafer_context = ctx
        elif hasattr(mpv, 'get_wafer_context'):
            ctx = mpv.get_wafer_context()
            if ctx is None:
                # Uninspected wafer geometry — not demo with fake NG/OK
                if hasattr(mpv, "load_wafer_context_empty"):
                    mpv.load_wafer_context_empty()
                elif hasattr(mpv, "load_wafer_context_demo"):
                    mpv.load_wafer_context_demo()
                ctx = mpv.get_wafer_context()
            if ctx is not None:
                ctx.apply_host_path(info)
                if hasattr(mpv, 'load_wafer_context'):
                    mpv.load_wafer_context(ctx)

        if hasattr(self, '_online_append_log') and self._online_append_log:
            if info.parsed_ok:
                self._online_append_log(
                    f"[CONTEXT] {info.breadcrumb()}",
                    SemiconductorTheme.ACCENT_PRIMARY,
                )
                self._online_append_log(
                    f"[PATH] Input  → {info.input_path or file_path}",
                    SemiconductorTheme.TEXT_SECONDARY,
                )
                self._online_append_log(
                    f"[PATH] Results→ {info.results_dir}",
                    SemiconductorTheme.TEXT_SECONDARY,
                )
            else:
                self._online_append_log(
                    f"[CONTEXT] Path not fully parsed (using FOV parent/Results): {file_path}",
                    SemiconductorTheme.ACCENT_WARNING,
                )
        return info

    def _update_context_judge_from_stats(self, mpv, stats_text: str = ""):
        """Push FOV OK/NG onto Wafer map after measurement (bin 8 / 1)."""
        info = getattr(self, '_online_host_path_info', None)
        ctx = None
        if hasattr(mpv, 'get_wafer_context'):
            ctx = mpv.get_wafer_context()
        if ctx is None:
            return

        col = row = fov = 0
        if isinstance(info, HostVolumePathInfo) and info.chip_col > 0:
            col, row = info.chip_col, info.chip_row
            fov = info.fov_index or ctx.selected_fov
        else:
            col, row = ctx.selected_col, ctx.selected_row
            fov = ctx.selected_fov
        if col <= 0 or row <= 0 or fov <= 0:
            return

        # NG if any object failed NG threshold (same rule as stats summary)
        ng_count = 0
        if getattr(mpv, 'object_stats', None):
            ng_threshold = 0.05
            seg = getattr(self, 'segmentation_tab', None)
            if seg is not None and hasattr(seg, 'ng_threshold_spin'):
                ng_threshold = seg.ng_threshold_spin.value() / 100.0
            ng_count = sum(
                1 for s in mpv.object_stats
                if (s.get('ratio', 0) >= ng_threshold if s.get('ratio') != float('inf') else True)
            )
        judge = BIN_NG if ng_count > 0 else BIN_OK
        ct = "NG" if judge == BIN_NG else "OK"
        ctx.set_fov_judge(col, row, fov, judge, ct_result=ct)

        panel = getattr(mpv, 'context_map_panel', None)
        if panel is not None and hasattr(panel, 'refresh'):
            panel.refresh()
        elif panel is not None:
            panel.set_context(ctx)

        # Chip FinalBin after rollup (wafer color). Pending until 9/9 OK or any NG.
        cell = ctx.chip_at(col, row)
        chip_bin = int(cell.final_bin) if cell is not None else BIN_PENDING
        if chip_bin == BIN_NG:
            chip_txt = "ChipFinal=NG"
        elif chip_bin == BIN_OK:
            chip_txt = "ChipFinal=OK (9/9)"
        else:
            g = int(cell.good_count) if cell is not None else 0
            b = int(cell.bad_count) if cell is not None else 0
            chip_txt = f"ChipFinal=Pend ({g}/9 OK, {b} NG) — wafer stays Pending until 9/9"

        if hasattr(self, '_online_append_log') and self._online_append_log:
            self._online_append_log(
                f"[MAP] Chip ({col},{row}) FOV P{fov} → {ct}  "
                f"(NG objs={ng_count})  ·  {chip_txt}",
                SemiconductorTheme.ACCENT_ERROR if judge == BIN_NG else SemiconductorTheme.ACCENT_SUCCESS,
            )

    @staticmethod
    def _align_mask_to_volume(mask, volume_shape):
        """Pad/crop mask to volume (Z,Y,X) so overlay never shape-mismatches."""
        if mask is None:
            return None
        arr = np.asarray(mask)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        elif arr.ndim == 4:
            arr = arr[:, :, :, 0]
        if arr.ndim != 3:
            return None
        out = np.zeros(volume_shape[:3], dtype=np.uint8)
        z = min(arr.shape[0], volume_shape[0])
        y = min(arr.shape[1], volume_shape[1])
        x = min(arr.shape[2], volume_shape[2])
        if z > 0 and y > 0 and x > 0:
            out[:z, :y, :x] = arr[:z, :y, :x]
        return out

    def _runtime_layers_for_seg(self):
        """Layers in runtime volume Z coords (after optional crop remap)."""
        recipe = getattr(self, '_online_recipe', None) or {}
        mode = recipe.get('input_mode', 'test_1layer')
        if mode != 'single':
            return None
        layers = recipe.get('layers')
        return layers if layers else None

    def on_online_folder_received(self, file_path):
        """Handle file path from Server.cpp — production FOV inspection pipeline.

        Config / INPUT_MODE / layers are locked when Online was turned ON
        (see ``_capture_online_recipe``).

        Flow:
          0. Resolve volume files from host path
          1. Optional ROI crop (XY/Z) + remap LAYER Z bands into crop space
          2. Load volume into 3D Viewer (operator sees data immediately)
          3. Optional enhancement → DLL segmentation (test_1layer or single split)
          4. Stitch masks → overlay + stats → save Results (FOV or source folder)
          5. Ready for next host path (keep DLL + config)
        """
        if self._online_processing:
            self.status_label.setText("ONLINE: Still processing previous file, skipping...")
            if hasattr(self, '_online_append_log') and self._online_append_log:
                self._online_append_log(
                    f"[SKIP] Busy — ignored: {file_path}", SemiconductorTheme.ACCENT_WARNING
                )
            return
        
        self._online_processing = True
        self._online_crop_meta = None
        self._online_host_path_info = None
        self._online_walltime_begin(file_path)
        self.status_label.setText(f"ONLINE: Processing {file_path}...")
        # Mark pipeline start immediately (secondary FILE RECEIVED log used to
        # appear only after this handler returned — after a slow temp copy).
        if hasattr(self, "_online_append_log") and self._online_append_log:
            self._online_append_log(
                f"[PIPELINE] start: {file_path}",
                SemiconductorTheme.TEXT_SECONDARY,
            )
        if hasattr(self, '_online_append_log') and self._online_append_log:
            self._online_append_log(f"[RECV] {file_path}", SemiconductorTheme.ACCENT_SUCCESS)

        # ---- Link host path → Wafer Map + Chip FOV UI + Results dir ----
        # Canonical:
        #   yy_mm_dd/LotID_FoupID/WaferID/ChipLoc/FOVLoc/Input.tiff
        #   → Results: .../FOVLoc/Results
        host_info = self._apply_host_path_to_context_ui(file_path)

        # Prefer resolved Input.tiff from parser when host sent FOV folder
        resolved_input = file_path
        if host_info and host_info.input_path:
            resolved_input = host_info.input_path
        
        # Server sends a file path - extract folder from it
        input_file = Path(resolved_input)
        if not input_file.exists():
            # Try as folder path (backward compat)
            folder = input_file
            if not folder.is_dir():
                # Still allow process if original path exists
                alt = Path(file_path)
                if alt.exists():
                    input_file = alt
                    folder = alt if alt.is_dir() else alt.parent
                else:
                    self.status_label.setText(f"ONLINE ERROR: Path not found: {file_path}")
                    self._online_processing = False
                    return
            self._online_source_file = str(folder) if folder.is_dir() else str(input_file)
        else:
            if input_file.is_dir():
                folder = input_file
                self._online_source_file = str(folder)
            else:
                folder = input_file.parent
                self._online_source_file = str(input_file)

        # Persistent FOV folder (parent of Input.tiff) — before temp crop rewrites
        self._online_source_folder = str(folder)
        if host_info and host_info.fov_dir:
            self._online_source_folder = host_info.fov_dir
        
        # ---- Find all volume .tif files in the folder ----
        # Prefer single Input.tiff when present (host mass-production layout)
        input_candidates = []
        for cand_name in ("Input.tiff", "Input.tif", "input.tiff", "input.tif"):
            c = Path(folder) / cand_name
            if c.is_file():
                input_candidates.append(c)
                break

        if input_file.exists() and input_file.is_file() and input_file.suffix.lower() in ('.tif', '.tiff'):
            volume_files = [input_file]
        elif input_candidates:
            volume_files = input_candidates
            self._online_source_file = str(input_candidates[0])
        else:
            volume_files = sorted([
                f for f in Path(folder).iterdir()
                if f.suffix.lower() in ('.tif', '.tiff')
                and '_config' not in f.name
                and 'bump3D' not in f.name
                and 'voidsOnly' not in f.name
                and 'voidsInBumps' not in f.name
                and f.parent.name.lower() != 'results'
            ]) if Path(folder).is_dir() else []
        
        if not volume_files:
            self.status_label.setText(f"ONLINE ERROR: No TIFF files in {folder}")
            self._online_processing = False
            return

        # Sort volume files by slice number (2D sequences)
        def extract_slice_number(filepath):
            match = re.search(r'slice(\d+)', filepath.name, re.IGNORECASE)
            return int(match.group(1)) if match else 0
        volume_files.sort(key=extract_slice_number)

        # Ensure recipe exists (toggle ON should have set it)
        if not getattr(self, '_online_recipe', None):
            self._capture_online_recipe()
        # Reset runtime layers to original Z (before this FOV's crop)
        recipe = self._online_recipe
        recipe['layers'] = (
            [dict(L) for L in recipe['layers_original']]
            if recipe.get('layers_original') else None
        )
        
        # ---- Step 0: Crop for Online mode if ROI is set ----
        roi_box = recipe.get('roi_box')
        if not roi_box and self.online_roi_active and self.online_roi and 'x1' in self.online_roi:
            roi_box = self.online_roi

        if roi_box and 'x1' in roi_box:
            start_crop = time.perf_counter()
            self._online_append_log("[CROPPING] Applying locked ROI to incoming data...", "#FFA000")
            
            cropped_folder, crop_meta = self._crop_volume_for_online(volume_files, roi_box)
            if cropped_folder:
                folder = Path(cropped_folder)
                volume_files = sorted([
                    f for f in folder.iterdir()
                    if f.suffix.lower() in ('.tif', '.tiff')
                ])
                volume_files.sort(key=extract_slice_number)
                self._online_crop_meta = crop_meta
                crop_time = time.perf_counter() - start_crop
                self._online_walltime_phase("roi_crop", elapsed_sec=crop_time)
                self._online_append_log(
                    f"[CROP DONE] {len(volume_files)} file(s) ({crop_time:.2f}s) "
                    f"offset Z0={crop_meta.get('z0', 0)}",
                    "#00BFA5",
                )
                # Remap LAYER Z into cropped volume coordinates
                if recipe.get('layers_original') and crop_meta is not None:
                    z_len = crop_meta.get('z_len')
                    if z_len is None and crop_meta.get('z1') is not None:
                        z_len = int(crop_meta['z1']) - int(crop_meta['z0'])
                    remapped = self._remap_layers_for_crop(
                        recipe['layers_original'],
                        z_offset=crop_meta.get('z0', 0),
                        z_len=z_len if z_len is not None else 10**9,
                    )
                    recipe['layers'] = remapped
                    n = len(remapped) if remapped else 0
                    self._online_append_log(
                        f"[LAYERS] Remapped {n} bands into crop Z-space",
                        "#00BFA5",
                    )
            else:
                crop_time = time.perf_counter() - start_crop
                self._online_walltime_phase(
                    "roi_crop",
                    elapsed_sec=crop_time,
                    note="fallback_full_volume",
                )
                self._online_append_log(
                    "[CROP FAILED] Proceeding with full volume as fallback",
                    SemiconductorTheme.ACCENT_ERROR,
                )
        else:
            self._online_walltime_phase("roi_crop", elapsed_sec=0.0, note="skipped")
        
        # Runtime working folder (may be temp crop)
        self._online_folder = str(folder)
        self._online_volume_files = volume_files
        self._online_results_dir = self._resolve_online_results_dir()
        self._online_append_log(
            f"[RESULTS] → {self._online_results_dir}", SemiconductorTheme.TEXT_SECONDARY
        )
        
        # ---- Step 1: Cleanup + load volume into Viewer (MPR first, no 3D) ----
        self._online_preflight_cleanup_for_new_volume()

        mpv = self.multiplanar_tab
        # Online: keep GPU 3D OFF so first paint is MPR-only (fast).
        if hasattr(mpv, "set_3d_volume_render_enabled"):
            try:
                mpv.set_3d_volume_render_enabled(False, sync_button=True)
            except Exception:
                pass

        # Prefer direct path — avoid Windows symlink-fail → full-file copy2 (10GB+).
        used_crop_temp = "online_crop_" in str(folder)
        self._online_temp_dir = None
        if used_crop_temp:
            # Cropped multi-slice stack lives in temp folder — load that folder.
            load_path = str(folder)
            load_note = "cropped folder"
        elif len(volume_files) == 1:
            load_path = str(volume_files[0])
            load_note = "direct file (no temp copy)"
        else:
            # Original multi-file stack: LoadVolumeThread supports directory scan.
            load_path = str(folder)
            load_note = "direct folder (no temp copy)"

        self._online_runtime_input_file = load_path
        if hasattr(self, "_online_append_log") and self._online_append_log:
            self._online_append_log(
                f"[LOAD] {load_note}: {load_path}",
                SemiconductorTheme.TEXT_SECONDARY,
            )

        # Compact stage progress — best-effort UI; must never block the pipeline
        try:
            self._open_online_progress("load")
        except Exception as e:
            print(f"[ONLINE] open progress failed: {e}")
            self._online_progress = None

        from inno3d.core.volume_store import large_volume_engine_enabled

        wt = getattr(self, "_online_mpr_walltime", None)
        if wt is not None:
            wt.set_meta("load_path", str(load_path))
            wt.set_meta("load_note", load_note)
            wt.set_meta("crop_used", bool(used_crop_temp))
            wt.set_meta("lve_enabled", bool(large_volume_engine_enabled(mpv)))
            mpv._online_mpr_walltime = wt

        mpv.load_thread = LoadVolumeThread(
            load_path,
            downsample_factor=1,
            use_large_volume_engine=large_volume_engine_enabled(mpv),
        )
        mpv.load_thread.progress.connect(
            lambda v, _m: self._tick_online_progress(v)
        )
        mpv.load_thread.finished.connect(self._on_online_volume_loaded)
        if wt is not None:
            wt.begin_load_thread()
        mpv.load_thread.start()
    
    def _on_online_volume_loaded(self, data, error):
        """Volume loaded into Viewer — paint MPR ASAP, then continue pipeline."""
        mpv = self.multiplanar_tab
        wt = getattr(self, "_online_mpr_walltime", None)
        if wt is not None:
            wt.mark("on_online_volume_loaded.start")
            thr = getattr(mpv, "load_thread", None)
            decode_s = getattr(thr, "decode_walltime_s", None)
            if decode_s is None:
                decode_s = getattr(thr, "walltime_total_sec", None)
            wt.end_load_thread_callback(decode_s)
            cb_s = wt.get_duration("load_volume_thread.callback_wall")
            if cb_s is not None and hasattr(self, "_online_append_log") and self._online_append_log:
                self._online_append_log(
                    f"[LOAD] decode finished in {cb_s:.1f}s",
                    SemiconductorTheme.TEXT_SECONDARY,
                )

        with online_walltime_phase(wt, "on_online_volume_loaded.handler"):
            if data is not None:
                # on_volume_loaded expects mpv.progress to exist - create a dummy one
                # (it will be closed by on_volume_loaded; _online_progress stays open for segmentation)
                mpv.progress = QProgressDialog("Loading...", None, 0, 100, self)
                mpv.progress.close()  # Hide it immediately - we use _online_progress instead
                self._tick_online_progress(100)

                # Keep 3D OFF for Online first paint (MPR only).
                if hasattr(mpv, "set_3d_volume_render_enabled"):
                    try:
                        mpv.set_3d_volume_render_enabled(False, sync_button=True)
                    except Exception:
                        pass

                # Display volume in Viewer (MPR). processEvents lets panes paint before Teaching sync.
                mpv.on_volume_loaded(data, error)
                try:
                    QApplication.processEvents()
                except Exception:
                    pass
                self._online_walltime_finalize_first_mpr()

                # Also set in segmentation tab for pipeline / manual cropping
                src_label = getattr(self, '_online_source_file', None) or (
                    str(self._online_volume_files[0])
                    if getattr(self, '_online_volume_files', None) else None
                )
                self.segmentation_tab.set_volume_data(
                    getattr(mpv, "volume_data", data),
                    src_label,
                )

                # Clamp/remap layers to actual loaded Z depth (safety after crop)
                if mpv.volume_data is not None:
                    z, y, x = mpv.volume_data.shape[:3]
                    recipe = getattr(self, '_online_recipe', None) or {}
                    layers = recipe.get('layers')
                    if layers:
                        clamped = []
                        for L in layers:
                            zs = max(0, min(int(L['z_start']), z))
                            ze = max(0, min(int(L['z_end']), z))
                            if ze > zs:
                                e = dict(L)
                                e['z_start'], e['z_end'] = zs, ze
                                clamped.append(e)
                        recipe['layers'] = clamped or None
                        if hasattr(self, '_online_append_log') and self._online_append_log:
                            self._online_append_log(
                                f"[VOLUME] {z}×{y}×{x}  |  runtime layers: "
                                f"{len(recipe['layers'] or [])}",
                                SemiconductorTheme.TEXT_SECONDARY,
                            )
                    # Sliders/slices already set by Viewer.on_volume_loaded → update_all_views
            elif error:
                self._online_walltime_abort(str(error))
                self.status_label.setText(f"ONLINE ERROR (volume): {error}")
                if hasattr(self, '_online_append_log') and self._online_append_log:
                    self._online_append_log(f"[ERROR] volume load: {error}", SemiconductorTheme.ACCENT_ERROR)
                self._close_online_progress(ok=False)
                self._online_cleanup()
                return

        # ---- Step 2: Enhancement (if enabled) → then Segmentation ----
        self._online_apply_dll_profiling()
        self._online_reset_profiling_report()
        if getattr(self, '_online_enhancement_enabled', False) and self._online_enhancement_model:
            self._run_online_enhancement()
        else:
            self._start_online_segmentation()
    
    def _get_online_volume_for_enhance(self):
        """Return (Z,Y,X) ndarray already loaded for this FOV, or None."""
        for src in (
            getattr(self.multiplanar_tab, "volume_data", None),
            getattr(self.segmentation_tab, "volume_data", None),
        ):
            if src is None:
                continue
            try:
                # VolumeStore proxy must be explicitly materialized for enhancement.
                if hasattr(src, "store") and hasattr(src.store, "get_level"):
                    arr = np.asarray(src.store.get_level(0))
                elif hasattr(src, "get_level"):
                    arr = np.asarray(src.get_level(0))
                else:
                    arr = np.asarray(src)
            except Exception:
                continue
            if arr.ndim == 2:
                arr = arr[np.newaxis]
            elif arr.ndim == 4:
                arr = arr[:, :, :, 0]
            if arr.ndim == 3 and arr.shape[0] > 0:
                return arr
        return None

    def _export_volume_slices_for_enhance(self, volume, dest_dir):
        """Write volume[Z,Y,X] as 0000.tif … for BumpVoid_ISP_ENH (2D-per-file).

        The enhancement DLL only reads single-plane TIFFs — multipage Input.tiff
        would otherwise become one 2D frame and destroy Z for layer split.
        """
        import tifffile

        os.makedirs(dest_dir, exist_ok=True)
        # Clear previous run leftovers
        for old in Path(dest_dir).glob("*.tif*"):
            try:
                old.unlink()
            except Exception:
                pass

        z = int(volume.shape[0])
        for i in range(z):
            path = os.path.join(dest_dir, f"{i:04d}.tif")
            tifffile.imwrite(path, np.ascontiguousarray(volume[i]))
        return z

    def _run_online_enhancement(self):
        """Run volume enhancement (ONNX DLL) before segmentation.

        Prefer in-memory ``ProcessVolumeBuffer`` (BumpVoid_ISP_ENH >= 0.0.1)
        so Online does not export thousands of temp TIFF slices. Fall back to
        per-slice folder export for older DLLs (multipage Input.tiff is not
        a valid folder input for the DLL).
        """
        import time as _time

        self._online_enhance_start_time = _time.time()

        volume = self._get_online_volume_for_enhance()
        if volume is None:
            self._online_append_log(
                "[ENHANCE FAILED] No volume in memory to enhance — using raw",
                SemiconductorTheme.ACCENT_ERROR,
            )
            self._start_online_segmentation()
            return

        volume = np.asarray(volume)
        n_slices = int(volume.shape[0])
        self._online_enhance_n_slices = n_slices

        # Enhanced output under stable results tree (archive / fallback assemble)
        results_root = getattr(self, "_online_results_dir", None) or self._resolve_online_results_dir()
        enhanced_output = os.path.join(results_root, "enhanced_volume")
        if os.path.isdir(enhanced_output):
            try:
                shutil.rmtree(enhanced_output)
            except Exception:
                pass
        os.makedirs(enhanced_output, exist_ok=True)
        self._online_enhanced_output = enhanced_output

        start_slice = getattr(self, "_online_enhancement_start", -1)
        end_slice = getattr(self, "_online_enhancement_end", -1)
        mode = getattr(self, "_online_enhancement_mode", "full")

        # Clamp layer range to actual Z
        if start_slice >= 0 and end_slice >= 0:
            start_slice = max(0, min(start_slice, n_slices))
            end_slice = max(0, min(end_slice, n_slices))
            if end_slice <= start_slice:
                start_slice, end_slice = -1, -1
                mode = "full"

        # Probe DLL for buffer API (load only — init happens in worker thread)
        use_buffer = False
        seg_tab = self.segmentation_tab
        try:
            from inno3d.core import enhanced_volume as _ev
            try:
                _ev.load_dll(seg_tab.dll_path)
            except Exception:
                pass
            use_buffer = bool(
                hasattr(_ev, "has_process_volume_buffer")
                and _ev.has_process_volume_buffer()
            )
        except Exception:
            use_buffer = False

        input_dir = None
        volume_u16 = None
        if use_buffer:
            # In-memory: DLL copies non-enhanced Z itself for layers mode
            self._online_enhance_base_volume = None
            volume_u16 = np.ascontiguousarray(volume, dtype=np.uint16)
            path_note = "in-memory buffer (no temp slice export)"
        else:
            # Legacy DLL: keep original for merge when mode=layers
            self._online_enhance_base_volume = np.asarray(volume).copy()
            base_temp = getattr(self, "_online_temp_dir", None) or tempfile.mkdtemp(
                prefix="online_enh_"
            )
            self._online_temp_dir = base_temp
            input_dir = os.path.join(base_temp, "enhance_slices")
            n_slices = self._export_volume_slices_for_enhance(volume, input_dir)
            self._online_enhance_n_slices = n_slices
            path_note = f"temp slice folder ({n_slices} files)"

        self._online_append_log("[ENHANCE] Starting ONNX enhancement...", "#CE93D8")
        self._online_append_log(
            f"[ENHANCE] Volume {volume.shape[0]}×{volume.shape[1]}×{volume.shape[2]} "
            f"— {path_note}",
            SemiconductorTheme.TEXT_SECONDARY,
        )
        self._online_append_log(f"[ENHANCE] Mode: {mode}", "#CE93D8")
        if start_slice >= 0 and end_slice >= 0:
            self._online_append_log(
                f"[ENHANCE] Slice range: [{start_slice}, {end_slice}) (selected layers)",
                SemiconductorTheme.TEXT_SECONDARY,
            )
        if input_dir:
            self._online_append_log(f"[ENHANCE] Input:  {input_dir}", SemiconductorTheme.TEXT_SECONDARY)
        else:
            self._online_append_log("[ENHANCE] Input:  RAM volume buffer", SemiconductorTheme.TEXT_SECONDARY)
        self._online_append_log(f"[ENHANCE] Output: {enhanced_output}", SemiconductorTheme.TEXT_SECONDARY)

        self._set_online_stage("enhance", 0)

        self._online_enhancement_thread = EnhancementThread(
            dll_dir=seg_tab.dll_path,
            model_path=self._online_enhancement_model,
            trt_cache_path=self._online_enhancement_trt_cache,
            use_gpu=self._online_enhancement_use_gpu,
            gpu_device_id=self._online_enhancement_gpu_id,
            input_dir=input_dir,
            output_dir=enhanced_output,
            start_slice=start_slice,
            end_slice=end_slice,
            volume_u16=volume_u16,
        )
        # Percent only — never surface model/CUDA/path detail to the dialog
        self._online_enhancement_thread.progress.connect(
            lambda v, _m: self._tick_online_progress(v)
        )
        self._online_enhancement_thread.finished.connect(self._on_online_enhancement_finished)
        self._online_enhancement_thread.start()

    def _assemble_enhanced_volume_stack(self, enhanced_dir):
        """Build full (Z,Y,X) stack from enhanced slice files + optional base volume.

        - full mode: stack all ``####.tif`` in order
        - layers mode: start from pre-enhance volume, overwrite enhanced Z only
          so LAYER_* Z indices still match the full FOV.
        """
        import tifffile

        enhanced_dir = Path(enhanced_dir)
        files = sorted(
            [f for f in enhanced_dir.iterdir() if f.suffix.lower() in (".tif", ".tiff")],
            key=lambda p: p.name,
        )
        if not files:
            return None, 0

        # Parse index from 0000.tif style; fallback to enumeration
        def _idx(p):
            stem = p.stem
            try:
                return int(stem)
            except ValueError:
                return None

        indexed = []
        for f in files:
            i = _idx(f)
            if i is not None:
                indexed.append((i, f))
            else:
                indexed.append((len(indexed), f))

        base = getattr(self, "_online_enhance_base_volume", None)
        n_expected = int(getattr(self, "_online_enhance_n_slices", 0) or 0)
        if base is not None:
            stack = np.asarray(base).copy()
            if stack.ndim == 2:
                stack = stack[np.newaxis]
            n_expected = stack.shape[0]
        else:
            # pure enhanced stack
            max_i = max(i for i, _ in indexed)
            n_expected = max(n_expected, max_i + 1, len(indexed))
            first = np.asarray(tifffile.imread(str(indexed[0][1])))
            if first.ndim > 2:
                first = first[0] if first.ndim == 3 else first[..., 0]
            stack = np.zeros((n_expected,) + first.shape, dtype=first.dtype)

        n_applied = 0
        for i, fpath in indexed:
            if i < 0 or i >= stack.shape[0]:
                continue
            sl = np.asarray(tifffile.imread(str(fpath)))
            if sl.ndim == 3:
                sl = sl[0]
            elif sl.ndim == 4:
                sl = sl[0, ..., 0] if sl.shape[-1] <= 4 else sl[0]
            if sl.shape != stack.shape[1:]:
                # skip mismatched geometry
                continue
            stack[i] = sl
            n_applied += 1

        return stack, n_applied

    def _on_online_enhancement_finished(self, success, error):
        """Enhancement done — rebuild full Z volume then run segmentation."""
        import time as _time
        import tifffile

        enhance_time = (
            _time.time() - self._online_enhance_start_time
            if hasattr(self, "_online_enhance_start_time")
            else 0
        )
        self._online_last_enhance_sec = float(enhance_time or 0)
        self._online_enhance_end_time = _time.time()

        if success:
            self._online_append_log(
                f"[ENHANCE DONE] Volume enhanced successfully ({enhance_time:.1f}s)",
                SemiconductorTheme.ACCENT_SUCCESS,
            )
            try:
                from inno3d.core import enhanced_volume
                self._online_log_dll_timing("ENH", enhanced_volume.get_last_timing)
            except Exception:
                pass

            enhanced_dir = self._online_enhanced_output
            enh_stack, n_applied = None, 0
            thr = getattr(self, "_online_enhancement_thread", None)
            if thr is not None and getattr(thr, "enhanced_volume", None) is not None:
                enh_stack = thr.enhanced_volume
                n_applied = int(getattr(self, "_online_enhance_n_slices", 0) or enh_stack.shape[0])
                if getattr(thr, "used_buffer_api", False):
                    self._online_append_log(
                        "[ENHANCE] Used in-memory ProcessVolumeBuffer (no temp slices)",
                        SemiconductorTheme.TEXT_SECONDARY,
                    )
            else:
                try:
                    enh_stack, n_applied = self._assemble_enhanced_volume_stack(enhanced_dir)
                except Exception as e:
                    enh_stack, n_applied = None, 0
                    self._online_append_log(
                        f"[ENHANCE WARN] assemble failed: {e}",
                        SemiconductorTheme.ACCENT_WARNING,
                    )

            if enh_stack is not None and enh_stack.shape[0] > 0:
                z, y, x = enh_stack.shape[:3]
                self._online_append_log(
                    f"[ENHANCE] Stack {z}×{y}×{x}  ({n_applied} slice file(s) applied)",
                    "#CE93D8",
                )

                # Write multipage TIFF for DLL segmentation path (single file, full Z)
                results_root = (
                    getattr(self, "_online_results_dir", None)
                    or self._resolve_online_results_dir()
                )
                multipage_path = os.path.join(results_root, "enhanced_volume", "Enhanced_Volume.tif")
                try:
                    tifffile.imwrite(multipage_path, enh_stack, imagej=True)
                    self._online_runtime_input_file = multipage_path
                    self._online_volume_files = [Path(multipage_path)]
                except Exception as e:
                    self._online_append_log(
                        f"[ENHANCE WARN] multipage write failed: {e}",
                        SemiconductorTheme.ACCENT_WARNING,
                    )
                    # fallback: keep slice folder; point at first slice + set volume in memory
                    self._online_runtime_input_file = str(
                        sorted(Path(enhanced_dir).glob("*.tif"))[0]
                    ) if list(Path(enhanced_dir).glob("*.tif")) else self._online_runtime_input_file

                # Viewer + Teaching tab must use full enhanced Z (layer split needs it)
                try:
                    mpv = self.multiplanar_tab
                    mpv.progress = QProgressDialog("Loading...", None, 0, 100, self)
                    mpv.progress.close()
                    mpv.on_volume_loaded(enh_stack, None)
                    self.segmentation_tab.set_volume_data(
                        enh_stack, getattr(self, "_online_source_file", None)
                    )
                    # Re-clamp layers to enhanced Z (should match original)
                    recipe = getattr(self, "_online_recipe", None) or {}
                    layers = recipe.get("layers")
                    if layers:
                        clamped = []
                        for L in layers:
                            zs = max(0, min(int(L["z_start"]), z))
                            ze = max(0, min(int(L["z_end"]), z))
                            if ze > zs:
                                e = dict(L)
                                e["z_start"], e["z_end"] = zs, ze
                                clamped.append(e)
                        recipe["layers"] = clamped or None
                except Exception as e:
                    self._online_append_log(
                        f"[ENHANCE WARN] could not refresh Viewer: {e}",
                        SemiconductorTheme.ACCENT_WARNING,
                    )
            else:
                self._online_append_log(
                    "[ENHANCE WARNING] No enhanced stack — using original volume",
                    SemiconductorTheme.ACCENT_WARNING,
                )
        else:
            self._online_append_log(
                f"[ENHANCE FAILED] {error} — falling back to raw volume",
                SemiconductorTheme.ACCENT_ERROR,
            )

        # Free large base copy
        if hasattr(self, "_online_enhance_base_volume"):
            self._online_enhance_base_volume = None

        # Proceed to segmentation regardless
        self._start_online_segmentation()
    
    def _online_dll_profiling_enabled(self) -> bool:
        """Read ENABLE_DLL_PROFILING from Teaching Params / config checkbox."""
        seg_tab = getattr(self, "segmentation_tab", None)
        if seg_tab is None:
            return False
        try:
            if hasattr(seg_tab, "_is_dll_profiling_enabled"):
                return bool(seg_tab._is_dll_profiling_enabled())
        except Exception:
            pass
        return bool(getattr(seg_tab, "_enable_dll_profiling", False))

    def _online_apply_dll_profiling(self) -> None:
        """Re-push SetProfiling after Online DLL reload (flag resets on reload)."""
        seg_tab = getattr(self, "segmentation_tab", None)
        if seg_tab is not None and hasattr(seg_tab, "_apply_dll_profiling_flag"):
            try:
                seg_tab._apply_dll_profiling_flag()
                return
            except Exception:
                pass
        # Fallback: apply directly from checkbox state
        enabled = self._online_dll_profiling_enabled()
        import logging
        applied = []
        try:
            from inno3d.core import bumpvoid
            if bumpvoid.set_profiling(enabled):
                applied.append("SEG")
        except Exception:
            pass
        try:
            from inno3d.core import bumpvoid_mes
            if bumpvoid_mes.set_profiling(enabled):
                applied.append("MES")
        except Exception:
            pass
        try:
            from inno3d.core import bumpvoid_b2b
            if bumpvoid_b2b.set_profiling(enabled):
                applied.append("B2B")
        except Exception:
            pass
        msg = (
            f"[DLL Profiling] ENABLED → {', '.join(applied)}"
            if enabled
            else "[DLL Profiling] disabled"
        )
        logging.getLogger("DLL_PROF").info(msg)
        print(msg)

    def _online_reset_profiling_report(self) -> None:
        seg_tab = self.segmentation_tab
        if seg_tab is None or not hasattr(seg_tab, "_reset_profiling_report"):
            return
        lines = []
        host_info = getattr(self, "_online_host_path_info", None)
        if host_info is not None and hasattr(host_info, "breadcrumb"):
            try:
                lines.append(f"Context: {host_info.breadcrumb()}")
            except Exception:
                pass
        rd = getattr(self, "_online_results_dir", None)
        if rd:
            lines.append(f"Results: {rd}")
        recipe = getattr(self, "_online_recipe", None) or {}
        mode = recipe.get("input_mode", "test_1layer")
        layers = recipe.get("layers") or []
        lines.append(f"INPUT_MODE: {mode}")
        if layers:
            lines.append(f"Layers: {len(layers)}")
        src = getattr(self, "_online_source_file", None) or getattr(self, "_online_runtime_input_file", None)
        if src:
            lines.append(f"Runtime input: {src}")

        mpv = getattr(self, "multiplanar_tab", None)
        vol = None
        if mpv is not None:
            vol = getattr(mpv, "volume_data", None)
        if vol is None and seg_tab is not None:
            vol = getattr(seg_tab, "volume_data", None)
        if vol is not None and seg_tab is not None:
            seg_tab.volume_data = vol

        gpu_id = getattr(self, "_online_enhancement_gpu_id", 0)
        if seg_tab is not None and hasattr(seg_tab, "enh_gpuid_spin") and seg_tab.enh_gpuid_spin is not None:
            try:
                gpu_id = int(seg_tab.enh_gpuid_spin.value())
            except Exception:
                pass

        from inno3d.core.dll_profiling import build_profiling_context_lines

        builder = seg_tab._get_profiling_report_builder()
        builder.reset()
        ctx = build_profiling_context_lines(
            volume=vol,
            spacing=seg_tab._profiling_voxel_spacing() if hasattr(seg_tab, "_profiling_voxel_spacing") else None,
            gpu_device_id=int(gpu_id or 0),
            extra_lines=lines,
        )
        builder.set_context(ctx, gpu_device_id=int(gpu_id or 0))

    def _online_log_dll_timing(self, module: str, getter, note=None) -> None:
        """Write stage timing into Inno3D_Logs + Online UI log."""
        import logging
        log = logging.getLogger("DLL_PROF")
        if not self._online_dll_profiling_enabled():
            return
        try:
            from inno3d.core.dll_profiling import format_timing_banner
            info = getter()
            seg_tab = self.segmentation_tab
            if seg_tab is not None and hasattr(seg_tab, "_record_profiling_section"):
                banner = seg_tab._record_profiling_section(module, info, note=note)
            else:
                banner = format_timing_banner(module, info)
            for line in banner.splitlines():
                log.info(line)
            print(banner)
            if hasattr(self, "_online_append_log") and self._online_append_log:
                # One compact line in Online panel; full report in file log
                if info and info.get("total_sec") is not None:
                    self._online_append_log(
                        f"[{module} PROF] total={info.get('total_sec', 0):.3f}s "
                        f"load={info.get('load_io_sec', 0):.3f}s "
                        f"algo={info.get('algo_sec', 0):.3f}s "
                        f"save={info.get('save_io_sec', 0):.3f}s "
                        f"h2d={info.get('h2d_sec', 0):.3f}s "
                        f"d2h={info.get('d2h_sec', 0):.3f}s",
                        "#80CBC4",
                    )
                else:
                    self._online_append_log(
                        f"[{module} PROF] (no timing detail — see Inno3D_Logs DLL_PROF)",
                        SemiconductorTheme.ACCENT_WARNING,
                    )
        except Exception as e:
            log.error("[%s] Online profiling read failed: %s", module, e)

    def _start_online_segmentation(self):
        """Start DLL segmentation in background (called after optional enhancement).

        Respects Teaching INPUT_MODE recipe:
          - test_1layer  → layers=None (one DLL pass on full runtime volume)
          - single       → layers=runtime Z bands (split → stitch)
        """
        self._online_apply_dll_profiling()
        seg_tab = self.segmentation_tab
        first_volume = (
            self._online_runtime_input_file
            if hasattr(self, '_online_runtime_input_file')
            else str(self._online_volume_files[0])
        )

        self._set_online_stage("segment", 0)
        
        # Stable Results dir (FOV context or source folder — not temp crop)
        results_dir = getattr(self, '_online_results_dir', None) or self._resolve_online_results_dir()
        self._online_results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        recipe = getattr(self, '_online_recipe', None) or {}
        mode = recipe.get('input_mode', 'test_1layer')
        runtime_layers = self._runtime_layers_for_seg()

        if runtime_layers:
            self._online_append_log(
                f"[SEG] INPUT_MODE={mode}  ·  {len(runtime_layers)} layer(s) split",
                "#FFA000",
            )
        else:
            self._online_append_log(
                f"[SEG] INPUT_MODE={mode}  ·  full-volume single pass",
                "#FFA000",
            )

        # Prefer volume already in Viewer / Teaching (avoid second disk load)
        mpv = self.multiplanar_tab
        vol = getattr(mpv, 'volume_data', None)
        if vol is None:
            vol = getattr(seg_tab, 'volume_data', None)

        # Run segmentation (hidden — no tab switch, pre-loaded config)
        seg_tab.set_paths_and_run(
            input_path=first_volume,
            output_path=results_dir,
            on_finished_callback=self._on_online_inspection_finished,
            layers=runtime_layers,
            volume_data=vol,
        )
        
        # Percent only — never surface DLL step names / thresholds to the dialog
        if seg_tab.inspection_thread:
            seg_tab.inspection_thread.progress.connect(
                lambda v, _m: self._tick_online_progress(v)
            )
    
    def _on_online_inspection_finished(self, result, error):
        """Segmentation done — stitch masks, overlay on Viewer, measure, save Results."""
        if error:
            self.status_label.setText(f"ONLINE ERROR (segmentation): {error}")
            if hasattr(self, '_online_append_log') and self._online_append_log:
                self._online_append_log(f"[ERROR] segmentation: {error}", SemiconductorTheme.ACCENT_ERROR)
            self._close_online_progress(ok=False)
            self._online_cleanup()
            return

        try:
            from inno3d.core import bumpvoid
            recipe = getattr(self, "_online_recipe", None) or {}
            n_layers = len(recipe.get("layers") or [])
            seg_note = "last layer only" if n_layers > 1 else None
            self._online_log_dll_timing("SEG", bumpvoid.get_last_timing, note=seg_note)
        except Exception:
            pass

        # Move to measure stage (keep dialog open; still no algorithm detail)
        self._set_online_stage("measure", 10)
        
        try:
            mpv = self.multiplanar_tab
            if mpv.volume_data is None:
                self.status_label.setText("ONLINE ERROR: No volume in Viewer to overlay")
                self._close_online_progress(ok=False)
                self._online_cleanup()
                return

            vol_shape = mpv.volume_data.shape[:3]
            recipe = getattr(self, '_online_recipe', None) or {}
            # Runtime layers (already crop-remapped) — must match DLL split order
            layers = list(recipe.get('layers') or [])
            if not layers and self.online_roi_active and isinstance(self.online_roi, dict):
                # last resort: original locked layers (no crop case)
                layers = list(self.online_roi.get('layers') or [])

            if isinstance(result, list):
                all_successful = all(getattr(r, 'success', False) for r in result)
                if not all_successful:
                    error_msg = next(
                        (
                            r.errorMessage.decode('utf-8')
                            for r in result
                            if not getattr(r, 'success', False)
                        ),
                        "Unknown error in mask generation",
                    )
                    self.status_label.setText(
                        f"ONLINE ERROR: Multi-layer Segmentation failed - {error_msg}"
                    )
                    self._close_online_progress(ok=False)
                    self._online_cleanup()
                    return

                self._set_online_stage("measure", 20)
                self._online_append_log(
                    f"[MERGE] Stitching {len(result)} layer result(s) → full volume",
                    "#FFA000",
                )
                bump_data = np.zeros(vol_shape, dtype=np.uint8)
                void_data = np.zeros(vol_shape, dtype=np.uint8)
                total_bump_t = total_void_t = total_t = 0.0

                for i, r in enumerate(result):
                    # Prefer layer_name match, else index order
                    layer = None
                    lname = getattr(r, 'layer_name', None)
                    if lname and layers:
                        for L in layers:
                            if L.get('name') == lname:
                                layer = L
                                break
                    if layer is None and i < len(layers):
                        layer = layers[i]
                    if layer is None:
                        # No layer metadata — paste from Z=0 (test_1layer-like chunks)
                        start_z = 0
                        end_z = vol_shape[0]
                    else:
                        start_z = max(0, int(layer['z_start']))
                        end_z = min(vol_shape[0], int(layer['z_end']))
                    chunk_size = max(0, end_z - start_z)

                    b_path = r.bumpOutputPath.decode('utf-8') if r.bumpOutputPath else None
                    if b_path and os.path.exists(b_path):
                        b_chunk = np.asarray(io.imread(b_path))
                        if b_chunk.ndim == 4:
                            b_chunk = b_chunk[:, :, :, 0]
                        if b_chunk.ndim == 2:
                            b_chunk = b_chunk[np.newaxis, :, :]
                        if chunk_size > 0 and b_chunk.shape[0] > chunk_size:
                            b_chunk = b_chunk[:chunk_size]
                        end_z_b = min(vol_shape[0], start_z + b_chunk.shape[0])
                        y = min(b_chunk.shape[1], vol_shape[1])
                        x = min(b_chunk.shape[2], vol_shape[2])
                        bump_data[start_z:end_z_b, :y, :x] = np.maximum(
                            bump_data[start_z:end_z_b, :y, :x],
                            b_chunk[: end_z_b - start_z, :y, :x],
                        )

                    v_path = r.voidOutputPath.decode('utf-8') if r.voidOutputPath else None
                    if v_path and os.path.exists(v_path):
                        v_chunk = np.asarray(io.imread(v_path))
                        if v_chunk.ndim == 4:
                            v_chunk = v_chunk[:, :, :, 0]
                        if v_chunk.ndim == 2:
                            v_chunk = v_chunk[np.newaxis, :, :]
                        if chunk_size > 0 and v_chunk.shape[0] > chunk_size:
                            v_chunk = v_chunk[:chunk_size]
                        end_z_v = min(vol_shape[0], start_z + v_chunk.shape[0])
                        y = min(v_chunk.shape[1], vol_shape[1])
                        x = min(v_chunk.shape[2], vol_shape[2])
                        void_data[start_z:end_z_v, :y, :x] = np.maximum(
                            void_data[start_z:end_z_v, :y, :x],
                            v_chunk[: end_z_v - start_z, :y, :x],
                        )

                    total_bump_t += float(getattr(r, 'bumpTime', 0) or 0)
                    total_void_t += float(getattr(r, 'voidTime', 0) or 0)
                    total_t += float(getattr(r, 'totalTime', 0) or 0)

                mpv.class1_data = bump_data
                mpv.class2_data = void_data
                time_info = (total_bump_t, total_void_t, total_t)

            else:
                if not (result and getattr(result, 'success', False)):
                    error_msg = (
                        result.errorMessage.decode('utf-8') if result else "Unknown error"
                    )
                    self.status_label.setText(
                        f"ONLINE ERROR: Segmentation failed - {error_msg}"
                    )
                    self._close_online_progress(ok=False)
                    self._online_cleanup()
                    return

                # ---- Full-volume single pass (test_1layer or single without layers) ----
                bump_path = (
                    result.bumpOutputPath.decode('utf-8') if result.bumpOutputPath else None
                )
                void_path = (
                    result.voidOutputPath.decode('utf-8') if result.voidOutputPath else None
                )
                bump_data = void_data = None
                if bump_path and os.path.exists(bump_path):
                    self._tick_online_progress(30)
                    bump_data = io.imread(bump_path)
                if void_path and os.path.exists(void_path):
                    self._tick_online_progress(40)
                    void_data = io.imread(void_path)

                mpv.class1_data = self._align_mask_to_volume(bump_data, vol_shape)
                mpv.class2_data = self._align_mask_to_volume(void_data, vol_shape)
                time_info = (
                    float(getattr(result, 'bumpTime', 0) or 0),
                    float(getattr(result, 'voidTime', 0) or 0),
                    float(getattr(result, 'totalTime', 0) or 0),
                )

            # Ensure aligned shape for both paths
            mpv.class1_data = self._align_mask_to_volume(mpv.class1_data, vol_shape)
            mpv.class2_data = self._align_mask_to_volume(mpv.class2_data, vol_shape)

            # Merge masks and enable overlays (same rules as Teaching / Viewer)
            mpv.merge_class_masks()

            for orientation in ['axial', 'coronal', 'sagittal']:
                c1_check = getattr(mpv, f'{orientation}_overlay_c1', None)
                if c1_check is not None:
                    c1_check.blockSignals(True)
                    c1_check.setChecked(True)
                    c1_check.blockSignals(False)
                c2_check = getattr(mpv, f'{orientation}_overlay_c2', None)
                if c2_check is not None:
                    c2_check.blockSignals(True)
                    c2_check.setChecked(True)
                    c2_check.blockSignals(False)
                opacity_slider = getattr(mpv, f'{orientation}_opacity_slider', None)
                if opacity_slider is not None:
                    opacity_slider.blockSignals(True)
                    opacity_slider.setValue(70)  # match Teaching default
                    opacity_slider.blockSignals(False)

            # Render MPR + 3D
            mpv.update_all_views()

            # Persist combined masks + stats for host / audit
            results_dir = getattr(self, '_online_results_dir', None) or self._resolve_online_results_dir()
            self._save_online_combined_masks(mpv, results_dir)

            # Pipeline: SEG (done) → MES → B2B
            self._set_online_stage("measure", 55)
            stats_text = self._calculate_online_object_stats(mpv, results_dir=results_dir)
            self._set_online_stage("measure", 80)
            b2b_text = self._run_online_b2b(mpv, results_dir=results_dir)
            self._set_online_stage("measure", 92)

            # Sync Teaching AFTER MES so Layer bands + stats match Viewer Online
            # (masks are FAR-cleaned; object_stats already have layer_name).
            try:
                _bands = self._online_layer_bands()
            except Exception:
                _bands = None
            _stats = list(getattr(mpv, "object_stats", None) or []) or None
            try:
                _sp = None
                if hasattr(mpv, "_get_3d_world_spacing"):
                    try:
                        _sp = mpv._get_3d_world_spacing()
                    except Exception:
                        _sp = getattr(mpv, "custom_spacing", None) or getattr(
                            mpv, "spacing", None
                        )
                self.segmentation_tab.set_segmentation_results(
                    mpv.class1_data,
                    mpv.class2_data,
                    layer_bands=_bands,
                    object_stats=_stats,
                    labeled_class1=getattr(mpv, "labeled_class1_data", None),
                    spacing=_sp,
                )
            except Exception as _sync_e:
                print(f"[ONLINE] Teaching sync failed: {_sync_e}")

            # Roll FOV judge → Chip FinalBin → Wafer map color (0/1/8)
            self._update_context_judge_from_stats(mpv, stats_text)

            b_time, v_time, t_time = time_info
            msg = (
                f"ONLINE: Complete - Bump:{b_time:.1f}s Void:{v_time:.1f}s "
                f"Total:{t_time:.1f}s | {stats_text}"
            )
            if b2b_text:
                msg = f"{msg} | {b2b_text}"
            # Status sink + audit log keep detail; on-screen progress stays stage-only
            self.status_label.setText("ONLINE: Complete")
            if hasattr(self, '_online_append_log') and self._online_append_log:
                self._online_append_log(msg, SemiconductorTheme.ACCENT_SUCCESS)
                self._online_append_log(
                    f"[DONE] Results saved → {results_dir}",
                    SemiconductorTheme.ACCENT_PRIMARY,
                )
                host_info = getattr(self, '_online_host_path_info', None)
                if isinstance(host_info, HostVolumePathInfo) and host_info.parsed_ok:
                    self._online_append_log(
                        f"[CONTEXT] {host_info.breadcrumb()}",
                        SemiconductorTheme.TEXT_SECONDARY,
                    )

            # Catalog FOV run → SQLite (Batch Review)
            try:
                self._record_online_run_to_db(
                    mpv,
                    results_dir=results_dir,
                    time_info=time_info,
                    stats_text=stats_text,
                    b2b_text=b2b_text or "",
                )
            except Exception as db_err:
                import traceback
                traceback.print_exc()
                if hasattr(self, '_online_append_log') and self._online_append_log:
                    self._online_append_log(
                        f"[DB WARN] inspection catalog: {db_err}",
                        SemiconductorTheme.ACCENT_WARNING,
                    )

            # Profiling-only text file in Results folder
            try:
                seg_tab = self.segmentation_tab
                if seg_tab is not None and hasattr(seg_tab, "_flush_profiling_report"):
                    footer = [
                        f"Wall-clock summary: Bump={b_time:.1f}s "
                        f"Void={v_time:.1f}s Total={t_time:.1f}s",
                    ]
                    if getattr(self, "_online_last_enhance_sec", 0) > 0:
                        footer.append(
                            f"Enhance wall time: {self._online_last_enhance_sec:.1f}s"
                        )
                    footer.append(
                        "Note: SEG/B2B stage breakdown is from the last DLL "
                        "layer pass when multi-layer."
                    )
                    prof_path = seg_tab._flush_profiling_report(
                        results_dir, footer_lines=footer
                    )
                    if prof_path and hasattr(self, "_online_append_log") and self._online_append_log:
                        self._online_append_log(
                            f"[PROF] Report → {prof_path}",
                            "#80CBC4",
                        )
            except Exception as prof_err:
                print(f"[ONLINE] Profiling report save failed: {prof_err}")

            self._close_online_progress(ok=True)
            self.switch_tab(0)

        except Exception as e:
            import traceback
            print(f"[ONLINE] Error rendering results: {e}")
            traceback.print_exc()
            self.status_label.setText(f"ONLINE ERROR: Failed to render results - {e}")
            if hasattr(self, '_online_append_log') and self._online_append_log:
                self._online_append_log(
                    f"[ERROR] render: {e}", SemiconductorTheme.ACCENT_ERROR
                )
            self._close_online_progress(ok=False)

        self._online_cleanup()

    def _save_online_combined_masks(self, mpv, results_dir):
        """Write stitched full-volume masks next to per-layer DLL outputs."""
        try:
            os.makedirs(results_dir, exist_ok=True)
            import tifffile
            if mpv.class1_data is not None:
                tifffile.imwrite(
                    os.path.join(results_dir, "online_combined_bump3D.tif"),
                    mpv.class1_data,
                    compression='zlib',
                )
            if mpv.class2_data is not None:
                tifffile.imwrite(
                    os.path.join(results_dir, "online_combined_voidsOnly.tif"),
                    mpv.class2_data,
                    compression='zlib',
                )
        except Exception as e:
            print(f"[ONLINE] Failed to save combined masks: {e}")
            if hasattr(self, '_online_append_log') and self._online_append_log:
                self._online_append_log(
                    f"[WARN] combined mask save: {e}", SemiconductorTheme.ACCENT_WARNING
                )
    
    def _online_measure_params(self):
        """Voxel / NG / FAR sizes from Teaching tab (same as Python Viewer path)."""
        vx = vy = vz = 1.0
        z4x = False
        ng = 0.05
        bump_min, bump_max = 0.0, 1e12
        void_min, void_max = 0.0, 1e12
        seg = getattr(self, "segmentation_tab", None)

        def parse_size(text):
            try:
                parts = [float(p.strip()) for p in str(text).lower().split("x") if p.strip()]
                if not parts:
                    return 0.0
                res = 1.0
                for p in parts:
                    res *= p
                return res
            except Exception:
                return 0.0

        if seg is not None:
            sw = getattr(seg, "param_widgets", None) or {}
            if "voxel_size_x" in sw:
                vx = float(sw["voxel_size_x"].value())
                vy = float(sw["voxel_size_y"].value())
                vz = float(sw["voxel_size_z"].value())
            if sw.get("z_stretched_4x") is not None:
                z4x = bool(sw["z_stretched_4x"].isChecked())
            if "bump_minimum_size" in sw:
                bump_min = parse_size(sw["bump_minimum_size"].text())
                bump_max = parse_size(sw["bump_maximum_size"].text()) or 1e12
                void_min = parse_size(sw["void_minimum_size"].text())
                void_max = parse_size(sw["void_maximum_size"].text()) or 1e12
            if hasattr(seg, "ng_threshold_spin") and seg.ng_threshold_spin is not None:
                ng = float(seg.ng_threshold_spin.value()) / 100.0
        return {
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "z4x": z4x,
            "ng": ng,
            "bump_min": bump_min,
            "bump_max": bump_max,
            "void_min": void_min,
            "void_max": void_max,
        }

    def _ensure_mes_loaded(self):
        """Load BumpVoidMes.dll from Teaching DLL folder (same package as SEG)."""
        from inno3d.core import bumpvoid_mes

        seg = getattr(self, "segmentation_tab", None)
        dll_dir = getattr(seg, "dll_path", None) if seg else None
        if not dll_dir:
            from inno3d.core.resources import default_dll_dir

            dll_dir = default_dll_dir() or ""
        if not dll_dir:
            raise RuntimeError("No DLL folder set in Teaching — browse V2 or V3 first")

        force = True
        if bumpvoid_mes.is_loaded():
            path = bumpvoid_mes.get_dll_path() or ""
            try:
                cur_dir = os.path.dirname(path) if path else ""
                same = (
                    cur_dir
                    and os.path.normcase(os.path.abspath(cur_dir))
                    == os.path.normcase(os.path.abspath(dll_dir))
                )
                sz = os.path.getsize(path) if path and os.path.isfile(path) else 0
            except OSError:
                same, sz = False, 0
            if same and sz >= 75000:
                return bumpvoid_mes
            bumpvoid_mes.unload_dll()

        ver = bumpvoid_mes.load_dll(dll_dir, force=force)
        mes_path = bumpvoid_mes.get_dll_path() or dll_dir
        print(f"[ONLINE MES] Loaded BumpVoidMes.dll version={ver} path={mes_path}")
        if hasattr(self, "_online_append_log") and self._online_append_log:
            self._online_append_log(
                f"[MES] BumpVoidMes.dll  v{ver}  ·  {mes_path}",
                "#00BFA5",
            )
        return bumpvoid_mes

    def _calculate_online_object_stats(self, mpv, results_dir=None):
        """Per-object stats via BumpVoidMes.dll only (no Python parity compare).

        Python measurement is used **only** if MES fails, so the Online path
        stays fast when the DLL works (typical production case).
        """
        try:
            if results_dir is None:
                results_dir = getattr(self, "_online_results_dir", None)
            if not results_dir and getattr(self, "_online_folder", None):
                results_dir = os.path.join(self._online_folder, "Results")
            if results_dir:
                os.makedirs(results_dir, exist_ok=True)

            params = self._online_measure_params()
            vx, vy, vz = params["vx"], params["vy"], params["vz"]
            z4x = params["z4x"]
            ng = params["ng"]

            mes_ok = False
            mes_stats = None
            mes_elapsed = 0.0

            # ── Primary: BumpVoidMes.dll ───────────────────────────────
            try:
                bumpvoid_mes = self._ensure_mes_loaded()
                self._online_apply_dll_profiling()
                if mpv.class1_data is None:
                    raise RuntimeError("No class1 (bump) mask for measurement")

                min_vox = int(max(0, round(params["bump_min"])))
                if hasattr(self, "_online_append_log") and self._online_append_log:
                    try:
                        _mi = bumpvoid_mes.get_product_info()
                        _mv = _mi.get("version") or "?"
                    except Exception:
                        _mv = "?"
                    self._online_append_log(
                        f"[MES] Measuring via BumpVoidMes.dll v{_mv} "
                        f"(minVoxels={min_vox}, z4x={z4x}, ng={ng*100:.2f}%)",
                        "#00BFA5",
                    )

                cleaned_bump_path = ""
                cleaned_void_path = ""
                if results_dir:
                    cleaned_bump_path = os.path.join(
                        results_dir, "online_combined_bump3D_FAR.tif"
                    )
                    cleaned_void_path = os.path.join(
                        results_dir, "online_combined_voidsOnly_FAR.tif"
                    )

                # FAR inside DLL (Teaching run_false_alarm_remover parity)
                # before any final visualize / B2B
                out = bumpvoid_mes.process_memory(
                    mpv.class1_data,
                    mpv.class2_data,
                    voxel_x=vx,
                    voxel_y=vy,
                    voxel_z=vz,
                    z_stretched_4x=z4x,
                    connectivity=26,
                    min_voxels=min_vox,  # CCL prefilter (optional)
                    start_slice=0,
                    end_slice=0,
                    ng_threshold=ng,
                    export_csv=False,
                    layer_name="Online",
                    return_labels=True,
                    apply_far=True,
                    bump_min_voxels=float(params["bump_min"]),
                    bump_max_voxels=float(params["bump_max"]) if params["bump_max"] > 0 else 1e300,
                    void_min_voxels=float(params["void_min"]),
                    void_max_voxels=float(params["void_max"]) if params["void_max"] > 0 else 1e300,
                    cleaned_bump_path=cleaned_bump_path,
                    cleaned_void_path=cleaned_void_path,
                    write_cleaned_masks=bool(results_dir),
                    return_cleaned_masks=True,
                )
                # (result, stats, labels, bump_clean, void_clean)
                res, mes_stats, mes_labels = out[0], out[1], out[2]
                bump_clean = out[3] if len(out) > 3 else None
                void_clean = out[4] if len(out) > 4 else None
                mes_elapsed = float(getattr(res, "elapsedSec", 0) or 0)
                far_rm = int(getattr(res, "farRemovedBumps", 0) or 0)
                far_cv = int(getattr(res, "farClearedVoids", 0) or 0)
                try:
                    self._online_log_dll_timing("MES", bumpvoid_mes.get_last_timing)
                except Exception:
                    pass

                # Apply FAR-cleaned masks for visualization + B2B (before display)
                if bump_clean is not None:
                    mpv.class1_data = bump_clean
                if void_clean is not None:
                    mpv.class2_data = void_clean
                if mes_labels is not None:
                    mpv.labeled_class1_data = mes_labels
                # rebuild merge overlay from cleaned masks
                if hasattr(mpv, "merge_class_masks"):
                    mpv.merge_class_masks()
                # overwrite main combined masks with FAR-cleaned (host audit)
                if results_dir and bump_clean is not None:
                    try:
                        import tifffile
                        tifffile.imwrite(
                            os.path.join(results_dir, "online_combined_bump3D.tif"),
                            bump_clean,
                            compression="zlib",
                        )
                        if void_clean is not None:
                            tifffile.imwrite(
                                os.path.join(results_dir, "online_combined_voidsOnly.tif"),
                                void_clean,
                                compression="zlib",
                            )
                    except Exception as se:
                        print(f"[ONLINE MES] re-save cleaned masks: {se}")

                # assign Layer from Z bands (Teaching column) BEFORE reindex
                layer_bands = self._online_layer_bands()
                for s in mes_stats:
                    if not s.get("layer_name") and layer_bands:
                        cz = float(s.get("centroid_z", 0) or 0)
                        for L in layer_bands:
                            if L["z_start"] <= cz < L["z_end"]:
                                s["layer_name"] = L["name"]
                                break
                        if not s.get("layer_name"):
                            best = min(
                                layer_bands,
                                key=lambda L: abs(
                                    cz - 0.5 * (L["z_start"] + L["z_end"])
                                ),
                            )
                            s["layer_name"] = best["name"]

                # Unified indexing: (1,1) = top-left XY, per-layer (Teaching-style, 1-based)
                mes_stats = bumpvoid_mes.reindex_grid_top_left(
                    mes_stats, voxel_x=vx, voxel_y=vy, per_layer=True
                )
                mpv.object_stats = mes_stats
                # Full-volume absolute Z (process_memory start_slice=0)
                mpv._measurement_start_slice = 0
                # Visualize only AFTER FAR (DLL already applied)
                if hasattr(mpv, "update_all_views"):
                    try:
                        for ori in ("axial", "coronal", "sagittal"):
                            mpv.render_slice(ori, preserve_camera=True)
                        mpv.render_3d()
                    except Exception:
                        pass
                if hasattr(mpv, "populate_object_stats_table"):
                    mpv.populate_object_stats_table(
                        mes_stats, ng_threshold=ng, source_label="MES+FAR"
                    )

                if results_dir:
                    csv_mes = os.path.join(results_dir, "object_statistics.csv")
                    bumpvoid_mes.write_python_format_csv(csv_mes, mes_stats, ng_threshold=ng)
                    print(f"[STATS] MES CSV exported: {csv_mes}")

                mes_ok = True
                if hasattr(self, "_online_append_log") and self._online_append_log:
                    self._online_append_log(
                        f"[MES] Done: {len(mes_stats)} objects in {mes_elapsed:.3f}s "
                        f"(FAR removed bumps={far_rm}, cleared voids={far_cv})",
                        SemiconductorTheme.ACCENT_SUCCESS,
                    )
                    if results_dir and cleaned_bump_path:
                        self._online_append_log(
                            f"[MES] FAR masks → online_combined_*_FAR.tif + overwrote combined",
                            SemiconductorTheme.TEXT_SECONDARY,
                        )
            except Exception as mes_err:
                import traceback

                traceback.print_exc()
                print(f"[ONLINE MES] failed, fallback Python: {mes_err}")
                if hasattr(self, "_online_append_log") and self._online_append_log:
                    self._online_append_log(
                        f"[MES FAIL] {mes_err} → Python fallback",
                        SemiconductorTheme.ACCENT_ERROR,
                    )

            # ── Python only if MES failed (emergency path) ─────────────
            if not mes_ok:
                try:
                    if hasattr(self, "_online_append_log") and self._online_append_log:
                        self._online_append_log(
                            "[MES] Running Python measurement fallback…",
                            SemiconductorTheme.ACCENT_WARNING,
                        )
                    mpv.run_object_analysis()
                    if results_dir:
                        csv_path = os.path.join(results_dir, "object_statistics.csv")
                        mpv.export_stats_csv(output_path=csv_path)
                        print(f"[STATS] Python fallback CSV: {csv_path}")
                except Exception as py_err:
                    import traceback

                    traceback.print_exc()
                    print(f"[ONLINE] Python measure fallback error: {py_err}")
                    if hasattr(self, "_online_append_log") and self._online_append_log:
                        self._online_append_log(
                            f"[PY FAIL] {py_err}",
                            SemiconductorTheme.ACCENT_ERROR,
                        )

            if not getattr(mpv, "object_stats", None):
                return "Stats: 0 objects"

            c1_count = len(mpv.object_stats)
            c2_count = sum(1 for s in mpv.object_stats if s.get("c2_volume", 0) > 0)
            c1_total_vol = sum(s.get("c1_volume", 0) for s in mpv.object_stats)
            c2_total_vol = sum(s.get("c2_volume", 0) for s in mpv.object_stats)
            ng_count = sum(
                1
                for s in mpv.object_stats
                if (
                    s.get("ratio", 0) >= ng
                    if s.get("ratio") != float("inf")
                    else True
                )
            )
            engine = "MES" if mes_ok else "Python"
            stats_text = (
                f"[{engine}] C1: {c1_count} obj ({c1_total_vol:,.0f} um^3) | "
                f"C2: {c2_count} obj ({c2_total_vol:,.0f} um^3) | "
                f"NG: {ng_count} obj"
            )
            print(f"[ONLINE STATS] {stats_text}")
            return stats_text

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[ONLINE] Stats calculation error: {e}")
            return "Stats: calculation error"

    def _ensure_b2b_loaded(self):
        """Load golden gap-capable BoundaryGPU.dll / BumpVoidB2B.dll from SEG DLL folder.

        Always re-pick if the currently loaded PE is summary-only (legacy ~73KB),
        so Online B2B matches Teaching / before UI:
          boundary_gap.csv  Src→Dst + Direction + Euclidean µm
        """
        from inno3d.core import bumpvoid_b2b

        seg = getattr(self, "segmentation_tab", None)
        dll_dir = getattr(seg, "dll_path", None) if seg else None
        if not dll_dir:
            from inno3d.core.resources import default_dll_dir

            dll_dir = default_dll_dir() or ""

        force = True
        if bumpvoid_b2b.is_loaded():
            path = bumpvoid_b2b.get_dll_path() or ""
            try:
                cur_dir = os.path.dirname(path) if path else ""
                same = (
                    cur_dir
                    and os.path.normcase(os.path.abspath(cur_dir))
                    == os.path.normcase(os.path.abspath(dll_dir))
                )
            except Exception:
                same = False
            if same and bumpvoid_b2b.is_gap_capable():
                force = False
            elif not bumpvoid_b2b.is_gap_capable():
                print(
                    f"[ONLINE B2B] Reloading: current DLL is summary-only "
                    f"({bumpvoid_b2b.get_dll_path()})"
                )
                force = True

        ver = bumpvoid_b2b.load_dll(dll_dir, force=force)
        path = bumpvoid_b2b.get_dll_path() or ""
        gap_ok = bumpvoid_b2b.is_gap_capable()
        fmt = "boundary_gap" if gap_ok else "boundary_summary(legacy)"
        print(f"[ONLINE B2B] Loaded {path} version={ver} format={fmt}")
        if hasattr(self, "_online_append_log") and self._online_append_log:
            color = "#FFB74D" if gap_ok else SemiconductorTheme.ACCENT_WARNING
            self._online_append_log(
                f"[B2B] {os.path.basename(path) or 'BoundaryGPU.dll'}  v{ver}  "
                f"·  {fmt}  ·  {path}",
                color,
            )
            if not gap_ok:
                self._online_append_log(
                    "[B2B] WARNING: summary-only DLL — table will show Bump/Label "
                    "not Src→Dst Dir. Deploy BumpVoid_ISP_B2B release BoundaryGPU.dll.",
                    SemiconductorTheme.ACCENT_WARNING,
                )
        return bumpvoid_b2b

    def _online_layer_bands(self):
        """Return list of dicts {name, z_start, z_end} for runtime Online layers."""
        recipe = getattr(self, "_online_recipe", None) or {}
        layers = list(recipe.get("layers") or [])
        if not layers and self.online_roi_active and isinstance(self.online_roi, dict):
            layers = list(self.online_roi.get("layers") or [])
        out = []
        for L in layers:
            try:
                zs = int(L.get("z_start", 0))
                ze = int(L.get("z_end", 0))
                if ze > zs:
                    out.append({
                        "name": str(L.get("name") or f"Layer_{len(out)+1}"),
                        "z_start": zs,
                        "z_end": ze,
                    })
            except Exception:
                continue
        return out

    def _run_online_b2b(self, mpv, results_dir=None):
        """After MES: run B2B (BoundaryGPU) per layer, merge tables into B2B tab.

        Pipeline step: receive → SEG → MES → **B2B**.
        Primary result: boundary_gap.csv (Src→Dst per direction Up/Down/Left/Right),
        matching Teaching / Send2Pixel reference (e.g. Layer_1/boundary_gap.csv).
        """
        try:
            if results_dir is None:
                results_dir = getattr(self, "_online_results_dir", None)
            if not results_dir:
                return ""

            if mpv.class1_data is None:
                return "B2B: no bump mask"

            params = self._online_measure_params()
            vx, vy, vz = params["vx"], params["vy"], params["vz"]
            # Teaching passes effective vz (already /4 if stretched) into BoundaryGPU
            vz_eff = (vz / 4.0) if params["z4x"] else vz

            bumpvoid_b2b = self._ensure_b2b_loaded()
            self._online_apply_dll_profiling()
            layers = self._online_layer_bands()
            if not layers:
                # Full-volume single pass
                layers = [{
                    "name": "Full",
                    "z_start": 0,
                    "z_end": int(mpv.class1_data.shape[0]),
                }]

            labeled = getattr(mpv, "labeled_class1_data", None)
            if labeled is None or labeled.shape != mpv.class1_data.shape:
                # Fallback if MES labels missing (Python-only path)
                from skimage import measure as _sk_measure
                labeled = _sk_measure.label(mpv.class1_data > 0).astype(np.uint32)
                mpv.labeled_class1_data = labeled

            object_stats = list(getattr(mpv, "object_stats", []) or [])
            b2b_root = os.path.join(results_dir, "B2B")
            os.makedirs(b2b_root, exist_ok=True)

            dll_info = ""
            try:
                dll_info = bumpvoid_b2b.get_dll_path() or ""
            except Exception:
                pass
            if hasattr(self, "_online_append_log") and self._online_append_log:
                self._online_append_log(
                    f"[B2B] Running BoundaryGPU on {len(layers)} layer(s) "
                    f"(vz_eff={vz_eff:.4f} um) · gap Src→Dst 4-dir",
                    "#FFB74D",
                )
                if dll_info:
                    self._online_append_log(
                        f"[B2B] DLL: {dll_info}",
                        SemiconductorTheme.TEXT_SECONDARY,
                    )

            layer_csvs = []
            ok_n = 0
            fail_n = 0
            for i, L in enumerate(layers):
                name = L["name"]
                z0, z1 = L["z_start"], L["z_end"]
                z0 = max(0, min(z0, mpv.class1_data.shape[0]))
                z1 = max(z0 + 1, min(z1, mpv.class1_data.shape[0]))
                bump_l = np.ascontiguousarray(mpv.class1_data[z0:z1] > 0, dtype=np.uint8) * 255
                # Labels are full-volume IDs; crop Z for this layer
                lab_l = np.ascontiguousarray(labeled[z0:z1], dtype=np.uint32)
                out_dir = os.path.join(b2b_root, name.replace(" ", "_"))
                os.makedirs(out_dir, exist_ok=True)
                try:
                    rc = bumpvoid_b2b.run_boundary_analysis(
                        bump_l,
                        lab_l,
                        voxel_z=vz_eff,
                        voxel_y=vy,
                        voxel_x=vx,
                        output_dir=out_dir,
                        object_stats=object_stats,
                    )
                    # Prefer golden gap CSV; fall back to legacy summary
                    layer_csv = bumpvoid_b2b.find_layer_csv(out_dir)
                    if rc == 1 and layer_csv:
                        layer_csvs.append((name, layer_csv))
                        ok_n += 1
                    elif rc == 0:
                        fail_n += 1  # no boundary voxels
                    else:
                        fail_n += 1
                        print(f"[ONLINE B2B] {name} rc={rc} csv={layer_csv}")
                except Exception as le:
                    fail_n += 1
                    print(f"[ONLINE B2B] {name} error: {le}")

                if hasattr(self, "_online_append_log") and self._online_append_log and (
                    i < 3 or i == len(layers) - 1
                ):
                    self._online_append_log(
                        f"[B2B] {name} Z[{z0}:{z1}] → {os.path.basename(out_dir)}",
                        SemiconductorTheme.TEXT_SECONDARY,
                    )

            rows = bumpvoid_b2b.aggregate_layer_summaries(layer_csvs)
            # Embed layer z_start so 3D gap lines map local layer Z → full volume Z
            zmap = {
                str(L.get("name", "")): int(L.get("z_start", 0))
                for L in layers
            }
            for r in rows:
                ln = str(r.get("Layer", "") or "")
                z0 = zmap.get(ln)
                if z0 is None:
                    z0 = zmap.get(ln.replace("_", " "))
                if z0 is None:
                    key = ln.replace(" ", "_").lower()
                    for k, v in zmap.items():
                        if str(k).replace(" ", "_").lower() == key:
                            z0 = v
                            break
                r["z_start"] = int(z0 or 0)

            combined_path = os.path.join(results_dir, "boundary_gap_all_layers.csv")
            if rows:
                combined_path = bumpvoid_b2b.write_combined_summary(combined_path, rows)
                print(f"[STATS] B2B combined CSV: {combined_path} ({len(rows)} rows)")
            try:
                self._online_log_dll_timing("B2B", bumpvoid_b2b.get_last_timing)
            except Exception:
                pass

            if hasattr(mpv, "populate_b2b_table"):
                mpv.populate_b2b_table(rows, source_label="B2B")
            # Switch UI to B2B tab so operator sees combined table
            if hasattr(mpv, "stats_tabs") and rows:
                try:
                    mpv.stats_tabs.setCurrentIndex(1)
                except Exception:
                    pass

            text = f"[B2B] {len(rows)} gaps · {ok_n}/{len(layers)} layers OK"
            if fail_n:
                text += f" · {fail_n} fail/empty"
            print(f"[ONLINE B2B] {text}")
            if hasattr(self, "_online_append_log") and self._online_append_log:
                self._online_append_log(text, "#FFB74D")
                if rows:
                    self._online_append_log(
                        f"[B2B] Combined table → {combined_path}",
                        SemiconductorTheme.TEXT_SECONDARY,
                    )
            return text

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[ONLINE B2B] error: {e}")
            if hasattr(self, "_online_append_log") and self._online_append_log:
                self._online_append_log(
                    f"[B2B FAIL] {e}", SemiconductorTheme.ACCENT_ERROR
                )
            return "B2B: error"
    
    def _record_online_run_to_db(
        self,
        mpv,
        results_dir: str,
        time_info,
        stats_text: str = "",
        b2b_text: str = "",
    ):
        """Persist FOV completion into SQLite for Batch Review."""
        from inno3d.core.inspection_db import record_online_fov_completion

        host_info = getattr(self, "_online_host_path_info", None)
        b_time, v_time, t_time = time_info if time_info else (0.0, 0.0, 0.0)
        enhance_sec = 0.0
        if hasattr(self, "_online_enhance_start_time") and hasattr(
            self, "_online_enhance_end_time"
        ):
            try:
                enhance_sec = float(
                    self._online_enhance_end_time - self._online_enhance_start_time
                )
            except Exception:
                enhance_sec = 0.0
        elif hasattr(self, "_online_enhance_start_time"):
            # finished handler may not have stored end; estimate from wall if present
            enhance_sec = float(getattr(self, "_online_last_enhance_sec", 0) or 0)

        stats = list(getattr(mpv, "object_stats", None) or [])
        n_obj = len(stats)
        n_ng = 0
        n_ok = 0
        ng_thr = 0.05
        try:
            if hasattr(self, "segmentation_tab") and hasattr(
                self.segmentation_tab, "ng_threshold_spin"
            ):
                ng_thr = float(self.segmentation_tab.ng_threshold_spin.value()) / 100.0
        except Exception:
            pass
        for s in stats:
            if not isinstance(s, dict):
                continue
            ratio = s.get("ratio", 0.0)
            try:
                rf = float(ratio) if ratio != float("inf") else 1e9
            except (TypeError, ValueError):
                rf = 0.0
            jud = s.get("judgment")
            if jud in (1, "NG", "ng"):
                n_ng += 1
            elif jud in (8, "OK", "ok"):
                n_ok += 1
            elif rf >= ng_thr:
                n_ng += 1
            else:
                n_ok += 1

        judgment = "OK"
        final_bin = 8  # FOV-level FinalBin (1=NG, 8=OK)
        # Prefer FOV-level rollup if context updated a die
        if n_ng > 0 and (n_ng / max(n_obj, 1)) >= 0.01:
            # any NG objects still OK FOV unless operator threshold marks FOV NG —
            # match Online map coloring when NG objects present and ratio high
            pass
        # Simple FOV judgment: NG if any NG object exceeds count heuristic OR stats_text says NG
        st_up = (stats_text or "").upper()
        if " NG" in f" {st_up}" or st_up.startswith("NG") or n_ng > 0 and n_ok == 0:
            judgment = "NG"
            final_bin = 1
        elif n_ng > max(3, n_obj // 20):
            judgment = "NG"
            final_bin = 1

        # Chip-level FinalBin for chips table / wafer map (not FOV bin).
        # Must match ChipCell.recompute_counts: NG if any FOV NG; OK only after 9/9 OK.
        chip_final_bin = 2  # BIN_PENDING
        try:
            if hasattr(mpv, "get_wafer_context"):
                ctx = mpv.get_wafer_context()
                if ctx is not None:
                    col = int(getattr(host_info, "chip_col", 0) or 0) if host_info else 0
                    row = int(getattr(host_info, "chip_row", 0) or 0) if host_info else 0
                    if col <= 0 or row <= 0:
                        col, row = int(ctx.selected_col or 0), int(ctx.selected_row or 0)
                    cell = ctx.chip_at(col, row) if col > 0 and row > 0 else None
                    if cell is not None:
                        cell.recompute_counts()
                        chip_final_bin = int(cell.final_bin)
        except Exception:
            chip_final_bin = 2

        artifacts = []
        if results_dir and os.path.isdir(results_dir):
            for name, kind in (
                ("object_statistics.csv", "csv_mes"),
                ("online_combined_bump3D.tif", "mask_bump"),
                ("online_combined_voidsOnly.tif", "mask_void"),
                ("online_combined_bump3D_FAR.tif", "mask_bump_far"),
                ("online_combined_voidsOnly_FAR.tif", "mask_void_far"),
                ("boundary_gap_all_layers.csv", "csv_b2b"),
            ):
                p = os.path.join(results_dir, name)
                if os.path.exists(p):
                    artifacts.append({"kind": kind, "path": p})
            enh_dir = os.path.join(results_dir, "enhanced_volume")
            if os.path.isdir(enh_dir):
                artifacts.append({"kind": "enhanced_dir", "path": enh_dir})
                for name, kind in (
                    ("Enhanced_Volume.tif", "enhanced_volume"),
                    ("Enhanced_Volume.tiff", "enhanced_volume"),
                ):
                    p = os.path.join(enh_dir, name)
                    if os.path.exists(p):
                        artifacts.append({"kind": kind, "path": p})
                        break

        timeline = []
        if getattr(self, "_online_enhancement_enabled", False):
            timeline.append(
                {
                    "step": "enhance",
                    "message": (
                        f"Enhance · provider={getattr(self, '_online_enhancement_mode', '')} "
                        f"{getattr(self, '_online_last_enhance_provider', '')}"
                    ).strip(),
                    "duration_sec": enhance_sec,
                }
            )
        timeline.append(
            {
                "step": "seg",
                "message": f"SEG · Bump {b_time:.1f}s Void {v_time:.1f}s",
                "duration_sec": float(t_time or 0),
            }
        )
        timeline.append(
            {
                "step": "mes",
                "message": stats_text or "MES",
                "duration_sec": 0,
            }
        )
        if b2b_text:
            timeline.append({"step": "b2b", "message": b2b_text, "duration_sec": 0})

        payload = {
            "host_path": getattr(self, "_online_source_file", None)
            or (host_info.raw_path if isinstance(host_info, HostVolumePathInfo) else ""),
            "input_path": (
                host_info.input_path
                if isinstance(host_info, HostVolumePathInfo)
                else getattr(self, "_online_source_file", "")
            ),
            "results_dir": results_dir or "",
            "date_folder": getattr(host_info, "date_folder", "") if host_info else "",
            "lot_foup_id": getattr(host_info, "lot_foup_id", "") if host_info else "",
            "lot_id": getattr(host_info, "lot_id", "") if host_info else "",
            "foup_id": getattr(host_info, "foup_id", "") if host_info else "",
            "wafer_id": getattr(host_info, "wafer_id", "") if host_info else "",
            "chip_col": int(getattr(host_info, "chip_col", 0) or 0) if host_info else 0,
            "chip_row": int(getattr(host_info, "chip_row", 0) or 0) if host_info else 0,
            "chip_folder": getattr(host_info, "chip_folder", "") if host_info else "",
            "fov_index": int(getattr(host_info, "fov_index", 0) or 0) if host_info else 0,
            "fov_folder": getattr(host_info, "fov_folder", "") if host_info else "",
            "judgment": judgment,
            "final_bin": final_bin,  # FOV judge (1/8)
            "chip_final_bin": chip_final_bin,  # die rollup for wafer map (1/2/8)
            "n_objects": n_obj,
            "n_ng": n_ng,
            "n_ok": n_ok,
            "enhance_sec": enhance_sec,
            "seg_sec": float(t_time or 0),
            "mes_sec": 0.0,
            "b2b_sec": 0.0,
            "total_sec": float(t_time or 0) + float(enhance_sec or 0),
            "enhance_provider": str(
                getattr(self, "_online_last_enhance_provider", "")
                or getattr(self, "_online_enhancement_mode", "")
            ),
            "config_path": getattr(self, "online_config_path", None)
            or (
                getattr(self.segmentation_tab, "config_path", "")
                if hasattr(self, "segmentation_tab")
                else ""
            ),
            "dll_dir": (
                getattr(self.segmentation_tab, "dll_path", "")
                if hasattr(self, "segmentation_tab")
                else ""
            ),
            "recipe_json": getattr(self, "_online_recipe", None) or {},
            "mes_objects": stats,
            "artifacts": artifacts,
            "timeline": timeline,
            "status": "ok",
        }
        run_id = record_online_fov_completion(payload)
        if hasattr(self, "_online_append_log") and self._online_append_log:
            self._online_append_log(
                f"[DB] FOV catalogued · run_id={run_id} · {judgment} · {n_obj} objects",
                "#CE93D8",
            )
        # Notify Batch Review tab if present
        br = getattr(self, "batch_review_tab", None)
        if br is not None and hasattr(br, "on_run_catalogued"):
            try:
                br.on_run_catalogued(payload)
            except Exception:
                pass
        elif br is not None and hasattr(br, "refresh_all"):
            try:
                br.refresh_all()
            except Exception:
                pass
        return run_id

    def _on_seg_inspection_done(self, volume_data, bump_data, void_data):
        """Handle SegmentationTab inspection_done signal - sync data to 3D Viewer tab"""
        pass  # Online drives Viewer explicitly via _on_online_inspection_finished
        
    def _crop_volume_for_online(self, files, roi):
        """Crop for Online Mode; return ``(crop_folder, meta)`` or ``(None, None)``.

        ``meta`` includes ``z0`` (original Z start) so LAYER bands can be remapped
        into cropped volume coordinates before DLL split / mask stitch.
        """
        try:
            crop_root = tempfile.mkdtemp(prefix="online_crop_")
            
            try:
                rx_min = int(min(float(roi['x1']), float(roi['x2'])))
                rx_max = int(max(float(roi['x1']), float(roi['x2'])))
                ry_min = int(min(float(roi['y1']), float(roi['y2'])))
                ry_max = int(max(float(roi['y1']), float(roi['y2'])))
                rz_min = int(min(float(roi['z_start']), float(roi['z_end'])))
                rz_max = int(max(float(roi['z_start']), float(roi['z_end'])))
            except (KeyError, ValueError, TypeError) as e:
                self._online_append_log(f"[ROI DATA ERROR] {e}", SemiconductorTheme.ACCENT_ERROR)
                return None, None
            
            self._online_append_log(
                f"[TARGET ROI] X[{rx_min}:{rx_max}] Y[{ry_min}:{ry_max}] Z[{rz_min}:{rz_max}]",
                "#FFA000",
            )

            if len(files) == 1:
                # --- CASE A: Single 3D TIFF Stack ---
                raw_data = io.imread(str(files[0]))
                img_stack = np.asarray(raw_data)
                if img_stack.ndim == 4:
                    img_stack = img_stack[:, :, :, 0]
                if img_stack.ndim < 3:
                    img_stack = img_stack[np.newaxis, :, :]
                
                sz, sy, sx = img_stack.shape[:3]
                
                cz1 = int(max(0, min(rz_min, sz - 1)))
                cz2 = int(max(1, min(rz_max, sz)))
                cy1 = int(max(0, min(ry_min, sy)))
                cy2 = int(max(1, min(ry_max, sy)))
                cx1 = int(max(0, min(rx_min, sx)))
                cx2 = int(max(1, min(rx_max, sx)))
                
                if cz2 <= cz1 or cy2 <= cy1 or cx2 <= cx1:
                    self._online_append_log(
                        "[CROP ERROR] Resulting volume is empty", SemiconductorTheme.ACCENT_ERROR
                    )
                    shutil.rmtree(crop_root, ignore_errors=True)
                    return None, None
                
                cropped = img_stack[cz1:cz2, cy1:cy2, cx1:cx2].copy()
                out_path = os.path.join(crop_root, "roi_crop_volume.tif")
                io.imsave(out_path, cropped, check_contrast=False)
                meta = {
                    'z0': cz1, 'z1': cz2, 'z_len': cz2 - cz1,
                    'y0': cy1, 'y1': cy2,
                    'x0': cx1, 'x1': cx2,
                    'orig_shape': (sz, sy, sx),
                    'crop_shape': cropped.shape[:3],
                }
                return crop_root, meta

            else:
                # --- CASE B: Multiple 2D Files ---
                num_files = len(files)
                cz1 = int(max(0, min(rz_min, num_files - 1)))
                cz2 = int(max(1, min(rz_max, num_files)))
                if cz2 <= cz1:
                    shutil.rmtree(crop_root, ignore_errors=True)
                    return None, None
                
                target_files = files[cz1:cz2]
                
                def process_and_save(idx_file):
                    idx, file_path = idx_file
                    try:
                        img = np.asarray(io.imread(str(file_path)))
                        if img.ndim == 3:
                            img = img[0] if img.shape[0] <= 4 else img[:, :, 0]
                        
                        h, w = img.shape[:2]
                        scx1 = int(max(0, min(rx_min, w)))
                        scx2 = int(max(1, min(rx_max, w)))
                        scy1 = int(max(0, min(ry_min, h)))
                        scy2 = int(max(1, min(ry_max, h)))
                        
                        if scx2 <= scx1 or scy2 <= scy1:
                            return False
                        
                        cropped = img[scy1:scy2, scx1:scx2].copy()
                        out_path = os.path.join(crop_root, f"slice{idx:04d}.tif")
                        io.imsave(out_path, cropped, check_contrast=False)
                        return True
                    except Exception:
                        return False

                with concurrent.futures.ThreadPoolExecutor(max_workers=min(12, len(target_files))) as executor:
                    results = list(executor.map(process_and_save, enumerate(target_files)))
                
                if any(results):
                    meta = {
                        'z0': cz1, 'z1': cz2, 'z_len': cz2 - cz1,
                        'y0': ry_min, 'y1': ry_max,
                        'x0': rx_min, 'x1': rx_max,
                        'orig_shape': (num_files, None, None),
                        'crop_shape': (cz2 - cz1, None, None),
                    }
                    return crop_root, meta
                
            shutil.rmtree(crop_root, ignore_errors=True)
            return None, None
            
        except Exception as e:
            print(f"Online crop utility error: {e}")
            import traceback
            traceback.print_exc()
            return None, None
            
    def _online_cleanup(self):
        """Cleanup temp files and prepare for next file - keep DLL + config + recipe"""
        # Ensure progress overlay is not left blocking the UI
        dlg = getattr(self, "_online_progress", None)
        if dlg is not None:
            try:
                if isinstance(dlg, OnlinePipelineProgress):
                    if dlg.isVisible() and dlg._stage not in ("done", "error"):
                        dlg.close()
                elif dlg.isVisible():
                    dlg.close()
            except Exception:
                pass

        # Cleanup temp dir (viewer load copies)
        if hasattr(self, '_online_temp_dir') and self._online_temp_dir:
            try:
                shutil.rmtree(self._online_temp_dir, ignore_errors=True)
            except Exception:
                pass
            self._online_temp_dir = None

        # Cleanup crop temp (results already saved to stable Results dir)
        crop_meta = getattr(self, '_online_crop_meta', None)
        # crop folder is _online_folder when crop was used
        runtime_folder = getattr(self, '_online_folder', None)
        if runtime_folder and 'online_crop_' in str(runtime_folder):
            try:
                shutil.rmtree(runtime_folder, ignore_errors=True)
            except Exception:
                pass
        
        # Cleanup enhanced volume temp dir (only if under temp / enhanced_volume helper)
        if hasattr(self, '_online_enhanced_output') and self._online_enhanced_output:
            enh = self._online_enhanced_output
            # Do not delete if it lives under the stable source Results tree
            src_folder = getattr(self, '_online_source_folder', '')
            if src_folder and str(enh).startswith(str(src_folder)):
                pass  # keep enhanced for audit
            else:
                try:
                    shutil.rmtree(enh, ignore_errors=True)
                except Exception:
                    pass
            self._online_enhanced_output = None
        
        # Release enhancement thread reference
        if hasattr(self, '_online_enhancement_thread'):
            self._online_enhancement_thread = None

        self._online_crop_meta = None
        
        # Keep last received source path visible in Teaching after Online job
        if hasattr(self, 'segmentation_tab') and getattr(self, '_online_source_file', None):
            seg_tab = self.segmentation_tab
            if hasattr(seg_tab, 'input_path_input'):
                seg_tab.input_path_input.setText(self._online_source_file)
            if getattr(seg_tab, 'config', None) is not None:
                seg_tab.config.inputPath = self._online_source_file.encode('utf-8')
            else:
                seg_tab._pending_input_path = self._online_source_file

        self._online_processing = False
