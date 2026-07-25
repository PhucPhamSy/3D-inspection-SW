# inno3d/features/viewer/align_sidebar.py
# -----------------------------------------------------------------------
# AlignSidebarMixin — VIEW TOOLS sidebar (right panel) for MultiPlanarView
#
# Extracted from tabs/viewer.py (Phase 3.5 continuation) — MOVE not COPY.
# Contains: _create_align_tools_host, _create_align_tools_rail,
#            set_align_tools_expanded, toggle_align_tools,
#            _rebalance_main_splitter_for_tools, _create_align_sidebar
#
# MRO: add AlignSidebarMixin to MultiPlanarView bases BEFORE QWidget.
# Define-module rule: every name below must be imported in THIS file.
# -----------------------------------------------------------------------
"""AlignSidebarMixin — right-side VIEW TOOLS panel builders."""

from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QToolButton, QSlider, QSpinBox, QDoubleSpinBox,
    QCheckBox, QFrame, QScrollArea, QAbstractSpinBox, QSizePolicy,
)

from inno3d.core.styles import SemiconductorTheme
from inno3d.core.tf_widgets import HistogramWLWidget
from inno3d.features.viewer.shared_widgets import _ui_icon, _tinted_ui_icon


def _style_compact_reset_button(btn):
    """Style a compact «Reset» QPushButton without clipping hover border.

    Global theme uses min-height 30px + padding 6×14. Shorter fixedHeight
    clips the bottom border on hover. Full override (normal/hover/pressed)
    keeps a complete 1px border inside the widget box.
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QSizePolicy

    btn.setCursor(Qt.PointingHandCursor)
    btn.setMinimumWidth(68)
    # Match global min-height so border is not clipped by fixed short height
    btn.setMinimumHeight(30)
    btn.setMaximumHeight(32)
    btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
    # Leave room for 1px border top+bottom inside the layout cell
    btn.setContentsMargins(0, 0, 0, 0)
    t = SemiconductorTheme
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {t.BG_LIGHT};
            border: 1px solid {t.BORDER_DEFAULT};
            color: {t.TEXT_PRIMARY};
            border-radius: 4px;
            padding: 4px 12px;
            font-weight: 600;
            font-size: 9pt;
            min-width: 64px;
            min-height: 28px;
        }}
        QPushButton:hover {{
            background-color: {t.BG_PANEL};
            border: 1px solid {t.BORDER_HOVER};
            color: {t.TEXT_PRIMARY};
        }}
        QPushButton:pressed {{
            background-color: {t.BG_MEDIUM};
            border: 1px solid {t.BORDER_ACTIVE};
        }}
    """)


class AlignSidebarMixin:
    """Mixin providing the right VIEW TOOLS sidebar UI for MultiPlanarView."""

    # Re-export for other mixins on MultiPlanarView (e.g. Volume3dMixin)
    _style_compact_reset_button = staticmethod(_style_compact_reset_button)

    def _create_align_tools_host(self):
        """Host for VIEW TOOLS: full sidebar (expanded) or thin reopen rail (collapsed)."""
        host = QWidget()
        host.setObjectName("AlignToolsHost")
        host.setStyleSheet(f"""
            QWidget#AlignToolsHost {{
                background: {SemiconductorTheme.BG_MEDIUM};
                border-left: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
        """)
        lay = QHBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.align_tools_rail = self._create_align_tools_rail()
        self.align_sidebar = self._create_align_sidebar()
        lay.addWidget(self.align_tools_rail, 0)
        lay.addWidget(self.align_sidebar, 1)

        # Default: full tools (offline). Rail hidden until collapse / Online.
        self.align_tools_rail.setVisible(False)
        self.align_sidebar.setVisible(True)
        host.setMinimumWidth(240)
        host.setMaximumWidth(240)
        return host

    def _create_align_tools_rail(self):
        """Slim strip shown when VIEW TOOLS is collapsed — one click to reopen."""
        rail = QFrame()
        rail.setObjectName("AlignToolsRail")
        rail.setFixedWidth(28)
        rail.setCursor(Qt.PointingHandCursor)
        rail.setToolTip(
            "Show VIEW TOOLS\n"
            "Window Leveling · Clip · Oblique MPR · Alignment"
        )
        rail.setStyleSheet(f"""
            QFrame#AlignToolsRail {{
                background: {SemiconductorTheme.BG_MEDIUM};
                border: none;
                border-right: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
            QFrame#AlignToolsRail:hover {{
                background: {SemiconductorTheme.BG_LIGHT};
            }}
            QToolButton#AlignToolsRailBtn {{
                background: transparent;
                border: none;
                color: {SemiconductorTheme.ACCENT_PRIMARY};
                padding: 4px 0;
            }}
            QToolButton#AlignToolsRailBtn:hover {{
                background: rgba(34, 174, 209, 0.15);
                border-radius: 4px;
            }}
            QLabel#AlignToolsRailLabel {{
                color: {SemiconductorTheme.TEXT_SECONDARY};
                font-size: 8pt;
                font-weight: 700;
                letter-spacing: 1px;
            }}
        """)
        v = QVBoxLayout(rail)
        v.setContentsMargins(2, 8, 2, 8)
        v.setSpacing(6)

        open_btn = QToolButton()
        open_btn.setObjectName("AlignToolsRailBtn")
        open_btn.setArrowType(Qt.LeftArrow)
        open_btn.setToolTip("Show VIEW TOOLS (Window Leveling, Clip, Align…)")
        open_btn.setFixedSize(24, 24)
        open_btn.setCursor(Qt.PointingHandCursor)
        open_btn.clicked.connect(lambda: self.set_align_tools_expanded(True))
        v.addWidget(open_btn, 0, Qt.AlignHCenter)

        gear = QToolButton()
        gear.setObjectName("AlignToolsRailBtn")
        gear.setIcon(_ui_icon("settings.svg", gear))
        gear.setIconSize(QSize(16, 16))
        gear.setToolTip("Show VIEW TOOLS")
        gear.setFixedSize(24, 24)
        gear.setCursor(Qt.PointingHandCursor)
        gear.clicked.connect(lambda: self.set_align_tools_expanded(True))
        v.addWidget(gear, 0, Qt.AlignHCenter)

        # Vertical-ish label (stacked letters — readable in 28px rail)
        for ch in ("T", "O", "O", "L", "S"):
            lab = QLabel(ch)
            lab.setObjectName("AlignToolsRailLabel")
            lab.setAlignment(Qt.AlignCenter)
            v.addWidget(lab, 0, Qt.AlignHCenter)

        v.addStretch(1)

        # Click empty rail area also expands
        def _rail_click(_event):
            self.set_align_tools_expanded(True)

        rail.mousePressEvent = _rail_click  # type: ignore[method-assign]
        return rail

    def set_align_tools_expanded(self, expanded, remember=True):
        """Expand (full 240px sidebar) or collapse (28px reopen rail) VIEW TOOLS.

        Online mode defaults to collapsed so MPR / CONTEXT / Statistics are larger.
        ``remember`` stores the choice while Online is ON (session preference).
        """
        expanded = bool(expanded)
        self._align_tools_expanded = expanded
        if remember and getattr(self, "_online_viewer_mode", False):
            self._align_tools_user_expanded = expanded

        if not hasattr(self, "align_tools_host"):
            return

        # Host stays hidden until a volume exists (tools need volume context)
        if self.volume_data is None and not expanded:
            # Still allow rail geometry updates if host already shown
            pass

        if hasattr(self, "align_sidebar") and self.align_sidebar is not None:
            self.align_sidebar.setVisible(expanded)
        if hasattr(self, "align_tools_rail") and self.align_tools_rail is not None:
            self.align_tools_rail.setVisible(not expanded)

        # Collapse button only relevant when panel is open
        if hasattr(self, "align_tools_collapse_btn") and self.align_tools_collapse_btn is not None:
            self.align_tools_collapse_btn.setVisible(expanded)

        if expanded:
            self.align_tools_host.setMinimumWidth(240)
            self.align_tools_host.setMaximumWidth(240)
        else:
            self.align_tools_host.setMinimumWidth(28)
            self.align_tools_host.setMaximumWidth(28)

        self._rebalance_main_splitter_for_tools(expanded)
        QTimer.singleShot(40, self._reposition_visible_overlays)

    def toggle_align_tools(self):
        """Toggle VIEW TOOLS expanded/collapsed."""
        self.set_align_tools_expanded(not getattr(self, "_align_tools_expanded", True))

    def set_online_viewer_layout(self, online):
        """Online ON → auto-collapse VIEW TOOLS (more room for MPR + stats).

        User can reopen via the thin TOOLS rail; choice is remembered for the
        Online session. Online OFF → restore full sidebar when volume is loaded.
        """
        online = bool(online)
        self._online_viewer_mode = online

        if online:
            # Fresh Online session: start collapsed unless already opened this session
            # (toggle ON always resets to collapsed for max inspection space)
            self._align_tools_user_expanded = False
            self._3d_render_force_on_online = False
            if self.volume_data is not None and hasattr(self, "align_tools_host"):
                self.align_tools_host.setVisible(True)
                self.set_align_tools_expanded(False, remember=True)
            elif hasattr(self, "align_tools_host") and self.align_tools_host.isVisible():
                self.set_align_tools_expanded(False, remember=True)
            # Large Online volumes: skip GPU 3D by default (user can re-enable via 3D btn)
            self.set_3d_volume_render_enabled(False, sync_button=True)
        else:
            # Manual mode: full tools when volume present
            if self.volume_data is not None and hasattr(self, "align_tools_host"):
                self.align_tools_host.setVisible(True)
                self.set_align_tools_expanded(True, remember=False)
            elif hasattr(self, "align_sidebar"):
                # No volume — keep hidden
                if hasattr(self, "align_tools_host"):
                    self.align_tools_host.setVisible(False)
            # Offline: restore 3D volume so B2B gap / MES surface review works
            self.set_3d_volume_render_enabled(True, sync_button=True)

        QTimer.singleShot(50, self._reposition_visible_overlays)


    def _rebalance_main_splitter_for_tools(self, tools_expanded):
        """Give space freed by collapsing tools to MPR grid (+ stats when Online)."""
        if not hasattr(self, "_main_splitter") or self._main_splitter is None:
            return
        sp = self._main_splitter
        sizes = sp.sizes()
        if len(sizes) < 3:
            return
        total = sum(sizes)
        if total <= 0:
            total = max(sp.width(), 1600)

        tools_w = 240 if tools_expanded else 28
        stats_visible = (
            hasattr(self, "stats_panel")
            and self.stats_panel is not None
            and self.stats_panel.isVisible()
        )
        if stats_visible:
            # Prefer keeping stats readable; surplus goes to MPR
            stats_w = max(sizes[2], 340) if getattr(self, "_online_viewer_mode", False) else max(sizes[2], 280)
            # Cap stats so grid stays usable
            stats_w = min(stats_w, max(280, total // 3))
            grid_w = max(480, total - tools_w - stats_w)
            # If total shrank, shrink stats first
            if grid_w + tools_w + stats_w > total:
                stats_w = max(260, total - tools_w - 480)
                grid_w = max(400, total - tools_w - stats_w)
        else:
            stats_w = 0 if sizes[2] == 0 else max(0, sizes[2])
            grid_w = max(400, total - tools_w - stats_w)

        sp.setSizes([grid_w, tools_w, stats_w])

    def _create_align_sidebar(self):
        """Create a dedicated right sidebar panel for alignment controls.
        
        Design: Carl Zeiss ZEN Industrial — consistent with left sidebar.
        Uses centralized CSS classes: system-text-btn, system-toggle-btn.

        Groups (WINDOW LEVELING / CLIP / OBLIQUE MPR / ALIGNMENT) are
        independently collapsible via arrow headers (▼ open · ▶ closed).
        Top bar has a collapse control (especially useful in Online mode).
        """
        panel = QWidget()
        panel.setObjectName("AlignSidebar")
        panel.setFixedWidth(240)
        panel.setStyleSheet(f"""
            QWidget#AlignSidebar {{
                background-color: {SemiconductorTheme.BG_MEDIUM};
                border: none;
            }}
            QScrollArea#AlignSidebarScroll {{
                background: transparent;
                border: none;
            }}
            QWidget#AlignSidebarTitleBar {{
                background: {SemiconductorTheme.BG_DARK};
                border-bottom: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
            }}
            QToolButton#AlignToolsCollapseBtn {{
                background: transparent;
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 4px;
                color: {SemiconductorTheme.ACCENT_PRIMARY};
                padding: 2px 6px;
                font-weight: 700;
                font-size: 8pt;
            }}
            QToolButton#AlignToolsCollapseBtn:hover {{
                background: rgba(34, 174, 209, 0.15);
                border-color: {SemiconductorTheme.ACCENT_PRIMARY};
            }}
        """)

        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Title bar: VIEW TOOLS + collapse (free space for Online) ──
        title_bar = QWidget()
        title_bar.setObjectName("AlignSidebarTitleBar")
        title_bar.setFixedHeight(32)
        title_row = QHBoxLayout(title_bar)
        title_row.setContentsMargins(8, 0, 6, 0)
        title_row.setSpacing(6)

        title_lbl = QLabel("VIEW TOOLS")
        title_lbl.setStyleSheet(
            f"color: {SemiconductorTheme.ACCENT_PRIMARY}; font-size: 8.5pt;"
            f" font-weight: 700; letter-spacing: 0.8px; background: transparent;"
        )
        title_row.addWidget(title_lbl, 0, Qt.AlignVCenter)
        title_row.addStretch(1)

        self.align_tools_collapse_btn = QToolButton()
        self.align_tools_collapse_btn.setObjectName("AlignToolsCollapseBtn")
        self.align_tools_collapse_btn.setText("Hide ›")
        self.align_tools_collapse_btn.setToolTip(
            "Hide VIEW TOOLS to enlarge MPR / Statistics.\n"
            "Reopen anytime from the thin TOOLS rail on the right of the views."
        )
        self.align_tools_collapse_btn.setCursor(Qt.PointingHandCursor)
        self.align_tools_collapse_btn.clicked.connect(lambda: self.set_align_tools_expanded(False))
        title_row.addWidget(self.align_tools_collapse_btn, 0, Qt.AlignVCenter)
        outer.addWidget(title_bar)

        scroll = QScrollArea()
        scroll.setObjectName("AlignSidebarScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        content.setObjectName("AlignSidebarContent")
        content.setStyleSheet(f"background-color: {SemiconductorTheme.BG_MEDIUM};")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self._align_sidebar_sections = {}

        # ── Section: WINDOW LEVELING for 3 MPR faces (XY / YZ / XZ) ──
        _wl_body, wl_lay = self._make_sidebar_section(
            "WINDOW LEVELING",
            layout,
            expanded=True,
            tooltip=(
                "Window/Level for the three MPR views (XY / YZ / XZ).\n"
                "Click header to collapse / expand."
            ),
        )

        mpr_wl_hint = QLabel("MPR · XY / YZ / XZ (linked)")
        mpr_wl_hint.setStyleSheet(
            f"color: {SemiconductorTheme.PRIMARY_DEFAULT}; font-size: 7.5pt; font-weight: 600;"
        )
        wl_lay.addWidget(mpr_wl_hint)

        self.mpr_hist_widget = HistogramWLWidget()
        self.mpr_hist_widget.setMinimumHeight(100)
        self.mpr_hist_widget.setMaximumHeight(120)
        self.mpr_hist_widget.setToolTip(
            "Drag Min/Max markers to adjust window width and level.\n"
            "Applies simultaneously to all three MPR planes."
        )
        self.mpr_hist_widget.rangeChanged.connect(self._on_mpr_wl_hist_changed)
        wl_lay.addWidget(self.mpr_hist_widget)

        sel_lbl = QLabel("Selected range")
        sel_lbl.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 7.5pt;"
            f" font-weight: 600; letter-spacing: 1px; margin-top: 2px;"
        )
        wl_lay.addWidget(sel_lbl)

        mpr_wl_row = QHBoxLayout()
        mpr_wl_row.setSpacing(4)
        self.mpr_wl_min_spin = QSpinBox()
        self.mpr_wl_min_spin.setRange(0, 65535)
        self.mpr_wl_min_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.mpr_wl_min_spin.setAlignment(Qt.AlignCenter)
        self.mpr_wl_min_spin.setToolTip("Window min intensity (lower black level)")
        self.mpr_wl_min_spin.valueChanged.connect(self._on_mpr_wl_spin_changed)
        mpr_wl_row.addWidget(self.mpr_wl_min_spin)

        self.mpr_wl_max_spin = QSpinBox()
        self.mpr_wl_max_spin.setRange(0, 65535)
        self.mpr_wl_max_spin.setValue(65535)
        self.mpr_wl_max_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.mpr_wl_max_spin.setAlignment(Qt.AlignCenter)
        self.mpr_wl_max_spin.setToolTip("Window max intensity (upper white level)")
        self.mpr_wl_max_spin.valueChanged.connect(self._on_mpr_wl_spin_changed)
        mpr_wl_row.addWidget(self.mpr_wl_max_spin)
        wl_lay.addLayout(mpr_wl_row)

        mpr_wl_info_row = QHBoxLayout()
        # 1px vertical slack so hover border is not clipped by tight layout
        mpr_wl_info_row.setContentsMargins(0, 1, 0, 2)
        mpr_wl_info_row.setSpacing(4)
        self.mpr_wl_info_label = QLabel("W: 65535   L: 32767.5")
        self.mpr_wl_info_label.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_DISABLED}; font-size: 7pt;"
        )
        self.mpr_wl_info_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        mpr_wl_info_row.addWidget(self.mpr_wl_info_label, 1)

        self.btn_mpr_wl_reset = QPushButton("Reset")
        # Compact Reset — full border on hover (no clipped bottom edge)
        self._style_compact_reset_button(self.btn_mpr_wl_reset)
        self.btn_mpr_wl_reset.setToolTip("Reset MPR window/level to full data range")
        self.btn_mpr_wl_reset.clicked.connect(self._reset_mpr_wl_to_data_range)
        mpr_wl_info_row.addWidget(self.btn_mpr_wl_reset)
        wl_lay.addLayout(mpr_wl_info_row)

        # ── Section: CLIP (Dragonfly-style clip box, linked MPR↔3D) ──
        _clip_body, clip_lay = self._make_sidebar_section(
            "CLIP",
            layout,
            expanded=True,
            tooltip="Dragonfly-style clip box (MPR + 3D). Click to collapse / expand.",
        )

        # Toolbar: Enable clip + Reset  |  Grid size
        clip_toolbar = QHBoxLayout()
        clip_toolbar.setSpacing(4)
        clip_toolbar.setContentsMargins(0, 0, 0, 0)

        self.btn_df_clip_enable = QPushButton("  CLIP BOX")
        self.btn_df_clip_enable.setProperty("class", "system-toggle-btn")
        self.btn_df_clip_enable.setCheckable(True)
        self.btn_df_clip_enable.setCursor(Qt.PointingHandCursor)
        self.btn_df_clip_enable.setIcon(_ui_icon("layers.svg", self.btn_df_clip_enable))
        self.btn_df_clip_enable.setIconSize(QSize(14, 14))
        self.btn_df_clip_enable.setToolTip(
            "Dragonfly-style clip box:\n"
            "• ON: show interactive box (MPR + 3D) and edit crop\n"
            "• OFF: hide the box UI but KEEP the current crop\n"
            "• Use Reset to restore the full uncropped volume\n"
            "• Drag faces on 3D / edges on MPR while the tool is ON"
        )
        self.btn_df_clip_enable.clicked.connect(self._df_clip_on_enable_toggled)
        clip_toolbar.addWidget(self.btn_df_clip_enable, 1)

        self.btn_df_clip_reset = QPushButton()
        self.btn_df_clip_reset.setProperty("class", "icon-button")
        # Theme-tint refresh glyph — raw SVG stroke (#DDE7F2) blends into light
        # icon-button backgrounds; TEXT_PRIMARY keeps contrast in both themes.
        self.btn_df_clip_reset.setProperty("theme_icon", True)
        self.btn_df_clip_reset.setProperty("icon_name", "refresh.svg")
        self.btn_df_clip_reset.setProperty("icon_size_w", 14)
        self.btn_df_clip_reset.setProperty("icon_size_h", 14)
        self.btn_df_clip_reset.setIcon(
            _tinted_ui_icon(
                "refresh.svg",
                color=SemiconductorTheme.TEXT_PRIMARY,
                size=14,
                widget=self.btn_df_clip_reset,
            )
        )
        self.btn_df_clip_reset.setIconSize(QSize(14, 14))
        self.btn_df_clip_reset.setFixedSize(28, 26)
        self.btn_df_clip_reset.setCursor(Qt.PointingHandCursor)
        self.btn_df_clip_reset.setToolTip(
            "Reset clip to full volume (clears crop).\n"
            "Unlike turning CLIP BOX off, this restores the original extent."
        )
        self.btn_df_clip_reset.clicked.connect(self._df_clip_reset)
        clip_toolbar.addWidget(self.btn_df_clip_reset)
        clip_lay.addLayout(clip_toolbar)

        grid_size_row = QHBoxLayout()
        grid_size_row.setSpacing(4)
        grid_size_lbl = QLabel("Grid size:")
        grid_size_lbl.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;"
        )
        grid_size_row.addWidget(grid_size_lbl)
        self.df_clip_grid_size_spin = QDoubleSpinBox()
        self.df_clip_grid_size_spin.setRange(1.0, 10000.0)
        self.df_clip_grid_size_spin.setValue(self._df_clip_grid_size)
        self.df_clip_grid_size_spin.setDecimals(1)
        self.df_clip_grid_size_spin.setSingleStep(10.0)
        self.df_clip_grid_size_spin.setSuffix(" µm")
        self.df_clip_grid_size_spin.setToolTip("Spacing of grid lines on clip faces (world units)")
        self.df_clip_grid_size_spin.valueChanged.connect(self._df_clip_on_options_changed)
        grid_size_row.addWidget(self.df_clip_grid_size_spin, 1)
        clip_lay.addLayout(grid_size_row)

        self.chk_df_clip_keep = QCheckBox("Keep box when volume hidden")
        self.chk_df_clip_keep.setChecked(self._df_clip_keep_when_hidden)
        self.chk_df_clip_keep.setStyleSheet(f"color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 8pt;")
        self.chk_df_clip_keep.setToolTip("Keep the clip box visible even if the volume is turned off")
        self.chk_df_clip_keep.stateChanged.connect(self._df_clip_on_options_changed)
        clip_lay.addWidget(self.chk_df_clip_keep)

        self.chk_df_clip_grid_on_obj = QCheckBox("Display grid on object")
        self.chk_df_clip_grid_on_obj.setChecked(self._df_clip_display_grid_on_object)
        self.chk_df_clip_grid_on_obj.setStyleSheet(f"color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 8pt;")
        self.chk_df_clip_grid_on_obj.setToolTip("Draw grid lines on the clipped faces of the 3D volume")
        self.chk_df_clip_grid_on_obj.stateChanged.connect(self._df_clip_on_options_changed)
        clip_lay.addWidget(self.chk_df_clip_grid_on_obj)

        clip_opts = QHBoxLayout()
        clip_opts.setSpacing(6)
        self.chk_df_clip_grid_lines = QCheckBox("Grid lines")
        self.chk_df_clip_grid_lines.setChecked(self._df_clip_show_grid)
        self.chk_df_clip_grid_lines.setStyleSheet(f"color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 8pt;")
        self.chk_df_clip_grid_lines.stateChanged.connect(self._df_clip_on_options_changed)
        clip_opts.addWidget(self.chk_df_clip_grid_lines)

        self.chk_df_clip_borders = QCheckBox("Borders")
        self.chk_df_clip_borders.setChecked(self._df_clip_show_borders)
        self.chk_df_clip_borders.setStyleSheet(f"color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 8pt;")
        self.chk_df_clip_borders.stateChanged.connect(self._df_clip_on_options_changed)
        clip_opts.addWidget(self.chk_df_clip_borders)

        self.chk_df_clip_axes = QCheckBox("Axes")
        self.chk_df_clip_axes.setChecked(self._df_clip_show_axes)
        self.chk_df_clip_axes.setStyleSheet(f"color: {SemiconductorTheme.TEXT_PRIMARY}; font-size: 8pt;")
        self.chk_df_clip_axes.setToolTip("Show RGB axis tripod at the clip-box origin")
        self.chk_df_clip_axes.stateChanged.connect(self._df_clip_on_options_changed)
        clip_opts.addWidget(self.chk_df_clip_axes)
        clip_opts.addStretch()
        clip_lay.addLayout(clip_opts)

        # Bound sliders: X / Y / Z  min–max (percent of extent)
        self._df_clip_bound_ctrls = {}
        _axis_colors = {
            'x': ('#FF5555', 'X (Sag)'),
            'y': ('#55FF55', 'Y (Cor)'),
            'z': ('#5588FF', 'Z (Axi)'),
        }
        for axis, (color, label) in _axis_colors.items():
            axis_lbl = QLabel(label)
            axis_lbl.setStyleSheet(
                f"color: {color}; font-weight: 700; font-size: 8pt; margin-top: 4px;"
            )
            clip_lay.addWidget(axis_lbl)

            row = QHBoxLayout()
            row.setSpacing(3)
            lo = QSlider(Qt.Horizontal)
            lo.setRange(0, 1000)
            lo.setValue(0)
            lo.setStyleSheet(
                f"QSlider::groove:horizontal {{ height: 3px; background: #555; }}"
                f"QSlider::handle:horizontal {{ background: {color}; width: 10px; "
                f"margin: -4px 0; border-radius: 5px; }}"
            )
            hi = QSlider(Qt.Horizontal)
            hi.setRange(0, 1000)
            hi.setValue(1000)
            hi.setStyleSheet(
                f"QSlider::groove:horizontal {{ height: 3px; background: #555; }}"
                f"QSlider::handle:horizontal {{ background: {color}; width: 10px; "
                f"margin: -4px 0; border-radius: 5px; }}"
            )
            pct = QLabel("0–100%")
            pct.setFixedWidth(48)
            pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            pct.setStyleSheet(f"color: {color}; font-size: 7.5pt;")
            lo.valueChanged.connect(lambda v, a=axis: self._df_clip_on_bound_slider(a, 'lo', v))
            hi.valueChanged.connect(lambda v, a=axis: self._df_clip_on_bound_slider(a, 'hi', v))
            row.addWidget(lo, 1)
            row.addWidget(hi, 1)
            row.addWidget(pct)
            clip_lay.addLayout(row)
            self._df_clip_bound_ctrls[axis] = {'lo': lo, 'hi': hi, 'pct': pct}

        # ── Section: OBLIQUE MPR ──────────────────────────────
        _obl_body, obl_lay = self._make_sidebar_section(
            "OBLIQUE MPR",
            layout,
            expanded=True,
            tooltip="Oblique MPR rotation controls. Click to collapse / expand.",
        )

        # ── Toggle: Rotation Handles (Dragonfly-style) ────────
        self.btn_oblique_handles = QPushButton("  ROTATION HANDLES")
        self.btn_oblique_handles.setProperty("class", "system-toggle-btn")
        self.btn_oblique_handles.setCheckable(True)
        self.btn_oblique_handles.setCursor(Qt.PointingHandCursor)
        self.btn_oblique_handles.setToolTip(
            "Show Dragonfly-style rotation arrow handles on crosshair.\n"
            "Hover over an arrow to highlight it, then drag to rotate\n"
            "the oblique slicing plane around that axis."
        )
        self.btn_oblique_handles.clicked.connect(self._toggle_oblique_handles)
        obl_lay.addWidget(self.btn_oblique_handles)

        # ── Toggle: 2D ROTATE (moved from left SYSTEM sidebar) ──
        self.rotate2d_btn = QPushButton("  2D ROTATE")
        self.rotate2d_btn.setProperty("class", "system-toggle-btn")
        self.rotate2d_btn.setCheckable(True)
        self.rotate2d_btn.setCursor(Qt.PointingHandCursor)
        self.rotate2d_btn.setIcon(_ui_icon("rotate-clockwise-2.svg", self.rotate2d_btn))
        self.rotate2d_btn.setIconSize(QSize(16, 16))
        self.rotate2d_btn.setToolTip(
            "Toggle 2D rotation mode on crosshair views.\n"
            "Drag on a 2D view to roll the camera for that plane."
        )
        self.rotate2d_btn.clicked.connect(self.toggle_rotate_mode)
        obl_lay.addWidget(self.rotate2d_btn)

        # ── Section: VIEW NAV (P0/P1 MPR interaction) ─────
        _nav_body, nav_lay = self._make_sidebar_section(
            "VIEW NAV",
            layout,
            expanded=True,
            tooltip="MPR zoom / pan / crosshair behaviour. Click to collapse / expand.",
        )
        self.btn_link_mpr_nav = QPushButton("  LINK MPR ZOOM/PAN")
        self.btn_link_mpr_nav.setProperty("class", "system-toggle-btn")
        self.btn_link_mpr_nav.setCheckable(True)
        self.btn_link_mpr_nav.setCursor(Qt.PointingHandCursor)
        self.btn_link_mpr_nav.setToolTip(
            "When ON, zoom and pan on one MPR pane are mirrored to the other panes.\n"
            "Crosshair is always linked. Default OFF (Dragonfly-style independent views).\n\n"
            "Mouse:\n"
            "  Scroll = zoom at cursor\n"
            "  Ctrl+Scroll = change slice\n"
            "  Middle-drag or Shift+Left-drag = pan\n"
            "  Left = crosshair (click to place / drag handles)\n"
            "  Double-click = fit pane\n"
            "  Ctrl+0 = 1:1  ·  R = reset view  ·  Ctrl+Shift+0 = fit"
        )
        self.btn_link_mpr_nav.toggled.connect(self._toggle_link_mpr_nav)
        nav_lay.addWidget(self.btn_link_mpr_nav)

        nav_hint = QLabel(
            "Scroll zoom · Mid/Shift-pan · LMB crosshair\n"
            "Dbl-click fit · Ctrl+0 1:1 · R reset"
        )
        nav_hint.setWordWrap(True)
        nav_hint.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 7.5pt; margin-top: 2px;"
        )
        nav_lay.addWidget(nav_hint)

        # ── Section: ALIGNMENT ───────────────────────────
        _align_body, align_lay = self._make_sidebar_section(
            "ALIGNMENT",
            layout,
            expanded=True,
            tooltip="Auto / manual volume alignment. Click to collapse / expand.",
        )

        # ── Auto Align: symmetric 2-column grid ────────────────
        auto_lbl = QLabel("Auto")
        auto_lbl.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 7.5pt;"
            f" font-weight: 600; letter-spacing: 1px; margin-top: 4px;"
        )
        align_lay.addWidget(auto_lbl)

        auto_grid = QGridLayout()
        auto_grid.setContentsMargins(0, 0, 0, 0)
        auto_grid.setSpacing(6)
        auto_grid.setColumnStretch(0, 1)
        auto_grid.setColumnStretch(1, 1)

        self.btn_auto_align = QPushButton("  ALIGN")
        self.btn_auto_align.setProperty("class", "system-text-btn")
        self.btn_auto_align.setToolTip("Automatically align HBM chip volume using PCA & Symmetry analysis")
        self.btn_auto_align.setCursor(Qt.PointingHandCursor)
        self.btn_auto_align.clicked.connect(self.align_hbm_volume)
        auto_grid.addWidget(self.btn_auto_align, 0, 0)

        self.btn_reset_align = QPushButton("  RESET")
        self.btn_reset_align.setProperty("class", "system-text-btn")
        self.btn_reset_align.setToolTip("Reset volume and masks to their original unaligned states")
        self.btn_reset_align.setCursor(Qt.PointingHandCursor)
        self.btn_reset_align.clicked.connect(self.reset_hbm_alignment)
        self.btn_reset_align.setEnabled(False)
        auto_grid.addWidget(self.btn_reset_align, 0, 1)

        align_lay.addLayout(auto_grid)

        # ── Manual Align ───────────────────────────────────────
        manual_lbl = QLabel("Manual")
        manual_lbl.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 7.5pt;"
            f" font-weight: 600; letter-spacing: 1px; margin-top: 6px;"
        )
        align_lay.addWidget(manual_lbl)

        self.btn_manual_align_mode = QPushButton("  Manual Align: OFF")
        self.btn_manual_align_mode.setProperty("class", "system-toggle-btn")
        self.btn_manual_align_mode.setCheckable(True)
        self.btn_manual_align_mode.setCursor(Qt.PointingHandCursor)
        self.btn_manual_align_mode.setToolTip(
            "Enable Manual Align Mode:\n"
            "1. Drag crosshair handles on any 2D view\n"
            "2. Only crosshair rotates (no reslicing)\n"
            "3. Click 'Commit' to apply rotation"
        )
        self.btn_manual_align_mode.clicked.connect(self._toggle_manual_align_mode)
        align_lay.addWidget(self.btn_manual_align_mode)

        # Live angle display
        angle_grid = QGridLayout()
        angle_grid.setSpacing(2)
        angle_grid.setContentsMargins(4, 4, 4, 4)

        _axis_lbl_style = f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;"

        lbl_x = QLabel("Roll (X):")
        lbl_x.setStyleSheet(_axis_lbl_style)
        angle_grid.addWidget(lbl_x, 0, 0)
        self.manual_angle_x_label = QLabel("+0.00°")
        self.manual_angle_x_label.setStyleSheet("font-weight: bold; color: #FF3333; font-size: 9pt;")
        angle_grid.addWidget(self.manual_angle_x_label, 0, 1)

        lbl_y = QLabel("Pitch (Y):")
        lbl_y.setStyleSheet(_axis_lbl_style)
        angle_grid.addWidget(lbl_y, 1, 0)
        self.manual_angle_y_label = QLabel("+0.00°")
        self.manual_angle_y_label.setStyleSheet("font-weight: bold; color: #33FF33; font-size: 9pt;")
        angle_grid.addWidget(self.manual_angle_y_label, 1, 1)

        lbl_z = QLabel("Yaw (Z):")
        lbl_z.setStyleSheet(_axis_lbl_style)
        angle_grid.addWidget(lbl_z, 2, 0)
        self.manual_angle_z_label = QLabel("+0.00°")
        self.manual_angle_z_label.setStyleSheet("font-weight: bold; color: #5588FF; font-size: 9pt;")
        angle_grid.addWidget(self.manual_angle_z_label, 2, 1)

        align_lay.addLayout(angle_grid)

        # Commit + Cancel: symmetric 2-column
        commit_grid = QGridLayout()
        commit_grid.setContentsMargins(0, 0, 0, 0)
        commit_grid.setSpacing(6)
        commit_grid.setColumnStretch(0, 1)
        commit_grid.setColumnStretch(1, 1)

        self.btn_commit_manual_align = QPushButton("  COMMIT")
        self.btn_commit_manual_align.setProperty("class", "system-text-btn")
        self.btn_commit_manual_align.setProperty("role", "success")
        self.btn_commit_manual_align.setToolTip("Apply the current rotation permanently to the volume data")
        self.btn_commit_manual_align.setCursor(Qt.PointingHandCursor)
        self.btn_commit_manual_align.clicked.connect(self.commit_manual_alignment)
        self.btn_commit_manual_align.setEnabled(False)
        commit_grid.addWidget(self.btn_commit_manual_align, 0, 0)

        self.btn_cancel_manual_align = QPushButton("  CANCEL")
        self.btn_cancel_manual_align.setProperty("class", "system-text-btn")
        self.btn_cancel_manual_align.setToolTip("Discard the preview rotation and reset angles to 0")
        self.btn_cancel_manual_align.setCursor(Qt.PointingHandCursor)
        self.btn_cancel_manual_align.clicked.connect(self._cancel_manual_align)
        self.btn_cancel_manual_align.setEnabled(False)
        commit_grid.addWidget(self.btn_cancel_manual_align, 0, 1)

        align_lay.addLayout(commit_grid)

        # ── Export ─────────────────────────────────────────────
        export_lbl = QLabel("Export")
        export_lbl.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 7.5pt;"
            f" font-weight: 600; letter-spacing: 1px; margin-top: 6px;"
        )
        align_lay.addWidget(export_lbl)

        self.btn_save_aligned = QPushButton("  SAVE VOLUME")
        self.btn_save_aligned.setProperty("class", "system-text-btn")
        self.btn_save_aligned.setToolTip("Save the current volume data to a TIFF stack or RAW file")
        self.btn_save_aligned.setCursor(Qt.PointingHandCursor)
        self.btn_save_aligned.clicked.connect(self.save_aligned_volume)
        align_lay.addWidget(self.btn_save_aligned)

        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)
        return panel

