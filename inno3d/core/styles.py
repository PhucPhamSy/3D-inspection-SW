"""
styles.py
Modern UI/UX Stylesheet for 3D Semiconductor Viewer
Theme: High-Tech / Cyberpunk / Computer Vision
Color Palette: Deep Dark Blue/Black with Neon Cyan & Orange Accents
"""

import os

from PyQt5.QtGui import QPalette, QColor, QFont
from PyQt5.QtCore import Qt

class SemiconductorTheme:
    """Color palette for industrial semiconductor inspection theme (2026 Edition)
    
    Design reference: Carl Zeiss ZEN · Keyence VHX-7000 · Olympus LEXT OLS5100
    Elevation model follows Material Design Dark with wider tonal gaps.
    """

    # ── Elevation System (5 levels — wide gaps for clear hierarchy) ──
    EL0 = "#0a0f14"  # canvas / app bg       (deepest)
    EL1 = "#111a24"  # sidebar, panels        (+7 lightness)
    EL2 = "#182636"  # cards, groupboxes      (+7)
    EL3 = "#1f3044"  # active panel, dropdowns (+7)
    EL4 = "#273c52"  # hover states, tooltips  (+7)

    # Legacy mappings
    BG_DARK = EL0
    BG_MEDIUM = EL1
    BG_LIGHT = EL2
    BG_PANEL = EL3

    # ── Semantic Surfaces ──
    SURFACE_DEFAULT = EL1
    SURFACE_CARD = EL2
    SURFACE_MODAL = EL3
    SURFACE_OVERLAY = "rgba(0, 0, 0, 0.45)"

    ON_SURFACE_PRIMARY = "#e4ecf4"
    ON_SURFACE_SECONDARY = "#98a8b8"
    ON_SURFACE_DISABLED = "#5c6c7c"

    # ── Brand / Accent ──
    PRIMARY_DEFAULT = "#22aed1"
    PRIMARY_HOVER = "#3cc4e8"
    PRIMARY_PRESSED = "#1488a8"
    
    SUCCESS_DEFAULT = "#2fb27a"
    WARNING_DEFAULT = "#eda800"
    DANGER_DEFAULT = "#d9534f"

    # Legacy accent aliases
    ACCENT_PRIMARY = PRIMARY_DEFAULT
    ACCENT_SECONDARY = PRIMARY_HOVER
    ACCENT_TERTIARY = "#7a8d9e"
    ACCENT_SUCCESS = SUCCESS_DEFAULT
    ACCENT_WARNING = WARNING_DEFAULT
    ACCENT_ERROR = DANGER_DEFAULT

    # ── Text ──
    TEXT_PRIMARY = ON_SURFACE_PRIMARY
    TEXT_SECONDARY = ON_SURFACE_SECONDARY
    TEXT_DISABLED = ON_SURFACE_DISABLED
    TEXT_ON_ACCENT = "#0a0f14"

    # ── Borders ──
    BORDER_DEFAULT = "#293848"
    BORDER_ACTIVE = PRIMARY_DEFAULT
    BORDER_HOVER = PRIMARY_HOVER
    BORDER_GLOW = "#22aed144"

    # ── Buttons ──
    BTN_PRIMARY_BG = "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #28bfe0, stop:1 #1488a8)"
    BTN_PRIMARY_HOVER = PRIMARY_HOVER
    BTN_SECONDARY_BG = EL3
    BTN_SECONDARY_HOVER = EL4

    # ── Visualization ──
    CROSSHAIR_COLOR = "#ffb800"
    CLASS1_COLOR = "#00ff9d"
    CLASS2_COLOR = "#ff4d00"

    CURRENT_THEME = "dark"
    _PALETTES = {
        "dark": {
            # Elevation — widened gaps
            "EL0": "#0a0f14", "EL1": "#111a24", "EL2": "#182636", "EL3": "#1f3044", "EL4": "#273c52",
            "BG_DARK": "#0a0f14", "BG_MEDIUM": "#111a24", "BG_LIGHT": "#182636", "BG_PANEL": "#1f3044",
            # Surfaces
            "SURFACE_DEFAULT": "#111a24", "SURFACE_CARD": "#182636", "SURFACE_MODAL": "#1f3044",
            "SURFACE_OVERLAY": "rgba(0, 0, 0, 0.45)",
            # Brand
            "PRIMARY_DEFAULT": "#22aed1", "PRIMARY_HOVER": "#3cc4e8", "PRIMARY_PRESSED": "#1488a8",
            "SUCCESS_DEFAULT": "#2fb27a", "WARNING_DEFAULT": "#eda800", "DANGER_DEFAULT": "#d9534f",
            "ACCENT_PRIMARY": "#22aed1", "ACCENT_SECONDARY": "#3cc4e8", "ACCENT_TERTIARY": "#7a8d9e",
            "ACCENT_SUCCESS": "#2fb27a", "ACCENT_WARNING": "#eda800", "ACCENT_ERROR": "#d9534f",
            # Text
            "TEXT_PRIMARY": "#e4ecf4", "TEXT_SECONDARY": "#98a8b8", "TEXT_DISABLED": "#5c6c7c",
            "TEXT_ON_ACCENT": "#0a0f14",
            # Borders
            "BORDER_DEFAULT": "#293848", "BORDER_ACTIVE": "#22aed1", "BORDER_HOVER": "#3cc4e8",
            "BORDER_GLOW": "#22aed144",
            # Buttons
            "BTN_PRIMARY_BG": "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #28bfe0, stop:1 #1488a8)",
            "BTN_PRIMARY_HOVER": "#3cc4e8", "BTN_SECONDARY_BG": "#1f3044", "BTN_SECONDARY_HOVER": "#273c52",
            # Vis
            "CROSSHAIR_COLOR": "#ffb800", "CLASS1_COLOR": "#00ff9d", "CLASS2_COLOR": "#ff4d00",
        },
        "light": {
            # Elevation — warmer / more neutral grays
            "EL0": "#f4f5f7", "EL1": "#eaecf0", "EL2": "#dde1e8", "EL3": "#cfd5de", "EL4": "#bfc8d2",
            "BG_DARK": "#f4f5f7", "BG_MEDIUM": "#eaecf0", "BG_LIGHT": "#dde1e8", "BG_PANEL": "#cfd5de",
            # Surfaces
            "SURFACE_DEFAULT": "#eaecf0", "SURFACE_CARD": "#dde1e8", "SURFACE_MODAL": "#cfd5de",
            "SURFACE_OVERLAY": "rgba(0, 0, 0, 0.12)",
            # Brand — desaturated 10% for softer appearance
            "PRIMARY_DEFAULT": "#0e87a8", "PRIMARY_HOVER": "#18a8d0", "PRIMARY_PRESSED": "#096980",
            "SUCCESS_DEFAULT": "#1a8a56", "WARNING_DEFAULT": "#ae7200", "DANGER_DEFAULT": "#be3a3a",
            "ACCENT_PRIMARY": "#0e87a8", "ACCENT_SECONDARY": "#18a8d0", "ACCENT_TERTIARY": "#6b7e90",
            "ACCENT_SUCCESS": "#1a8a56", "ACCENT_WARNING": "#ae7200", "ACCENT_ERROR": "#be3a3a",
            # Text
            "TEXT_PRIMARY": "#1a2836", "TEXT_SECONDARY": "#4d6074", "TEXT_DISABLED": "#7c8e9e",
            "TEXT_ON_ACCENT": "#ffffff",
            # Borders
            "BORDER_DEFAULT": "#c0ccd8", "BORDER_ACTIVE": "#0e87a8", "BORDER_HOVER": "#18a8d0",
            "BORDER_GLOW": "#0e87a844",
            # Buttons
            "BTN_PRIMARY_BG": "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #18a6cc, stop:1 #0e87a8)",
            "BTN_PRIMARY_HOVER": "#18a8d0", "BTN_SECONDARY_BG": "#dde1e8", "BTN_SECONDARY_HOVER": "#cfd5de",
            # Vis
            "CROSSHAIR_COLOR": "#c88200", "CLASS1_COLOR": "#00a060", "CLASS2_COLOR": "#cc4818",
        },
    }

    @classmethod
    def set_theme(cls, theme_name="dark"):
        mode = (theme_name or "dark").strip().lower()
        if mode not in cls._PALETTES:
            mode = "dark"
        palette = cls._PALETTES[mode]
        for key, value in palette.items():
            setattr(cls, key, value)
        cls.CURRENT_THEME = mode

    @classmethod
    def is_light(cls):
        return cls.CURRENT_THEME == "light"

    @classmethod
    def vtk_bg(cls):
        """Return (r, g, b) float tuple for VTK renderer background."""
        if cls.is_light():
            return (0.957, 0.961, 0.969)  # #f4f5f7
        return (0.039, 0.059, 0.078)       # #0a0f14

    @classmethod
    def mpl_colors(cls):
        """Return dict of matplotlib styling colors for the current theme."""
        if cls.is_light():
            return dict(
                face="#eaecf0", axes="#cfd5de", text="#1a2836",
                grid_alpha=0.35, spine="#a8b5c2",
                legend_face="#dde1e8", legend_text="#1a2836",
            )
        return dict(
            face="#111a24", axes="#111a24", text="#e4ecf4",
            grid_alpha=0.25, spine="#293848",
            legend_face="#182636", legend_text="#e4ecf4",
        )

    @classmethod
    def table_stylesheet(cls):
        """Return a standardized QTableWidget stylesheet for the current theme."""
        is_light = cls.is_light()
        hdr_bg = cls.BG_PANEL
        hdr_text = cls.ACCENT_PRIMARY
        alt_row = "rgba(0, 0, 0, 0.02)" if is_light else "rgba(255, 255, 255, 0.015)"
        sel_bg = f"rgba({','.join(str(int(cls.PRIMARY_DEFAULT.lstrip('#')[i:i+2],16)) for i in (0,2,4))}, 0.18)"
        grid_color = cls.BORDER_DEFAULT
        return f"""
            QTableWidget, QTableView {{
                background-color: {cls.BG_LIGHT};
                alternate-background-color: {alt_row};
                color: {cls.TEXT_PRIMARY};
                gridline-color: {grid_color};
                border: 1px solid {cls.BORDER_DEFAULT};
                border-radius: 4px;
                font-size: 8.5pt;
            }}
            QTableWidget::item, QTableView::item {{
                padding: 3px 6px;
            }}
            QTableWidget::item:selected, QTableView::item:selected {{
                background-color: {sel_bg};
                color: {cls.TEXT_PRIMARY};
            }}
            QHeaderView::section {{
                background-color: {hdr_bg};
                color: {hdr_text};
                border: none;
                border-bottom: 2px solid {cls.ACCENT_PRIMARY};
                border-right: 1px solid {grid_color};
                padding: 4px 6px;
                padding-right: 22px;  /* room for Excel funnel chip */
                font-weight: 700;
                font-size: 8pt;
            }}
            QHeaderView::section:last {{
                border-right: none;
            }}
        """

    @classmethod
    def sidebar_section_style(cls):
        """Return inline stylesheet for sidebar section header labels."""
        bdr = cls.BORDER_DEFAULT
        return (
            f"color: {cls.TEXT_DISABLED}; font-weight: 700; font-size: 7.5pt;"
            f" letter-spacing: 2px; text-transform: uppercase;"
            f" margin-top: 8px; margin-bottom: 4px; padding-bottom: 5px;"
            f" border-bottom: 1px solid {bdr};"
        )




def get_main_stylesheet():
    """Get modern application stylesheet — fully theme-aware (2026 Edition)"""

    T = SemiconductorTheme
    is_light = T.is_light()

    # ── Derived tokens ──────────────────────────────────────────────────
    # Buttons
    btn_bg      = "rgba(226, 234, 243, 235)" if is_light else "rgba(36, 50, 64, 220)"
    btn_hover   = "rgba(210, 223, 236, 240)" if is_light else "rgba(44, 61, 79, 235)"
    btn_press   = "rgba(195, 211, 227, 245)" if is_light else "rgba(28, 40, 53, 240)"
    btn_dis     = "rgba(210, 220, 232, 180)" if is_light else "rgba(20, 28, 38, 170)"
    btn_dis_bdr = "rgba(180, 193, 207, 0.55)" if is_light else "rgba(43, 54, 66, 0.4)"
    btn_h_text  = T.TEXT_PRIMARY

    # Inputs
    inp_bg      = "rgba(255, 255, 255, 235)" if is_light else "rgba(5, 5, 16, 200)"
    inp_focus   = "rgba(244, 249, 255, 245)" if is_light else "rgba(10, 15, 30, 230)"
    focus_ring  = f"rgba({','.join(str(int(T.PRIMARY_DEFAULT.lstrip('#')[i:i+2],16)) for i in (0,2,4))}, 0.6)"

    # Checkboxes
    chk_bg      = "rgba(255, 255, 255, 235)" if is_light else "rgba(5, 5, 16, 200)"
    chk_bdr     = "rgba(165, 180, 198, 0.8)" if is_light else "rgba(45, 55, 72, 0.8)"
    chk_hov_bdr = f"rgba({','.join(str(int(T.PRIMARY_DEFAULT.lstrip('#')[i:i+2],16)) for i in (0,2,4))}, 0.5)"
    chk_hov_bg  = f"rgba({','.join(str(int(T.PRIMARY_DEFAULT.lstrip('#')[i:i+2],16)) for i in (0,2,4))}, 0.05)"

    # Groups
    grp_bg      = "rgba(250, 252, 255, 235)" if is_light else "rgba(19, 22, 41, 200)"
    grp_bdr     = "rgba(165, 180, 198, 0.6)" if is_light else "rgba(45, 55, 72, 0.6)"

    # Scrollbars
    sb_track    = "rgba(225, 232, 240, 180)" if is_light else "rgba(11, 17, 24, 160)"
    sb_handle   = ("qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                   "stop:0 rgba(165,185,208,200), stop:1 rgba(145,168,195,200))"
                   if is_light else
                   "qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                   "stop:0 rgba(60,85,115,200), stop:1 rgba(80,110,145,200))")
    sb_handle_h = ("qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                   f"stop:0 {T.PRIMARY_DEFAULT}, stop:1 {T.PRIMARY_HOVER})")
    sb_handle_p = T.PRIMARY_DEFAULT
    sb_btn      = "rgba(230, 237, 244, 220)" if is_light else "rgba(22, 33, 45, 220)"
    sb_btn_bdr  = "rgba(180, 193, 207, 0.5)" if is_light else "rgba(43, 54, 66, 0.5)"
    sb_arrow_c  = "rgba(80, 100, 125, 200)" if is_light else "rgba(160, 175, 190, 200)"

    # Combo
    combo_popup = T.BG_LIGHT if is_light else T.BG_PANEL

    # Nav tabs
    nav_text    = "rgba(20, 33, 46, 0.62)" if is_light else "rgba(255, 255, 255, 0.5)"
    nav_h_text  = "rgba(20, 33, 46, 0.92)" if is_light else "rgba(255, 255, 255, 0.85)"
    nav_h_bg    = f"rgba({','.join(str(int(T.PRIMARY_DEFAULT.lstrip('#')[i:i+2],16)) for i in (0,2,4))}, 0.10)"
    nav_c_bg    = f"rgba({','.join(str(int(T.PRIMARY_DEFAULT.lstrip('#')[i:i+2],16)) for i in (0,2,4))}, 0.14)"

    # Misc
    tooltip_bg  = "rgba(255, 255, 255, 245)" if is_light else "rgba(19, 22, 41, 240)"
    tooltip_bdr = f"rgba({','.join(str(int(T.PRIMARY_DEFAULT.lstrip('#')[i:i+2],16)) for i in (0,2,4))}, 0.45)"
    progress_bg = "rgba(255, 255, 255, 235)" if is_light else "rgba(5, 5, 16, 200)"

    # Icon button
    icon_bg     = "rgba(220, 228, 237, 0.80)" if is_light else "rgba(22, 31, 42, 0.65)"
    icon_bdr    = "rgba(175, 192, 210, 0.8)"  if is_light else "rgba(43, 54, 66, 0.8)"
    icon_hov_bg = f"rgba({','.join(str(int(T.PRIMARY_DEFAULT.lstrip('#')[i:i+2],16)) for i in (0,2,4))}, 0.12)"
    icon_hov_bdr= f"rgba({','.join(str(int(T.PRIMARY_DEFAULT.lstrip('#')[i:i+2],16)) for i in (0,2,4))}, 0.5)"
    icon_h_text = T.TEXT_PRIMARY if is_light else "#ffffff"

    # Role danger
    dgr_bg      = "rgba(217, 83, 79, 0.12)" if is_light else "rgba(217, 83, 79, 0.18)"
    dgr_text    = T.DANGER_DEFAULT if is_light else "#ffbab8"
    dgr_bdr     = "rgba(217, 83, 79, 0.45)" if is_light else "rgba(217, 83, 79, 0.55)"
    dgr_hov_bg  = "rgba(217, 83, 79, 0.22)" if is_light else "rgba(217, 83, 79, 0.3)"
    dgr_hov_txt = T.DANGER_DEFAULT if is_light else "#ffd5d4"

    # Role toggle
    toggle_chk  = f"rgba({','.join(str(int(T.PRIMARY_DEFAULT.lstrip('#')[i:i+2],16)) for i in (0,2,4))}, 0.2)"
    toggle_bdr  = f"rgba({','.join(str(int(T.PRIMARY_DEFAULT.lstrip('#')[i:i+2],16)) for i in (0,2,4))}, 0.75)"

    # Load icon button
    load_bg     = "rgba(230, 237, 244, 0.85)" if is_light else "rgba(24, 34, 47, 0.8)"
    load_bdr    = "rgba(175, 192, 210, 0.9)" if is_light else "rgba(43, 54, 66, 0.9)"
    load_hov    = "rgba(215, 228, 240, 0.95)" if is_light else "rgba(31, 47, 64, 0.92)"

    # Sidebar
    side_hov_bg = "rgba(200, 212, 225, 0.55)" if is_light else "rgba(26, 31, 53, 180)"
    side_chk_bg = f"rgba({','.join(str(int(T.PRIMARY_DEFAULT.lstrip('#')[i:i+2],16)) for i in (0,2,4))}, 0.08)"

    # Ghost hover
    ghost_hov_bg  = f"rgba({','.join(str(int(T.PRIMARY_DEFAULT.lstrip('#')[i:i+2],16)) for i in (0,2,4))}, 0.12)"
    ghost_hov_bdr = f"rgba({','.join(str(int(T.PRIMARY_DEFAULT.lstrip('#')[i:i+2],16)) for i in (0,2,4))}, 0.45)"

    # System / VIEW TOOLS button hover — shared left + right sidebars
    # (matches AlignTools "Hide" / toggle-checked cyan wash, not flat EL3 gray)
    _pr = ",".join(str(int(T.PRIMARY_DEFAULT.lstrip("#")[i : i + 2], 16)) for i in (0, 2, 4))
    sys_hov_bg = f"rgba({_pr}, 0.15)"
    sys_hov_bg_strong = f"rgba({_pr}, 0.22)"

    # List-btn hover
    list_hov_bg = f"rgba({','.join(str(int(T.PRIMARY_DEFAULT.lstrip('#')[i:i+2],16)) for i in (0,2,4))}, 0.08)"
    list_hov_bdr= f"rgba({','.join(str(int(T.PRIMARY_DEFAULT.lstrip('#')[i:i+2],16)) for i in (0,2,4))}, 0.45)"

    # Clear button
    clr_text    = T.DANGER_DEFAULT if is_light else "#ffb6b3"
    clr_h_text  = T.DANGER_DEFAULT if is_light else "#ffd5d4"

    # Capsule disabled
    cap_dis_bg  = "rgba(225, 230, 238, 150)" if is_light else "rgba(10, 12, 25, 150)"
    cap_dis_bdr = "rgba(180, 193, 207, 0.2)" if is_light else "rgba(45, 55, 72, 0.2)"

    # Slider groove
    sl_groove   = "rgba(170, 185, 204, 0.8)" if is_light else "rgba(45, 55, 72, 0.8)"

    return f"""
    /* ==================== GLOBAL ==================== */
    * {{
        font-family: 'Inter', 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
        font-size: 9pt;
    }}

    QMainWindow {{
        background-color: {T.BG_DARK};
    }}

    QWidget {{
        background: none;
        color: {T.TEXT_PRIMARY};
    }}

    QStackedWidget {{
        background: {T.BG_DARK};
    }}

    QSplitter {{
        background: {T.BG_DARK};
    }}

    /* ==================== TAB WIDGET ==================== */
    QTabWidget::pane {{
        border: 1px solid {T.BORDER_DEFAULT};
        background: {T.BG_MEDIUM};
        border-radius: 4px;
        top: -1px;
    }}

    QTabBar::tab {{
        background: {T.BG_DARK};
        color: {T.TEXT_SECONDARY};
        padding: 8px 16px;
        min-width: 160px;
        border: 1px solid transparent;
        border-bottom: 2px solid {T.BORDER_DEFAULT};
        font-weight: 600;
        font-size: 10pt;
        margin-right: 2px;
    }}

    QTabBar::tab:selected {{
        color: {T.ACCENT_PRIMARY};
        background: {T.BG_MEDIUM};
        border-bottom: 2px solid {T.ACCENT_PRIMARY};
    }}

    QTabBar::tab:hover:!selected {{
        color: {T.TEXT_PRIMARY};
        background: {T.BG_LIGHT};
    }}

    /* ==================== BUTTONS ==================== */
    QPushButton {{
        background-color: {btn_bg};
        border: 1px solid {T.BORDER_DEFAULT};
        color: {T.TEXT_PRIMARY};
        border-radius: 4px;
        padding: 6px 14px;
        font-weight: 600;
        font-size: 9pt;
        min-height: 30px; /* Secondary default size */
    }}

    QPushButton:hover {{
        background-color: {btn_hover};
        border: 1px solid {T.BORDER_HOVER};
        color: {btn_h_text};
    }}

    QPushButton:pressed {{
        background-color: {btn_press};
        border: 1px solid {T.BORDER_ACTIVE};
    }}

    QToolButton {{
        background-color: {btn_bg};
        border: 1px solid {T.BORDER_DEFAULT};
        color: {T.TEXT_PRIMARY};
        border-radius: 5px;
        padding: 3px 6px;
    }}

    QToolButton:hover {{
        background-color: {btn_hover};
        border: 1px solid {T.BORDER_HOVER};
        color: {btn_h_text};
    }}

    QToolButton:pressed {{
        background-color: {btn_press};
        border: 1px solid {T.BORDER_ACTIVE};
    }}

    /* ── Role: primary ── */
    QPushButton[role="primary"], QPushButton[class="primary"] {{
        background: {T.BTN_PRIMARY_BG};
        color: {T.TEXT_ON_ACCENT};
        border: 1px solid {T.PRIMARY_DEFAULT};
        font-weight: 700;
        min-height: 34px;
    }}

    QPushButton[role="primary"]:hover, QPushButton[class="primary"]:hover {{
        background: {T.BTN_PRIMARY_HOVER};
        color: {T.TEXT_ON_ACCENT};
    }}

    /* ── Role: cta ── */
    QPushButton[role="cta"], QPushButton[class="cta"] {{
        background: {T.BTN_PRIMARY_BG};
        color: {T.TEXT_ON_ACCENT};
        border: 1px solid {T.PRIMARY_DEFAULT};
        font-weight: 800;
        min-height: 40px;
        font-size: 10pt;
    }}

    QPushButton[role="cta"]:hover, QPushButton[class="cta"]:hover {{
        background: {T.BTN_PRIMARY_HOVER};
        color: {T.TEXT_ON_ACCENT};
    }}

    /* ── Role: secondary ── */
    QPushButton[role="secondary"] {{
        background: {T.BTN_SECONDARY_BG};
        color: {T.TEXT_PRIMARY};
        border: 1px solid {T.BORDER_DEFAULT};
    }}

    QPushButton[role="secondary"]:hover {{
        background: {T.BTN_SECONDARY_HOVER};
    }}

    /* ── Role: danger ── */
    QPushButton[role="danger"], QPushButton[class="danger"] {{
        background: {dgr_bg};
        color: {dgr_text};
        border: 1px solid {dgr_bdr};
    }}

    QPushButton[role="danger"]:hover, QPushButton[class="danger"]:hover {{
        background: {dgr_hov_bg};
        color: {dgr_hov_txt};
    }}

    /* ── Role: ghost ── */
    QPushButton[role="ghost"] {{
        background: transparent;
        border: 1px solid transparent;
        color: {T.TEXT_SECONDARY};
        min-height: 28px;
    }}

    QPushButton[role="ghost"]:hover {{
        background: {ghost_hov_bg};
        border: 1px solid {ghost_hov_bdr};
        color: {T.TEXT_PRIMARY};
    }}

    /* ── Disabled global override ── */
    QPushButton:disabled {{
        background-color: {btn_dis};
        color: {T.TEXT_DISABLED};
        border: 1px solid {btn_dis_bdr};
    }}

    /* ── Role: toggle ── */
    QPushButton[role="toggle"]:checked {{
        background: {toggle_chk};
        border: 1px solid {toggle_bdr};
        color: {T.TEXT_PRIMARY};
    }}

    /* ── Sizes ── */
    QPushButton[size="sm"] {{
        min-height: 28px;
        padding: 4px 10px;
        font-size: 8.5pt;
    }}
    QPushButton[size="md"] {{
        min-height: 32px;
    }}
    QPushButton[size="lg"] {{
        min-height: 40px;
        font-size: 9.5pt;
    }}

    /* ── Icon Buttons ── */
    QPushButton[class="icon-button"], QToolButton[class="icon-button"] {{
        background: {icon_bg};
        color: {T.TEXT_SECONDARY};
        border: 1px solid {icon_bdr};
        border-radius: 5px;
        min-width: 24px;
        min-height: 24px;
        padding: 2px;
    }}

    QPushButton[class="icon-button"]:hover, QToolButton[class="icon-button"]:hover {{
        background: {icon_hov_bg};
        border: 1px solid {icon_hov_bdr};
        color: {icon_h_text};
    }}

    QPushButton[class="icon-button"]:pressed, QToolButton[class="icon-button"]:pressed {{
        background: {ghost_hov_bg};
    }}

    /* ── Load Icon Button ── */
    QPushButton[class="load-icon-btn"], QToolButton[class="load-icon-btn"] {{
        background: {load_bg};
        border: 1px solid {load_bdr};
        border-radius: 6px;
        min-height: 38px;
        min-width: 40px;
        padding: 0;
    }}

    QPushButton[class="load-icon-btn"]:hover, QToolButton[class="load-icon-btn"]:hover {{
        border: 1px solid {icon_hov_bdr};
        background: {load_hov};
    }}

    QPushButton[class="load-icon-btn"][loaded="true"], QToolButton[class="load-icon-btn"][loaded="true"] {{
        border: 1px solid rgba(47, 178, 122, 0.9);
        background: rgba(47, 178, 122, 0.22);
        color: {T.TEXT_PRIMARY};
    }}

    /* ==================== INPUTS ==================== */
    QLineEdit, QSpinBox, QDoubleSpinBox {{
        background-color: {inp_bg};
        border: 1px solid {T.BORDER_DEFAULT};
        border-radius: 5px;
        padding: 5px 8px;
        color: {T.TEXT_PRIMARY};
        selection-background-color: {T.ACCENT_PRIMARY};
        selection-color: {T.TEXT_ON_ACCENT};
    }}

    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 1px solid {focus_ring};
        background-color: {inp_focus};
    }}

    /* ==================== SLIDERS ==================== */
    QSlider::groove:horizontal {{
        border: none;
        height: 4px;
        background: {sl_groove};
        border-radius: 2px;
    }}

    QSlider::sub-page:horizontal {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {T.PRIMARY_PRESSED},
            stop:1 {T.PRIMARY_DEFAULT});
        border-radius: 2px;
    }}

    QSlider::handle:horizontal {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #ffffff, stop:1 #c8dce8);
        border: 2px solid {chk_hov_bdr};
        width: 14px;
        height: 14px;
        margin: -6px 0;
        border-radius: 8px;
    }}

    QSlider::handle:horizontal:hover {{
        background: {T.PRIMARY_DEFAULT};
        border: 2px solid {T.PRIMARY_HOVER};
        width: 16px;
        height: 16px;
        margin: -7px 0;
        border-radius: 9px;
    }}

    /* ==================== CHECKBOX ==================== */
    QCheckBox {{
        spacing: 8px;
        color: {T.TEXT_PRIMARY};
        font-size: 9pt;
    }}

    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border: 2px solid {chk_bdr};
        border-radius: 4px;
        background: {chk_bg};
    }}

    QCheckBox::indicator:unchecked:hover {{
        border-color: {chk_hov_bdr};
        background: {chk_hov_bg};
    }}

    QCheckBox::indicator:checked {{
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 {T.PRIMARY_DEFAULT}, stop:1 {T.PRIMARY_PRESSED});
        border-color: {T.PRIMARY_DEFAULT};
    }}

    /* ==================== GROUPS ==================== */
    QGroupBox {{
        background-color: {grp_bg};
        border: 1px solid {grp_bdr};
        border-radius: 8px;
        margin-top: 24px;
        font-weight: bold;
        color: {T.ACCENT_PRIMARY};
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 8px;
        left: 10px;
        top: 0px;
    }}

    /* ==================== DIALOGS ==================== */
    QDialog, QMessageBox {{
        background-color: {T.BG_DARK};
        color: {T.TEXT_PRIMARY};
    }}

    QMessageBox QLabel {{
        color: {T.TEXT_PRIMARY};
        font-size: 10pt;
    }}

    QMessageBox QPushButton {{
        min-width: 80px;
    }}

    /* ==================== SCROLLBAR (Vertical) ==================== */
    QScrollBar:vertical {{
        background: {sb_track};
        width: 14px;
        margin: 14px 0 14px 0;
        border: 1px solid {sb_btn_bdr};
        border-radius: 3px;
    }}
    QScrollBar::handle:vertical {{
        background: {sb_handle};
        min-height: 28px;
        border-radius: 3px;
        border: 1px solid rgba(100, 130, 165, 80);
    }}
    QScrollBar::handle:vertical:hover {{
        background: {sb_handle_h};
        border: 1px solid {T.PRIMARY_DEFAULT};
    }}
    QScrollBar::handle:vertical:pressed {{
        background: {sb_handle_p};
    }}
    QScrollBar::sub-line:vertical {{
        border: 1px solid {sb_btn_bdr};
        background: {sb_btn};
        height: 14px;
        subcontrol-position: top;
        subcontrol-origin: margin;
        border-top-left-radius: 3px;
        border-top-right-radius: 3px;
    }}
    QScrollBar::add-line:vertical {{
        border: 1px solid {sb_btn_bdr};
        background: {sb_btn};
        height: 14px;
        subcontrol-position: bottom;
        subcontrol-origin: margin;
        border-bottom-left-radius: 3px;
        border-bottom-right-radius: 3px;
    }}
    QScrollBar::sub-line:vertical:hover, QScrollBar::add-line:vertical:hover {{
        background: {icon_hov_bg};
        border-color: {icon_hov_bdr};
    }}
    QScrollBar::up-arrow:vertical {{
        width: 8px; height: 6px; image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-bottom: 6px solid {sb_arrow_c};
    }}
    QScrollBar::down-arrow:vertical {{
        width: 8px; height: 6px; image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 6px solid {sb_arrow_c};
    }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: none;
    }}

    /* ==================== SCROLLBAR (Horizontal) ==================== */
    QScrollBar:horizontal {{
        background: {sb_track};
        height: 14px;
        margin: 0 14px 0 14px;
        border: 1px solid {sb_btn_bdr};
        border-radius: 3px;
    }}
    QScrollBar::handle:horizontal {{
        background: {sb_handle};
        min-width: 28px;
        border-radius: 3px;
        border: 1px solid rgba(100, 130, 165, 80);
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {sb_handle_h};
        border: 1px solid {T.PRIMARY_DEFAULT};
    }}
    QScrollBar::handle:horizontal:pressed {{
        background: {sb_handle_p};
    }}
    QScrollBar::sub-line:horizontal {{
        border: 1px solid {sb_btn_bdr};
        background: {sb_btn};
        width: 14px;
        subcontrol-position: left;
        subcontrol-origin: margin;
        border-top-left-radius: 3px;
        border-bottom-left-radius: 3px;
    }}
    QScrollBar::add-line:horizontal {{
        border: 1px solid {sb_btn_bdr};
        background: {sb_btn};
        width: 14px;
        subcontrol-position: right;
        subcontrol-origin: margin;
        border-top-right-radius: 3px;
        border-bottom-right-radius: 3px;
    }}
    QScrollBar::sub-line:horizontal:hover, QScrollBar::add-line:horizontal:hover {{
        background: {icon_hov_bg};
        border-color: {icon_hov_bdr};
    }}
    QScrollBar::left-arrow:horizontal {{
        width: 6px; height: 8px; image: none;
        border-top: 4px solid transparent;
        border-bottom: 4px solid transparent;
        border-right: 6px solid {sb_arrow_c};
    }}
    QScrollBar::right-arrow:horizontal {{
        width: 6px; height: 8px; image: none;
        border-top: 4px solid transparent;
        border-bottom: 4px solid transparent;
        border-left: 6px solid {sb_arrow_c};
    }}
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
        background: none;
    }}

    /* ==================== COMBOBOX ==================== */
    QComboBox {{
        background-color: {inp_bg};
        border: 1px solid {T.BORDER_DEFAULT};
        border-radius: 5px;
        padding: 5px 8px;
        color: {T.TEXT_PRIMARY};
        min-height: 20px;
    }}

    QComboBox:hover {{
        border-color: {chk_hov_bdr};
    }}

    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}

    QComboBox QAbstractItemView {{
        background: {combo_popup};
        border: 1px solid {T.BORDER_DEFAULT};
        color: {T.TEXT_PRIMARY};
        selection-background-color: {toggle_chk};
        selection-color: {T.TEXT_PRIMARY};
    }}

    /* ==================== CUSTOM CLASSES ==================== */
    .toolbar {{
        background-color: {T.BG_MEDIUM};
        border-bottom: 1px solid {T.BORDER_DEFAULT};
    }}

    .heading {{
        font-family: 'Segoe UI', sans-serif;
        font-size: 18pt;
        font-weight: bold;
        color: {T.ACCENT_PRIMARY};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .status-bar {{
        background-color: {T.BG_MEDIUM};
        color: {T.TEXT_SECONDARY};
        border-top: 1px solid {T.BORDER_DEFAULT};
    }}

    .panel-header {{
        background-color: {T.BG_LIGHT};
        border-bottom: 1px solid {T.BORDER_DEFAULT};
        border-radius: 4px 4px 0 0;
        padding: 5px;
    }}

    /* ==================== SIDEBAR BUTTONS ==================== */
    QPushButton[class="sidebar-btn"] {{
        background-color: transparent;
        border: none;
        border-left: 3px solid transparent;
        color: {T.TEXT_SECONDARY};
        text-align: left;
        padding: 12px 15px;
        font-size: 10pt;
        font-weight: 600;
        border-radius: 0;
    }}

    QPushButton[class="sidebar-btn"]:hover {{
        background-color: {side_hov_bg};
        color: {T.TEXT_PRIMARY};
    }}

    QPushButton[class="sidebar-btn"][checked="true"] {{
        background-color: {side_chk_bg};
        border-left: 3px solid {T.ACCENT_PRIMARY};
        color: {T.ACCENT_PRIMARY};
        font-weight: bold;
    }}

    /* ==================== LIST BUTTONS ==================== */
    QPushButton[class="list-btn"] {{
        background-color: transparent;
        border: none;
        border-left: 2px solid transparent;
        color: {T.TEXT_PRIMARY};
        text-align: left;
        padding: 8px 4px 8px 8px;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 400;
        font-size: 9.5pt;
        border-radius: 0;
    }}
    QPushButton[class="list-btn"]:hover {{
        color: {T.TEXT_PRIMARY};
        background-color: {list_hov_bg};
        border-left: 2px solid {list_hov_bdr};
    }}
    QPushButton[class="list-btn"][loaded="true"] {{
        color: {T.ACCENT_PRIMARY};
        font-weight: 600;
        border-left: 2px solid {T.ACCENT_PRIMARY};
    }}
    QPushButton[class="list-btn"]:checked {{
        color: {T.ACCENT_PRIMARY};
        font-weight: 600;
        background-color: {list_hov_bg};
        border-left: 2px solid {T.ACCENT_PRIMARY};
    }}

    /* ==================== CLEAR BUTTON ==================== */
    QPushButton[class="clear-btn"] {{
        background-color: transparent;
        border: 1px solid {dgr_bdr};
        border-radius: 5px;
        color: {clr_text};
        text-align: center;
        padding: 0px 12px;
        margin: 2px 0px;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        font-size: 8.5pt;
        min-height: 34px;
        max-height: 34px;
    }}
    QPushButton[class="clear-btn"]:hover {{
        color: {clr_h_text};
        background-color: {dgr_hov_bg};
        border: 1px solid {T.DANGER_DEFAULT};
    }}
    QPushButton[class="clear-btn"]:pressed {{
        background-color: rgba(255, 60, 60, 0.2);
        color: {T.DANGER_DEFAULT};
    }}

    /* ==================== CROSSHAIR TOGGLE (legacy) ==================== */
    QPushButton[class="crosshair-btn"] {{
        background-color: transparent;
        border: none;
        border-left: 2px solid transparent;
        color: {T.TEXT_SECONDARY};
        text-align: left;
        padding: 8px 4px 8px 8px;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 400;
        font-size: 9.5pt;
    }}
    QPushButton[class="crosshair-btn"]:hover {{
        color: {T.TEXT_PRIMARY};
        background-color: rgba(255, 184, 0, 0.04);
    }}
    QPushButton[class="crosshair-btn"]:checked {{
        color: {T.ACCENT_WARNING};
        font-weight: 600;
        border-left: 2px solid {T.ACCENT_WARNING};
    }}

    /* ==================== SYSTEM TEXT BUTTON (Carl Zeiss Industrial) ==================== */
    /* Standard action button — uniform height, clear border, left-aligned text.
       Hover matches right VIEW TOOLS (cyan wash + cyan border), not flat EL3 gray. */
    QPushButton[class="system-text-btn"] {{
        background-color: {T.EL2};
        border: 1px solid {T.BORDER_DEFAULT};
        border-radius: 5px;
        color: {T.TEXT_PRIMARY};
        text-align: center;
        padding: 0px 12px;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        font-size: 8.5pt;
        letter-spacing: 0.5px;
        min-height: 34px;
        max-height: 34px;
    }}
    QPushButton[class="system-text-btn"]:hover {{
        background-color: {sys_hov_bg};
        border: 1px solid {T.BORDER_HOVER};
        color: {T.TEXT_PRIMARY};
    }}
    QPushButton[class="system-text-btn"]:pressed {{
        background-color: {T.EL4};
        border: 1px solid {T.BORDER_ACTIVE};
    }}
    QPushButton[class="system-text-btn"]:disabled {{
        background-color: {T.EL1};
        border: 1px solid rgba(41, 56, 72, 0.4);
        color: {T.TEXT_DISABLED};
    }}
    QPushButton[class="system-text-btn"][loaded="true"] {{
        border: 1px solid rgba(47, 178, 122, 0.7);
        color: {T.ACCENT_SUCCESS};
    }}
    /* Loaded must not kill hover — same cyan feedback as VIEW TOOLS */
    QPushButton[class="system-text-btn"][loaded="true"]:hover {{
        background-color: {sys_hov_bg};
        border: 1px solid {T.BORDER_HOVER};
        color: {T.ACCENT_SUCCESS};
    }}
    QPushButton[class="system-text-btn"][role="success"] {{
        border: 1px solid rgba(47, 178, 122, 0.6);
    }}
    QPushButton[class="system-text-btn"][role="success"]:hover {{
        background-color: {sys_hov_bg};
        border: 1px solid {T.BORDER_HOVER};
        color: {T.ACCENT_SUCCESS};
    }}
    QPushButton[class="system-text-btn"][role="success"]:disabled {{
        border: 1px solid rgba(41, 56, 72, 0.4);
        background-color: {T.EL1};
        color: {T.TEXT_DISABLED};
    }}

    /* ==================== SYSTEM TOGGLE BUTTON (Carl Zeiss Industrial) ==================== */
    /* Toggle button — same dimensions as text btn but with clear ON/OFF state */
    QPushButton[class="system-toggle-btn"] {{
        background-color: {T.EL2};
        border: 1px solid {T.BORDER_DEFAULT};
        border-radius: 5px;
        color: {T.TEXT_SECONDARY};
        text-align: center;
        padding: 0px 12px;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        font-size: 8.5pt;
        letter-spacing: 0.5px;
        min-height: 34px;
        max-height: 34px;
    }}
    QPushButton[class="system-toggle-btn"]:hover {{
        background-color: {sys_hov_bg};
        border: 1px solid {T.BORDER_HOVER};
        color: {T.TEXT_PRIMARY};
    }}
    QPushButton[class="system-toggle-btn"]:checked {{
        background-color: {sys_hov_bg};
        border: 1px solid {T.PRIMARY_DEFAULT};
        color: {T.ACCENT_PRIMARY};
        font-weight: 700;
    }}
    QPushButton[class="system-toggle-btn"]:checked:hover {{
        background-color: {sys_hov_bg_strong};
        border: 1px solid {T.PRIMARY_HOVER};
    }}
    QPushButton[class="system-toggle-btn"]:pressed {{
        background-color: {T.EL4};
        border: 1px solid {T.BORDER_ACTIVE};
    }}

    /* ==================== SYSTEM ICON BUTTON (Carl Zeiss Industrial) ==================== */
    /* Square icon button — matches height of system-text-btn */
    QPushButton[class="system-icon-btn"] {{
        background-color: {T.EL2};
        border: 1px solid {T.BORDER_DEFAULT};
        border-radius: 5px;
        color: {T.TEXT_SECONDARY};
        padding: 0px;
        min-width: 34px;
        max-width: 34px;
        min-height: 34px;
        max-height: 34px;
    }}
    QPushButton[class="system-icon-btn"]:hover {{
        background-color: {sys_hov_bg};
        border: 1px solid {T.BORDER_HOVER};
        color: {T.TEXT_PRIMARY};
    }}
    QPushButton[class="system-icon-btn"]:pressed {{
        background-color: {T.EL4};
        border: 1px solid {T.BORDER_ACTIVE};
    }}

    /* Icon button that expands equally in a full-width row (under CROSSHAIR) */
    QPushButton[class="system-icon-btn-fill"] {{
        background-color: {T.EL2};
        border: 1px solid {T.BORDER_DEFAULT};
        border-radius: 5px;
        color: {T.TEXT_SECONDARY};
        padding: 0px;
        min-width: 34px;
        min-height: 34px;
        max-height: 34px;
    }}
    QPushButton[class="system-icon-btn-fill"]:hover {{
        background-color: {sys_hov_bg};
        border: 1px solid {T.BORDER_HOVER};
        color: {T.TEXT_PRIMARY};
    }}
    QPushButton[class="system-icon-btn-fill"]:pressed {{
        background-color: {T.EL4};
        border: 1px solid {T.BORDER_ACTIVE};
    }}

    /* ==================== TITLE BAR & NAV TABS ==================== */
    #CustomTitleBar {{
        background-color: {T.BG_MEDIUM};
        border: none;
    }}

    QPushButton[class="nav-tab"] {{
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        color: {nav_text};
        padding: 0 20px;
        margin: 0;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
        font-size: 8pt;
        text-transform: uppercase;
        letter-spacing: 1px;
        max-height: 28px;
        min-height: 28px;
    }}

    QPushButton[class="nav-tab"]:hover {{
        color: {nav_h_text};
        background-color: {nav_h_bg};
    }}

    QPushButton[class="nav-tab"]:checked {{
        color: {T.ACCENT_PRIMARY};
        background-color: {nav_c_bg};
        border-bottom: 2px solid {T.ACCENT_PRIMARY};
    }}

    /* ==================== CAPSULE BUTTONS ==================== */
    QPushButton[class="capsule-btn"] {{
        background-color: {T.BTN_SECONDARY_BG};
        border: 1px solid {T.BORDER_DEFAULT};
        border-radius: 6px;
        color: {T.TEXT_PRIMARY};
        padding: 6px 14px;
        font-weight: 600;
        font-size: 8.5pt;
        text-align: center;
    }}

    QPushButton[class="capsule-btn"]:hover {{
        background-color: {T.BTN_SECONDARY_HOVER};
        border: 1px solid {T.BORDER_HOVER};
        color: {btn_h_text};
    }}

    QPushButton[class="capsule-btn"]:pressed {{
        background-color: {btn_press};
        border: 1px solid {T.BORDER_ACTIVE};
    }}

    QPushButton[class="capsule-btn"]:disabled {{
        background-color: {cap_dis_bg};
        border: 1px solid {cap_dis_bdr};
        color: {T.TEXT_DISABLED};
    }}

    /* ==================== TOOLTIP ==================== */
    QToolTip {{
        background-color: {tooltip_bg};
        color: {T.TEXT_PRIMARY};
        border: 1px solid {tooltip_bdr};
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 8.5pt;
    }}

    /* ==================== PROGRESS BAR ==================== */
    QProgressBar {{
        background-color: {progress_bg};
        border: 1px solid {T.BORDER_DEFAULT};
        border-radius: 6px;
        text-align: center;
        color: {T.TEXT_PRIMARY};
        font-size: 8pt;
        font-weight: 600;
        min-height: 16px;
    }}

    QProgressBar::chunk {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {T.PRIMARY_PRESSED},
            stop:1 {T.PRIMARY_DEFAULT});
        border-radius: 5px;
    }}
    """


def normalize_theme_name(theme_name):
    mode = (theme_name or "").strip().lower()
    return mode if mode in ("dark", "light") else "dark"


def apply_theme(app, theme_name=None):
    """Apply theme to QApplication"""

    if theme_name is None:
        theme_name = os.environ.get("INNO3D_THEME", SemiconductorTheme.CURRENT_THEME)
    SemiconductorTheme.set_theme(normalize_theme_name(theme_name))

    # Set stylesheet
    app.setStyleSheet(get_main_stylesheet())
    
    # Set palette for fallback
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(SemiconductorTheme.BG_DARK))
    palette.setColor(QPalette.WindowText, QColor(SemiconductorTheme.TEXT_PRIMARY))
    palette.setColor(QPalette.Base, QColor(SemiconductorTheme.BG_DARK))
    palette.setColor(QPalette.AlternateBase, QColor(SemiconductorTheme.BG_LIGHT))
    palette.setColor(QPalette.ToolTipBase, QColor(SemiconductorTheme.BG_MEDIUM))
    palette.setColor(QPalette.ToolTipText, QColor(SemiconductorTheme.TEXT_PRIMARY))
    palette.setColor(QPalette.Text, QColor(SemiconductorTheme.TEXT_PRIMARY))
    palette.setColor(QPalette.Button, QColor(SemiconductorTheme.BG_LIGHT))
    palette.setColor(QPalette.ButtonText, QColor(SemiconductorTheme.TEXT_PRIMARY))
    palette.setColor(QPalette.Link, QColor(SemiconductorTheme.ACCENT_PRIMARY))
    palette.setColor(QPalette.Highlight, QColor(SemiconductorTheme.ACCENT_PRIMARY))
    palette.setColor(QPalette.HighlightedText, QColor(SemiconductorTheme.TEXT_ON_ACCENT))
    
    app.setPalette(palette)
    
    # Font
    font = QFont("Segoe UI", 9)
    app.setFont(font)


def get_class_colors():
    """Get colors for segmentation classes"""
    return {
        128: [0.0, 1.0, 0.62],  # Class 1 - Neon Green
        255: [1.0, 0.30, 0.0]   # Class 2 - Neon Orange
    }
