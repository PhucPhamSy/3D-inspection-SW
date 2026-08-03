"""Help tab for product information, license, and update guidance."""

from PyQt5.QtCore import Qt, QSignalBlocker, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from inno3d.app.settings import (
    VOLUME_3D_BUDGET_MB_OPTIONS,
    get_3d_upload_budget_mb,
    get_large_volume_engine_enabled,
    set_3d_upload_budget_mb,
    set_large_volume_engine_enabled,
)
from inno3d.core.styles import SemiconductorTheme


class HelpTab(QWidget):
    theme_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._load_volume_settings()
        self._show_license()
        self.set_current_theme(SemiconductorTheme.CURRENT_THEME)
        self._apply_theme_styles()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        self.title = QLabel("HELP CENTER")
        root.addWidget(self.title)

        self.subtitle = QLabel("Product documents and software information")
        root.addWidget(self.subtitle)

        actions = QHBoxLayout()
        actions.setSpacing(8)

        self.btn_view_license = self._make_section_button("View License", self._show_license)
        self.btn_check_updates = self._make_section_button(
            "Check for updates..", self._show_updates
        )
        self.btn_about = self._make_section_button("About", self._show_about)

        actions.addWidget(self.btn_view_license)
        actions.addWidget(self.btn_check_updates)
        actions.addWidget(self.btn_about)
        actions.addStretch()

        self.theme_label = QLabel("Theme")
        self.theme_label.setStyleSheet("font-weight: 600;")
        actions.addWidget(self.theme_label)

        self.theme_combo = QComboBox()
        self.theme_combo.setMinimumWidth(110)
        self.theme_combo.addItem("Dark", "dark")
        self.theme_combo.addItem("Light", "light")
        self.theme_combo.currentIndexChanged.connect(self._on_theme_combo_changed)
        actions.addWidget(self.theme_combo)

        root.addLayout(actions)

        self.volume_settings_title = QLabel("Large Volume 3D Settings")
        root.addWidget(self.volume_settings_title)

        self.volume_settings_help = QLabel(
            "Higher upload budget improves 3D smoothness but uses more memory and GPU resources."
        )
        self.volume_settings_help.setWordWrap(True)
        root.addWidget(self.volume_settings_help)

        volume_row = QHBoxLayout()
        volume_row.setSpacing(8)

        self.budget_label = QLabel("3D Upload Budget (MB)")
        volume_row.addWidget(self.budget_label)

        self.budget_combo = QComboBox()
        self.budget_combo.setMinimumWidth(90)
        for mb in VOLUME_3D_BUDGET_MB_OPTIONS:
            self.budget_combo.addItem(str(mb), mb)
        self.budget_combo.currentIndexChanged.connect(self._on_budget_combo_changed)
        volume_row.addWidget(self.budget_combo)

        self.lve_checkbox = QCheckBox("Enable Large Volume Engine")
        self.lve_checkbox.stateChanged.connect(self._on_lve_checkbox_changed)
        volume_row.addWidget(self.lve_checkbox)
        volume_row.addStretch()
        root.addLayout(volume_row)

        self.volume_settings_notice = QLabel("")
        self.volume_settings_notice.setWordWrap(True)
        root.addWidget(self.volume_settings_notice)

        self.content_title = QLabel("")
        root.addWidget(self.content_title)

        self.content_text = QTextEdit()
        self.content_text.setReadOnly(True)
        root.addWidget(self.content_text, 1)

    def _make_section_button(self, text, callback):
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setProperty("class", "capsule-btn")
        btn.setProperty("role", "secondary")
        btn.clicked.connect(callback)
        return btn

    def _apply_theme_styles(self):
        self.title.setStyleSheet(
            f"color: {SemiconductorTheme.ACCENT_PRIMARY}; font-weight: bold; font-size: 12pt;"
        )
        self.subtitle.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 9pt;")
        self.theme_label.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 9pt; font-weight: 700;"
        )
        self.volume_settings_title.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_PRIMARY}; font-weight: 700; font-size: 9pt;"
        )
        self.volume_settings_help.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;"
        )
        self.budget_label.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 9pt;"
        )
        self.volume_settings_notice.setStyleSheet(
            f"color: {SemiconductorTheme.ACCENT_PRIMARY}; font-size: 8pt;"
        )
        self.content_title.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_PRIMARY}; font-weight: 700; font-size: 10pt;"
        )
        self.content_text.setStyleSheet(
            f"""
            QTextEdit {{
                background: {SemiconductorTheme.BG_DARK};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 6px;
                padding: 10px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 9pt;
            }}
            """
        )

    def set_current_theme(self, theme_name):
        mode = (theme_name or "dark").strip().lower()
        idx = self.theme_combo.findData(mode)
        if idx < 0:
            idx = self.theme_combo.findData("dark")
        blocker = QSignalBlocker(self.theme_combo)
        self.theme_combo.setCurrentIndex(idx)
        del blocker
        self._apply_theme_styles()

    def _on_theme_combo_changed(self, _index):
        theme = self.theme_combo.currentData() or "dark"
        self.theme_changed.emit(theme)

    def _load_volume_settings(self):
        mb = get_3d_upload_budget_mb()
        idx = self.budget_combo.findData(mb)
        if idx < 0:
            idx = self.budget_combo.findData(VOLUME_3D_BUDGET_MB_OPTIONS[0])
        with QSignalBlocker(self.budget_combo):
            self.budget_combo.setCurrentIndex(max(0, idx))
        with QSignalBlocker(self.lve_checkbox):
            self.lve_checkbox.setChecked(get_large_volume_engine_enabled())

    def _show_volume_settings_notice(self):
        self.volume_settings_notice.setText(
            "Saved. New values apply on the next volume load or 3D re-render."
        )

    def _on_budget_combo_changed(self, _index):
        mb = self.budget_combo.currentData()
        if mb is None:
            return
        set_3d_upload_budget_mb(int(mb))
        self._show_volume_settings_notice()

    def _on_lve_checkbox_changed(self, _state):
        set_large_volume_engine_enabled(self.lve_checkbox.isChecked())
        self._show_volume_settings_notice()

    def _set_active_title(self, title):
        self.content_title.setText(title)

    def _show_license(self):
        self._set_active_title("View License")
        self.content_text.setPlainText(
            "INNO3D SOFTWARE LICENSE AGREEMENT\n"
            "Version 0.0.1\n"
            "Effective Date: April 23, 2026\n\n"
            "1. Grant of License\n"
            "Innometry grants the customer a limited, non-exclusive, non-transferable "
            "license to install and use Inno3D for internal inspection and analysis workflows.\n\n"
            "2. Permitted Use\n"
            "The software may be used only for authorized production, quality, and engineering "
            "activities within the customer organization.\n\n"
            "3. Restrictions\n"
            "You may not redistribute, sublicense, reverse engineer, decompile, disassemble, "
            "or create derivative works of the software unless explicitly permitted by written agreement.\n\n"
            "4. Intellectual Property\n"
            "All rights, title, and interest in and to Inno3D, including software, interface, "
            "and documentation, remain the property of Innometry and its licensors.\n\n"
            "5. Warranty and Liability\n"
            "The software is provided as-is for professional use. Innometry is not liable for "
            "indirect, incidental, or consequential damages arising from use of the software.\n\n"
            "6. Support\n"
            "Support, maintenance scope, and update eligibility are governed by your service or "
            "commercial agreement.\n\n"
            "7. Termination\n"
            "This license terminates automatically if license terms are violated. Upon termination, "
            "all use of the software must stop immediately.\n\n"
            "For commercial licensing details, contact: sales@innometry.com"
        )

    def _show_updates(self):
        self._set_active_title("Check for updates..")
        self.content_text.setPlainText(
            "Current installed version: 0.0.1\n"
            "Update channel: Production\n\n"
            "Automatic update service is not configured in this build.\n"
            "Please contact your administrator or Innometry support team to receive "
            "the latest approved installer package.\n\n"
            "Contact: sales@innometry.com"
        )

    def _show_about(self):
        self._set_active_title("About")
        self.content_text.setPlainText(
            "Software name: Inno3D\n"
            "Software version: 0.0.1\n"
            "Software license: --\n"
            "Advisor: Kim Donglok\n"
            "Developer: Pham Sy Phuc\n"
            "Contact: sales@innometry.com"
        )
