"""Help tab for product documents and application settings."""

from PyQt5.QtCore import Qt, QSignalBlocker, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from inno3d.app.settings import (
    VOLUME_3D_BUDGET_MB_OPTIONS,
    get_3d_upload_budget_mb,
    get_large_volume_engine_enabled,
    get_online_fdc_enabled,
    get_online_fdc_interval_sec,
    get_online_fdc_target_host,
    get_online_fdc_target_port,
    set_3d_upload_budget_mb,
    set_large_volume_engine_enabled,
    set_online_fdc_settings,
)
from inno3d.core.styles import SemiconductorTheme


class HelpTab(QWidget):
    theme_changed = pyqtSignal(str)
    fdc_settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._load_volume_settings()
        self._load_fdc_settings()
        self._show_license()
        self.set_current_theme(SemiconductorTheme.CURRENT_THEME)
        self._apply_theme_styles()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        self.title = QLabel("HELP CENTER")
        root.addWidget(self.title)

        self.subtitle = QLabel("Product documents and software settings")
        root.addWidget(self.subtitle)

        self.inner_tabs = QTabWidget()
        self.inner_tabs.setObjectName("HelpInnerTabs")
        root.addWidget(self.inner_tabs, 1)

        self.docs_page = self._build_documents_page()
        self.settings_page = self._build_settings_page()
        self.inner_tabs.addTab(self.docs_page, "Documents")
        self.inner_tabs.addTab(self.settings_page, "Settings")

    def _build_documents_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(10)

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
        layout.addLayout(actions)

        self.content_title = QLabel("")
        layout.addWidget(self.content_title)

        self.content_text = QTextEdit()
        self.content_text.setReadOnly(True)
        layout.addWidget(self.content_text, 1)
        return page

    def _build_settings_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(10)

        # Appearance
        self.appearance_title = QLabel("Appearance")
        layout.addWidget(self.appearance_title)

        theme_row = QHBoxLayout()
        theme_row.setSpacing(8)
        self.theme_label = QLabel("Theme")
        theme_row.addWidget(self.theme_label)

        self.theme_combo = QComboBox()
        self.theme_combo.setMinimumWidth(110)
        self.theme_combo.addItem("Dark", "dark")
        self.theme_combo.addItem("Light", "light")
        self.theme_combo.currentIndexChanged.connect(self._on_theme_combo_changed)
        theme_row.addWidget(self.theme_combo)
        theme_row.addStretch()
        layout.addLayout(theme_row)

        # Large Volume 3D
        self.volume_settings_title = QLabel("Large Volume 3D Settings")
        layout.addWidget(self.volume_settings_title)

        self.volume_settings_help = QLabel(
            "Higher upload budget improves 3D smoothness but uses more memory and GPU resources."
        )
        self.volume_settings_help.setWordWrap(True)
        layout.addWidget(self.volume_settings_help)

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
        layout.addLayout(volume_row)

        self.volume_settings_notice = QLabel("")
        self.volume_settings_notice.setWordWrap(True)
        layout.addWidget(self.volume_settings_notice)

        # FDC
        self.fdc_settings_title = QLabel("FDC Monitor → Recon PC")
        layout.addWidget(self.fdc_settings_title)

        self.fdc_settings_help = QLabel(
            "Inspection PC #1 → Recon PC #1 (e.g. 192.168.1.112) · "
            "Inspection PC #2 → Recon PC #2 (e.g. 192.168.1.115). "
            "Always-on TCP send (independent of Online mode)."
        )
        self.fdc_settings_help.setWordWrap(True)
        layout.addWidget(self.fdc_settings_help)

        fdc_row = QHBoxLayout()
        fdc_row.setSpacing(8)

        self.fdc_enabled_check = QCheckBox("Enable FDC")
        fdc_row.addWidget(self.fdc_enabled_check)

        self.fdc_host_label = QLabel("Recon IP")
        fdc_row.addWidget(self.fdc_host_label)

        self.fdc_host_input = QLineEdit()
        self.fdc_host_input.setPlaceholderText("192.168.1.112")
        self.fdc_host_input.setMinimumWidth(140)
        fdc_row.addWidget(self.fdc_host_input)

        self.fdc_port_label = QLabel("Port")
        fdc_row.addWidget(self.fdc_port_label)

        self.fdc_port_spin = QSpinBox()
        self.fdc_port_spin.setRange(1, 65535)
        self.fdc_port_spin.setValue(8100)
        self.fdc_port_spin.setMinimumWidth(90)
        fdc_row.addWidget(self.fdc_port_spin)

        self.fdc_interval_label = QLabel("Interval (s)")
        fdc_row.addWidget(self.fdc_interval_label)

        self.fdc_interval_spin = QDoubleSpinBox()
        self.fdc_interval_spin.setRange(0.5, 60.0)
        self.fdc_interval_spin.setSingleStep(0.5)
        self.fdc_interval_spin.setDecimals(1)
        self.fdc_interval_spin.setValue(1.0)
        self.fdc_interval_spin.setMinimumWidth(80)
        fdc_row.addWidget(self.fdc_interval_spin)

        self.fdc_save_btn = QPushButton("Save & Apply")
        self.fdc_save_btn.setCursor(Qt.PointingHandCursor)
        self.fdc_save_btn.setProperty("class", "capsule-btn")
        self.fdc_save_btn.setProperty("role", "secondary")
        self.fdc_save_btn.clicked.connect(self._on_fdc_save_clicked)
        fdc_row.addWidget(self.fdc_save_btn)
        fdc_row.addStretch()
        layout.addLayout(fdc_row)

        self.fdc_settings_notice = QLabel("")
        self.fdc_settings_notice.setWordWrap(True)
        layout.addWidget(self.fdc_settings_notice)

        layout.addStretch(1)
        return page

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
        self.inner_tabs.setStyleSheet(
            f"""
            QTabWidget#HelpInnerTabs::pane {{
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 6px;
                top: -1px;
                background: {SemiconductorTheme.BG_MEDIUM};
            }}
            QTabWidget#HelpInnerTabs QTabBar::tab {{
                background: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_SECONDARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 6px 14px;
                margin-right: 2px;
                font-weight: 600;
            }}
            QTabWidget#HelpInnerTabs QTabBar::tab:selected {{
                background: {SemiconductorTheme.BG_MEDIUM};
                color: {SemiconductorTheme.ACCENT_PRIMARY};
            }}
            """
        )
        self.theme_label.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 9pt; font-weight: 700;"
        )
        for title in (
            self.appearance_title,
            self.volume_settings_title,
            self.fdc_settings_title,
        ):
            title.setStyleSheet(
                f"color: {SemiconductorTheme.TEXT_PRIMARY}; font-weight: 700; font-size: 9pt;"
            )
        self.volume_settings_help.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;"
        )
        self.fdc_settings_help.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;"
        )
        self.budget_label.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 9pt;"
        )
        self.volume_settings_notice.setStyleSheet(
            f"color: {SemiconductorTheme.ACCENT_PRIMARY}; font-size: 8pt;"
        )
        for lbl in (
            self.fdc_host_label,
            self.fdc_port_label,
            self.fdc_interval_label,
        ):
            lbl.setStyleSheet(
                f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 9pt;"
            )
        self.fdc_settings_notice.setStyleSheet(
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

    def _load_fdc_settings(self):
        with QSignalBlocker(self.fdc_enabled_check):
            self.fdc_enabled_check.setChecked(get_online_fdc_enabled())
        self.fdc_host_input.setText(get_online_fdc_target_host())
        with QSignalBlocker(self.fdc_port_spin):
            self.fdc_port_spin.setValue(int(get_online_fdc_target_port()))
        with QSignalBlocker(self.fdc_interval_spin):
            self.fdc_interval_spin.setValue(float(get_online_fdc_interval_sec()))

    def _on_fdc_save_clicked(self):
        host = self.fdc_host_input.text().strip() or "127.0.0.1"
        self.fdc_host_input.setText(host)
        set_online_fdc_settings(
            enabled=self.fdc_enabled_check.isChecked(),
            target_host=host,
            target_port=int(self.fdc_port_spin.value()),
            interval_sec=float(self.fdc_interval_spin.value()),
        )
        self.fdc_settings_notice.setText(
            f"Saved. FDC target → {host}:{int(self.fdc_port_spin.value())} "
            f"every {float(self.fdc_interval_spin.value()):.1f}s "
            f"({'ON' if self.fdc_enabled_check.isChecked() else 'OFF'})."
        )
        self.fdc_settings_changed.emit()

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
