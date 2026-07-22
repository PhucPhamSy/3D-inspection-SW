"""
Foundation Pre-Training Widget — Self-Supervised Learning Dashboard.

Provides UI controls for:
- BYOL pre-training configuration (epochs, crop size, etc.)
- Training progress monitoring
- Pre-trained checkpoint management
- One-click weight loading for fine-tuning
"""

import os
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QSpinBox, QGroupBox, QFileDialog, QLineEdit,
    QFrame, QDoubleSpinBox,
)
from PyQt5.QtGui import QFont, QColor

from inno3d.core.styles import SemiconductorTheme


class FoundationWidget(QWidget):
    """
    Self-Supervised Foundation Pre-Training Dashboard.
    """

    run_pretrain = pyqtSignal()       # Start pre-training
    stop_pretrain = pyqtSignal()      # Stop pre-training
    load_weights = pyqtSignal(str)    # Load pretrained weights path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header
        header = QLabel("🧬 FOUNDATION PRE-TRAINING (Self-Supervised)")
        header.setStyleSheet("""
            font-size: 10pt; font-weight: bold;
            color: #4fc3f7;
            letter-spacing: 1.5px;
        """)
        layout.addWidget(header)

        desc = QLabel(
            "Train the encoder on UNLABELED data.\n"
            "No labels needed — AI learns structure patterns from raw volumes."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"""
            color: {SemiconductorTheme.TEXT_SECONDARY};
            font-size: 7.5pt; font-style: italic;
            padding: 2px 0;
        """)
        layout.addWidget(desc)

        # Configuration row 1: Epochs + Batch Size
        row1 = QHBoxLayout()

        lbl_ep = QLabel("Epochs:")
        lbl_ep.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;")
        row1.addWidget(lbl_ep)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(5, 500)
        self.spin_epochs.setValue(50)
        self.spin_epochs.setFixedWidth(60)
        self._style_spin(self.spin_epochs)
        row1.addWidget(self.spin_epochs)

        lbl_bs = QLabel("Batch:")
        lbl_bs.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;")
        row1.addWidget(lbl_bs)
        self.spin_bs = QSpinBox()
        self.spin_bs.setRange(4, 64)
        self.spin_bs.setValue(16)
        self.spin_bs.setFixedWidth(55)
        self._style_spin(self.spin_bs)
        row1.addWidget(self.spin_bs)

        lbl_crop = QLabel("Crop:")
        lbl_crop.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;")
        row1.addWidget(lbl_crop)
        self.spin_crop = QSpinBox()
        self.spin_crop.setRange(32, 256)
        self.spin_crop.setValue(96)
        self.spin_crop.setSingleStep(16)
        self.spin_crop.setFixedWidth(55)
        self._style_spin(self.spin_crop)
        row1.addWidget(self.spin_crop)

        row1.addStretch()
        layout.addLayout(row1)

        # Configuration row 2: LR
        row2 = QHBoxLayout()
        lbl_lr = QLabel("Learning Rate:")
        lbl_lr.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8pt;")
        row2.addWidget(lbl_lr)
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(1e-5, 1e-2)
        self.spin_lr.setValue(3e-4)
        self.spin_lr.setSingleStep(1e-4)
        self.spin_lr.setDecimals(5)
        self.spin_lr.setFixedWidth(90)
        self._style_spin(self.spin_lr)
        row2.addWidget(self.spin_lr)
        row2.addStretch()
        layout.addLayout(row2)

        # Buttons row
        btn_row = QHBoxLayout()

        self.btn_run = QPushButton("🧬 Start Pre-Training")
        self.btn_run.setFixedHeight(32)
        self.btn_run.setStyleSheet(f"""
            QPushButton {{
                background: {SemiconductorTheme.BG_LIGHT};
                color: #4fc3f7;
                border: 1px solid #4fc3f7;
                border-radius: 4px;
                font-size: 9pt;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #4fc3f7;
                color: {SemiconductorTheme.BG_DARK};
            }}
            QPushButton:disabled {{
                background: {SemiconductorTheme.BG_MEDIUM};
                color: {SemiconductorTheme.TEXT_DISABLED};
                border-color: {SemiconductorTheme.BORDER_DEFAULT};
            }}
        """)
        self.btn_run.clicked.connect(self.run_pretrain.emit)
        btn_row.addWidget(self.btn_run)

        self.btn_stop = QPushButton("⬛ Stop")
        self.btn_stop.setFixedHeight(32)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(f"""
            QPushButton {{
                background: {SemiconductorTheme.BG_LIGHT};
                color: #ef5350;
                border: 1px solid #ef5350;
                border-radius: 4px;
                font-size: 9pt;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #ef5350;
                color: white;
            }}
            QPushButton:disabled {{
                background: {SemiconductorTheme.BG_MEDIUM};
                color: {SemiconductorTheme.TEXT_DISABLED};
                border-color: {SemiconductorTheme.BORDER_DEFAULT};
            }}
        """)
        self.btn_stop.clicked.connect(self.stop_pretrain.emit)
        btn_row.addWidget(self.btn_stop)

        layout.addLayout(btn_row)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(16)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {SemiconductorTheme.BG_DARK};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 4px;
                text-align: center;
                color: {SemiconductorTheme.TEXT_SECONDARY};
                font-size: 7.5pt;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0288d1, stop:1 #4fc3f7);
                border-radius: 3px;
            }}
        """)
        layout.addWidget(self.progress_bar)

        # Status + loss display
        self._lbl_status = QLabel("Ready. Load a volume and press Start.")
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_DISABLED}; font-size: 7.5pt;"
        )
        layout.addWidget(self._lbl_status)

        self._lbl_loss = QLabel("")
        self._lbl_loss.setStyleSheet(
            f"color: #4fc3f7; font-size: 9pt; font-weight: bold;"
        )
        layout.addWidget(self._lbl_loss)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {SemiconductorTheme.BORDER_DEFAULT};")
        layout.addWidget(sep)

        # Load pre-trained weights section
        load_lbl = QLabel("📦 Load Pre-Trained Encoder")
        load_lbl.setStyleSheet(
            f"color: #81c784; font-size: 8.5pt; font-weight: bold;"
        )
        layout.addWidget(load_lbl)

        load_row = QHBoxLayout()
        self.txt_pretrained_path = QLineEdit()
        self.txt_pretrained_path.setPlaceholderText("foundation_pretrained.pth...")
        self.txt_pretrained_path.setStyleSheet(f"""
            QLineEdit {{
                background: {SemiconductorTheme.BG_DARK};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 3px;
                padding: 4px;
                font-size: 8pt;
            }}
        """)
        load_row.addWidget(self.txt_pretrained_path)

        btn_browse = QPushButton("📂")
        btn_browse.setFixedSize(30, 28)
        btn_browse.setStyleSheet(f"""
            QPushButton {{
                background: {SemiconductorTheme.BG_LIGHT};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background: {SemiconductorTheme.ACCENT_PRIMARY};
                color: white;
            }}
        """)
        btn_browse.clicked.connect(self._browse_weights)
        load_row.addWidget(btn_browse)
        layout.addLayout(load_row)

        self.btn_load = QPushButton("⬆ Load into Current Model")
        self.btn_load.setFixedHeight(28)
        self.btn_load.setStyleSheet(f"""
            QPushButton {{
                background: {SemiconductorTheme.BG_LIGHT};
                color: #81c784;
                border: 1px solid #81c784;
                border-radius: 4px;
                font-size: 8.5pt;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #81c784;
                color: {SemiconductorTheme.BG_DARK};
            }}
        """)
        self.btn_load.clicked.connect(
            lambda: self.load_weights.emit(self.txt_pretrained_path.text())
        )
        layout.addWidget(self.btn_load)

        self._lbl_load_status = QLabel("")
        self._lbl_load_status.setWordWrap(True)
        self._lbl_load_status.setStyleSheet(
            f"color: {SemiconductorTheme.TEXT_DISABLED}; font-size: 7.5pt;"
        )
        layout.addWidget(self._lbl_load_status)

        # Container style
        self.setStyleSheet(f"""
            FoundationWidget {{
                background: {SemiconductorTheme.BG_MEDIUM};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 6px;
            }}
        """)

    def _style_spin(self, spin):
        spin.setStyleSheet(f"""
            QSpinBox, QDoubleSpinBox {{
                background: {SemiconductorTheme.BG_MEDIUM};
                color: {SemiconductorTheme.TEXT_PRIMARY};
                border: 1px solid {SemiconductorTheme.BORDER_DEFAULT};
                border-radius: 3px;
                padding: 2px;
                font-size: 8pt;
            }}
        """)

    def _browse_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Pre-Trained Weights",
            "", "PyTorch Weights (*.pth);;All Files (*)"
        )
        if path:
            self.txt_pretrained_path.setText(path)

    # ── Public update methods ──

    def set_progress(self, epoch, total, loss, message):
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(epoch)
        self._lbl_status.setText(message)
        self._lbl_loss.setText(f"BYOL Loss: {loss:.4f}")

    def set_running(self, running):
        self.btn_run.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.btn_run.setText(
            "⏳ Pre-Training..." if running else "🧬 Start Pre-Training"
        )
        self.progress_bar.setVisible(running)

    def set_finished(self, msg, saved_path):
        self._lbl_status.setText(msg)
        self._lbl_status.setStyleSheet(
            f"color: #4fc3f7; font-size: 7.5pt;"
        )
        if saved_path:
            self.txt_pretrained_path.setText(saved_path)

    def set_load_status(self, msg, success=True):
        color = "#81c784" if success else "#ef5350"
        self._lbl_load_status.setText(msg)
        self._lbl_load_status.setStyleSheet(
            f"color: {color}; font-size: 7.5pt;"
        )
