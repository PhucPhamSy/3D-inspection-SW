"""C-P1–P4 Homology Cross page for the Analysis workbench."""
from __future__ import annotations

import csv
import os
from typing import Any, Callable, Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from inno3d.core.homology import (
    CompareMode,
    HomologyResult,
    build_homology_table,
    homologous_delta,
    hsi_rank,
    loso_impact,
    nested_variance,
    pairwise_homology_matrix,
    propose_m1_peers,
)
from inno3d.core.styles import SemiconductorTheme


class HomologyPage(QWidget):
    """Interactive Die↔Die (M1) / Wafer↔Wafer (M2) / FOV-bias (M5) + variance/HSI/export."""

    def __init__(self, db: Any, get_run_ids: Callable[[], List[str]], parent=None):
        super().__init__(parent)
        self._db = db
        self._get_run_ids = get_run_ids
        self._run_ids: List[str] = []
        self._last_result: Optional[HomologyResult] = None
        self._last_variance: Optional[Dict[str, Any]] = None
        self._last_hsi: Optional[List[Dict[str, Any]]] = None
        self._last_table: Optional[List[Dict[str, Any]]] = None
        self._last_mes_rows: Optional[List[Dict[str, Any]]] = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        controls = QFrame()
        controls.setStyleSheet(
            f"QFrame {{ background: {SemiconductorTheme.BG_MEDIUM}; border: 1px solid "
            f"{SemiconductorTheme.BORDER_DEFAULT}; border-radius: 5px; }}"
        )
        grid = QGridLayout(controls)
        grid.setContentsMargins(8, 6, 8, 6)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Die↔Die", CompareMode.M1_DIE)
        self.mode_combo.addItem("Wafer↔Wafer", CompareMode.M2_WAFER)
        self.mode_combo.addItem("FOV Bias", CompareMode.M5_FOV_BIAS)
        self.mode_combo.currentIndexChanged.connect(self._update_mode_banner)
        self.anchor_combo = QComboBox()
        self.anchor_combo.currentIndexChanged.connect(
            lambda _index: self._rebuild_peer_list()
        )
        self.metric_combo = QComboBox()
        self.metric_combo.addItem("SOH", "soh")
        self.metric_combo.addItem("Ratio", "ratio")
        self.aggregate_combo = QComboBox()
        self.aggregate_combo.addItem("median Δ", "median")
        self.aggregate_combo.addItem("mean Δ", "mean")
        self.lcl_spin = self._spin(-1.5)
        self.ucl_spin = self._spin(1.5)
        self.grade1_spin = self._spin(10.0, 0.0, 100.0)
        self.grade2_spin = self._spin(30.0, 0.0, 100.0)
        self.propose_button = QPushButton("Propose peers")
        self.run_button = QPushButton("Run H²C²")
        self.run_button.setProperty("role", "primary")
        self.propose_button.clicked.connect(self.propose_peers)
        self.run_button.clicked.connect(self.run_comparison)

        fields = [
            ("Mode", self.mode_combo), ("Anchor", self.anchor_combo),
            ("Metric", self.metric_combo), ("Aggregate", self.aggregate_combo),
            ("LCL", self.lcl_spin), ("UCL", self.ucl_spin),
            ("Grade1 %Tol", self.grade1_spin), ("Grade2 %Tol", self.grade2_spin),
        ]
        for col, (label, widget) in enumerate(fields):
            grid.addWidget(QLabel(label), 0, col)
            grid.addWidget(widget, 1, col)
        grid.addWidget(self.propose_button, 1, len(fields))
        grid.addWidget(self.run_button, 1, len(fields) + 1)
        root.addWidget(controls)

        self.warning_banner = QLabel(
            "⚠ FOV bias study — volumes are NOT class-matched. Do not use for die yield blame."
        )
        self.warning_banner.setStyleSheet(
            f"color: {SemiconductorTheme.ACCENT_WARNING}; background: {SemiconductorTheme.BG_PANEL}; "
            f"border: 1px solid {SemiconductorTheme.ACCENT_WARNING}; padding: 5px; font-size: 9pt;"
        )
        self.warning_banner.setVisible(False)
        root.addWidget(self.warning_banner)

        selection_row = QHBoxLayout()
        selection_row.addWidget(self._section("PEER RUNS (analysis set)", self._build_peer_list()), 1)
        selection_row.addWidget(self._section("RESULT SUMMARY", self._build_summary()), 1)
        root.addLayout(selection_row)

        # C-P3/P4 action row
        action_row = QHBoxLayout()
        self.pair_button = QPushButton("Pair matrix")
        self.loso_button = QPushButton("LOSO")
        self.export_button = QPushButton("Export evidence pack")
        self.pair_button.clicked.connect(self.run_pair_matrix)
        self.loso_button.clicked.connect(self.run_loso)
        self.export_button.clicked.connect(self.export_evidence_pack)
        self.export_button.setEnabled(False)
        action_row.addWidget(self.pair_button)
        action_row.addWidget(self.loso_button)
        action_row.addWidget(self.export_button)
        action_row.addStretch(1)
        root.addLayout(action_row)

        self.status_label = QLabel("Load at least two FOV runs into the analysis set.")
        self.status_label.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 9pt; padding: 2px;")
        root.addWidget(self.status_label)

        self.result_table = QTableWidget(0, 8)
        self.result_table.setHorizontalHeaderLabels(
            ["SHK layer", "FOV", "grid_r", "grid_c", "m_anchor", "m_peer", "Δ", "z"]
        )
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_table.setStyleSheet(self._table_style())
        root.addWidget(self._section("MATCHED HOMOLOGOUS SITES", self.result_table), 1)

        insights = QHBoxLayout()
        self.hotspot_label = self._insight_label("Hotspots: —")
        self.orphan_label = self._insight_label("Orphans / missing rate: —")
        insights.addWidget(self._section("INSIGHT — TOP |Δ|", self.hotspot_label), 2)
        insights.addWidget(self._section("INSIGHT — COVERAGE", self.orphan_label), 1)
        root.addLayout(insights)

        # C-P2: variance stack + HSI ranking (filled after successful Run)
        p2_row = QHBoxLayout()
        self.variance_label = self._insight_label("Run H²C² with multi-chip data to see variance stack.")
        mono = QFont("Consolas", 9)
        if not mono.exactMatch():
            mono = QFont("Courier New", 9)
        self.variance_label.setFont(mono)
        self.hsi_table = QTableWidget(0, 6)
        self.hsi_table.setHorizontalHeaderLabels(
            ["layer", "FOV", "grid_r", "grid_c", "HSI", "IQR"]
        )
        self.hsi_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.hsi_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.hsi_table.setAlternatingRowColors(True)
        self.hsi_table.horizontalHeader().setStretchLastSection(True)
        self.hsi_table.setMaximumHeight(140)
        self.hsi_table.setStyleSheet(self._table_style())
        p2_row.addWidget(self._section("VARIANCE STACK", self.variance_label), 1)
        p2_row.addWidget(self._section("HSI HOTSPOTS (lowest first)", self.hsi_table), 2)
        root.addLayout(p2_row)

        # C-P3: pair / LOSO readout
        self.p3_text = QTextEdit()
        self.p3_text.setReadOnly(True)
        self.p3_text.setMaximumHeight(110)
        self.p3_text.setPlaceholderText("Pair matrix / LOSO results appear here after Run (≥3 units) or via buttons.")
        self.p3_text.setStyleSheet(
            f"QTextEdit {{ background: {SemiconductorTheme.BG_DARK}; color: {SemiconductorTheme.TEXT_PRIMARY}; "
            f"border: 0; font-family: Consolas, 'Courier New', monospace; font-size: 8pt; }}"
        )
        root.addWidget(self._section("PAIR MATRIX / LOSO", self.p3_text))
        self.refresh_runs()

    def _spin(self, value: float, minimum: float = -9999.0, maximum: float = 9999.0) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(2)
        spin.setSingleStep(0.1)
        spin.setValue(value)
        spin.setMaximumWidth(75)
        return spin

    def _build_peer_list(self) -> QListWidget:
        self.peer_list = QListWidget()
        self.peer_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.peer_list.setMaximumHeight(82)
        self.peer_list.setStyleSheet(self._table_style())
        return self.peer_list

    def _build_summary(self) -> QWidget:
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setContentsMargins(4, 0, 4, 0)
        self.summary_labels: Dict[str, QLabel] = {}
        for row, (key, title) in enumerate((
            ("coverage", "Coverage"), ("mean", "Mean Δ"), ("median", "Median Δ"), ("matched", "Matched N"),
        )):
            layout.addWidget(QLabel(title + ":"), row, 0)
            value = QLabel("—")
            value.setStyleSheet(f"color: {SemiconductorTheme.ACCENT_PRIMARY}; font-weight: bold;")
            layout.addWidget(value, row, 1)
            self.summary_labels[key] = value
        return widget

    def _section(self, title: str, content: QWidget) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background: {SemiconductorTheme.BG_MEDIUM}; border: 1px solid "
            f"{SemiconductorTheme.BORDER_DEFAULT}; border-radius: 4px; }}"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(6, 5, 6, 6)
        label = QLabel(title)
        label.setStyleSheet(f"color: {SemiconductorTheme.ACCENT_PRIMARY}; font-size: 8pt; font-weight: bold; border: none;")
        layout.addWidget(label)
        layout.addWidget(content)
        return frame

    def _insight_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {SemiconductorTheme.TEXT_SECONDARY}; font-size: 8.5pt; border: none;")
        return label

    @staticmethod
    def _table_style() -> str:
        return (
            f"QTableWidget, QListWidget {{ background: {SemiconductorTheme.BG_DARK}; "
            f"color: {SemiconductorTheme.TEXT_PRIMARY}; border: 0; font-size: 8pt; "
            f"gridline-color: {SemiconductorTheme.BORDER_DEFAULT}; }}"
            f"QHeaderView::section {{ background: {SemiconductorTheme.BG_PANEL}; "
            f"color: {SemiconductorTheme.TEXT_SECONDARY}; padding: 3px; "
            f"border: 1px solid {SemiconductorTheme.BORDER_DEFAULT}; }}"
        )

    def refresh_runs(self) -> None:
        """Synchronize anchor and peer controls with the current analysis set."""
        previous_anchor = self.anchor_combo.currentData()
        previous_checked = {
            self.peer_list.item(i).data(Qt.UserRole)
            for i in range(self.peer_list.count())
            if self.peer_list.item(i).checkState() == Qt.Checked
        }
        self._run_ids = list(dict.fromkeys(str(x) for x in self._get_run_ids() if x))
        self.anchor_combo.blockSignals(True)
        self.anchor_combo.clear()
        for run_id in self._run_ids:
            self.anchor_combo.addItem(run_id, run_id)
        index = self.anchor_combo.findData(previous_anchor)
        self.anchor_combo.setCurrentIndex(index if index >= 0 else 0)
        self.anchor_combo.blockSignals(False)
        self._rebuild_peer_list(previous_checked)
        self.status_label.setText(
            "Select an anchor and one or more peer runs, then run H²C²."
            if len(self._run_ids) >= 2 else "Load at least two DB-backed FOV runs into the analysis set."
        )

    def _rebuild_peer_list(self, checked: set = None) -> None:
        checked = checked or set()
        anchor = self.anchor_combo.currentData()
        self.peer_list.clear()
        for run_id in self._run_ids:
            if run_id == anchor:
                continue
            item = QListWidgetItem(run_id)
            item.setData(Qt.UserRole, run_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if run_id in checked else Qt.Unchecked)
            self.peer_list.addItem(item)

    def _peer_ids(self) -> List[str]:
        return [
            self.peer_list.item(i).data(Qt.UserRole)
            for i in range(self.peer_list.count())
            if self.peer_list.item(i).checkState() == Qt.Checked
        ]

    def _update_mode_banner(self) -> None:
        self.warning_banner.setVisible(self.mode_combo.currentData() == CompareMode.M5_FOV_BIAS)

    def propose_peers(self) -> None:
        anchor = self.anchor_combo.currentData()
        if not anchor:
            QMessageBox.information(self, "Homology Cross", "Choose an anchor run first.")
            return
        proposed = propose_m1_peers(self._db, anchor)
        proposed_ids = {str(row.get("run_id") or "") for row in proposed}
        available = proposed_ids & set(self._run_ids)
        for i in range(self.peer_list.count()):
            item = self.peer_list.item(i)
            item.setCheckState(Qt.Checked if item.data(Qt.UserRole) in available else Qt.Unchecked)
        outside = sorted(proposed_ids - set(self._run_ids))
        if available:
            self.status_label.setText(f"Selected {len(available)} proposed M1 peer(s) in the analysis set.")
        elif outside:
            self.status_label.setText("M1 peers were found, but none are loaded in the analysis set: " + ", ".join(outside))
        else:
            self.status_label.setText("No eligible M1 peers found for the selected anchor.")

    def run_comparison(self) -> None:
        anchor = self.anchor_combo.currentData()
        peers = self._peer_ids()
        if not anchor or not peers:
            QMessageBox.information(self, "Homology Cross", "Choose one anchor and check at least one peer run.")
            return
        mode = self.mode_combo.currentData()
        metric = self.metric_combo.currentData()
        try:
            rows = self._db.list_mes_joined(run_ids=self._run_ids)
            result = homologous_delta(
                rows, mode=mode, anchor_run_id=anchor,
                peer_run_ids=peers, metric=metric,
                aggregate=self.aggregate_combo.currentData(),
            )
        except Exception as exc:
            self.status_label.setText(f"Homology query failed: {exc}")
            return
        if result.blocked:
            self._clear_results()
            self.status_label.setText("Comparison blocked: " + result.block_reason)
            return
        self._show_result(result)
        self._last_mes_rows = list(rows or [])
        # C-P2 variance / HSI from the full analysis-set homology table
        try:
            table = build_homology_table(rows, mode)
            self._last_table = table
            tol = max(abs(self.ucl_spin.value() - self.lcl_spin.value()), 1e-9)
            var = nested_variance(table, metric=metric)
            hsi = hsi_rank(table, metric=metric, tol=tol)
            self._last_variance = var
            self._last_hsi = hsi
            self._show_variance_hsi(var, hsi)
            self._maybe_auto_p3(table, mode, metric)
        except Exception as exc:
            self.variance_label.setText(f"Variance/HSI failed: {exc}")
            self.hsi_table.setRowCount(0)
            self._last_variance = None
            self._last_hsi = None

    def _maybe_auto_p3(self, table: List[Dict[str, Any]], mode: CompareMode, metric: str) -> None:
        """Auto-fill pair matrix when ≥3 units are present after a successful Run."""
        try:
            labels, matrix = pairwise_homology_matrix(table, mode, metric=metric)
            if len(labels) >= 3:
                self._fill_pair_text(labels, matrix)
            else:
                self.p3_text.setPlainText(
                    f"Pair matrix skipped (need ≥3 units; found {len(labels)}). "
                    "Use Pair matrix / LOSO buttons after loading more strata."
                )
        except Exception as exc:
            self.p3_text.setPlainText(f"Auto pair matrix failed: {exc}")

    def _ensure_table(self) -> Optional[List[Dict[str, Any]]]:
        if self._last_table:
            return self._last_table
        if not self._run_ids:
            return None
        try:
            rows = self._db.list_mes_joined(run_ids=self._run_ids)
            mode = self.mode_combo.currentData()
            table = build_homology_table(rows, mode)
            self._last_mes_rows = list(rows or [])
            self._last_table = table
            return table
        except Exception as exc:
            self.status_label.setText(f"MES load failed: {exc}")
            return None

    def run_pair_matrix(self) -> None:
        table = self._ensure_table()
        if not table:
            QMessageBox.information(self, "Homology Cross", "Load MES runs (or Run H²C²) before Pair matrix.")
            return
        mode = self.mode_combo.currentData()
        metric = self.metric_combo.currentData()
        try:
            labels, matrix = pairwise_homology_matrix(table, mode, metric=metric)
        except Exception as exc:
            QMessageBox.warning(self, "Pair matrix", str(exc))
            return
        if len(labels) < 2:
            self.p3_text.setPlainText("Need ≥2 units for a pairwise matrix.")
            return
        self._fill_pair_text(labels, matrix)
        self.status_label.setText(f"Pair matrix: {len(labels)} unit(s), metric={metric}.")

    def _fill_pair_text(self, labels: List[str], matrix: List[List[float]]) -> None:
        header = "unit\\unit | " + " | ".join(f"{lab:>10}" for lab in labels)
        lines = ["PAIRWISE median_|Δ|", header, "-" * len(header)]
        for i, lab in enumerate(labels):
            cells = " | ".join(f"{matrix[i][j]:10.4f}" for j in range(len(labels)))
            lines.append(f"{lab:>9} | {cells}")
        self.p3_text.setPlainText("\n".join(lines))

    def run_loso(self) -> None:
        table = self._ensure_table()
        if not table:
            QMessageBox.information(self, "Homology Cross", "Load MES runs (or Run H²C²) before LOSO.")
            return
        mode = self.mode_combo.currentData()
        metric = self.metric_combo.currentData()
        leave = "wafer" if mode == CompareMode.M2_WAFER else "chip"
        try:
            ranked = loso_impact(table, mode, leave_level=leave, metric=metric)
        except Exception as exc:
            QMessageBox.warning(self, "LOSO", str(exc))
            return
        if not ranked:
            self.p3_text.setPlainText("LOSO: no strata with usable homologous means.")
            return
        lines = [
            f"LOSO leave_level={leave}  L_full={ranked[0].get('L_full', 0):.6f}",
            f"{'impact':>10}  {'n_obs':>6}  level_key",
            "-" * 48,
        ]
        for row in ranked[:20]:
            lines.append(
                f"{float(row['impact']):10.6f}  {int(row['n_obs']):6d}  {row['level_key']}"
            )
        self.p3_text.setPlainText("\n".join(lines))
        self.status_label.setText(f"LOSO ranked {len(ranked)} {leave} stratum(s).")

    def export_evidence_pack(self) -> None:
        if self._last_result is None or not self._last_result.rows:
            QMessageBox.information(
                self, "Export evidence pack",
                "No successful H²C² run yet. Run Homology first, then export.",
            )
            return
        folder = QFileDialog.getExistingDirectory(self, "Export evidence pack — choose folder")
        if not folder:
            return
        try:
            written = self._write_evidence_csvs(folder)
        except Exception as exc:
            QMessageBox.critical(self, "Export evidence pack", f"Failed to write CSVs:\n{exc}")
            return
        self.status_label.setText(f"Evidence pack written ({len(written)} file(s)): {folder}")
        QMessageBox.information(
            self, "Export evidence pack",
            "Wrote:\n" + "\n".join(written),
        )

    def _write_evidence_csvs(self, folder: str) -> List[str]:
        written: List[str] = []
        result = self._last_result
        assert result is not None

        delta_path = os.path.join(folder, "homology_delta_pairs.csv")
        delta_fields = [
            "layer_id", "fov_index", "grid_row", "grid_col", "peer_run_id",
            "m_anchor", "m_peer", "delta", "z_score", "coverage",
        ]
        with open(delta_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=delta_fields, extrasaction="ignore")
            writer.writeheader()
            for row in result.rows:
                writer.writerow({k: row.get(k, "") for k in delta_fields})
        written.append(delta_path)

        var_path = os.path.join(folder, "homology_variance_stack.csv")
        var = self._last_variance or {}
        with open(var_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["component", "pct", "ok", "n_obs", "n_shk", "method", "reason"])
            for key in ("lot", "foup", "wafer", "chip", "residual"):
                writer.writerow([
                    key, var.get(key, ""), var.get("ok", ""),
                    var.get("n_obs", ""), var.get("n_shk", ""),
                    var.get("method", ""), var.get("reason", ""),
                ])
        written.append(var_path)

        hsi_path = os.path.join(folder, "homology_hsi.csv")
        hsi_fields = ["layer_id", "fov_index", "grid_row", "grid_col", "hsi", "iqr", "n_strata", "mean_m"]
        with open(hsi_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=hsi_fields, extrasaction="ignore")
            writer.writeheader()
            for row in self._last_hsi or []:
                writer.writerow({k: row.get(k, "") for k in hsi_fields})
        written.append(hsi_path)
        return written

    def _clear_results(self) -> None:
        self.result_table.setRowCount(0)
        for label in self.summary_labels.values():
            label.setText("—")
        self.hotspot_label.setText("Hotspots: —")
        self.orphan_label.setText("Orphans / missing rate: —")
        self.variance_label.setText("Run H²C² with multi-chip data to see variance stack.")
        self.hsi_table.setRowCount(0)
        self._last_result = None
        self._last_variance = None
        self._last_hsi = None
        self.export_button.setEnabled(False)
        self.p3_text.clear()

    @staticmethod
    def _bar(pct: float, width: int = 10) -> str:
        filled = max(0, min(width, int(round(pct / 100.0 * width))))
        return "█" * filled + "░" * (width - filled)

    def _show_variance_hsi(self, var: Dict[str, Any], hsi_rows: List[Dict[str, Any]]) -> None:
        if not var.get("ok"):
            reason = var.get("reason") or "insufficient strata"
            self.variance_label.setText(f"Unavailable — {reason}")
        else:
            lines = []
            for key, title in (
                ("lot", "Lot"), ("foup", "FOUP"), ("wafer", "Wafer"),
                ("chip", "Chip"), ("residual", "Residual"),
            ):
                pct = float(var.get(key) or 0.0)
                lines.append(f"{title:<8} {pct:5.1f}%  {self._bar(pct)}")
            lines.append(f"({var.get('n_shk', 0)} SHK · {var.get('n_obs', 0)} obs)")
            self.variance_label.setText("\n".join(lines))

        top = list(hsi_rows[:12])
        self.hsi_table.setRowCount(len(top))
        for row_idx, row in enumerate(top):
            values = (
                row["layer_id"], f"FOV_P{row['fov_index']}", row["grid_row"], row["grid_col"],
                f"{row['hsi']:.3f}", f"{row['iqr']:.4f}",
            )
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col == 4 and float(row["hsi"]) < 0.5:
                    item.setForeground(Qt.red)
                self.hsi_table.setItem(row_idx, col, item)

    def _show_result(self, result: HomologyResult) -> None:
        self._last_result = result
        self.export_button.setEnabled(bool(result.rows))
        self.result_table.setRowCount(len(result.rows))
        for row_idx, row in enumerate(result.rows):
            values = (
                row["layer_id"], f"FOV_P{row['fov_index']}", row["grid_row"], row["grid_col"],
                f"{row['m_anchor']:.4f}", f"{row['m_peer']:.4f}",
                f"{row['delta']:.4f}", f"{row['z_score']:.3f}",
            )
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col in (6, 7) and float(row["delta"]) < self.lcl_spin.value():
                    item.setForeground(Qt.red)
                elif col in (6, 7) and float(row["delta"]) > self.ucl_spin.value():
                    item.setForeground(Qt.yellow)
                self.result_table.setItem(row_idx, col, item)
        denominator = result.matched_n + result.orphans_anchor + result.orphans_peer
        coverage = (result.matched_n / denominator * 100.0) if denominator else 0.0
        self.summary_labels["coverage"].setText(f"{coverage:.1f}%")
        self.summary_labels["mean"].setText(f"{result.mean_delta:.4f}")
        self.summary_labels["median"].setText(f"{result.median_delta:.4f}")
        self.summary_labels["matched"].setText(str(result.matched_n))
        hotspots = sorted(result.rows, key=lambda row: abs(float(row["delta"])), reverse=True)[:5]
        self.hotspot_label.setText(
            " · ".join(
                f"{row['layer_id']}/P{row['fov_index']} r{row['grid_row']}c{row['grid_col']}: {row['delta']:+.4f}"
                for row in hotspots
            ) or "No matched sites."
        )
        self.orphan_label.setText(
            f"Anchor {result.orphans_anchor} · peer {result.orphans_peer} · missing {result.missing_rate:.1%}"
        )
        self.status_label.setText(
            f"Completed {result.mode.value}: {result.matched_n} matched site(s), p95 |Δ| {result.p95_abs_delta:.4f}."
        )
