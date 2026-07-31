"""
Recon PC TCP Sender & Volume Latency Tester GUI.

A standalone PyQt5 utility application to run on Recon PC for selecting volume files,
triggering TCP packets to Inspection PC, and measuring throughput & latency.
"""

import os
import socket
import struct
import sys
import time
from pathlib import Path

from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


RECV_PACKET_SIZE = 520
SEND_PACKET_SIZE = 528


class LatencyTesterThread(QThread):
    """Background worker thread to perform file benchmark and TCP trigger sending"""

    log_signal = pyqtSignal(str, str)  # (message, color)
    result_signal = pyqtSignal(dict)  # Results dictionary

    def __init__(self, ip: str, port: int, volume_path: str, counter: int, flags: list):
        super().__init__()
        self.ip = ip
        self.port = port
        self.volume_path = volume_path
        self.counter = counter
        self.flags = flags

    def run(self):
        res = {
            "success": False,
            "ip": self.ip,
            "port": self.port,
            "path": self.volume_path,
            "counter": self.counter,
            "file_exists": False,
            "file_size_mb": 0.0,
            "read_speed_mbps": 0.0,
            "file_check_ms": 0.0,
            "conn_ms": 0.0,
            "send_ms": 0.0,
            "ack_ms": 0.0,
            "total_rtt_ms": 0.0,
            "ack_status": [],
            "error": "",
        }

        # 1. File Access & Read Benchmark
        self.log_signal.emit(f"Checking volume file access: '{self.volume_path}'...", "#00bcd4")
        p = Path(self.volume_path)

        t0 = time.perf_counter()
        exists = p.exists()
        t_exists = (time.perf_counter() - t0) * 1000
        res["file_check_ms"] = t_exists
        res["file_exists"] = exists

        if exists:
            try:
                size_bytes = p.stat().st_size
                size_mb = size_bytes / (1024 * 1024)
                res["file_size_mb"] = size_mb
                self.log_signal.emit(
                    f"File exists ({size_mb:.2f} MB). Measuring throughput...", "#4caf50"
                )

                t_read_start = time.perf_counter()
                chunk_size = 10 * 1024 * 1024  # 10 MB
                total_read = 0
                with open(p, "rb") as f:
                    while chunk := f.read(chunk_size):
                        total_read += len(chunk)
                        if total_read >= 200 * 1024 * 1024:  # Cap benchmark at 200MB
                            break
                t_read_end = time.perf_counter()
                elapsed = t_read_end - t_read_start
                read_mb = total_read / (1024 * 1024)
                speed = read_mb / elapsed if elapsed > 0 else 0
                res["read_speed_mbps"] = speed
                self.log_signal.emit(
                    f"Throughput: {speed:.2f} MB/s (Read {read_mb:.1f} MB in {elapsed:.3f} s)",
                    "#8bc34a",
                )
            except Exception as e:
                self.log_signal.emit(f"Warning reading file: {e}", "#ff9800")
        else:
            self.log_signal.emit(
                f"Warning: File path does not exist locally or drive is unmapped!", "#ff9800"
            )

        # 2. TCP Trigger Communication
        self.log_signal.emit(f"Connecting to Inspection PC at {self.ip}:{self.port}...", "#2196f3")
        try:
            t_begin = time.perf_counter()

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)

            t_c0 = time.perf_counter()
            sock.connect((self.ip, self.port))
            t_c1 = time.perf_counter()
            res["conn_ms"] = (t_c1 - t_c0) * 1000
            self.log_signal.emit(f"TCP Connected in {res['conn_ms']:.3f} ms", "#4caf50")

            # Build 520-byte packet
            path_encoded = self.volume_path.encode("utf-8", errors="replace")[:512].ljust(
                512, b"\x00"
            )
            bool_flags = struct.pack("<4?", *self.flags)
            packet = struct.pack("<i", self.counter) + path_encoded + bool_flags

            t_s0 = time.perf_counter()
            sock.sendall(packet)
            t_s1 = time.perf_counter()
            res["send_ms"] = (t_s1 - t_s0) * 1000
            self.log_signal.emit(
                f"Sent 520-byte TCP trigger in {res['send_ms']:.3f} ms", "#4caf50"
            )

            # Receive 528-byte ACK
            t_a0 = time.perf_counter()
            ack_data = b""
            while len(ack_data) < SEND_PACKET_SIZE:
                chunk = sock.recv(SEND_PACKET_SIZE - len(ack_data))
                if not chunk:
                    break
                ack_data += chunk
            t_a1 = time.perf_counter()

            res["ack_ms"] = (t_a1 - t_a0) * 1000
            res["total_rtt_ms"] = (t_a1 - t_begin) * 1000

            if len(ack_data) == SEND_PACKET_SIZE:
                res_counter = struct.unpack_from("<i", ack_data, 0)[0]
                res_status = struct.unpack_from("<3i", ack_data, 516)
                res["ack_status"] = list(res_status)
                res["success"] = True

                self.log_signal.emit(
                    f"SUCCESS: Received ACK in {res['ack_ms']:.3f} ms (Total RTT: {res['total_rtt_ms']:.3f} ms)",
                    "#00e676",
                )
            else:
                res["error"] = f"Incomplete ACK packet ({len(ack_data)}/528 bytes)"
                self.log_signal.emit(f"ERROR: {res['error']}", "#ff5252")

            sock.close()

        except Exception as e:
            res["error"] = str(e)
            self.log_signal.emit(f"TCP ERROR: {e}", "#ff5252")

        self.result_signal.emit(res)


class ReconSimulatorMainWindow(QMainWindow):
    """Main UI Window for Recon PC simulator"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Recon PC - Volume TCP Sender & Latency Tester")
        self.resize(1000, 680)
        self._setup_style()
        self._init_ui()

    def _setup_style(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #1e1e24;
                color: #e0e0e0;
            }
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
                color: #e0e0e0;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #33333d;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: #25252d;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: #00bcd4;
            }
            QLineEdit, QSpinBox {
                background-color: #141418;
                border: 1px solid #3d3d4a;
                border-radius: 4px;
                padding: 6px 10px;
                color: #ffffff;
                selection-background-color: #00bcd4;
            }
            QLineEdit:focus, QSpinBox:focus {
                border: 1px solid #00bcd4;
            }
            QPushButton {
                background-color: #00838f;
                color: #ffffff;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #00acc1;
            }
            QPushButton:pressed {
                background-color: #006064;
            }
            QPushButton#btnSend {
                background-color: #00c853;
                font-size: 14px;
                padding: 10px 20px;
            }
            QPushButton#btnSend:hover {
                background-color: #00e676;
            }
            QTableWidget {
                background-color: #141418;
                border: 1px solid #33333d;
                gridline-color: #2a2a35;
            }
            QHeaderView::section {
                background-color: #25252d;
                color: #00bcd4;
                font-weight: bold;
                border: 1px solid #33333d;
                padding: 4px;
            }
            QTextEdit {
                background-color: #121215;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                border: 1px solid #33333d;
                color: #a9b7c6;
            }
            """
        )

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Header Title
        header = QLabel("📡 RECON PC VOLUME TRANSMITTER & LATENCY BENCHMARK")
        header.setFont(QFont("Segoe UI", 16, QFont.Bold))
        header.setStyleSheet("color: #00e5ff; margin-bottom: 5px;")
        layout.addWidget(header)

        # 1. Connection & Target Volume Settings Group
        config_group = QGroupBox("Target Inspection PC & Volume Selection")
        cfg_layout = QVBoxLayout(config_group)

        form_layout = QFormLayout()

        # IP & Port
        ip_port_layout = QHBoxLayout()
        self.txt_ip = QLineEdit("192.168.1.133")
        self.txt_ip.setPlaceholderText("e.g. 192.168.1.133 (Fiber) or 192.168.1.113 (LAN)")

        self.spn_port = QSpinBox()
        self.spn_port.setRange(1, 65535)
        self.spn_port.setValue(8000)
        self.spn_port.setFixedWidth(100)

        ip_port_layout.addWidget(QLabel("IP:"))
        ip_port_layout.addWidget(self.txt_ip, 2)
        ip_port_layout.addWidget(QLabel("Port:"))
        ip_port_layout.addWidget(self.spn_port, 1)

        form_layout.addRow("Inspection PC Address:", ip_port_layout)

        # Volume Path Selector
        path_layout = QHBoxLayout()
        self.txt_path = QLineEdit(r"Z:\volume_compensated.tif")
        btn_browse_file = QPushButton("Browse File...")
        btn_browse_file.clicked.connect(self._browse_file)
        btn_browse_dir = QPushButton("Browse Folder...")
        btn_browse_dir.clicked.connect(self._browse_folder)

        path_layout.addWidget(self.txt_path, 3)
        path_layout.addWidget(btn_browse_file)
        path_layout.addWidget(btn_browse_dir)

        form_layout.addRow("Input Volume Path:", path_layout)

        # Packet Counter & Flags
        pkt_layout = QHBoxLayout()
        self.spn_counter = QSpinBox()
        self.spn_counter.setRange(1, 9999999)
        self.spn_counter.setValue(1)
        self.spn_counter.setFixedWidth(120)

        self.chk_flag0 = QCheckBox("Flag 0 (Enable)")
        self.chk_flag0.setChecked(True)
        self.chk_flag1 = QCheckBox("Flag 1")
        self.chk_flag2 = QCheckBox("Flag 2")
        self.chk_flag3 = QCheckBox("Flag 3")

        pkt_layout.addWidget(QLabel("Packet Counter:"))
        pkt_layout.addWidget(self.spn_counter)
        pkt_layout.addSpacing(20)
        pkt_layout.addWidget(self.chk_flag0)
        pkt_layout.addWidget(self.chk_flag1)
        pkt_layout.addWidget(self.chk_flag2)
        pkt_layout.addWidget(self.chk_flag3)
        pkt_layout.addStretch()

        form_layout.addRow("Packet Payload Options:", pkt_layout)
        cfg_layout.addLayout(form_layout)

        # Send Button
        btn_layout = QHBoxLayout()
        self.btn_send = QPushButton("🚀 SEND TCP TRIGGER & BENCHMARK SPEED")
        self.btn_send.setObjectName("btnSend")
        self.btn_send.clicked.connect(self._start_transmission)
        btn_layout.addWidget(self.btn_send)

        cfg_layout.addLayout(btn_layout)
        layout.addWidget(config_group)

        # 2. Results Splitter (Table Summary + Log Console)
        splitter = QSplitter(Qt.Vertical)

        # Results Summary Table
        self.tbl_results = QTableWidget(0, 6)
        self.tbl_results.setHorizontalHeaderLabels(
            ["Time", "Counter", "File Speed (MB/s)", "Conn (ms)", "ACK (ms)", "Total RTT (ms)"]
        )
        self.tbl_results.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        splitter.addWidget(self.tbl_results)

        # Log Console Text
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        splitter.addWidget(self.log_console)

        splitter.setSizes([200, 250])
        layout.addWidget(splitter)

        # Status Bar Info
        self.lbl_status = QLabel("Ready to transmit volume trigger.")
        self.statusBar().addWidget(self.lbl_status)

    def _browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Volume File", r"Z:\", "TIFF Files (*.tif *.tiff);;All Files (*.*)"
        )
        if file_path:
            self.txt_path.setText(os.path.normpath(file_path))

    def _browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Volume Folder", r"Z:\ ")
        if folder_path:
            self.txt_path.setText(os.path.normpath(folder_path))

    def _log(self, text: str, color: str = "#e0e0e0"):
        ts = time.strftime("%H:%M:%S")
        formatted = f'<span style="color:#757575;">[{ts}]</span> <span style="color:{color};">{text}</span>'
        self.log_console.append(formatted)

    def _start_transmission(self):
        ip = self.txt_ip.text().strip()
        port = self.spn_port.value()
        volume_path = self.txt_path.text().strip()
        counter = self.spn_counter.value()
        flags = [
            self.chk_flag0.isChecked(),
            self.chk_flag1.isChecked(),
            self.chk_flag2.isChecked(),
            self.chk_flag3.isChecked(),
        ]

        if not volume_path:
            self._log("Error: Please select a valid volume file/folder path!", "#ff5252")
            return

        self.btn_send.setEnabled(False)
        self.lbl_status.setText(f"Transmitting to {ip}:{port}...")
        self._log("=" * 60, "#555555")
        self._log(f"Starting test for Counter #{counter} -> Path: '{volume_path}'", "#00e5ff")

        self.worker = LatencyTesterThread(ip, port, volume_path, counter, flags)
        self.worker.log_signal.connect(self._log)
        self.worker.result_signal.connect(self._on_results)
        self.worker.start()

    def _on_results(self, res: dict):
        self.btn_send.setEnabled(True)

        if res["success"]:
            self.lbl_status.setText(
                f"Transmission OK! Total RTT: {res['total_rtt_ms']:.2f} ms | Read Speed: {res['read_speed_mbps']:.1f} MB/s"
            )
            # Auto-increment counter for next trigger
            self.spn_counter.setValue(self.spn_counter.value() + 1)
        else:
            self.lbl_status.setText(f"Transmission Failed: {res.get('error', 'Unknown error')}")

        # Add row to results table
        row = self.tbl_results.rowCount()
        self.tbl_results.insertRow(row)

        ts = time.strftime("%H:%M:%S")
        self.tbl_results.setItem(row, 0, QTableWidgetItem(ts))
        self.tbl_results.setItem(row, 1, QTableWidgetItem(str(res["counter"])))
        self.tbl_results.setItem(
            row, 2, QTableWidgetItem(f"{res['read_speed_mbps']:.1f} MB/s")
        )
        self.tbl_results.setItem(row, 3, QTableWidgetItem(f"{res['conn_ms']:.2f} ms"))
        self.tbl_results.setItem(row, 4, QTableWidgetItem(f"{res['ack_ms']:.2f} ms"))

        rtt_item = QTableWidgetItem(f"{res['total_rtt_ms']:.2f} ms")
        if res["success"]:
            rtt_item.setForeground(Qt.green)
        else:
            rtt_item.setForeground(Qt.red)
        self.tbl_results.setItem(row, 5, rtt_item)

        self.tbl_results.scrollToBottom()


def main():
    app = QApplication(sys.argv)
    window = ReconSimulatorMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
