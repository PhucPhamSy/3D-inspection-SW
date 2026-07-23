# inno3d/features/online/server.py
# -----------------------------------------------------------------------
# Online server threads — extracted from inno3d/modes/online.py (Phase 5)
#
# OnlineServerThread: TCP server, packet decode, client handling
# OnlineLoadThread: background volume load for online FOV
# -----------------------------------------------------------------------

import os
import socket
import struct
import threading
import time
import traceback
from pathlib import Path as _Path

import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal

# ==================== PACKET STRUCTURES ====================
RECV_PACKET_SIZE = 520
SEND_PACKET_SIZE = 528
FIXED_PORT = 8000


def _log(msg):
    """Print timestamped log message to console"""
    ts = time.strftime("%H:%M:%S")
    print(f"[ONLINE {ts}] {msg}")


class OnlineServerThread(QThread):
    """TCP server that listens for file paths from Server.cpp"""
    
    folder_received = pyqtSignal(str)        # Emits file/folder path
    status_update = pyqtSignal(str)          # Status messages for status bar
    server_error = pyqtSignal(str)           # Error messages
    
    def __init__(self, port=FIXED_PORT):
        super().__init__()
        self.port = port
        self._running = False
        self._server_socket = None
    
    def run(self):
        self._running = True
        _log(f"=== SERVER STARTING on port {self.port} ===")
        try:
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # SO_REUSEADDR allows immediate rebind after close (avoids TIME_WAIT)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_socket.settimeout(1.0)  # 1 second timeout for accept()
            
            _log(f"Binding to 0.0.0.0:{self.port}...")
            self._server_socket.bind(('0.0.0.0', self.port))
            
            # Verify the actual bound port
            actual_addr = self._server_socket.getsockname()
            actual_port = actual_addr[1]
            _log(f"Bound successfully - actual address: {actual_addr[0]}:{actual_port}")
            
            if actual_port != self.port:
                _log(f"WARNING: Port mismatch! Requested {self.port}, got {actual_port}")
                self.server_error.emit(f"ONLINE ERROR: Port mismatch - got {actual_port} instead of {self.port}")
                return
            
            self._server_socket.listen(5)
            _log(f"Listening on port {actual_port} - waiting for Server.cpp connections...")
            self.status_update.emit(f"ONLINE: Listening on port {actual_port} - Waiting for connection...")
            
            connection_count = 0
            while self._running:
                try:
                    client_socket, addr = self._server_socket.accept()
                    connection_count += 1
                    _log(f"--- Connection #{connection_count} from {addr[0]} on port {actual_port} (client src port: {addr[1]}) ---")
                    self.status_update.emit(f"ONLINE: Client connected to port {actual_port} from {addr[0]} - Connection #{connection_count}")
                    self._handle_client(client_socket, connection_count)
                except socket.timeout:
                    continue  # Check _running flag and retry
                except OSError as e:
                    if self._running:
                        _log(f"OSError in accept loop: {e}")
                        raise
                    break
            
            _log("=== SERVER LOOP ENDED (stopped) ===")
                    
        except Exception as e:
            _log(f"=== SERVER ERROR: {e} ===")
            if self._running:
                self.server_error.emit(f"ONLINE ERROR: {str(e)}")
        finally:
            self._cleanup()
    
    def _handle_client(self, client_socket, conn_id):
        """Handle a single client connection â€” matches ClientTGVVoid.cpp protocol"""
        try:
            # Receive the packet (520 bytes = int32 + char[512] + 4 bools)
            _log(f"  [{conn_id}] Waiting for {RECV_PACKET_SIZE} bytes...")
            data = b''
            while len(data) < RECV_PACKET_SIZE:
                chunk = client_socket.recv(RECV_PACKET_SIZE - len(data))
                if not chunk:
                    _log(f"  [{conn_id}] Client disconnected (recv returned empty)")
                    break
                data += chunk
                _log(f"  [{conn_id}] Received {len(chunk)} bytes (total: {len(data)}/{RECV_PACKET_SIZE})")
            
            if len(data) == RECV_PACKET_SIZE:
                # Parse: int32(4) + char[512](512) + 4 bools(4)
                counter = struct.unpack_from('<i', data, 0)[0]
                raw_path = data[4:4+512]  # char[512] = 512 bytes
                
                # Decode as ASCII/UTF-8 and strip null terminators
                folder_path = raw_path.decode('utf-8', errors='replace').split('\x00', 1)[0].strip()
                
                # Parse bool flags
                bool_bytes = data[4+512:4+512+4]
                bools = [bool(b) for b in bool_bytes]
                
                _log(f"  [{conn_id}] Parsed packet:")
                _log(f"    Counter: {counter}")
                _log(f"    Path:    '{folder_path}'")
                _log(f"    Bools:   {bools}")
                
                self.status_update.emit(f"ONLINE: Received #{counter}: {folder_path}")
                
                # Build response: int32(4) + char[512](512) + 3Ã—int32(12) = 528 bytes
                # Matches _EXAMPLE_SEND_PACKET in ClientBumpVoid.cpp
                result_path_encoded = folder_path.encode('utf-8', errors='replace')
                result_path_encoded = result_path_encoded[:512].ljust(512, b'\x00')
                response = struct.pack('<i', counter) + result_path_encoded + struct.pack('<3i', 0, 0, 1)
                
                _log(f"  [{conn_id}] Sending response ({len(response)} bytes)...")
                client_socket.send(response)
                _log(f"  [{conn_id}] Response sent OK")
                
                # Emit signal to process the file
                _log(f"  [{conn_id}] Emitting folder_received signal: '{folder_path}'")
                self.folder_received.emit(folder_path)
            else:
                _log(f"  [{conn_id}] ERROR: Incomplete packet ({len(data)}/{RECV_PACKET_SIZE} bytes)")
                self.status_update.emit(f"ONLINE: Incomplete packet ({len(data)}/{RECV_PACKET_SIZE} bytes)")
                
        except Exception as e:
            _log(f"  [{conn_id}] ERROR handling client: {e}")
            self.status_update.emit(f"ONLINE: Client error: {str(e)}")
        finally:
            client_socket.close()
            _log(f"  [{conn_id}] Connection closed")
    
    def stop(self):
        """Stop the server cleanly"""
        _log("=== STOP requested ===")
        self._running = False
        # Force-close the socket to unblock accept()
        self._cleanup()
    
    def _cleanup(self):
        """Close the server socket"""
        if self._server_socket:
            _log("Closing server socket...")
            try:
                self._server_socket.close()
            except Exception as e:
                _log(f"Error closing socket: {e}")
            self._server_socket = None
            _log("Server socket closed")


class OnlineLoadThread(QThread):
    """Background thread to load volume + class1 + class2 from a folder progressively"""
    
    status_update = pyqtSignal(str)          # Progress messages
    volume_loaded = pyqtSignal(object)       # volume data
    class1_loaded = pyqtSignal(object)       # class1 data
    class2_loaded = pyqtSignal(object)       # class2 data
    error_occurred = pyqtSignal(str)         # error message
    
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
    
    @staticmethod
    def _ensure_3d(data):
        """Convert multi-channel image stack to single-channel (grayscale)"""
        if data.ndim == 4:
            # (Z, Y, X, C) -> take first channel
            data = data[:, :, :, 0]
        elif data.ndim == 2:
            data = data[np.newaxis, :, :]
        return data
    
    def run(self):
        try:
            folder = Path(self.folder_path)
            if not folder.exists():
                self.error_occurred.emit(f"Folder not found: {self.folder_path}")
                return
            
            # Get all tiff/tif files
            all_files = sorted([
                f for f in folder.iterdir() 
                if f.suffix.lower() in ('.tif', '.tiff')
            ])
            
            if not all_files:
                self.error_occurred.emit(f"No TIFF files in: {self.folder_path}")
                return
            
            # Classify files by pattern
            volume_files = []
            class1_files = []
            class2_files = []
            
            for f in all_files:
                fname = f.name
                if '_bump3D' in fname:
                    class1_files.append(f)
                elif '_voidsOnly' in fname and '_voidsOnlyFlatten' not in fname:
                    class2_files.append(f)
                elif re.match(r'.*slice\d+\.tiff - $', fname, re.IGNORECASE):
                    volume_files.append(f)
            
            # Sort by slice number
            def extract_slice_number(filepath):
                match = re.search(r'slice(\d+)', filepath.name, re.IGNORECASE)
                return int(match.group(1)) if match else 0
            
            volume_files.sort(key=extract_slice_number)
            class1_files.sort(key=extract_slice_number)
            class2_files.sort(key=extract_slice_number)
            
            # 1. Load and emit volume first
            if volume_files:
                self.status_update.emit(f"Loading volume ({len(volume_files)} slices)...")
                if len(volume_files) == 1:
                    volume_data = io.imread(str(volume_files[0]))
                else:
                    volume_data = np.array([io.imread(str(f)) for f in volume_files])
                volume_data = self._ensure_3d(volume_data)
                self.status_update.emit(f"Volume loaded: {volume_data.shape} ({volume_data.dtype})")
                self.volume_loaded.emit(volume_data)
            
            # 2. Load and emit class 1
            if class1_files:
                self.status_update.emit(f"Loading Class 1 ({len(class1_files)} files)...")
                if len(class1_files) == 1:
                    class1_data = io.imread(str(class1_files[0]))
                else:
                    class1_data = np.array([io.imread(str(f)) for f in class1_files])
                class1_data = self._ensure_3d(class1_data)
                self.status_update.emit(f"Class 1 loaded: {class1_data.shape}")
                self.class1_loaded.emit(class1_data)
            
            # 3. Load and emit class 2
            if class2_files:
                self.status_update.emit(f"Loading Class 2 ({len(class2_files)} files)...")
                if len(class2_files) == 1:
                    class2_data = io.imread(str(class2_files[0]))
                else:
                    class2_data = np.array([io.imread(str(f)) for f in class2_files])
                class2_data = self._ensure_3d(class2_data)
                self.status_update.emit(f"Class 2 loaded: {class2_data.shape}")
                self.class2_loaded.emit(class2_data)
            
            self.status_update.emit("ONLINE: All data loaded successfully")
            
        except Exception as e:
            import traceback
            self.error_occurred.emit(f"{str(e)}\n{traceback.format_exc()}")


