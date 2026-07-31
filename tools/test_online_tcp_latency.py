"""
Benchmark & Testing script for Online Mode TCP Communication & Network File Read Speed.

Usage:
    python tools/test_online_tcp_latency.py [--ip 192.168.1.113] [--port 8000] [--path "Z:\\volume_compensated.tif"]
"""

import argparse
import os
import socket
import struct
import time
from pathlib import Path


RECV_PACKET_SIZE = 520
SEND_PACKET_SIZE = 528


def test_network_file_access(file_path: str):
    """Measure file existence check and read throughput on the mapped drive Z:\\"""
    print("\n" + "=" * 60)
    print(f" [1/2] MEASURING NETWORK FILE ACCESS: '{file_path}'")
    print("=" * 60)

    p = Path(file_path)
    
    t_start = time.perf_counter()
    exists = p.exists()
    t_exists = time.perf_counter() - t_start

    print(f" -> File Existence Check: {exists} (Latency: {t_exists * 1000:.3f} ms)")

    if not exists:
        print(f" [!] WARNING: Path '{file_path}' does not exist or is not accessible on this PC.")
        return

    try:
        size_bytes = p.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        print(f" -> File Size: {size_bytes:,} bytes ({size_mb:.2f} MB)")

        # Measure reading speed (read up to 200MB)
        t_read_start = time.perf_counter()
        read_chunk_size = 10 * 1024 * 1024  # 10 MB chunks
        total_read = 0
        
        with open(p, "rb") as f:
            while chunk := f.read(read_chunk_size):
                total_read += len(chunk)
                if total_read >= 200 * 1024 * 1024:
                    break

        t_read_end = time.perf_counter()
        elapsed_read = t_read_end - t_read_start
        actual_mb = total_read / (1024 * 1024)
        speed_mbps = actual_mb / elapsed_read if elapsed_read > 0 else 0

        print(f" -> Read Performance: Read {actual_mb:.2f} MB in {elapsed_read:.4f} sec")
        print(f" -> Network Transfer Throughput: {speed_mbps:.2f} MB/s (via Fiber/Network link)")

    except Exception as e:
        print(f" [!] Error reading file: {e}")


def test_tcp_trigger_latency(ip: str, port: int, file_path: str, counter: int = 1):
    """Send TCP packet trigger and measure exact roundtrip latency to Inspection PC"""
    print("\n" + "=" * 60)
    print(f" [2/2] TESTING TCP TRIGGER LATENCY TO INSPECTION PC ({ip}:{port})")
    print("=" * 60)

    try:
        t_begin = time.perf_counter()

        # 1. Connect
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)  # 5 sec timeout
        
        t_conn_start = time.perf_counter()
        sock.connect((ip, port))
        t_conn_end = time.perf_counter()

        conn_ms = (t_conn_end - t_conn_start) * 1000
        print(f" [1] TCP Connection Established: {conn_ms:.3f} ms")

        # 2. Build 520-byte RECV_PACKET
        # int32 counter (4) + char[512] path (512) + 4 bools (4) = 520 bytes
        path_encoded = file_path.encode("utf-8", errors="replace")[:512].ljust(512, b"\x00")
        bool_flags = struct.pack("<4?", True, False, False, False)
        packet = struct.pack("<i", counter) + path_encoded + bool_flags

        assert len(packet) == RECV_PACKET_SIZE, f"Packet size mismatch: {len(packet)} != 520"

        # 3. Send Packet
        t_send_start = time.perf_counter()
        sock.sendall(packet)
        t_send_end = time.perf_counter()

        send_ms = (t_send_end - t_send_start) * 1000
        print(f" [2] Packet of {len(packet)} bytes sent successfully: {send_ms:.3f} ms")

        # 4. Receive 528-byte ACK Response
        t_ack_start = time.perf_counter()
        ack_data = b""
        while len(ack_data) < SEND_PACKET_SIZE:
            chunk = sock.recv(SEND_PACKET_SIZE - len(ack_data))
            if not chunk:
                break
            ack_data += chunk
        t_ack_end = time.perf_counter()

        ack_ms = (t_ack_end - t_ack_start) * 1000
        total_roundtrip_ms = (t_ack_end - t_begin) * 1000

        if len(ack_data) == SEND_PACKET_SIZE:
            res_counter = struct.unpack_from("<i", ack_data, 0)[0]
            res_path_raw = ack_data[4:516].split(b"\x00", 1)[0].decode("utf-8", errors="replace")
            res_status = struct.unpack_from("<3i", ack_data, 516)

            print(f" [3] Received ACK response ({len(ack_data)} bytes): {ack_ms:.3f} ms")
            print(f"     -> Response Counter : {res_counter}")
            print(f"     -> Response Path    : '{res_path_raw}'")
            print(f"     -> Response Status  : {list(res_status)}")
            print("-" * 60)
            print(f" SUCCESS: TOTAL TCP ROUND-TRIP TIME: {total_roundtrip_ms:.3f} ms")
            print("-" * 60)
        else:
            print(f" [!] Incomplete ACK packet received: {len(ack_data)}/{SEND_PACKET_SIZE} bytes")

        sock.close()

    except Exception as e:
        print(f" [!] TCP CONNECTION ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test TCP latency & volume file access time.")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Inspection PC IP Address (Default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="TCP Port (Default: 8000)")
    parser.add_argument("--path", type=str, default=r"Z:\volume_compensated.tif", help="Volume File Path")
    
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" INNO3D ONLINE TCP LATENCY & NETWORK SPEED BENCHMARK")
    print("=" * 60)
    print(f" Target Inspection PC IP : {args.ip}")
    print(f" Target TCP Port         : {args.port}")
    print(f" Target Volume Path      : {args.path}")

    # Step 1: Benchmark network drive read speed
    test_network_file_access(args.path)

    # Step 2: Benchmark TCP signal latency
    test_tcp_trigger_latency(args.ip, args.port, args.path)


if __name__ == "__main__":
    main()
