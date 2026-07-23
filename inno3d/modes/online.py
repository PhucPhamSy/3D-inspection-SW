"""
Online mode support: TCP server, auto-load pipeline, and MainWindow mixin.

Protocol matches Server.cpp / ClientBumpVoid.cpp:
  Port: 8000
  RECV: int32 counter + char Value1[512] + 4 bools = 520 bytes
  SEND: int32 counter + char Value1[512] + 3 int32 values = 528 bytes
"""
# inno3d/modes/online.py  ─── thin façade (Phase 5)
# All logic lives in inno3d/features/online/:
#   server.py     — OnlineServerThread, OnlineLoadThread, _log
#   controller.py — OnlineModeMixin
#
# This file re-exports everything so existing `from inno3d.modes.online import ...`
# call sites keep working without modification.

import time
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import *
from skimage import io

from inno3d.core.styles import SemiconductorTheme
from inno3d.core.ui_system import OnlinePipelineProgress
from inno3d.core.view_support import LoadVolumeThread
from inno3d.core.wafer_context import (
    BIN_NG, BIN_OK, BIN_PENDING,
    HostVolumePathInfo, parse_host_volume_path,
)
from inno3d.tabs.teaching import EnhancementThread

# ── Re-export everything from features/ ──────────────────────────────────
from inno3d.features.online.server import (
    OnlineServerThread,
    OnlineLoadThread,
    FIXED_PORT,
    RECV_PACKET_SIZE,
    SEND_PACKET_SIZE,
    _log,
)
from inno3d.features.online.controller import OnlineModeMixin

__all__ = [
    "OnlineServerThread",
    "OnlineLoadThread",
    "OnlineModeMixin",
    "FIXED_PORT",
    "RECV_PACKET_SIZE",
    "SEND_PACKET_SIZE",
    "_log",
]
