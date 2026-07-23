"""Text file helpers — portable encoding for config / recipe files.

Korean Windows locales default ``open(path, 'r')`` to **cp949**. Our ship
configs are UTF-8 (box-drawing comments ``─``, µm notes, etc.) and fail with::

    'cp949' codec can't decode byte 0xe2 in position …

Always prefer UTF-8 when reading human-edited config under ``config/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

PathLike = Union[str, Path]

# Prefer UTF-8; then common Windows locales; latin-1 never fails as last resort.
_DECODE_ORDER: tuple[str, ...] = (
    "utf-8-sig",  # handles BOM
    "utf-8",
    "cp949",
    "cp1252",
    "latin-1",
)


def read_text_auto(path: PathLike, encodings: Iterable[str] | None = None) -> str:
    """Read a text file trying UTF-8 first, then locale-friendly fallbacks.

    Parameters
    ----------
    path:
        File path.
    encodings:
        Optional custom codec order. Defaults to UTF-8 → cp949 → cp1252 → latin-1.

    Returns
    -------
    str
        Decoded file contents.

    Raises
    ------
    FileNotFoundError / OSError
        If the path cannot be read as bytes.
    """
    data = Path(path).read_bytes()
    order = tuple(encodings) if encodings is not None else _DECODE_ORDER
    last_err: Exception | None = None
    for enc in order:
        try:
            return data.decode(enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    # Should not reach here if latin-1 is in order
    if last_err is not None:
        return data.decode("utf-8", errors="replace")
    return data.decode("utf-8", errors="replace")
