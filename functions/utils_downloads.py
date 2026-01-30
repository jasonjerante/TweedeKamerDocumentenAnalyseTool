# app/utils_downloads.py
from __future__ import annotations

import io
import zipfile
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import streamlit as st

DEFAULT_TIMEOUT = 30


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_bytes(url: str, headers: Optional[Dict[str, str]] = None) -> bytes:
    """
    Download bytes van URL (gecached 1 uur).
    """
    if not url:
        raise ValueError("Lege download URL")

    resp = requests.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.content


def build_zip(files: List[Tuple[str, bytes]]) -> bytes:
    """
    files: lijst (filename, bytes)
    returns: zip bytes
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, data in files:
            zf.writestr(filename, data)
    buf.seek(0)
    return buf.read()
