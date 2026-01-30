# app/utils_filenames.py
from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from typing import Any, Dict, Optional


def slugify(text: str, max_len: int = 60) -> str:
    """
    Maak een veilige bestandsnaam-slug:
    - lower
    - unicode normalisatie
    - spaties -> -
    - alleen [a-z0-9-_]
    """
    if not text:
        return "document"

    # Normalize unicode (strip accents)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()

    # Replace whitespace with hyphen
    text = re.sub(r"\s+", "-", text)

    # Remove invalid chars
    text = re.sub(r"[^a-z0-9\-_]", "", text)

    # Collapse multiple hyphens
    text = re.sub(r"-{2,}", "-", text).strip("-_")

    if not text:
        text = "document"

    return text[:max_len]


def _safe_date_str(value: Any) -> str:
    """
    Probeer date/datetime/str om te zetten naar YYYY-MM-DD.
    Als het niet lukt: 'unknown-date'
    """
    if value is None:
        return "unknown-date"
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, str):
        # Verwacht dat het al 'YYYY-MM-DD' is, anders proberen we grof te parsen
        v = value.strip()
        # Quick accept
        if re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            return v
        # Try: YYYYMMDD
        if re.match(r"^\d{8}$", v):
            return f"{v[0:4]}-{v[4:6]}-{v[6:8]}"
    return "unknown-date"


def guess_extension(doc: Dict[str, Any]) -> str:
    """
    Bepaal extensie. Verwacht doc['file_ext'] of doc['download_url'].
    """
    ext = (doc.get("file_ext") or "").lower().strip(".")
    if ext:
        return ext

    url = (doc.get("download_url") or doc.get("source_url") or "").lower()
    for candidate in ("pdf", "docx", "doc", "html", "txt"):
        if url.endswith(f".{candidate}"):
            return candidate
    # fallback
    return "bin"


def make_unique_filename(doc: Dict[str, Any], title_max_len: int = 60) -> str:
    """
    Pattern:
      {dossier_id}_{date}_{doc_id}_{slug(title)[:60]}.{ext}

    doc moet idealiter hebben:
      dossier_id, doc_id, title, date, file_ext/download_url
    """
    dossier_id = str(doc.get("dossier_id") or doc.get("dossiernummer") or "unknown-dossier").strip()
    doc_id = str(doc.get("doc_id") or doc.get("id") or doc.get("document_id") or "unknown-doc").strip()
    date_str = _safe_date_str(doc.get("date") or doc.get("datum") or doc.get("publication_date"))

    title = str(doc.get("title") or doc.get("naam") or "document").strip()
    title_slug = slugify(title, max_len=title_max_len)

    ext = guess_extension(doc)

    return f"{dossier_id}_{date_str}_{doc_id}_{title_slug}.{ext}"
