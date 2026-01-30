# functions/ui_master_detail.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from functions.utils_downloads import fetch_bytes
from functions.utils_filenames import make_unique_filename


def _format_doc_option(doc: Dict[str, Any]) -> str:
    date_str = str(doc.get("date") or "")
    doctype = str(doc.get("doctype") or "")
    title = str(doc.get("title") or "Document")
    dossier = str(doc.get("dossier_id") or "unknown-dossier")

    bits = [b for b in [date_str, doctype] if b]
    meta = " · ".join(bits) if bits else ""
    if meta:
        return f"[{dossier}] {title}  —  {meta}"
    return f"[{dossier}] {title}"


def render_master_detail(
    docs: List[Dict[str, Any]],
    *,
    title: str = "Resultaten",
    default_sort: str = "Nieuwste eerst",
) -> None:
    """
    Master–detail UI:
      - Links: lijst (selecteer 1 document)
      - Rechts: detailpaneel + download

    Vereist per doc (minimaal):
      doc_id, title, dossier_id, date, doctype, download_url
    """
    if not docs:
        st.info("Geen resultaten om te tonen.")
        return

    # -------- Controls (top) --------
    st.subheader(title)

    if "selected_doc_id" not in st.session_state:
        st.session_state["selected_doc_id"] = None

    topbar = st.columns([2, 1, 1])
    with topbar[0]:
        q = st.text_input(
            "Filter resultaten (titel/doctype/dossier)",
            placeholder="Typ om te filteren…",
            key="md_filter_q",
        )
    with topbar[1]:
        sort_mode = st.selectbox(
            "Sorteren",
            [default_sort, "Oudste eerst", "Titel A–Z", "Titel Z–A"],
            key="md_sort_mode",
        )
    with topbar[2]:
        if st.button("Reset selectie", use_container_width=True):
            st.session_state["selected_doc_id"] = None

    # -------- Filter + sort docs --------
    filtered = docs
    if q and q.strip():
        ql = q.strip().lower()
        def match(d: Dict[str, Any]) -> bool:
            hay = " ".join(
                [
                    str(d.get("title") or ""),
                    str(d.get("doctype") or ""),
                    str(d.get("dossier_id") or ""),
                    str(d.get("doc_id") or ""),
                ]
            ).lower()
            return ql in hay
        filtered = [d for d in docs if match(d)]

    def safe_date(d: Dict[str, Any]) -> str:
        # YYYY-MM-DD string; lexicographic works
        return str(d.get("date") or "")

    if sort_mode == "Nieuwste eerst":
        filtered = sorted(filtered, key=safe_date, reverse=True)
    elif sort_mode == "Oudste eerst":
        filtered = sorted(filtered, key=safe_date)
    elif sort_mode == "Titel A–Z":
        filtered = sorted(filtered, key=lambda d: str(d.get("title") or "").lower())
    elif sort_mode == "Titel Z–A":
        filtered = sorted(filtered, key=lambda d: str(d.get("title") or "").lower(), reverse=True)

    st.caption(f"Toont **{len(filtered):,}** van **{len(docs):,}** documenten.")

    # -------- Layout: master (left) / detail (right) --------
    left, right = st.columns([0.48, 0.52], gap="large")

    with left:
        st.markdown("### Selecteer een document")

        if not filtered:
            st.warning("Geen resultaten na filter. Verwijder de filtertekst.")
            return

        # options = doc_id list; display uses formatted label
        options = [str(d.get("doc_id")) for d in filtered]
        labels = {str(d.get("doc_id")): _format_doc_option(d) for d in filtered}

        # Default selection: keep previous if still exists else first
        current = st.session_state.get("selected_doc_id")
        if current not in labels:
            current = options[0]
            st.session_state["selected_doc_id"] = current

        selected_doc_id = st.radio(
            "Resultaten",
            options=options,
            format_func=lambda x: labels.get(x, x),
            key="md_selected_radio",
        )
        st.session_state["selected_doc_id"] = selected_doc_id

    with right:
        st.markdown("### Detail")

        selected_id = st.session_state.get("selected_doc_id")
        doc = next((d for d in docs if str(d.get("doc_id")) == str(selected_id)), None)

        if not doc:
            st.info("Selecteer links een document om details te zien.")
            return

        title_txt = str(doc.get("title") or "Document")
        dossier = str(doc.get("dossier_id") or "unknown-dossier")
        date_str = str(doc.get("date") or "")
        doctype = str(doc.get("doctype") or "")
        download_url = str(doc.get("download_url") or "")

        st.markdown(f"## {title_txt}")
        meta_bits = [b for b in [f"Dossier: {dossier}", date_str, doctype] if b and b.strip()]
        if meta_bits:
            st.caption(" · ".join(meta_bits))

        # Unieke bestandsnaam zichtbaar
        unique_name = make_unique_filename(doc)
        st.code(unique_name, language="text")

        # Bronlink
        if download_url:
            st.markdown(f"**Bron:** {download_url}")

        st.divider()

        # Download knop
        if download_url:
            try:
                data = fetch_bytes(download_url)
                st.download_button(
                    "Download document",
                    data=data,
                    file_name=unique_name,
                    use_container_width=True,
                    key=f"md_dl_{selected_id}",
                )
            except Exception:
                st.warning("Download niet beschikbaar (URL of netwerk).")
        else:
            st.info("Geen download URL beschikbaar voor dit document.")
