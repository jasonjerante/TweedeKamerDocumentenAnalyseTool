# app/ui_results_downloads.py
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import streamlit as st

from app.utils_downloads import build_zip, fetch_bytes
from app.utils_filenames import make_unique_filename


def group_by_dossier(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups = defaultdict(list)
    for doc in results:
        dossier_id = str(doc.get("dossier_id") or doc.get("dossiernummer") or "unknown-dossier")
        groups[dossier_id].append(doc)
    return dict(groups)


def render_results_with_downloads(results: List[Dict[str, Any]]) -> None:
    """
    - Groepeert zoekresultaten per dossier
    - Toont per document: unieke naam + metadata + download knop
    - Optioneel: selecteer meerdere docs -> zip per dossier
    """
    if not results:
        st.info("Geen resultaten. Pas je zoekopdracht/filters aan.")
        return

    groups = group_by_dossier(results)

    st.caption(f"Gevonden documenten: **{len(results)}** · Dossiers: **{len(groups)}**")

    # Zorg dat selectie persistent is
    if "selected_doc_ids" not in st.session_state:
        st.session_state.selected_doc_ids = set()

    for dossier_id, docs in groups.items():
        # Sort op datum (nieuwste boven)
        docs_sorted = sorted(
            docs,
            key=lambda d: str(d.get("date") or d.get("datum") or d.get("publication_date") or ""),
            reverse=True,
        )

        dossier_title = docs_sorted[0].get("dossier_title") or docs_sorted[0].get("dossier_naam") or ""
        header = f"{dossier_id}"
        if dossier_title:
            header += f" — {dossier_title}"

        with st.expander(f"{header}  ·  {len(docs_sorted)} documenten", expanded=False):
            # Dossier-level download geselecteerd (ZIP)
            colA, colB, colC = st.columns([1, 1, 2])
            with colA:
                st.write("")
                st.write("**Selectie**")
            with colB:
                select_all = st.button("Selecteer alles", key=f"sel_all_{dossier_id}")
                if select_all:
                    for d in docs_sorted:
                        st.session_state.selected_doc_ids.add(str(d.get("doc_id") or d.get("id") or d.get("document_id")))
            with colC:
                clear_all = st.button("Selectie wissen", key=f"clr_all_{dossier_id}")
                if clear_all:
                    for d in docs_sorted:
                        doc_id = str(d.get("doc_id") or d.get("id") or d.get("document_id"))
                        if doc_id in st.session_state.selected_doc_ids:
                            st.session_state.selected_doc_ids.remove(doc_id)

            st.divider()

            # Documentenlijst
            for d in docs_sorted:
                doc_id = str(d.get("doc_id") or d.get("id") or d.get("document_id") or "")
                title = d.get("title") or d.get("naam") or "Onbekende titel"
                doctype = d.get("doctype") or d.get("type") or ""
                date_str = str(d.get("date") or d.get("datum") or d.get("publication_date") or "")
                download_url = d.get("download_url") or d.get("url") or d.get("source_url") or ""

                unique_name = make_unique_filename(d)

                row1, row2, row3 = st.columns([0.10, 0.62, 0.28])

                with row1:
                    checked = st.checkbox(
                        "",
                        value=(doc_id in st.session_state.selected_doc_ids),
                        key=f"chk_{dossier_id}_{doc_id}",
                    )
                    if checked:
                        st.session_state.selected_doc_ids.add(doc_id)
                    else:
                        st.session_state.selected_doc_ids.discard(doc_id)

                with row2:
                    meta_bits = [b for b in [date_str, doctype] if b]
                    meta = " · ".join(meta_bits)
                    st.markdown(f"**{title}**")
                    if meta:
                        st.caption(meta)
                    st.code(unique_name, language="text")

                with row3:
                    # Direct download knop
                    if download_url:
                        try:
                            data = fetch_bytes(download_url)
                            st.download_button(
                                label="Download",
                                data=data,
                                file_name=unique_name,
                                mime=None,  # laat browser bepalen
                                key=f"dl_{dossier_id}_{doc_id}",
                                use_container_width=True,
                            )
                        except Exception as e:
                            st.warning("Download niet beschikbaar")
                    else:
                        st.caption("Geen download URL")

                st.divider()

            # ZIP download van geselecteerde docs (in dit dossier)
            selected_docs = []
            for d in docs_sorted:
                doc_id = str(d.get("doc_id") or d.get("id") or d.get("document_id") or "")
                if doc_id in st.session_state.selected_doc_ids:
                    selected_docs.append(d)

            if selected_docs:
                st.subheader("Download selectie als ZIP")
                zip_files = []
                errors = 0
                for d in selected_docs:
                    url = d.get("download_url") or d.get("url") or d.get("source_url") or ""
                    if not url:
                        errors += 1
                        continue
                    try:
                        data = fetch_bytes(url)
                        zip_files.append((make_unique_filename(d), data))
                    except Exception:
                        errors += 1

                if zip_files:
                    zip_name = f"{dossier_id}_selected_documents.zip"
                    zip_bytes = build_zip(zip_files)
                    st.download_button(
                        label=f"Download ZIP ({len(zip_files)} bestanden)",
                        data=zip_bytes,
                        file_name=zip_name,
                        key=f"zip_{dossier_id}",
                        use_container_width=True,
                    )
                if errors:
                    st.caption(f"Let op: {errors} geselecteerde items konden niet worden gedownload.")
