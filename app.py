# app.py
# ---------------------------------------------
# Tweede Kamer Analyse Tool — UI/UX refresh
# - Filters in een vaste linkerkolom (geen sidebar)
# - Duidelijke secties + cards/containers
# - Expliciete AND/OR keuze, NOT apart
# - Eén duidelijke actieknop + reset knoppen
# - Rustige empty-state op main
# ---------------------------------------------

# ---- imports ----
import json
import os
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression

# Functies ophalen van /functions
from functions.ui_filters import render_filters_ui, apply_facet_filters
from functions.ui_topics import compute_topics, render_topic_cards
from functions.ui_master_detail import render_master_detail

DOCS_PATH = "data/documents.parquet"
META_PATH = "data/metadata.json"

BASE_URL = "https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0/Document({})/resource"

# -----------------------------
# Page config + light styling
# -----------------------------
st.set_page_config(layout="wide", page_title="Tweede Kamer Analyse Tool")

st.markdown(
    """
<style>
/* iets meer ademruimte bovenaan */
.block-container { padding-top: 1.2rem; }

/* cards iets strakker */
[data-testid="stVerticalBlockBorderWrapper"]{
  border-radius: 12px;
}

/* buttons net iets robuuster */
.stButton > button {
  border-radius: 10px;
  padding: 0.55rem 0.75rem;
}

/* compacte captions */
.small-muted { color: rgba(49, 51, 63, 0.65); font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Header
# -----------------------------
st.title("De Tweede Kamer Analyse Tool")
st.write(
    "Deze tool maakt openbare Tweede Kamer-stukken overzichtelijk en doorzoekbaar. "
    "Gebouwd door Jason Stuve. Vragen/opmerkingen? Neem contact op via LinkedIn: "
    "https://www.linkedin.com/in/jkpstuve/"
)

# -----------------------------
# Helpers
# -----------------------------
def load_meta() -> dict:
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"updated_utc": None, "rows_total": 0}


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not os.path.exists(DOCS_PATH):
        return pd.DataFrame()

    df = pd.read_parquet(DOCS_PATH)

    # Zorg dat DatumRegistratie netjes is (kan al datetime zijn)
    if "DatumRegistratie" in df.columns:
        df["DatumRegistratie"] = pd.to_datetime(df["DatumRegistratie"], errors="coerce", utc=True)

    return df


def row_contains_terms(
    row: pd.Series,
    include_terms: list[str],
    include_logic: str,
    exclude_terms: list[str],
) -> bool:
    text = " ".join(row.astype(str).str.lower())

    # NOT: als één exclude term voorkomt -> direct false
    if exclude_terms and any(t.lower() in text for t in exclude_terms):
        return False

    # Geen include terms? dan matcht alles (handig als je alleen facets gebruikt)
    if not include_terms:
        return True

    if include_logic == "AND":
        return all(t.lower() in text for t in include_terms)
    return any(t.lower() in text for t in include_terms)


def perform_search_with_progress(
    df: pd.DataFrame,
    include_terms: list[str],
    include_logic: str,
    exclude_terms: list[str],
    chunk_size: int = 5000,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    n = len(df)
    progress = st.progress(0)
    status = st.empty()

    matches = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = df.iloc[start:end]

        mask = chunk.apply(
            lambda r: row_contains_terms(r, include_terms, include_logic, exclude_terms),
            axis=1,
        )
        if mask.any():
            matches.append(chunk[mask])

        pct = int((end / n) * 100)
        progress.progress(pct)
        status.caption(f"Zoeken… {end:,}/{n:,} documenten gecontroleerd ({pct}%).")

    progress.empty()
    status.empty()

    if matches:
        return pd.concat(matches, ignore_index=False).copy()
    return pd.DataFrame()


def results_df_to_docs(results: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Zet jouw results DataFrame om naar list[dict] met keys die de Master–Detail UI verwacht.
    Verwachte keys in UI:
      dossier_id, doc_id, title, date, doctype, download_url, file_ext
    """
    docs: List[Dict[str, Any]] = []

    col_doc_id = "Id" if "Id" in results.columns else None
    col_title = "Onderwerp" if "Onderwerp" in results.columns else ("Titel" if "Titel" in results.columns else None)
    col_date = "DatumRegistratie" if "DatumRegistratie" in results.columns else None
    col_type = "Soort" if "Soort" in results.columns else None

    dossier_candidates = ["Dossiernummer", "DossierNummer", "Dossier", "Zaaknummer", "ZaakNummer"]
    col_dossier = next((c for c in dossier_candidates if c in results.columns), None)

    for _, row in results.iterrows():
        doc_id = str(row.get(col_doc_id)) if col_doc_id else ""
        if not doc_id or doc_id == "nan":
            continue

        title = (row.get(col_title) if col_title else None) or "Document"
        doctype = (row.get(col_type) if col_type else None) or ""
        dossier_id = (row.get(col_dossier) if col_dossier else None) or "unknown-dossier"

        date_val = row.get(col_date) if col_date else None
        if pd.notna(date_val):
            try:
                dt = pd.to_datetime(date_val, errors="coerce", utc=True)
                date_val = None if pd.isna(dt) else dt.date().isoformat()
            except Exception:
                date_val = None
        else:
            date_val = None

        download_url = BASE_URL.format(doc_id)

        docs.append(
            {
                "dossier_id": str(dossier_id),
                "doc_id": doc_id,
                "title": str(title),
                "date": date_val,
                "doctype": str(doctype),
                "download_url": download_url,
                "file_ext": "pdf",
            }
        )

    return docs


def has_any_filters(spec) -> bool:
    return bool(spec.include_terms) or any(spec.facet_includes.values()) or any(spec.facet_excludes.values())


# -----------------------------
# Load dataset
# -----------------------------
meta = load_meta()
df = load_data()

if df.empty:
    st.warning("Nog geen dataset gevonden. Wacht tot de GitHub Action een eerste update heeft gedaan.")
    st.info("Tip: je kunt de workflow ook handmatig starten via GitHub → Actions → ‘Daily data update’ → Run workflow.")
    st.stop()

st.caption(f"Laatst bijgewerkt (UTC): {meta.get('updated_utc')} | Aantal rijen: {meta.get('rows_total')}")

with st.expander("Dataset preview", expanded=False):
    st.dataframe(df.head(200), use_container_width=True)

# -----------------------------
# Layout: Filterkolom + Main
# -----------------------------
filter_col, main_col = st.columns([1.1, 3.9], gap="large")

# We bewaren resultaten in session_state zodat de UI niet "leeg" voelt na interacties.
if "last_results" not in st.session_state:
    st.session_state["last_results"] = None
if "last_results_with_topics" not in st.session_state:
    st.session_state["last_results_with_topics"] = None
if "last_topics" not in st.session_state:
    st.session_state["last_topics"] = None
if "last_runtime_s" not in st.session_state:
    st.session_state["last_runtime_s"] = None

# -----------------------------
# Filters UI (left column)
# -----------------------------
with filter_col:
    st.markdown("## 🔍 Zoeken & Filters")

    # Card: Zoekinput + facet filters (we hergebruiken jouw bestaande render_filters_ui)
    with st.container(border=True):
        st.markdown("### Zoeken")
        st.markdown('<div class="small-muted">AND = specifieker · OR = breder · NOT = uitsluiten</div>', unsafe_allow_html=True)

        # render_filters_ui verwacht df en levert spec met include_terms/include_logic/exclude_terms en facets
        # Let op: als render_filters_ui zelf labels toont als "AND/OR/NOT", is dat oké.
        spec = render_filters_ui(df)

    with st.container(border=True):
        st.markdown("### Uitvoering")
        chunk_size = st.slider(
            "Zoeksnelheid (chunk size)",
            min_value=2000,
            max_value=20000,
            value=5000,
            step=1000,
            help="Groter = sneller, maar kan zwaarder zijn voor je machine/browser.",
        )

        run_search = st.button("🔎 Zoeken", use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Reset topic", use_container_width=True):
                st.session_state["topic_filter"] = None
                # laat vorige resultaten staan; alleen topic resetten
                st.rerun()
        with c2:
            if st.button("Reset alles", use_container_width=True):
                st.session_state.clear()
                st.rerun()

    st.caption("💡 Volgende stap: filters deelbaar maken via URL query parameters.")

# -----------------------------
# Main column: empty state / results
# -----------------------------
with main_col:
    # Header card (maakt main minder "leeg")
    with st.container(border=True):
        st.markdown("### Overzicht")
        left_a, left_b, left_c = st.columns(3)
        left_a.metric("Dataset (rijen)", f"{len(df):,}")
        left_b.metric("Laatst bijgewerkt (UTC)", meta.get("updated_utc") or "-")
        if st.session_state["last_results"] is None:
            left_c.metric("Resultaten", "-")
        else:
            left_c.metric("Resultaten", f"{len(st.session_state['last_results']):,}")

    # Empty state: als nog nooit gezocht
    if not run_search and st.session_state["last_results"] is None:
        st.info("👈 Stel links je filters in en klik op **Zoeken**.")
        st.stop()

    # Als user klikt op zoeken: valideer
    if run_search:
        if not has_any_filters(spec):
            st.warning("Je hebt nog geen zoekterm of filters ingevuld.")
            st.stop()

        # -----------------------------
        # Execute search
        # -----------------------------
        t0 = time.time()
        with st.spinner("Bezig met zoeken… even geduld"):
            # 1) facet filters
            df_prefiltered = apply_facet_filters(df, spec)

            st.caption(
                f"Zoeken in {len(df_prefiltered):,} documenten "
                f"(na facet filters; totaal was {len(df):,})."
            )

            # 2) text search
            results = perform_search_with_progress(
                df_prefiltered,
                include_terms=spec.include_terms,
                include_logic=spec.include_logic,
                exclude_terms=spec.exclude_terms,
                chunk_size=int(chunk_size),
            )

        dt = time.time() - t0

        st.session_state["last_results"] = results
        st.session_state["last_runtime_s"] = dt

        # Precompute topics meteen (zodat filtering snel voelt)
        if len(results) > 0:
            results_with_topics, topics = compute_topics(results)
            st.session_state["last_results_with_topics"] = results_with_topics
            st.session_state["last_topics"] = topics
        else:
            st.session_state["last_results_with_topics"] = None
            st.session_state["last_topics"] = None

    # Gebruik altijd de laatste resultaten (zodat UI niet springt)
    results = st.session_state["last_results"]
    runtime_s = st.session_state["last_runtime_s"]

    if results is None:
        st.info("👈 Stel links je filters in en klik op **Zoeken**.")
        st.stop()

    # Status
    if len(results) == 0:
        st.warning("Geen resultaten gevonden.")
        st.markdown(
            """
**Suggesties:**
- Gebruik **OR** in plaats van **AND**
- Verwijder één of meer **NOT-termen**
- Probeer een **algemener** zoekwoord
"""
        )
        st.stop()

    st.success(f"Klaar! {len(results):,} resultaten gevonden" + (f" in {runtime_s:.1f}s." if runtime_s else "."))

    # -----------------------------
    # Topics (clustering)
    # -----------------------------
    st.divider()
    st.subheader("Topics (clustering)")
    st.caption("Klik op een topic om de resultaten te filteren.")

    results_with_topics = st.session_state["last_results_with_topics"]
    topics = st.session_state["last_topics"]

    if results_with_topics is None or topics is None:
        # fallback
        results_with_topics, topics = compute_topics(results)

    selected_topic_id = render_topic_cards(topics)

    if selected_topic_id is not None:
        filtered = results_with_topics[results_with_topics["topic_id"] == selected_topic_id].copy()
        st.info(f"Topic-filter actief: **{len(filtered):,}** documenten.")
        results_to_show = filtered
    else:
        results_to_show = results_with_topics

    # -----------------------------
    # Results: Master–Detail
    # -----------------------------
    st.divider()
    docs = results_df_to_docs(results_to_show)
    render_master_detail(docs, title="Resultaten")

    # -----------------------------
    # Debug dataframe
    # -----------------------------
    with st.expander("Ruwe resultaten (DataFrame)", expanded=False):
        st.dataframe(results_to_show, use_container_width=True)

    # -----------------------------
    # Analytics
    # -----------------------------
    st.divider()
    st.subheader("Analyse")

    tmp = results_to_show.copy()
    if "DatumRegistratie" in tmp.columns:
        tmp["DatumRegistratie"] = pd.to_datetime(tmp["DatumRegistratie"], errors="coerce", utc=True)
        tmp = tmp.dropna(subset=["DatumRegistratie"])
    else:
        tmp = pd.DataFrame()

    if tmp.empty:
        st.info("Geen (bruikbare) DatumRegistratie-waarden gevonden voor trendanalyse.")
    else:
        st.markdown("#### Time Trend per Month")
        tmp["Month"] = tmp["DatumRegistratie"].dt.to_period("M").dt.to_timestamp()
        trend_data = tmp.groupby("Month").size().reset_index(name="Count").sort_values("Month")

        fig = px.line(trend_data, x="Month", y="Count", title="Aantal documenten per maand")
        st.plotly_chart(fig, use_container_width=True)

        # Lineaire regressie trendline
        if len(trend_data) >= 2:
            X = np.arange(len(trend_data)).reshape(-1, 1)
            y = trend_data["Count"].values
            model = LinearRegression().fit(X, y)
            trend_line = model.predict(X)

            trend_data["Trend"] = trend_line
            fig2 = px.line(trend_data, x="Month", y=["Count", "Trend"], title="Trend (Count vs. lineaire regressie)")
            st.plotly_chart(fig2, use_container_width=True)

            slope = float(model.coef_[0])
            direction = "stijgend" if slope > 0 else "dalend"
            st.info(f"Trend is **{direction}** (helling ≈ **{slope:.2f}** documenten per maand-index).")

    st.divider()
    st.subheader("Verdeling documenttypes")

    if "Soort" in results_to_show.columns:
        breakdown = results_to_show["Soort"].fillna("Onbekend").value_counts().reset_index()
        breakdown.columns = ["Document Type", "Count"]
        st.dataframe(breakdown, use_container_width=True)

        fig3 = px.pie(breakdown, values="Count", names="Document Type", title="Verdeling documenttypes")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Kolom 'Soort' is niet aanwezig in de dataset.")

    # -----------------------------
    # About / Methodology
    # -----------------------------
    with st.expander("Over deze tool", expanded=False):
        st.markdown(
            """
**Data:** Tweede Kamer OData API (download resource per document)  
**Update:** Dagelijks via GitHub Actions  
**Zoeken:** Full-row text match + AND/OR/NOT + facets  
**Topics:** TF-IDF + KMeans clustering (keywords per topic)  
**Auteur:** Jason Stuve  
"""
        )

# klaargemaakt voor gebruik door Jason Stuve op maandag 12 januari 2026
