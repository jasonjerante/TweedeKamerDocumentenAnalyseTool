# app.py
# ---- imports ----
import json
import os
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

st.set_page_config(layout="wide")
st.title("De Tweede Kamer Analyse Tool")
st.write(
    "Deze tool maakt openbare Tweede Kamer-stukken overzichtelijk en doorzoekbaar. "
    "Gebouwd door Jason Stuve. Mochten er vragen of opmerkingen zijn, neem gerust contact op via Linkedin: "
    "https://www.linkedin.com/in/jkpstuve/"
)


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
    Zet jouw results DataFrame om naar list[dict] met keys die de download UI verwacht.
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


# ---- Load dataset ----
meta = load_meta()
df = load_data()

if df.empty:
    st.warning("Nog geen dataset gevonden. Wacht tot de GitHub Action een eerste update heeft gedaan.")
    st.info("Tip: je kunt de workflow ook handmatig starten via GitHub → Actions → ‘Daily data update’ → Run workflow.")
    st.stop()

st.caption(f"Laatst bijgewerkt (UTC): {meta.get('updated_utc')} | Aantal rijen: {meta.get('rows_total')}")

with st.expander("Dataset preview", expanded=False):
    st.dataframe(df.head(200), use_container_width=True)

st.divider()
st.subheader("Zoeken")

# Gebruik een form zodat de app niet bij elke toetsaanslag opnieuw gaat zoeken.
with st.form("search_form", clear_on_submit=False):
    spec = render_filters_ui(df)

    st.caption("Tip: gebruik AND om specifieker te zoeken. NOT helpt om ruis weg te halen.")
    chunk_size = st.slider("Zoeksnelheid (chunk size)", min_value=2000, max_value=20000, value=5000, step=1000)

    submitted = st.form_submit_button("Zoeken")

if not submitted:
    st.info("Vul één of meerdere zoektermen in en klik op **Zoeken**.")
    st.stop()

if not spec.include_terms and not any(spec.facet_includes.values()) and not any(spec.facet_excludes.values()):
    st.warning("Je hebt nog geen zoekterm of filters ingevuld.")
    st.stop()

# ---- Execute search with feedback ----
with st.spinner("Bezig met zoeken… even geduld"):
    # 1) Eerst facet filters toepassen (sneller zoeken)
    df_prefiltered = apply_facet_filters(df, spec)

    st.caption(
        f"Zoeken in {len(df_prefiltered):,} documenten "
        f"(na facet filters; totaal was {len(df):,})."
    )

    # 2) Daarna tekst search met AND/OR/NOT
    results = perform_search_with_progress(
        df_prefiltered,
        include_terms=spec.include_terms,
        include_logic=spec.include_logic,
        exclude_terms=spec.exclude_terms,
        chunk_size=int(chunk_size),
    )

st.success(f"Klaar! {len(results):,} resultaten gevonden.")

if len(results) == 0:
    st.info("Geen resultaten. Probeer een andere term, of gebruik OR i.p.v. AND.")
    st.stop()

# ---- Topics (clustering) ----
st.divider()

# Bereken topics op basis van de huidige zoekresultaten
results_with_topics, topics = compute_topics(results)

# Render cards en lees selectie
selected_topic_id = render_topic_cards(topics)

# Filter resultaten als er een topic gekozen is
if selected_topic_id is not None:
    results = results_with_topics[results_with_topics["topic_id"] == selected_topic_id].copy()
    st.success(f"Topic-filter actief: {len(results):,} documenten in dit topic.")
else:
    # geen filter -> gebruik originele results
    results = results_with_topics


# ---- UX Upgrade: Master–Detail resultaten ----
st.divider()
docs = results_df_to_docs(results)

render_master_detail(
    docs,
    title="Search Results (Master–Detail)",
)

# ---- Optional: dataset tabel nog steeds beschikbaar voor debug ----
with st.expander("Ruwe resultaten (DataFrame)", expanded=False):
    st.dataframe(results, use_container_width=True)

with st.expander("Downloadlijst per dossier (oude view)", expanded=False):
    render_results_with_downloads(docs)


# ---- Analytics ----
st.divider()
st.subheader("Time Trend per Month")

tmp = results.copy()
if "DatumRegistratie" not in tmp.columns:
    st.warning("Kolom 'DatumRegistratie' is niet aanwezig in de dataset, trendgrafiek kan niet worden gemaakt.")
    st.stop()

tmp["DatumRegistratie"] = pd.to_datetime(tmp["DatumRegistratie"], errors="coerce", utc=True)
tmp = tmp.dropna(subset=["DatumRegistratie"])

if tmp.empty:
    st.warning("Geen geldige DatumRegistratie-waarden gevonden voor deze selectie.")
    st.stop()

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

if "Soort" in results.columns:
    breakdown = results["Soort"].fillna("Onbekend").value_counts().reset_index()
    breakdown.columns = ["Document Type", "Count"]
    st.dataframe(breakdown, use_container_width=True)

    fig3 = px.pie(breakdown, values="Count", names="Document Type", title="Verdeling documenttypes")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Kolom 'Soort' is niet aanwezig in de dataset.")

# klaargemaakt voor gebruik door Jason Stuve op maandag 12 januari 2026
