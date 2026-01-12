import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression

DOCS_PATH = "data/documents.parquet"
META_PATH = "data/metadata.json"

BASE_URL = "https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0/Document({})/resource"

st.set_page_config(layout="wide")
st.title("De Tweede Kamer Analyse Tool")
st.write("Deze tool maakt openbare Tweede Kamer-stukken overzichtelijk en doorzoekbaar.")


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


def row_contains_terms(row: pd.Series, terms: list[str], logic: str) -> bool:
    text = " ".join(row.astype(str).str.lower())
    if logic == "AND":
        return all(t.lower() in text for t in terms)
    return any(t.lower() in text for t in terms)


def perform_search(df: pd.DataFrame, terms: list[str], logic: str) -> pd.DataFrame:
    if df.empty or not terms:
        return pd.DataFrame()
    mask = df.apply(lambda r: row_contains_terms(r, terms, logic), axis=1)
    return df[mask].copy()


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

search_mode = st.radio("Kies de zoekmodus:", ("Een item", "Meerdere items"), horizontal=True)

terms = []
logic = "OR"

if search_mode == "Een item":
    t = st.text_input("Zoekterm")
    if t.strip():
        terms = [t.strip()]
        logic = "OR"
else:
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        t1 = st.text_input("Zoekterm 1")
    with c2:
        t2 = st.text_input("Zoekterm 2")
    with c3:
        logic = st.radio("Logica", ("AND", "OR"), horizontal=True)

    if t1.strip():
        terms.append(t1.strip())
    if t2.strip():
        terms.append(t2.strip())

if terms:
    results = perform_search(df, terms, logic)

    st.subheader("Search Results")
    st.write(f"Resultaten: **{len(results)}**")
    st.dataframe(results, use_container_width=True)

    if len(results) > 0:
        st.divider()
        st.subheader("Time Trend per Month")

        # Bereid trend data
        tmp = results.copy()
        tmp["DatumRegistratie"] = pd.to_datetime(tmp["DatumRegistratie"], errors="coerce", utc=True)
        tmp = tmp.dropna(subset=["DatumRegistratie"])
        tmp["Month"] = tmp["DatumRegistratie"].dt.to_period("M").dt.to_timestamp()

        trend_data = tmp.groupby("Month").size().reset_index(name="Count").sort_values("Month")

        fig = px.line(trend_data, x="Month", y="Count", title="Aantal documenten per maand")
        st.plotly_chart(fig, use_container_width=True)

        # Regressie (lineaire trend)
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

        st.divider()
        st.subheader("Document download links")

        # Toon max 50 links om pagina snel te houden
        max_links = min(50, len(results))
        for _, row in results.head(max_links).iterrows():
            doc_id = row.get("Id")
            onderwerp = row.get("Onderwerp") or row.get("Titel") or "Document"
            if pd.notna(doc_id):
                link = BASE_URL.format(doc_id)
                st.markdown(f"- [{onderwerp}]({link})")
else:
    st.info("Vul één of meerdere zoektermen in om resultaten te zien.")
