# functions/ui_topics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TopicInfo:
    topic_id: int
    label: str
    n_docs: int
    delta: int  # last30 - prev30
    top_keywords: List[str]
    top_dossiers: List[str]


def _pick_text_series(df: pd.DataFrame) -> pd.Series:
    """
    Kies een tekstveld om op te clusteren (Onderwerp > Titel > fallback).
    """
    for col in ["Onderwerp", "Titel", "Samenvatting", "TitelKort"]:
        if col in df.columns:
            s = df[col].fillna("").astype(str)
            # Als er echt veel lege strings zijn, proberen we de volgende kolom
            if (s.str.len() > 0).mean() > 0.2:
                return s
    # fallback: concat van alle kolommen (duurder, maar werkt)
    return df.astype(str).agg(" ".join, axis=1)


def _find_date_series(df: pd.DataFrame) -> Optional[pd.Series]:
    for col in ["DatumRegistratie", "Datum", "Publicatiedatum", "PublicationDate"]:
        if col in df.columns:
            return pd.to_datetime(df[col], errors="coerce", utc=True)
    return None


def _find_dossier_series(df: pd.DataFrame) -> Optional[pd.Series]:
    for col in ["Dossiernummer", "DossierNummer", "Dossier", "Zaaknummer", "ZaakNummer"]:
        if col in df.columns:
            return df[col].fillna("unknown-dossier").astype(str)
    return None


def _choose_k(n_docs: int) -> int:
    """
    Heuristiek: genoeg topics om interessant te zijn, niet te veel voor de UI.
    """
    if n_docs < 30:
        return 3
    if n_docs < 80:
        return 5
    if n_docs < 200:
        return 7
    return 10


@st.cache_data(show_spinner=False, ttl=60 * 30)
def compute_topics(df: pd.DataFrame, k: Optional[int] = None, random_state: int = 42) -> Tuple[pd.DataFrame, List[TopicInfo]]:
    """
    Returns:
      df_out: df + kolom 'topic_id'
      topics: lijst met TopicInfo (voor cards)
    """
    if df is None or df.empty:
        return df, []

    df_work = df.copy()
    text = _pick_text_series(df_work)

    # Als alles leeg is, geen topics
    if text.fillna("").str.strip().eq("").all():
        return df, []

    n_docs = len(df_work)
    k = k or _choose_k(n_docs)
    k = max(2, min(k, 12))
    if n_docs < k:
        k = max(2, min(3, n_docs))

    # Vectorize
    vectorizer = TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="dutch",
    )
    X = vectorizer.fit_transform(text)

    # Cluster
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    topic_ids = model.fit_predict(X)
    df_work["topic_id"] = topic_ids

    # Keywords per topic (top TF-IDF centroid features)
    feature_names = np.array(vectorizer.get_feature_names_out())
    centroids = model.cluster_centers_

    # Trend (last 30 vs prev 30)
    dt = _find_date_series(df_work)
    now = pd.Timestamp.utcnow()
    if dt is None:
        dt = pd.Series([pd.NaT] * len(df_work), index=df_work.index)

    last30_mask = (dt >= (now - pd.Timedelta(days=30))) & dt.notna()
    prev30_mask = (dt >= (now - pd.Timedelta(days=60))) & (dt < (now - pd.Timedelta(days=30))) & dt.notna()

    dossier_s = _find_dossier_series(df_work)

    topics: List[TopicInfo] = []
    for tid in range(k):
        idx = np.where(topic_ids == tid)[0]
        n = int(len(idx))

        # top keywords
        top_idx = centroids[tid].argsort()[::-1][:6]
        keywords = feature_names[top_idx].tolist()
        label = ", ".join(keywords[:4])

        # delta
        delta = 0
        if dt.notna().any():
            d_last = int(last30_mask[df_work["topic_id"] == tid].sum())
            d_prev = int(prev30_mask[df_work["topic_id"] == tid].sum())
            delta = d_last - d_prev

        # top dossiers
        top_dossiers: List[str] = []
        if dossier_s is not None:
            top_dossiers = (
                dossier_s[df_work["topic_id"] == tid]
                .value_counts()
                .head(3)
                .index
                .tolist()
            )

        topics.append(
            TopicInfo(
                topic_id=int(tid),
                label=label if label else f"Topic {tid}",
                n_docs=n,
                delta=int(delta),
                top_keywords=keywords,
                top_dossiers=top_dossiers,
            )
        )

    # sort: grootste topics eerst
    topics = sorted(topics, key=lambda t: t.n_docs, reverse=True)

    return df_work, topics


def render_topic_cards(topics: List[TopicInfo], *, key_prefix: str = "topics") -> Optional[int]:
    """
    Render cards. Returns: selected topic_id (of None).
    Gebruikt session_state['topic_filter'] om selectie vast te houden.
    """
    if not topics:
        st.info("Geen topics beschikbaar voor deze selectie.")
        return None

    st.markdown("### Topics (clustering)")
    st.caption("Klik op **Bekijk documenten** om de resultaten te filteren op dat topic.")

    # Reset / clear
    cols_top = st.columns([1, 6])
    with cols_top[0]:
        if st.button("Reset topic-filter", key=f"{key_prefix}_reset"):
            st.session_state["topic_filter"] = None

    selected = st.session_state.get("topic_filter", None)

    # 3 columns grid
    cols = st.columns(3)
    for i, t in enumerate(topics):
        col = cols[i % 3]
        with col:
            with st.container(border=True):
                st.markdown(f"**{t.label}**")
                st.caption(f"{t.n_docs} documenten")

                # Trend line
                if t.delta != 0:
                    arrow = "↑" if t.delta > 0 else "↓"
                    st.write(f"Trend (30d vs 30d): {arrow} **{t.delta:+d}**")
                else:
                    st.write("Trend (30d vs 30d): **0**")

                # Keywords chips
                st.caption("Keywords: " + " · ".join(t.top_keywords[:5]))

                if t.top_dossiers:
                    st.caption("Top dossiers: " + " · ".join(t.top_dossiers))

                btn_label = "Geselecteerd ✅" if selected == t.topic_id else "Bekijk documenten"
                if st.button(btn_label, key=f"{key_prefix}_btn_{t.topic_id}", use_container_width=True):
                    st.session_state["topic_filter"] = t.topic_id

    return st.session_state.get("topic_filter", None)
