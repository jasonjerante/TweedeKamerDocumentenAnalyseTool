# functions/ui_filters.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st


@dataclass
class FilterSpec:
    include_terms: List[str]
    include_logic: str  # "AND" or "OR"
    exclude_terms: List[str]
    facet_includes: Dict[str, List[str]]
    facet_excludes: Dict[str, List[str]]


def _parse_terms(raw: str) -> List[str]:
    """
    Parse input uit textarea:
    - split op newline en comma
    - strip
    - unieke terms, volgorde behouden
    """
    if not raw:
        return []
    parts = []
    for line in raw.splitlines():
        for p in line.split(","):
            t = p.strip()
            if t:
                parts.append(t)

    seen = set()
    out = []
    for t in parts:
        if t.lower() not in seen:
            out.append(t)
            seen.add(t.lower())
    return out


def _query_summary(spec: FilterSpec) -> str:
    parts = []

    if spec.include_terms:
        joiner = f" {spec.include_logic} "
        inc = joiner.join([f'"{t}"' for t in spec.include_terms])
        parts.append(f"Include: {inc}")

    if spec.exclude_terms:
        exc = " OR ".join([f'"{t}"' for t in spec.exclude_terms])
        parts.append(f"NOT: {exc}")

    for col, vals in spec.facet_includes.items():
        if vals:
            v = " OR ".join([f'"{x}"' for x in vals])
            parts.append(f"{col}: {v}")

    for col, vals in spec.facet_excludes.items():
        if vals:
            v = " OR ".join([f'"{x}"' for x in vals])
            parts.append(f"NOT {col}: {v}")

    if not parts:
        return "Geen filters ingesteld."
    return "  |  ".join(parts)


def apply_facet_filters(df: pd.DataFrame, spec: FilterSpec) -> pd.DataFrame:
    """
    Pas alleen facet filters toe (categorische filters). Tekst-terms doe je in je search loop.
    """
    out = df

    # Includes
    for col, vals in spec.facet_includes.items():
        if vals and col in out.columns:
            out = out[out[col].isin(vals)]

    # Excludes (NOT)
    for col, vals in spec.facet_excludes.items():
        if vals and col in out.columns:
            out = out[~out[col].isin(vals)]

    return out


def render_filters_ui(df: pd.DataFrame) -> FilterSpec:
    """
    Render een UI voor:
    - Include terms + AND/OR
    - Exclude terms (NOT)
    - Facet include + NOT (nu: Soort + optionele extra’s als ze bestaan)
    """
    st.markdown("### Filters (AND / OR / NOT)")

    col1, col2 = st.columns([2, 1])

    with col1:
        raw_include = st.text_area(
            "Zoektermen (include)",
            placeholder="Voorbeelden:\nenergie\nkernenergie, gas\nasiel",
            help="Je kunt meerdere termen invoeren, gescheiden door komma’s of nieuwe regels.",
            height=110,
        )
        include_terms = _parse_terms(raw_include)

    with col2:
        include_logic = "OR"
        if len(include_terms) >= 2:
            include_logic = st.radio("Logica tussen include-termen", ("AND", "OR"), horizontal=False)
        else:
            st.caption("Logica verschijnt bij ≥ 2 termen.")

        raw_exclude = st.text_area(
            "NOT-termen (exclude)",
            placeholder="Bijv.\nconcept\nbijlage",
            help="Alles wat één van deze woorden bevat wordt uitgesloten.",
            height=110,
        )
        exclude_terms = _parse_terms(raw_exclude)

    st.divider()

    facet_includes: Dict[str, List[str]] = {}
    facet_excludes: Dict[str, List[str]] = {}

    # Facet filters (uitbreidbaar): pak kolommen die in jouw dataset bestaan
    facet_candidates = ["Soort", "Vergaderjaar", "DocumentSoort", "Organisatie", "Kamer"]  # safe proberen
    available_facets = [c for c in facet_candidates if c in df.columns]

    if available_facets:
        st.markdown("### Facet filters (optioneel)")
        for col in available_facets:
            values = (
                df[col]
                .dropna()
                .astype(str)
                .value_counts()
                .index.tolist()
            )

            with st.expander(f"Filter op: {col}", expanded=False):
                cA, cB = st.columns(2)
                with cA:
                    inc = st.multiselect(f"{col} (include)", options=values, default=[])
                with cB:
                    exc = st.multiselect(f"{col} (NOT)", options=values, default=[])

                facet_includes[col] = inc
                facet_excludes[col] = exc
    else:
        st.caption("Geen standaard facet-kolommen gevonden (zoals 'Soort').")

    spec = FilterSpec(
        include_terms=include_terms,
        include_logic=include_logic,
        exclude_terms=exclude_terms,
        facet_includes=facet_includes,
        facet_excludes=facet_excludes,
    )

    st.info(_query_summary(spec))
    return spec
