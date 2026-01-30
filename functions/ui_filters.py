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
    Render zoekfilters en retourneer een FilterSpec (dataclass)
    """

    # --------
    # Include terms
    # --------
    include_raw = st.text_area(
        "Zoektermen (include)",
        placeholder="energie, kernenergie",
        help="Meerdere termen scheiden met een komma of nieuwe regel",
    )
    include_terms = _parse_terms(include_raw)

    # --------
    # Logica (alleen bij ≥ 2 termen)
    # --------
    if len(include_terms) >= 2:
        include_logic = st.radio(
            "Zoeklogica",
            ["AND", "OR"],
            horizontal=True,
            help="AND = alle termen moeten voorkomen · OR = minstens één term",
        )
    else:
        include_logic = "AND"

    # --------
    # NOT-termen (altijd direct onder logica)
    # --------
    exclude_raw = st.text_area(
        "NOT-termen (exclude)",
        placeholder="concept",
        help="Meerdere termen scheiden met een komma of nieuwe regel",
    )
    exclude_terms = _parse_terms(exclude_raw)

    st.divider()

    # --------
    # Facet filters
    # --------
    facet_includes: Dict[str, List[str]] = {}
    facet_excludes: Dict[str, List[str]] = {}

    if "Soort" in df.columns:
        facet_includes["Soort"] = st.multiselect(
            "Filter op soort",
            options=sorted(df["Soort"].dropna().unique()),
        )

    if "Vergaderjaar" in df.columns:
        facet_includes["Vergaderjaar"] = st.multiselect(
            "Filter op vergaderjaar",
            options=sorted(df["Vergaderjaar"].dropna().unique()),
        )

    spec = FilterSpec(
        include_terms=include_terms,
        include_logic=include_logic,
        exclude_terms=exclude_terms,
        facet_includes=facet_includes,
        facet_excludes=facet_excludes,
    )

    # (optioneel) query summary tonen:
    st.info(_query_summary(spec))

    return spec

