# functions/usage_analytics.py
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DB_PATH = os.getenv("USAGE_DB_PATH", "data/usage.db")


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_usage_db() -> None:
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS search_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            session_id TEXT NOT NULL,
            include_terms TEXT,
            include_logic TEXT,
            exclude_terms TEXT,
            facet_includes_json TEXT,
            facet_excludes_json TEXT,
            results_count INTEGER,
            runtime_s REAL
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_search_events_ts ON search_events(ts_utc);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_search_events_session ON search_events(session_id);")
    conn.commit()
    conn.close()


def log_search_event(
    *,
    session_id: str,
    include_terms: List[str],
    include_logic: str,
    exclude_terms: List[str],
    facet_includes: Dict[str, List[str]],
    facet_excludes: Dict[str, List[str]],
    results_count: int,
    runtime_s: float,
) -> None:
    conn = _connect()
    ts_utc = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """
        INSERT INTO search_events (
            ts_utc, session_id, include_terms, include_logic, exclude_terms,
            facet_includes_json, facet_excludes_json, results_count, runtime_s
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            ts_utc,
            session_id,
            ",".join(include_terms) if include_terms else "",
            include_logic or "",
            ",".join(exclude_terms) if exclude_terms else "",
            json.dumps(facet_includes or {}, ensure_ascii=False),
            json.dumps(facet_excludes or {}, ensure_ascii=False),
            int(results_count),
            float(runtime_s),
        ),
    )
    conn.commit()
    conn.close()


def get_trending_terms(days: int = 7, limit: int = 15) -> List[Dict[str, Any]]:
    """
    Return: [{term: str, count: int}, ...]
    """
    conn = _connect()
    rows = conn.execute(
        """
        SELECT include_terms
        FROM search_events
        WHERE ts_utc >= datetime('now', ?)
        """,
        (f"-{days} days",),
    ).fetchall()
    conn.close()

    freq: Dict[str, int] = {}
    for (raw,) in rows:
        if not raw:
            continue
        for t in [x.strip() for x in raw.split(",") if x.strip()]:
            key = t.lower()
            freq[key] = freq.get(key, 0) + 1

    items = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return [{"term": k, "count": v} for k, v in items]


def get_trending_facets(days: int = 7, limit: int = 15) -> List[Dict[str, Any]]:
    """
    Return: [{facet: "Soort", value: "Motie", count: 12}, ...]
    """
    conn = _connect()
    rows = conn.execute(
        """
        SELECT facet_includes_json
        FROM search_events
        WHERE ts_utc >= datetime('now', ?)
        """,
        (f"-{days} days",),
    ).fetchall()
    conn.close()

    freq: Dict[tuple, int] = {}
    for (raw,) in rows:
        if not raw:
            continue
        try:
            d = json.loads(raw)
        except Exception:
            continue
        for facet, values in (d or {}).items():
            for val in values or []:
                key = (facet, str(val))
                freq[key] = freq.get(key, 0) + 1

    items = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return [{"facet": k[0], "value": k[1], "count": v} for k, v in items]


def get_usage_kpis(days: int = 7) -> Dict[str, Any]:
    conn = _connect()
    row = conn.execute(
        """
        SELECT
            COUNT(*) as searches,
            COUNT(DISTINCT session_id) as sessions,
            AVG(results_count) as avg_results,
            AVG(runtime_s) as avg_runtime
        FROM search_events
        WHERE ts_utc >= datetime('now', ?)
        """,
        (f"-{days} days",),
    ).fetchone()
    conn.close()

    return {
        "searches": int(row[0] or 0),
        "sessions": int(row[1] or 0),
        "avg_results": float(row[2] or 0.0),
        "avg_runtime": float(row[3] or 0.0),
    }
