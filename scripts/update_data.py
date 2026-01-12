import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests

DATA_DIR = "data"
DOCS_PATH = os.path.join(DATA_DIR, "documents.parquet")
META_PATH = os.path.join(DATA_DIR, "metadata.json")

BASE_ENDPOINT = "https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0/Document"

SELECT_FIELDS = [
    "Onderwerp",
    "Vergaderjaar",
    "DatumRegistratie",
    "DatumOntvangst",
    "Aanhangselnummer",
    "Titel",
    "Soort",
    "DocumentNummer",
    "Id",
]

DEFAULT_START_ISO = "2019-11-30T00:00:00Z"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_metadata() -> Dict:
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "last_successful_fetch_iso": DEFAULT_START_ISO,
        "rows_total": 0,
        "updated_utc": None,
    }


def save_metadata(meta: Dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def fetch_odata_all(url: str, params: Optional[Dict]) -> List[Dict]:
    """
    Fetch OData paginated results following @odata.nextLink.
    First request uses `params`; subsequent calls use the nextLink URL directly.
    """
    out: List[Dict] = []
    next_url: Optional[str] = url
    next_params: Optional[Dict] = params

    while next_url:
        r = requests.get(next_url, params=next_params, timeout=60)
        r.raise_for_status()
        payload = r.json()

        out.extend(payload.get("value", []))

        next_url = payload.get("@odata.nextLink")
        next_params = None  # nextLink already includes query

    return out


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    # Keep timezone-aware datetimes for analysis flexibility
    for col in ["DatumRegistratie", "DatumOntvangst"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def load_existing() -> pd.DataFrame:
    if os.path.exists(DOCS_PATH):
        return pd.read_parquet(DOCS_PATH)
    return pd.DataFrame(columns=SELECT_FIELDS)


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    meta = load_metadata()
    last_iso = meta.get("last_successful_fetch_iso") or DEFAULT_START_ISO

    params = {
        "$select": ",".join(SELECT_FIELDS),
        # Gebruik "ge" om niets te missen; dedupe gebeurt op Id.
        "$filter": f"Verwijderd eq false and DatumRegistratie ge {last_iso}",
        "$orderby": "DatumRegistratie asc",
    }

    print(f"[update] Fetching since: {last_iso}")
    new_rows = fetch_odata_all(BASE_ENDPOINT, params)

    if not new_rows:
        # Geen updates: update alleen metadata timestamp (optioneel)
        meta["updated_utc"] = utc_now_iso()
        save_metadata(meta)
        print("[update] No new rows. Metadata updated.")
        return

    new_df = pd.DataFrame(new_rows)
    new_df = normalize_dates(new_df)

    old_df = load_existing()
    combined = pd.concat([old_df, new_df], ignore_index=True)

    # Dedupe op Id: behoud laatste versie
    if "DatumRegistratie" in combined.columns:
        combined = combined.sort_values("DatumRegistratie")
    combined = combined.drop_duplicates(subset=["Id"], keep="last").reset_index(drop=True)

    combined.to_parquet(DOCS_PATH, index=False)

    meta["last_successful_fetch_iso"] = utc_now_iso()
    meta["rows_total"] = int(len(combined))
    meta["updated_utc"] = meta["last_successful_fetch_iso"]
    save_metadata(meta)

    print(f"[update] Saved {DOCS_PATH}")
    print(f"[update] Total rows: {len(combined)}")


if __name__ == "__main__":
    main()
