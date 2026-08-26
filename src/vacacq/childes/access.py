"""childes-db 2026.1 access via Redivis (datapages.childes_db v1.4)."""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from vacacq import CACHE

DB_VERSION = "2026.1"
REDIVIS_TAG = "v1.4"
ORGANIZATION = "datapages"
DATASET = "childes_db:b6q6"
ENGLISH_COLLECTIONS = ("Eng-NA", "Eng-UK")


def _dataset(version: str = REDIVIS_TAG):
    import redivis

    return redivis.organization(ORGANIZATION).dataset(DATASET, version=version)


def query(sql: str, version: str = REDIVIS_TAG) -> pd.DataFrame:
    """Run SQL against the pinned childes-db Redivis dataset."""
    tmp = CACHE / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp)
    os.environ["TMP"] = str(tmp)
    os.environ["TEMP"] = str(tmp)
    CACHE.mkdir(parents=True, exist_ok=True)
    table = _dataset(version).query(sql)
    try:
        return table.to_pandas_dataframe(max_parallelization=1)
    except TypeError:
        return table.to_pandas_dataframe()


def fetch_table(name: str, version: str = REDIVIS_TAG) -> pd.DataFrame:
    return _dataset(version).table(name).to_pandas_dataframe()


def quote_sql(values: list[str]) -> str:
    escaped = []
    for v in values:
        escaped.append("'" + v.replace("\\", "\\\\").replace("'", "\\'") + "'")
    return ", ".join(escaped)


def try_query(sql: str) -> Optional[pd.DataFrame]:
    """Return a frame or None if Redivis is unreachable / unauthenticated."""
    try:
        out = query(sql)
        err = CACHE / "redivis_error.txt"
        if err.exists():
            err.unlink()
        return out
    except Exception as exc:  # noqa: BLE001 — surface later in coverage JSON
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / "redivis_error.txt").write_text(str(exc), encoding="utf-8")
        return None


def redivis_token_present() -> bool:
    return bool(os.environ.get("REDIVIS_API_TOKEN"))
