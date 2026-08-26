"""Load Eng-NA/Eng-UK analysis tokens from Redivis or a local parquet cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vacacq import CACHE
from vacacq.childes.access import ENGLISH_COLLECTIONS, quote_sql, try_query
from vacacq.childes.strata import (
    CHILD_ROLES,
    DAYS_PER_MONTH,
    PARENT_ROLES,
    age_to_months,
    apply_s7_2_exclusions,
    is_s7_2_corpus,
)
from vacacq.parse.fill import annotate_existing_parses

ROLES = tuple(sorted(CHILD_ROLES | PARENT_ROLES))

TOKEN_COLUMNS = """
  t.id,
  t.utterance_id,
  t.transcript_id,
  t.token_order,
  t.gloss,
  t.part_of_speech,
  t.stem,
  t.suffix,
  t.clitic,
  t.gra_index,
  t.gra_head,
  t.gra_relation,
  t.collection_name,
  t.corpus_name,
  t.speaker_role,
  t.utterance_type,
  t.target_child_age,
  t.target_child_id,
  t.target_child_name,
  tr.filename
"""


def _tokens_sql(
    *,
    corpora: list[str] | None = None,
    roles: tuple[str, ...] | None = None,
) -> str:
    roles = roles or ROLES
    where = [
        f"t.collection_name IN ({quote_sql(list(ENGLISH_COLLECTIONS))})",
        f"t.speaker_role IN ({quote_sql(list(roles))})",
        f"t.target_child_age >= {18 * DAYS_PER_MONTH}",
        f"t.target_child_age < {72 * DAYS_PER_MONTH}",
    ]
    if corpora:
        where.append(f"t.corpus_name IN ({quote_sql(corpora)})")
    return f"""
SELECT {TOKEN_COLUMNS}
FROM token t
LEFT JOIN transcript tr ON t.transcript_id = tr.id
WHERE {" AND ".join(where)}
"""


TOKENS_SQL = _tokens_sql()


def _annotate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["in_s7_2"] = df["corpus_name"].astype(str).map(is_s7_2_corpus)
    if "target_child_age" in df.columns and "target_child_age_days" not in df.columns:
        df["target_child_age_days"] = df["target_child_age"]
        df["target_child_age"] = df["target_child_age"].map(age_to_months)
    if "filename" in df.columns:
        child = df.loc[df["speaker_role"].isin(CHILD_ROLES)]
        parent = df.loc[df["speaker_role"].isin(PARENT_ROLES)]
        other = df.loc[~df["speaker_role"].isin(CHILD_ROLES | PARENT_ROLES)]
        child = apply_s7_2_exclusions(child, stratum="child")
        parent = apply_s7_2_exclusions(parent, stratum="parent_all")
        df = pd.concat([child, parent, other], ignore_index=True)
    df = annotate_existing_parses(df)
    return df


def load_analysis_tokens(
    path: str | Path | None = None,
    *,
    limit: int | None = None,
    cache: bool = False,
    corpora: list[str] | None = None,
    roles: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Load tokens. Prefer `path`, then a local parquet cache, then Redivis."""
    if path is not None:
        df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
        return _annotate(df)

    cached = CACHE / "tokens_eng.parquet"
    if cached.exists() and limit is None and corpora is None and roles is None:
        return _annotate(pd.read_parquet(cached))

    sql = _tokens_sql(corpora=corpora, roles=roles)
    if limit is not None:
        sql = sql + f" LIMIT {int(limit)}"
    df = try_query(sql)
    if df is None:
        return pd.DataFrame()
    df = _annotate(df)
    if cache and corpora is None and roles is None:
        CACHE.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cached, index=False)
    return df


def load_corpora_tokens(
    corpora: list[str],
    *,
    roles: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Fetch one corpus at a time, caching each parquet under data/cache/."""
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    for corpus in corpora:
        path = CACHE / f"tokens_{corpus}.parquet"
        if path.exists():
            print(f"cache hit {corpus}: {path}")
            frames.append(_annotate(pd.read_parquet(path)))
            continue
        print(f"fetching {corpus} from childes-db 2026.1 …", flush=True)
        df = try_query(_tokens_sql(corpora=[corpus], roles=roles))
        if df is None:
            err = (CACHE / "redivis_error.txt").read_text(encoding="utf-8") if (CACHE / "redivis_error.txt").exists() else "unknown"
            raise RuntimeError(f"Redivis failed for {corpus}: {err[:400]}")
        if df.empty:
            print(f"{corpus}: 0 rows in the 18–72 month child/parent window")
            df.to_parquet(path, index=False)
            continue
        df = _annotate(df)
        df.to_parquet(path, index=False)
        print(f"wrote {path} ({len(df)} tokens)", flush=True)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
