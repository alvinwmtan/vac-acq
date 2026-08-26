"""Load Eng-NA/Eng-UK analysis tokens from Redivis or a local parquet cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vacacq import CACHE
from vacacq.childes.access import ENGLISH_COLLECTIONS, quote_sql, try_query
from vacacq.childes.strata import (
    AGE_MAX,
    AGE_MIN,
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
    age_unit: str = "days",
) -> str:
    roles = roles or ROLES
    if age_unit == "none":
        age_pred = None
    elif age_unit == "months":
        age_pred = f"t.target_child_age >= {AGE_MIN} AND t.target_child_age < {AGE_MAX}"
    else:
        age_pred = (
            f"t.target_child_age >= {AGE_MIN * DAYS_PER_MONTH} AND "
            f"t.target_child_age < {AGE_MAX * DAYS_PER_MONTH}"
        )
    where = [
        f"t.collection_name IN ({quote_sql(list(ENGLISH_COLLECTIONS))})",
        f"t.speaker_role IN ({quote_sql(list(roles))})",
    ]
    if age_pred:
        where.append(age_pred)
    if corpora:
        where.append(f"t.corpus_name IN ({quote_sql(corpora)})")
    return f"""
SELECT {TOKEN_COLUMNS}
FROM token t
LEFT JOIN transcript tr ON t.transcript_id = tr.id
WHERE {" AND ".join(where)}
"""


TOKENS_SQL = _tokens_sql()


def corpus_fetch_diag(corpus: str) -> dict:
    sql = f"""
SELECT
  COUNT(*) AS n,
  COUNTIF(target_child_age IS NULL) AS n_null_age,
  MIN(target_child_age) AS min_age,
  MAX(target_child_age) AS max_age,
  COUNTIF(speaker_role IN ({quote_sql(list(ROLES))})) AS n_cds_roles
FROM token
WHERE collection_name IN ({quote_sql(list(ENGLISH_COLLECTIONS))})
  AND corpus_name IN ({quote_sql([corpus])})
"""
    df = try_query(sql)
    if df is None or df.empty:
        err = CACHE / "redivis_error.txt"
        return {"error": err.read_text(encoding="utf-8") if err.exists() else "query failed"}
    return df.iloc[0].to_dict()


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
    keep_na_age = False
    if df.empty and corpora:
        for unit in ("months", "none"):
            sql = _tokens_sql(corpora=corpora, roles=roles, age_unit=unit)
            if limit is not None:
                sql = sql + f" LIMIT {int(limit)}"
            alt = try_query(sql)
            if alt is None:
                return pd.DataFrame()
            if not alt.empty:
                df = alt
                keep_na_age = unit == "none"
                break
    df = _annotate(df)
    if keep_na_age and not df.empty and "target_child_age" in df.columns:
        age = df["target_child_age"]
        df = df.loc[age.isna() | ((age >= AGE_MIN) & (age < AGE_MAX))].copy()
    if cache and corpora is None and roles is None:
        CACHE.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cached, index=False)
    return df


def list_english_corpora() -> list[str]:
    cov = CACHE / "coverage.csv"
    if cov.exists():
        return pd.read_csv(cov)["corpus_name"].astype(str).drop_duplicates().tolist()
    from vacacq.childes.coverage import audit_coverage

    df = audit_coverage()
    return [] if df.empty else df["corpus_name"].astype(str).drop_duplicates().tolist()


def load_corpora_tokens(
    corpora: list[str],
    *,
    roles: tuple[str, ...] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch one corpus at a time, caching each parquet under data/cache/."""
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    for corpus in corpora:
        path = CACHE / f"tokens_{corpus}.parquet"
        if path.exists() and not force:
            try:
                cached = pd.read_parquet(path)
            except Exception:
                cached = pd.DataFrame()
            if cached.empty:
                path.unlink(missing_ok=True)
            else:
                print(f"cache hit {corpus}: {path}")
                frames.append(_annotate(cached))
                continue
        print(f"fetching {corpus} from childes-db 2026.1 …", flush=True)
        df = load_analysis_tokens(corpora=[corpus], roles=roles)
        if df.empty:
            err_path = CACHE / "redivis_error.txt"
            if err_path.exists():
                err = err_path.read_text(encoding="utf-8")
                raise RuntimeError(f"Redivis failed for {corpus}: {err[:400]}")
            print(f"{corpus}: 0 rows in the {AGE_MIN:.0f}–{AGE_MAX:.0f} month child/parent window")
            continue
        df.to_parquet(path, index=False)
        print(f"wrote {path} ({len(df)} tokens)", flush=True)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
