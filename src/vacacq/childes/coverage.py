"""UD %mor/%gra coverage audit for Eng-NA and Eng-UK."""

from __future__ import annotations

import json

import pandas as pd

from vacacq import CACHE
from vacacq.childes.access import ENGLISH_COLLECTIONS, quote_sql, try_query
from vacacq.childes.strata import included_corpus_names, s7_2


COVERAGE_SQL = f"""
SELECT
  collection_name,
  corpus_name,
  COUNT(*) AS n_tokens,
  COUNTIF(part_of_speech IS NOT NULL AND part_of_speech != '') AS n_pos,
  COUNTIF(gra_relation IS NOT NULL AND gra_relation != '') AS n_gra,
  COUNTIF(
    part_of_speech IS NOT NULL AND part_of_speech != ''
    AND gra_relation IS NOT NULL AND gra_relation != ''
  ) AS n_parsed
FROM token
WHERE collection_name IN ({quote_sql(list(ENGLISH_COLLECTIONS))})
GROUP BY collection_name, corpus_name
ORDER BY collection_name, corpus_name
"""


def audit_coverage() -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    df = try_query(COVERAGE_SQL)
    if df is None:
        empty = pd.DataFrame(
            columns=[
                "collection_name",
                "corpus_name",
                "n_tokens",
                "n_pos",
                "n_gra",
                "n_parsed",
                "pos_rate",
                "gra_rate",
                "parsed_rate",
                "in_s7_2",
            ]
        )
        note = {
            "status": "redivis_unavailable",
            "detail": "Could not query childes-db 2026.1. Set REDIVIS_API_TOKEN or log in via the redivis client.",
            "error_file": str(CACHE / "redivis_error.txt"),
        }
        (CACHE / "coverage_status.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
        empty.to_parquet(CACHE / "coverage.parquet", index=False)
        return empty

    df["pos_rate"] = df["n_pos"] / df["n_tokens"].clip(lower=1)
    df["gra_rate"] = df["n_gra"] / df["n_tokens"].clip(lower=1)
    df["parsed_rate"] = df["n_parsed"] / df["n_tokens"].clip(lower=1)
    names = included_corpus_names()
    df["in_s7_2"] = df["corpus_name"].str.lower().isin(names)
    missing = sorted(
        n for n in {c.lower() for c in s7_2()["included"]} if n not in set(df["corpus_name"].str.lower())
    )
    # Bloom70/Bloom73 are stored as Bloom in 2026.1
    missing = [n for n in missing if n not in {"bloom70", "bloom73"} or "bloom" not in set(df["corpus_name"].str.lower())]
    df.to_parquet(CACHE / "coverage.parquet", index=False)
    df.to_csv(CACHE / "coverage.csv", index=False)
    zero = df.loc[df["parsed_rate"] == 0, "corpus_name"].tolist()
    low = df.loc[df["parsed_rate"] < 0.5, ["collection_name", "corpus_name", "parsed_rate"]].to_dict(orient="records")
    (CACHE / "coverage_status.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "n_corpora": int(len(df)),
                "n_s7_2": int(df["in_s7_2"].sum()),
                "n_new": int((~df["in_s7_2"]).sum()),
                "mean_parsed_rate": float(df["parsed_rate"].mean()),
                "s7_2_missing_from_2026_1": missing,
                "zero_parse_corpora": zero,
                "parsed_rate_below_0.5": low,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return df


def parsed_corpus_names(*, min_rate: float = 0.0) -> set[str] | None:
    """Corpus names with parsed_rate > min_rate. None if coverage has not been audited."""
    path = CACHE / "coverage.parquet"
    if not path.exists():
        path = CACHE / "coverage.csv"
    if not path.exists():
        return None
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "parsed_rate" not in df.columns or "corpus_name" not in df.columns:
        return None
    kept = df.loc[df["parsed_rate"] > min_rate, "corpus_name"].astype(str)
    return set(kept)


def drop_unparsed_corpora(frame: pd.DataFrame, *, min_rate: float = 0.0) -> pd.DataFrame:
    """Drop rows whose corpus has no (or below-threshold) UD parses."""
    if frame.empty or "corpus_name" not in frame.columns:
        return frame
    names = parsed_corpus_names(min_rate=min_rate)
    if names is None:
        return frame
    return frame.loc[frame["corpus_name"].astype(str).isin(names)].copy()
