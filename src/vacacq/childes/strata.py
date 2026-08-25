"""Age strata, speaker roles, and S7.2 include/exclude matching."""

from __future__ import annotations

import pandas as pd

from vacacq.io import load_json

CHILD_ROLES = {"Target_Child", "Child"}
PARENT_ROLES = {"Mother", "Father"}

# childes-db 2026.1 Redivis stores target_child_age in days (365.25/12 per month).
DAYS_PER_MONTH = 365.25 / 12

# months
CHILD_I = (18.0, 36.0)
CHILD_II = (36.0, 72.0)
PARENT_YOUNG_MAX = 37.0
PARENT_ALL_MAX = 72.0


def age_to_months(age: float | None) -> float | None:
    """Convert 2026.1 day-valued ages to months; leave already-monthly values alone."""
    if age is None:
        return None
    try:
        val = float(age)
    except (TypeError, ValueError):
        return None
    if val != val:
        return None
    if val > 120:
        return val / DAYS_PER_MONTH
    return val


def s7_2() -> dict:
    return load_json("s7_2_included_corpora.json")


def included_corpus_names() -> set[str]:
    spec = s7_2()
    names = {c.lower() for c in spec["included"]}
    for alias in spec.get("name_aliases", {}):
        names.add(alias.lower())
    return names


def filename_matches(filename: str, patterns: list[str]) -> bool:
    fn = (filename or "").replace("\\", "/")
    low = fn.lower()
    for pat in patterns:
        p = pat.replace("\\", "/").lower()
        if p.rstrip("/") in low:
            return True
    return False


def is_s7_2_corpus(corpus_name: str) -> bool:
    return (corpus_name or "").lower() in included_corpus_names()


def apply_s7_2_exclusions(
    transcripts: pd.DataFrame,
    *,
    stratum: str,
) -> pd.DataFrame:
    """Drop S7.2 excluded files. `stratum` is child, parent_young, or parent_all."""
    spec = s7_2()
    patterns = list(spec["filename_exclusions"])
    if stratum in {"parent_young", "parent_all"}:
        patterns += spec["parent_young_and_parent_all_file_exclusions"]
    if stratum == "parent_all":
        patterns += spec["parent_all_only_file_exclusions"]
    col = "filename" if "filename" in transcripts.columns else "transcript_filename"
    if col not in transcripts.columns:
        return transcripts
    mask = ~transcripts[col].astype(str).map(lambda x: filename_matches(x, patterns))
    return transcripts.loc[mask].copy()


def assign_strata(role: str, target_child_age_months: float | None) -> list[str]:
    """Return all matching Chapter 7 strata. Parent All includes Parent Young."""
    role_n = (role or "").strip()
    age = target_child_age_months
    out: list[str] = []
    if role_n in CHILD_ROLES and age is not None:
        if CHILD_I[0] <= age < CHILD_I[1]:
            out.append("Child I")
        if CHILD_II[0] <= age < CHILD_II[1]:
            out.append("Child II")
    if role_n in PARENT_ROLES and age is not None:
        if age < PARENT_YOUNG_MAX:
            out.append("Parent Young")
        if age < PARENT_ALL_MAX:
            out.append("Parent All")
    return out


def assign_stratum(role: str, target_child_age_months: float | None) -> str | None:
    """Narrowest matching stratum (Child I/II, Parent Young, else Parent All)."""
    strata = assign_strata(role, target_child_age_months)
    if not strata:
        return None
    for preferred in ("Child I", "Child II", "Parent Young", "Parent All"):
        if preferred in strata:
            return preferred
    return strata[0]


def parent_all_mask(stratum: str) -> bool:
    return stratum in {"Parent Young", "Parent All"}
