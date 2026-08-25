"""QC against UD-English-CHILDES and S7.6 inclusion thresholds."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vacacq import CACHE
from vacacq.childes.extract import extract_vacs
from vacacq.io import load_json

S7_6 = {
    "function_agreement_min": 0.70,
    "function_precision_min": 0.60,
    "token_min": 30,
    "form_precision_mean": 0.969,
    "function_precision_mean": 0.875,
    "recall_mean": 0.705,
    "per_construction": {
        "VO": {"form": 0.956, "function": 0.942, "recall": 0.675},
        "VOO": {"form": 0.947, "function": 0.947, "recall": 0.467},
        "VL": {"form": 0.971, "function": 0.887, "recall": 0.716},
        "VOL": {"form": 0.970, "function": 0.852, "recall": 0.706},
    },
    "noisy": ["VL-to", "VL-in", "VOL-in", "VL-about", "VOL-about"],
}


def conllu_to_tokens(path: Path, utterance_prefix: str = "ud") -> pd.DataFrame:
    """Convert a UD CoNLL-U file to the extractor token table."""
    rows = []
    sent_i = 0
    tok_i = 0
    with path.open(encoding="utf-8") as fh:
        pending: list[dict] = []
        for line in fh:
            line = line.strip()
            if not line:
                if pending:
                    sent_i += 1
                    uid = f"{utterance_prefix}-{sent_i}"
                    for row in pending:
                        row["utterance_id"] = uid
                        rows.append(row)
                    pending = []
                continue
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if "-" in parts[0] or "." in parts[0]:
                continue
            tok_i += 1
            feats = parts[5]
            pending.append(
                {
                    "id": tok_i,
                    "token_order": int(parts[0]),
                    "gloss": parts[1],
                    "stem": parts[2],
                    "part_of_speech": parts[3],
                    "suffix": feats if feats != "_" else "",
                    "gra_index": int(parts[0]),
                    "gra_head": int(parts[6]) if parts[6] != "_" else 0,
                    "gra_relation": parts[7],
                    "parse_source": "ud_english_childes",
                    "speaker_role": "Target_Child",
                    "utterance_type": "declarative",
                }
            )
        if pending:
            sent_i += 1
            uid = f"{utterance_prefix}-{sent_i}"
            for row in pending:
                row["utterance_id"] = uid
                rows.append(row)
    return pd.DataFrame(rows)


def extract_ud_treebank(path: Path) -> pd.DataFrame:
    tokens = conllu_to_tokens(path)
    return extract_vacs(tokens, extractor="ud")


def qualitative_leads(occupancy: pd.DataFrame, stratum: str | None = None) -> dict:
    """Check S7.7–S7.12 lead-verb identity on an occupancy table."""
    expected = load_json("expected_leads.json")
    work = occupancy.copy()
    if stratum:
        work = work.loc[work["stratum"] == stratum]
    recovered = {}
    missing = {}
    for cons, verbs in expected["schematic"].items():
        sub = work.loc[work["construction"] == cons]
        if sub.empty:
            missing[cons] = verbs
            continue
        top = (
            sub.groupby("verb")["vac_freq"].sum().sort_values(ascending=False).head(3).index.str.lower().tolist()
        )
        recovered[cons] = top
        miss = [v for v in verbs if v not in top]
        if miss:
            missing[cons] = miss
    for key, verbs in expected["prep_specific"].items():
        if key.endswith("_parents"):
            vac = key.replace("_parents", "")
            sub = work.loc[work["stratum"].astype(str).str.startswith("Parent")]
        elif key.endswith("_children"):
            vac = key.replace("_children", "")
            sub = work.loc[work["stratum"].astype(str).str.startswith("Child")]
        else:
            vac = key
            sub = work
        cons, prep = vac.split("-", 1)
        sub = sub.loc[(sub["construction"] == cons) & (sub["preposition"].astype(str) == prep)]
        top = (
            sub.groupby("verb")["vac_freq"].sum().sort_values(ascending=False).head(3).index.str.lower().tolist()
            if not sub.empty
            else []
        )
        recovered[key] = top
        miss = [v for v in verbs if v not in top]
        if miss:
            missing[key] = miss
    return {"recovered": recovered, "missing": missing, "expected": expected}


def handcheck_sample(hits: pd.DataFrame, n_per: int = 50, seed: int = 0) -> pd.DataFrame:
    """50 parent + 50 child rows per schematic VAC for form/function coding."""
    schematic = hits.loc[hits["construction"].isin(["VL", "VOL", "VOO", "VO"])].copy()
    if "extra" in schematic.columns:
        schematic = schematic.loc[~schematic["extra"].fillna(False)]
    schematic["side"] = schematic["speaker_role"].map(
        lambda r: "child" if r in {"Target_Child", "Child"} else "parent" if r in {"Mother", "Father"} else "other"
    )
    parts = []
    for cons, grp in schematic.groupby("construction"):
        for side in ("parent", "child"):
            pool = grp.loc[grp["side"] == side]
            take = min(n_per, len(pool))
            if take:
                parts.append(pool.sample(take, random_state=seed))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not out.empty:
        out["form_ok"] = pd.NA
        out["function_ok"] = pd.NA
        CACHE.mkdir(parents=True, exist_ok=True)
        out.to_csv(CACHE / "qc_handcheck_sample.csv", index=False)
    return out


def write_qc_report(occupancy: pd.DataFrame, hits: pd.DataFrame | None = None) -> dict:
    leads = qualitative_leads(occupancy)
    report = {
        "s7_6_thresholds": S7_6,
        "lead_verbs": leads,
        "handcheck_csv": str(CACHE / "qc_handcheck_sample.csv") if hits is not None else None,
    }
    if hits is not None:
        handcheck_sample(hits)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "qc_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
