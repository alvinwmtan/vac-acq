"""Silver-parse unparsed Eng-NA/Eng-UK utterances. Never overwrite 2026.1 tags."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from vacacq import CACHE

NONWORDS = {"xxx", "yyy", "www"}
PARSE_SOURCES = ("childesdb", "batchalign2", "stanza_gloss")


def _has_tag(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    return text.notna() & (text.str.strip() != "") & (text.str.lower() != "nan")


def annotate_existing_parses(tokens: pd.DataFrame) -> pd.DataFrame:
    """Mark tokens that already have 2026.1 UD tags as parse_source=childesdb."""
    df = tokens.copy()
    if "parse_source" not in df.columns:
        df["parse_source"] = pd.Series(pd.NA, index=df.index, dtype="string")
    parsed = _has_tag(df["part_of_speech"]) & _has_tag(df["gra_relation"])
    missing = df["parse_source"].isna() | (df["parse_source"].astype(str).str.strip() == "")
    df.loc[parsed & missing, "parse_source"] = "childesdb"
    return df


def is_nonword(gloss: object) -> bool:
    g = str(gloss or "").strip().lower()
    return g in NONWORDS or g.startswith("&")


def utterance_fully_unparsed(tokens: pd.DataFrame) -> bool:
    """True when content tokens have no POS (unparsed file), not sporadic retraces."""
    content = tokens.loc[~tokens["gloss"].map(is_nonword)]
    if content.empty:
        return False
    return not _has_tag(content["part_of_speech"]).any()


def _utterance_text(tokens: pd.DataFrame) -> str:
    return " ".join(str(g) for g in tokens.sort_values("token_order")["gloss"].tolist())


def _align_stanza_to_tokens(tokens: pd.DataFrame, sentence) -> pd.DataFrame:
    """Greedy lowercase gloss alignment of a Stanza sentence onto CHILDES tokens."""
    df = tokens.sort_values("token_order").copy()
    words = list(sentence.words)
    wi = 0
    pos = [None] * len(df)
    head = [None] * len(df)
    rel = [None] * len(df)
    index = [None] * len(df)
    lemma = [None] * len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if is_nonword(row.get("gloss")):
            continue
        gloss = str(row.get("gloss") or "").lower().strip()
        if not gloss or wi >= len(words):
            continue
        w = words[wi]
        if w.text.lower().strip() == gloss or w.text.lower().strip().startswith(gloss[:3]):
            pos[i] = w.upos
            index[i] = w.id
            head[i] = w.head
            rel[i] = w.deprel
            lemma[i] = w.lemma
            wi += 1
            continue
        if w.upos == "PUNCT":
            wi += 1
            if wi < len(words) and words[wi].text.lower().strip() == gloss:
                w = words[wi]
                pos[i] = w.upos
                index[i] = w.id
                head[i] = w.head
                rel[i] = w.deprel
                lemma[i] = w.lemma
                wi += 1
    df["silver_pos"] = pos
    df["silver_gra_index"] = index
    df["silver_gra_head"] = head
    df["silver_gra_relation"] = rel
    df["silver_lemma"] = lemma
    return df


def _apply_silver(tokens: pd.DataFrame, silver: pd.DataFrame, source: str) -> pd.DataFrame:
    """Write silver tags only where 2026.1 tags are empty."""
    out = tokens.copy()
    for _, row in silver.iterrows():
        idx = out.index[out["id"] == row["id"]] if "id" in out.columns else []
        if len(idx) != 1:
            continue
        i = idx[0]
        if not _has_tag(out.loc[[i], "part_of_speech"]).iloc[0] and row.get("silver_pos"):
            out.at[i, "part_of_speech"] = row["silver_pos"]
            out.at[i, "parse_source"] = source
        if not _has_tag(out.loc[[i], "gra_relation"]).iloc[0] and row.get("silver_gra_relation"):
            out.at[i, "gra_index"] = row["silver_gra_index"]
            out.at[i, "gra_head"] = row["silver_gra_head"]
            out.at[i, "gra_relation"] = row["silver_gra_relation"]
            out.at[i, "parse_source"] = source
        if (not _has_tag(out.loc[[i], "stem"]).iloc[0] if "stem" in out.columns else True) and row.get("silver_lemma"):
            out.at[i, "stem"] = row["silver_lemma"]
    return out


def parse_gloss_stanza(tokens: pd.DataFrame, nlp) -> pd.DataFrame:
    """Stanza-on-gloss fallback. `nlp` is a stanza.Pipeline."""
    text = _utterance_text(tokens)
    doc = nlp(text)
    if not doc.sentences:
        return tokens
    silver = _align_stanza_to_tokens(tokens, doc.sentences[0])
    return _apply_silver(tokens, silver, "stanza_gloss")


def pos_agreement(gold: pd.Series, pred: pd.Series) -> float:
    mask = _has_tag(gold) & _has_tag(pred)
    if not mask.any():
        return float("nan")
    return float((gold[mask].astype(str) == pred[mask].astype(str)).mean())


def batchalign2_available() -> bool:
    return shutil.which("batchalign") is not None


def run_batchalign2_chat(chat_path: Path, lang: str = "eng") -> subprocess.CompletedProcess | None:
    """Run Batchalign2 morphotag if the CLI is installed. Does not overwrite 2026.1."""
    if not batchalign2_available():
        return None
    return subprocess.run(
        ["batchalign", "morphotag", f"--lang={lang}", str(chat_path)],
        check=False,
        capture_output=True,
        text=True,
    )


def fill_unparsed(
    tokens: pd.DataFrame,
    *,
    nlp=None,
    use_stanza: bool = True,
) -> pd.DataFrame:
    """Fill fully unparsed utterances. Existing tags are never overwritten."""
    tokens = annotate_existing_parses(tokens)
    if tokens.empty or "utterance_id" not in tokens.columns:
        return tokens
    filled_parts = []
    n_filled = 0
    n_skipped_partial = 0
    for _, grp in tokens.groupby("utterance_id", sort=False):
        if not utterance_fully_unparsed(grp):
            if not _has_tag(grp["part_of_speech"]).all():
                n_skipped_partial += 1
            filled_parts.append(grp)
            continue
        if use_stanza:
            if nlp is None:
                import stanza

                nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,lemma,depparse", verbose=False)
            filled_parts.append(parse_gloss_stanza(grp.copy(), nlp))
            n_filled += 1
        else:
            filled_parts.append(grp)
    out = pd.concat(filled_parts, ignore_index=True) if filled_parts else tokens
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "parse_fill_status.json").write_text(
        json.dumps(
            {
                "n_utterances_filled": n_filled,
                "n_partial_missing_left_unparsed": n_skipped_partial,
                "batchalign2_cli": batchalign2_available(),
                "stanza_used": use_stanza,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def sanity_check_parsed_sample(tokens: pd.DataFrame, nlp, n: int = 50) -> dict:
    """POS agreement of Stanza-on-gloss vs stored 2026.1 tags on already-parsed utterances."""
    parsed = tokens.loc[_has_tag(tokens["part_of_speech"])]
    if parsed.empty:
        return {"n": 0, "pos_agreement": None}
    sample_ids = parsed["utterance_id"].drop_duplicates().head(n)
    scores = []
    for uid in sample_ids:
        grp = tokens.loc[tokens["utterance_id"] == uid].copy()
        pred = parse_gloss_stanza(grp.copy(), nlp)
        scores.append(pos_agreement(grp["part_of_speech"], pred["part_of_speech"]))
    scores = [s for s in scores if s == s]
    return {
        "n": len(scores),
        "pos_agreement": float(sum(scores) / len(scores)) if scores else None,
        "note": "Large mismatch means Stanza version ≠ TalkBank Batchalign; pin that version if documented.",
    }
