"""Stratified VAC hit sample from corpora added between childes-db 2021.1 and 2026.1."""

from __future__ import annotations

import re
from pathlib import Path

import lemminflect
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from vacacq import CACHE, DATA
from vacacq.childes.strata import CHILD_ROLES
from vacacq.stats.acquisition import MIN_ADULT_SHARE, MIN_TOKENS, SCHEMATIC, adult_frame_share

HITS_PATH = CACHE / "hits_filled.parquet"
TOKENS_PATH = CACHE / "tokens_filled.parquet"
CORPORA_2021 = CACHE / "corpora_2021_1.csv"
OUT_PATH = DATA / "eval" / "test_set.csv"
N_PER_CONSTRUCTION = 100
N_VERBS = 10
N_ALTS = 5
SEED = 20261
MIN_SHARE_TOKENS = 50
MAX_DISTRACTOR_SHARE = 0.10
MIN_SENT_TOKENS = 4
MAX_SENT_TOKENS = 14
_XXX = re.compile(r"\bxxx\b", re.IGNORECASE)
_WORD = re.compile(r"^[A-Za-z]+(?:'[A-Za-z]+)?$")
_VERB_TAGS = ("VBG", "VBN", "VBD", "VBZ", "VBP", "VB")
_ALT_COLS = tuple(f"alt_{i}" for i in range(1, N_ALTS + 1))
_OK_UTT_TYPES = {"declarative", "question", "imperative_emphatic"}
_FILLERS = frozenset(
    "uh um er ah oh ow hmm hm huh ugh eh mm mhm uhhuh uhuh oops oop op wow hey yeah yes yep".split()
)
_NONSTANDARD = frozenset("gonna wanna gotta hafta gimme lemme dunno".split())
_BAD_START = frozenset("and but or so then well now here yeah yes okay ok oh because baby dad daddy mom mommy mama papa honey sweetie just actually maybe".split())
_PRONOUNS = frozenset("i you he she it we they me him her us them".split())
_SPANISH = frozenset(
    "ese esos esa esas vamos está estan hay para los las del yo tú tu su se te le lo sí muy pero este esta esto poner mira que qué el la un una con por nadie alguien va voy van tener tiene hacer silla sillas también siempre ahora porque cuando él ella ellos nosotros estoy estás quiero puedes puede".split()
)
_NON_ENGLISH_VERBS = frozenset(
    "tener hacer ir ver dar mirar haber ser estar poner querer traer comer abrir".split()
)
_BAD_GLOSS = frozenset("xxx yyy www xx yy cname tname sname".split())


def contains_xxx(sentence: str) -> bool:
    return bool(_XXX.search(sentence or ""))


def parquet_corpus_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    pf = pq.ParquetFile(path)
    if "corpus_name" not in pf.schema_arrow.names:
        return set()
    names: set[str] = set()
    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i, columns=["corpus_name"]).to_pandas()
        names.update(df["corpus_name"].astype(str).unique())
        del df
    return names


def corpora_added_in_2026_1() -> list[str]:
    old = set(pd.read_csv(CORPORA_2021)["corpus_name"].astype(str))
    return sorted(parquet_corpus_names(TOKENS_PATH) - old)


def top_frame_verbs(
    share: pd.DataFrame,
    *,
    n: int = N_VERBS,
) -> dict[str, list[str]]:
    """Up to n verbs whose highest adult schematic-VAC share is this frame, ranked by n_frame."""
    work = share.copy()
    work["verb"] = work["verb"].astype(str).str.lower()
    work = work.loc[work["verb"].str.fullmatch(r"[a-z]+", na=False)]
    work = work.loc[~work["verb"].isin(_NON_ENGLISH_VERBS)]
    work = work.sort_values(["share", "n_frame"], ascending=False)
    pref = work.drop_duplicates("verb", keep="first")
    out: dict[str, list[str]] = {}
    for cons in SCHEMATIC:
        sub = pref.loc[pref["construction"] == cons].sort_values(
            ["n_frame", "share"], ascending=False
        )
        out[cons] = sub["verb"].tolist()[:n]
    return out


def stratified_hits(
    hits: pd.DataFrame,
    *,
    n_per_construction: int = N_PER_CONSTRUCTION,
    seed: int = SEED,
) -> pd.DataFrame:
    """Round-robin over verb strata until the per-construction budget is filled."""
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    work = hits.loc[hits["construction"].isin(SCHEMATIC)].copy()
    work["verb"] = work["verb"].astype(str).str.lower()
    for cons in SCHEMATIC:
        sub = work.loc[work["construction"] == cons].copy()
        if sub.empty:
            continue
        sub["_stratum"] = sub["verb"].astype(str)
        strata = sub["_stratum"].unique().tolist()
        rng.shuffle(strata)
        bags: dict[str, np.ndarray] = {}
        for s in strata:
            idx = sub.index[sub["_stratum"] == s].to_numpy().copy()
            rng.shuffle(idx)
            bags[s] = idx
        picked: list[int] = []
        round_i = 0
        n_take = min(n_per_construction, int(len(sub)))
        while len(picked) < n_take:
            progressed = False
            for s in strata:
                if len(picked) >= n_take:
                    break
                bag = bags[s]
                if round_i < len(bag):
                    picked.append(int(bag[round_i]))
                    progressed = True
            if not progressed:
                break
            round_i += 1
        parts.append(sub.loc[picked].drop(columns=["_stratum"]))
    if not parts:
        return work.iloc[0:0]
    return pd.concat(parts, ignore_index=True)


def tokens_for_utterances(utterance_ids: set, tokens_path: Path = TOKENS_PATH) -> pd.DataFrame:
    if not utterance_ids:
        return pd.DataFrame()
    pf = pq.ParquetFile(tokens_path)
    want = (
        "utterance_id",
        "token_order",
        "gloss",
        "stem",
        "suffix",
        "part_of_speech",
        "gra_relation",
        "utterance_type",
    )
    cols = [c for c in want if c in pf.schema_arrow.names]
    chunks: list[pd.DataFrame] = []
    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i, columns=cols).to_pandas()
        keep = df.loc[df["utterance_id"].isin(utterance_ids)]
        if not keep.empty:
            chunks.append(keep)
        del df
    if not chunks:
        return pd.DataFrame()
    tok = pd.concat(chunks, ignore_index=True)
    if "suffix" in tok.columns:
        tok = tok.rename(columns={"suffix": "morph_suffix"})
    if "token_order" in tok.columns:
        tok = tok.sort_values(["utterance_id", "token_order"], kind="mergesort")
    return tok


def _join_gloss(grp: pd.DataFrame) -> str:
    words = [str(g).strip() for g in grp["gloss"].tolist() if pd.notna(g) and str(g).strip()]
    return " ".join(words)


def sentences_for_utterances(utterance_ids: set, tokens_path: Path = TOKENS_PATH) -> dict:
    """Reconstruct space-joined gloss strings for a small set of utterance ids."""
    tok = tokens_for_utterances(utterance_ids, tokens_path)
    if tok.empty:
        return {}
    return {uid: _join_gloss(grp) for uid, grp in tok.groupby("utterance_id", sort=False)}


def _content_tokens(grp: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for rec in grp.itertuples(index=False):
        gloss = rec.gloss
        if gloss is None or (isinstance(gloss, float) and pd.isna(gloss)):
            continue
        gloss = str(gloss).strip()
        if not gloss:
            continue
        stem = getattr(rec, "stem", None)
        if stem is None or (isinstance(stem, float) and pd.isna(stem)):
            stem = ""
        morph = getattr(rec, "morph_suffix", None)
        if morph is None or (isinstance(morph, float) and pd.isna(morph)):
            morph = ""
        pos = getattr(rec, "part_of_speech", None)
        if pos is None or (isinstance(pos, float) and pd.isna(pos)):
            pos = ""
        rows.append({"gloss": gloss, "stem": str(stem).strip(), "morph": str(morph).strip(), "pos": str(pos).strip().lower()})
    return rows


def _is_propn(pos) -> bool:
    p = str(pos or "").strip().lower().replace(":", " ").replace("_", " ")
    parts = p.split()
    return "propn" in parts or (len(parts) >= 2 and parts[0] == "n" and parts[1].startswith("prop"))


def _norm_rel(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip().lower().replace("_", "-").replace(":", "-")


def has_proper_noun(tokens: pd.DataFrame) -> bool:
    if tokens.empty or "part_of_speech" not in tokens.columns:
        return False
    return tokens["part_of_speech"].map(_is_propn).any()


def is_well_formed_sentence(tokens: pd.DataFrame) -> bool:
    if tokens.empty:
        return False
    utt = ""
    if "utterance_type" in tokens.columns:
        raw = tokens["utterance_type"].iloc[0]
        utt = "" if raw is None or (isinstance(raw, float) and pd.isna(raw)) else str(raw).strip().lower()
    if utt and utt not in _OK_UTT_TYPES:
        return False
    rows = _content_tokens(tokens)
    if not (MIN_SENT_TOKENS <= len(rows) <= MAX_SENT_TOKENS):
        return False
    first = rows[0]["gloss"].lower()
    if first in _BAD_START:
        return False
    if first == "come" and len(rows) > 1 and rows[1]["gloss"].lower() == "on":
        return False
    if first == "look" and len(rows) > 1 and rows[1]["gloss"].lower() in {"it", "and"}:
        return False
    if len(rows) >= 2 and rows[0]["gloss"].lower() in _PRONOUNS and rows[1]["gloss"].lower() in _PRONOUNS:
        return False
    if (
        len(rows) >= 2
        and rows[0]["pos"] == "verb"
        and rows[1]["pos"] == "verb"
        and rows[1]["gloss"].lower() != "me"
    ):
        return False
    for i, row in enumerate(rows):
        g = row["gloss"]
        gl = g.lower()
        if gl in _BAD_GLOSS or gl in _FILLERS or gl in _NONSTANDARD or gl in _SPANISH:
            return False
        if row["pos"] == "intj":
            return False
        if not _WORD.fullmatch(g):
            return False
        if i and gl == rows[i - 1]["gloss"].lower():
            return False
    glosses = [r["gloss"].lower() for r in rows]
    for i in range(len(glosses) - 3):
        if glosses[i : i + 2] == glosses[i + 2 : i + 4]:
            return False
    if glosses.count("if") >= 2:
        return False
    rels = tokens["gra_relation"].map(_norm_rel) if "gra_relation" in tokens.columns else pd.Series("", index=tokens.index)
    has_nsubj = bool(rels.str.startswith("nsubj").any())
    root = tokens.loc[rels.eq("root")] if "part_of_speech" in tokens.columns else tokens.iloc[0:0]
    has_verb_root = (not root.empty) and root["part_of_speech"].astype(str).str.strip().str.lower().eq("verb").any()
    if not has_verb_root:
        return False
    first_pos = rows[0]["pos"]
    imperative_shape = first_pos == "verb" or first in {"let's", "lets"}
    if not has_nsubj and not imperative_shape:
        return False
    if not has_nsubj:
        n_verbs = sum(r["pos"] == "verb" for r in rows)
        if n_verbs > 1 and first not in {"let", "let's", "lets"}:
            return False
        lemma0 = (rows[0]["stem"] or rows[0]["gloss"]).lower()
        if rows[0]["pos"] == "verb" and rows[0]["gloss"].lower() != lemma0:
            return False
    return True


def eval_utterance_ok(tokens: pd.DataFrame) -> bool:
    return (not has_proper_noun(tokens)) and is_well_formed_sentence(tokens)


def _surface_forms(lemma: str) -> set[str]:
    forms = lemminflect.getAllInflections(lemma.lower()) or {}
    out: set[str] = {lemma.lower()}
    for tag in _VERB_TAGS:
        out.update(v.lower() for v in forms.get(tag, ()))
    return out


def _stem_of(row: dict) -> str:
    stem = (row.get("stem") or "").lower()
    return stem or row["gloss"].lower()


def _verb_index(rows: list[dict], lemma: str) -> int | None:
    lemma = lemma.lower()
    inflected = _surface_forms(lemma)
    buckets: dict[str, list[int]] = {"verb": [], "aux": [], "other": []}
    for i, row in enumerate(rows):
        gloss = row["gloss"].lower()
        if _stem_of(row) != lemma and gloss != lemma and gloss not in inflected:
            continue
        pos = row.get("pos") or ""
        if pos == "verb":
            buckets["verb"].append(i)
        elif pos == "aux":
            buckets["aux"].append(i)
        else:
            buckets["other"].append(i)
    for key in ("verb", "aux", "other"):
        if buckets[key]:
            return buckets[key][0]
    return None


def verb_span(tokens: pd.DataFrame, lemma: str) -> dict | None:
    rows = _content_tokens(tokens)
    i = _verb_index(rows, lemma)
    if i is None:
        return None
    return {
        "prefix": " ".join(r["gloss"] for r in rows[:i]),
        "inflected_verb": rows[i]["gloss"],
        "suffix": " ".join(r["gloss"] for r in rows[i + 1 :]),
        "morph": rows[i]["morph"],
    }


def split_at_verb(tokens: pd.DataFrame, lemma: str) -> tuple[str, str, str] | None:
    span = verb_span(tokens, lemma)
    if span is None:
        return None
    return span["prefix"], span["inflected_verb"], span["suffix"]


def inflection_tag(lemma: str, gloss: str, morph: str = "") -> str:
    m = (morph or "").lower()
    if "part pres" in m or m.startswith("ger"):
        return "VBG"
    if "part past" in m:
        return "VBN"
    if "past" in m:
        return "VBD"
    if "pres s3" in m or m.endswith("s3"):
        return "VBZ"
    g = (gloss or "").lower()
    forms = lemminflect.getAllInflections((lemma or "").lower()) or {}
    for tag in _VERB_TAGS:
        if g in {v.lower() for v in forms.get(tag, ())}:
            return tag
    if g.endswith("ing"):
        return "VBG"
    if g != (lemma or "").lower() and g.endswith("s"):
        return "VBZ"
    return "VB"


def inflect_lemma(lemma: str, tag: str) -> str:
    lemma = lemma.lower()
    forms = lemminflect.getInflection(lemma, tag=tag)
    if forms:
        return forms[0]
    if tag == "VBN":
        forms = lemminflect.getInflection(lemma, tag="VBD")
        if forms:
            return forms[0]
    forms = lemminflect.getInflection(lemma, tag="VB")
    return forms[0] if forms else lemma


def match_case(template: str, word: str) -> str:
    if not template or not word:
        return word
    if template.isupper():
        return word.upper()
    if template[0].isupper():
        return word[0].upper() + word[1:]
    return word


def inflect_like(surface: str, source_lemma: str, target_lemma: str, morph: str = "") -> str:
    tag = inflection_tag(source_lemma, surface, morph)
    return match_case(surface, inflect_lemma(target_lemma, tag))


def weak_frame_lemmas(
    share: pd.DataFrame,
    *,
    max_share: float = MAX_DISTRACTOR_SHARE,
) -> dict[str, pd.DataFrame]:
    """Lemmas whose adult share in a frame is < max_share (unattested → 0)."""
    verbs = share.drop_duplicates("verb")[["verb", "n_verb"]]
    verbs = verbs.loc[verbs["verb"].astype(str).str.fullmatch(r"[a-z]+", na=False)]
    out: dict[str, pd.DataFrame] = {}
    for cons in SCHEMATIC:
        frame = share.loc[share["construction"] == cons, ["verb", "share"]]
        merged = verbs.merge(frame, on="verb", how="left")
        merged["share"] = merged["share"].fillna(0.0)
        out[cons] = merged.loc[merged["share"] < max_share].sort_values("n_verb", ascending=False)
    return out


def pick_alt_forms(
    pool: pd.DataFrame,
    source_lemma: str,
    surface: str,
    morph: str,
    rng: np.random.Generator,
    *,
    n: int = N_ALTS,
) -> list[str]:
    cand = pool.loc[pool["verb"].astype(str) != source_lemma]
    if cand.empty:
        return []
    weights = cand["n_verb"].to_numpy(dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(cand))
    order = rng.choice(len(cand), size=len(cand), replace=False, p=weights / weights.sum())
    alts: list[str] = []
    seen: set[str] = {surface.lower()}
    for i in order:
        lemma = str(cand.iloc[int(i)]["verb"])
        form = inflect_like(surface, source_lemma, lemma, morph)
        if not form or form.lower() in seen:
            continue
        alts.append(form)
        seen.add(form.lower())
        if len(alts) == n:
            break
    return alts


def utterance_ids_containing_xxx(utterance_ids: set, tokens_path: Path = TOKENS_PATH) -> set:
    """Utterance ids whose gloss includes the CHILDES unintelligible token xxx."""
    if not utterance_ids:
        return set()
    pf = pq.ParquetFile(tokens_path)
    cols = [c for c in ("utterance_id", "gloss") if c in pf.schema_arrow.names]
    bad: set = set()
    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i, columns=cols).to_pandas()
        keep = df.loc[df["utterance_id"].isin(utterance_ids)]
        if keep.empty:
            continue
        gloss = keep["gloss"].astype("string").str.strip().str.lower()
        bad.update(keep.loc[gloss.eq("xxx"), "utterance_id"].tolist())
        del df
    return bad


def strong_frame_pairs(
    hits: pd.DataFrame,
    *,
    min_share: float = MIN_ADULT_SHARE,
    min_tokens: int = MIN_TOKENS,
) -> pd.DataFrame:
    """(construction, verb) pairs where ≥ min_share of adult VAC uses are in that frame."""
    share = adult_frame_share(hits, min_tokens=min_tokens)
    return share.loc[share["share"] >= min_share, ["construction", "verb"]].drop_duplicates()


def build_test_set(
    *,
    n_per_construction: int = N_PER_CONSTRUCTION,
    seed: int = SEED,
) -> pd.DataFrame:
    added = set(corpora_added_in_2026_1())
    hits = pd.read_parquet(HITS_PATH)
    share = adult_frame_share(hits, min_tokens=MIN_SHARE_TOKENS)
    share.to_csv(CACHE / "adult_frame_share.csv", index=False)
    top = top_frame_verbs(share)
    strong = pd.DataFrame(
        [{"construction": cons, "verb": verb} for cons, verbs in top.items() for verb in verbs]
    )
    weak = weak_frame_lemmas(share)
    pool = hits.loc[hits["corpus_name"].astype(str).isin(added)].copy()
    if "speaker_role" in pool.columns:
        pool = pool.loc[~pool["speaker_role"].isin(CHILD_ROLES)]
    if "extra" in pool.columns:
        pool = pool.loc[~pool["extra"].fillna(False)]
    pool["verb"] = pool["verb"].astype(str).str.lower()
    if strong.empty:
        return strong
    pool = pool.merge(strong, on=["construction", "verb"], how="inner")
    tok = tokens_for_utterances(set(pool["utterance_id"].tolist()))
    ok_ids = {uid for uid, grp in tok.groupby("utterance_id") if eval_utterance_ok(grp)}
    pool = pool.loc[pool["utterance_id"].isin(ok_ids)]
    sample = stratified_hits(pool, n_per_construction=n_per_construction * 2, seed=seed)
    rng = np.random.default_rng(seed)
    rows = []
    counts = {c: 0 for c in SCHEMATIC}
    for rec in sample.itertuples(index=False):
        cons = str(rec.construction)
        if counts[cons] >= n_per_construction:
            continue
        grp = tok.loc[tok["utterance_id"] == rec.utterance_id]
        if grp.empty:
            continue
        sent = _join_gloss(grp)
        if not sent or contains_xxx(sent):
            continue
        lemma = str(rec.verb).lower()
        if lemma in _NON_ENGLISH_VERBS:
            continue
        span = verb_span(grp, lemma)
        if span is None:
            continue
        alts = pick_alt_forms(weak[cons], lemma, span["inflected_verb"], span["morph"], rng)
        if len(alts) < N_ALTS:
            continue
        counts[cons] += 1
        prep = rec.preposition
        prep_out = "" if prep is None or (isinstance(prep, float) and pd.isna(prep)) else str(prep).lower()
        age = rec.target_child_age
        row = {
            "item_id": f"{cons.lower()}_{counts[cons]:04d}",
            "construction_type": cons,
            "verb": lemma,
            "preposition": prep_out,
            "sentence": sent,
            "prefix": span["prefix"],
            "inflected_verb": span["inflected_verb"],
            "suffix": span["suffix"],
        }
        for col, form in zip(_ALT_COLS, alts):
            row[col] = form
        row["corpus"] = rec.corpus_name
        row["child_id"] = rec.target_child_id
        row["child_age"] = None if pd.isna(age) else round(float(age), 2)
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.reset_index(drop=True)


def write_test_set(path: Path = OUT_PATH, **kwargs) -> Path:
    df = build_test_set(**kwargs)
    for col in ("prefix", "suffix"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
