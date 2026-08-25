"""UD-native VAC extractor (primary) plus S7.4 linear-window sensitivity."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from vacacq.io import load_json

VERB_POS = {"verb"}
ADP_POS = {"adp"}
NP_POS = {"noun", "propn", "pron", "det", "adj", "num"}
OBJ_RELS = {"obj"}
IOBJ_RELS = {"iobj"}
OBL_RELS = {"obl", "obl:lmod"}
PRT_RELS = {"compound:prt", "prt", "compound:prt:lvc"}
PASSIVE_RELS = {"nsubj:pass", "aux:pass"}
QUESTION_TYPES = {"question", "?", "q"}
LATE_PREPS = {"near", "onto"}


def _norm_pos(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip().lower()


def _norm_rel(val) -> str:
    """CHAT %gra uses ROOT/OBJ/COMPOUND-PRT; UD uses root/obj/compound:prt."""
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip().lower().replace("_", "-").replace("-", ":")


def _normalize_tags(tokens: pd.DataFrame) -> pd.DataFrame:
    df = tokens.copy()
    pos = df["part_of_speech"].astype("string").str.strip().str.lower().fillna("")
    df["part_of_speech"] = pos
    if "gra_relation" in df.columns:
        rel = df["gra_relation"].astype("string").str.strip().str.lower().fillna("")
        df["gra_relation"] = rel.str.replace("_", "-", regex=False).str.replace("-", ":", regex=False)
    return df


@lru_cache(maxsize=1)
def prepositions() -> dict:
    return load_json("s7_3_prepositions.json")


@lru_cache(maxsize=1)
def ditransitive_verbs() -> frozenset[str]:
    return frozenset(v.lower() for v in load_json("s7_5_ditransitive.json")["verbs"])


def locative_preps(construction: str, stratum: str | None = None) -> set[str]:
    spec = prepositions()
    if construction == "VL":
        preps = set(spec["vl_and_vol"]) | set(spec["vl_only"])
    elif construction == "VOL":
        preps = set(spec["vl_and_vol"]) | set(spec["vol_only"])
    else:
        preps = set()
    if stratum in {"Child I", "Parent Young"}:
        preps -= set(spec["parent_all_child_ii_only"])
    return {p.lower() for p in preps}


def _lemma(row: pd.Series) -> str:
    stem = row.get("stem")
    gloss = row.get("gloss")
    val = stem if isinstance(stem, str) and stem.strip() else gloss
    return (str(val) if val is not None else "").lower().strip()


def _key(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return val


def _node_id(row: pd.Series):
    if "gra_index" in row.index and pd.notna(row.get("gra_index")):
        return _key(row["gra_index"])
    return _key(row.get("id"))


def _children(tokens: pd.DataFrame, node_id) -> pd.DataFrame:
    if "gra_head" not in tokens.columns or node_id is None:
        return tokens.iloc[0:0]
    heads = tokens["gra_head"].map(_key)
    return tokens.loc[heads == node_id]


def _is_question(tokens: pd.DataFrame) -> bool:
    for col in ("utterance_type", "type"):
        if col in tokens.columns:
            utt_type = tokens[col].iloc[0]
            if isinstance(utt_type, str) and utt_type.strip().lower() in QUESTION_TYPES:
                return True
    if "gloss" in tokens.columns:
        glosses = tokens["gloss"].astype(str).tolist()
        if any(g.strip() == "?" for g in glosses):
            return True
    return False


def _is_passive(tokens: pd.DataFrame, node_id) -> bool:
    rels = _children(tokens, node_id)["gra_relation"]
    if rels.empty:
        return False
    return rels.astype(str).isin(PASSIVE_RELS).any()


def _prep_lemma_from_obl(tokens: pd.DataFrame, obl_row: pd.Series) -> str | None:
    """ADP case-marker of an oblique, or the obl token itself if it is ADP."""
    if str(obl_row.get("part_of_speech") or "").lower() == "adp":
        return _lemma(obl_row)
    kids = _children(tokens, _node_id(obl_row))
    adp = kids.loc[kids["part_of_speech"].isin(ADP_POS)] if not kids.empty else kids
    if adp.empty and not kids.empty and "gra_relation" in kids.columns:
        adp = kids.loc[kids["gra_relation"].astype(str).isin({"case", "case:loc"})]
    if adp.empty:
        return None
    return _lemma(adp.iloc[0])


def _particle_prep(kids: pd.DataFrame, allowed: set[str]) -> str | None:
    if kids.empty:
        return None
    prt = kids.loc[kids["gra_relation"].astype(str).isin(PRT_RELS)]
    for _, row in prt.iterrows():
        lemma = _lemma(row)
        if lemma in allowed:
            return lemma
    return None


def _verb_rows(tokens: pd.DataFrame) -> pd.DataFrame:
    verbs = tokens.loc[tokens["part_of_speech"].isin(VERB_POS)]
    if "suffix" in tokens.columns:
        ger = tokens["suffix"].astype(str).str.contains(r"Ger|VerbForm=Ger", na=False)
        verbs = pd.concat([verbs, tokens.loc[ger]])
    if verbs.empty:
        return verbs
    subset = "id" if "id" in verbs.columns else ("gra_index" if "gra_index" in verbs.columns else None)
    if subset:
        verbs = verbs.drop_duplicates(subset=[subset])
    return verbs


@dataclass
class VacHit:
    utterance_id: object
    verb: str
    construction: str
    preposition: str | None
    parse_source: str
    extractor: str
    extra: bool = False


def extract_utterance_ud(
    tokens: pd.DataFrame,
    stratum: str | None = None,
    *,
    already_normalized: bool = False,
    extras: bool = True,
) -> list[VacHit]:
    """Primary UD-native extractor for one utterance (already ordered)."""
    hits: list[VacHit] = []
    if tokens.empty or _is_question(tokens):
        return hits
    parse_source = tokens["parse_source"].iloc[0] if "parse_source" in tokens.columns else "childesdb"
    utt_id = tokens["utterance_id"].iloc[0]
    if not already_normalized:
        tokens = _normalize_tags(tokens)
    ditr = ditransitive_verbs()
    vl_preps = locative_preps("VL", stratum)
    vol_preps = locative_preps("VOL", stratum)
    loc_preps = vl_preps | vol_preps

    for _, verb in _verb_rows(tokens).iterrows():
        nid = _node_id(verb)
        if _is_passive(tokens, nid):
            continue
        kids = _children(tokens, nid)
        rels = set(kids["gra_relation"].astype(str)) if not kids.empty else set()
        n_obj = int(kids["gra_relation"].astype(str).isin(OBJ_RELS).sum()) if not kids.empty else 0
        has_obj = n_obj >= 1
        has_iobj = bool(rels & IOBJ_RELS) or n_obj >= 2
        loc_prep = None
        if not kids.empty:
            obls = kids.loc[kids["gra_relation"].astype(str).isin(OBL_RELS)]
            for _, obl in obls.iterrows():
                p = _prep_lemma_from_obl(tokens, obl)
                if p and p in loc_preps:
                    loc_prep = p
                    break
            if loc_prep is None:
                loc_prep = _particle_prep(kids, loc_preps)
        vlemma = _lemma(verb)

        if loc_prep and not has_obj and loc_prep in vl_preps:
            hits.append(VacHit(utt_id, vlemma, "VL", loc_prep, parse_source, "ud"))
        if loc_prep and has_obj and loc_prep in vol_preps:
            hits.append(VacHit(utt_id, vlemma, "VOL", loc_prep, parse_source, "ud"))
        if has_obj and has_iobj and vlemma in ditr:
            hits.append(VacHit(utt_id, vlemma, "VOO", None, parse_source, "ud"))
        if has_obj and not has_iobj and not (loc_prep and loc_prep in vol_preps):
            hits.append(VacHit(utt_id, vlemma, "VO", None, parse_source, "ud"))

        if extras:
            if "xcomp" in rels and has_obj:
                hits.append(VacHit(utt_id, vlemma, "resultative", None, parse_source, "ud", extra=True))
            if not has_obj and not has_iobj and loc_prep is None:
                hits.append(VacHit(utt_id, vlemma, "intransitive", None, parse_source, "ud", extra=True))
            if has_obj and loc_prep == "to" and vlemma in ditr:
                hits.append(VacHit(utt_id, vlemma, "prep_dative", "to", parse_source, "ud", extra=True))

    if extras and "gra_relation" in tokens.columns:
        cops = tokens.loc[tokens["gra_relation"].astype(str) == "cop"]
        for _, cop in cops.iterrows():
            hits.append(VacHit(utt_id, _lemma(cop), "copular", None, parse_source, "ud", extra=True))
    return hits


def extract_utterance_s74(tokens: pd.DataFrame, stratum: str | None = None) -> list[VacHit]:
    """S7.4 linear windows on UD POS (sensitivity check)."""
    hits: list[VacHit] = []
    if tokens.empty or _is_question(tokens):
        return hits
    tokens = tokens.sort_values("token_order").reset_index(drop=True)
    parse_source = tokens["parse_source"].iloc[0] if "parse_source" in tokens.columns else "childesdb"
    utt_id = tokens["utterance_id"].iloc[0]
    tokens = _normalize_tags(tokens)
    ditr = ditransitive_verbs()
    vl_preps = locative_preps("VL", stratum)
    vol_preps = locative_preps("VOL", stratum)
    n = len(tokens)

    def pos(i: int) -> str:
        return str(tokens.iloc[i].get("part_of_speech") or "")

    def lem(i: int) -> str:
        return _lemma(tokens.iloc[i])

    verbish = set(VERB_POS)
    for i, row in tokens.iterrows():
        suffix = str(row.get("suffix") or "")
        if pos(i) not in verbish and "Ger" not in suffix:
            continue
        nid = _node_id(row)
        if _is_passive(tokens, nid):
            continue
        vlemma = lem(i)
        kids = _children(tokens, nid)
        obj_ids = set()
        iobj_ids = set()
        if not kids.empty:
            obj_ids = set(kids.loc[kids["gra_relation"].astype(str).isin(OBJ_RELS), "id"])
            iobj_ids = set(kids.loc[kids["gra_relation"].astype(str).isin(IOBJ_RELS), "id"])

        if i + 1 < n and pos(i + 1) in ADP_POS and lem(i + 1) in vl_preps and not obj_ids:
            hits.append(VacHit(utt_id, vlemma, "VL", lem(i + 1), parse_source, "s74"))

        for gap in range(1, 4):
            j = i + 1 + gap
            if j >= n:
                break
            if pos(j) in ADP_POS and lem(j) in vol_preps:
                if not all(pos(k) in NP_POS for k in range(i + 1, j)):
                    continue
                mid_ids = set(tokens.iloc[i + 1 : j]["id"])
                if obj_ids and mid_ids.intersection(obj_ids):
                    hits.append(VacHit(utt_id, vlemma, "VOL", lem(j), parse_source, "s74"))
                    break

        if vlemma in ditr:
            k = i + 1
            chunks = []
            while k < n and len(chunks) < 2:
                if pos(k) not in NP_POS:
                    break
                start = k
                while k < n and pos(k) in NP_POS:
                    k += 1
                chunks.append((start, k - 1))
            if len(chunks) == 2:
                ids0 = set(tokens.iloc[chunks[0][0] : chunks[0][1] + 1]["id"])
                ids1 = set(tokens.iloc[chunks[1][0] : chunks[1][1] + 1]["id"])
                if (ids0 & obj_ids or ids0 & iobj_ids) and (ids1 & obj_ids or ids1 & iobj_ids):
                    hits.append(VacHit(utt_id, vlemma, "VOO", None, parse_source, "s74"))

        if i + 1 < n and pos(i + 1) in NP_POS:
            k = i + 1
            while k < n and pos(k) in NP_POS:
                k += 1
            np_ids = set(tokens.iloc[i + 1 : k]["id"])
            if np_ids & obj_ids and not iobj_ids:
                hits.append(VacHit(utt_id, vlemma, "VO", None, parse_source, "s74"))
    return hits


def hits_to_frame(hits: list[VacHit], meta: dict | None = None) -> pd.DataFrame:
    rows = []
    for h in hits:
        row = {
            "utterance_id": h.utterance_id,
            "verb": h.verb,
            "construction": h.construction,
            "preposition": h.preposition,
            "parse_source": h.parse_source,
            "extractor": h.extractor,
            "extra": h.extra,
        }
        if meta:
            row.update(meta)
        rows.append(row)
    return pd.DataFrame(rows)


META_COLS = (
    "corpus_name",
    "collection_name",
    "speaker_role",
    "stratum",
    "in_s7_2",
    "target_child_age",
    "target_child_id",
    "target_child_name",
    "filename",
    "num_tokens",
)


def extract_vacs(
    tokens: pd.DataFrame,
    *,
    extractor: str = "ud",
    stratum_col: str = "stratum",  # kept for callers; prep cuts happen in stats
    extras: bool = True,
) -> pd.DataFrame:
    """Extract VAC instances from a token table with UD columns and utterance_id."""
    if tokens.empty:
        return hits_to_frame([])
    tokens = _normalize_tags(tokens)
    verb_utts = set(tokens.loc[tokens["part_of_speech"].isin(VERB_POS), "utterance_id"].tolist())
    if verb_utts:
        tokens = tokens.loc[tokens["utterance_id"].isin(verb_utts)]
    frames = []
    for utt_id, grp in tokens.groupby("utterance_id", sort=False):
        grp = grp.sort_values("token_order") if "token_order" in grp.columns else grp
        if extractor == "ud":
            hits = extract_utterance_ud(grp, stratum=None, already_normalized=True, extras=extras)
        elif extractor == "s74":
            hits = extract_utterance_s74(grp, stratum=None)
        else:
            raise ValueError(extractor)
        meta = {"utterance_id": utt_id}
        for col in META_COLS:
            if col in grp.columns:
                meta[col] = grp[col].iloc[0]
        frames.append(hits_to_frame(hits, meta))
    if not frames:
        return hits_to_frame([])
    return pd.concat(frames, ignore_index=True)


def filter_hits_for_stratum(hits: pd.DataFrame, stratum: str) -> pd.DataFrame:
    """Drop near/onto from Child I and Parent Young (S7.3)."""
    if hits.empty:
        return hits
    if stratum not in {"Child I", "Parent Young"}:
        return hits
    prep = hits["preposition"].fillna("").str.lower()
    return hits.loc[~prep.isin(LATE_PREPS)].copy()
