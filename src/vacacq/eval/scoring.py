"""Verb-in-frame ranking and frame-affinity scoring for BabyLM checkpoints."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from vacacq import CACHE, DATA
from vacacq.io import load_json, load_jsonl


def ranking_frames() -> list[dict]:
    return load_jsonl(DATA / "eval" / "ranking_frames.jsonl")


def affinity_items() -> list[dict]:
    return load_jsonl(DATA / "eval" / "affinity_items.jsonl")


def verb_sets() -> dict:
    return load_json("eval/verb_sets.json")


def checkpoints() -> dict:
    return load_json("checkpoints.json")


def expand_ranking_items() -> pd.DataFrame:
    """Cartesian product of frames × occupant/distractor verbs."""
    rows = []
    sets = verb_sets()
    for frame in ranking_frames():
        cons = frame["construction"]
        verbs = sets.get(cons, {})
        for kind, lst in (("occupant", verbs.get("occupants", [])), ("distractor", verbs.get("distractors", []))):
            for verb in lst:
                sent = frame["template"].format(verb=verb)
                rows.append(
                    {
                        "item_id": frame["id"],
                        "construction": cons,
                        "preposition": frame.get("preposition"),
                        "verb": verb,
                        "kind": kind,
                        "sentence": sent,
                    }
                )
    return pd.DataFrame(rows)


def _device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_lm(model_id: str, architecture: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.unk_token
    if architecture == "gpt2":
        model = AutoModelForCausalLM.from_pretrained(model_id)
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_id)
    model.eval()
    model.to(_device())
    return model, tok


def causal_mean_logprob(model, tokenizer, text: str) -> float:
    """Length-normalized token logprob (harmony)."""
    import torch
    import torch.nn.functional as F

    enc = tokenizer(text, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    input_ids = enc["input_ids"]
    with torch.no_grad():
        logits = model(**enc).logits
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    target = input_ids[:, 1:]
    token_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return float(token_lp.mean().item())


def causal_verb_span_logprob(model, tokenizer, sentence: str, verb: str) -> float:
    """Mean logprob of the verb-token span given the prefix."""
    import torch
    import torch.nn.functional as F

    lower = sentence.lower()
    idx = lower.find(verb.lower())
    if idx < 0:
        return causal_mean_logprob(model, tokenizer, sentence)
    prefix = sentence[:idx]
    enc_full = tokenizer(sentence, return_tensors="pt")
    enc_pref = tokenizer(prefix, return_tensors="pt") if prefix.strip() else None
    start = enc_pref["input_ids"].shape[1] if enc_pref is not None else 1
    enc_full = {k: v.to(model.device) for k, v in enc_full.items()}
    ids = enc_full["input_ids"]
    with torch.no_grad():
        logits = model(**enc_full).logits
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    target = ids[:, 1:]
    # verb tokens correspond to positions start-1 ... in the shifted target
    lo = max(start - 1, 0)
    hi = target.shape[1]
    span = logp[:, lo:hi, :].gather(-1, target[:, lo:hi].unsqueeze(-1)).squeeze(-1)
    if span.numel() == 0:
        return causal_mean_logprob(model, tokenizer, sentence)
    return float(span.mean().item())


def masked_verb_logprob(model, tokenizer, sentence: str, verb: str) -> float:
    """Mean logprob of gold verb pieces with the verb span masked."""
    import torch
    import torch.nn.functional as F

    enc = tokenizer(sentence, return_tensors="pt")
    ids = enc["input_ids"][0]
    verb_ids = tokenizer(verb, add_special_tokens=False)["input_ids"]
    if not verb_ids:
        return float("nan")
    # find first occurrence of verb_ids in ids
    start = None
    seq = ids.tolist()
    for i in range(len(seq) - len(verb_ids) + 1):
        if seq[i : i + len(verb_ids)] == verb_ids:
            start = i
            break
    if start is None:
        # fall back: mask tokenizer.mask_token inserted at verb string location
        masked = sentence
        idx = sentence.lower().find(verb.lower())
        if idx >= 0:
            masked = sentence[:idx] + tokenizer.mask_token + sentence[idx + len(verb) :]
        enc = tokenizer(masked, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        mask_pos = (enc["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
        if mask_pos[1].numel() == 0:
            return float("nan")
        with torch.no_grad():
            logits = model(**enc).logits
        pos = int(mask_pos[1][0])
        logp = F.log_softmax(logits[0, pos], dim=-1)
        return float(logp[verb_ids[0]].item())

    masked = ids.clone()
    for j in range(len(verb_ids)):
        masked[start + j] = tokenizer.mask_token_id
    batch = {
        "input_ids": masked.unsqueeze(0).to(model.device),
        "attention_mask": enc["attention_mask"].to(model.device),
    }
    with torch.no_grad():
        logits = model(**batch).logits
    lps = []
    for j, vid in enumerate(verb_ids):
        logp = F.log_softmax(logits[0, start + j], dim=-1)
        lps.append(float(logp[vid].item()))
    return float(np.mean(lps))


def spearman_with_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    if len(x) < 3:
        return {"rho": float("nan"), "p": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r, _ = stats.spearmanr(x[idx], y[idx])
        if r == r:
            boots.append(r)
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5])
    else:
        lo = hi = float("nan")
    return {
        "rho": float(rho) if rho == rho else float("nan"),
        "p": float(p) if p == p else float("nan"),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def score_ranking(model, tokenizer, architecture: str, items: pd.DataFrame) -> pd.DataFrame:
    scores = []
    for rec in items.to_dict(orient="records"):
        if architecture == "gpt2":
            harmony = causal_mean_logprob(model, tokenizer, rec["sentence"])
            span = causal_verb_span_logprob(model, tokenizer, rec["sentence"], rec["verb"])
            score = harmony
        else:
            span = masked_verb_logprob(model, tokenizer, rec["sentence"], rec["verb"])
            score = span
            harmony = span
        rec = dict(rec)
        rec["harmony"] = harmony
        rec["verb_span_logprob"] = span
        rec["score"] = score
        scores.append(rec)
    return pd.DataFrame(scores)


def score_affinity(model, tokenizer, architecture: str, items: list[dict] | None = None) -> pd.DataFrame:
    rows = []
    for item in items or affinity_items():
        if architecture == "gpt2":
            g = causal_mean_logprob(model, tokenizer, item["grammatical"])
            c = causal_mean_logprob(model, tokenizer, item["control"])
        else:
            g = masked_verb_logprob(model, tokenizer, item["grammatical"], item["verb"])
            c = masked_verb_logprob(model, tokenizer, item["control"], item["verb"])
        rows.append(
            {
                "item_id": item["id"],
                "construction": item["construction"],
                "verb": item["verb"],
                "grammatical_score": g,
                "control_score": c,
                "affinity": g - c,
            }
        )
    return pd.DataFrame(rows)


def ranking_correlations(scores: pd.DataFrame, occupancy: pd.DataFrame) -> pd.DataFrame:
    """Spearman ρ of model ranking vs training Verb–VAC frequency and ΔP_cw."""
    rows = []
    occ = occupancy.copy()
    occ["verb"] = occ["verb"].str.lower()
    for cons, grp in scores.groupby("construction"):
        ranks = grp.groupby("verb")["score"].mean().reset_index()
        sub = occ.loc[occ["construction"] == cons] if "construction" in occ.columns else occ
        if "vac" in sub.columns:
            # occupancy may be vac-specific; take mean over matching frames
            sub = sub.groupby("verb", as_index=False).agg(vac_freq=("vac_freq", "sum"), delta_p_cw=("delta_p_cw", "mean"))
        merged = ranks.merge(sub, on="verb", how="inner")
        if merged.empty:
            continue
        for col, name in (("vac_freq", "frequency"), ("delta_p_cw", "delta_p")):
            if col not in merged.columns or merged[col].isna().all():
                continue
            stats_d = spearman_with_ci(merged["score"].to_numpy(), merged[col].to_numpy())
            stats_d.update({"construction": cons, "predictor": name, "n": int(len(merged))})
            rows.append(stats_d)
    return pd.DataFrame(rows)


def acquired(ranking: pd.DataFrame, affinity: pd.DataFrame) -> pd.DataFrame:
    """A VAC is acquired if ranking ρ vs occupancy > 0 (CI) and mean affinity > 0."""
    rows = []
    aff_mean = affinity.groupby("construction")["affinity"].agg(["mean", "sem", "count"]).reset_index()
    for cons, grp in ranking.groupby("construction"):
        freq = grp.loc[grp["predictor"] == "frequency"]
        if freq.empty:
            continue
        row = freq.iloc[0]
        rho_pos = row["ci_low"] > 0 if row["ci_low"] == row["ci_low"] else row["rho"] > 0 and row["p"] < 0.05
        a = aff_mean.loc[aff_mean["construction"] == cons]
        mean_aff = float(a["mean"].iloc[0]) if not a.empty else float("nan")
        aff_pos = mean_aff > 0
        rows.append(
            {
                "construction": cons,
                "rho_frequency": row["rho"],
                "rho_ci_low": row["ci_low"],
                "mean_affinity": mean_aff,
                "acquired": bool(rho_pos and aff_pos),
            }
        )
    return pd.DataFrame(rows)


def score_track(
    track: str,
    epochs: list[int] | None = None,
    occupancy: pd.DataFrame | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, pd.DataFrame]:
    spec = checkpoints()["tracks"][track]
    architecture = spec["architecture"]
    epochs = epochs or spec["epochs"]
    items = expand_ranking_items()
    rank_frames = []
    aff_frames = []
    if dry_run:
        CACHE.mkdir(parents=True, exist_ok=True)
        items.to_csv(CACHE / "eval_ranking_items.csv", index=False)
        return {"items": items, "ranking": pd.DataFrame(), "affinity": pd.DataFrame()}

    for epoch in epochs:
        model_id = spec["pattern"].format(epoch=epoch)
        model, tok = load_lm(model_id, architecture)
        ranked = score_ranking(model, tok, architecture, items)
        ranked["epoch"] = epoch
        ranked["track"] = track
        ranked["model_id"] = model_id
        aff = score_affinity(model, tok, architecture)
        aff["epoch"] = epoch
        aff["track"] = track
        aff["model_id"] = model_id
        rank_frames.append(ranked)
        aff_frames.append(aff)
        del model
    ranking = pd.concat(rank_frames, ignore_index=True)
    affinity = pd.concat(aff_frames, ignore_index=True)
    corrs = []
    acquired_rows = []
    if occupancy is not None and not occupancy.empty:
        for epoch, grp in ranking.groupby("epoch"):
            c = ranking_correlations(grp, occupancy)
            c["epoch"] = epoch
            c["track"] = track
            corrs.append(c)
            a = affinity.loc[affinity["epoch"] == epoch]
            acq = acquired(c, a)
            acq["epoch"] = epoch
            acq["track"] = track
            acquired_rows.append(acq)
    out = {
        "ranking": ranking,
        "affinity": affinity,
        "correlations": pd.concat(corrs, ignore_index=True) if corrs else pd.DataFrame(),
        "acquired": pd.concat(acquired_rows, ignore_index=True) if acquired_rows else pd.DataFrame(),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_csv(CACHE / f"babylm_{track}_{name}.csv", index=False)
    return out
