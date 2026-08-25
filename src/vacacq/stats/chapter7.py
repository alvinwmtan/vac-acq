"""Chapter 7 statistics: Zipf, selectivity, ΔP_cw, WordNet betweenness, parent→child, MLU-w."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

from vacacq import CACHE
from vacacq.childes.extract import filter_hits_for_stratum
from vacacq.childes.strata import assign_strata

SCHEMATIC = {"VL", "VOL", "VOO", "VO"}


def explode_strata(hits: pd.DataFrame) -> pd.DataFrame:
    """Duplicate parent hits so Parent Young also counts in Parent All."""
    if hits.empty:
        return hits
    rows = []
    for rec in hits.to_dict(orient="records"):
        age = rec.get("target_child_age")
        role = rec.get("speaker_role")
        assigned = rec.get("stratum")
        strata = assign_strata(role, age) if age is not None and role else []
        if not strata and assigned:
            strata = [assigned]
        if not strata:
            strata = [None]
        for s in strata:
            row = dict(rec)
            row["stratum"] = s
            rows.append(row)
    out = pd.DataFrame(rows)
    parts = []
    for stratum, grp in out.groupby("stratum", dropna=False):
        parts.append(filter_hits_for_stratum(grp, stratum) if isinstance(stratum, str) else grp)
    return pd.concat(parts, ignore_index=True) if parts else out


def construction_key(row) -> str:
    prep = row.get("preposition")
    cons = row.get("construction")
    if prep and cons in {"VL", "VOL"}:
        return f"{cons}-{prep}"
    return str(cons)


def occupancy_table(hits: pd.DataFrame, verb_totals: pd.DataFrame | None = None) -> pd.DataFrame:
    """Verb–VAC frequency plus ΔP_cw. verb_totals: stratum, verb, n_verb_tokens."""
    if hits.empty:
        return pd.DataFrame()
    work = hits.loc[~hits["extra"].fillna(False)].copy() if "extra" in hits.columns else hits.copy()
    work["vac"] = work.apply(construction_key, axis=1)
    grouped = (
        work.groupby(["stratum", "vac", "construction", "preposition", "verb", "parse_source"], dropna=False)
        .size()
        .reset_index(name="vac_freq")
    )
    if verb_totals is None or verb_totals.empty:
        grouped["delta_p_cw"] = np.nan
        grouped["p_verb_vac"] = np.nan
        grouped["p_verb_not_vac"] = np.nan
        return grouped

    vt = verb_totals.rename(columns={"n_verb_tokens": "n_verb"})
    merged = grouped.merge(vt, on=["stratum", "verb"], how="left")
    vac_n = work.groupby(["stratum", "vac"]).size().rename("n_vac")
    merged = merged.merge(vac_n, on=["stratum", "vac"], how="left")
    stratum_n = vt.groupby("stratum")["n_verb"].sum().rename("n_stratum")
    merged = merged.merge(stratum_n, on="stratum", how="left")
    merged["n_verb"] = merged["n_verb"].fillna(merged["vac_freq"])
    merged["p_verb_vac"] = merged["vac_freq"] / merged["n_vac"].clip(lower=1)
    not_vac = (merged["n_stratum"] - merged["n_vac"]).clip(lower=1)
    verb_outside = (merged["n_verb"] - merged["vac_freq"]).clip(lower=0)
    merged["p_verb_not_vac"] = verb_outside / not_vac
    merged["delta_p_cw"] = merged["p_verb_vac"] - merged["p_verb_not_vac"]
    return merged


def zipf_fit(frequencies: pd.Series) -> dict:
    """log10 frequency vs log10 rank. Returns R² and slope γ."""
    freqs = np.asarray(frequencies, dtype=float)
    freqs = freqs[freqs > 0]
    freqs = np.sort(freqs)[::-1]
    if freqs.size < 2:
        return {"r2": float("nan"), "gamma": float("nan"), "n_types": int(freqs.size)}
    rank = np.arange(1, freqs.size + 1, dtype=float)
    x = np.log10(rank)
    y = np.log10(freqs)
    slope, intercept, r, _, _ = stats.linregress(x, y)
    return {
        "r2": float(r**2),
        "gamma": float(slope),
        "intercept": float(intercept),
        "n_types": int(freqs.size),
    }


def selectivity_one_minus_tau2(vac_freq: pd.Series, corpus_freq: pd.Series) -> dict:
    """1 − τ² of VAC-rank vs corpus-rank, including verbs that never occur in the VAC."""
    df = pd.DataFrame({"vac": vac_freq, "corpus": corpus_freq}).fillna(0)
    if len(df) < 3:
        return {"tau": float("nan"), "one_minus_tau2": float("nan"), "n": int(len(df))}
    tau, _ = stats.kendalltau(df["vac"].rank(ascending=False), df["corpus"].rank(ascending=False))
    tau = float(tau) if tau == tau else float("nan")
    return {"tau": tau, "one_minus_tau2": 1 - tau**2 if tau == tau else float("nan"), "n": int(len(df))}


def _ensure_wordnet():
    import nltk
    from nltk.corpus import wordnet as wn

    try:
        wn.synsets("go", pos=wn.VERB)
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    return wn


def wordnet_betweenness(verbs: list[str], sim_threshold: float = 0.5) -> pd.DataFrame:
    """Frequency-blind betweenness on WordNet path-similarity graph (verbs, max synset pair)."""
    import networkx as nx

    wn = _ensure_wordnet()
    unique = sorted({v.lower() for v in verbs if v})
    synsets = {}
    for v in unique:
        synsets[v] = wn.synsets(v, pos=wn.VERB)
    graph = nx.Graph()
    graph.add_nodes_from(unique)
    for i, a in enumerate(unique):
        sa = synsets[a]
        if not sa:
            continue
        for b in unique[i + 1 :]:
            sb = synsets[b]
            if not sb:
                continue
            best = 0.0
            for x in sa:
                for y in sb:
                    sim = x.path_similarity(y)
                    if sim and sim > best:
                        best = sim
            if best > sim_threshold:
                graph.add_edge(a, b, weight=best)
    bc = nx.betweenness_centrality(graph, normalized=True, weight=None)
    return pd.DataFrame({"verb": list(bc), "betweenness": list(bc.values())})


def parent_child_models(occ: pd.DataFrame) -> pd.DataFrame:
    """Pearson r and lm(log10 child ~ log10 parent freq + ΔP + log10 BC) per VAC pair."""
    from sklearn.linear_model import LinearRegression

    pairs = [("Child I", "Parent Young"), ("Child II", "Parent All")]
    rows = []
    for child, parent in pairs:
        for vac, grp in occ.groupby("vac"):
            c = grp.loc[grp["stratum"] == child, ["verb", "vac_freq", "delta_p_cw", "betweenness"]]
            p = grp.loc[grp["stratum"] == parent, ["verb", "vac_freq", "delta_p_cw", "betweenness"]]
            merged = p.merge(c, on="verb", suffixes=("_parent", "_child"))
            if merged.empty:
                continue
            x = np.log10(merged["vac_freq_parent"].clip(lower=1))
            y = np.log10(merged["vac_freq_child"].clip(lower=1))
            r, pval = stats.pearsonr(x, y) if len(merged) >= 3 else (float("nan"), float("nan"))
            row = {
                "child_stratum": child,
                "parent_stratum": parent,
                "vac": vac,
                "n": int(len(merged)),
                "pearson_r": float(r) if r == r else float("nan"),
                "pearson_p": float(pval) if pval == pval else float("nan"),
            }
            feats = merged[["vac_freq_parent", "delta_p_cw_parent", "betweenness_parent"]].copy()
            feats["log10_parent"] = np.log10(feats["vac_freq_parent"].clip(lower=1))
            feats["log10_bc"] = np.log10(feats["betweenness_parent"].clip(lower=1e-12))
            X = feats[["log10_parent", "delta_p_cw_parent", "log10_bc"]].fillna(0)
            if len(merged) >= 4 and X.var().min() > 0:
                model = LinearRegression().fit(X, y)
                row["lm_r2"] = float(model.score(X, y))
                row["coef_log10_parent"] = float(model.coef_[0])
                row["coef_delta_p"] = float(model.coef_[1])
                row["coef_log10_bc"] = float(model.coef_[2])
            else:
                row["lm_r2"] = float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def cumulative_vs_mlu(
    hits: pd.DataFrame,
    mlu: pd.DataFrame,
) -> pd.DataFrame:
    """Cumulative verb-type frequency vs speaker MLU-w in Child I."""
    child = hits.loc[hits["stratum"] == "Child I"].copy()
    if child.empty or mlu.empty:
        return pd.DataFrame()
    key = "utterance_id"
    if key not in mlu.columns:
        return pd.DataFrame()
    child = child.merge(mlu[[key, "mlu_w"]], on=key, how="left")
    child = child.sort_values("mlu_w")
    seen: dict[tuple, set[str]] = defaultdict(set)
    rows = []
    for _, row in child.iterrows():
        vac = construction_key(row)
        seen[vac].add(row["verb"])
        rows.append({"vac": vac, "mlu_w": row["mlu_w"], "n_types": len(seen[vac])})
    return pd.DataFrame(rows)


def mlu_w_by_utterance(tokens: pd.DataFrame) -> pd.DataFrame:
    """Word MLU proxy: mean num_tokens per speaker, attached to each utterance."""
    if tokens.empty:
        return pd.DataFrame()
    utt = tokens.drop_duplicates("utterance_id")
    if "num_tokens" in utt.columns:
        out = utt[["utterance_id"]].copy()
        out["mlu_w"] = utt["num_tokens"].astype(float)
        return out
    counts = tokens.groupby("utterance_id").size().rename("mlu_w").reset_index()
    counts = counts.rename(columns={"utterance_id": "utterance_id"})
    return counts


def lead_verbs(occ: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    if occ.empty:
        return occ
    occ = occ.sort_values(["stratum", "vac", "vac_freq"], ascending=[True, True, False])
    return occ.groupby(["stratum", "vac"], dropna=False).head(k)


def run_chapter7_stats(
    hits: pd.DataFrame,
    verb_totals: pd.DataFrame | None = None,
    tokens: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    exploded = explode_strata(hits)
    occ = occupancy_table(exploded, verb_totals)
    zipf_rows = []
    sel_rows = []
    for (stratum, vac, source), grp in occ.groupby(["stratum", "vac", "parse_source"], dropna=False):
        z = zipf_fit(grp["vac_freq"])
        z.update({"stratum": stratum, "vac": vac, "parse_source": source})
        zipf_rows.append(z)
        if verb_totals is not None and not verb_totals.empty:
            vt = verb_totals.loc[verb_totals["stratum"] == stratum]
            vac_f = grp.set_index("verb")["vac_freq"]
            corp = vt.set_index("verb")["n_verb_tokens"]
            aligned = pd.DataFrame({"vac": vac_f, "corpus": corp}).fillna(0)
            sel = selectivity_one_minus_tau2(aligned["vac"], aligned["corpus"])
            sel.update({"stratum": stratum, "vac": vac, "parse_source": source})
            sel_rows.append(sel)
    zipf_df = pd.DataFrame(zipf_rows)
    verbs = occ["verb"].dropna().unique().tolist() if not occ.empty else []
    try:
        bc = wordnet_betweenness(verbs) if verbs else pd.DataFrame(columns=["verb", "betweenness"])
    except Exception as exc:  # noqa: BLE001
        bc = pd.DataFrame(columns=["verb", "betweenness"])
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / "wordnet_error.txt").write_text(str(exc), encoding="utf-8")
    if not occ.empty and not bc.empty:
        occ = occ.merge(bc, on="verb", how="left")
    elif not occ.empty:
        occ["betweenness"] = np.nan
    models = parent_child_models(occ) if not occ.empty else pd.DataFrame()
    mlu_curve = pd.DataFrame()
    if tokens is not None:
        mlu_curve = cumulative_vs_mlu(exploded, mlu_w_by_utterance(tokens))
    leads = lead_verbs(occ)
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {
        "occupancy": occ,
        "zipf": zipf_df,
        "selectivity": pd.DataFrame(sel_rows),
        "parent_child": models,
        "leads": leads,
        "cumulative_mlu": mlu_curve,
    }
    for name, df in out.items():
        path = CACHE / f"stats_{name}.csv"
        df.to_csv(path, index=False)
    (CACHE / "stats_status.json").write_text(
        json.dumps({"tables": list(out), "n_hits": int(len(exploded))}, indent=2),
        encoding="utf-8",
    )
    return out


def verb_token_totals(tokens: pd.DataFrame) -> pd.DataFrame:
    """N = all verb tokens per stratum (ΔP denominator)."""
    from vacacq.childes.extract import VERB_POS, _norm_pos

    if tokens.empty:
        return pd.DataFrame(columns=["stratum", "verb", "n_verb_tokens"])
    verbs = tokens.loc[tokens["part_of_speech"].map(_norm_pos).isin(VERB_POS)].copy()
    verbs["verb"] = verbs["stem"].fillna(verbs["gloss"]).astype(str).str.lower()
    rows = []
    for _, row in verbs.iterrows():
        for s in assign_strata(row.get("speaker_role"), row.get("target_child_age")):
            rows.append({"stratum": s, "verb": row["verb"]})
    if not rows:
        return pd.DataFrame(columns=["stratum", "verb", "n_verb_tokens"])
    return pd.DataFrame(rows).groupby(["stratum", "verb"]).size().reset_index(name="n_verb_tokens")
