"""Acquisition order: child age-of-acquisition and model epoch curves."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from vacacq import CACHE
from vacacq.childes.strata import AGE_MAX, AGE_MIN, CHILD_ROLES, PARENT_ROLES

SCHEMATIC = ("VL", "VOL", "VO", "VOO")
PREP_VACS = {"VL", "VOL"}
DEFAULT_BIN_MONTHS = 3.0
MIN_CHILDREN = 5
MIN_CHILDREN_VOO = 3
MIN_TOKENS = 10


def item_key(row) -> str:
    cons = str(row.get("construction") or "")
    prep = row.get("preposition")
    verb = str(row.get("verb") or "").lower()
    if cons in PREP_VACS and isinstance(prep, str) and prep:
        return f"{cons}-{prep.lower()}|{verb}"
    return f"{cons}|{verb}"


def vac_label(row) -> str:
    cons = str(row.get("construction") or "")
    prep = row.get("preposition")
    if cons in PREP_VACS and isinstance(prep, str) and prep:
        return f"{cons}-{prep.lower()}"
    return cons


def _schematic_hits(hits: pd.DataFrame) -> pd.DataFrame:
    """Expand locative hits to schematic + prep grains. VO/VOO stay verb-level."""
    if hits.empty:
        return hits
    work = hits.copy()
    if "extra" in work.columns:
        work = work.loc[~work["extra"].fillna(False)]
    work = work.loc[work["construction"].isin(SCHEMATIC)]
    if work.empty:
        return work
    work["verb"] = work["verb"].astype(str).str.lower()
    rows: list[dict] = []
    for rec in work.to_dict(orient="records"):
        cons = str(rec.get("construction") or "")
        verb = rec["verb"]
        schematic = dict(rec)
        schematic["vac"] = cons
        schematic["grain"] = "schematic"
        schematic["item"] = f"{cons}|{verb}"
        # Pool locative prepositions at schematic grain (VL|go, not VL-to|go).
        if cons in PREP_VACS:
            schematic["preposition"] = None
        rows.append(schematic)
        prep = rec.get("preposition")
        if cons in PREP_VACS and isinstance(prep, str) and prep.strip():
            fine = dict(rec)
            fine["vac"] = f"{cons}-{prep.lower()}"
            fine["grain"] = "prep"
            fine["item"] = f"{fine['vac']}|{verb}"
            rows.append(fine)
    return pd.DataFrame(rows)


def child_first_ages(hits: pd.DataFrame) -> pd.DataFrame:
    """Per-child first age (months) for each verb / verb+prep in a VAC."""
    kids = _schematic_hits(hits)
    if kids.empty or "target_child_age" not in kids.columns:
        return pd.DataFrame()
    kids = kids.loc[kids["speaker_role"].isin(CHILD_ROLES)]
    kids = kids.loc[kids["target_child_age"].notna()]
    if kids.empty:
        return kids
    child_col = "target_child_id" if "target_child_id" in kids.columns else "target_child_name"
    cols = [child_col, "vac", "verb", "item", "construction", "preposition", "grain"]
    cols = [c for c in cols if c in kids.columns]
    return (
        kids.groupby(cols, dropna=False)["target_child_age"]
        .agg(first_age="min", n="size")
        .reset_index()
        .rename(columns={child_col: "child_id"})
    )


def _min_children(vac: str) -> int:
    return MIN_CHILDREN_VOO if str(vac).startswith("VOO") else MIN_CHILDREN


def acquisition_order(first_ages: pd.DataFrame, *, min_children: int | None = None) -> pd.DataFrame:
    """Median first-age across children; rank within each VAC (1 = earliest)."""
    if first_ages.empty:
        return first_ages
    group_cols = [c for c in ("vac", "verb", "item", "construction", "preposition", "grain") if c in first_ages.columns]
    summary = (
        first_ages.groupby(group_cols, dropna=False)
        .agg(
            n_children=("child_id", "nunique"),
            n_tokens=("n", "sum"),
            median_first_age=("first_age", "median"),
            mean_first_age=("first_age", "mean"),
            pooled_first_age=("first_age", "min"),
        )
        .reset_index()
    )
    keep = []
    for vac, grp in summary.groupby("vac"):
        thresh = min_children if min_children is not None else _min_children(vac)
        kept = grp.loc[grp["n_children"] >= thresh]
        if kept.empty:
            kept = grp.sort_values("n_tokens", ascending=False).head(8)
        keep.append(kept)
    summary = pd.concat(keep, ignore_index=True) if keep else summary.iloc[0:0]
    summary["aoa_rank"] = summary.groupby("vac")["median_first_age"].rank(method="min")
    return summary.sort_values(["vac", "aoa_rank", "verb"])


def parent_order(hits: pd.DataFrame, *, min_tokens: int = MIN_TOKENS) -> pd.DataFrame:
    """CDS input order: rank by token frequency (and median age of occurrence)."""
    cds = _schematic_hits(hits)
    if cds.empty:
        return cds
    cds = cds.loc[cds["speaker_role"].isin(PARENT_ROLES)]
    if cds.empty:
        return cds
    group_cols = [c for c in ("vac", "verb", "item", "construction", "preposition", "grain") if c in cds.columns]
    summary = (
        cds.groupby(group_cols, dropna=False)
        .agg(
            n_tokens=("utterance_id", "size"),
            median_age=("target_child_age", "median"),
        )
        .reset_index()
    )
    summary = summary.loc[summary["n_tokens"] >= min_tokens]
    summary["input_rank"] = summary.groupby("vac")["n_tokens"].rank(method="min", ascending=False)
    return summary.sort_values(["vac", "input_rank", "verb"])


def age_bins(bin_months: float = DEFAULT_BIN_MONTHS) -> np.ndarray:
    return np.arange(AGE_MIN, AGE_MAX + bin_months, bin_months)


def assign_age_bins(ages, bin_months: float = DEFAULT_BIN_MONTHS) -> np.ndarray:
    a = np.asarray(ages, dtype=float)
    b = np.floor((a - AGE_MIN) / bin_months) * bin_months + AGE_MIN
    return np.clip(b, AGE_MIN, AGE_MAX - bin_months).astype(int)


def child_token_exposure(
    tokens: pd.DataFrame,
    *,
    bin_months: float = DEFAULT_BIN_MONTHS,
) -> pd.Series:
    """Count of child tokens (any word) in each age bin.

    This is the sampling density of the corpus, not VAC occupancy. Use it to
    inverse-weight token CDFs so high-coverage ages do not dominate the shape.
    """
    if tokens.empty or "target_child_age" not in tokens.columns:
        return pd.Series(dtype="int64")
    kids = tokens
    if "speaker_role" in kids.columns:
        kids = kids.loc[kids["speaker_role"].isin(CHILD_ROLES)]
    kids = kids.loc[kids["target_child_age"].notna()]
    if kids.empty:
        return pd.Series(dtype="int64")
    bins = pd.Series(assign_age_bins(kids["target_child_age"], bin_months), index=kids.index)
    return bins.value_counts().sort_index()


def cumulative_bin_share(
    counts: pd.Series,
    bins: list[int],
    exposure: pd.Series | None = None,
) -> tuple[list[float], list[float]]:
    """CDF over age bins, optionally inverse-weighted by overall token exposure.

    Unweighted: mass in bin b is n_v(b). Weighted: mass is n_v(b) / N(b), the
    verb's share of all child tokens in that bin. The CDF is the running sum
    of those masses, renormalized to 1.
    """
    def _get(series: pd.Series | None, b: int) -> float:
        if series is None or series.empty:
            return 0.0
        try:
            idx = series.index.astype(int)
        except (TypeError, ValueError):
            idx = series.index
        lookup = pd.Series(series.to_numpy(), index=idx)
        if b not in lookup.index:
            return 0.0
        return float(lookup.loc[b])

    weights: list[float] = []
    for b in bins:
        n = _get(counts, b)
        if exposure is None:
            weights.append(n)
            continue
        total = _get(exposure, b)
        weights.append((n / total) if total > 0 else 0.0)
    denom = float(sum(weights)) or 1.0
    running = 0.0
    shares: list[float] = []
    for w in weights:
        running += w
        shares.append(running / denom)
    return shares, weights


def cumulative_age_curves(
    hits: pd.DataFrame,
    *,
    bin_months: float = DEFAULT_BIN_MONTHS,
    top_n: int = 8,
    order: pd.DataFrame | None = None,
    exposure: pd.Series | None = None,
) -> pd.DataFrame:
    """Cumulative share of tokens by age bin.

    Schematic VL, VOL, VO, and VOO are always emitted when those hits exist.
    Locatives additionally get verb+preposition series (VL-to, VOL-off, …).
    Pass ``exposure`` (child tokens of any word per bin) to inverse-weight the
    CDF by sampling density.
    """
    kids = _schematic_hits(hits)
    if kids.empty:
        return pd.DataFrame()
    kids = kids.loc[kids["speaker_role"].isin(CHILD_ROLES)]
    kids = kids.loc[kids["target_child_age"].notna()]
    if kids.empty:
        return kids
    kids["age_bin"] = assign_age_bins(kids["target_child_age"], bin_months)
    keep: set[str] = set()
    if order is not None and not order.empty:
        for vac, grp in order.groupby("vac"):
            keep.update(grp.sort_values("aoa_rank").head(top_n)["item"].tolist())
    # Required schematic families: if they did not make the order cutoff, still plot them.
    for vac in SCHEMATIC:
        vac_items = kids.loc[kids["vac"] == vac]
        if vac_items.empty:
            continue
        already = {i for i in keep if i.startswith(f"{vac}|")}
        if already:
            continue
        freq = vac_items.groupby("item").size().sort_values(ascending=False).head(top_n)
        keep.update(freq.index.tolist())
    if keep:
        kids = kids.loc[kids["item"].isin(keep)]
    bins = [int(b) for b in age_bins(bin_months)[:-1]]
    rows = []
    for (vac, verb, item), grp in kids.groupby(["vac", "verb", "item"], dropna=False):
        counts = grp.groupby("age_bin").size()
        shares, weights = cumulative_bin_share(counts, bins, exposure)
        grain = grp["grain"].iloc[0] if "grain" in grp.columns else ("schematic" if vac in SCHEMATIC else "prep")
        n_tokens = int(counts.sum())
        for b, share, weight in zip(bins, shares, weights):
            rows.append(
                {
                    "vac": vac,
                    "verb": verb,
                    "item": item,
                    "grain": grain,
                    "age_months": b,
                    "cumulative_share": share,
                    "bin_tokens": int(counts.get(b, 0)),
                    "bin_weight": weight,
                    "n_tokens": n_tokens,
                }
            )
    return pd.DataFrame(rows)


def child_observation_spans(tokens: pd.DataFrame) -> pd.DataFrame:
    """First and last child-speech age (months) per child."""
    if tokens.empty or "target_child_age" not in tokens.columns:
        return pd.DataFrame()
    kids = tokens.loc[tokens["speaker_role"].isin(CHILD_ROLES)]
    kids = kids.loc[kids["target_child_age"].notna()]
    if kids.empty:
        return kids
    child_col = "target_child_id" if "target_child_id" in kids.columns else "target_child_name"
    out = (
        kids.groupby(child_col, dropna=False)
        .agg(first_obs=("target_child_age", "min"), last_obs=("target_child_age", "max"))
        .reset_index()
        .rename(columns={child_col: "child_id"})
    )
    if "target_child_name" in kids.columns and child_col != "target_child_name":
        names = kids.groupby(child_col)["target_child_name"].agg(
            lambda s: s.dropna().astype(str).iloc[0] if s.dropna().size else ""
        )
        out["child_name"] = out["child_id"].map(names)
    return out


def right_censored_times(
    first_ages: pd.DataFrame,
    spans: pd.DataFrame,
    items: pd.DataFrame,
) -> pd.DataFrame:
    """One row per child × item: event at first production, else censored at last observation."""
    if first_ages.empty or spans.empty or items.empty:
        return pd.DataFrame()
    item_cols = [c for c in ("vac", "verb", "item", "construction", "grain") if c in items.columns]
    catalog = items[item_cols].drop_duplicates()
    children = spans.loc[spans["last_obs"].notna(), ["child_id", "last_obs"]].drop_duplicates("child_id")
    grid = catalog.merge(children, how="cross")
    events = first_ages[["child_id", "item", "first_age"]].drop_duplicates(["child_id", "item"])
    out = grid.merge(events, on=["child_id", "item"], how="left")
    produced = out["first_age"].notna()
    out["event"] = produced.astype(int)
    out["time"] = np.where(produced, out["first_age"], out["last_obs"])
    return out


def kaplan_meier_curve(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Event times and Kaplan–Meier survival S(t) = P(AoA > t)."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    ok = np.isfinite(time)
    time, event = time[ok], event[ok]
    if time.size == 0:
        return np.array([]), np.array([])
    order = np.argsort(time, kind="mergesort")
    time, event = time[order], event[order]
    surv_t: list[float] = []
    surv_s: list[float] = []
    survival = 1.0
    for t in np.unique(time):
        at_t = time == t
        deaths = int(event[at_t].sum())
        n_risk = int((time >= t).sum())
        if deaths > 0 and n_risk > 0:
            survival *= 1.0 - deaths / n_risk
            surv_t.append(float(t))
            surv_s.append(float(survival))
            if survival <= 0:
                break
    return np.asarray(surv_t), np.asarray(surv_s)


def kaplan_meier_median(time: np.ndarray, event: np.ndarray) -> float:
    """Smallest t with KM S(t) ≤ 0.5. NaN if the median is not reached."""
    times, surv = kaplan_meier_curve(time, event)
    if times.size == 0:
        return float("nan")
    hit = np.flatnonzero(surv <= 0.5)
    if hit.size == 0:
        return float("nan")
    return float(times[int(hit[0])])


def bootstrap_km_median_ci(
    time: np.ndarray,
    event: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI for the KM median by resampling children."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    n = time.size
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        meds[i] = kaplan_meier_median(time[idx], event[idx])
    finite = meds[np.isfinite(meds)]
    if finite.size < max(20, int(0.5 * n_boot)):
        lo = float(np.nanmin(finite)) if finite.size else float("nan")
        return lo, float("nan")
    q = 100.0 * np.array([alpha / 2.0, 1.0 - alpha / 2.0])
    lo, hi = np.percentile(finite, q)
    return float(lo), float(hi)


def censored_aoa_summary(
    times: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """KM median first-production age per VAC × verb, with bootstrap 95% CI."""
    if times.empty:
        return times
    rows = []
    for (vac, verb, item), grp in times.groupby(["vac", "verb", "item"], dropna=False):
        time = grp["time"].to_numpy(dtype=float)
        event = grp["event"].to_numpy(dtype=int)
        median = kaplan_meier_median(time, event)
        lo, hi = bootstrap_km_median_ci(time, event, n_boot=n_boot, seed=seed)
        times_km, surv = kaplan_meier_curve(time, event)
        rows.append(
            {
                "vac": vac,
                "verb": verb,
                "item": item,
                "n_risk": int(len(grp)),
                "n_event": int(event.sum()),
                "n_censored": int((event == 0).sum()),
                "naive_median": float(grp.loc[grp["event"] == 1, "time"].median()) if int(event.sum()) else float("nan"),
                "km_median": median,
                "km_ci_low": lo,
                "km_ci_high": hi,
                "km_s_end": float(surv[-1]) if surv.size else 1.0,
                "median_reached": bool(median == median),
            }
        )
    out = pd.DataFrame(rows)
    out["aoa_rank"] = out.groupby("vac")["km_median"].rank(method="min")
    return out.sort_values(["vac", "aoa_rank", "verb"])


def model_verb_curves(ranking: pd.DataFrame) -> pd.DataFrame:
    """Mean ranking score per epoch × VAC × verb. Emits schematic VO/VOO/VL/VOL plus prep variants."""
    if ranking.empty:
        return ranking
    work = ranking.copy()
    work["verb"] = work["verb"].astype(str).str.lower()
    schematic = work.copy()
    schematic["vac"] = schematic["construction"].astype(str)
    schematic["grain"] = "schematic"
    schematic["item"] = schematic["vac"] + "|" + schematic["verb"]
    fine = work.copy()
    fine["vac"] = fine.apply(vac_label, axis=1)
    fine["grain"] = np.where(fine["vac"].isin(SCHEMATIC), "schematic", "prep")
    fine["item"] = fine["vac"] + "|" + fine["verb"]
    fine = fine.loc[fine["vac"] != fine["construction"].astype(str)]
    work = pd.concat([schematic, fine], ignore_index=True)
    return (
        work.groupby(
            ["track", "epoch", "vac", "verb", "item", "kind", "construction", "preposition", "grain"],
            dropna=False,
        )["score"]
        .mean()
        .reset_index(name="score")
    )


def model_acquisition_order(curves: pd.DataFrame) -> pd.DataFrame:
    """First epoch at which an occupant outscores the mean distractor in that VAC."""
    if curves.empty:
        return curves
    rows = []
    for (track, vac), grp in curves.groupby(["track", "vac"]):
        dist = grp.loc[grp["kind"] == "distractor"].groupby("epoch")["score"].mean()
        occ = grp.loc[grp["kind"] == "occupant"]
        for verb, vg in occ.groupby("verb"):
            vg = vg.sort_values("epoch")
            acquire = None
            for rec in vg.to_dict(orient="records"):
                dmean = dist.get(rec["epoch"], np.nan)
                if dmean == dmean and rec["score"] > dmean:
                    acquire = int(rec["epoch"])
                    break
            last = vg.iloc[-1]
            rows.append(
                {
                    "track": track,
                    "vac": vac,
                    "verb": verb,
                    "item": last["item"],
                    "grain": last.get("grain", "schematic" if vac in SCHEMATIC else "prep"),
                    "construction": last["construction"],
                    "preposition": last.get("preposition"),
                    "acquire_epoch": acquire,
                    "final_score": float(last["score"]),
                    "final_distractor_mean": float(dist.get(last["epoch"], np.nan)) if last["epoch"] in dist.index else float("nan"),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["epoch_rank"] = out.groupby(["track", "vac"])["acquire_epoch"].rank(method="min")
    return out.sort_values(["track", "vac", "epoch_rank", "verb"])


def compare_orders(child_order: pd.DataFrame, model_order: pd.DataFrame) -> pd.DataFrame:
    """Spearman ρ of child median-AoA rank vs model acquire-epoch rank."""
    from scipy import stats

    rows = []
    if child_order.empty or model_order.empty:
        return pd.DataFrame()
    child = child_order.copy()
    child["verb"] = child["verb"].str.lower()
    for (track, vac), grp in model_order.groupby(["track", "vac"]):
        merged = grp.merge(child.loc[child["vac"] == vac, ["verb", "aoa_rank", "median_first_age"]], on="verb")
        merged = merged.dropna(subset=["epoch_rank", "aoa_rank"])
        if len(merged) < 3:
            continue
        rho, p = stats.spearmanr(merged["aoa_rank"], merged["epoch_rank"])
        rows.append(
            {
                "track": track,
                "vac": vac,
                "n_verbs": int(len(merged)),
                "spearman_rho": float(rho) if rho == rho else float("nan"),
                "p": float(p) if p == p else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def run_acquisition(
    hits: pd.DataFrame,
    ranking: pd.DataFrame | None = None,
    *,
    bin_months: float = DEFAULT_BIN_MONTHS,
    observation_spans: pd.DataFrame | None = None,
    n_boot: int = 1000,
    cache_prefix: str = "acq",
) -> dict[str, pd.DataFrame]:
    first = child_first_ages(hits)
    order = acquisition_order(first)
    curves = cumulative_age_curves(hits, bin_months=bin_months, order=order)
    cds = parent_order(hits)
    out: dict[str, pd.DataFrame] = {
        "child_first_ages": first,
        "child_order": order,
        "child_age_curves": curves,
        "parent_order": cds,
    }
    if observation_spans is not None and not observation_spans.empty and not order.empty:
        schematic_order = []
        for vac in SCHEMATIC:
            grp = order.loc[order["vac"] == vac].sort_values("aoa_rank").head(8)
            if not grp.empty:
                schematic_order.append(grp)
        items = (
            pd.concat(schematic_order, ignore_index=True)[["vac", "verb", "item"]].drop_duplicates()
            if schematic_order
            else pd.DataFrame()
        )
        if not items.empty:
            times = right_censored_times(first, observation_spans, items)
            out["censored_times"] = times
            out["censored_aoa"] = censored_aoa_summary(times, n_boot=n_boot)
    if ranking is not None and not ranking.empty:
        mcurves = model_verb_curves(ranking)
        morder = model_acquisition_order(mcurves)
        out["model_curves"] = mcurves
        out["model_order"] = morder
        out["order_correlation"] = compare_orders(order, morder)
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_csv(CACHE / f"{cache_prefix}_{name}.csv", index=False)
    present = sorted(curves["vac"].dropna().unique().tolist()) if not curves.empty else []
    (CACHE / f"{cache_prefix}_status.json").write_text(
        json.dumps(
            {
                **{k: int(len(v)) for k, v in out.items()},
                "curve_vacs": present,
                "schematic_included": [v for v in SCHEMATIC if v in present],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out
