"""Extract schematic VACs from tokens_filled.parquet and write acquisition tables."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from vacacq import CACHE
from vacacq.childes.extract import extract_vacs
from vacacq.childes.strata import CHILD_ROLES
from vacacq.parse.fill import keep_parsed_utterances
from vacacq.stats.acquisition import (
    SCHEMATIC,
    age_bins,
    assign_age_bins,
    censored_aoa_summary,
    child_observation_spans,
    child_token_exposure,
    cumulative_bin_share,
    right_censored_times,
    run_acquisition,
)

FILLED = CACHE / "tokens_filled.parquet"
HITS_ALL = CACHE / "hits_filled.parquet"
SPANS = CACHE / "spans_filled.parquet"
EXPOSURE = CACHE / "acq_filled_token_exposure.csv"
CANVAS = Path(
    "/Users/alvintan/.cursor/projects/Users-alvintan-Documents-03-Research-explorations-vac-acq"
    "/canvases/vac-acquisition-curves.canvas.tsx"
)

TOKEN_COLS = [
    "id",
    "utterance_id",
    "token_order",
    "gloss",
    "part_of_speech",
    "stem",
    "suffix",
    "gra_index",
    "gra_head",
    "gra_relation",
    "collection_name",
    "corpus_name",
    "speaker_role",
    "utterance_type",
    "target_child_age",
    "target_child_id",
    "target_child_name",
    "filename",
    "in_s7_2",
    "parse_source",
]

AGES = [18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69]
VAC_TITLES = {
    "VL": "VL intransitive locative",
    "VOL": "VOL caused motion",
    "VO": "VO transitive",
    "VOO": "VOO ditransitive",
}


def _merge_spans(parts: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(parts, ignore_index=True)
    if df.empty:
        return df
    agg = {"first_obs": "min", "last_obs": "max"}
    if "child_name" in df.columns:
        agg["child_name"] = "first"
    return df.groupby("child_id", dropna=False).agg(agg).reset_index()


def extract_all() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pf = pq.ParquetFile(FILLED)
    n_rg = pf.num_row_groups
    print(f"filled file: {FILLED} ({pf.metadata.num_rows} tokens, {n_rg} row groups)", flush=True)
    available = [c for c in TOKEN_COLS if c in pf.schema_arrow.names]
    hit_parts: list[pd.DataFrame] = []
    span_parts: list[pd.DataFrame] = []
    n_tokens = 0
    n_parsed = 0
    corpora: set[str] = set()
    sources: dict[str, int] = {}
    carry: pd.DataFrame | None = None
    for i in range(n_rg):
        df = pf.read_row_group(i, columns=available).to_pandas()
        if carry is not None:
            df = pd.concat([carry, df], ignore_index=True)
        if i < n_rg - 1 and not df.empty:
            last_uid = df["utterance_id"].iloc[-1]
            carry = df.loc[df["utterance_id"].eq(last_uid)].copy()
            df = df.loc[~df["utterance_id"].eq(last_uid)]
        else:
            carry = None
        n_tokens += len(df)
        if "corpus_name" in df.columns:
            corpora.update(df["corpus_name"].dropna().astype(str).unique().tolist())
        if "parse_source" in df.columns:
            for src, n in df["parse_source"].fillna("unparsed").astype(str).value_counts().items():
                sources[str(src)] = sources.get(str(src), 0) + int(n)
        before = len(df)
        df = keep_parsed_utterances(df)
        n_parsed += len(df)
        print(
            f"[{i + 1}/{n_rg}] {before} tokens, {len(df)} parsed utterances kept",
            flush=True,
        )
        if df.empty:
            continue
        span_parts.append(child_observation_spans(df))
        hits = extract_vacs(df, extras=False)
        print(f"  extracted {len(hits)} hits", flush=True)
        if not hits.empty:
            hit_parts.append(hits)
        del df
    hits = pd.concat(hit_parts, ignore_index=True) if hit_parts else pd.DataFrame()
    spans = _merge_spans(span_parts) if span_parts else pd.DataFrame()
    hits.to_parquet(HITS_ALL, index=False)
    spans.to_parquet(SPANS, index=False)
    meta = {
        "n_tokens": n_tokens,
        "n_parsed_tokens": n_parsed,
        "n_hits": int(len(hits)),
        "n_children": int(len(spans)),
        "n_corpora": len(corpora),
        "corpora": sorted(corpora),
        "parse_source_tokens": sources,
    }
    print(f"wrote {HITS_ALL} ({len(hits)} hits)", flush=True)
    print(f"wrote {SPANS} ({len(spans)} children)", flush=True)
    print(json.dumps(meta, indent=2), flush=True)
    return hits, spans, meta


def _finite(val) -> float | None:
    if val is None:
        return None
    try:
        x = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, 2)


def frequent_schematic_order(order: pd.DataFrame, *, top_n: int = 8) -> pd.DataFrame:
    """Top-n verbs by child tokens per schematic VAC, re-ranked by naive median AoA.

    With hundreds of children the 5-child earliest-median ranking is dominated by
    rare noun lemmas. Frequency occupancy is the Ellis Ch.7 comparison set.
    """
    work = order.loc[order["vac"].isin(SCHEMATIC)].copy()
    if "grain" in work.columns:
        work = work.loc[work["grain"].eq("schematic") | work["grain"].isna()]
    parts = []
    for vac in SCHEMATIC:
        grp = work.loc[work["vac"] == vac].sort_values("n_tokens", ascending=False).head(top_n)
        if grp.empty:
            continue
        grp = grp.copy()
        grp["aoa_rank"] = grp["median_first_age"].rank(method="min")
        parts.append(grp.sort_values(["aoa_rank", "verb"]))
    return pd.concat(parts, ignore_index=True) if parts else work.iloc[0:0]


def schematic_token_curves(
    hits: pd.DataFrame,
    items: pd.DataFrame,
    exposure: pd.Series | None = None,
) -> pd.DataFrame:
    """Cumulative child-token share curves for selected schematic items only.

    When ``exposure`` is given (child tokens of any word per age bin), each
    bin's mass is n_verb / N_all so CHILDES sampling density does not set the
    CDF shape.
    """
    if hits.empty or items.empty:
        return pd.DataFrame()
    keep = set(items["item"].astype(str))
    kids = hits.loc[hits["speaker_role"].isin(CHILD_ROLES)]
    kids = kids.loc[kids["construction"].isin(SCHEMATIC)]
    kids = kids.loc[kids["target_child_age"].notna()].copy()
    if kids.empty:
        return kids
    kids["verb"] = kids["verb"].astype(str).str.lower()
    kids["vac"] = kids["construction"].astype(str)
    kids["item"] = kids["vac"] + "|" + kids["verb"]
    kids = kids.loc[kids["item"].isin(keep)]
    bin_months = 3.0
    kids["age_bin"] = assign_age_bins(kids["target_child_age"], bin_months)
    bins = [int(b) for b in age_bins(bin_months)[:-1]]
    rows = []
    for (vac, verb, item), grp in kids.groupby(["vac", "verb", "item"], dropna=False):
        counts = grp.groupby("age_bin").size()
        shares, weights = cumulative_bin_share(counts, bins, exposure)
        n_tokens = int(counts.sum())
        for b, share, weight in zip(bins, shares, weights):
            rows.append(
                {
                    "vac": vac,
                    "verb": verb,
                    "item": item,
                    "grain": "schematic",
                    "age_months": b,
                    "cumulative_share": share,
                    "bin_tokens": int(counts.get(b, 0)),
                    "bin_weight": weight,
                    "n_tokens": n_tokens,
                    "exposure_tokens": int(exposure.get(b, 0)) if exposure is not None else 0,
                }
            )
    return pd.DataFrame(rows)


def child_token_exposure_from_parquet(path: Path) -> pd.Series:
    """Stream speaker_role + age from the filled token file; count child tokens per bin."""
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    cols = [c for c in ("speaker_role", "target_child_age") if c in names]
    parts: list[pd.Series] = []
    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i, columns=cols).to_pandas()
        parts.append(child_token_exposure(df))
        del df
    if not parts:
        return pd.Series(dtype="int64")
    out = pd.concat(parts).groupby(level=0).sum().sort_index()
    out.index = out.index.astype(int)
    out.index.name = "age_months"
    out.name = "n_tokens"
    return out


def _verb_series(order: pd.DataFrame, curves: pd.DataFrame, vac: str) -> list[dict]:
    grp = order.loc[order["vac"] == vac].sort_values("aoa_rank")
    if "grain" in grp.columns:
        grp = grp.loc[grp["grain"].eq("schematic") | grp["grain"].isna()]
    grp = grp.head(8)
    out = []
    for rec in grp.to_dict(orient="records"):
        item = rec["item"]
        shares = (
            curves.loc[(curves["item"] == item) & (curves["vac"] == vac)]
            .sort_values("age_months")
        )
        share_map = {int(r.age_months): float(r.cumulative_share) for r in shares.itertuples()}
        n_tokens = int(shares["n_tokens"].iloc[0]) if not shares.empty else int(rec.get("n_tokens") or 0)
        out.append(
            {
                "verb": str(rec["verb"]),
                "medianFirstAge": round(float(rec["median_first_age"]), 1),
                "nChildren": int(rec["n_children"]),
                "nTokens": n_tokens,
                "aoaRank": int(rec["aoa_rank"]),
                "shares": [round(share_map.get(a, 0.0), 4) for a in AGES],
            }
        )
    return out


def _rle_ages(ages: list[float]) -> list[list[float | int]]:
    out: list[list[float | int]] = []
    for a in ages:
        if out and out[-1][0] == a:
            out[-1][1] += 1
        else:
            out.append([a, 1])
    return out


def _event_ages(item: str, times: pd.DataFrame, first_ages: pd.DataFrame) -> list[float]:
    if not times.empty and "event" in times.columns and "item" in times.columns:
        ev = times.loc[(times["item"].astype(str) == item) & (times["event"] == 1), "time"]
    elif not first_ages.empty and "item" in first_ages.columns:
        ev = first_ages.loc[first_ages["item"].astype(str) == item, "first_age"]
    else:
        ev = pd.Series(dtype=float)
    return sorted(round(float(x), 2) for x in ev.tolist() if pd.notna(x))


def _first_prod(aoa: pd.DataFrame, times: pd.DataFrame, first_ages: pd.DataFrame | None = None) -> dict:
    first_ages = first_ages if first_ages is not None else pd.DataFrame()
    facets = []
    for vac in SCHEMATIC:
        rows = aoa.loc[aoa["vac"] == vac].copy()
        if rows.empty:
            continue
        rows["_ord"] = rows["km_median"].rank(method="min", na_option="bottom")
        rows = rows.sort_values(["_ord", "verb"])
        verbs = []
        for rec in rows.to_dict(orient="records"):
            ages = _event_ages(str(rec["item"]), times, first_ages)
            verbs.append(
                {
                    "verb": str(rec["verb"]),
                    "km": _finite(rec.get("km_median")),
                    "lo": _finite(rec.get("km_ci_low")),
                    "hi": _finite(rec.get("km_ci_high")),
                    "naive": _finite(rec.get("naive_median")),
                    "nEvent": int(rec.get("n_event") or 0),
                    "nCensored": int(rec.get("n_censored") or 0),
                    "nRisk": int(rec.get("n_risk") or 0),
                    "sEnd": round(float(rec.get("km_s_end") or 0.0), 3),
                    "ages": _rle_ages(ages),
                }
            )
        facets.append({"vac": vac, "title": VAC_TITLES[vac], "verbs": verbs})
    return {"xMin": 18, "xMax": 72, "facets": facets}


def _fmt(n: int) -> str:
    return f"{n:,}"


def write_canvas(
    tables: dict[str, pd.DataFrame],
    hits: pd.DataFrame,
    meta: dict,
    exposure: pd.Series | None = None,
) -> None:
    order = tables["child_order"]
    curves = tables["child_age_curves"]
    schematic_curves = curves.loc[curves["vac"].isin(SCHEMATIC)]
    if "grain" in schematic_curves.columns:
        schematic_curves = schematic_curves.loc[
            schematic_curves["grain"].eq("schematic") | schematic_curves["grain"].isna()
        ]
    series = {vac: _verb_series(order, schematic_curves, vac) for vac in SCHEMATIC}
    first_prod = _first_prod(
        tables.get("censored_aoa", pd.DataFrame()),
        tables.get("censored_times", pd.DataFrame()),
        tables.get("child_first_ages", pd.DataFrame()),
    )
    kids = hits.loc[hits["speaker_role"].isin(CHILD_ROLES)] if "speaker_role" in hits.columns else hits
    kids = kids.loc[kids["construction"].isin(SCHEMATIC)] if "construction" in kids.columns else kids
    child_tokens = kids.groupby("construction").size().to_dict() if not kids.empty else {}
    n_risk = int(meta.get("n_children") or 0)
    n_corpora = int(meta.get("n_corpora") or 0)
    sources = meta.get("parse_source_tokens") or {}
    stanza = int(sources.get("stanza_gloss") or 0)
    gold = int(sources.get("childesdb") or 0)
    vo = series["VO"]
    voo = series["VOO"]
    want = next((v for v in vo if v["verb"] == "want"), vo[0] if vo else None)
    getv = next((v for v in vo if v["verb"] == "get"), None)
    give = next((v for v in voo if v["verb"] == "give"), voo[0] if voo else None)
    km_vo = {f["vac"]: f["verbs"] for f in first_prod["facets"]}
    want_km = next((v for v in km_vo.get("VO", []) if v["verb"] == "want"), None)
    get_km = next((v for v in km_vo.get("VO", []) if v["verb"] == "get"), None)
    give_km = next((v for v in km_vo.get("VOO", []) if v["verb"] == "give"), None)

    def _age(v: dict | None, key: str) -> str:
        if not v or v.get(key) is None:
            return "not reached"
        return f"{v[key]:.0f}" if key == "km" else f"{v[key]:.1f}"

    vo_line = "want/get"
    if want_km or get_km:
        bits = []
        if want_km:
            bits.append(f"want { _age(want_km, 'km') }")
        if get_km:
            bits.append(f"get { _age(get_km, 'km') }")
        vo_line = " / ".join(bits) + " months (KM)"
    give_line = "give"
    if give_km:
        give_line = f"give { _age(give_km, 'km') } months (KM; {give_km['nEvent']} producers)"

    if exposure is None:
        exposure = load_or_build_exposure() if FILLED.exists() or EXPOSURE.exists() else None
    sampling_callout = ""
    if exposure is not None and len(exposure):
        exp_lo = int(exposure.min())
        exp_hi = int(exposure.max())
        exp_lo_age = int(exposure.idxmin())
        exp_hi_age = int(exposure.idxmax())
        exp_total = int(exposure.sum())
        sampling_callout = f"""      <Callout tone="info" title="CDFs adjusted for sampling density">
        Child tokens (any word) per 3-month bin range from {_fmt(exp_lo)}
        ({exp_lo_age} mo) to {_fmt(exp_hi)} ({exp_hi_age} mo); {_fmt(exp_total)}
        child tokens between 18 and 72 months. Each verb&apos;s CDF uses
        verb-tokens / all-tokens in that bin, then renormalizes, so the shape
        tracks the word rather than how much CHILDES data happens to exist.
      </Callout>
"""
    dumps = json.dumps
    ts = f'''import {{
  Callout,
  Card,
  CardBody,
  CardHeader,
  Grid,
  H1,
  H2,
  H3,
  LineChart,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
}} from "cursor/canvas";

const AGES = {dumps(AGES)};
const AGE_LABELS = AGES.map((a) => String(a));

type VerbSeries = {{
  verb: string;
  medianFirstAge: number;
  nChildren: number;
  nTokens: number;
  aoaRank: number;
  shares: number[];
}};

const VL: VerbSeries[] = {dumps(series["VL"])};
const VOL: VerbSeries[] = {dumps(series["VOL"])};
const VO: VerbSeries[] = {dumps(series["VO"])};
const VOO: VerbSeries[] = {dumps(series["VOO"])};

type FirstProdVerb = {{
  verb: string;
  km: number | null;
  lo: number | null;
  hi: number | null;
  naive: number | null;
  nEvent: number;
  nCensored: number;
  nRisk: number;
  sEnd: number;
  ages: [number, number][];
}};

type FirstProdFacet = {{
  vac: string;
  title: string;
  verbs: FirstProdVerb[];
}};

type FirstProdPayload = {{
  xMin: number;
  xMax: number;
  facets: FirstProdFacet[];
}};

const FIRST_PROD = {dumps(first_prod, separators=(",", ":"))} as FirstProdPayload;
const N_CHILDREN = {n_risk};
const N_CORPORA = {n_corpora};
const SOURCE_NOTE = "tokens_filled.parquet · {n_corpora} corpora · {_fmt(n_risk)} children · childes-db 2026.1 + Stanza-on-gloss";

function expandAges(rle: [number, number][]): number[] {{
  const out: number[] = [];
  for (const [age, n] of rle) {{
    for (let i = 0; i < n; i++) out.push(age);
  }}
  return out;
}}

function swarmPoints(
  ages: number[],
  xOf: (age: number) => number,
  yMid: number,
): {{ x: number; y: number }}[] {{
  const r = 2.3;
  const bins = new Map<number, number[]>();
  ages.forEach((age, i) => {{
    const key = Math.round(xOf(age) / 3) * 3;
    const list = bins.get(key) ?? [];
    list.push(i);
    bins.set(key, list);
  }});
  const out = ages.map((age) => ({{ x: xOf(age), y: yMid }}));
  for (const idxs of bins.values()) {{
    const n = idxs.length;
    const step = n > 1 ? Math.min(r * 1.65, 16 / (n - 1)) : 0;
    idxs.forEach((i, k) => {{
      out[i].y = yMid + (k - (n - 1) / 2) * step;
    }});
  }}
  return out;
}}

function FirstProductionFacet({{ facet }}: {{ facet: FirstProdFacet }}) {{
  const theme = useHostTheme();
  const verbs = facet.verbs;
  const padL = 52;
  const padR = 18;
  const padT = 10;
  const padB = 28;
  const rowH = 26;
  const width = 560;
  const innerW = width - padL - padR;
  const height = padT + verbs.length * rowH + padB;
  const xMin = FIRST_PROD.xMin;
  const xMax = FIRST_PROD.xMax;
  const xOf = (age: number) => padL + ((age - xMin) / (xMax - xMin)) * innerW;
  const ticks = [18, 24, 36, 48, 60, 72];
  const ink = theme.text.primary;
  const muted = theme.text.tertiary;
  const grid = theme.stroke.tertiary;
  const axis = theme.stroke.secondary;
  const childFill = theme.accent.primary;

  return (
    <Stack gap={{6}}>
      <H3>{{facet.title}}</H3>
      <Text tone="secondary" size="small">
        X-axis: age (months). Y-axis: verb, earliest KM median at the top.
      </Text>
      <svg
        width="100%"
        viewBox={{`0 0 ${{width}} ${{height}}`}}
        role="img"
        aria-label={{`${{facet.vac}} first-production ages`}}
        style={{{{ display: "block" }}}}
      >
        {{ticks.map((t) => (
          <line
            key={{`g-${{t}}`}}
            x1={{xOf(t)}}
            x2={{xOf(t)}}
            y1={{padT}}
            y2={{height - padB}}
            stroke={{grid}}
            strokeWidth={{1}}
          />
        ))}}
        {{verbs.map((v, i) => {{
          const y = padT + i * rowH + rowH / 2;
          const pts = swarmPoints(expandAges(v.ages), xOf, y);
          const yTop = y - 5;
          const yBot = y + 5;
          return (
            <g key={{v.verb}}>
              <text
                x={{padL - 8}}
                y={{y + 4}}
                textAnchor="end"
                fill={{theme.text.secondary}}
                fontSize="11"
                fontFamily="ui-sans-serif, system-ui, sans-serif"
              >
                {{v.verb}}
              </text>
              {{pts.map((p, j) => (
                <circle
                  key={{j}}
                  cx={{p.x}}
                  cy={{p.y}}
                  r={{2.2}}
                  fill={{childFill}}
                  fillOpacity={{0.55}}
                />
              ))}}
              {{v.km != null && v.lo != null && v.hi != null ? (
                <g>
                  <line
                    x1={{xOf(v.lo)}}
                    x2={{xOf(v.hi)}}
                    y1={{y}}
                    y2={{y}}
                    stroke={{ink}}
                    strokeWidth={{1.75}}
                  />
                  <line x1={{xOf(v.lo)}} x2={{xOf(v.lo)}} y1={{yTop}} y2={{yBot}} stroke={{ink}} strokeWidth={{1.75}} />
                  <line x1={{xOf(v.hi)}} x2={{xOf(v.hi)}} y1={{yTop}} y2={{yBot}} stroke={{ink}} strokeWidth={{1.75}} />
                  <circle cx={{xOf(v.km)}} cy={{y}} r={{4.2}} fill={{ink}} />
                </g>
              ) : v.lo != null ? (
                <g>
                  <line
                    x1={{xOf(v.lo)}}
                    x2={{xOf(xMax)}}
                    y1={{y}}
                    y2={{y}}
                    stroke={{ink}}
                    strokeWidth={{1.5}}
                    strokeDasharray="3 3"
                  />
                  <polygon
                    points={{`${{xOf(xMax)}},${{y}} ${{xOf(xMax) - 7}},${{y - 4}} ${{xOf(xMax) - 7}},${{y + 4}}`}}
                    fill={{ink}}
                  />
                </g>
              ) : null}}
            </g>
          );
        }})}}
        <line
          x1={{padL}}
          x2={{width - padR}}
          y1={{height - padB}}
          y2={{height - padB}}
          stroke={{axis}}
          strokeWidth={{1}}
        />
        {{ticks.map((t) => (
          <text
            key={{`t-${{t}}`}}
            x={{xOf(t)}}
            y={{height - 8}}
            textAnchor="middle"
            fill={{muted}}
            fontSize="10"
            fontFamily="ui-sans-serif, system-ui, sans-serif"
          >
            {{t}}
          </text>
        ))}}
      </svg>
      <Text tone="tertiary" size="small">
        X-axis: child age (months). Y-axis: verb. Source: {{SOURCE_NOTE}}.
        Right-censored Kaplan–Meier.
      </Text>
    </Stack>
  );
}}

function pct(shares: number[]): number[] {{
  return shares.map((s) => Math.round(s * 1000) / 10);
}}

function chartSeries(verbs: VerbSeries[], minChildren: number) {{
  return verbs
    .filter((v) => v.nChildren >= minChildren)
    .map((v) => ({{ name: v.verb, data: pct(v.shares) }}));
}}

function fmtKm(v: FirstProdVerb): string {{
  return v.km == null ? "not reached" : v.km.toFixed(1);
}}

function fmtCi(v: FirstProdVerb): string {{
  if (v.lo != null && v.hi != null) {{
    return `${{v.lo.toFixed(1)}}–${{v.hi.toFixed(1)}}`;
  }}
  if (v.lo != null) {{
    return `≥ ${{v.lo.toFixed(1)}}`;
  }}
  return "—";
}}

function kmRows(facet: FirstProdFacet, series: VerbSeries[]) {{
  const tokens = new Map(series.map((s) => [s.verb, s.nTokens]));
  return facet.verbs.map((v, i) => [
    String(i + 1),
    v.verb,
    fmtKm(v),
    fmtCi(v),
    String(v.nEvent),
    (tokens.get(v.verb) ?? 0).toLocaleString(),
  ]);
}}

function VacChart({{
  title,
  verbs,
  minChildren,
}}: {{
  title: string;
  verbs: VerbSeries[];
  minChildren: number;
}}) {{
  return (
    <Stack gap={{8}}>
      <H3>{{title}}</H3>
      <Text tone="secondary" size="small">
        Sampling-adjusted CDF: each 3-month bin is weighted by this verb&apos;s
        tokens divided by all child tokens (any word) in that bin, then
        accumulated and renormalized to 100%. Lines are verbs produced by at
        least {{minChildren}} children. Y-axis is cumulative weighted share (%).
      </Text>
      <LineChart
        categories={{AGE_LABELS}}
        series={{chartSeries(verbs, minChildren)}}
        yMin={{0}}
        yMax={{100}}
        valueSuffix="%"
        height={{220}}
        fill
      />
    </Stack>
  );
}}

export default function VacAcquisitionCurves() {{
  return (
    <Stack gap={{24}}>
      <Stack gap={{8}}>
        <H1>Child acquisition curves include VO and VOO</H1>
        <Text tone="secondary">
          Schematic VL, VOL, VO, and VOO on {{N_CORPORA}} English CHILDES corpora
          from tokens_filled.parquet (childes-db 2026.1 gold parses plus
          Stanza-on-gloss fills). Remaining fully unparsed utterances are
          dropped. Each panel shows the eight most frequent child-produced
          verbs in that construction, ranked by naive median first age.
          Line-chart CDFs are inverse-weighted by total child tokens in each
          age bin so corpus coverage does not set the shape. Locative
          verb+preposition series are in the CSV tables. Model epoch curves
          are not scored yet.
        </Text>
      </Stack>

      <Callout tone="info" title="Filled parses">
        Gold childes-db tags cover {_fmt(gold)} tokens; Stanza-on-gloss added
        {_fmt(stanza)} tokens. Extraction keeps utterances with POS and %gra
        so silver fills are included and leftover unparsed speech is not.
      </Callout>

{sampling_callout}

      <Grid columns={{4}} gap={{16}}>
        <Stat value="{_fmt(int(child_tokens.get("VL", 0)))}" label="Child VL tokens" />
        <Stat value="{_fmt(int(child_tokens.get("VOL", 0)))}" label="Child VOL tokens" />
        <Stat value="{_fmt(int(child_tokens.get("VO", 0)))}" label="Child VO tokens" />
        <Stat value="{_fmt(int(child_tokens.get("VOO", 0)))}" label="Child VOO tokens" />
      </Grid>

      <Callout tone="info" title="VO and VOO are verb-level">
        Transitive VO and ditransitive VOO have no preposition grain. KM median
        first age: {vo_line}. {give_line}. The 5-child earliest-median ranking
        on 962 children is dominated by rare lemmas, so the plots use
        token-frequency occupants instead.
      </Callout>

      <Grid columns={{2}} gap={{24}}>
        <VacChart title="VL intransitive locative" verbs={{VL}} minChildren={{10}} />
        <VacChart title="VOL caused motion" verbs={{VOL}} minChildren={{10}} />
        <VacChart title="VO transitive" verbs={{VO}} minChildren={{16}} />
        <VacChart title="VOO ditransitive" verbs={{VOO}} minChildren={{8}} />
      </Grid>

      <Stack gap={{8}}>
        <H2>First production by child, right-censored</H2>
        <Text tone="secondary">
          Each small dot is one child&apos;s first observed use of that
          verb+construction. Children who were recorded but never produced the
          combination are treated as right-censored at their last parsed
          child-speech session and are not plotted as dots. The large ink marker
          is the Kaplan–Meier median age of acquisition; whiskers are a 95%
          bootstrap CI. If the KM curve never falls to 50%, the median is not
          reached and a dashed arrow shows the lower bound (if any).
          Late-entering children are still in the risk set (no left truncation).
        </Text>
      </Stack>

      <Grid columns={{2}} gap={{24}}>
        <FirstProductionFacet facet={{FIRST_PROD.facets[0]}} />
        <FirstProductionFacet facet={{FIRST_PROD.facets[1]}} />
        <FirstProductionFacet facet={{FIRST_PROD.facets[2]}} />
        <FirstProductionFacet facet={{FIRST_PROD.facets[3]}} />
      </Grid>

      <Stack gap={{8}}>
        <H2>Kaplan–Meier median first age</H2>
        <Text tone="secondary" size="small">
          Same eight verbs as the plots above. Rank 1 = earliest KM median.
          Medians that are not reached (survival never falls to 50%) are listed
          last. The interval is a 95% bootstrap CI; a lower bound only means
          the median itself was not reached. Producers are children with an
          observed first use (n_event); never-producers are right-censored.
          Source: data/cache/acq_filled_canvas_censored_aoa.csv.
        </Text>
      </Stack>

      <Grid columns={{2}} gap={{16}}>
        <Card>
          <CardHeader>VL intransitive locative</CardHeader>
          <CardBody>
            <Table
              headers={{["Rank", "Verb", "KM median (mo)", "95% CI", "Producers", "Tokens"]}}
              columnAlign={{["right", "left", "right", "right", "right", "right"]}}
              rows={{kmRows(FIRST_PROD.facets[0], VL)}}
              striped
            />
          </CardBody>
        </Card>
        <Card>
          <CardHeader>VOL caused motion</CardHeader>
          <CardBody>
            <Table
              headers={{["Rank", "Verb", "KM median (mo)", "95% CI", "Producers", "Tokens"]}}
              columnAlign={{["right", "left", "right", "right", "right", "right"]}}
              rows={{kmRows(FIRST_PROD.facets[1], VOL)}}
              striped
            />
          </CardBody>
        </Card>
        <Card>
          <CardHeader>VO transitive</CardHeader>
          <CardBody>
            <Table
              headers={{["Rank", "Verb", "KM median (mo)", "95% CI", "Producers", "Tokens"]}}
              columnAlign={{["right", "left", "right", "right", "right", "right"]}}
              rows={{kmRows(FIRST_PROD.facets[2], VO)}}
              striped
            />
          </CardBody>
        </Card>
        <Card>
          <CardHeader>VOO ditransitive</CardHeader>
          <CardBody>
            <Table
              headers={{["Rank", "Verb", "KM median (mo)", "95% CI", "Producers", "Tokens"]}}
              columnAlign={{["right", "left", "right", "right", "right", "right"]}}
              rows={{kmRows(FIRST_PROD.facets[3], VOO)}}
              striped
            />
          </CardBody>
        </Card>
      </Grid>
    </Stack>
  );
}}
'''
    # The template above used doubled braces for TSX. Python f-string already
    # interpolated stats; remaining `{{` become `{`. Write as-is.
    CANVAS.write_text(ts, encoding="utf-8")
    print(f"wrote {CANVAS}", flush=True)


def load_or_build_exposure() -> pd.Series:
    if EXPOSURE.exists():
        df = pd.read_csv(EXPOSURE)
        s = df.set_index(df.columns[0])[df.columns[1]]
        s.index = s.index.astype(int)
        s.index.name = "age_months"
        s.name = "n_tokens"
        return s
    print(f"counting child tokens per age bin from {FILLED}...", flush=True)
    s = child_token_exposure_from_parquet(FILLED)
    s.to_csv(EXPOSURE, header=True)
    print(f"wrote {EXPOSURE} ({int(s.sum()):,} child tokens)", flush=True)
    return s


def build_canvas_tables(
    hits: pd.DataFrame,
    spans: pd.DataFrame,
    order: pd.DataFrame,
    first_ages: pd.DataFrame,
    exposure: pd.Series | None = None,
) -> dict[str, pd.DataFrame]:
    if exposure is None:
        exposure = load_or_build_exposure()
    freq = frequent_schematic_order(order)
    curves = schematic_token_curves(hits, freq, exposure=exposure)
    curves.to_csv(CACHE / "acq_filled_canvas_age_curves.csv", index=False)
    keep = set(freq["item"].astype(str))
    first = first_ages.loc[first_ages["item"].astype(str).isin(keep)]
    times = right_censored_times(first, spans, freq)
    aoa = censored_aoa_summary(times, n_boot=1000)
    freq.to_csv(CACHE / "acq_filled_canvas_order.csv", index=False)
    aoa.to_csv(CACHE / "acq_filled_canvas_censored_aoa.csv", index=False)
    return {
        "child_order": freq,
        "child_age_curves": curves,
        "child_first_ages": first,
        "censored_times": times,
        "censored_aoa": aoa,
    }


def rebuild_curves_only() -> int:
    """Recompute sampling-adjusted CDFs without rerunning Kaplan–Meier."""
    hits = pd.read_parquet(HITS_ALL)
    order = pd.read_csv(CACHE / "acq_filled_child_order.csv")
    meta = json.loads((CACHE / "acq_filled_meta.json").read_text(encoding="utf-8"))
    exposure = load_or_build_exposure()
    freq = frequent_schematic_order(order)
    curves = schematic_token_curves(hits, freq, exposure=exposure)
    curves.to_csv(CACHE / "acq_filled_canvas_age_curves.csv", index=False)
    aoa_path = CACHE / "acq_filled_canvas_censored_aoa.csv"
    aoa = pd.read_csv(aoa_path) if aoa_path.exists() else pd.DataFrame()
    first_path = CACHE / "acq_filled_child_first_ages.csv"
    first = pd.read_csv(first_path) if first_path.exists() else pd.DataFrame()
    tables = {
        "child_order": freq,
        "child_age_curves": curves,
        "child_first_ages": first,
        "censored_times": pd.DataFrame(),
        "censored_aoa": aoa,
    }
    print({"curves": len(curves), "exposure_bins": int(len(exposure))}, flush=True)
    write_canvas(tables, hits, meta, exposure=exposure)
    return 0


def rebuild_canvas_only() -> int:
    hits = pd.read_parquet(HITS_ALL)
    spans = pd.read_parquet(SPANS)
    order = pd.read_csv(CACHE / "acq_filled_child_order.csv")
    first = pd.read_csv(CACHE / "acq_filled_child_first_ages.csv")
    meta = json.loads((CACHE / "acq_filled_meta.json").read_text(encoding="utf-8"))
    exposure = load_or_build_exposure()
    tables = build_canvas_tables(hits, spans, order, first, exposure=exposure)
    print({k: len(v) for k, v in tables.items()}, flush=True)
    print(tables["censored_aoa"].to_string(index=False), flush=True)
    write_canvas(tables, hits, meta, exposure=exposure)
    return 0


def main() -> int:
    if not FILLED.exists():
        print(f"missing {FILLED}", file=sys.stderr)
        return 1
    if HITS_ALL.exists() and SPANS.exists():
        print(f"reusing {HITS_ALL} and {SPANS}", flush=True)
        hits = pd.read_parquet(HITS_ALL)
        spans = pd.read_parquet(SPANS)
        meta = {
            "n_tokens": int(pd.read_parquet(FILLED, columns=["corpus_name"]).shape[0]),
            "n_parsed_tokens": None,
            "n_hits": int(len(hits)),
            "n_children": int(len(spans)),
            "n_corpora": int(hits["corpus_name"].nunique()) if "corpus_name" in hits.columns else 0,
            "corpora": sorted(hits["corpus_name"].dropna().astype(str).unique().tolist())
            if "corpus_name" in hits.columns
            else [],
            "parse_source_tokens": {},
        }
        src = pd.read_parquet(FILLED, columns=["parse_source"])["parse_source"].fillna("unparsed")
        meta["parse_source_tokens"] = src.astype(str).value_counts().to_dict()
        meta["n_tokens"] = int(len(src))
    else:
        hits, spans, meta = extract_all()
    if hits.empty:
        print("no hits", file=sys.stderr)
        return 1
    tables = run_acquisition(hits, observation_spans=spans, cache_prefix="acq_filled")
    print({k: len(v) for k, v in tables.items()}, flush=True)
    canvas_tables = build_canvas_tables(
        hits,
        spans,
        tables["child_order"],
        tables["child_first_ages"],
    )
    write_canvas(canvas_tables, hits, meta)
    (CACHE / "acq_filled_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--curves-only":
        raise SystemExit(rebuild_curves_only())
    if len(sys.argv) > 1 and sys.argv[1] == "--canvas-only":
        raise SystemExit(rebuild_canvas_only())
    raise SystemExit(main())
