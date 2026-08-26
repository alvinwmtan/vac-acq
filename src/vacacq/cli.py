"""CLI for coverage, parse-fill, VAC extract, Chapter 7 stats, QC, and BabyLM scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vacacq import CACHE
from vacacq.childes.access import DB_VERSION, REDIVIS_TAG, redivis_token_present
from vacacq.childes.coverage import audit_coverage
from vacacq.childes.extract import extract_vacs
from vacacq.childes.fetch import load_analysis_tokens
from vacacq.eval.scoring import expand_ranking_items, score_track
from vacacq.parse.fill import fill_unparsed, sanity_check_parsed_sample
from vacacq.qc.gold import extract_ud_treebank, write_qc_report
from vacacq.stats.acquisition import run_acquisition
from vacacq.stats.chapter7 import run_chapter7_stats, verb_token_totals


def _cmd_coverage(_args: argparse.Namespace) -> int:
    df = audit_coverage()
    print(f"childes-db {DB_VERSION} (Redivis {REDIVIS_TAG})")
    print(f"token present: {redivis_token_present()}")
    status = CACHE / "coverage_status.json"
    if status.exists():
        print(status.read_text(encoding="utf-8"))
    if df.empty:
        print("Coverage table is empty. Set REDIVIS_API_TOKEN or cache tokens locally.")
        return 1
    print(df.to_string(index=False))
    return 0


def _cmd_parse_fill(args: argparse.Namespace) -> int:
    tokens = load_analysis_tokens(args.tokens, limit=args.limit, cache=args.cache)
    if tokens.empty:
        print("No tokens to fill. Provide --tokens parquet/csv or Redivis access.")
        return 1
    nlp = None
    if args.sanity:
        import stanza

        nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,lemma,depparse", verbose=False)
        report = sanity_check_parsed_sample(tokens, nlp, n=args.sanity_n)
        print(json.dumps(report, indent=2))
    filled = fill_unparsed(tokens, nlp=nlp, use_stanza=not args.skip_stanza)
    out = Path(args.out) if args.out else CACHE / "tokens_filled.parquet"
    CACHE.mkdir(parents=True, exist_ok=True)
    filled.to_parquet(out, index=False)
    print(f"wrote {out} ({len(filled)} tokens)")
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    tokens = load_analysis_tokens(args.tokens, limit=args.limit)
    if tokens.empty:
        print("No tokens. Provide --tokens or Redivis access.")
        return 1
    if args.subset == "s7_2":
        tokens = tokens.loc[tokens["in_s7_2"]].copy()
    if args.skip_unparsed:
        from vacacq.childes.coverage import drop_unparsed_corpora
        from vacacq.parse.fill import keep_parsed_utterances

        before = len(tokens)
        tokens = drop_unparsed_corpora(tokens, min_rate=args.min_parsed_rate)
        tokens = keep_parsed_utterances(tokens)
        print(f"skip-unparsed: {before} → {len(tokens)} tokens")
    hits = extract_vacs(tokens, extractor=args.extractor)
    out = Path(args.out) if args.out else CACHE / f"hits_{args.extractor}.parquet"
    CACHE.mkdir(parents=True, exist_ok=True)
    hits.to_parquet(out, index=False)
    hits.to_csv(out.with_suffix(".csv"), index=False)
    print(f"wrote {out} ({len(hits)} hits)")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    import pandas as pd

    hits = pd.read_parquet(args.hits) if args.hits.endswith(".parquet") else pd.read_csv(args.hits)
    tokens = None
    verb_totals = None
    if args.tokens:
        tokens = load_analysis_tokens(args.tokens)
        verb_totals = verb_token_totals(tokens)
    tables = run_chapter7_stats(hits, verb_totals=verb_totals, tokens=tokens)
    for name, df in tables.items():
        print(f"{name}: {len(df)} rows")
    return 0


def _cmd_qc(args: argparse.Namespace) -> int:
    import pandas as pd

    occupancy = pd.read_csv(args.occupancy)
    hits = pd.read_csv(args.hits) if args.hits else None
    report = write_qc_report(occupancy, hits)
    print(json.dumps(report, indent=2, default=str))
    if args.ud_conllu:
        tb_hits = extract_ud_treebank(Path(args.ud_conllu))
        tb_hits.to_csv(CACHE / "qc_ud_english_childes_hits.csv", index=False)
        print(f"UD-English-CHILDES hits: {len(tb_hits)}")
    return 0


def _curve_corpora(args: argparse.Namespace) -> list[str]:
    from vacacq.childes.coverage import parsed_corpus_names

    if args.corpora:
        names = list(args.corpora)
    elif args.subset == "all":
        names = sorted(parsed_corpus_names(min_rate=args.min_parsed_rate) or [])
        if not names:
            raise SystemExit("No parsed corpora in coverage. Run `vacacq coverage` first.")
    else:
        names = ["Brown", "Manchester", "Wells", "Hall", "Belfast", "Brent", "NewEngland"]
    if args.skip_unparsed:
        allowed = parsed_corpus_names(min_rate=args.min_parsed_rate)
        if allowed:
            skipped = [c for c in names if c not in allowed]
            names = [c for c in names if c in allowed]
            if skipped:
                print(f"skipping unparsed corpora: {skipped}")
    cov_path = CACHE / "coverage.csv"
    if cov_path.exists() and names:
        import pandas as pd

        cov = pd.read_csv(cov_path)
        size = {str(r.corpus_name): int(r.n_tokens) for r in cov.itertuples()}
        names = sorted(names, key=lambda c: size.get(c, 0))
    return names


def _extract_corpora(
    corpora: list[str],
    *,
    subset: str,
    skip_unparsed: bool,
    min_parsed_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch, extract, and cache per corpus. Returns (hits, observation_spans)."""
    import pandas as pd

    from vacacq.childes.coverage import drop_unparsed_corpora
    from vacacq.childes.extract import extract_vacs
    from vacacq.childes.fetch import load_corpora_tokens
    from vacacq.childes.strata import CHILD_ROLES, PARENT_ROLES
    from vacacq.parse.fill import keep_parsed_utterances
    from vacacq.stats.acquisition import child_observation_spans

    hit_frames: list[pd.DataFrame] = []
    span_frames: list[pd.DataFrame] = []
    roles = tuple(sorted(CHILD_ROLES | PARENT_ROLES))
    for corpus in corpora:
        hits_path = CACHE / f"hits_{corpus}.parquet"
        try:
            tok = load_corpora_tokens([corpus], roles=roles)
        except RuntimeError as exc:
            print(f"{corpus}: {exc}")
            continue
        if tok.empty:
            print(f"{corpus}: no tokens")
            continue
        if subset == "s7_2" and "in_s7_2" in tok.columns:
            tok = tok.loc[tok["in_s7_2"]].copy()
        if skip_unparsed:
            tok = drop_unparsed_corpora(tok, min_rate=min_parsed_rate)
            tok = keep_parsed_utterances(tok)
        if tok.empty:
            print(f"{corpus}: no parsed tokens after filters")
            continue
        if hits_path.exists():
            hits = pd.read_parquet(hits_path)
            print(f"cache hit hits_{corpus}.parquet ({len(hits)} hits)")
        else:
            print(f"extracting {corpus} ({len(tok)} tokens) …", flush=True)
            hits = extract_vacs(tok, extractor="ud", extras=False)
            CACHE.mkdir(parents=True, exist_ok=True)
            hits.to_parquet(hits_path, index=False)
            print(f"wrote {hits_path} ({len(hits)} hits)", flush=True)
        if not hits.empty:
            hit_frames.append(hits)
        child_tok = tok.loc[tok["speaker_role"].isin(CHILD_ROLES)] if "speaker_role" in tok.columns else tok
        spans = child_observation_spans(child_tok)
        if not spans.empty:
            span_frames.append(spans)
        del tok
    hits = pd.concat(hit_frames, ignore_index=True) if hit_frames else pd.DataFrame()
    spans = pd.concat(span_frames, ignore_index=True) if span_frames else pd.DataFrame()
    if not spans.empty and "child_id" in spans.columns:
        spans = (
            spans.groupby("child_id", dropna=False)
            .agg(
                first_obs=("first_obs", "min"),
                last_obs=("last_obs", "max"),
                **({"child_name": ("child_name", "first")} if "child_name" in spans.columns else {}),
            )
            .reset_index()
        )
    return hits, spans


def _cmd_curves(args: argparse.Namespace) -> int:
    import pandas as pd

    if args.hits:
        hits = pd.read_parquet(args.hits) if args.hits.endswith(".parquet") else pd.read_csv(args.hits)
        corpora = _curve_corpora(args) if args.corpora else (
            sorted(hits["corpus_name"].dropna().astype(str).unique()) if "corpus_name" in hits.columns else []
        )
        _, spans = _extract_corpora(
            [c for c in corpora if (CACHE / f"tokens_{c}.parquet").exists()],
            subset=args.subset,
            skip_unparsed=args.skip_unparsed,
            min_parsed_rate=args.min_parsed_rate,
        )
        print(f"hits: {len(hits)}; observation spans: {len(spans)} children")
    else:
        corpora = _curve_corpora(args)
        print(f"corpora ({len(corpora)}): {', '.join(corpora)}")
        hits, spans = _extract_corpora(
            corpora,
            subset=args.subset,
            skip_unparsed=args.skip_unparsed,
            min_parsed_rate=args.min_parsed_rate,
        )
        if hits.empty:
            print("No tokens/hits. Provide --hits or Redivis access.")
            return 1
        out_hits = CACHE / ("hits_curves_all.parquet" if args.subset == "all" else "hits_curves.parquet")
        CACHE.mkdir(parents=True, exist_ok=True)
        hits.to_parquet(out_hits, index=False)
        print(f"extracted {len(hits)} hits from {len(corpora)} corpora → {out_hits}")
        print(f"observation spans: {len(spans)} children")

    ranking = None
    if args.ranking:
        ranking = pd.read_csv(args.ranking)
    elif args.score_models:
        frames = []
        if args.gpt2_epochs:
            out = score_track("gpt2_100m", epochs=args.gpt2_epochs)
            frames.append(out["ranking"])
        if args.distilbert_epochs:
            out = score_track("distilbert_100m", epochs=args.distilbert_epochs)
            frames.append(out["ranking"])
        ranking = pd.concat(frames, ignore_index=True) if frames else None

    prefix = "acq"
    if args.subset == "all":
        prefix = "acq_all_parsed" if args.skip_unparsed else "acq_all"
    elif args.skip_unparsed:
        prefix = "acq_parsed"
    tables = run_acquisition(
        hits,
        ranking,
        bin_months=args.bin_months,
        observation_spans=spans,
        cache_prefix=prefix,
    )
    for name, df in tables.items():
        print(f"{name}: {len(df)} rows")
    curves = tables.get("child_age_curves")
    if curves is not None and not curves.empty:
        schematic = sorted(v for v in curves["vac"].unique() if v in ("VL", "VOL", "VO", "VOO"))
        print(f"schematic age curves: {schematic}")
        missing = [v for v in ("VL", "VOL", "VO", "VOO") if v not in set(curves["vac"])]
        if missing:
            print(f"warning: no child age curves for {missing} (no hits in this sample)")
    models = tables.get("model_curves")
    if models is not None and not models.empty:
        schematic_m = sorted(v for v in models["vac"].unique() if v in ("VL", "VOL", "VO", "VOO"))
        print(f"schematic model curves: {schematic_m}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    import pandas as pd

    occupancy = None
    if args.occupancy:
        occupancy = pd.read_csv(args.occupancy)
    out = score_track(args.track, epochs=args.epochs, occupancy=occupancy, dry_run=args.dry_run)
    for name, df in out.items():
        print(f"{name}: {len(df)} rows")
    if args.dry_run:
        items = expand_ranking_items()
        print(f"ranking item expansions: {len(items)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vacacq", description="VAC acquisition: CHILDES Ch.7 + BabyLM probes")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("coverage", help="Audit UD %mor/%gra coverage on childes-db 2026.1")
    c.set_defaults(func=_cmd_coverage)

    pf = sub.add_parser("parse-fill", help="Silver-parse fully unparsed utterances")
    pf.add_argument("--tokens", help="Local parquet/csv of tokens")
    pf.add_argument("--limit", type=int)
    pf.add_argument("--cache", action="store_true")
    pf.add_argument("--out")
    pf.add_argument("--skip-stanza", action="store_true")
    pf.add_argument("--sanity", action="store_true", help="POS agreement vs 2026.1 on a parsed sample")
    pf.add_argument("--sanity-n", type=int, default=25)
    pf.set_defaults(func=_cmd_parse_fill)

    ex = sub.add_parser("extract", help="Extract VAC instances")
    ex.add_argument("--tokens", help="Local parquet/csv of tokens")
    ex.add_argument("--limit", type=int)
    ex.add_argument("--extractor", choices=("ud", "s74"), default="ud")
    ex.add_argument("--subset", choices=("all", "s7_2"), default="s7_2")
    ex.add_argument("--out")
    ex.add_argument(
        "--skip-unparsed",
        action="store_true",
        help="Drop zero-parse corpora and fully unparsed utterances",
    )
    ex.add_argument("--min-parsed-rate", type=float, default=0.0)
    ex.set_defaults(func=_cmd_extract)

    st = sub.add_parser("stats", help="Chapter 7 Zipf / ΔP / parent–child stats")
    st.add_argument("--hits", required=True)
    st.add_argument("--tokens", help="Token table for verb totals and MLU-w")
    st.set_defaults(func=_cmd_stats)

    qc = sub.add_parser("qc", help="Lead-verb check and hand-coding sample")
    qc.add_argument("--occupancy", required=True)
    qc.add_argument("--hits")
    qc.add_argument("--ud-conllu", help="Path to UD_English-CHILDES CoNLL-U")
    qc.set_defaults(func=_cmd_qc)

    ev = sub.add_parser("eval", help="Score BabyLM ranking + affinity")
    ev.add_argument("--track", choices=("gpt2_100m", "distilbert_100m"), default="gpt2_100m")
    ev.add_argument("--epochs", type=int, nargs="*")
    ev.add_argument("--occupancy", help="CSV of Verb–VAC frequency / ΔP (training analog)")
    ev.add_argument("--dry-run", action="store_true")
    ev.set_defaults(func=_cmd_eval)

    cu = sub.add_parser(
        "curves",
        help="Child age and model epoch curves for VL, VOL, VO, and VOO",
    )
    cu.add_argument("--hits", help="Existing VAC hits parquet/csv (skips extract)")
    cu.add_argument(
        "--subset",
        choices=("s7_2", "all"),
        default="s7_2",
        help="s7_2: Ellis Table S7.2 files only. all: every Eng-NA/Eng-UK corpus listed in --corpora",
    )
    cu.add_argument(
        "--corpora",
        nargs="*",
        default=None,
        help="Corpora to fetch. Default: S7.2 starter list, or all parsed corpora with --subset all",
    )
    cu.add_argument("--ranking", help="BabyLM ranking CSV from `vacacq eval`")
    cu.add_argument("--score-models", action="store_true", help="Score GPT-2 and DistilBERT epoch subsets")
    cu.add_argument("--gpt2-epochs", type=int, nargs="*", default=[1, 6, 12, 18, 24])
    cu.add_argument("--distilbert-epochs", type=int, nargs="*", default=[1, 15, 30, 45, 60])
    cu.add_argument("--bin-months", type=float, default=3.0)
    cu.add_argument(
        "--skip-unparsed",
        action="store_true",
        help="Drop zero-parse corpora and fully unparsed utterances",
    )
    cu.add_argument(
        "--min-parsed-rate",
        type=float,
        default=0.0,
        help="With --skip-unparsed, keep corpora with parsed_rate above this (default: skip only 0%%)",
    )
    cu.set_defaults(func=_cmd_curves)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    raise SystemExit(args.func(args))
