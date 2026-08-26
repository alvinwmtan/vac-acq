# VAC acquisition (`vacacq`)

Reproduce Ellis, Römer & O'Donnell (2016), Chapter 7 — *VACs in Parent and Child Language* — on **childes-db 2026.1**, then ask whether Shah 2023 BabyLM 100M models acquire the same verb–argument constructions (VACs).

The usage-based claim under test: children learn constructions from caregiver input statistics of **frequency**, **contingency** (ΔP<sub>cw</sub>), and **semantic prototypicality** (WordNet betweenness). We re-test that on Universal Dependencies parses, then compare **child age-of-acquisition order** with **model epoch order** at the verb and verb+preposition grain (schematic **VL, VOL, VO, VOO**).

Environment is managed with **[uv](https://docs.astral.sh/uv/)** (not pip/venv).

## Objectives

1. **Conceptual replication of Ch. 7** on Eng-NA + Eng-UK in childes-db **2026.1** (Redivis `v1.4`), not a classic-MOR / GRASP rerun.
2. **Fill unparsed English** with silver UD (`parse_source ∈ {childesdb, batchalign2, stanza_gloss}`); never overwrite 2026.1 tags. Retraces and `xxx`/`yyy` stay empty.
3. **Extract VACs** with a UD-native query (primary) and the original S7.4 linear windows (sensitivity only).
4. **Re-estimate** Zipf, selectivity `1 − τ²`, ΔP<sub>cw</sub>, WordNet betweenness, parent→child regressions, and cumulative use vs **MLU-w** (not MLU-m).
5. **Probe BabyLM 100M** GPT-2 and DistilBERT checkpoints with evals **(1) verb-in-frame ranking** and **(4) frame vs control affinity** only.
6. **Acquisition curves:** child age (months) and model epoch, including schematic **VO** and **VOO** as well as locative VL/VOL and prep-specific variants.

Pinned data: childes-db **2026.1** / Redivis `datapages.childes_db` **v1.4**. Inventories live in `data/` (S7.2 corpora, S7.3 prepositions, S7.5 ditransitives, BabyLM checkpoint IDs, eval items).

## Phases

### Phase A — CHILDES (childes-db 2026.1)

| Step | Goal |
|---|---|
| Coverage | Audit UD `%mor`/`%gra` by Eng-NA/Eng-UK corpus; S7.2 vs new 2026.1 corpora |
| Parse-fill | Batchalign2 `morphotag` on CHAT if available; else Stanza on the utterance gloss |
| Extract | VL / VOL / VOO / VO (+ extras); questions and passives excluded; VOO restricted to S7.5 |
| QC | Lead verbs vs S7.7–S7.12 on the S7.2 subset; optional UD-English-CHILDES CoNLL-U; hand-check sample |
| Stats | Zipf, ΔP<sub>cw</sub>, Kendall selectivity, WordNet betweenness, parent→child `lm`, cumulative vs MLU-w; split by `parse_source` |
| Curves | Median first age and cumulative token share by age bin, **verb-level for VO/VOO** and **verb+prep for VL/VOL** |

**Strata:** Child I 12–36 months (Ch. 7 started at 18); Child II 36–72; Parent Young = CDS to children &lt; 37 months; Parent All = CDS to children &lt; 72 months (Parent All includes Parent Young). Analysis tokens are **12–72 months**. Ages on Redivis are stored in **days**; the pipeline converts with `days / (365.25/12)`.

**Primary extractor (UD-native):** locative `obl`/`obl:lmod` or `compound:prt` with an S7.3 preposition; `obj`/`iobj` for VOL/VO/VOO. POS is lowercase UD (`verb`, `adp`); `%gra` is CHAT-style (`OBJ`, `COMPOUND-PRT`) — both are normalized.

### Phase B — BabyLM 100M

Checkpoints (epoch = developmental time):

- Causal: `Raj-Sanjay-Shah/babyLM_100M_gpt2_epoch-{1…24}`
- Masked: `Raj-Sanjay-Shah/babyLM_100M_distilbert_epoch-{1…60}`

The training mix is not CHILDES alone. Occupancy/ΔP should be measured on the 100M corpus when available; Parent All is only a secondary analog.

| Eval | What is scored |
|---|---|
| (1) Ranking | Verb slot in frozen frames; Spearman ρ vs training Verb–VAC frequency and ΔP<sub>cw</sub> |
| (4) Affinity | P(verb \| VAC) − P(verb \| control frame); should be &gt; 0 and track corpus ΔP |

A checkpoint has **acquired** construction *C* if ranking ρ vs occupancy is significantly positive **and** mean affinity is significantly &gt; 0. Minimal pairs, nonce/spugged items, and prototype-echo (evals 2, 3, 5) are out of the first pass.

## Setup

```bash
uv sync --dev
export REDIVIS_API_TOKEN=...    # required for live 2026.1 queries
uv run vacacq --help
uv run pytest
```

Outputs go to `data/cache/` (gitignored). Local token parquets skip Redivis.

## CLI

All commands: `uv run vacacq <command>`.

### Phase A

```bash
# 1. Parse coverage by English corpus (S7.2 vs new)
uv run vacacq coverage

# 2. Silver-parse fully unparsed utterances (never overwrites 2026.1 tags)
uv run vacacq parse-fill --tokens data/cache/tokens_eng.parquet
uv run vacacq parse-fill --tokens data/cache/tokens_eng.parquet --sanity   # POS agreement vs stored tags
uv run vacacq parse-fill --skip-stanza   # skip Stanza fallback

# 3. Extract VAC instances
uv run vacacq extract --subset s7_2 --extractor ud
uv run vacacq extract --subset all --extractor ud --tokens data/cache/tokens_Brown.parquet
uv run vacacq extract --extractor s74          # S7.4 linear windows (sensitivity only)
uv run vacacq extract --limit 5000             # Redivis smoke test

# 4. Chapter 7 statistics
uv run vacacq stats --hits data/cache/hits_ud.parquet --tokens data/cache/tokens_eng.parquet

# 5. Lead-verb QC + hand-coding sample
uv run vacacq qc --occupancy data/cache/stats_occupancy.csv --hits data/cache/hits_ud.csv
uv run vacacq qc --occupancy data/cache/stats_occupancy.csv --ud-conllu path/to/en_childes-ud-test.conllu
```

### Phase B

```bash
# Expand the frozen ranking/affinity item bank (no model download)
uv run vacacq eval --dry-run

# Score one track / selected epochs
uv run vacacq eval --track gpt2_100m --epochs 1 12 24 --occupancy data/cache/stats_occupancy.csv
uv run vacacq eval --track distilbert_100m --epochs 1 30 60
```

### Acquisition curves (children + models)

Schematic **VL, VOL, VO, VOO** are always included. Locatives also get verb+preposition curves (e.g. `VL-to`, `VOL-off`).

```bash
# Child age curves from listed S7.2 corpora (fetches Redivis if not cached)
uv run vacacq curves --corpora Brown Wells Belfast --bin-months 3

# Reuse extracted hits
uv run vacacq curves --hits data/cache/hits_curves.parquet --bin-months 3

# Same, but skip zero-parse corpora and fully unparsed utterances
uv run vacacq curves --hits data/cache/hits_curves.parquet --skip-unparsed
# Optional: also drop sparse parses (Cruttenden is ~9%)
# uv run vacacq curves --hits data/cache/hits_curves.parquet --skip-unparsed --min-parsed-rate 0.5

# Child curves + score model epoch subsets
uv run vacacq curves --hits data/cache/hits_curves.parquet --score-models \
  --gpt2-epochs 1 6 12 18 24 \
  --distilbert-epochs 1 15 30 45 60

# Child curves + already-scored ranking CSV
uv run vacacq curves --hits data/cache/hits_curves.parquet \
  --ranking data/cache/babylm_gpt2_100m_ranking.csv
```

Tables written under `data/cache/`: `acq_child_order.csv`, `acq_child_age_curves.csv`, `acq_parent_order.csv`, and (if models were scored) `acq_model_order.csv`, `acq_model_curves.csv`. With `--skip-unparsed` the same tables are written as `acq_parsed_*.csv`.

### Common flags

| Flag | Commands | Meaning |
|---|---|---|
| `--tokens` | parse-fill, extract, stats | Local parquet/csv instead of Redivis |
| `--limit N` | parse-fill, extract | Cap Redivis rows (debug) |
| `--subset s7_2\|all` | extract | 2013-like corpora vs all Eng-NA/Eng-UK |
| `--extractor ud\|s74` | extract | UD-native (default) vs linear-window check |
| `--out PATH` | parse-fill, extract | Output parquet |
| `--track` | eval | `gpt2_100m` or `distilbert_100m` |
| `--epochs` | eval | Checkpoint numbers to score |
| `--dry-run` | eval | Item expansion only |
| `--hits` | stats, qc, curves | VAC instance table |
| `--occupancy` | qc, eval | Verb–VAC frequency / ΔP table |
| `--corpora` | curves | Corpus names to fetch when `--hits` is omitted |
| `--bin-months` | curves | Child age bin width (default 3) |
| `--score-models` | curves | Run GPT-2 and DistilBERT epoch loops |

## Data files

| Path | Contents |
|---|---|
| `data/s7_2_included_corpora.json` | Ch. 7 corpus list, filename exclusions, 2026.1 aliases (`Bloom`) |
| `data/s7_3_prepositions.json` | Included / excluded locative prepositions |
| `data/s7_5_ditransitive.json` | VOO verb list (including original no-hit types) |
| `data/expected_leads.json` | S7.7–S7.12 lead verbs (COME/GO, PUT, GET, GIVE, …) |
| `data/checkpoints.json` | BabyLM 100M GPT-2 and DistilBERT epoch IDs |
| `data/eval/ranking_frames.jsonl` | Verb-in-frame templates |
| `data/eval/affinity_items.jsonl` | Grammatical vs control frames |
| `data/eval/verb_sets.json` | Occupants and frequency-matched distractors |

## Implementation notes

- **childesr MySQL 2021.1** is not used. All live access is Python `redivis` against 2026.1.
- Cornell and Korman from Table S7.2 are missing in 2026.1; Bloom70/Bloom73 are stored as `Bloom`.
- 2026.1 `suffix` is CHAT-style (`Ger S`), not a UD feature chain; gerunds are detected via `Ger` in `suffix`.
- Stats and curves should be reported pooled and split by `parse_source`.
- BabyLM occupancy belongs on the 100M training mix; Parent All ΔP is a fallback analog only.
