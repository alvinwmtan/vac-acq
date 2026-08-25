import numpy as np
import pandas as pd
import pytest

from fixtures import tokens
from vacacq.childes.strata import assign_strata, apply_s7_2_exclusions, is_s7_2_corpus
from vacacq.eval.scoring import expand_ranking_items, spearman_with_ci
from vacacq.parse.fill import annotate_existing_parses, fill_unparsed, utterance_fully_unparsed
from vacacq.stats.chapter7 import explode_strata, zipf_fit


def test_parent_all_includes_parent_young():
    assert assign_strata("Mother", 24.0) == ["Parent Young", "Parent All"]
    assert assign_strata("Target_Child", 24.0) == ["Child I"]
    assert assign_strata("Target_Child", 48.0) == ["Child II"]


def test_age_to_months_days():
    from vacacq.childes.strata import DAYS_PER_MONTH, age_to_months

    assert age_to_months(24.0) == 24.0
    assert abs(age_to_months(18 * DAYS_PER_MONTH) - 18.0) < 1e-6


def test_s7_2_exclusions():
    assert is_s7_2_corpus("Brown")
    assert is_s7_2_corpus("Bloom")
    assert not is_s7_2_corpus("Providence")
    tr = pd.DataFrame({"filename": ["Bernstein/Interview/foo.cha", "Brown/Adam/adam01.cha"]})
    kept = apply_s7_2_exclusions(tr, stratum="child")
    assert list(kept["filename"]) == ["Brown/Adam/adam01.cha"]


def test_explode_parent_strata():
    hits = pd.DataFrame(
        {
            "verb": ["go"],
            "construction": ["VL"],
            "preposition": ["to"],
            "speaker_role": ["Mother"],
            "target_child_age": [24.0],
            "stratum": ["Parent Young"],
            "extra": [False],
            "parse_source": ["childesdb"],
        }
    )
    out = explode_strata(hits)
    assert set(out["stratum"]) == {"Parent Young", "Parent All"}


def test_zipf_perfect_series():
    # Zipf: f = C / rank  → log f = log C - 1 * log rank, R² = 1
    rank = pd.Series([1000 / r for r in range(1, 21)])
    fit = zipf_fit(rank)
    assert fit["r2"] > 0.999
    assert abs(fit["gamma"] + 1) < 0.05


def test_ranking_item_bank_covers_schematic_vacs():
    items = expand_ranking_items()
    assert set(items["construction"]) == {"VL", "VOL", "VOO", "VO"}
    assert {"occupant", "distractor"} <= set(items["kind"])
    assert len(items) > 50


def test_spearman_ci_positive():
    import numpy as np

    x = np.arange(10, dtype=float)
    stats = spearman_with_ci(x, x, n_boot=200)
    assert stats["rho"] == pytest.approx(1.0)
    assert stats["ci_low"] > 0.5


def test_parse_fill_skips_retraces_and_does_not_overwrite():
    parsed = tokens(
        [
            {"gloss": "she", "part_of_speech": "PRON", "gra_head": 2, "gra_relation": "nsubj"},
            {"gloss": "go", "part_of_speech": "VERB", "gra_head": 0, "gra_relation": "root"},
            {"gloss": "home", "part_of_speech": "", "gra_head": None, "gra_relation": ""},
        ],
        utterance_id="partial",
    )
    parsed.loc[parsed["gloss"] == "home", "part_of_speech"] = pd.NA
    assert not utterance_fully_unparsed(parsed)
    marked = annotate_existing_parses(parsed)
    filled = fill_unparsed(marked, use_stanza=False)
    assert filled.loc[filled["gloss"] == "go", "parse_source"].iloc[0] == "childesdb"
    assert pd.isna(filled.loc[filled["gloss"] == "home", "part_of_speech"].iloc[0]) or filled.loc[
        filled["gloss"] == "home", "part_of_speech"
    ].iloc[0] in {"", None}


def test_parse_fill_writes_stanza_on_fully_unparsed(monkeypatch):
    unparsed = tokens(
        [
            {"gloss": "she", "part_of_speech": "", "gra_head": None, "gra_relation": ""},
            {"gloss": "go", "part_of_speech": "", "gra_head": None, "gra_relation": ""},
            {"gloss": "home", "part_of_speech": "", "gra_head": None, "gra_relation": ""},
        ],
        utterance_id="gap1",
    )
    for col in ("part_of_speech", "gra_relation", "stem"):
        unparsed[col] = pd.NA

    class Word:
        def __init__(self, text, upos, wid, head, deprel, lemma):
            self.text = text
            self.upos = upos
            self.id = wid
            self.head = head
            self.deprel = deprel
            self.lemma = lemma

    class Sent:
        words = [
            Word("she", "PRON", 1, 2, "nsubj", "she"),
            Word("go", "VERB", 2, 0, "root", "go"),
            Word("home", "ADV", 3, 2, "advmod", "home"),
        ]

    class Doc:
        sentences = [Sent()]

    filled = fill_unparsed(unparsed, nlp=lambda text: Doc(), use_stanza=True)
    assert filled.loc[filled["gloss"] == "go", "part_of_speech"].iloc[0] == "VERB"
    assert filled.loc[filled["gloss"] == "go", "parse_source"].iloc[0] == "stanza_gloss"


def test_child_acquisition_order_and_age_curve():
    from vacacq.stats.acquisition import acquisition_order, child_first_ages, cumulative_age_curves, model_acquisition_order, model_verb_curves

    rows = []
    for child, go_age, come_age in (("c1", 20.0, 30.0), ("c2", 22.0, 33.0), ("c3", 21.0, 28.0),
                                   ("c4", 24.0, 36.0), ("c5", 19.0, 31.0)):
        rows.append(
            {
                "utterance_id": f"{child}-go",
                "verb": "go",
                "construction": "VL",
                "preposition": "to",
                "extra": False,
                "speaker_role": "Target_Child",
                "target_child_age": go_age,
                "target_child_id": child,
            }
        )
        rows.append(
            {
                "utterance_id": f"{child}-come",
                "verb": "come",
                "construction": "VL",
                "preposition": "to",
                "extra": False,
                "speaker_role": "Target_Child",
                "target_child_age": come_age,
                "target_child_id": child,
            }
        )
    hits = pd.DataFrame(rows)
    order = acquisition_order(child_first_ages(hits), min_children=5)
    ranked = order.loc[order["vac"] == "VL-to"].sort_values("aoa_rank")
    assert list(ranked["verb"]) == ["go", "come"]
    assert "VL" in set(order["vac"])
    curves = cumulative_age_curves(hits, bin_months=3, order=order)
    go = curves.loc[(curves["verb"] == "go") & (curves["vac"] == "VL-to")].sort_values("age_months")
    come = curves.loc[(curves["verb"] == "come") & (curves["vac"] == "VL-to")].sort_values("age_months")
    mid = 24
    assert float(go.loc[go["age_months"] == mid, "cumulative_share"].iloc[0]) > float(
        come.loc[come["age_months"] == mid, "cumulative_share"].iloc[0]
    )


def test_vo_and_voo_are_in_age_curves():
    from vacacq.stats.acquisition import acquisition_order, child_first_ages, cumulative_age_curves

    rows = []
    for child in ("c1", "c2", "c3", "c4", "c5"):
        rows.append(
            {
                "utterance_id": f"{child}-get",
                "verb": "get",
                "construction": "VO",
                "preposition": None,
                "extra": False,
                "speaker_role": "Target_Child",
                "target_child_age": 22.0,
                "target_child_id": child,
            }
        )
        rows.append(
            {
                "utterance_id": f"{child}-give",
                "verb": "give",
                "construction": "VOO",
                "preposition": None,
                "extra": False,
                "speaker_role": "Target_Child",
                "target_child_age": 34.0,
                "target_child_id": child,
            }
        )
        rows.append(
            {
                "utterance_id": f"{child}-eat",
                "verb": "eat",
                "construction": "VO",
                "preposition": None,
                "extra": False,
                "speaker_role": "Target_Child",
                "target_child_age": 28.0,
                "target_child_id": child,
            }
        )
    hits = pd.DataFrame(rows)
    order = acquisition_order(child_first_ages(hits), min_children=5)
    assert {"VO", "VOO"} <= set(order["vac"])
    curves = cumulative_age_curves(hits, bin_months=3, order=order)
    assert {"VO", "VOO"} <= set(curves["vac"])
    vo = order.loc[order["vac"] == "VO"].sort_values("aoa_rank")
    assert list(vo["verb"])[:2] == ["get", "eat"]


def test_schematic_vl_pools_prepositions():
    from vacacq.stats.acquisition import acquisition_order, child_first_ages, cumulative_age_curves

    rows = []
    for child in ("c1", "c2", "c3", "c4", "c5"):
        for prep, age in (("to", 24.0), ("off", 30.0)):
            rows.append(
                {
                    "utterance_id": f"{child}-go-{prep}",
                    "verb": "go",
                    "construction": "VL",
                    "preposition": prep,
                    "extra": False,
                    "speaker_role": "Target_Child",
                    "target_child_age": age,
                    "target_child_id": child,
                }
            )
    hits = pd.DataFrame(rows)
    first = child_first_ages(hits)
    schematic = first.loc[first["vac"] == "VL"]
    assert len(schematic) == 5
    assert (schematic["first_age"] == 24.0).all()
    order = acquisition_order(first, min_children=5)
    assert list(order.loc[order["vac"] == "VL", "verb"]) == ["go"]
    curves = cumulative_age_curves(hits, bin_months=3, order=order)
    assert "VL" in set(curves["vac"])
    assert "VL-to" in set(curves["vac"])
    assert "VOO" not in set(curves["vac"])


def test_model_acquire_epoch_order():
    from vacacq.stats.acquisition import model_acquisition_order, model_verb_curves

    rows = []
    for epoch in (1, 6, 12):
        rows.append(
            {
                "track": "gpt2_100m",
                "epoch": epoch,
                "construction": "VOO",
                "preposition": None,
                "verb": "give",
                "kind": "occupant",
                "score": -4.0 + epoch * 0.4,
            }
        )
        rows.append(
            {
                "track": "gpt2_100m",
                "epoch": epoch,
                "construction": "VOO",
                "preposition": None,
                "verb": "tell",
                "kind": "occupant",
                "score": -3.0 + epoch * 0.05,
            }
        )
        rows.append(
            {
                "track": "gpt2_100m",
                "epoch": epoch,
                "construction": "VOO",
                "preposition": None,
                "verb": "sleep",
                "kind": "distractor",
                "score": -3.2,
            }
        )
    curves = model_verb_curves(pd.DataFrame(rows))
    assert "VOO" in set(curves["vac"])
    order = model_acquisition_order(curves)
    give = order.loc[order["verb"] == "give"].iloc[0]
    tell = order.loc[order["verb"] == "tell"].iloc[0]
    assert give["acquire_epoch"] == 6
    assert tell["epoch_rank"] < give["epoch_rank"] or tell["acquire_epoch"] < give["acquire_epoch"]


def test_km_median_is_later_than_naive_when_nonproducers_are_followed():
    from vacacq.stats.acquisition import kaplan_meier_median

    # 4 produce at 20, 2 at 40; 4 never produce and are followed to 60.
    time = np.array([20, 20, 20, 20, 40, 40, 60, 60, 60, 60], dtype=float)
    event = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    naive = float(np.median(time[event == 1]))
    km = kaplan_meier_median(time, event)
    assert naive == 20.0
    assert km == 40.0


def test_right_censored_times_mark_nonproducers():
    from vacacq.stats.acquisition import right_censored_times

    first = pd.DataFrame(
        {
            "child_id": ["a", "b"],
            "vac": ["VO", "VO"],
            "verb": ["get", "get"],
            "item": ["VO|get", "VO|get"],
            "first_age": [22.0, 30.0],
        }
    )
    spans = pd.DataFrame(
        {
            "child_id": ["a", "b", "c"],
            "last_obs": [40.0, 36.0, 48.0],
        }
    )
    items = pd.DataFrame({"vac": ["VO"], "verb": ["get"], "item": ["VO|get"]})
    times = right_censored_times(first, spans, items)
    assert len(times) == 3
    censored = times.loc[times["child_id"] == "c"].iloc[0]
    assert censored["event"] == 0
    assert censored["time"] == 48.0
    assert int(times["event"].sum()) == 2
