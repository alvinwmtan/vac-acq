import numpy as np
import pandas as pd

from vacacq.eval.test_set import (
    contains_xxx,
    eval_utterance_ok,
    has_proper_noun,
    inflect_like,
    is_well_formed_sentence,
    pick_alt_forms,
    split_at_verb,
    strong_frame_pairs,
    stratified_hits,
    top_frame_verbs,
    weak_frame_lemmas,
)


def test_stratified_hits_covers_constructions_and_verbs():
    rows = []
    for cons, verbs in (
        ("VL", ["go", "come", "fall"]),
        ("VOL", ["put", "take"]),
        ("VO", ["want", "get", "see"]),
        ("VOO", ["give", "tell"]),
    ):
        for verb in verbs:
            for i in range(5):
                rows.append(
                    {
                        "utterance_id": f"{cons}-{verb}-{i}",
                        "verb": verb,
                        "construction": cons,
                        "preposition": "in" if cons in {"VL", "VOL"} else None,
                        "corpus_name": "Champaign",
                    }
                )
    hits = pd.DataFrame(rows)
    sample = stratified_hits(hits, n_per_construction=6, seed=1)
    assert set(sample["construction"]) == {"VL", "VOL", "VO", "VOO"}
    assert sample.groupby("construction").size().min() == 6
    vo = sample.loc[sample["construction"] == "VO"]
    assert set(vo["verb"]) == {"want", "get", "see"}
    vl = sample.loc[sample["construction"] == "VL"]
    assert set(vl["verb"]) == {"go", "come", "fall"}
    assert vl.groupby("verb").size().max() - vl.groupby("verb").size().min() <= 1


def test_top_frame_verbs_uses_preferred_frame_not_majority():
    rows = []
    for _ in range(80):
        rows.append({"verb": "put", "construction": "VOL", "speaker_role": "Mother", "extra": False})
    for _ in range(20):
        rows.append({"verb": "put", "construction": "VO", "speaker_role": "Mother", "extra": False})
    for _ in range(45):
        rows.append({"verb": "stick", "construction": "VOL", "speaker_role": "Mother", "extra": False})
    for _ in range(40):
        rows.append({"verb": "stick", "construction": "VO", "speaker_role": "Mother", "extra": False})
    for _ in range(15):
        rows.append({"verb": "stick", "construction": "VL", "speaker_role": "Mother", "extra": False})
    for _ in range(70):
        rows.append({"verb": "want", "construction": "VO", "speaker_role": "Mother", "extra": False})
    for _ in range(20):
        rows.append({"verb": "want", "construction": "VOL", "speaker_role": "Mother", "extra": False})
    for i in range(12):
        lemma = f"vol{chr(ord('a') + i)}"
        for _ in range(30):
            rows.append({"verb": lemma, "construction": "VOL", "speaker_role": "Mother", "extra": False})
        for _ in range(5):
            rows.append({"verb": lemma, "construction": "VO", "speaker_role": "Mother", "extra": False})
    from vacacq.stats.acquisition import adult_frame_share

    share = adult_frame_share(pd.DataFrame(rows), min_tokens=20)
    top = top_frame_verbs(share, n=10)
    assert top["VOL"][:2] == ["put", "stick"]
    assert "want" not in top["VOL"]
    assert len(top["VOL"]) == 10
    rares = [f"vol{chr(ord('a') + i)}" for i in range(12)]
    assert sum(v in top["VOL"] for v in rares) == 8


def test_contains_xxx_is_token_not_substring_of_other_words():
    assert contains_xxx("when she xxx glasses")
    assert contains_xxx("XXX dropping him wagon off")
    assert not contains_xxx("and then he waddled on the rocks")
    assert not contains_xxx("")


def test_strong_frame_pairs_drop_low_affinity_verbs():
    rows = []
    for i in range(20):
        rows.append({"verb": "go", "construction": "VL", "speaker_role": "Mother", "extra": False})
    for i in range(5):
        rows.append({"verb": "go", "construction": "VO", "speaker_role": "Mother", "extra": False})
    for i in range(8):
        rows.append({"verb": "get", "construction": "VO", "speaker_role": "Mother", "extra": False})
    for i in range(8):
        rows.append({"verb": "get", "construction": "VL", "speaker_role": "Mother", "extra": False})
    for i in range(4):
        rows.append({"verb": "get", "construction": "VOL", "speaker_role": "Mother", "extra": False})
    pairs = strong_frame_pairs(pd.DataFrame(rows))
    assert ("VL", "go") in set(zip(pairs["construction"], pairs["verb"]))
    assert ("VL", "get") not in set(zip(pairs["construction"], pairs["verb"]))
    assert ("VO", "get") not in set(zip(pairs["construction"], pairs["verb"]))


def test_split_at_verb_uses_stem_not_substring():
    tok = pd.DataFrame(
        {
            "gloss": ["the", "bread", "that", "goes", "around", "the", "hotdog"],
            "stem": ["the", "bread", "that", "go", "around", "the", "hotdog"],
            "morph_suffix": ["", "", "", "Fin Ind Pres S3", "", "", ""],
            "part_of_speech": ["det", "noun", "sconj", "verb", "adp", "det", "noun"],
        }
    )
    assert split_at_verb(tok, "go") == ("the bread that", "goes", "around the hotdog")


def test_inflect_like_matches_tense_and_case():
    assert inflect_like("goes", "go", "put") == "puts"
    assert inflect_like("went", "go", "give") == "gave"
    assert inflect_like("going", "go", "sit") == "sitting"
    assert inflect_like("gone", "go", "eat") == "eaten"
    assert inflect_like("Look", "look", "want") == "Want"
    assert inflect_like("goes", "go", "put", morph="Fin Ind Pres S3") == "puts"


def test_weak_frame_lemmas_are_under_ten_percent():
    from vacacq.stats.acquisition import adult_frame_share

    rows = []
    for _ in range(20):
        rows.append({"verb": "go", "construction": "VL", "speaker_role": "Mother", "extra": False})
    rows.append({"verb": "go", "construction": "VO", "speaker_role": "Mother", "extra": False})
    for _ in range(20):
        rows.append({"verb": "want", "construction": "VO", "speaker_role": "Mother", "extra": False})
    share = adult_frame_share(pd.DataFrame(rows), min_tokens=10)
    weak = weak_frame_lemmas(share)
    assert "want" in set(weak["VL"]["verb"])
    assert "go" not in set(weak["VL"]["verb"])
    assert "go" in set(weak["VO"]["verb"])
    assert "want" not in set(weak["VO"]["verb"])


def test_pick_alt_forms_inflects_and_skips_source():
    pool = pd.DataFrame(
        {"verb": ["put", "have", "do", "see", "make", "go"], "n_verb": [100, 90, 80, 70, 60, 50]}
    )
    alts = pick_alt_forms(pool, "go", "goes", "Fin Ind Pres S3", np.random.default_rng(0))
    assert len(alts) == 5
    assert all(a.endswith("s") for a in alts)
    assert "goes" not in {a.lower() for a in alts}


def _utt(gloss, pos, rel, utt_type="declarative"):
    n = len(gloss)
    return pd.DataFrame(
        {
            "gloss": gloss,
            "stem": gloss,
            "part_of_speech": pos,
            "gra_relation": rel,
            "utterance_type": [utt_type] * n,
            "morph_suffix": [""] * n,
        }
    )


def test_has_proper_noun_from_pos():
    assert has_proper_noun(_utt(["Peter", "falls"], ["propn", "verb"], ["nsubj", "root"]))
    assert has_proper_noun(_utt(["Robin", "fell"], ["n:prop", "verb"], ["nsubj", "root"]))
    assert not has_proper_noun(_utt(["he", "fell"], ["pron", "verb"], ["nsubj", "root"]))


def test_well_formed_keeps_simple_declaratives():
    tok = _utt(
        ["he", "slid", "off", "the", "roof"],
        ["pron", "verb", "adp", "det", "noun"],
        ["nsubj", "root", "case", "det", "obl"],
    )
    assert is_well_formed_sentence(tok)
    assert eval_utterance_ok(tok)


def test_well_formed_drops_fragments_fillers_and_continuations():
    fragment = _utt(
        ["the", "bread", "that", "goes", "around", "the", "hotdog"],
        ["det", "noun", "sconj", "verb", "adp", "det", "noun"],
        ["det", "root", "mark", "acl-relcl", "case", "det", "obl"],
    )
    assert not is_well_formed_sentence(fragment)
    filler = _utt(
        ["uh", "he", "came", "through"],
        ["intj", "pron", "verb", "adp"],
        ["discourse", "nsubj", "root", "case"],
    )
    assert not is_well_formed_sentence(filler)
    cont = _utt(
        ["and", "he", "goes", "to", "the", "river"],
        ["cconj", "pron", "verb", "adp", "det", "noun"],
        ["cc", "nsubj", "root", "case", "det", "obl"],
    )
    assert not is_well_formed_sentence(cont)
    named = _utt(
        ["Peter", "Pan", "falls", "down"],
        ["propn", "propn", "verb", "adv"],
        ["nsubj", "flat", "root", "advmod"],
    )
    assert is_well_formed_sentence(named)
    assert not eval_utterance_ok(named)
    repeats = _utt(
        ["he", "he", "fell", "down"],
        ["pron", "pron", "verb", "adv"],
        ["nsubj", "nsubj", "root", "advmod"],
    )
    assert not is_well_formed_sentence(repeats)
    stacked = _utt(
        ["go", "fly", "back", "up"],
        ["verb", "verb", "adv", "adv"],
        ["root", "xcomp", "advmod", "advmod"],
    )
    assert not is_well_formed_sentence(stacked)
    restart = _utt(
        ["there", "was", "there", "was", "a", "bug"],
        ["pron", "verb", "pron", "verb", "det", "noun"],
        ["expl", "root", "expl", "parataxis", "det", "nsubj"],
    )
    assert not is_well_formed_sentence(restart)
    two_pron = _utt(
        ["she", "we", "can", "load", "her"],
        ["pron", "pron", "aux", "verb", "pron"],
        ["nsubj", "nsubj", "aux", "root", "obj"],
    )
    assert not is_well_formed_sentence(two_pron)
