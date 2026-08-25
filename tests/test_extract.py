from fixtures import ALL, COPULAR, PASSIVE, QUESTION, VL, VO, VOL, VOL_OFF, VOL_ONTO, VOO
from vacacq.childes.extract import extract_utterance_s74, extract_utterance_ud, extract_vacs, filter_hits_for_stratum


def _cons(hits):
    return {(h.construction, h.verb, h.preposition, h.extra) for h in hits}


def test_ud_vl_to():
    hits = extract_utterance_ud(VL)
    assert ("VL", "go", "to", False) in _cons(hits)


def test_ud_vol_on():
    hits = extract_utterance_ud(VOL)
    assert ("VOL", "put", "on", False) in _cons(hits)


def test_ud_voo_give():
    hits = extract_utterance_ud(VOO)
    assert ("VOO", "give", None, False) in _cons(hits)


def test_ud_vo_eat():
    hits = extract_utterance_ud(VO)
    assert ("VO", "eat", None, False) in _cons(hits)


def test_ud_excludes_passive_and_questions():
    assert extract_utterance_ud(PASSIVE) == []
    assert extract_utterance_ud(QUESTION) == []


def test_ud_particle_vol_off():
    hits = extract_utterance_ud(VOL_OFF)
    assert ("VOL", "take", "off", False) in _cons(hits)


def test_onto_dropped_for_child_i():
    hits = extract_vacs(VOL_ONTO)
    kept = filter_hits_for_stratum(hits, "Child I")
    dropped = kept.loc[(kept["construction"] == "VOL") & (kept["preposition"] == "onto")]
    assert dropped.empty
    parent_all = filter_hits_for_stratum(hits, "Parent All")
    assert not parent_all.loc[(parent_all["construction"] == "VOL") & (parent_all["preposition"] == "onto")].empty


def test_copular_extra():
    hits = extract_utterance_ud(COPULAR)
    assert ("copular", "be", None, True) in _cons(hits)


def test_s74_vl_and_vol_agree_on_linear_cases():
    vl = {(h.construction, h.verb, h.preposition) for h in extract_utterance_s74(VL)}
    vol = {(h.construction, h.verb, h.preposition) for h in extract_utterance_s74(VOL)}
    assert ("VL", "go", "to") in vl
    assert ("VOL", "put", "on") in vol


def test_2026_1_chat_gra_and_lowercase_pos():
    """2026.1 stores verb/adp and OBJ/COMPOUND-PRT, not VERB/compound:prt."""
    from fixtures import tokens

    frame = tokens(
        [
            {"gloss": "he", "part_of_speech": "pron", "gra_head": 2, "gra_relation": "NSUBJ"},
            {"gloss": "take", "part_of_speech": "verb", "gra_head": 0, "gra_relation": "ROOT", "stem": "take"},
            {"gloss": "the", "part_of_speech": "det", "gra_head": 4, "gra_relation": "DET"},
            {"gloss": "hat", "part_of_speech": "noun", "gra_head": 2, "gra_relation": "OBJ"},
            {"gloss": "off", "part_of_speech": "adp", "gra_head": 2, "gra_relation": "COMPOUND-PRT"},
        ],
        utterance_id="chat1",
    )
    hits = extract_utterance_ud(frame)
    assert ("VOL", "take", "off", False) in _cons(hits)


def test_extract_vacs_batch():
    hits = extract_vacs(ALL)
    schematic = hits.loc[~hits["extra"]]
    assert set(schematic["construction"]) >= {"VL", "VOL", "VOO", "VO"}
