"""Shared builders for synthetic UD utterances."""

from __future__ import annotations

import pandas as pd


def tokens(rows: list[dict], **meta) -> pd.DataFrame:
    defaults = {
        "parse_source": "childesdb",
        "utterance_type": "declarative",
        "suffix": "",
        "corpus_name": "Brown",
        "collection_name": "Eng-NA",
        "speaker_role": "Mother",
        "target_child_age": 24.0,
        "in_s7_2": True,
        "filename": "Brown/Adam/adam01.cha",
        "num_tokens": len(rows),
    }
    defaults.update(meta)
    out = []
    for i, row in enumerate(rows, start=1):
        rec = {
            "id": 1000 + i,
            "token_order": i,
            "gra_index": i,
            "utterance_id": defaults.get("utterance_id", "u1"),
            **{k: v for k, v in defaults.items() if k != "utterance_id"},
        }
        rec.update(row)
        rec.setdefault("stem", rec["gloss"].lower())
        out.append(rec)
    return pd.DataFrame(out)


# she go to the park  (VL-to)
VL = tokens(
    [
        {"gloss": "she", "part_of_speech": "PRON", "gra_head": 2, "gra_relation": "nsubj"},
        {"gloss": "go", "part_of_speech": "VERB", "gra_head": 0, "gra_relation": "root", "stem": "go"},
        {"gloss": "to", "part_of_speech": "ADP", "gra_head": 5, "gra_relation": "case"},
        {"gloss": "the", "part_of_speech": "DET", "gra_head": 5, "gra_relation": "det"},
        {"gloss": "park", "part_of_speech": "NOUN", "gra_head": 2, "gra_relation": "obl"},
    ],
    utterance_id="vl1",
)

# she put the cup on the table  (VOL-on)
VOL = tokens(
    [
        {"gloss": "she", "part_of_speech": "PRON", "gra_head": 2, "gra_relation": "nsubj"},
        {"gloss": "put", "part_of_speech": "VERB", "gra_head": 0, "gra_relation": "root", "stem": "put"},
        {"gloss": "the", "part_of_speech": "DET", "gra_head": 4, "gra_relation": "det"},
        {"gloss": "cup", "part_of_speech": "NOUN", "gra_head": 2, "gra_relation": "obj"},
        {"gloss": "on", "part_of_speech": "ADP", "gra_head": 7, "gra_relation": "case"},
        {"gloss": "the", "part_of_speech": "DET", "gra_head": 7, "gra_relation": "det"},
        {"gloss": "table", "part_of_speech": "NOUN", "gra_head": 2, "gra_relation": "obl"},
    ],
    utterance_id="vol1",
)

# they give him the book  (VOO)
VOO = tokens(
    [
        {"gloss": "they", "part_of_speech": "PRON", "gra_head": 2, "gra_relation": "nsubj"},
        {"gloss": "give", "part_of_speech": "VERB", "gra_head": 0, "gra_relation": "root", "stem": "give"},
        {"gloss": "him", "part_of_speech": "PRON", "gra_head": 2, "gra_relation": "iobj"},
        {"gloss": "the", "part_of_speech": "DET", "gra_head": 5, "gra_relation": "det"},
        {"gloss": "book", "part_of_speech": "NOUN", "gra_head": 2, "gra_relation": "obj"},
    ],
    utterance_id="voo1",
)

# she eat the apple  (VO)
VO = tokens(
    [
        {"gloss": "she", "part_of_speech": "PRON", "gra_head": 2, "gra_relation": "nsubj"},
        {"gloss": "eat", "part_of_speech": "VERB", "gra_head": 0, "gra_relation": "root", "stem": "eat"},
        {"gloss": "the", "part_of_speech": "DET", "gra_head": 4, "gra_relation": "det"},
        {"gloss": "apple", "part_of_speech": "NOUN", "gra_head": 2, "gra_relation": "obj"},
    ],
    utterance_id="vo1",
)

# the apple was eaten  (passive — exclude)
PASSIVE = tokens(
    [
        {"gloss": "the", "part_of_speech": "DET", "gra_head": 2, "gra_relation": "det"},
        {"gloss": "apple", "part_of_speech": "NOUN", "gra_head": 4, "gra_relation": "nsubj:pass"},
        {"gloss": "was", "part_of_speech": "AUX", "gra_head": 4, "gra_relation": "aux:pass", "stem": "be"},
        {"gloss": "eaten", "part_of_speech": "VERB", "gra_head": 0, "gra_relation": "root", "stem": "eat"},
    ],
    utterance_id="pass1",
)

QUESTION = tokens(
    [
        {"gloss": "what", "part_of_speech": "PRON", "gra_head": 3, "gra_relation": "obj"},
        {"gloss": "did", "part_of_speech": "AUX", "gra_head": 3, "gra_relation": "aux", "stem": "do"},
        {"gloss": "she", "part_of_speech": "PRON", "gra_head": 3, "gra_relation": "nsubj"},
        {"gloss": "eat", "part_of_speech": "VERB", "gra_head": 0, "gra_relation": "root", "stem": "eat"},
        {"gloss": "?", "part_of_speech": "PUNCT", "gra_head": 4, "gra_relation": "punct"},
    ],
    utterance_id="q1",
    utterance_type="question",
)

# he take the hat off  (VOL-off via compound:prt)
VOL_OFF = tokens(
    [
        {"gloss": "he", "part_of_speech": "PRON", "gra_head": 2, "gra_relation": "nsubj"},
        {"gloss": "take", "part_of_speech": "VERB", "gra_head": 0, "gra_relation": "root", "stem": "take"},
        {"gloss": "the", "part_of_speech": "DET", "gra_head": 4, "gra_relation": "det"},
        {"gloss": "hat", "part_of_speech": "NOUN", "gra_head": 2, "gra_relation": "obj"},
        {"gloss": "off", "part_of_speech": "ADP", "gra_head": 2, "gra_relation": "compound:prt"},
    ],
    utterance_id="voloff1",
)

# she put the bag onto the shelf  (VOL-onto; Child I / Parent Young should drop)
VOL_ONTO = tokens(
    [
        {"gloss": "she", "part_of_speech": "PRON", "gra_head": 2, "gra_relation": "nsubj"},
        {"gloss": "put", "part_of_speech": "VERB", "gra_head": 0, "gra_relation": "root", "stem": "put"},
        {"gloss": "the", "part_of_speech": "DET", "gra_head": 4, "gra_relation": "det"},
        {"gloss": "bag", "part_of_speech": "NOUN", "gra_head": 2, "gra_relation": "obj"},
        {"gloss": "onto", "part_of_speech": "ADP", "gra_head": 7, "gra_relation": "case"},
        {"gloss": "the", "part_of_speech": "DET", "gra_head": 7, "gra_relation": "det"},
        {"gloss": "shelf", "part_of_speech": "NOUN", "gra_head": 2, "gra_relation": "obl"},
    ],
    utterance_id="onto1",
)

# copular: she is happy
COPULAR = tokens(
    [
        {"gloss": "she", "part_of_speech": "PRON", "gra_head": 3, "gra_relation": "nsubj"},
        {"gloss": "is", "part_of_speech": "AUX", "gra_head": 3, "gra_relation": "cop", "stem": "be"},
        {"gloss": "happy", "part_of_speech": "ADJ", "gra_head": 0, "gra_relation": "root"},
    ],
    utterance_id="cop1",
)

ALL = pd.concat(
    [VL, VOL, VOO, VO, PASSIVE, QUESTION, VOL_OFF, VOL_ONTO, COPULAR],
    ignore_index=True,
)
