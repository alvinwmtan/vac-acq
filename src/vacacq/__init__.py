"""VAC acquisition: CHILDES reproduction and BabyLM probing."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = DATA / "cache"


def data_path(*parts: str) -> Path:
    return DATA.joinpath(*parts)


def main(argv: list[str] | None = None) -> None:
    from vacacq.cli import main as cli_main

    cli_main(argv)
