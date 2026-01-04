#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from fontTools.ttLib import TTCollection, TTFont


@dataclass
class FontFilterStats:
    corpus_in: str
    corpus_out: str
    fonts_list: str
    normalize: str
    remove_space: bool
    in_lines: int
    kept_lines: int
    removed_lines: int
    unsupported_unique: int
    unsupported_top50: List[Tuple[str, int]]


def normalize_text(text: str, mode: str) -> str:
    if mode == "none":
        return text
    if mode == "nfkc":
        return unicodedata.normalize("NFKC", text)
    if mode == "nfkc_lower":
        return unicodedata.normalize("NFKC", text).lower()
    raise ValueError(f"unknown normalize mode: {mode}")


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def remove_spaces(text: str) -> str:
    return re.sub(r"\s+", "", text).strip()


def _load_codepoints(font_path: Path) -> Set[int]:
    if font_path.suffix.lower() == ".ttc":
        ttc = TTCollection(str(font_path))
        ttf = ttc.fonts[0]
    else:
        ttf = TTFont(
            str(font_path),
            0,
            allowVID=0,
            ignoreDecompileErrors=True,
            fontNumber=-1,
        )
    cps: Set[int] = set()
    for table in ttf["cmap"].tables:
        cps.update(table.cmap.keys())
    ttf.close()
    return cps


def load_supported_codepoints(fonts_list: Path) -> Set[int]:
    font_paths = [Path(line.strip()) for line in fonts_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not font_paths:
        raise SystemExit(f"empty fonts_list: {fonts_list}")
    supported: Set[int] = set()
    for p in font_paths:
        if not p.exists():
            raise SystemExit(f"font not found: {p}")
        supported |= _load_codepoints(p)
    return supported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter corpus lines by whether they are renderable by ANY font in fonts_list."
    )
    parser.add_argument("--corpus-in", required=True, help="input corpus txt")
    parser.add_argument("--corpus-out", required=True, help="output corpus txt")
    parser.add_argument("--fonts-list", required=True, help="fonts_list file (1 abs path per line)")
    parser.add_argument(
        "--normalize",
        default="nfkc_lower",
        choices=["none", "nfkc", "nfkc_lower"],
        help="normalization before checking",
    )
    parser.add_argument("--keep-space", action="store_true", help="keep spaces; default remove all whitespace")
    parser.add_argument("--stats", default=None, help="write stats json path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_in = Path(args.corpus_in).expanduser().resolve()
    corpus_out = Path(args.corpus_out).expanduser().resolve()
    fonts_list = Path(args.fonts_list).expanduser().resolve()

    if not corpus_in.is_file():
        raise SystemExit(f"--corpus-in not found: {corpus_in}")
    if not fonts_list.is_file():
        raise SystemExit(f"--fonts-list not found: {fonts_list}")

    supported_cps = load_supported_codepoints(fonts_list)
    remove_space_flag = not args.keep_space

    raw_lines = corpus_in.read_text(encoding="utf-8", errors="ignore").splitlines()
    kept: List[str] = []
    removed = 0
    counter: Counter = Counter()

    for raw in raw_lines:
        s = raw.strip()
        if not s:
            continue
        s = normalize_text(s, str(args.normalize))
        s = remove_spaces(s) if remove_space_flag else collapse_spaces(s)
        if not s:
            continue
        unsupported = [ch for ch in s if ord(ch) not in supported_cps]
        if unsupported:
            removed += 1
            counter.update(unsupported)
            continue
        kept.append(s)

    corpus_out.parent.mkdir(parents=True, exist_ok=True)
    corpus_out.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    stats = FontFilterStats(
        corpus_in=str(corpus_in),
        corpus_out=str(corpus_out),
        fonts_list=str(fonts_list),
        normalize=str(args.normalize),
        remove_space=remove_space_flag,
        in_lines=len(raw_lines),
        kept_lines=len(kept),
        removed_lines=removed,
        unsupported_unique=len(counter),
        unsupported_top50=[(ch, int(cnt)) for ch, cnt in counter.most_common(50)],
    )

    stats_json = json.dumps(stats.__dict__, ensure_ascii=False, indent=2)
    print("[OK] font filter stats:")
    print(stats_json)
    if args.stats:
        stats_path = Path(args.stats).expanduser().resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(stats_json + "\n", encoding="utf-8")
        print(f"[OK] stats written: {stats_path}")


if __name__ == "__main__":
    main()
