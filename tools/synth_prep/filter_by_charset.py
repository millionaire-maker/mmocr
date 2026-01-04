#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class FilterStats:
    in_path: str
    out_path: str
    charset_path: str
    normalize: str
    remove_space: bool
    in_lines: int
    kept_lines: int
    removed_lines: int
    oov_unique: int
    oov_top50: List[Tuple[str, int]]


def read_charset(charset_path: Path) -> List[str]:
    chars: List[str] = []
    for line in charset_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        chars.append(line[0])
    return chars


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


def filter_file(
    in_path: Path,
    out_path: Path,
    charset_set: set,
    normalize: str,
    remove_space_flag: bool,
) -> FilterStats:
    raw_lines = in_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    oov_counter: Counter = Counter()
    kept: List[str] = []
    removed = 0

    for raw in raw_lines:
        s = raw.strip()
        if not s:
            continue
        s = normalize_text(s, normalize)
        s = remove_spaces(s) if remove_space_flag else collapse_spaces(s)
        if not s:
            continue

        oov_chars = [ch for ch in s if ch not in charset_set]
        if oov_chars:
            removed += 1
            oov_counter.update(oov_chars)
            continue
        kept.append(s)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    top50 = [(ch, int(cnt)) for ch, cnt in oov_counter.most_common(50)]
    return FilterStats(
        in_path=str(in_path),
        out_path=str(out_path),
        charset_path="",  # filled by caller
        normalize=normalize,
        remove_space=remove_space_flag,
        in_lines=len(raw_lines),
        kept_lines=len(kept),
        removed_lines=removed,
        oov_unique=len(oov_counter),
        oov_top50=top50,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter corpus lines by charset (drop lines with OOV chars).")
    parser.add_argument("--in", dest="in_path", required=True, help="input txt path")
    parser.add_argument("--out", dest="out_path", required=True, help="output txt path")
    parser.add_argument("--charset", required=True, help="charset file path")
    parser.add_argument(
        "--normalize",
        default="nfkc_lower",
        choices=["none", "nfkc", "nfkc_lower"],
        help="normalization mode used before checking",
    )
    parser.add_argument("--keep-space", action="store_true", help="keep spaces; default remove if charset has no space")
    parser.add_argument("--stats", default=None, help="write stats json to this path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_path).expanduser().resolve()
    out_path = Path(args.out_path).expanduser().resolve()
    charset_path = Path(args.charset).expanduser().resolve()

    if not in_path.is_file():
        raise SystemExit(f"--in not found: {in_path}")
    if not charset_path.is_file():
        raise SystemExit(f"--charset not found: {charset_path}")

    charset_chars = read_charset(charset_path)
    charset_set = set(charset_chars)
    remove_space_flag = (" " not in charset_set) and (not args.keep_space)

    stats = filter_file(
        in_path=in_path,
        out_path=out_path,
        charset_set=charset_set,
        normalize=str(args.normalize),
        remove_space_flag=remove_space_flag,
    )
    stats.charset_path = str(charset_path)

    stats_json = json.dumps(stats.__dict__, ensure_ascii=False, indent=2)
    print("[OK] filter stats:")
    print(stats_json)
    if args.stats:
        stats_path = Path(args.stats).expanduser().resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(stats_json + "\n", encoding="utf-8")
        print(f"[OK] stats written: {stats_path}")


if __name__ == "__main__":
    main()
