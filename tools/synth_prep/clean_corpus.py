#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import shutil
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PREFIX_RULES: Sequence[Tuple[str, re.Pattern]] = (
    # 001\txxx / 001 xxx
    ("digits_tab", re.compile(r"^\s*\d{1,6}\t+\s*")),
    # (23) xxx / （23）xxx / [23] xxx / 【23】xxx
    ("bracket_digits", re.compile(r"^\s*[\(\[\{（【]\s*\d{1,6}\s*[\)\]\}）】]\s*")),
    # 23) xxx / 23）xxx
    ("digits_rparen", re.compile(r"^\s*\d{1,6}\s*[)）]\s*")),
    # 23. xxx / 23、xxx / 23: xxx / 23：xxx / 23-xxx
    ("digits_punct", re.compile(r"^\s*\d{1,6}\s*[\.．。、,:：\-—]\s*")),
    # 23] xxx / 23】xxx
    ("digits_rbracket", re.compile(r"^\s*\d{1,6}\s*[\]】]\s*")),
    # 23 xxx  (only short index, avoid stripping year-like prefixes)
    ("digits_space_short", re.compile(r"^\s*\d{1,3}\s+(?=\S)")),
)


@dataclass
class CleanStats:
    in_path: str
    out_path: str
    charset_path: str
    normalize: str
    remove_space: bool
    original_lines: int
    after_clean_lines: int
    after_dedup_lines: int
    prefix_rule_hits: Dict[str, int]


def read_charset(charset_path: Path) -> List[str]:
    chars: List[str] = []
    for line in charset_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        chars.append(line[0])
    return chars


def backup_file(src: Path) -> Optional[Path]:
    if not src.exists():
        return None
    bak = src.with_suffix(src.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(src, bak)
        return bak
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak2 = src.with_suffix(src.suffix + f".bak.{ts}")
    shutil.copy2(src, bak2)
    return bak2


def normalize_text(text: str, mode: str) -> str:
    if mode == "none":
        return text
    if mode == "nfkc":
        return unicodedata.normalize("NFKC", text)
    if mode == "nfkc_lower":
        return unicodedata.normalize("NFKC", text).lower()
    raise ValueError(f"unknown normalize mode: {mode}")


def strip_prefix(text: str, hits: Counter) -> str:
    for name, pat in PREFIX_RULES:
        m = pat.match(text)
        if m:
            hits[name] += 1
            return text[m.end() :]
    return text


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def remove_spaces(text: str) -> str:
    return re.sub(r"\s+", "", text).strip()


def clean_lines(
    lines: Iterable[str],
    normalize: str,
    remove_space_flag: bool,
) -> Tuple[List[str], Counter]:
    hits: Counter = Counter()
    cleaned: List[str] = []
    for raw in lines:
        s = raw.strip("\n\r")
        s = s.lstrip("\ufeff")
        s = s.strip()
        if not s:
            continue
        s = strip_prefix(s, hits)
        s = s.strip()
        if not s:
            continue
        s = normalize_text(s, normalize)
        s = remove_spaces(s) if remove_space_flag else collapse_spaces(s)
        if not s:
            continue
        cleaned.append(s)
    return cleaned, hits


def stable_dedup(lines: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in lines:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean list corpus: strip index prefixes, normalize, and dedup.")
    parser.add_argument("--in", dest="in_path", required=True, help="input txt path")
    parser.add_argument("--out", dest="out_path", required=True, help="output txt path (will not overwrite input)")
    parser.add_argument("--charset", required=True, help="charset file path (1 char per line, use first char)")
    parser.add_argument(
        "--normalize",
        default="nfkc_lower",
        choices=["none", "nfkc", "nfkc_lower"],
        help="text normalization mode",
    )
    parser.add_argument(
        "--keep-space",
        action="store_true",
        help="keep spaces (collapse multiple spaces). Default: remove if charset has no space.",
    )
    parser.add_argument("--stats", default=None, help="write stats json to this path")
    parser.add_argument("--no-backup", action="store_true", help="do not create *.bak backup")
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
    if out_path == in_path:
        raise SystemExit("--out must be different from --in (to avoid overwrite)")

    charset_chars = read_charset(charset_path)
    charset_set = set(charset_chars)
    remove_space_flag = (" " not in charset_set) and (not args.keep_space)

    if not args.no_backup:
        bak = backup_file(in_path)
        print(f"[OK] backup: {bak}")

    raw_lines = in_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    cleaned, hits = clean_lines(raw_lines, normalize=str(args.normalize), remove_space_flag=remove_space_flag)
    deduped = stable_dedup(cleaned)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(deduped) + ("\n" if deduped else ""), encoding="utf-8")

    stats = CleanStats(
        in_path=str(in_path),
        out_path=str(out_path),
        charset_path=str(charset_path),
        normalize=str(args.normalize),
        remove_space=remove_space_flag,
        original_lines=len(raw_lines),
        after_clean_lines=len(cleaned),
        after_dedup_lines=len(deduped),
        prefix_rule_hits=dict(hits),
    )

    print("[OK] clean stats:")
    print(json.dumps(stats.__dict__, ensure_ascii=False, indent=2))

    if args.stats:
        stats_path = Path(args.stats).expanduser().resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats.__dict__, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] stats written: {stats_path}")


if __name__ == "__main__":
    main()
