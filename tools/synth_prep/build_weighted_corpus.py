#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass
class MixStats:
    base_path: str
    hard_path: str
    out_path: str
    total: int
    hard_ratio: float
    seed: int
    min_len: int
    max_len: int
    base_in: int
    base_usable: int
    hard_in: int
    hard_usable: int
    sampled_base: int
    sampled_hard: int
    out_lines: int


def read_lines(path: Path, min_len: int, max_len: int) -> List[str]:
    raw = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: List[str] = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        if len(s) < min_len:
            continue
        if max_len > 0 and len(s) > max_len:
            continue
        out.append(s)
    return out


def sample_lines(rng: random.Random, lines: Sequence[str], n: int) -> List[str]:
    if n <= 0:
        return []
    if len(lines) >= n:
        return rng.sample(list(lines), n)
    # fallback: with replacement
    return [rng.choice(list(lines)) for _ in range(n)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a weighted mixed corpus for list-mode generation.")
    parser.add_argument("--base", required=True, help="base corpus txt (main)")
    parser.add_argument("--hard", required=True, help="hard corpus txt (curved/hard cases)")
    parser.add_argument("--out", required=True, help="output mixed corpus txt")
    parser.add_argument("--total", type=int, default=500, help="total lines to sample")
    parser.add_argument("--hard-ratio", type=float, default=0.4, help="ratio of hard lines in output")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--min-len", type=int, default=2, help="drop lines shorter than this length")
    parser.add_argument("--max-len", type=int, default=0, help="drop lines longer than this length, 0 means no limit")
    parser.add_argument("--stats", default=None, help="write stats json path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.base).expanduser().resolve()
    hard_path = Path(args.hard).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not base_path.is_file():
        raise SystemExit(f"--base not found: {base_path}")
    if not hard_path.is_file():
        raise SystemExit(f"--hard not found: {hard_path}")
    if int(args.total) <= 0:
        raise SystemExit("--total must be > 0")

    rng = random.Random(int(args.seed))
    base_lines = read_lines(base_path, min_len=int(args.min_len), max_len=int(args.max_len))
    hard_lines = read_lines(hard_path, min_len=int(args.min_len), max_len=int(args.max_len))
    if not base_lines:
        raise SystemExit(
            f"no usable base lines after min_len={args.min_len}, max_len={args.max_len}: {base_path}"
        )
    if not hard_lines:
        raise SystemExit(
            f"no usable hard lines after min_len={args.min_len}, max_len={args.max_len}: {hard_path}"
        )

    hard_n = int(round(int(args.total) * float(args.hard_ratio)))
    hard_n = max(0, min(int(args.total), hard_n))
    base_n = int(args.total) - hard_n

    sampled_base = sample_lines(rng, base_lines, base_n)
    sampled_hard = sample_lines(rng, hard_lines, hard_n)

    mixed = sampled_base + sampled_hard
    rng.shuffle(mixed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(mixed) + "\n", encoding="utf-8")

    stats = MixStats(
        base_path=str(base_path),
        hard_path=str(hard_path),
        out_path=str(out_path),
        total=int(args.total),
        hard_ratio=float(args.hard_ratio),
        seed=int(args.seed),
        min_len=int(args.min_len),
        max_len=int(args.max_len),
        base_in=len(base_path.read_text(encoding="utf-8", errors="ignore").splitlines()),
        base_usable=len(base_lines),
        hard_in=len(hard_path.read_text(encoding="utf-8", errors="ignore").splitlines()),
        hard_usable=len(hard_lines),
        sampled_base=len(sampled_base),
        sampled_hard=len(sampled_hard),
        out_lines=len(mixed),
    )

    print("[OK] mixed corpus stats:")
    print(json.dumps(stats.__dict__, ensure_ascii=False, indent=2))
    if args.stats:
        stats_path = Path(args.stats).expanduser().resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats.__dict__, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] stats written: {stats_path}")


if __name__ == "__main__":
    main()
