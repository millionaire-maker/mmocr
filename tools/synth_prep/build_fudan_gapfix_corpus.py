#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import lmdb


LABEL_KEY_CANDIDATES: Sequence[str] = (
    "label-%09d",
    "label-%08d",
    "label_%09d",
    "label_%08d",
    "label-%d",
    "label_%d",
    "labels-%09d",
    "labels_%09d",
)


@dataclass
class BuildStats:
    fudan_lmdb: str
    hard_txt: str
    out_txt: str
    total: int
    hard_ratio: float
    seed: int
    normalize: str
    max_len: int
    strip_leading_punct: bool
    charset_path: Optional[str]
    fudan_total: int
    fudan_kept: int
    fudan_skipped_oov: int
    fudan_skipped_too_long: int
    fudan_skipped_empty: int
    hard_in: int
    hard_usable: int
    out_len_hist_top20: List[Tuple[int, int]]


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


def strip_leading_list_punct(text: str) -> str:
    # Common artifact in some corpora: leading "、" used as list marker.
    return re.sub(r"^、+", "", text).strip()


def _try_read_num_samples(txn: lmdb.Transaction) -> Optional[int]:
    for key in (b"num-samples", b"num_samples", b"numSamples"):
        val = txn.get(key)
        if not val:
            continue
        try:
            return int(val.decode("utf-8", errors="ignore"))
        except Exception:  # noqa: BLE001
            continue
    return None


def _detect_label_template(txn: lmdb.Transaction, start_candidates: Sequence[int]) -> Tuple[Optional[int], Optional[str]]:
    for start in start_candidates:
        for tmpl in LABEL_KEY_CANDIDATES:
            key = (tmpl % start).encode("utf-8")
            if txn.get(key) is not None:
                return start, tmpl
    return None, None


def load_fudan_labels_with_freq(
    lmdb_dir: Path,
    normalize: str,
    max_len: int,
    strip_leading_punct: bool,
    charset_set: Optional[set],
) -> Tuple[List[str], Dict[str, int]]:
    env = lmdb.open(
        str(lmdb_dir),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=64,
    )
    kept: List[str] = []
    skipped_too_long = 0
    skipped_empty = 0
    skipped_oov = 0

    with env.begin(write=False) as txn:
        num_samples = _try_read_num_samples(txn)
        if num_samples is None:
            raise RuntimeError(f"num-samples not found in lmdb: {lmdb_dir}")
        start, tmpl = _detect_label_template(txn, start_candidates=(0, 1))
        if start is None or tmpl is None:
            raise RuntimeError(f"cannot detect label key template in lmdb: {lmdb_dir}")

        for idx in range(start, start + int(num_samples)):
            val = txn.get((tmpl % idx).encode("utf-8"))
            if val is None:
                continue
            s = val.decode("utf-8", errors="ignore").strip()
            if not s:
                skipped_empty += 1
                continue
            s = normalize_text(s, normalize)
            s = collapse_spaces(s)
            if strip_leading_punct:
                s = strip_leading_list_punct(s)
            if not s:
                skipped_empty += 1
                continue
            if max_len > 0 and len(s) > max_len:
                skipped_too_long += 1
                continue
            if charset_set is not None and any(ch not in charset_set for ch in s):
                skipped_oov += 1
                continue
            kept.append(s)

    env.close()
    meta = {
        "fudan_total": int(num_samples),
        "fudan_kept": len(kept),
        "fudan_skipped_oov": int(skipped_oov),
        "fudan_skipped_too_long": int(skipped_too_long),
        "fudan_skipped_empty": int(skipped_empty),
    }
    return kept, meta


def load_lines(path: Path, max_len: int, normalize: str, strip_leading_punct: bool) -> List[str]:
    out: List[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.strip()
        if not s:
            continue
        s = normalize_text(s, normalize)
        s = collapse_spaces(s)
        if strip_leading_punct:
            s = strip_leading_list_punct(s)
        if not s:
            continue
        if max_len > 0 and len(s) > max_len:
            continue
        out.append(s)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build a list-corpus file by sampling Fudan LMDB labels (with frequency) + hard-curve corpus."
    )
    ap.add_argument(
        "--fudan-lmdb",
        default="data/fudan/scene/scene_train",
        help="Fudan LMDB dir for label sampling",
    )
    ap.add_argument(
        "--hard-txt",
        default="data/synth_assets/list_corpus/hard_curve.cleaned.filtered.fontok.txt",
        help="hard/curved corpus txt (will be oversampled)",
    )
    ap.add_argument(
        "--out-txt",
        default="data/synth_assets/list_corpus/full_pretrain_mix_fudan_gapfix/mixed_pretrain.txt",
        help="output corpus txt",
    )
    ap.add_argument("--total", type=int, default=240000, help="total lines to write")
    ap.add_argument("--hard-ratio", type=float, default=0.12, help="ratio of hard lines in output")
    ap.add_argument("--seed", type=int, default=3407, help="random seed")
    ap.add_argument(
        "--normalize",
        default="nfkc_lower",
        choices=["none", "nfkc", "nfkc_lower"],
        help="normalization mode for both sources",
    )
    ap.add_argument(
        "--charset",
        default="data/charset/charset_rec_cn_en.txt",
        help="charset file (1 char per line). If set, drop lines containing OOV chars",
    )
    ap.add_argument("--max-len", type=int, default=30, help="drop lines longer than this length (0 means no limit)")
    ap.add_argument(
        "--keep-leading-punct",
        action="store_true",
        help="keep leading '、' list marker; default strips it",
    )
    ap.add_argument("--stats", default=None, help="write stats json path")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    fudan_lmdb = Path(args.fudan_lmdb).expanduser().resolve()
    hard_txt = Path(args.hard_txt).expanduser().resolve()
    out_txt = Path(args.out_txt).expanduser().resolve()
    stats_path = Path(args.stats).expanduser().resolve() if args.stats else None

    if not fudan_lmdb.is_dir():
        raise SystemExit(f"--fudan-lmdb not found: {fudan_lmdb}")
    if not hard_txt.is_file():
        raise SystemExit(f"--hard-txt not found: {hard_txt}")
    if int(args.total) <= 0:
        raise SystemExit("--total must be > 0")

    strip_leading = not bool(args.keep_leading_punct)

    charset_set = None
    charset_path: Optional[Path] = None
    if args.charset:
        charset_path = Path(args.charset).expanduser().resolve()
        if not charset_path.is_file():
            raise SystemExit(f"--charset not found: {charset_path}")
        charset_set = set(
            line[0]
            for line in charset_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line
        )

    fudan_labels, meta = load_fudan_labels_with_freq(
        lmdb_dir=fudan_lmdb,
        normalize=str(args.normalize),
        max_len=int(args.max_len),
        strip_leading_punct=strip_leading,
        charset_set=charset_set,
    )
    if not fudan_labels:
        raise SystemExit("no usable labels from Fudan LMDB after filtering")

    hard_lines = load_lines(
        hard_txt,
        max_len=int(args.max_len),
        normalize=str(args.normalize),
        strip_leading_punct=strip_leading,
    )
    if not hard_lines:
        raise SystemExit("no usable hard lines after filtering")

    rng = random.Random(int(args.seed))

    hard_n = int(round(int(args.total) * float(args.hard_ratio)))
    hard_n = max(0, min(int(args.total), hard_n))
    base_n = int(args.total) - hard_n

    mixed: List[str] = []
    mixed.extend(rng.choice(fudan_labels) for _ in range(base_n))
    mixed.extend(rng.choice(hard_lines) for _ in range(hard_n))
    rng.shuffle(mixed)

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(mixed) + "\n", encoding="utf-8")

    hist = Counter(len(s) for s in mixed)
    top20 = sorted(((int(k), int(v)) for k, v in hist.items()), key=lambda x: x[0])[:20]

    stats = BuildStats(
        fudan_lmdb=str(fudan_lmdb),
        hard_txt=str(hard_txt),
        out_txt=str(out_txt),
        total=int(args.total),
        hard_ratio=float(args.hard_ratio),
        seed=int(args.seed),
        normalize=str(args.normalize),
        max_len=int(args.max_len),
        strip_leading_punct=strip_leading,
        charset_path=str(charset_path) if charset_path else None,
        fudan_total=int(meta["fudan_total"]),
        fudan_kept=int(meta["fudan_kept"]),
        fudan_skipped_oov=int(meta["fudan_skipped_oov"]),
        fudan_skipped_too_long=int(meta["fudan_skipped_too_long"]),
        fudan_skipped_empty=int(meta["fudan_skipped_empty"]),
        hard_in=len(hard_txt.read_text(encoding="utf-8", errors="ignore").splitlines()),
        hard_usable=len(hard_lines),
        out_len_hist_top20=top20,
    )

    stats_json = json.dumps(stats.__dict__, ensure_ascii=False, indent=2)
    print("[OK] corpus built:")
    print(stats_json)
    if stats_path is not None:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(stats_json + "\n", encoding="utf-8")
        print(f"[OK] stats written: {stats_path}")


if __name__ == "__main__":
    main()
