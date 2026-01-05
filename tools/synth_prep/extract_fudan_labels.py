#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import lmdb


LABEL_KEY_PATTERNS: Sequence[re.Pattern] = (
    re.compile(rb"^label[-_](\d+)$"),
    re.compile(rb"^labels[-_](\d+)$"),
)


@dataclass
class ExtractStats:
    lmdb_dir: str
    charset_path: Optional[str]
    normalize: str
    remove_space: bool
    lmdb_num_samples_meta: Optional[int]
    lmdb_start_index: Optional[int]
    lmdb_label_tmpl: Optional[str]
    lmdb_missing_label_keys: int
    lmdb_decode_errors: int
    lmdb_empty_labels: int
    extracted_total: int
    extracted_kept: int
    extracted_filtered_oov: int
    extracted_dedup: int
    base_in_lines: Optional[int] = None
    base_kept: Optional[int] = None
    base_filtered_oov: Optional[int] = None
    base_dedup: Optional[int] = None
    merged_total: Optional[int] = None
    merged_added_from_lmdb: Optional[int] = None
    merged_dup_skipped: Optional[int] = None


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


def strip_leading_list_punct(text: str) -> str:
    # Common artifact in some corpora: leading "、" used as list marker.
    return re.sub(r"^、+", "", text).strip()


def stable_dedup(lines: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in lines:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def contains_oov(text: str, charset_set: Optional[set]) -> bool:
    if charset_set is None:
        return False
    for ch in text:
        if ch not in charset_set:
            return True
    return False


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
    candidates = (
        "label-%09d",
        "label-%08d",
        "label_%09d",
        "label_%08d",
        "label-%d",
        "label_%d",
        "labels-%09d",
        "labels_%09d",
    )
    for start in start_candidates:
        for tmpl in candidates:
            key = (tmpl % start).encode("utf-8")
            if txn.get(key) is not None:
                return start, tmpl
    return None, None


def _scan_label_keys(txn: lmdb.Transaction, limit: Optional[int] = None) -> List[Tuple[int, bytes]]:
    found: List[Tuple[int, bytes]] = []
    with txn.cursor() as cursor:
        for i, (k, _v) in enumerate(cursor):
            if limit is not None and i >= limit:
                break
            for pat in LABEL_KEY_PATTERNS:
                m = pat.match(k)
                if m:
                    try:
                        idx = int(m.group(1))
                    except Exception:  # noqa: BLE001
                        continue
                    found.append((idx, k))
                    break
    found.sort(key=lambda x: x[0])
    return found


def extract_labels_from_lmdb(
    lmdb_dir: Path,
    normalize: str,
    remove_space_flag: bool,
    charset_set: Optional[set],
) -> Tuple[List[str], Dict[str, object]]:
    env = lmdb.open(
        str(lmdb_dir),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=64,
    )
    missing = 0
    decode_errors = 0
    empty_labels = 0
    filtered_oov = 0

    extracted: List[str] = []
    meta: Dict[str, object] = {
        "lmdb_num_samples_meta": None,
        "lmdb_start_index": None,
        "lmdb_label_tmpl": None,
        "lmdb_missing_label_keys": 0,
        "lmdb_decode_errors": 0,
        "lmdb_empty_labels": 0,
        "extracted_total": 0,
        "extracted_kept": 0,
        "extracted_filtered_oov": 0,
    }

    with env.begin(write=False) as txn:
        num_samples = _try_read_num_samples(txn)
        meta["lmdb_num_samples_meta"] = num_samples

        if num_samples is not None:
            start, tmpl = _detect_label_template(txn, start_candidates=(0, 1))
            meta["lmdb_start_index"] = start
            meta["lmdb_label_tmpl"] = tmpl

            if start is not None and tmpl is not None:
                for idx in range(start, start + num_samples):
                    key = (tmpl % idx).encode("utf-8")
                    val = txn.get(key)
                    if val is None:
                        missing += 1
                        continue
                    try:
                        s = val.decode("utf-8", errors="ignore")
                    except Exception:  # noqa: BLE001
                        decode_errors += 1
                        continue
                    s = s.strip()
                    if not s:
                        empty_labels += 1
                        continue
                    s = normalize_text(s, normalize)
                    s = remove_spaces(s) if remove_space_flag else collapse_spaces(s)
                    if not s:
                        empty_labels += 1
                        continue
                    if contains_oov(s, charset_set):
                        filtered_oov += 1
                        continue
                    extracted.append(s)
            else:
                # Fallback: scan keys for label-xxxx
                key_pairs = _scan_label_keys(txn)
                for _idx, key in key_pairs:
                    val = txn.get(key)
                    if val is None:
                        missing += 1
                        continue
                    try:
                        s = val.decode("utf-8", errors="ignore")
                    except Exception:  # noqa: BLE001
                        decode_errors += 1
                        continue
                    s = s.strip()
                    if not s:
                        empty_labels += 1
                        continue
                    s = normalize_text(s, normalize)
                    s = remove_spaces(s) if remove_space_flag else collapse_spaces(s)
                    if not s:
                        empty_labels += 1
                        continue
                    if contains_oov(s, charset_set):
                        filtered_oov += 1
                        continue
                    extracted.append(s)
        else:
            # Fallback: scan keys for label-xxxx
            key_pairs = _scan_label_keys(txn)
            for _idx, key in key_pairs:
                val = txn.get(key)
                if val is None:
                    missing += 1
                    continue
                try:
                    s = val.decode("utf-8", errors="ignore")
                except Exception:  # noqa: BLE001
                    decode_errors += 1
                    continue
                s = s.strip()
                if not s:
                    empty_labels += 1
                    continue
                s = normalize_text(s, normalize)
                s = remove_spaces(s) if remove_space_flag else collapse_spaces(s)
                if not s:
                    empty_labels += 1
                    continue
                if contains_oov(s, charset_set):
                    filtered_oov += 1
                    continue
                extracted.append(s)

    meta["lmdb_missing_label_keys"] = missing
    meta["lmdb_decode_errors"] = decode_errors
    meta["lmdb_empty_labels"] = empty_labels
    meta["extracted_total"] = len(extracted) + filtered_oov
    meta["extracted_kept"] = len(extracted)
    meta["extracted_filtered_oov"] = filtered_oov
    return extracted, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract labels from Fudan scene LMDB.")
    parser.add_argument("--lmdb-dir", required=True, help="lmdb dir, e.g. data/fudan/scene/scene_train")
    parser.add_argument("--out", required=True, help="output labels txt path")
    parser.add_argument("--charset", default=None, help="charset file path; if set, filter OOV labels")
    parser.add_argument(
        "--normalize",
        default="nfkc_lower",
        choices=["none", "nfkc", "nfkc_lower"],
        help="normalization mode",
    )
    parser.add_argument("--keep-space", action="store_true", help="keep spaces; default remove if charset has no space")
    parser.add_argument(
        "--keep-leading-punct",
        action="store_true",
        help="keep leading list punct like '、xxx' (default: strip it)",
    )
    parser.add_argument("--merge-base", default=None, help="base corpus txt to merge into")
    parser.add_argument("--out-merged", default=None, help="merged output txt path (required if --merge-base)")
    parser.add_argument("--stats", default=None, help="write stats json to this path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lmdb_dir = Path(args.lmdb_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not lmdb_dir.is_dir():
        raise SystemExit(f"--lmdb-dir not found: {lmdb_dir}")

    charset_set = None
    remove_space_flag = True
    strip_leading_flag = not bool(args.keep_leading_punct)
    charset_path: Optional[Path] = None
    if args.charset:
        charset_path = Path(args.charset).expanduser().resolve()
        if not charset_path.is_file():
            raise SystemExit(f"--charset not found: {charset_path}")
        charset_chars = read_charset(charset_path)
        charset_set = set(charset_chars)
        remove_space_flag = (" " not in charset_set) and (not args.keep_space)
    else:
        remove_space_flag = not args.keep_space

    extracted, meta = extract_labels_from_lmdb(
        lmdb_dir=lmdb_dir,
        normalize=str(args.normalize),
        remove_space_flag=remove_space_flag,
        charset_set=charset_set,
    )

    if strip_leading_flag:
        extracted = [strip_leading_list_punct(s) for s in extracted if strip_leading_list_punct(s)]

    deduped = stable_dedup(extracted)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(deduped) + ("\n" if deduped else ""), encoding="utf-8")

    stats = ExtractStats(
        lmdb_dir=str(lmdb_dir),
        charset_path=str(charset_path) if charset_path else None,
        normalize=str(args.normalize),
        remove_space=remove_space_flag,
        lmdb_num_samples_meta=meta.get("lmdb_num_samples_meta"),  # type: ignore[arg-type]
        lmdb_start_index=meta.get("lmdb_start_index"),  # type: ignore[arg-type]
        lmdb_label_tmpl=meta.get("lmdb_label_tmpl"),  # type: ignore[arg-type]
        lmdb_missing_label_keys=int(meta.get("lmdb_missing_label_keys", 0)),
        lmdb_decode_errors=int(meta.get("lmdb_decode_errors", 0)),
        lmdb_empty_labels=int(meta.get("lmdb_empty_labels", 0)),
        extracted_total=int(meta.get("extracted_total", 0)),
        extracted_kept=int(meta.get("extracted_kept", 0)),
        extracted_filtered_oov=int(meta.get("extracted_filtered_oov", 0)),
        extracted_dedup=len(deduped),
    )

    # Optional merge
    if args.merge_base:
        base_path = Path(args.merge_base).expanduser().resolve()
        if not base_path.is_file():
            raise SystemExit(f"--merge-base not found: {base_path}")
        if not args.out_merged:
            raise SystemExit("--out-merged is required when using --merge-base")
        out_merged = Path(args.out_merged).expanduser().resolve()

        base_raw = base_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        stats.base_in_lines = len(base_raw)

        base_cleaned: List[str] = []
        base_filtered_oov = 0
        base_empty = 0
        for s in base_raw:
            s = s.strip()
            if not s:
                base_empty += 1
                continue
            s = normalize_text(s, str(args.normalize))
            s = remove_spaces(s) if remove_space_flag else collapse_spaces(s)
            if strip_leading_flag:
                s = strip_leading_list_punct(s)
            if not s:
                base_empty += 1
                continue
            if contains_oov(s, charset_set):
                base_filtered_oov += 1
                continue
            base_cleaned.append(s)
        base_deduped = stable_dedup(base_cleaned)

        stats.base_kept = len(base_cleaned)
        stats.base_filtered_oov = base_filtered_oov
        stats.base_dedup = len(base_deduped)

        merged: List[str] = []
        seen = set()
        for s in base_deduped:
            if s in seen:
                continue
            seen.add(s)
            merged.append(s)
        added = 0
        dup = 0
        for s in deduped:
            if s in seen:
                dup += 1
                continue
            seen.add(s)
            merged.append(s)
            added += 1

        out_merged.parent.mkdir(parents=True, exist_ok=True)
        out_merged.write_text("\n".join(merged) + ("\n" if merged else ""), encoding="utf-8")

        stats.merged_total = len(merged)
        stats.merged_added_from_lmdb = added
        stats.merged_dup_skipped = dup

    stats_json = json.dumps(stats.__dict__, ensure_ascii=False, indent=2)
    print("[OK] extract stats:")
    print(stats_json)
    if args.stats:
        stats_path = Path(args.stats).expanduser().resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(stats_json + "\n", encoding="utf-8")
        print(f"[OK] stats written: {stats_path}")


if __name__ == "__main__":
    main()
