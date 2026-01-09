#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import random
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import lmdb
import numpy as np


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

IMAGE_KEY_CANDIDATES: Sequence[str] = (
    "image-%09d",
    "image-%08d",
    "image_%09d",
    "image_%08d",
    "image-%d",
    "image_%d",
    "img-%09d",
    "img_%09d",
)


def read_charset(charset_path: Path) -> List[str]:
    chars: List[str] = []
    for line in charset_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line:
            continue
        chars.append(line[0])
    return chars


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


def _detect_template(
    txn: lmdb.Transaction, candidates: Sequence[str], start_candidates: Sequence[int]
) -> Tuple[Optional[int], Optional[str]]:
    for start in start_candidates:
        for tmpl in candidates:
            key = (tmpl % start).encode("utf-8")
            if txn.get(key) is not None:
                return start, tmpl
    return None, None


def _sniff_image_format(img_bytes: bytes) -> str:
    if img_bytes.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if img_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if img_bytes.startswith(b"RIFF") and b"WEBP" in img_bytes[:16]:
        return "webp"
    return "other"


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    # CJK Unified Ideographs + Extension A + Compatibility Ideographs
    return (
        0x3400 <= o <= 0x4DBF
        or 0x4E00 <= o <= 0x9FFF
        or 0xF900 <= o <= 0xFAFF
        or 0x20000 <= o <= 0x2A6DF
        or 0x2A700 <= o <= 0x2B73F
        or 0x2B740 <= o <= 0x2B81F
        or 0x2B820 <= o <= 0x2CEAF
    )


def char_bucket(ch: str) -> str:
    if ch.isspace():
        return "space"
    if "0" <= ch <= "9":
        return "digit"
    if ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
        return "latin"
    if _is_cjk(ch):
        return "cjk"
    if unicodedata.category(ch).startswith("P"):
        return "punct"
    return "other"


def safe_percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    if q <= 0:
        return float(min(values))
    if q >= 100:
        return float(max(values))
    xs = sorted(values)
    k = (len(xs) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(xs[int(k)])
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return float(d0 + d1)


def summarize(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(values),
        "min": float(min(values)),
        "mean": float(statistics.fmean(values)),
        "p10": safe_percentile(values, 10),
        "p50": safe_percentile(values, 50),
        "p90": safe_percentile(values, 90),
        "p95": safe_percentile(values, 95),
        "p99": safe_percentile(values, 99),
        "max": float(max(values)),
    }


@dataclass
class LabelStats:
    num_samples_meta: Optional[int]
    start_index: Optional[int]
    label_tmpl: Optional[str]
    missing_label_keys: int
    decode_errors: int
    empty_labels: int
    kept_labels: int
    label_len: Dict[str, Optional[float]]
    char_bucket_counts: Dict[str, int]
    top_chars: List[Tuple[str, int]]
    charset_path: Optional[str]
    oov_labels: int
    oov_chars_top50: List[Tuple[str, int]]


@dataclass
class ImageStats:
    sampled: int
    decode_fail: int
    missing_image_keys: int
    format_counts: Dict[str, int]
    channels_counts: Dict[str, int]
    width: Dict[str, Optional[float]]
    height: Dict[str, Optional[float]]
    aspect: Dict[str, Optional[float]]
    bytes_len: Dict[str, Optional[float]]
    mean_intensity: Dict[str, Optional[float]]
    std_intensity: Dict[str, Optional[float]]
    lap_var: Dict[str, Optional[float]]
    resize_sx_to_256: Dict[str, Optional[float]]
    resize_sy_to_64: Dict[str, Optional[float]]


@dataclass
class DatasetAnalysis:
    lmdb_dir: str
    label: LabelStats
    images: Optional[ImageStats]


def analyze_labels(
    lmdb_dir: Path,
    charset_path: Optional[Path],
) -> LabelStats:
    charset_set = None
    if charset_path is not None:
        charset_set = set(read_charset(charset_path))

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
    kept = 0

    lengths: List[float] = []
    bucket_counts: Dict[str, int] = defaultdict(int)
    char_counts: Counter = Counter()
    oov_labels = 0
    oov_chars: Counter = Counter()

    with env.begin(write=False) as txn:
        num_samples = _try_read_num_samples(txn)
        start, tmpl = _detect_template(txn, LABEL_KEY_CANDIDATES, start_candidates=(0, 1))

        if num_samples is None or start is None or tmpl is None:
            raise RuntimeError(
                f"Cannot detect label keys in lmdb: {lmdb_dir} "
                f"(num-samples={num_samples}, start={start}, tmpl={tmpl})"
            )

        for idx in range(start, start + num_samples):
            key = (tmpl % idx).encode("utf-8")
            val = txn.get(key)
            if val is None:
                missing += 1
                continue
            try:
                s = val.decode("utf-8", errors="ignore").strip()
            except Exception:  # noqa: BLE001
                decode_errors += 1
                continue
            if not s:
                empty_labels += 1
                continue

            kept += 1
            lengths.append(float(len(s)))
            has_oov = False
            for ch in s:
                bucket_counts[char_bucket(ch)] += 1
                char_counts[ch] += 1
                if charset_set is not None and ch not in charset_set:
                    has_oov = True
                    oov_chars[ch] += 1
            if has_oov:
                oov_labels += 1

    env.close()

    top_chars = [(c, int(n)) for c, n in char_counts.most_common(80)]
    oov_top = [(c, int(n)) for c, n in oov_chars.most_common(50)]
    return LabelStats(
        num_samples_meta=num_samples,
        start_index=start,
        label_tmpl=tmpl,
        missing_label_keys=missing,
        decode_errors=decode_errors,
        empty_labels=empty_labels,
        kept_labels=kept,
        label_len=summarize(lengths),
        char_bucket_counts={k: int(v) for k, v in bucket_counts.items()},
        top_chars=top_chars,
        charset_path=str(charset_path) if charset_path else None,
        oov_labels=int(oov_labels),
        oov_chars_top50=oov_top,
    )


def analyze_images(
    lmdb_dir: Path,
    sample_n: int,
    seed: int,
    resize_target: Tuple[int, int],
) -> ImageStats:
    env = lmdb.open(
        str(lmdb_dir),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=64,
    )
    missing = 0
    decode_fail = 0

    fmt_counts: Dict[str, int] = defaultdict(int)
    ch_counts: Dict[str, int] = defaultdict(int)

    widths: List[float] = []
    heights: List[float] = []
    aspects: List[float] = []
    byte_lens: List[float] = []
    mean_ints: List[float] = []
    std_ints: List[float] = []
    lap_vars: List[float] = []
    sx_list: List[float] = []
    sy_list: List[float] = []

    with env.begin(write=False) as txn:
        num_samples = _try_read_num_samples(txn)
        start_lab, _tmpl_lab = _detect_template(txn, LABEL_KEY_CANDIDATES, start_candidates=(0, 1))
        start_img, tmpl_img = _detect_template(txn, IMAGE_KEY_CANDIDATES, start_candidates=(0, 1))
        if num_samples is None or start_lab is None or start_img is None or tmpl_img is None:
            raise RuntimeError(
                f"Cannot detect image keys in lmdb: {lmdb_dir} "
                f"(num-samples={num_samples}, start_label={start_lab}, start_img={start_img}, tmpl_img={tmpl_img})"
            )

        start = start_img
        total = int(num_samples)
        k = min(int(sample_n), total)
        rng = random.Random(int(seed))
        sampled_indices = rng.sample(range(start, start + total), k=k)

        target_w, target_h = int(resize_target[0]), int(resize_target[1])

        for idx in sampled_indices:
            key = (tmpl_img % idx).encode("utf-8")
            val = txn.get(key)
            if val is None:
                missing += 1
                continue
            img_bytes = bytes(val)
            fmt_counts[_sniff_image_format(img_bytes)] += 1
            byte_lens.append(float(len(img_bytes)))

            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                decode_fail += 1
                continue

            h, w = img.shape[:2]
            c = img.shape[2] if img.ndim == 3 else 1
            ch_counts[str(c)] += 1
            widths.append(float(w))
            heights.append(float(h))
            aspects.append(float(w) / float(h) if h > 0 else 0.0)

            mean_ints.append(float(img.mean()))
            std_ints.append(float(img.std()))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            lap_vars.append(float(lap.var()))

            if w > 0 and h > 0:
                sx_list.append(float(target_w) / float(w))
                sy_list.append(float(target_h) / float(h))

    env.close()

    return ImageStats(
        sampled=int(sample_n),
        decode_fail=int(decode_fail),
        missing_image_keys=int(missing),
        format_counts={k: int(v) for k, v in fmt_counts.items()},
        channels_counts={k: int(v) for k, v in ch_counts.items()},
        width=summarize(widths),
        height=summarize(heights),
        aspect=summarize(aspects),
        bytes_len=summarize(byte_lens),
        mean_intensity=summarize(mean_ints),
        std_intensity=summarize(std_ints),
        lap_var=summarize(lap_vars),
        resize_sx_to_256=summarize(sx_list),
        resize_sy_to_64=summarize(sy_list),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze RecogLMDBDataset (labels + sampled images) statistics.")
    ap.add_argument("--lmdb-dir", required=True, help="lmdb dir, e.g. data/fudan/scene/scene_train")
    ap.add_argument("--out", default=None, help="write analysis json to this path")
    ap.add_argument("--charset", default=None, help="charset file path; if set, report OOV stats")
    ap.add_argument("--no-images", action="store_true", help="skip image sampling stats")
    ap.add_argument("--sample-images", type=int, default=20000, help="number of images to sample (default 20000)")
    ap.add_argument("--seed", type=int, default=0, help="random seed for sampling")
    ap.add_argument(
        "--resize-target",
        default="256,64",
        help="target size used by training pipeline for warp resize, default 256,64",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    lmdb_dir = Path(args.lmdb_dir).expanduser().resolve()
    if not lmdb_dir.is_dir():
        raise SystemExit(f"--lmdb-dir not found: {lmdb_dir}")

    charset_path = None
    if args.charset:
        charset_path = Path(args.charset).expanduser().resolve()
        if not charset_path.is_file():
            raise SystemExit(f"--charset not found: {charset_path}")

    try:
        w_str, h_str = str(args.resize_target).split(",", 1)
        resize_target = (int(w_str), int(h_str))
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"--resize-target must be like 256,64 (got {args.resize_target})") from e

    label_stats = analyze_labels(lmdb_dir=lmdb_dir, charset_path=charset_path)
    image_stats = None
    if not args.no_images:
        image_stats = analyze_images(
            lmdb_dir=lmdb_dir,
            sample_n=int(args.sample_images),
            seed=int(args.seed),
            resize_target=resize_target,
        )

    analysis = DatasetAnalysis(
        lmdb_dir=str(lmdb_dir),
        label=label_stats,
        images=image_stats,
    )

    out_json = json.dumps(asdict(analysis), ensure_ascii=False, indent=2)
    print(out_json)
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_json + "\n", encoding="utf-8")
        print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()

