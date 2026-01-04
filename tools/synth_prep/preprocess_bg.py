#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image


@dataclass
class ItemReport:
    src: str
    dst: str
    ok: bool
    src_size: Optional[Tuple[int, int]] = None
    dst_size: Optional[Tuple[int, int]] = None
    error: Optional[str] = None


def iter_images(src_dir: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    for path in sorted(src_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def _composite_to_rgb(img: Image.Image, bg_color=(255, 255, 255)) -> Image.Image:
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, bg_color + (255,))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return img.convert("RGB")


def _resize_keep_ratio(size: Tuple[int, int], max_side: int) -> Tuple[int, int]:
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {size}")
    current_max = max(width, height)
    scale = max_side / float(current_max)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return new_w, new_h


def _safe_out_path(dst_dir: Path, src_path: Path) -> Path:
    base = src_path.stem
    out = dst_dir / f"{base}.jpg"
    if not out.exists():
        return out
    for i in range(1, 10000):
        cand = dst_dir / f"{base}_{i}.jpg"
        if not cand.exists():
            return cand
    raise RuntimeError(f"cannot find available output name for {src_path}")


def process_one(src_path: Path, dst_dir: Path, max_side: int, quality: int = 95) -> ItemReport:
    out_path = _safe_out_path(dst_dir, src_path)
    try:
        with Image.open(src_path) as img:
            src_size = (img.width, img.height)
            img_rgb = _composite_to_rgb(img)
            new_size = _resize_keep_ratio((img_rgb.width, img_rgb.height), max_side)
            if new_size != (img_rgb.width, img_rgb.height):
                img_rgb = img_rgb.resize(new_size, Image.Resampling.LANCZOS)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img_rgb.save(out_path, format="JPEG", quality=quality, optimize=True)
        return ItemReport(
            src=str(src_path),
            dst=str(out_path),
            ok=True,
            src_size=src_size,
            dst_size=new_size,
        )
    except Exception as exc:  # noqa: BLE001
        return ItemReport(src=str(src_path), dst=str(out_path), ok=False, error=str(exc))


def write_report(reports: List[ItemReport], report_path: Path, src_dir: Path, dst_dir: Path, max_side: int) -> None:
    ok_count = sum(1 for r in reports if r.ok)
    with report_path.open("w", encoding="utf-8") as f:
        f.write("背景图预处理报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"src_dir: {src_dir}\n")
        f.write(f"dst_dir: {dst_dir}\n")
        f.write(f"max_side: {max_side}\n")
        f.write(f"total: {len(reports)}\n")
        f.write(f"success: {ok_count}\n")
        f.write(f"failed: {len(reports) - ok_count}\n")
        f.write("\n明细（src_size -> dst_size | status | dst）\n")
        f.write("-" * 60 + "\n")
        for r in reports:
            src_size = f"{r.src_size[0]}x{r.src_size[1]}" if r.src_size else "-"
            dst_size = f"{r.dst_size[0]}x{r.dst_size[1]}" if r.dst_size else "-"
            status = "OK" if r.ok else f"FAIL: {r.error}"
            f.write(f"{r.src}\t{src_size}\t->\t{dst_size}\t{status}\t{r.dst}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess background images for synth data generation.")
    parser.add_argument("--src", required=True, help="source dir, e.g. data/synth_assets/bg")
    parser.add_argument("--dst", required=True, help="output dir, e.g. data/synth_assets/bg_proc")
    parser.add_argument("--max-side", type=int, default=1280, help="resize longest side to this value")
    parser.add_argument("--report", default=None, help="report path (txt); default: <dst>_report.txt")
    parser.add_argument("--quality", type=int, default=95, help="jpeg quality")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_dir = Path(args.src).expanduser().resolve()
    dst_dir = Path(args.dst).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve() if args.report else Path(f"{dst_dir}_report.txt")

    if not src_dir.is_dir():
        raise SystemExit(f"--src is not a directory: {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    reports: List[ItemReport] = []
    for img_path in iter_images(src_dir):
        reports.append(process_one(img_path, dst_dir, max_side=int(args.max_side), quality=int(args.quality)))

    write_report(reports, report_path, src_dir, dst_dir, int(args.max_side))
    print(f"[OK] processed {sum(1 for r in reports if r.ok)}/{len(reports)} images")
    print(f"[OK] outputs: {dst_dir}")
    print(f"[OK] report: {report_path}")


if __name__ == "__main__":
    main()
