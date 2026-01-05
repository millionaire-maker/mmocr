#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image


@dataclass
class ItemReport:
    src: str
    dst: str
    ok: bool
    skipped: bool = False
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
    if current_max <= max_side:
        return width, height
    scale = max_side / float(current_max)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return new_w, new_h


def _out_path_by_rel(src_dir: Path, dst_dir: Path, src_path: Path) -> Path:
    rel = src_path.relative_to(src_dir)
    return (dst_dir / rel).with_suffix(".jpg")


def process_one(
    src_dir: Path,
    src_path: Path,
    dst_dir: Path,
    max_side: int,
    quality: int = 95,
    overwrite: bool = False,
) -> ItemReport:
    out_path = _out_path_by_rel(src_dir, dst_dir, src_path)
    try:
        with Image.open(src_path) as img:
            src_size = (img.width, img.height)
            img_rgb = _composite_to_rgb(img)
            new_size = _resize_keep_ratio((img_rgb.width, img_rgb.height), max_side)
            if out_path.exists() and not overwrite:
                return ItemReport(
                    src=str(src_path),
                    dst=str(out_path),
                    ok=True,
                    skipped=True,
                    src_size=src_size,
                    dst_size=new_size,
                )

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
    skip_count = sum(1 for r in reports if r.ok and r.skipped)
    with report_path.open("w", encoding="utf-8") as f:
        f.write("背景图预处理报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"src_dir: {src_dir}\n")
        f.write(f"dst_dir: {dst_dir}\n")
        f.write(f"max_side: {max_side}\n")
        f.write(f"total: {len(reports)}\n")
        f.write(f"success: {ok_count}\n")
        f.write(f"skipped: {skip_count}\n")
        f.write(f"failed: {len(reports) - ok_count}\n")
        f.write("\n明细（src_size -> dst_size | status | dst）\n")
        f.write("-" * 60 + "\n")
        for r in reports:
            src_size = f"{r.src_size[0]}x{r.src_size[1]}" if r.src_size else "-"
            dst_size = f"{r.dst_size[0]}x{r.dst_size[1]}" if r.dst_size else "-"
            if r.ok and r.skipped:
                status = "SKIP"
            else:
                status = "OK" if r.ok else f"FAIL: {r.error}"
            f.write(f"{r.src}\t{src_size}\t->\t{dst_size}\t{status}\t{r.dst}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess background images for synth data generation.")
    parser.add_argument("--src", required=True, help="source dir, e.g. data/synth_assets/bg")
    parser.add_argument("--dst", required=True, help="output dir, e.g. data/synth_assets/bg_proc")
    parser.add_argument("--max-side", type=int, default=1280, help="resize longest side to this value")
    parser.add_argument("--report", default=None, help="report path (txt); default: <dst>_report.txt")
    parser.add_argument("--quality", type=int, default=95, help="jpeg quality")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="if set, backup existing dst dir and rebuild from scratch",
    )
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing files in dst dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_dir = Path(args.src).expanduser().resolve()
    dst_dir = Path(args.dst).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve() if args.report else Path(f"{dst_dir}_report.txt")

    if not src_dir.is_dir():
        raise SystemExit(f"--src is not a directory: {src_dir}")
    if args.reset and dst_dir.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = dst_dir.with_name(dst_dir.name + f"_bak_{ts}")
        shutil.move(str(dst_dir), str(backup_dir))
        print(f"[OK] backup existing dst_dir -> {backup_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    reports: List[ItemReport] = []
    for img_path in iter_images(src_dir):
        reports.append(
            process_one(
                src_dir=src_dir,
                src_path=img_path,
                dst_dir=dst_dir,
                max_side=int(args.max_side),
                quality=int(args.quality),
                overwrite=bool(args.overwrite),
            )
        )

    write_report(reports, report_path, src_dir, dst_dir, int(args.max_side))
    ok = sum(1 for r in reports if r.ok)
    skip = sum(1 for r in reports if r.ok and r.skipped)
    print(f"[OK] processed {ok}/{len(reports)} images (skipped={skip})")
    print(f"[OK] outputs: {dst_dir}")
    print(f"[OK] report: {report_path}")


if __name__ == "__main__":
    main()
