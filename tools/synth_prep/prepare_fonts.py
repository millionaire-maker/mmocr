#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import shutil
import sys
import urllib.error
import urllib.request
from urllib.parse import unquote
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from fontTools.ttLib import TTCollection, TTFont
from PIL import ImageFont


GOOGLE_FONTS: Sequence[Tuple[str, str, str]] = (
    # name, url, license
    ("NotoSansSC-wght", "https://raw.githubusercontent.com/google/fonts/main/ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf", "SIL OFL 1.1"),
    ("NotoSerifSC-wght", "https://raw.githubusercontent.com/google/fonts/main/ofl/notoserifsc/NotoSerifSC%5Bwght%5D.ttf", "SIL OFL 1.1"),
    ("ZCOOLKuaiLe-Regular", "https://github.com/google/fonts/raw/main/ofl/zcoolkuaile/ZCOOLKuaiLe-Regular.ttf", "SIL OFL 1.1"),
    ("ZCOOLQingKeHuangYou-Regular", "https://github.com/google/fonts/raw/main/ofl/zcoolqingkehuangyou/ZCOOLQingKeHuangYou-Regular.ttf", "SIL OFL 1.1"),
    ("ZCOOLXiaoWei-Regular", "https://github.com/google/fonts/raw/main/ofl/zcoolxiaowei/ZCOOLXiaoWei-Regular.ttf", "SIL OFL 1.1"),
    ("MaShanZheng-Regular", "https://github.com/google/fonts/raw/main/ofl/mashanzheng/MaShanZheng-Regular.ttf", "SIL OFL 1.1"),
    ("LiuJianMaoCao-Regular", "https://github.com/google/fonts/raw/main/ofl/liujianmaocao/LiuJianMaoCao-Regular.ttf", "SIL OFL 1.1"),
    ("LongCang-Regular", "https://github.com/google/fonts/raw/main/ofl/longcang/LongCang-Regular.ttf", "SIL OFL 1.1"),
    ("ZhiMangXing-Regular", "https://github.com/google/fonts/raw/main/ofl/zhimangxing/ZhiMangXing-Regular.ttf", "SIL OFL 1.1"),
)


RECOMMENDED_MANUAL_FONTS: Sequence[str] = (
    "LXGW WenKai / 霞鹜文楷 (SIL OFL 1.1)",
    "Smiley Sans / 得意黑 (SIL OFL 1.1)",
    "Source Han Sans/Serif SC / 思源黑体/思源宋体 (SIL OFL 1.1)",
    "WenQuanYi Zen Hei / 文泉驿正黑 (GPL/开放字体许可，需自行确认商业合规)",
)


@dataclass
class FontItem:
    path: str
    source: str
    license: str
    ok_load_pil: bool
    sample_size: int
    missing: int
    missing_ratio: float
    kept: bool
    error: Optional[str] = None


def read_charset(charset_path: Path) -> List[str]:
    chars: List[str] = []
    for line in charset_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        chars.append(line[0])
    return chars


def download_file(url: str, dst_path: Path, timeout: int = 60) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_bytes(data)


def ensure_download_fonts(dst_dir: Path) -> Tuple[List[Path], List[str]]:
    downloaded: List[Path] = []
    errors: List[str] = []
    for name, url, _license in GOOGLE_FONTS:
        fname = os.path.basename(unquote(url))
        if not fname:
            fname = f"{name}.ttf"
        out_path = dst_dir / fname
        if out_path.exists() and out_path.stat().st_size > 0:
            downloaded.append(out_path)
            continue
        try:
            download_file(url, out_path)
            downloaded.append(out_path)
        except urllib.error.URLError as exc:
            errors.append(f"download failed: {url} ({exc})")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"download failed: {url} ({exc})")
    return downloaded, errors


def iter_system_fonts() -> Iterable[Path]:
    roots = [
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
        Path.home() / ".local/share/fonts",
    ]
    exts = {".ttf", ".otf", ".ttc", ".TTF", ".OTF", ".TTC"}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in exts:
                yield path


def pick_local_fonts(candidates_limit: int = 200) -> List[Path]:
    # Heuristic: pick fonts whose filename hints Chinese/CJK.
    keywords = (
        "cjk",
        "CJK",
        "NotoSansCJK",
        "NotoSerifCJK",
        "NotoSansSC",
        "NotoSerifSC",
        "SourceHan",
        "WenQuanYi",
        "DroidSansFallback",
        "wqy",
        "ukai",
        "uming",
        "simhei",
        "simsun",
        "kaiti",
        "fangsong",
        "heiti",
        "songti",
        "yuanti",
        "wenkai",
    )
    out: List[Path] = []
    for p in iter_system_fonts():
        fn = p.name
        if any(k in fn for k in keywords):
            out.append(p)
            if len(out) >= candidates_limit:
                break
    return sorted(set(out))


def link_or_copy(src: Path, dst: Path, mode: str) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    if mode == "symlink":
        dst.symlink_to(src)
        return dst
    if mode == "copy":
        shutil.copy2(src, dst)
        return dst
    raise ValueError(f"unknown mode: {mode}")


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
    codepoints: Set[int] = set()
    for table in ttf["cmap"].tables:
        codepoints.update(table.cmap.keys())
    ttf.close()
    return codepoints


def check_one_font(
    font_path: Path,
    sample_chars: Sequence[str],
    missing_ratio_thr: float,
) -> Tuple[bool, bool, int, float, Optional[str]]:
    # 1) PIL load check
    try:
        _ = ImageFont.truetype(str(font_path), 32)
        ok_pil = True
    except Exception as exc:  # noqa: BLE001
        return False, False, len(sample_chars), 1.0, f"PIL load failed: {exc}"

    # 2) cmap coverage check
    try:
        cps = _load_codepoints(font_path)
    except Exception as exc:  # noqa: BLE001
        return ok_pil, False, len(sample_chars), 1.0, f"fontTools load failed: {exc}"

    missing = 0
    for ch in sample_chars:
        if ord(ch) not in cps:
            missing += 1
    ratio = missing / max(1, len(sample_chars))
    kept = ratio <= missing_ratio_thr
    return ok_pil, True, missing, ratio, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Chinese fonts and generate fonts_list/chn.txt")
    parser.add_argument("--charset", required=True, help="charset file path")
    parser.add_argument("--dst-dir", required=True, help="dst font dir, e.g. data/synth_assets/fonts/chn")
    parser.add_argument("--list-out", required=True, help="fonts list output path, 1 abs path per line")
    parser.add_argument("--report", required=True, help="report json path")
    parser.add_argument("--mode", default="auto", choices=["auto", "download", "local"], help="download or local scan")
    parser.add_argument("--link-mode", default="copy", choices=["copy", "symlink"], help="copy or symlink local fonts")
    parser.add_argument("--sample-size", type=int, default=1000, help="sample charset size for missing check")
    parser.add_argument("--missing-ratio-thr", type=float, default=0.10, help="drop font if missing_ratio > thr")
    parser.add_argument("--seed", type=int, default=0, help="random seed for sampling charset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    charset_path = Path(args.charset).expanduser().resolve()
    dst_dir = Path(args.dst_dir).expanduser().resolve()
    list_out = Path(args.list_out).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()

    if not charset_path.is_file():
        raise SystemExit(f"--charset not found: {charset_path}")

    charset_chars = read_charset(charset_path)
    if not charset_chars:
        raise SystemExit(f"empty charset: {charset_path}")

    rng = random.Random(int(args.seed))
    sample_size = min(int(args.sample_size), len(charset_chars))
    sample_chars = rng.sample(charset_chars, sample_size)

    dst_dir.mkdir(parents=True, exist_ok=True)

    used_mode = args.mode
    download_errors: List[str] = []
    font_candidates: List[Tuple[Path, str, str]] = []

    if args.mode in ("auto", "download"):
        downloaded, errs = ensure_download_fonts(dst_dir)
        download_errors.extend(errs)
        for _name, url, lic in GOOGLE_FONTS:
            fname = os.path.basename(unquote(url)) or f"{_name}.ttf"
            p = (dst_dir / fname).resolve()
            if p.exists() and p.stat().st_size > 0:
                font_candidates.append((p, url, lic))
        if args.mode == "auto" and not font_candidates:
            used_mode = "local"

    if used_mode == "local":
        picked = pick_local_fonts()
        for src in picked:
            dst = dst_dir / src.name
            try:
                out = link_or_copy(src, dst, args.link_mode).resolve()
                font_candidates.append((out, f"local:{src}", "UNKNOWN"))
            except Exception as exc:  # noqa: BLE001
                download_errors.append(f"local import failed: {src} ({exc})")

    # de-dup candidates by realpath
    uniq: Dict[str, Tuple[Path, str, str]] = {}
    for p, src, lic in font_candidates:
        uniq[str(p)] = (p, src, lic)
    font_candidates = list(uniq.values())

    items: List[FontItem] = []
    kept_paths: List[str] = []

    for font_path, source, lic in sorted(font_candidates, key=lambda x: x[0].name.lower()):
        ok_pil, ok_cmap, missing, ratio, err = check_one_font(
            font_path=font_path,
            sample_chars=sample_chars,
            missing_ratio_thr=float(args.missing_ratio_thr),
        )
        kept = ok_pil and ok_cmap and (ratio <= float(args.missing_ratio_thr))
        item = FontItem(
            path=str(font_path),
            source=source,
            license=lic,
            ok_load_pil=ok_pil,
            sample_size=sample_size,
            missing=missing,
            missing_ratio=ratio,
            kept=kept,
            error=err,
        )
        items.append(item)
        if kept:
            kept_paths.append(str(font_path))

    list_out.parent.mkdir(parents=True, exist_ok=True)
    list_out.write_text("\n".join(kept_paths) + ("\n" if kept_paths else ""), encoding="utf-8")

    report = {
        "charset": str(charset_path),
        "sample_size": sample_size,
        "missing_ratio_thr": float(args.missing_ratio_thr),
        "mode": used_mode,
        "download_errors": download_errors,
        "recommended_manual_fonts": list(RECOMMENDED_MANUAL_FONTS),
        "fonts_total": len(items),
        "fonts_kept": len(kept_paths),
        "fonts": [asdict(it) for it in items],
        "fonts_list": str(list_out),
        "fonts_dir": str(dst_dir),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] fonts kept: {len(kept_paths)}/{len(items)}")
    print(f"[OK] fonts dir: {dst_dir}")
    print(f"[OK] fonts list: {list_out}")
    print(f"[OK] report: {report_path}")
    if download_errors:
        print("[WARN] some download/import errors happened, see report.json")


if __name__ == "__main__":
    main()
