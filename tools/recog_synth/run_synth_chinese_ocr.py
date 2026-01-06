import argparse
import os
import shutil
import subprocess
import sys
import unicodedata

import yaml


def normalize_label(text: str) -> str:
    text = text.replace("\ufeff", "").strip()
    return unicodedata.normalize("NFKC", text)


def load_and_patch_config(config_path: str, curve_ratio: float, out_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if curve_ratio >= 0:
        curve_cfg = cfg.get("curve", {})
        curve_cfg["enable"] = curve_ratio > 0
        curve_cfg["fraction"] = float(curve_ratio)
        cfg["curve"] = curve_cfg
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return out_path


def build_cmd(args, config_path: str):
    repo_root = args.repo_root
    cmd = [
        sys.executable,
        os.path.join(repo_root, "main.py"),
        "--num_img",
        str(args.num_images),
        "--length",
        str(args.length),
        "--output_dir",
        args.out_root,
        "--tag",
        args.tag,
        "--config_file",
        config_path,
        "--chars_file",
        args.chars_file,
        "--fonts_list",
        args.fonts_list,
        "--corpus_dir",
        args.corpus_dir,
        "--corpus_mode",
        args.corpus_mode,
        "--bg_dir",
        args.bg_dir,
        "--img_height",
        str(args.img_height),
        "--img_width",
        str(args.img_width),
    ]
    if args.clip_max_chars:
        cmd.append("--clip_max_chars")
    if args.strict:
        cmd.append("--strict")
    if args.num_processes:
        cmd.extend(["--num_processes", str(args.num_processes)])
    return cmd


def _count_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
    return n


def get_existing_samples(save_dir: str) -> int:
    label_candidates = [
        os.path.join(save_dir, "labels.txt"),
        os.path.join(save_dir, "tmp_labels.txt"),
    ]
    for p in label_candidates:
        if os.path.exists(p):
            return _count_lines(p)
    return 0


def convert_labels(save_dir: str, out_root: str, tag: str, out_txt: str):
    label_candidates = [
        os.path.join(save_dir, "tmp_labels.txt"),
        os.path.join(save_dir, "labels.txt"),
    ]
    label_path = None
    for cand in label_candidates:
        if os.path.exists(cand):
            label_path = cand
            break
    if label_path is None:
        raise FileNotFoundError(f"No labels file found under {save_dir}")
    rel_prefix = tag
    count = 0
    with open(label_path, "r", encoding="utf-8") as f, open(
        out_txt, "w", encoding="utf-8"
    ) as out_f:
        for line in f:
            if not line.strip():
                continue
            if " " not in line:
                continue
            img_id, label = line.strip().split(" ", 1)
            label = normalize_label(label)
            rel_path = os.path.join(rel_prefix, f"{img_id}.jpg")
            out_f.write(f"{rel_path}\t{label}\n")
            count += 1
    print(f"Rewrote {count} labels to {out_txt} from {label_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wrapper to run text_renderer for Chinese synthetic OCR data."
    )
    default_repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "3rdparty", "synth_chinese_ocr")
    )
    parser.add_argument(
        "--repo-root",
        default=default_repo_root,
        help="Path to text_renderer repo.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1000,
        help="Number of images to generate.",
    )
    parser.add_argument(
        "--target-num-images",
        type=int,
        default=0,
        help="If >0, generate until total images under the tag reaches this target (resume-safe).",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=10,
        help="Text length hint (used by random/chn/eng mode; harmless for list mode).",
    )
    parser.add_argument(
        "--out-root",
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "synth_rec_ch")
        ),
        help="Output root for generated dataset.",
    )
    parser.add_argument("--tag", default="train", help="Subfolder name under out-root.")
    parser.add_argument(
        "--curve-ratio",
        type=float,
        default=0.0,
        help="Fraction of curved samples (sets curve.fraction in config).",
    )
    parser.add_argument(
        "--chars-file",
        default=None,
        help="Chars file for generator.",
    )
    parser.add_argument(
        "--fonts-list",
        default=None,
        help="Fonts list file.",
    )
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help="Corpus directory.",
    )
    parser.add_argument(
        "--bg-dir",
        default=None,
        help="Background images dir for generator (passed to --bg_dir).",
    )
    parser.add_argument(
        "--corpus-mode",
        default="chn",
        choices=["random", "chn", "eng", "list"],
        help="Corpus mode for generator.",
    )
    parser.add_argument("--img-height", type=int, default=32)
    parser.add_argument("--img-width", type=int, default=0)
    parser.add_argument(
        "--clip-max-chars",
        action="store_true",
        help="Truncate too-long labels to fit fixed img_width (passes --clip_max_chars).",
    )
    parser.add_argument("--strict", action="store_true", help="Enable strict font check.")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=0,
        help="Processes for generation, 0 means use all cores.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing tag folder before generation.",
    )
    parser.add_argument(
        "--base-config",
        default=None,
        help="Base config path to start from.",
    )
    parser.add_argument(
        "--out-txt",
        default=None,
        help="Path to save unified label txt.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.repo_root = os.path.abspath(args.repo_root)
    args.out_root = os.path.abspath(args.out_root)
    if args.chars_file is None:
        args.chars_file = os.path.join(args.repo_root, "data", "chars", "chn.txt")
    if args.fonts_list is None:
        args.fonts_list = os.path.join(args.repo_root, "data", "fonts_list", "chn.txt")
    if args.corpus_dir is None:
        args.corpus_dir = os.path.join(args.repo_root, "data", "corpus")
    if args.bg_dir is None:
        args.bg_dir = os.path.join(args.repo_root, "data", "bg")
    if args.base_config is None:
        args.base_config = os.path.join(args.repo_root, "configs", "default.yaml")
    if args.out_txt is None:
        args.out_txt = os.path.join(args.out_root, f"{args.tag}.txt")

    # Make all paths absolute since we run subprocess under repo_root.
    args.chars_file = os.path.abspath(args.chars_file)
    args.fonts_list = os.path.abspath(args.fonts_list)
    args.corpus_dir = os.path.abspath(args.corpus_dir)
    args.bg_dir = os.path.abspath(args.bg_dir)

    args.out_txt = os.path.abspath(args.out_txt)
    args.base_config = os.path.abspath(args.base_config)

    os.makedirs(args.out_root, exist_ok=True)
    save_dir = os.path.join(args.out_root, args.tag)
    if args.clean and os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    if args.clean and os.path.exists(args.out_txt):
        os.remove(args.out_txt)
    os.makedirs(save_dir, exist_ok=True)

    if int(getattr(args, "target_num_images", 0)) > 0:
        existing = get_existing_samples(save_dir)
        target = int(args.target_num_images)
        remaining = target - existing
        if remaining <= 0:
            print(f"[SKIP] target-num-images satisfied: existing={existing}, target={target}")
            convert_labels(save_dir, args.out_root, args.tag, args.out_txt)
            return
        print(f"[RESUME] existing={existing}, remaining={remaining}, target={target}")
        args.num_images = remaining

    tmp_config = os.path.join(args.out_root, f"{args.tag}_patched.yaml")
    patched_config = load_and_patch_config(args.base_config, args.curve_ratio, tmp_config)

    cmd = build_cmd(args, patched_config)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=args.repo_root, check=True)

    convert_labels(save_dir, args.out_root, args.tag, args.out_txt)


if __name__ == "__main__":
    main()
