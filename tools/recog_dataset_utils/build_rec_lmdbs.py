import argparse
import json
import os
import shutil
import subprocess
import sys


def txt_to_jsonl(txt_path: str, jsonl_path: str):
    with open(txt_path, "r", encoding="utf-8") as f_in, open(
        jsonl_path, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            img_path, label = parts
            record = {"filename": img_path, "text": label}
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")


def ensure_output_dir(path: str, overwrite: bool):
    if os.path.exists(path):
        if overwrite:
            shutil.rmtree(path)
        else:
            raise FileExistsError(f"{path} already exists.")
    os.makedirs(path, exist_ok=True)


def run_lmdb_converter(repo_root: str, jsonl_path: str, lmdb_out: str, img_root: str):
    converter = os.path.join(repo_root, "tools/dataset_converters/textrecog/lmdb_converter.py")
    cmd = [
        sys.executable,
        converter,
        jsonl_path,
        lmdb_out,
        "--img-root",
        img_root,
        "--label-format",
        "jsonl",
        "--batch-size",
        "2000",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)


def build_from_txt(label_txt: str, lmdb_out: str, img_root: str, repo_root: str, overwrite: bool):
    if not os.path.exists(label_txt):
        raise FileNotFoundError(f"Label file not found: {label_txt}")
    ensure_output_dir(lmdb_out, overwrite)
    jsonl_path = os.path.splitext(label_txt)[0] + ".jsonl"
    txt_to_jsonl(label_txt, jsonl_path)
    run_lmdb_converter(repo_root, jsonl_path, lmdb_out, img_root)


def combine_txt(inputs, output, repo_root):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as out_f:
        for txt in inputs:
            base_dir = os.path.dirname(txt)
            with open(txt, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    img_rel, label = line.rstrip("\n").split("\t", 1)
                    if not os.path.isabs(img_rel):
                        abs_path = os.path.normpath(os.path.join(base_dir, img_rel))
                    else:
                        abs_path = img_rel
                    rel_repo = os.path.relpath(abs_path, repo_root)
                    out_f.write(f"{rel_repo}\t{label}\n")
    return output


def ensure_symlink_or_copy(source: str, target: str, overwrite: bool):
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source LMDB not found: {source}")
    if os.path.exists(target):
        if overwrite:
            if os.path.islink(target):
                os.unlink(target)
            else:
                shutil.rmtree(target)
        else:
            print(f"Target exists, skip: {target}")
            return target
    os.makedirs(os.path.dirname(target), exist_ok=True)
    try:
        os.symlink(source, target)
        print(f"Created symlink {target} -> {source}")
    except OSError:
        shutil.copytree(source, target)
        print(f"Copied LMDB {source} to {target}")
    return target


def handle_fudan_train(repo_root: str, overwrite: bool):
    source = os.path.join(repo_root, "data/fudan/scene/scene_train")
    target = os.path.join(repo_root, "data/recog_lmdb/fudan_scene_train.lmdb")
    return ensure_symlink_or_copy(source, target, overwrite)


def handle_fudan_all(repo_root: str, overwrite: bool):
    splits = {
        "train": "scene_train",
        "val": "scene_val",
        "test": "scene_test",
    }
    targets = []
    for split, folder in splits.items():
        source = os.path.join(repo_root, "data/fudan/scene", folder)
        target = os.path.join(repo_root, f"data/recog_lmdb/fudan_scene_{split}.lmdb")
        targets.append(ensure_symlink_or_copy(source, target, overwrite))
    return targets


def parse_args():
    parser = argparse.ArgumentParser(description="Build recognition LMDBs via lmdb_converter.")
    parser.add_argument(
        "--target",
        required=True,
        choices=[
            "fudan_scene_train",
            "fudan_scene_all",
            "mega_train_from_raw",
            "synth_rec_ch_train",
            "mega_train_all",
        ],
        help="Which LMDB to build.",
    )
    parser.add_argument(
        "--label-txt",
        help="Optional label txt override (img_path<TAB>label).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing LMDB before building.",
    )
    parser.add_argument(
        "--repo-root",
        default="/home/yzy/mmocr",
        help="MMOCR repo root.",
    )
    parser.add_argument(
        "--img-root",
        default=None,
        help="Optional override for image root used by lmdb_converter.",
    )
    parser.add_argument(
        "--extra-inputs",
        nargs="+",
        default=[],
        help="Extra txt inputs when target=mega_train_all.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = args.repo_root
    os.makedirs(os.path.join(repo_root, "data/recog_lmdb"), exist_ok=True)

    if args.target == "fudan_scene_train":
        handle_fudan_train(repo_root, args.overwrite)
        return
    if args.target == "fudan_scene_all":
        handle_fudan_all(repo_root, args.overwrite)
        return

    if args.target == "mega_train_from_raw":
        label_txt = args.label_txt or os.path.join(repo_root, "data/recog_mega/mega_train_from_raw.txt")
        lmdb_out = os.path.join(repo_root, "data/recog_lmdb/mega_train_from_raw.lmdb")
        img_root = args.img_root or repo_root
        build_from_txt(label_txt, lmdb_out, img_root, repo_root, args.overwrite)
    elif args.target == "synth_rec_ch_train":
        label_txt = args.label_txt or os.path.join(repo_root, "data/synth_rec_ch/train.txt")
        lmdb_out = os.path.join(repo_root, "data/recog_lmdb/synth_rec_ch_train.lmdb")
        img_root = args.img_root or os.path.join(repo_root, "data/synth_rec_ch")
        build_from_txt(label_txt, lmdb_out, img_root, repo_root, args.overwrite)
    elif args.target == "mega_train_all":
        base_inputs = [
            args.label_txt or os.path.join(repo_root, "data/recog_mega/mega_train_from_raw.txt"),
            os.path.join(repo_root, "data/synth_rec_ch/train.txt"),
        ]
        base_inputs.extend(args.extra_inputs)
        combined_txt = os.path.join(repo_root, "data/recog_mega/mega_train_all.txt")
        combine_txt(base_inputs, combined_txt, repo_root)
        lmdb_out = os.path.join(repo_root, "data/recog_lmdb/mega_train_all.lmdb")
        img_root = args.img_root or repo_root
        build_from_txt(combined_txt, lmdb_out, img_root, repo_root, args.overwrite)


if __name__ == "__main__":
    main()
