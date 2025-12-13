import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from mmocr.utils import dump_ocr_data

DEFAULT_SEED = 42


def safe_symlink(src: Path, dst: Path):
    """Create a symlink and gracefully handle relative paths."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        src_abs = src if src.is_absolute() else (Path.cwd() / src).resolve()
    except FileNotFoundError:
        # resolve may fail on broken links; fall back to absolute path
        src_abs = src if src.is_absolute() else (Path.cwd() / src)
    try:
        rel_src = src_abs.relative_to(dst.parent.resolve())
        dst.symlink_to(rel_src)
    except ValueError:
        dst.symlink_to(src_abs)
    except FileExistsError:
        pass


def split_train_val(
        infos: List[Dict],
        val_ratio: float,
        seed: int = DEFAULT_SEED) -> Tuple[List[Dict], List[Dict]]:
    if val_ratio <= 0 or len(infos) < 2:
        return infos, []
    rnd = random.Random(seed)
    rnd.shuffle(infos)
    val_num = max(1, int(len(infos) * val_ratio))
    val_infos = infos[:val_num]
    train_infos = infos[val_num:]
    return train_infos, val_infos


def rewrite_and_link(
    infos: Iterable[Dict],
    src_root: Path,
    out_root: Path,
    source_tag: str = '',
) -> List[Dict]:
    rewritten = []
    for info in infos:
        new_info = dict(info)
        fname = Path(info['file_name'])
        if fname.is_absolute():
            src_img = fname
        else:
            src_img = src_root / fname
            if not src_img.exists() and (src_root / 'train').exists():
                candidate = src_root / 'train' / fname
                if candidate.exists():
                    src_img = candidate
        dst_img = out_root / fname.name
        safe_symlink(src_img, dst_img)
        new_info['file_name'] = str(Path('imgs') / dst_img.name)
        if source_tag:
            new_info['source'] = source_tag
        rewritten.append(new_info)
    return rewritten


def dump_split(train_infos: List[Dict],
               val_infos: List[Dict],
               out_dir: Path,
               task: str = 'textdet'):
    out_dir.mkdir(parents=True, exist_ok=True)
    train_json = out_dir / 'instances_train.json'
    val_json = out_dir / 'instances_val.json'
    dump_ocr_data(train_infos, str(train_json), task)
    if val_infos:
        dump_ocr_data(val_infos, str(val_json), task)
    else:
        val_json.write_text('{"metainfo": {"dataset_type": "TextDetDataset", '
                            '"task_name": "textdet", "category": [{"id": 0, '
                            '"name": "text"}]}, "data_list": []}\\n')
    return train_json, val_json
