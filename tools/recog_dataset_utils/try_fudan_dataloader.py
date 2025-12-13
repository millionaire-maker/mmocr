import argparse
from copy import deepcopy

from mmengine.config import Config
from mmengine.runner import Runner
from mmocr.utils import register_all_modules


def clamp_dataset_indices(dataset_cfg, max_num=64):
    """Limit dataset indices recursively to keep sanity check fast."""
    if not isinstance(dataset_cfg, dict):
        return
    ds_type = dataset_cfg.get("type")
    if ds_type == "ConcatDataset":
        for sub_ds in dataset_cfg.get("datasets", []):
            clamp_dataset_indices(sub_ds, max_num)
    elif ds_type == "RepeatDataset":
        clamp_dataset_indices(dataset_cfg.get("dataset", {}), max_num)
    else:
        dataset_cfg.setdefault("indices", list(range(max_num)))


def main():
    parser = argparse.ArgumentParser(description="Sanity check Fudan LMDB dataloader.")
    parser.add_argument(
        "--config",
        default="configs/textrecog/svtr/svtr-small_fudan_scene_baseline.py",
        help="Config file to build the dataloader.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=2,
        help="Number of batches to iterate for sanity check.",
    )
    args = parser.parse_args()

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    cfg.setdefault("work_dir", "./work_dirs/tmp_try_fudan_loader")

    cfg.train_dataloader = deepcopy(cfg.train_dataloader)
    cfg.train_dataloader["batch_size"] = min(
        cfg.train_dataloader.get("batch_size", 8), 8
    )
    cfg.train_dataloader["num_workers"] = 0
    cfg.train_dataloader["persistent_workers"] = False
    dataset_cfg = cfg.train_dataloader["dataset"]
    if dataset_cfg.get("pipeline") is None:
        dataset_cfg["pipeline"] = cfg.train_pipeline
    # Use a smaller split for quick check when LMDB is huge.
    if dataset_cfg.get("ann_file") == "fudan_scene_train.lmdb":
        dataset_cfg["ann_file"] = "fudan_scene_val.lmdb"
    if dataset_cfg.get("type") == "ConcatDataset":
        for sub_ds in dataset_cfg.get("datasets", []):
            if sub_ds.get("type") == "RepeatDataset":
                inner_ds = sub_ds.get("dataset", {})
                if inner_ds.get("ann_file") == "fudan_scene_train.lmdb":
                    inner_ds["ann_file"] = "fudan_scene_val.lmdb"
    clamp_dataset_indices(dataset_cfg, max_num=64)

    print(
        f"[Info] Using config {args.config}, "
        f"batch_size={cfg.train_dataloader['batch_size']}, "
        f"num_workers={cfg.train_dataloader['num_workers']}"
    )

    runner = Runner.from_cfg(cfg)
    print("[Info] Runner built, constructing dataloader...")
    dataloader = runner.build_dataloader(cfg.train_dataloader)
    print("[Info] Start iterating dataloader")
    data_iter = iter(dataloader)
    for idx in range(args.num_batches):
        batch = next(data_iter)
        data_samples = batch["data_samples"]
        labels = [sample.gt_text.item for sample in data_samples]
        print(f"[Batch {idx+1}] batch_size={len(data_samples)}, first_label={labels[0]}")


if __name__ == "__main__":
    main()
