
import os
import json
import numpy as np
import pandas as pd
import mmengine
from pathlib import Path
from mmengine.config import Config
from mmocr.registry import DATASETS, METRICS
from mmengine.registry import init_default_scope
from mmocr.utils import register_all_modules
from torch.utils.data import DataLoader
from tqdm import tqdm
from mmengine.structures import InstanceData
from mmocr.structures import TextDetDataSample

# --- Configuration ---
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = str(REPO_ROOT / "outputs" / "cascade_demo" / "run004" / "fullval_G3_vs_G5")
CONFIG_PATH = str(
    REPO_ROOT
    / "configs"
    / "textdet"
    / "fcenet"
    / "fcenet_r50dcnv2_fpn_1500e_art_rctw_rects_finetune.py"
)

def load_preds(path):
    print(f"Loading preds from {path}...")
    data = mmengine.load(path)
    # Convert to dict for fast lookup
    pred_dict = {x['img_path']: x for x in data}
    return pred_dict

def run_eval(pred_dict, dataloader, metric_name):
    # Build Metric
    # We use MultiDatasetHmeanIOUMetric from config
    # But checking config, it needs 'dataset_prefixes' to work automatically
    # We can perform the "Splitting" based on img_path manually if needed, 
    # OR we can just use the config's metric definition which SHOULD handle it.
    
    cfg = Config.fromfile(CONFIG_PATH)
    evaluator = METRICS.build(cfg.val_evaluator)
    
    print(f"Running Evaluation ({metric_name})...")
    
    for batch in tqdm(dataloader):
        # Batch is list of TextDetDataSample (with GT) due to collate_fn=lambda x:x
        # But wait, DataLoader yields what?
        # If I use the same dataloader setup as H1
        item = batch[0]
        data_sample_gt = item['data_samples']
        img_path = data_sample_gt.img_path
        
        # Get Pred
        if img_path in pred_dict:
            pred_info = pred_dict[img_path]
            polys = pred_info['polygons']
            scores = pred_info['scores']
        else:
            polys = []
            scores = []
            
        # Construct Pred InstanceData
        pred_instances = InstanceData()
        pred_instances.polygons = convert_polys_to_numpy_list(polys) # list of np array
        pred_instances.scores = np.array(scores, dtype=np.float32)
        
        # Inject into data_sample (creating a copy/proxy)
        # The metric process expects data_samples to have both gt_instances AND pred_instances
        data_sample_gt.pred_instances = pred_instances
        
        # Process
        evaluator.process(None, [data_sample_gt])
        
    metrics = evaluator.evaluate(len(dataloader.dataset))
    return metrics

def convert_polys_to_numpy_list(polys):
    out = []
    for p in polys:
        # p is list suitable for json. convert to np
        out.append(np.array(p).reshape(-1, 2))
    return out

def main():
    register_all_modules(init_default_scope=True)
    init_default_scope('mmocr')

    g3_pred_path = os.path.join(OUT_ROOT, "G3/preds.pkl")
    g5_pred_path = os.path.join(OUT_ROOT, "G5/preds.pkl")
    if not (os.path.exists(g3_pred_path) and os.path.exists(g5_pred_path)):
        print("Preds not found, please run step_H_export/h1_run_inference.py first.")
        print(f"Missing: {g3_pred_path} or {g5_pred_path}")
        return
    
    cfg = Config.fromfile(CONFIG_PATH)
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: x)
    
    # Run G3
    g3_preds = load_preds(g3_pred_path)
    g3_metrics = run_eval(g3_preds, dataloader, "G3")
    with open(os.path.join(OUT_ROOT, "G3/metrics_overall.json"), 'w') as f:
        json.dump(g3_metrics, f, indent=2)
        
    # Run G5
    g5_preds = load_preds(g5_pred_path)
    g5_metrics = run_eval(g5_preds, dataloader, "G5")
    with open(os.path.join(OUT_ROOT, "G5/metrics_overall.json"), 'w') as f:
        json.dump(g5_metrics, f, indent=2)
        
    # Generate CSVs
    # Compare Overall
    overall_data = []
    for name, m in [("G3", g3_metrics), ("G5", g5_metrics)]:
        overall_data.append({
            "Method": name,
            "Hmean": m.get('icdar/hmean', 0),
            "Precision": m.get('icdar/precision', 0),
            "Recall": m.get('icdar/recall', 0)
        })
    df_overall = pd.DataFrame(overall_data)
    df_overall.to_csv(os.path.join(OUT_ROOT, "compare_overall.csv"), index=False)
    
    # Compare By Dataset (ArT, RCTW, ReCTS)
    # The keys in metrics look like 'art/hmean', 'rctw/hmean' etc.
    dataset_data = []
    prefixes = ['art', 'rctw', 'rects']
    for prefix in prefixes:
        for name, m in [("G3", g3_metrics), ("G5", g5_metrics)]:
            dataset_data.append({
                "Dataset": prefix.upper(),
                "Method": name,
                "Hmean": m.get(f'{prefix}/hmean', 0),
                "Precision": m.get(f'{prefix}/precision', 0),
                "Recall": m.get(f'{prefix}/recall', 0)
            })
    df_dataset = pd.DataFrame(dataset_data)
    df_dataset.to_csv(os.path.join(OUT_ROOT, "compare_by_dataset.csv"), index=False)
    
    print("Metrics Computed.")

if __name__ == "__main__":
    main()
