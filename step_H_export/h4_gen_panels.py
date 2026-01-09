
import os
import json
import cv2
import numpy as np
import mmengine
from pathlib import Path
from mmengine.config import Config
from mmocr.registry import MODELS
from mmengine.runner import load_checkpoint
from mmocr.utils import register_all_modules
from h1_run_inference import PolicyRunner, run_model_inference, clean_s1_polygons, _poly_points
from tqdm import tqdm

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
DBNET_CONFIG = str(
    REPO_ROOT
    / "work_dirs"
    / "dbnetpp_r50_finetune_art_rctw_rects"
    / "dbnetpp_resnet50_fpnc_1200e_art_rctw_rects_finetune.py"
)
DBNET_CKPT = str(
    REPO_ROOT
    / "work_dirs"
    / "dbnetpp_r50_finetune_art_rctw_rects"
    / "best_icdar_hmean_epoch_93.pth"
)
FCENET_CKPT = str(
    REPO_ROOT
    / "work_dirs"
    / "fcenet_r50dcnv2_fpn_finetune_art_rctw_rects"
    / "best_icdar_hmean_epoch_96.pth"
)

def put_text_box(img, txt, col):
    cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)
    # Add box count if possible?

def draw_poly(img, poly, color):
    pts = _poly_points(poly).astype(np.int32)
    cv2.polylines(img, [pts], True, color, 2)

def main():
    register_all_modules(init_default_scope=True)
    
    # Load List
    interest_path = os.path.join(OUT_ROOT, "interest_list.json")
    if not os.path.exists(interest_path):
        print("Interest list not found, run H1 first.")
        return
        
    interest_data = mmengine.load(interest_path)
    # Merge lists
    img_paths = set()
    img_paths.update(interest_data['top_s2_acc'])
    img_paths.update(interest_data['random_art'])
    img_list = list(img_paths)
    
    # Load Models
    print("Loading Models...")
    cfg = Config.fromfile(CONFIG_PATH)
    cfg_s1 = Config.fromfile(DBNET_CONFIG)
    model_s1 = MODELS.build(cfg_s1.model)
    load_checkpoint(model_s1, DBNET_CKPT, map_location='cpu')
    if hasattr(model_s1.det_head, 'postprocessor'):
        model_s1.det_head.postprocessor.text_repr_type = 'poly'
    model_s1.cuda().eval()
    
    model_s2 = MODELS.build(cfg.model)
    load_checkpoint(model_s2, FCENET_CKPT, map_location='cpu')
    model_s2.cuda().eval()
    
    g3_runner = PolicyRunner('G3', model_s2)
    g5_runner = PolicyRunner('G5', model_s2)
    
    out_dirs = {
        'G3': os.path.join(OUT_ROOT, "viz/G3/panels"),
        'G5': os.path.join(OUT_ROOT, "viz/G5/panels"),
        'SBS': os.path.join(OUT_ROOT, "viz/side_by_side")
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)
        
    print(f"Generating Panels for {len(img_list)} images...")
    
    for img_path in tqdm(img_list):
        if not os.path.exists(img_path): continue
        bn = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        # Prepare Visuals
        v_orig = img.copy()
        v_db_raw = img.copy()
        v_db_clean = img.copy()
        v_fce = img.copy()
        v_g3 = img.copy()
        v_g5 = img.copy()
        
        # S1
        s1_polys_raw, s1_scores_raw = run_model_inference(model_s1, img)
        s1_polys = []
        for p in s1_polys_raw: # Robust loop
             if hasattr(p, 'cpu'): s1_polys.append(p.cpu().numpy().tolist())
             elif isinstance(p, np.ndarray): s1_polys.append(p.tolist())
             else: s1_polys.append(p)
             
        if hasattr(s1_scores_raw, 'cpu'): s1_scores = s1_scores_raw.cpu().numpy().tolist()
        elif isinstance(s1_scores_raw, np.ndarray): s1_scores = s1_scores_raw.tolist()
        else: s1_scores = list(s1_scores_raw)

        for p in s1_polys: draw_poly(v_db_raw, p, (255, 100, 0))
        
        # Clean S1
        s1_clean, s1_clean_sc, _, _ = clean_s1_polygons(s1_polys, s1_scores)
        for p in s1_clean: draw_poly(v_db_clean, p, (255, 255, 0))
        
        # S2
        s2_polys_full_raw, _ = run_model_inference(model_s2, img)
        s2_polys_full = []
        for p in s2_polys_full_raw:
             if hasattr(p, 'cpu'): s2_polys_full.append(p.cpu().numpy().tolist())
             elif isinstance(p, np.ndarray): s2_polys_full.append(p.tolist())
             else: s2_polys_full.append(p)
             
        for p in s2_polys_full: draw_poly(v_fce, p, (0, 255, 0))
        
        # G3
        g3_polys, _, _, _ = g3_runner.process_image(img, s1_clean, s1_clean_sc, s2_polys_full)
        for p in g3_polys: draw_poly(v_g3, p, (0, 0, 255))
        
        # G5
        g5_polys, _, _, _ = g5_runner.process_image(img, s1_clean, s1_clean_sc, s2_polys_full)
        for p in g5_polys: draw_poly(v_g5, p, (0, 0, 255))
        
        # Assemble Panels
        put_text_box(v_orig, "Original", (255,255,255))
        put_text_box(v_db_raw, "DBNet-Raw", (255,100,0))
        put_text_box(v_db_clean, "DBNet-Clean", (255,255,0))
        put_text_box(v_fce, "FCENet-Full", (0,255,0))
        put_text_box(v_g3, "Final-G3", (0,0,255))
        put_text_box(v_g5, "Final-G5", (0,0,255))
        
        panel_g3 = np.hstack([v_orig, v_db_raw, v_db_clean, v_fce, v_g3])
        panel_g5 = np.hstack([v_orig, v_db_raw, v_db_clean, v_fce, v_g5])
        
        cv2.imwrite(os.path.join(out_dirs['G3'], f"{bn}_panel.jpg"), panel_g3)
        cv2.imwrite(os.path.join(out_dirs['G5'], f"{bn}_panel.jpg"), panel_g5)
        
        # Side-by-Side: Original | Final-G3 | Final-G5
        # Or more detailed? Let's do Original | Final-G3 | Final-G5
        # User asked for "G3 vs G5 final side by side".
        put_text_box(v_g3, "G3 (Strict)", (0,0,255))
        put_text_box(v_g5, "G5 (TopK)", (0,0,255))
        panel_sbs = np.hstack([v_orig, v_g3, v_g5])
        cv2.imwrite(os.path.join(out_dirs['SBS'], f"{bn}_sbs.jpg"), panel_sbs)

    print("Vis Done.")

if __name__ == "__main__":
    main()
