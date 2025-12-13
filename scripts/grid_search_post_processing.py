import argparse
import os
import os.path as osp
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import RUNNERS
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Grid Search Post-processing')
    parser.add_argument('config', help='Test config file path')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--work-dir', help='Work directory', default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        # Use a temp dir or the config's work_dir
        pass

    # Build runner once
    # We need to make sure the runner doesn't dump results to the same file every time if that causes issues
    # But for metrics it should be fine.
    runner = Runner.from_cfg(cfg)
    
    # Load weights
    runner.load_or_resume()
    
    # Grid
    min_text_scores = [0.3, 0.4, 0.5, 0.6, 0.7]
    unclip_ratios = [1.5, 1.8, 2.0, 2.2, 2.5, 3.0]
    
    results = []
    
    print(f"Starting Grid Search on {len(min_text_scores) * len(unclip_ratios)} combinations...")
    
    for score in min_text_scores:
        for unclip in unclip_ratios:
            print(f"\n--- Testing min_text_score={score}, unclip_ratio={unclip} ---")
            
            # Update model postprocessor parameters
            # Handle DataParallel/DistributedDataParallel if necessary
            model = runner.model
            if hasattr(model, 'module'):
                model = model.module
                
            # Access postprocessor
            # For DBNet: model.det_head.postprocessor
            # For FCENet: model.det_head.postprocessor (check if similar)
            if hasattr(model, 'det_head') and hasattr(model.det_head, 'postprocessor'):
                postprocessor = model.det_head.postprocessor
                # Update attributes
                if hasattr(postprocessor, 'min_text_score'):
                    postprocessor.min_text_score = score
                if hasattr(postprocessor, 'unclip_ratio'):
                    postprocessor.unclip_ratio = unclip
                # For FCENet, parameters might be different (fourier_degree, etc.)
                # But user asked for DBNet/FCENet thresholds.
                # FCENet uses 'score_thr' in Postprocessor?
                # Let's assume DBNet for now as per priority.
            else:
                print("Warning: Could not find postprocessor in model.det_head")
            
            # Run test
            metrics = runner.test()
            
            # Extract Hmean
            hmean = 0
            for k, v in metrics.items():
                if 'hmean' in k:
                    hmean = v
                    break
            
            print(f">>> Result: min_text_score={score}, unclip_ratio={unclip} -> Hmean={hmean}")
            results.append({'score': score, 'unclip': unclip, 'hmean': hmean, 'metrics': metrics})
            
    # Sort and print best
    results.sort(key=lambda x: x['hmean'], reverse=True)
    print("\n================ Top 5 Results ================")
    for i in range(min(5, len(results))):
        res = results[i]
        print(f"Rank {i+1}: Hmean={res['hmean']:.4f} (score={res['score']}, unclip={res['unclip']})")
        
    # Save results to file
    import json
    with open(osp.join(runner.work_dir, 'grid_search_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {osp.join(runner.work_dir, 'grid_search_results.json')}")

if __name__ == '__main__':
    main()
