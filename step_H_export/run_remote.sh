
#!/bin/bash
# 确保在 mmocr 工程根目录下运行
# Usage: bash step_H_export/run_remote.sh

export PYTHONPATH=.

echo ">>> Step 1: Running Inference (G3 & G5)..."
python step_H_export/h1_run_inference.py

echo ">>> Step 2: Computing Metrics..."
python step_H_export/h2_compute_metrics.py

echo ">>> Step 3: Generating Visualizations..."
python step_H_export/h4_gen_panels.py

echo ">>> Step 4: Plotting Results..."
python step_H_export/h5_plot_results.py

echo ">>> All Done! Results are in output directory defined in scripts."
