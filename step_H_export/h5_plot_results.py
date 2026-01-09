
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration ---
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = str(REPO_ROOT / "outputs" / "cascade_demo" / "run004" / "fullval_G3_vs_G5")
PLOT_DIR = os.path.join(OUT_ROOT, "plots")

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Overall
    df_overall = pd.read_csv(os.path.join(OUT_ROOT, "compare_overall.csv"))
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_overall, x="Method", y="Hmean", palette="viridis")
    plt.title("Overall Hmean Comparison")
    plt.ylim(0, 1.0)
    for i, row in df_overall.iterrows():
        plt.text(i, row.Hmean + 0.01, f"{row.Hmean:.4f}", ha='center')
    plt.savefig(os.path.join(PLOT_DIR, "hmean_overall_bar.png"))
    plt.close()
    
    # By Dataset
    df_ds = pd.read_csv(os.path.join(OUT_ROOT, "compare_by_dataset.csv"))
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_ds, x="Dataset", y="Hmean", hue="Method", palette="viridis")
    plt.title("Hmean by Dataset")
    plt.ylim(0, 1.0)
    plt.savefig(os.path.join(PLOT_DIR, "hmean_by_dataset_bar.png"))
    plt.close()
    
    print("Plots Generated.")

if __name__ == "__main__":
    main()
