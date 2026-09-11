import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.figsize": (10, 6),
})

MULTI_DIR = r'c:\Users\rahma\Documents\Playground\LLM-Project\results\multi_model'
FIGURES_DIR = os.path.join(MULTI_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load Qwen results
rag_df = pd.read_csv(os.path.join(MULTI_DIR, "qwen2.5-0.5b_rag_results.csv"))
sce_df = pd.read_csv(os.path.join(MULTI_DIR, "qwen2.5-0.5b_sce_results.csv"))

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rouge_l_scores = [scorer.score(str(r['best_answer']), str(r['generated_answer']))['rougeL'].fmeasure
                  for _, r in rag_df.iterrows()]
actual_halluc = [1 if r < 0.2 else 0 for r in rouge_l_scores]
sce_scores = sce_df['semantic_similarity_score'].values

thresholds = np.arange(0.05, 1.00, 0.05)
rows = []

for t in thresholds:
    t = round(t, 2)
    flags = [1 if s < t else 0 for s in sce_scores]
    n_flagged = sum(flags)
    rate = n_flagged / len(flags) * 100
    
    flagged_rl = [rl for rl, f in zip(rouge_l_scores, flags) if f == 1]
    consistent_rl = [rl for rl, f in zip(rouge_l_scores, flags) if f == 0]
    
    avg_flagged = np.mean(flagged_rl) if flagged_rl else 0.0
    avg_consistent = np.mean(consistent_rl) if consistent_rl else 0.0
    
    tp = sum(1 for a, f in zip(actual_halluc, flags) if a == 1 and f == 1)
    fp = sum(1 for a, f in zip(actual_halluc, flags) if a == 0 and f == 1)
    fn = sum(1 for a, f in zip(actual_halluc, flags) if a == 1 and f == 0)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    rows.append({
        "Threshold (τ)": t,
        "Flagged Count": n_flagged,
        "Hallucination Rate (%)": round(rate, 2),
        "Avg ROUGE-L (Flagged)": round(avg_flagged, 4),
        "Avg ROUGE-L (Consistent)": round(avg_consistent, 4),
        "Quality Gap (Δ ROUGE-L)": round(avg_consistent - avg_flagged, 4),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1 Score": round(f1, 3)
    })

sens_df = pd.DataFrame(rows)
sens_csv_path = os.path.join(MULTI_DIR, "qwen_threshold_sensitivity.csv")
sens_df.to_csv(sens_csv_path, index=False)
print(f"Saved sensitivity CSV to: {sens_csv_path}")

print("\n=== SELECTED THRESHOLDS FOR QWEN2.5-0.5B ===")
sel_df = sens_df[sens_df["Threshold (τ)"].isin([0.3, 0.4, 0.5, 0.6, 0.7])]
print(sel_df[["Threshold (τ)", "Hallucination Rate (%)", "Avg ROUGE-L (Flagged)", "Avg ROUGE-L (Consistent)", "Quality Gap (Δ ROUGE-L)", "F1 Score"]].to_string(index=False))

# --- PLOT 1: Dual-Axis Sensitivity Chart for Qwen ---
fig, ax1 = plt.subplots(figsize=(10, 6))

color_rate = "#e67e22"
color_consistent = "#2ecc71"
color_flagged = "#e74c3c"

ax1.set_xlabel("SCE Threshold (τ)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Hallucination Rate (%)", color=color_rate, fontsize=12, fontweight="bold")
line1 = ax1.plot(sens_df["Threshold (τ)"], sens_df["Hallucination Rate (%)"],
                 color=color_rate, marker="o", linewidth=2.5, label="Hallucination Rate (%)")
ax1.tick_params(axis="y", labelcolor=color_rate)
ax1.set_ylim(0, 105)

# Secondary axis for ROUGE-L
ax2 = ax1.twinx()
ax2.set_ylabel("Mean ROUGE-L F1", color="#2c3e50", fontsize=12, fontweight="bold")
line2 = ax2.plot(sens_df["Threshold (τ)"], sens_df["Avg ROUGE-L (Consistent)"],
                 color=color_consistent, marker="s", linestyle="--", linewidth=2, label="Avg ROUGE-L (Consistent)")
line3 = ax2.plot(sens_df["Threshold (τ)"], sens_df["Avg ROUGE-L (Flagged)"],
                 color=color_flagged, marker="^", linestyle=":", linewidth=2, label="Avg ROUGE-L (Flagged)")
ax2.tick_params(axis="y", labelcolor="#2c3e50")
ax2.set_ylim(0, 0.85)

# Add vertical reference line at tau=0.5 and optimal F1 threshold
ax1.axvline(x=0.5, color="#7f8c8d", linestyle="-.", alpha=0.8, label="Default Threshold (τ=0.5)")
best_f1_idx = sens_df["F1 Score"].idxmax()
best_tau = sens_df.loc[best_f1_idx, "Threshold (τ)"]
best_f1 = sens_df.loc[best_f1_idx, "F1 Score"]
ax1.axvline(x=best_tau, color="#9b59b6", linestyle=":", linewidth=2, label=f"Optimal F1 Threshold (τ={best_tau}, F1={best_f1:.3f})")

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines] + ["Default Threshold (τ=0.5)", f"Optimal F1 (τ={best_tau})"]
ax1.legend(lines + [plt.Line2D([0], [0], color="#7f8c8d", linestyle="-."),
                    plt.Line2D([0], [0], color="#9b59b6", linestyle=":")],
           labels, loc="center left", framealpha=0.9)

plt.title("Qwen2.5-0.5B: Threshold Sensitivity Analysis (Hallucination Rate & ROUGE-L Separation)", fontsize=13, fontweight="bold", pad=12)
fig.tight_layout()

chart_path = os.path.join(FIGURES_DIR, "fig_qwen_threshold_sensitivity.png")
fig.savefig(chart_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved Qwen sensitivity chart to: {chart_path}")

# Copy to artifact folder
artifact_dir = r"C:\Users\rahma\.gemini\antigravity\brain\c2e60530-0333-492e-8d37-b3f2b62d2ce3"
import shutil
shutil.copy2(chart_path, os.path.join(artifact_dir, "fig_qwen_threshold_sensitivity.png"))
print(f"Copied to artifact dir: {os.path.join(artifact_dir, 'fig_qwen_threshold_sensitivity.png')}")
