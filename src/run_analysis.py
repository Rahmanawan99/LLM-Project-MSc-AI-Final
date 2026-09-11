import json
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "analysis")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "Baseline LLM": "#e74c3c",
    "RAG": "#3498db",
    "RAG + SCE": "#2ecc71",
}
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.figsize": (10, 6),
})


def log(message):
    print(f"[ANALYSIS] {message}")


log("Loading result CSV files...")
baseline_df = pd.read_csv(os.path.join(RESULTS_DIR, "baseline_results.csv"))
rag_df = pd.read_csv(os.path.join(RESULTS_DIR, "rag_results.csv"))
sce_df = pd.read_csv(os.path.join(RESULTS_DIR, "sce_results.csv"))

log(f"Baseline rows: {len(baseline_df)}")
log(f"RAG rows: {len(rag_df)}")
log(f"SCE rows: {len(sce_df)}")

try:
    from datasets import load_dataset

    tqa = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    tqa_df = tqa.to_pandas()[["question", "category", "type"]]
    HAS_CATEGORIES = True
except Exception as exc:
    log(f"Could not load TruthfulQA categories: {exc}")
    HAS_CATEGORIES = False
    tqa_df = None

for df in [baseline_df, rag_df, sce_df]:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("")

log("Loading sentence-transformers and ROUGE models...")
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def compute_rouge(generated, reference):
    scores = scorer.score(str(reference), str(generated))
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


def compute_semantic_similarity(texts_a, texts_b, batch_size=64):
    log(f"Encoding {len(texts_a)} text pairs...")
    embeddings_a = embed_model.encode(texts_a, batch_size=batch_size, show_progress_bar=True)
    embeddings_b = embed_model.encode(texts_b, batch_size=batch_size, show_progress_bar=True)
    similarities = []
    for left, right in zip(embeddings_a, embeddings_b):
        similarities.append(float(util.cos_sim(left, right)[0][0]))
    return similarities


log("Computing baseline metrics...")
baseline_rouge = [compute_rouge(str(row["generated_answer"]), str(row["best_answer"])) for _, row in baseline_df.iterrows()]
baseline_df["rouge1_f"] = [row["rouge1_f"] for row in baseline_rouge]
baseline_df["rouge2_f"] = [row["rouge2_f"] for row in baseline_rouge]
baseline_df["rougeL_f"] = [row["rougeL_f"] for row in baseline_rouge]
baseline_df["sem_sim_to_answer"] = compute_semantic_similarity(
    baseline_df["generated_answer"].astype(str).tolist(),
    baseline_df["best_answer"].astype(str).tolist(),
)

log("Computing RAG metrics...")
rag_rouge = [compute_rouge(str(row["generated_answer"]), str(row["best_answer"])) for _, row in rag_df.iterrows()]
rag_df["rouge1_f"] = [row["rouge1_f"] for row in rag_rouge]
rag_df["rouge2_f"] = [row["rouge2_f"] for row in rag_rouge]
rag_df["rougeL_f"] = [row["rougeL_f"] for row in rag_rouge]
rag_df["sem_sim_to_answer"] = compute_semantic_similarity(
    rag_df["generated_answer"].astype(str).tolist(),
    rag_df["best_answer"].astype(str).tolist(),
)
rag_df["sce_score"] = compute_semantic_similarity(
    rag_df["generated_answer"].astype(str).tolist(),
    rag_df["retrieved_context"].astype(str).tolist(),
)

log("Computing SCE metrics...")
sce_df["best_answer"] = rag_df["best_answer"].values
sce_rouge = [compute_rouge(str(row["generated_answer"]), str(row.get("best_answer", ""))) for _, row in sce_df.iterrows()]
sce_df["rouge1_f"] = [row["rouge1_f"] for row in sce_rouge]
sce_df["rouge2_f"] = [row["rouge2_f"] for row in sce_rouge]
sce_df["rougeL_f"] = [row["rougeL_f"] for row in sce_rouge]
sce_df["sem_sim_to_answer"] = compute_semantic_similarity(
    sce_df["generated_answer"].astype(str).tolist(),
    sce_df["best_answer"].astype(str).tolist(),
)

for df in [baseline_df, rag_df, sce_df]:
    df["answer_length"] = df["generated_answer"].astype(str).str.len()


def make_summary(label, df, has_sce=False):
    summary = {
        "Condition": label,
        "N": len(df),
        "Avg Answer Length": round(df["answer_length"].mean(), 2),
        "ROUGE-1 F1": round(df["rouge1_f"].mean(), 4),
        "ROUGE-2 F1": round(df["rouge2_f"].mean(), 4),
        "ROUGE-L F1": round(df["rougeL_f"].mean(), 4),
        "Sem. Sim. to Ground Truth": round(df["sem_sim_to_answer"].mean(), 4),
    }
    if has_sce:
        sce_col = "sce_score" if "sce_score" in df.columns else "semantic_similarity_score"
        summary["Avg SCE Score"] = round(df[sce_col].mean(), 4)
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            rate = (df[sce_col] < threshold).mean() * 100
            summary[f"Halluc. Rate (t={threshold})"] = round(rate, 2)
    return summary


summaries = [
    make_summary("Baseline LLM", baseline_df),
    make_summary("RAG", rag_df, has_sce=True),
    make_summary("RAG + SCE", sce_df, has_sce=True),
]
summary_df = pd.DataFrame(summaries)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "full_comparison_table.csv"), index=False)
print(summary_df.to_string(index=False))

log("Running statistical tests...")
stat_results = []

for metric, column in [("ROUGE-L", "rougeL_f"), ("Sem. Sim.", "sem_sim_to_answer")]:
    baseline_vals = baseline_df[column].values
    rag_vals = rag_df[column].values
    stat_w, p_w = stats.wilcoxon(baseline_vals, rag_vals, alternative="two-sided")
    diff = rag_vals - baseline_vals
    cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0
    improvement = ((rag_vals.mean() - baseline_vals.mean()) / baseline_vals.mean() * 100) if baseline_vals.mean() != 0 else float("inf")
    stat_results.append(
        {
            "Comparison": f"Baseline vs RAG ({metric})",
            "Baseline Mean": round(baseline_vals.mean(), 4),
            "RAG Mean": round(rag_vals.mean(), 4),
            "Improvement (%)": round(improvement, 2),
            "Wilcoxon Statistic": round(stat_w, 2),
            "p-value": f"{p_w:.2e}",
            "Significant (a=0.05)": "Yes" if p_w < 0.05 else "No",
            "Cohen's d": round(cohens_d, 4),
            "Effect Size": "Large" if abs(cohens_d) >= 0.8 else "Medium" if abs(cohens_d) >= 0.5 else "Small",
        }
    )

flagged_mask = sce_df["hallucination_flag"] == 1
not_flagged_mask = sce_df["hallucination_flag"] == 0
for metric, column in [("ROUGE-L", "rougeL_f"), ("Sem. Sim.", "sem_sim_to_answer")]:
    flagged_vals = sce_df.loc[flagged_mask, column].values
    consistent_vals = sce_df.loc[not_flagged_mask, column].values
    stat_mw, p_mw = stats.mannwhitneyu(flagged_vals, consistent_vals, alternative="two-sided")
    pooled_std = np.sqrt((flagged_vals.std() ** 2 + consistent_vals.std() ** 2) / 2)
    d = (consistent_vals.mean() - flagged_vals.mean()) / pooled_std if pooled_std > 0 else 0
    stat_results.append(
        {
            "Comparison": f"SCE Flagged vs Consistent ({metric})",
            "Baseline Mean": round(flagged_vals.mean(), 4),
            "RAG Mean": round(consistent_vals.mean(), 4),
            "Improvement (%)": "N/A",
            "Wilcoxon Statistic": round(stat_mw, 2),
            "p-value": f"{p_mw:.2e}",
            "Significant (a=0.05)": "Yes" if p_mw < 0.05 else "No",
            "Cohen's d": round(d, 4),
            "Effect Size": "Large" if abs(d) >= 0.8 else "Medium" if abs(d) >= 0.5 else "Small",
        }
    )

stat_df = pd.DataFrame(stat_results)
stat_df.to_csv(os.path.join(OUTPUT_DIR, "statistical_tests.csv"), index=False)


def bootstrap_ci(data, n_boot=5000, ci=0.95):
    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)
    return round(lower, 4), round(upper, 4)


ci_results = []
for label, df in [("Baseline LLM", baseline_df), ("RAG", rag_df), ("RAG + SCE", sce_df)]:
    for metric, column in [("ROUGE-L", "rougeL_f"), ("Sem. Sim.", "sem_sim_to_answer")]:
        lower, upper = bootstrap_ci(df[column].values)
        ci_results.append(
            {
                "Condition": label,
                "Metric": metric,
                "Mean": round(df[column].mean(), 4),
                "95% CI Lower": lower,
                "95% CI Upper": upper,
            }
        )

ci_df = pd.DataFrame(ci_results)
ci_df.to_csv(os.path.join(OUTPUT_DIR, "confidence_intervals.csv"), index=False)

log("Running threshold sensitivity analysis...")
thresholds = np.arange(0.05, 1.0, 0.05)
sce_col = "semantic_similarity_score"

sensitivity_data = []
for threshold in thresholds:
    flagged = (sce_df[sce_col] < threshold).sum()
    total = len(sce_df)
    rate = flagged / total * 100
    flagged_mask = sce_df[sce_col] < threshold
    if flagged_mask.sum() > 0:
        avg_rouge_flagged = sce_df.loc[flagged_mask, "rougeL_f"].mean()
        avg_rouge_consistent = sce_df.loc[~flagged_mask, "rougeL_f"].mean() if (~flagged_mask).sum() > 0 else 0
    else:
        avg_rouge_flagged = 0
        avg_rouge_consistent = sce_df["rougeL_f"].mean()

    sensitivity_data.append(
        {
            "Threshold": round(threshold, 2),
            "Flagged Count": flagged,
            "Hallucination Rate (%)": round(rate, 2),
            "Avg ROUGE-L (Flagged)": round(avg_rouge_flagged, 4),
            "Avg ROUGE-L (Consistent)": round(avg_rouge_consistent, 4),
        }
    )

sensitivity_df = pd.DataFrame(sensitivity_data)
sensitivity_df.to_csv(os.path.join(OUTPUT_DIR, "threshold_sensitivity.csv"), index=False)

if HAS_CATEGORIES:
    log("Computing category breakdown...")
    baseline_df["category"] = tqa_df["category"].values
    rag_df["category"] = tqa_df["category"].values
    sce_df["category"] = tqa_df["category"].values

    cat_data = []
    for cat in sorted(tqa_df["category"].unique()):
        b_mask = baseline_df["category"] == cat
        r_mask = rag_df["category"] == cat
        s_mask = sce_df["category"] == cat

        cat_data.append(
            {
                "Category": cat,
                "Count": b_mask.sum(),
                "Baseline ROUGE-L": round(baseline_df.loc[b_mask, "rougeL_f"].mean(), 4),
                "RAG ROUGE-L": round(rag_df.loc[r_mask, "rougeL_f"].mean(), 4),
                "ROUGE-L Improvement": round(rag_df.loc[r_mask, "rougeL_f"].mean() - baseline_df.loc[b_mask, "rougeL_f"].mean(), 4),
                "Baseline Sem. Sim.": round(baseline_df.loc[b_mask, "sem_sim_to_answer"].mean(), 4),
                "RAG Sem. Sim.": round(rag_df.loc[r_mask, "sem_sim_to_answer"].mean(), 4),
                "Avg SCE Score": round(sce_df.loc[s_mask, "semantic_similarity_score"].mean(), 4),
                "Halluc. Rate (%)": round(sce_df.loc[s_mask, "hallucination_flag"].mean() * 100, 2),
            }
        )

    cat_df = pd.DataFrame(cat_data).sort_values("Halluc. Rate (%)", ascending=False)
    cat_df.to_csv(os.path.join(OUTPUT_DIR, "category_breakdown.csv"), index=False)

log("Identifying the worst failures...")
sce_failures = sce_df[sce_df["hallucination_flag"] == 1].copy()
sce_failures = sce_failures.sort_values("rougeL_f", ascending=True).head(20)
sce_failures_out = sce_failures[["question", "retrieved_context", "generated_answer", "best_answer", "semantic_similarity_score", "rougeL_f"]].copy()
sce_failures_out.columns = ["Question", "Retrieved Context", "Generated Answer", "Best Answer", "SCE Score", "ROUGE-L"]
sce_failures_out.to_csv(os.path.join(OUTPUT_DIR, "worst_failures.csv"), index=False)

low_rouge_mask = sce_df["rougeL_f"] < 0.2
tp = int((low_rouge_mask & (sce_df["hallucination_flag"] == 1)).sum())
fp = int((~low_rouge_mask & (sce_df["hallucination_flag"] == 1)).sum())
fn = int((low_rouge_mask & (sce_df["hallucination_flag"] == 0)).sum())
tn = int((~low_rouge_mask & (sce_df["hallucination_flag"] == 0)).sum())

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

sce_classifier_stats = {
    "True Positives": tp,
    "False Positives": fp,
    "False Negatives": fn,
    "True Negatives": tn,
    "Precision": round(precision, 4),
    "Recall": round(recall, 4),
    "F1 Score": round(f1, 4),
}

with open(os.path.join(OUTPUT_DIR, "sce_classifier_stats.json"), "w") as file:
    json.dump(sce_classifier_stats, file, indent=2)

log("Generating figures...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics_to_plot = [
    ("ROUGE-L F1", [baseline_df["rougeL_f"].mean(), rag_df["rougeL_f"].mean(), sce_df["rougeL_f"].mean()]),
    ("Semantic Sim. to Ground Truth", [baseline_df["sem_sim_to_answer"].mean(), rag_df["sem_sim_to_answer"].mean(), sce_df["sem_sim_to_answer"].mean()]),
    ("ROUGE-1 F1", [baseline_df["rouge1_f"].mean(), rag_df["rouge1_f"].mean(), sce_df["rouge1_f"].mean()]),
]
conditions = ["Baseline LLM", "RAG", "RAG + SCE"]
for ax, (metric_name, values) in zip(axes, metrics_to_plot):
    bars = ax.bar(conditions, values, color=[COLORS[name] for name in conditions], edgecolor="white", linewidth=1.5)
    ax.set_title(metric_name, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
plt.suptitle("Cross-Condition Metric Comparison (TruthfulQA, N=817)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig1_metrics_comparison.png"), bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
rouge_data = [baseline_df["rougeL_f"].values, rag_df["rougeL_f"].values, sce_df["rougeL_f"].values]
box1 = axes[0].boxplot(rouge_data, labels=conditions, patch_artist=True, showfliers=False, medianprops={"color": "black", "linewidth": 2})
for patch, color in zip(box1["boxes"], [COLORS[name] for name in conditions]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_title("ROUGE-L F1 Distribution", fontweight="bold")
axes[0].set_ylabel("ROUGE-L F1 Score")

semantic_data = [baseline_df["sem_sim_to_answer"].values, rag_df["sem_sim_to_answer"].values, sce_df["sem_sim_to_answer"].values]
box2 = axes[1].boxplot(semantic_data, labels=conditions, patch_artist=True, showfliers=False, medianprops={"color": "black", "linewidth": 2})
for patch, color in zip(box2["boxes"], [COLORS[name] for name in conditions]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_title("Semantic Similarity to Ground Truth", fontweight="bold")
axes[1].set_ylabel("Cosine Similarity")
plt.suptitle("Score Distributions Across Conditions", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig2_box_distributions.png"), bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(sce_df["semantic_similarity_score"], bins=30, color=COLORS["RAG + SCE"], edgecolor="white", alpha=0.8, label="SCE Score Distribution")
ax.axvline(x=0.5, color="red", linestyle="--", linewidth=2, label="Threshold (t = 0.5)")
ax.axvline(x=sce_df["semantic_similarity_score"].mean(), color="orange", linestyle="-.", linewidth=2, label=f"Mean = {sce_df['semantic_similarity_score'].mean():.3f}")
ax.set_xlabel("Semantic Consistency Score (Cosine Similarity)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of SCE Scores (RAG + SCE, N=817)", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig3_sce_distribution.png"), bbox_inches="tight")
plt.close()

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(sensitivity_df["Threshold"], sensitivity_df["Hallucination Rate (%)"], "o-", color=COLORS["Baseline LLM"], linewidth=2, markersize=6, label="Hallucination Rate (%)")
ax1.set_xlabel("SCE Threshold (t)", fontsize=12)
ax1.set_ylabel("Hallucination Rate (%)", color=COLORS["Baseline LLM"], fontsize=12)
ax1.tick_params(axis="y", labelcolor=COLORS["Baseline LLM"])
ax2 = ax1.twinx()
ax2.plot(sensitivity_df["Threshold"], sensitivity_df["Avg ROUGE-L (Flagged)"], "s--", color=COLORS["RAG"], linewidth=2, markersize=5, label="Avg ROUGE-L (Flagged)")
ax2.plot(sensitivity_df["Threshold"], sensitivity_df["Avg ROUGE-L (Consistent)"], "^--", color=COLORS["RAG + SCE"], linewidth=2, markersize=5, label="Avg ROUGE-L (Consistent)")
ax2.set_ylabel("Average ROUGE-L F1", fontsize=12)
ax1.axvline(x=0.5, color="gray", linestyle=":", alpha=0.6)
ax1.annotate("t = 0.5", xy=(0.5, 50), fontsize=10, color="gray", ha="center")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
ax1.set_title("SCE Threshold Sensitivity Analysis", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig4_threshold_sensitivity.png"), bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
cm = np.array([[tp, fp], [fn, tn]])
im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted\nHallucination", "Predicted\nConsistent"])
ax.set_yticklabels(["Actual\nHallucination\n(ROUGE-L<0.2)", "Actual\nConsistent\n(ROUGE-L>=0.2)"])
for i in range(2):
    for j in range(2):
        text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=text_color, fontsize=18, fontweight="bold")
ax.set_title(f"SCE Hallucination Detection\nPrecision={precision:.2f}  Recall={recall:.2f}  F1={f1:.2f}", fontweight="bold")
plt.colorbar(im)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig5_confusion_matrix.png"), bbox_inches="tight")
plt.close()

if HAS_CATEGORIES:
    cat_plot = cat_df.set_index("Category")[["Baseline ROUGE-L", "RAG ROUGE-L", "Baseline Sem. Sim.", "RAG Sem. Sim.", "Halluc. Rate (%)"]].copy()
    cat_plot_norm = cat_plot.copy()
    cat_plot_norm["Halluc. Rate (%)"] = cat_plot_norm["Halluc. Rate (%)"] / 100

    fig, ax = plt.subplots(figsize=(12, max(8, len(cat_plot) * 0.5)))
    im = ax.imshow(cat_plot_norm.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(cat_plot.columns)))
    ax.set_xticklabels(cat_plot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cat_plot)))
    ax.set_yticklabels(cat_plot.index)
    for i in range(len(cat_plot)):
        for j in range(len(cat_plot.columns)):
            val = cat_plot.iloc[i, j]
            fmt = f"{val:.1f}%" if "Halluc" in cat_plot.columns[j] else f"{val:.3f}"
            ax.text(j, i, fmt, ha="center", va="center", fontsize=7, fontweight="bold")
    ax.set_title("Performance by TruthfulQA Category", fontweight="bold", fontsize=13)
    plt.colorbar(im, label="Score (higher = better, except Halluc. Rate)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig6_category_heatmap.png"), bbox_inches="tight")
    plt.close()

if HAS_CATEGORIES:
    fig, ax = plt.subplots(figsize=(12, 6))
    cat_sorted = cat_df.sort_values("ROUGE-L Improvement", ascending=True)
    bars = ax.barh(cat_sorted["Category"], cat_sorted["ROUGE-L Improvement"], color=[COLORS["RAG + SCE"] if value > 0 else COLORS["Baseline LLM"] for value in cat_sorted["ROUGE-L Improvement"]], edgecolor="white")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("ROUGE-L Improvement (RAG - Baseline)")
    ax.set_title("Per-Category ROUGE-L Improvement: Baseline -> RAG", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig7_category_improvement.png"), bbox_inches="tight")
    plt.close()

log("Saving enriched data files...")
baseline_df.to_csv(os.path.join(OUTPUT_DIR, "baseline_results_enriched.csv"), index=False)
rag_df.to_csv(os.path.join(OUTPUT_DIR, "rag_results_enriched.csv"), index=False)
sce_df.to_csv(os.path.join(OUTPUT_DIR, "sce_results_enriched.csv"), index=False)

print("\nAnalysis complete.")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Figures directory: {FIGURES_DIR}")
