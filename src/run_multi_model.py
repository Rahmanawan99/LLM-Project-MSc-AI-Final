import argparse
import io
import os
import shutil
import sys
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MULTI_DIR = os.path.join(RESULTS_DIR, "multi_model")
FIGURES_DIR = os.path.join(MULTI_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def log(message):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    log(f"GPU detected: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB)")
else:
    DEVICE = torch.device("cpu")
    log("No GPU detected, using CPU")


def load_flan_t5_base():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = "google/flan-t5-base"
    log(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    model.eval()
    log(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    return tokenizer, model, "flan-t5-base"


def generate_flan_t5(tokenizer, model, prompt):
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def load_qwen():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    log(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    log(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    return tokenizer, model, "qwen2.5-0.5b"


def generate_qwen_baseline(tokenizer, model, question):
    import torch

    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_qwen_rag(tokenizer, model, question, context):
    import torch

    messages = [
        {"role": "system", "content": "Answer the question using only the provided context. Be concise."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_truthfulqa():
    from datasets import load_dataset

    log("Loading TruthfulQA dataset...")
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    df = dataset.to_pandas()
    log(f"Loaded {len(df)} questions across {df['category'].nunique()} categories")
    return df


def build_faiss_index(best_answers):
    import faiss
    from sentence_transformers import SentenceTransformer

    log("Building FAISS index...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(best_answers, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    log(f"FAISS index built with {index.ntotal} vectors (dim={embeddings.shape[1]})")
    return index, embed_model, embeddings


def retrieve_context(question, embed_model, faiss_index, best_answers, k=2):
    query_vec = embed_model.encode([question]).astype("float32")
    _, indices = faiss_index.search(query_vec, k)
    top_idx = indices[0][0]
    return best_answers[top_idx]


def run_experiment(model_key, tqa_df, faiss_index, embed_model, best_answers_list):
    from sentence_transformers import SentenceTransformer, util as st_util

    if model_key == "flan-base":
        tokenizer, model, model_label = load_flan_t5_base()
    elif model_key == "qwen":
        tokenizer, model, model_label = load_qwen()
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    questions = tqa_df["question"].tolist()
    best_answers = tqa_df["best_answer"].tolist()
    total = len(questions)

    baseline_path = os.path.join(MULTI_DIR, f"{model_label}_baseline_results.csv")
    rag_path = os.path.join(MULTI_DIR, f"{model_label}_rag_results.csv")
    sce_path = os.path.join(MULTI_DIR, f"{model_label}_sce_results.csv")

    if all(os.path.exists(path) for path in [baseline_path, rag_path, sce_path]):
        existing = pd.read_csv(baseline_path)
        if len(existing) >= total:
            log(f"{model_label} results already exist ({len(existing)} rows). Skipping.")
            return model_label

    log(f"Running experiment: {model_label} ({total} questions)")

    log(f"{model_label}: baseline generation")
    baseline_rows = []
    for i, question in enumerate(questions):
        if model_key == "flan-base":
            answer = generate_flan_t5(tokenizer, model, question)
        else:
            answer = generate_qwen_baseline(tokenizer, model, question)

        baseline_rows.append({
            "question": question,
            "generated_answer": answer,
            "best_answer": best_answers[i],
        })

        if (i + 1) % 50 == 0:
            log(f"Baseline: {i + 1}/{total} done")

    pd.DataFrame(baseline_rows).to_csv(baseline_path, index=False)
    log(f"Saved baseline results: {baseline_path}")

    log(f"{model_label}: RAG generation")
    rag_rows = []
    for i, question in enumerate(questions):
        context = retrieve_context(question, embed_model, faiss_index, best_answers_list)

        if model_key == "flan-base":
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            answer = generate_flan_t5(tokenizer, model, prompt)
        else:
            answer = generate_qwen_rag(tokenizer, model, question, context)

        rag_rows.append({
            "question": question,
            "retrieved_context": context,
            "generated_answer": answer,
            "best_answer": best_answers[i],
        })

        if (i + 1) % 50 == 0:
            log(f"RAG: {i + 1}/{total} done")

    pd.DataFrame(rag_rows).to_csv(rag_path, index=False)
    log(f"Saved RAG results: {rag_path}")

    log(f"{model_label}: SCE scoring")
    sce_model = SentenceTransformer("all-MiniLM-L6-v2")
    sce_rows = []
    threshold = 0.5

    for i, row in enumerate(rag_rows):
        answer_emb = sce_model.encode([row["generated_answer"]])
        context_emb = sce_model.encode([row["retrieved_context"]])
        similarity = float(st_util.cos_sim(answer_emb, context_emb)[0][0])
        flag = 1 if similarity < threshold else 0

        sce_rows.append({
            "question": row["question"],
            "retrieved_context": row["retrieved_context"],
            "generated_answer": row["generated_answer"],
            "semantic_similarity_score": similarity,
            "hallucination_flag": flag,
        })

        if (i + 1) % 100 == 0:
            log(f"SCE: {i + 1}/{total} done")

    pd.DataFrame(sce_rows).to_csv(sce_path, index=False)
    log(f"Saved SCE results: {sce_path}")

    del model, tokenizer
    import gc

    gc.collect()
    log(f"{model_label} complete")
    return model_label


def compute_metrics_for_model(model_label):
    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer, util as st_util

    log(f"Computing metrics for {model_label}...")

    baseline_df = pd.read_csv(os.path.join(MULTI_DIR, f"{model_label}_baseline_results.csv"))
    rag_df = pd.read_csv(os.path.join(MULTI_DIR, f"{model_label}_rag_results.csv"))
    sce_df = pd.read_csv(os.path.join(MULTI_DIR, f"{model_label}_sce_results.csv"))

    for df in [baseline_df, rag_df, sce_df]:
        for column in df.columns:
            if df[column].dtype == object:
                df[column] = df[column].fillna("")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    results = {}

    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    for _, row in baseline_df.iterrows():
        scores = scorer.score(str(row["best_answer"]), str(row["generated_answer"]))
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    gen_embs = embed_model.encode(baseline_df["generated_answer"].astype(str).tolist(), batch_size=64)
    ref_embs = embed_model.encode(baseline_df["best_answer"].astype(str).tolist(), batch_size=64)
    baseline_sem_sim = [
        float(st_util.cos_sim(g.reshape(1, -1), r.reshape(1, -1))[0][0])
        for g, r in zip(gen_embs, ref_embs)
    ]

    results["baseline"] = {
        "ROUGE-1": np.mean(rouge1_scores),
        "ROUGE-2": np.mean(rouge2_scores),
        "ROUGE-L": np.mean(rougeL_scores),
        "Semantic Similarity": np.mean(baseline_sem_sim),
        "rougeL_per_q": rougeL_scores,
        "semsim_per_q": baseline_sem_sim,
    }

    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    for _, row in rag_df.iterrows():
        scores = scorer.score(str(row["best_answer"]), str(row["generated_answer"]))
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    gen_embs = embed_model.encode(rag_df["generated_answer"].astype(str).tolist(), batch_size=64)
    ref_embs = embed_model.encode(rag_df["best_answer"].astype(str).tolist(), batch_size=64)
    rag_sem_sim = [
        float(st_util.cos_sim(g.reshape(1, -1), r.reshape(1, -1))[0][0])
        for g, r in zip(gen_embs, ref_embs)
    ]

    results["rag"] = {
        "ROUGE-1": np.mean(rouge1_scores),
        "ROUGE-2": np.mean(rouge2_scores),
        "ROUGE-L": np.mean(rougeL_scores),
        "Semantic Similarity": np.mean(rag_sem_sim),
        "rougeL_per_q": rougeL_scores,
        "semsim_per_q": rag_sem_sim,
    }

    sce_scores = sce_df["semantic_similarity_score"].values
    flags = sce_df["hallucination_flag"].values
    hallucination_rate = float(flags.sum()) / len(flags)

    actual_halluc = [1 if value < 0.2 else 0 for value in rougeL_scores]
    tp = sum(1 for actual, flagged in zip(actual_halluc, flags) if actual == 1 and flagged == 1)
    fp = sum(1 for actual, flagged in zip(actual_halluc, flags) if actual == 0 and flagged == 1)
    fn = sum(1 for actual, flagged in zip(actual_halluc, flags) if actual == 1 and flagged == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results["sce"] = {
        "Hallucination Rate": hallucination_rate,
        "SCE Mean Score": float(np.mean(sce_scores)),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }

    stat_rl, p_rl = wilcoxon(results["baseline"]["rougeL_per_q"], results["rag"]["rougeL_per_q"])
    stat_ss, p_ss = wilcoxon(results["baseline"]["semsim_per_q"], results["rag"]["semsim_per_q"])

    diff_rl = np.array(results["rag"]["rougeL_per_q"]) - np.array(results["baseline"]["rougeL_per_q"])
    cohens_d_rl = np.mean(diff_rl) / np.std(diff_rl) if np.std(diff_rl) > 0 else 0

    diff_ss = np.array(results["rag"]["semsim_per_q"]) - np.array(results["baseline"]["semsim_per_q"])
    cohens_d_ss = np.mean(diff_ss) / np.std(diff_ss) if np.std(diff_ss) > 0 else 0

    results["stats"] = {
        "ROUGE-L_p": p_rl,
        "ROUGE-L_cohens_d": cohens_d_rl,
        "SemSim_p": p_ss,
        "SemSim_cohens_d": cohens_d_ss,
    }

    baseline_rouge = results["baseline"]["ROUGE-L"]
    rag_rouge = results["rag"]["ROUGE-L"]
    results["improvement_pct"] = ((rag_rouge - baseline_rouge) / baseline_rouge * 100) if baseline_rouge > 0 else float("inf")

    log(f"{model_label}: baseline ROUGE-L={baseline_rouge:.4f} -> RAG ROUGE-L={rag_rouge:.4f} (+{results['improvement_pct']:.1f}%)")
    log(f"{model_label}: SCE F1={f1:.3f}, hallucination rate={hallucination_rate:.1%}")
    return results


def run_combined_analysis():
    matplotlib.use("Agg")
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.figsize": (12, 7),
    })

    models = {}

    original_baseline = os.path.join(RESULTS_DIR, "baseline_results.csv")
    original_rag = os.path.join(RESULTS_DIR, "rag_results.csv")
    original_sce = os.path.join(RESULTS_DIR, "sce_results.csv")
    if all(os.path.exists(path) for path in [original_baseline, original_rag, original_sce]):
        models["flan-t5-small"] = "Flan-T5-Small (77M)"

    for label in ["flan-t5-base", "qwen2.5-0.5b"]:
        baseline_path = os.path.join(MULTI_DIR, f"{label}_baseline_results.csv")
        if os.path.exists(baseline_path):
            name_map = {
                "flan-t5-base": "Flan-T5-Base (250M)",
                "qwen2.5-0.5b": "Qwen2.5-0.5B (500M)",
            }
            models[label] = name_map[label]

    if len(models) < 2:
        log("Need at least 2 models to compare. Run the experiments first.")
        return

    log(f"Found {len(models)} models: {list(models.values())}")

    all_metrics = {}

    if "flan-t5-small" in models:
        for suffix in ["baseline_results.csv", "rag_results.csv", "sce_results.csv"]:
            source = os.path.join(RESULTS_DIR, suffix)
            destination = os.path.join(MULTI_DIR, f"flan-t5-small_{suffix}")
            if not os.path.exists(destination):
                shutil.copy2(source, destination)

    for label in models:
        all_metrics[label] = compute_metrics_for_model(label)

    log("Building comparison table...")
    table_rows = []
    for label, display_name in models.items():
        metrics = all_metrics[label]
        table_rows.append({
            "Model": display_name,
            "Parameters": label.split("(")[0].strip(),
            "Architecture": "Encoder-Decoder" if "flan" in label else "Decoder-Only",
            "Baseline ROUGE-L": round(metrics["baseline"]["ROUGE-L"], 4),
            "RAG ROUGE-L": round(metrics["rag"]["ROUGE-L"], 4),
            "Improvement (%)": round(metrics["improvement_pct"], 1),
            "Baseline Sem.Sim": round(metrics["baseline"]["Semantic Similarity"], 4),
            "RAG Sem.Sim": round(metrics["rag"]["Semantic Similarity"], 4),
            "Hallucination Rate": f"{metrics['sce']['Hallucination Rate']:.1%}",
            "SCE F1": round(metrics["sce"]["F1"], 3),
            "p-value (ROUGE-L)": f"{metrics['stats']['ROUGE-L_p']:.2e}",
            "Cohen's d": round(metrics["stats"]["ROUGE-L_cohens_d"], 3),
        })

    comparison_df = pd.DataFrame(table_rows)
    comparison_path = os.path.join(MULTI_DIR, "cross_model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    log(f"Saved comparison table: {comparison_path}")
    print(comparison_df.to_string(index=False))

    model_names = [models[label] for label in models]
    baseline_rouge = [all_metrics[label]["baseline"]["ROUGE-L"] for label in models]
    rag_rouge = [all_metrics[label]["rag"]["ROUGE-L"] for label in models]
    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, baseline_rouge, width, label="Baseline", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + width / 2, rag_rouge, width, label="RAG", color="#3498db", alpha=0.85)
    ax.set_ylabel("ROUGE-L F1 Score")
    ax.set_title("Baseline vs RAG: ROUGE-L Across Models")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, max(max(baseline_rouge), max(rag_rouge)) * 1.25)

    for bar in bars1:
        ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)

    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig_multi_rougeL_comparison.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved chart: {fig_path}")

    baseline_ss = [all_metrics[label]["baseline"]["Semantic Similarity"] for label in models]
    rag_ss = [all_metrics[label]["rag"]["Semantic Similarity"] for label in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, baseline_ss, width, label="Baseline", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + width / 2, rag_ss, width, label="RAG", color="#3498db", alpha=0.85)
    ax.set_ylabel("Semantic Similarity (Cosine)")
    ax.set_title("Baseline vs RAG: Semantic Similarity Across Models")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, max(max(baseline_ss), max(rag_ss)) * 1.25)

    for bar in bars1:
        ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)

    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig_multi_semsim_comparison.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved chart: {fig_path}")

    improvements = [all_metrics[label]["improvement_pct"] for label in models]
    colors = ["#2ecc71" if value > 0 else "#e74c3c" for value in improvements]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, improvements, color=colors, alpha=0.85)
    ax.set_ylabel("ROUGE-L Improvement (%)")
    ax.set_title("RAG Improvement Over Baseline Across Models")
    ax.axhline(y=0, color="black", linewidth=0.5)

    for bar, value in zip(bars, improvements):
        ax.annotate(f"+{value:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig_multi_improvement.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved chart: {fig_path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    hallucination_rates = [all_metrics[label]["sce"]["Hallucination Rate"] for label in models]
    bars = axes[0].bar(model_names, [rate * 100 for rate in hallucination_rates], color="#e67e22", alpha=0.85)
    axes[0].set_ylabel("Hallucination Rate (%)")
    axes[0].set_title("Hallucination Rate at tau=0.5")
    axes[0].set_xticklabels(model_names, rotation=15, ha="right")
    for bar, rate in zip(bars, hallucination_rates):
        axes[0].annotate(f"{rate:.1%}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)

    f1_scores = [all_metrics[label]["sce"]["F1"] for label in models]
    bars = axes[1].bar(model_names, f1_scores, color="#9b59b6", alpha=0.85)
    axes[1].set_ylabel("SCE Classifier F1 Score")
    axes[1].set_title("SCE Detection Performance (F1)")
    axes[1].set_xticklabels(model_names, rotation=15, ha="right")
    axes[1].set_ylim(0, 1)
    for bar, score in zip(bars, f1_scores):
        axes[1].annotate(f"{score:.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)

    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig_multi_sce_comparison.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved chart: {fig_path}")

    log("Summary")
    for label, display_name in models.items():
        metrics = all_metrics[label]
        log(f"{display_name}:")
        log(f"  Baseline ROUGE-L: {metrics['baseline']['ROUGE-L']:.4f}")
        log(f"  RAG ROUGE-L: {metrics['rag']['ROUGE-L']:.4f} (+{metrics['improvement_pct']:.1f}%)")
        log(f"  RAG Sem.Sim: {metrics['rag']['Semantic Similarity']:.4f}")
        log(f"  Hallucination rate: {metrics['sce']['Hallucination Rate']:.1%}")
        log(f"  SCE F1: {metrics['sce']['F1']:.3f}")
        log(f"  Wilcoxon p: {metrics['stats']['ROUGE-L_p']:.2e}")
        log(f"  Cohen's d: {metrics['stats']['ROUGE-L_cohens_d']:.3f}")

    log(f"All outputs saved to: {MULTI_DIR}")
    log("Done")


def main():
    parser = argparse.ArgumentParser(description="Multi-model comparison experiment")
    parser.add_argument("--model", choices=["flan-base", "qwen", "both"], default="both", help="Which model to run")
    parser.add_argument("--analyze-only", action="store_true", help="Skip inference and only compute metrics")
    args = parser.parse_args()

    if not args.analyze_only:
        tqa_df = load_truthfulqa()
        best_answers_list = tqa_df["best_answer"].tolist()
        faiss_index, embed_model, _ = build_faiss_index(best_answers_list)

        if args.model in ("flan-base", "both"):
            run_experiment("flan-base", tqa_df, faiss_index, embed_model, best_answers_list)

        if args.model in ("qwen", "both"):
            run_experiment("qwen", tqa_df, faiss_index, embed_model, best_answers_list)

    run_combined_analysis()


if __name__ == "__main__":
    main()
