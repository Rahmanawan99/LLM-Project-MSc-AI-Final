"""
Multi-Model Comparison Experiment
==================================
Runs the full pipeline (Baseline + RAG + SCE) on two additional models:
  1. Flan-T5-Base  (250M params, encoder-decoder)
  2. Qwen2.5-0.5B-Instruct (500M params, autoregressive decoder-only)

Then computes all metrics and generates comparison tables and figures.

Usage:
    python run_multi_model.py                    # Run both models
    python run_multi_model.py --model flan-base   # Run only Flan-T5-Base
    python run_multi_model.py --model qwen        # Run only Qwen2.5-0.5B
    python run_multi_model.py --analyze-only       # Skip inference, just analyze
"""

import os
import sys
import io
import csv
import time
import json
import argparse
import warnings
import numpy as np
import pandas as pd

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# ─── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MULTI_DIR = os.path.join(RESULTS_DIR, "multi_model")
FIGURES_DIR = os.path.join(MULTI_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Logging ─────────────────────────────────────────────────────────────────
def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─── Device ──────────────────────────────────────────────────────────────────
import torch
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    log(f"GPU detected: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB)")
else:
    DEVICE = torch.device("cpu")
    log("No GPU detected, using CPU")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Model Definitions
# ═══════════════════════════════════════════════════════════════════════════════

def load_flan_t5_base():
    """Load Flan-T5-Base (250M params, encoder-decoder)."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_name = "google/flan-t5-base"
    log(f"Downloading/loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    model.eval()
    log(f"  Loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return tokenizer, model, "flan-t5-base"


def generate_flan_t5(tokenizer, model, prompt):
    """Generate answer using Flan-T5 (same method as original experiment)."""
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            num_beams=5,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def load_qwen():
    """Load Qwen2.5-0.5B-Instruct (500M params, autoregressive)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    log(f"Downloading/loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # CPU needs float32
    )
    model.eval()
    log(f"  Loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return tokenizer, model, "qwen2.5-0.5b"


def generate_qwen_baseline(tokenizer, model, question):
    """Generate baseline answer using Qwen (no context)."""
    import torch
    messages = [
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )
    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_qwen_rag(tokenizer, model, question, context):
    """Generate RAG answer using Qwen (with retrieved context)."""
    import torch
    messages = [
        {"role": "system", "content": "Answer the question using only the provided context. Be concise."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Data Loading (TruthfulQA + FAISS)
# ═══════════════════════════════════════════════════════════════════════════════

def load_truthfulqa():
    """Load TruthfulQA dataset."""
    from datasets import load_dataset
    log("Loading TruthfulQA dataset...")
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    df = ds.to_pandas()
    log(f"  Loaded {len(df)} questions, {df['category'].nunique()} categories")
    return df


def build_faiss_index(best_answers):
    """Build FAISS index from best_answer strings (same as original experiment)."""
    import faiss
    from sentence_transformers import SentenceTransformer

    log("Building FAISS index from best_answers...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(best_answers, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    log(f"  FAISS index built: {index.ntotal} vectors, dim={embeddings.shape[1]}")
    return index, embed_model, embeddings


def retrieve_context(question, embed_model, faiss_index, best_answers, k=2):
    """Retrieve top-k documents; return the top-1 as context (matching original)."""
    query_vec = embed_model.encode([question]).astype("float32")
    distances, indices = faiss_index.search(query_vec, k)
    top_idx = indices[0][0]
    return best_answers[top_idx]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Run Experiments
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(model_key, tqa_df, faiss_index, embed_model, best_answers_list):
    """Run full Baseline + RAG + SCE pipeline for a given model."""
    from sentence_transformers import SentenceTransformer, util as st_util

    # Load the model
    if model_key == "flan-base":
        tokenizer, model, model_label = load_flan_t5_base()
    elif model_key == "qwen":
        tokenizer, model, model_label = load_qwen()
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    questions = tqa_df["question"].tolist()
    best_answers = tqa_df["best_answer"].tolist()
    n = len(questions)

    # Output CSV paths
    baseline_path = os.path.join(MULTI_DIR, f"{model_label}_baseline_results.csv")
    rag_path = os.path.join(MULTI_DIR, f"{model_label}_rag_results.csv")
    sce_path = os.path.join(MULTI_DIR, f"{model_label}_sce_results.csv")

    # Check if already done (resume support)
    if all(os.path.exists(p) for p in [baseline_path, rag_path, sce_path]):
        existing = pd.read_csv(baseline_path)
        if len(existing) >= n:
            log(f"  ✓ {model_label} results already exist ({len(existing)} rows). Skipping.")
            return model_label

    log(f"{'='*60}")
    log(f"Running experiment: {model_label} ({n} questions)")
    log(f"{'='*60}")

    # --- Baseline ---
    log(f"[{model_label}] Stage 1/3: Baseline generation...")
    baseline_rows = []
    for i, q in enumerate(questions):
        if model_key == "flan-base":
            ans = generate_flan_t5(tokenizer, model, q)
        else:
            ans = generate_qwen_baseline(tokenizer, model, q)

        baseline_rows.append({
            "question": q,
            "generated_answer": ans,
            "best_answer": best_answers[i],
        })

        if (i + 1) % 50 == 0:
            log(f"  Baseline: {i+1}/{n} done")

    pd.DataFrame(baseline_rows).to_csv(baseline_path, index=False)
    log(f"  ✓ Baseline saved: {baseline_path}")

    # --- RAG ---
    log(f"[{model_label}] Stage 2/3: RAG generation...")
    rag_rows = []
    for i, q in enumerate(questions):
        ctx = retrieve_context(q, embed_model, faiss_index, best_answers_list)

        if model_key == "flan-base":
            prompt = f"Context: {ctx}\n\nQuestion: {q}\n\nAnswer:"
            ans = generate_flan_t5(tokenizer, model, prompt)
        else:
            ans = generate_qwen_rag(tokenizer, model, q, ctx)

        rag_rows.append({
            "question": q,
            "retrieved_context": ctx,
            "generated_answer": ans,
            "best_answer": best_answers[i],
        })

        if (i + 1) % 50 == 0:
            log(f"  RAG: {i+1}/{n} done")

    pd.DataFrame(rag_rows).to_csv(rag_path, index=False)
    log(f"  ✓ RAG saved: {rag_path}")

    # --- SCE ---
    log(f"[{model_label}] Stage 3/3: Computing SCE scores...")
    sce_model = SentenceTransformer("all-MiniLM-L6-v2")
    sce_rows = []
    threshold = 0.5

    for i, row in enumerate(rag_rows):
        ans_emb = sce_model.encode([row["generated_answer"]])
        ctx_emb = sce_model.encode([row["retrieved_context"]])
        sim = float(st_util.cos_sim(ans_emb, ctx_emb)[0][0])
        flag = 1 if sim < threshold else 0

        sce_rows.append({
            "question": row["question"],
            "retrieved_context": row["retrieved_context"],
            "generated_answer": row["generated_answer"],
            "semantic_similarity_score": sim,
            "hallucination_flag": flag,
        })

        if (i + 1) % 100 == 0:
            log(f"  SCE: {i+1}/{n} done")

    pd.DataFrame(sce_rows).to_csv(sce_path, index=False)
    log(f"  ✓ SCE saved: {sce_path}")

    # Free memory
    del model, tokenizer
    import gc
    gc.collect()

    log(f"  ✓ {model_label} complete!")
    return model_label


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Metrics Computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics_for_model(model_label):
    """Compute ROUGE-L, semantic similarity, and SCE stats for one model."""
    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer, util as st_util

    log(f"Computing metrics for {model_label}...")

    baseline_df = pd.read_csv(os.path.join(MULTI_DIR, f"{model_label}_baseline_results.csv"))
    rag_df = pd.read_csv(os.path.join(MULTI_DIR, f"{model_label}_rag_results.csv"))
    sce_df = pd.read_csv(os.path.join(MULTI_DIR, f"{model_label}_sce_results.csv"))

    # Fill NaN
    for df in [baseline_df, rag_df, sce_df]:
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].fillna("")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    results = {}

    # --- Baseline metrics ---
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    for _, row in baseline_df.iterrows():
        scores = scorer.score(str(row["best_answer"]), str(row["generated_answer"]))
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    gen_embs = embed_model.encode(baseline_df["generated_answer"].astype(str).tolist(), batch_size=64)
    ref_embs = embed_model.encode(baseline_df["best_answer"].astype(str).tolist(), batch_size=64)
    baseline_sem_sim = [float(st_util.cos_sim(g.reshape(1, -1), r.reshape(1, -1))[0][0])
                        for g, r in zip(gen_embs, ref_embs)]

    results["baseline"] = {
        "ROUGE-1": np.mean(rouge1_scores),
        "ROUGE-2": np.mean(rouge2_scores),
        "ROUGE-L": np.mean(rougeL_scores),
        "Semantic Similarity": np.mean(baseline_sem_sim),
        "rougeL_per_q": rougeL_scores,
        "semsim_per_q": baseline_sem_sim,
    }

    # --- RAG metrics ---
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    for _, row in rag_df.iterrows():
        scores = scorer.score(str(row["best_answer"]), str(row["generated_answer"]))
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    gen_embs = embed_model.encode(rag_df["generated_answer"].astype(str).tolist(), batch_size=64)
    ref_embs = embed_model.encode(rag_df["best_answer"].astype(str).tolist(), batch_size=64)
    rag_sem_sim = [float(st_util.cos_sim(g.reshape(1, -1), r.reshape(1, -1))[0][0])
                   for g, r in zip(gen_embs, ref_embs)]

    results["rag"] = {
        "ROUGE-1": np.mean(rouge1_scores),
        "ROUGE-2": np.mean(rouge2_scores),
        "ROUGE-L": np.mean(rougeL_scores),
        "Semantic Similarity": np.mean(rag_sem_sim),
        "rougeL_per_q": rougeL_scores,
        "semsim_per_q": rag_sem_sim,
    }

    # --- SCE metrics ---
    sce_scores = sce_df["semantic_similarity_score"].values
    flags = sce_df["hallucination_flag"].values
    halluc_rate = float(flags.sum()) / len(flags)

    # SCE classifier evaluation (same as original: ROUGE-L < 0.2 = actual hallucination)
    actual_halluc = [1 if r < 0.2 else 0 for r in rougeL_scores]
    tp = sum(1 for a, f in zip(actual_halluc, flags) if a == 1 and f == 1)
    fp = sum(1 for a, f in zip(actual_halluc, flags) if a == 0 and f == 1)
    fn = sum(1 for a, f in zip(actual_halluc, flags) if a == 1 and f == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results["sce"] = {
        "Hallucination Rate": halluc_rate,
        "SCE Mean Score": float(np.mean(sce_scores)),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }

    # --- Statistical tests (Baseline vs RAG) ---
    from scipy.stats import wilcoxon
    stat_rl, p_rl = wilcoxon(results["baseline"]["rougeL_per_q"],
                              results["rag"]["rougeL_per_q"])
    stat_ss, p_ss = wilcoxon(results["baseline"]["semsim_per_q"],
                              results["rag"]["semsim_per_q"])

    # Cohen's d
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

    # ROUGE-L improvement
    bl_rl = results["baseline"]["ROUGE-L"]
    rag_rl = results["rag"]["ROUGE-L"]
    results["improvement_pct"] = ((rag_rl - bl_rl) / bl_rl * 100) if bl_rl > 0 else float("inf")

    log(f"  {model_label}: Baseline ROUGE-L={bl_rl:.4f} → RAG ROUGE-L={rag_rl:.4f} "
        f"(+{results['improvement_pct']:.1f}%)")
    log(f"  {model_label}: SCE F1={f1:.3f}, Halluc Rate={halluc_rate:.1%}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Combined Analysis & Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def run_combined_analysis():
    """Load all model results, compute metrics, build comparison tables & charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    # Original Flan-T5-Small results
    orig_baseline = os.path.join(RESULTS_DIR, "baseline_results.csv")
    orig_rag = os.path.join(RESULTS_DIR, "rag_results.csv")
    orig_sce = os.path.join(RESULTS_DIR, "sce_results.csv")
    if all(os.path.exists(p) for p in [orig_baseline, orig_rag, orig_sce]):
        models["flan-t5-small"] = "Flan-T5-Small (77M)"

    # New models
    for label in ["flan-t5-base", "qwen2.5-0.5b"]:
        bp = os.path.join(MULTI_DIR, f"{label}_baseline_results.csv")
        if os.path.exists(bp):
            nice_name = {
                "flan-t5-base": "Flan-T5-Base (250M)",
                "qwen2.5-0.5b": "Qwen2.5-0.5B (500M)",
            }[label]
            models[label] = nice_name

    if len(models) < 2:
        log("ERROR: Need at least 2 models to compare. Run experiments first.")
        return

    log(f"Found {len(models)} models: {list(models.values())}")

    # Compute metrics for each
    all_metrics = {}

    # Handle original flan-t5-small differently (results in different location)
    if "flan-t5-small" in models:
        # Symlink/copy original results to multi_model dir for uniform processing
        import shutil
        for suffix in ["baseline_results.csv", "rag_results.csv", "sce_results.csv"]:
            src = os.path.join(RESULTS_DIR, suffix)
            dst = os.path.join(MULTI_DIR, f"flan-t5-small_{suffix}")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    for label in models:
        all_metrics[label] = compute_metrics_for_model(label)

    # ─── Table 1: Cross-Model Comparison ─────────────────────────────────────
    log("Building comparison table...")
    table_rows = []
    for label, nice_name in models.items():
        m = all_metrics[label]
        table_rows.append({
            "Model": nice_name,
            "Parameters": label.split("(")[0].strip(),
            "Architecture": "Encoder-Decoder" if "flan" in label else "Decoder-Only",
            "Baseline ROUGE-L": round(m["baseline"]["ROUGE-L"], 4),
            "RAG ROUGE-L": round(m["rag"]["ROUGE-L"], 4),
            "Improvement (%)": round(m["improvement_pct"], 1),
            "Baseline Sem.Sim": round(m["baseline"]["Semantic Similarity"], 4),
            "RAG Sem.Sim": round(m["rag"]["Semantic Similarity"], 4),
            "Halluc. Rate": f"{m['sce']['Hallucination Rate']:.1%}",
            "SCE F1": round(m["sce"]["F1"], 3),
            "p-value (ROUGE-L)": f"{m['stats']['ROUGE-L_p']:.2e}",
            "Cohen's d": round(m["stats"]["ROUGE-L_cohens_d"], 3),
        })

    comp_df = pd.DataFrame(table_rows)
    comp_path = os.path.join(MULTI_DIR, "cross_model_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    log(f"  ✓ Saved: {comp_path}")

    # Print table
    log("\n" + "=" * 80)
    log("CROSS-MODEL COMPARISON TABLE")
    log("=" * 80)
    print(comp_df.to_string(index=False))
    print()

    # ─── Figure A: ROUGE-L Comparison (Baseline vs RAG per model) ────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = [models[l] for l in models]
    baseline_rougeL = [all_metrics[l]["baseline"]["ROUGE-L"] for l in models]
    rag_rougeL = [all_metrics[l]["rag"]["ROUGE-L"] for l in models]

    x = np.arange(len(model_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, baseline_rougeL, width, label="Baseline", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + width/2, rag_rougeL, width, label="RAG", color="#3498db", alpha=0.85)

    ax.set_ylabel("ROUGE-L F1 Score")
    ax.set_title("Baseline vs RAG: ROUGE-L Across Models")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, max(max(baseline_rougeL), max(rag_rougeL)) * 1.25)

    # Add value labels
    for bar in bars1:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)

    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig_multi_rougeL_comparison.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    log(f"  ✓ Saved: {fig_path}")

    # ─── Figure B: Semantic Similarity Comparison ────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    baseline_ss = [all_metrics[l]["baseline"]["Semantic Similarity"] for l in models]
    rag_ss = [all_metrics[l]["rag"]["Semantic Similarity"] for l in models]

    bars1 = ax.bar(x - width/2, baseline_ss, width, label="Baseline", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + width/2, rag_ss, width, label="RAG", color="#3498db", alpha=0.85)

    ax.set_ylabel("Semantic Similarity (Cosine)")
    ax.set_title("Baseline vs RAG: Semantic Similarity Across Models")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, max(max(baseline_ss), max(rag_ss)) * 1.25)

    for bar in bars1:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)

    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig_multi_semsim_comparison.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    log(f"  ✓ Saved: {fig_path}")

    # ─── Figure C: RAG Improvement % Bar Chart ───────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    improvements = [all_metrics[l]["improvement_pct"] for l in models]
    colors = ["#2ecc71" if imp > 0 else "#e74c3c" for imp in improvements]
    bars = ax.bar(model_names, improvements, color=colors, alpha=0.85)

    ax.set_ylabel("ROUGE-L Improvement (%)")
    ax.set_title("RAG Improvement Over Baseline Across Models")
    ax.axhline(y=0, color="black", linewidth=0.5)

    for bar, imp in zip(bars, improvements):
        ax.annotate(f"+{imp:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center",
                    fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig_multi_improvement.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    log(f"  ✓ Saved: {fig_path}")

    # ─── Figure D: SCE Performance Across Models ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Hallucination rate
    halluc_rates = [all_metrics[l]["sce"]["Hallucination Rate"] for l in models]
    bars = axes[0].bar(model_names, [h * 100 for h in halluc_rates], color="#e67e22", alpha=0.85)
    axes[0].set_ylabel("Hallucination Rate (%)")
    axes[0].set_title("Hallucination Rate at τ=0.5")
    axes[0].set_xticklabels(model_names, rotation=15, ha="right")
    for bar, h in zip(bars, halluc_rates):
        axes[0].annotate(f"{h:.1%}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)

    # Right: SCE F1
    f1_scores = [all_metrics[l]["sce"]["F1"] for l in models]
    bars = axes[1].bar(model_names, f1_scores, color="#9b59b6", alpha=0.85)
    axes[1].set_ylabel("SCE Classifier F1 Score")
    axes[1].set_title("SCE Detection Performance (F1)")
    axes[1].set_xticklabels(model_names, rotation=15, ha="right")
    axes[1].set_ylim(0, 1)
    for bar, f in zip(bars, f1_scores):
        axes[1].annotate(f"{f:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)

    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig_multi_sce_comparison.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    log(f"  ✓ Saved: {fig_path}")

    # ─── Summary ─────────────────────────────────────────────────────────────
    log("\n" + "=" * 80)
    log("SUMMARY")
    log("=" * 80)
    for label, nice_name in models.items():
        m = all_metrics[label]
        log(f"  {nice_name}:")
        log(f"    Baseline ROUGE-L: {m['baseline']['ROUGE-L']:.4f}")
        log(f"    RAG ROUGE-L:      {m['rag']['ROUGE-L']:.4f} (+{m['improvement_pct']:.1f}%)")
        log(f"    RAG Sem.Sim:      {m['rag']['Semantic Similarity']:.4f}")
        log(f"    Halluc Rate:      {m['sce']['Hallucination Rate']:.1%}")
        log(f"    SCE F1:           {m['sce']['F1']:.3f}")
        log(f"    Wilcoxon p:       {m['stats']['ROUGE-L_p']:.2e}")
        log(f"    Cohen's d:        {m['stats']['ROUGE-L_cohens_d']:.3f}")
        log("")

    log("All outputs saved to: " + MULTI_DIR)
    log("Done! ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Multi-model comparison experiment")
    parser.add_argument("--model", choices=["flan-base", "qwen", "both"],
                        default="both", help="Which model to run (default: both)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip inference, just compute metrics and charts")
    args = parser.parse_args()

    if not args.analyze_only:
        # Load data and build FAISS
        tqa_df = load_truthfulqa()
        best_answers_list = tqa_df["best_answer"].tolist()
        faiss_index, embed_model, _ = build_faiss_index(best_answers_list)

        # Run experiments
        if args.model in ("flan-base", "both"):
            run_experiment("flan-base", tqa_df, faiss_index, embed_model, best_answers_list)

        if args.model in ("qwen", "both"):
            run_experiment("qwen", tqa_df, faiss_index, embed_model, best_answers_list)

    # Always run analysis at the end
    run_combined_analysis()


if __name__ == "__main__":
    main()
