import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
import pandas as pd
import numpy as np

MULTI_DIR = r'c:\Users\rahma\Documents\Playground\LLM-Project\results\multi_model'

models = {
    "Flan-T5-Small (77M)": "flan-t5-small",
    "Flan-T5-Base (250M)": "flan-t5-base",
    "Qwen2.5-0.5B (500M)": "qwen2.5-0.5b"
}

print("=== SCE CLASSIFIER PERFORMANCE (ALL 3 MODELS, threshold=0.5, ground-truth: ROUGE-L < 0.2) ===")

rows = []
for display_name, prefix in models.items():
    rag_csv = os.path.join(MULTI_DIR, f"{prefix}_rag_results.csv")
    sce_csv = os.path.join(MULTI_DIR, f"{prefix}_sce_results.csv")
    
    rag_df = pd.read_csv(rag_csv)
    sce_df = pd.read_csv(sce_csv)
    
    # Compute ROUGE-L
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    rouge_l_scores = []
    for _, r in rag_df.iterrows():
        score = scorer.score(str(r['best_answer']), str(r['generated_answer']))
        rouge_l_scores.append(score['rougeL'].fmeasure)
        
    actual_halluc = [1 if r < 0.2 else 0 for r in rouge_l_scores]
    flags = sce_df['hallucination_flag'].tolist() # 1 = flagged, 0 = consistent
    
    tp = sum(1 for a, f in zip(actual_halluc, flags) if a == 1 and f == 1)
    fp = sum(1 for a, f in zip(actual_halluc, flags) if a == 0 and f == 1)
    fn = sum(1 for a, f in zip(actual_halluc, flags) if a == 1 and f == 0)
    tn = sum(1 for a, f in zip(actual_halluc, flags) if a == 0 and f == 0)
    
    n = len(actual_halluc)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    acc = (tp + tn) / n
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    actual_halluc_rate = sum(actual_halluc) / n
    flagged_rate = sum(flags) / n
    
    rows.append({
        "Model": display_name,
        "Actual Hallucinations (ROUGE-L < 0.2)": f"{sum(actual_halluc)} ({actual_halluc_rate:.1%})",
        "Flagged by SCE (τ=0.5)": f"{sum(flags)} ({flagged_rate:.1%})",
        "True Positives (TP)": tp,
        "False Positives (FP)": fp,
        "True Negatives (TN)": tn,
        "False Negatives (FN)": fn,
        "Accuracy": f"{acc:.3f} ({acc*100:.1f}%)",
        "Precision": f"{prec:.3f}",
        "Recall": f"{rec:.3f}",
        "F1 Score": f"{f1:.3f}",
        "Specificity": f"{spec:.3f}"
    })

res_df = pd.DataFrame(rows)
print(res_df.to_string(index=False))

# Also transpose for a clean table layout
transposed = res_df.set_index("Model").T
print("\n=== TRANSPOSED FOR WORD TABLE ===")
print(transposed.to_string())
