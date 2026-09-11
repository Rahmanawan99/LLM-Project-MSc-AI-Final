# LLM Project MSc AI Final

Source code for an MSc Artificial Intelligence final project investigating retrieval-augmented generation (RAG) and semantic consistency evaluation (SCE) with open language models.

## What This Repository Contains

- `src/`: scripts for model inference, RAG experiments, SCE scoring, and analysis.
- `notebooks/`: exploratory versions of the dataset, baseline, RAG, and analysis workflow.
- `data/`: small example documents and an embedding demonstration.
- `results/`: committed raw model outputs and derived analysis tables. These files are intentionally retained for inspection and comparison.
- `Other support codes/`: optional checks for a local thesis document. These scripts require a document path supplied by the user.

## Requirements

- Python 3.10 or newer.
- A CPU can run the workflows, but model inference is substantially faster with a compatible GPU.
- Several gigabytes of free disk space are required for downloaded datasets and model weights.

Install the pinned Python dependencies in a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pinned PyTorch package targets the CPU. For a CUDA installation, use the official PyTorch installation selector and then install the remaining dependencies from `requirements.txt`.

## Workflow

Run notebooks from the `notebooks/` directory so their relative result paths resolve correctly. The intended order is:

1. `01_dataset_exploration.ipynb`
2. `02_baseline_llm.ipynb`
3. `03_rag_pipeline.ipynb`
4. `04_semantic_consistency.ipynb`
5. `05_analysis.ipynb`
6. `05b_evaluation_analysis.ipynb`

For the multi-model workflow, run from the repository root:

```bash
python src/run_multi_model.py --model both
python src/run_multi_model.py --analyze-only
```

The scripts download models and the TruthfulQA dataset through Hugging Face on first use. Results and generated figures are written below `results/`. The raw CSV files in this repository are generated artifacts, not hand-authored ground truth.

## Data, Models, and Attribution

This project uses the validation split of the [TruthfulQA generation dataset](https://huggingface.co/datasets/truthfulqa/truthful_qa), the following language models, and the following embedding model:

- [google/flan-t5-small](https://huggingface.co/google/flan-t5-small)
- [google/flan-t5-base](https://huggingface.co/google/flan-t5-base)
- [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

Review the upstream dataset and model cards before redistributing this repository or its generated outputs. Their licenses and terms are separate from this project's MIT license. The research also depends on Hugging Face Datasets, Transformers, PyTorch, Sentence Transformers, FAISS, pandas, NumPy, SciPy, scikit-learn, Matplotlib, ROUGE, and Jupyter.

## Important Evaluation Limitations

The RAG index is built from TruthfulQA validation answers and queried with the same validation questions. This means the experiment is a proof-of-concept and may leak evaluation answers through retrieval; it should not be interpreted as an unbiased held-out RAG benchmark.

SCE uses cosine similarity between a generated answer and retrieved context. The threshold-based `hallucination_flag` is a proxy signal, not a verified factual label. The reported ROUGE-L threshold comparison is also a heuristic evaluation and should not be described as ground-truth hallucination detection without additional human or independently verified annotation.

The CSV files contain unfiltered model generations, including incorrect or sensitive responses to benchmark questions. They are provided for research analysis only and must not be treated as medical, legal, financial, safety, or other professional advice.

## License

The original source code is released under the [MIT License](LICENSE). External datasets, model weights, and generated artifacts may have separate terms; see their upstream sources before reuse or redistribution.
