import streamlit as st
import torch
import faiss
import numpy as np
import random
import plotly.graph_objects as go
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from rouge_score import rouge_scorer

st.set_page_config(page_title="Master's Thesis: RAG+SCE Live Evaluation", layout="wide")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- CACHED LOADERS ---
@st.cache_resource(show_spinner="Loading TruthfulQA Dataset & Indexing (Once)...")
def load_and_index_dataset():
    # Load TruthfulQA validation split
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    df = dataset.to_pandas()
    
    # We will use the 'best_answer' as our ground-truth knowledge base (as described in the thesis)
    documents = df['best_answer'].tolist()
    
    # Load embedding model
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    
    # Encode all documents
    embeddings = embedder.encode(documents, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    
    # Build FAISS Index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return df, index, documents, embedder

@st.cache_resource(show_spinner="Loading Model into Memory...")
def load_llm(model_choice):
    if "qwen" in model_choice.lower():
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float16).to(DEVICE)
        model_type = "causal"
    elif "base" in model_choice.lower():
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(DEVICE)
        model_type = "seq2seq"
    else:
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(DEVICE)
        model_type = "seq2seq"
    return tokenizer, model, model_type

# --- HELPER FUNCTIONS ---
def generate_answer(tokenizer, model, model_type, prompt):
    if model_type == "causal":
        messages = [
            {"role": "system", "content": "Answer the question using only the provided context. Be concise. If there is no context, do your best."},
            {"role": "user", "content": prompt}
        ]
        text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text_prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(**inputs, max_new_tokens=50, num_beams=5)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer

def calculate_rouge(prediction, target):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(target, prediction)
    return scores['rougeL'].fmeasure

def plot_sce_gauge(score, threshold):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Semantic Consistency Evaluation (SCE)", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [0, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, threshold], 'color': "lightcoral"},
                {'range': [threshold, 1], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# --- APP UI ---
st.title("🎓 Master's Thesis Live Evaluation: RAG + SCE")
st.markdown("### Investigating Factual Hallucinations in Small Open-Source LLMs")
st.markdown("_Demonstrating live metric computation (ROUGE-L & Cosine Similarity) on the TruthfulQA Benchmark._")

# Load Backend
df_truthful, faiss_index, kb_documents, embedder = load_and_index_dataset()

st.sidebar.header("🔬 Experimental Controls")
model_choice = st.sidebar.selectbox("Select Generator Model", ["Flan-T5-Small (77M)", "Flan-T5-Base (250M)", "Qwen2.5-0.5B-Instruct (500M)"])
tau = st.sidebar.slider("SCE Hallucination Threshold (τ)", 0.0, 1.0, 0.5, 0.05)

categories = ["All Categories"] + sorted(df_truthful['category'].unique().tolist())
if 'prev_category' not in st.session_state:
    st.session_state.prev_category = "All Categories"
selected_category = st.sidebar.selectbox("Filter Question Category", categories)

tokenizer, model, model_type = load_llm(model_choice)

# State management for random sampling
if 'current_idx' not in st.session_state or selected_category != st.session_state.prev_category:
    st.session_state.prev_category = selected_category
    if selected_category == "All Categories":
        st.session_state.current_idx = random.randint(0, len(df_truthful)-1)
    else:
        subset = df_truthful[df_truthful['category'] == selected_category]
        st.session_state.current_idx = random.choice(subset.index.tolist())

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    if st.button("🎲 Sample Random Question", type="primary"):
        if selected_category == "All Categories":
            st.session_state.current_idx = random.randint(0, len(df_truthful)-1)
        else:
            subset = df_truthful[df_truthful['category'] == selected_category]
            st.session_state.current_idx = random.choice(subset.index.tolist())

# Extract Current Question Data
idx = st.session_state.current_idx
question = df_truthful.iloc[idx]['question']
best_answer = df_truthful.iloc[idx]['best_answer']
category = df_truthful.iloc[idx]['category']

st.markdown("---")
st.subheader("1. Ground Truth (TruthfulQA)")
st.info(f"**Question:** {question}\n\n**Category:** {category}\n\n**Best Answer:** {best_answer}")

# Process
with st.spinner("Retrieving Vectors and Generating Outputs..."):
    # Retrieval
    q_emb = embedder.encode([question], convert_to_numpy=True)
    distances, indices = faiss_index.search(q_emb, 1)
    retrieved_context = kb_documents[indices[0][0]]
    l2_distance = distances[0][0]
    
    # Baseline
    if model_type == "causal":
        baseline_prompt = f"Question: {question}\n\nAnswer:"
        rag_prompt = f"Context: {retrieved_context}\n\nQuestion: {question}\n\nAnswer in a complete, detailed sentence:"
    else:
        baseline_prompt = f"Question: {question}\n\nAnswer:"
        rag_prompt = f"Context: {retrieved_context}\n\nQuestion: {question}\n\nAnswer in a complete, detailed sentence:"
        
    baseline_ans = generate_answer(tokenizer, model, model_type, baseline_prompt)
    rag_ans = generate_answer(tokenizer, model, model_type, rag_prompt)
    
    # Metrics
    base_rouge = calculate_rouge(baseline_ans, best_answer)
    rag_rouge = calculate_rouge(rag_ans, best_answer)
    
    # SCE
    ans_emb = embedder.encode([rag_ans], convert_to_numpy=True)
    ctx_emb = embedder.encode([retrieved_context], convert_to_numpy=True)
    
    ans_norm = ans_emb / np.linalg.norm(ans_emb, axis=1, keepdims=True)
    ctx_norm = ctx_emb / np.linalg.norm(ctx_emb, axis=1, keepdims=True)
    sce_score = float(np.dot(ans_norm, ctx_norm.T)[0][0])
    
st.markdown("---")
st.subheader("2. FAISS Vector Retrieval")
st.code(f"Retrieved Context: {retrieved_context}\nL2 Distance: {l2_distance:.4f} (Lower is closer in semantic space)", language="text")

st.markdown("---")
st.subheader("3. Model Generation & Performance")
col_base, col_rag = st.columns(2)

with col_base:
    st.markdown("#### Baseline LLM (Zero-Shot)")
    st.write(f"> {baseline_ans}")
    st.metric("ROUGE-L (vs Ground Truth)", f"{base_rouge:.3f}")

with col_rag:
    st.markdown("#### RAG-Enhanced LLM")
    st.write(f"> {rag_ans}")
    st.metric("ROUGE-L (vs Ground Truth)", f"{rag_rouge:.3f}", delta=f"{rag_rouge - base_rouge:.3f}")

st.markdown("---")
st.subheader("4. Post-Hoc Hallucination Detection (SCE)")
col_sce1, col_sce2 = st.columns([1, 1])

with col_sce1:
    if sce_score < tau:
        st.error(f"🚨 **FLAGGED AS HALLUCINATION**\n\nThe generation is semantically unsupported by the retrieved context. (Score: {sce_score:.3f} < {tau})")
    else:
        st.success(f"✅ **CONSISTENT / FACTUALLY GROUNDED**\n\nThe generation faithfully adheres to the retrieved context. (Score: {sce_score:.3f} >= {tau})")
    
    st.markdown("""
    **How this is calculated live:**
    1. The RAG answer is encoded via `all-MiniLM-L6-v2`.
    2. The retrieved context is encoded.
    3. The Cosine Similarity between the two 384-dimensional vectors is computed.
    """)

with col_sce2:
    fig = plot_sce_gauge(sce_score, tau)
    st.plotly_chart(fig, use_container_width=True)
