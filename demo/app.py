import streamlit as st
import torch
import json
import os
import faiss
import numpy as np
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

st.set_page_config(page_title="Thesis Demo: RAG + SCE", layout="wide")

# --- PATHS & SETUP ---
KB_PATH = "burger_kb.json"
FEEDBACK_FILE = "demo_feedback.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)

@st.cache_resource
def build_kb_index():
    embedder = load_embedding_model()
    with open(KB_PATH, 'r') as f:
        kb_data = json.load(f)
    
    documents = [item['text'] for item in kb_data]
    embeddings = embedder.encode(documents, convert_to_numpy=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, documents

@st.cache_resource(show_spinner=True)
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

def generate_answer(tokenizer, model, model_type, prompt):
    if model_type == "causal":
        messages = [
            {"role": "system", "content": "Answer the question using only the provided context. Be concise. If there is no context, do your best."},
            {"role": "user", "content": prompt}
        ]
        text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text_prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
        # Extract only the generated text
        answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(**inputs, max_new_tokens=50, num_beams=5)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer

# --- APP UI ---
st.title("🍔 Burger Haven AI: RAG + SCE Thesis Demo")
st.markdown("""
This web application demonstrates the core pipeline of my MSc Thesis. It compares a **Baseline LLM**, a **RAG-enhanced LLM**, and a **RAG + Semantic Consistency Evaluation (SCE)** system using a custom knowledge base (a fictional Burger Shop).
""")

# Sidebar settings
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.selectbox(
    "Select LLM Backbone",
    ["Flan-T5-Small (77M)", "Flan-T5-Base (250M)", "Qwen2.5-0.5B-Instruct (500M)"]
)
tau = st.sidebar.slider("SCE Hallucination Threshold (τ)", 0.0, 1.0, 0.5, 0.05, 
                        help="Answers with semantic similarity below this threshold are flagged as hallucinations.")

st.sidebar.markdown("---")
st.sidebar.subheader("📚 Current Knowledge Base")
with st.sidebar.expander("View Burger Haven KB"):
    with open(KB_PATH, 'r') as f:
        st.json(json.load(f))

# Initialize backend
embedder = load_embedding_model()
index, documents = build_kb_index()

# Load model based on selection (cached)
tokenizer, model, model_type = load_llm(model_choice)

# Main interaction
user_q = st.text_input("💬 Ask the Burger Haven Assistant a question:", placeholder="e.g. Do you sell pizza? or What time do you close on Sundays?")

if st.button("Generate Responses", type="primary") and user_q:
    with st.spinner("Processing pipeline..."):
        # 1. Retrieval
        q_emb = embedder.encode([user_q], convert_to_numpy=True)
        distances, indices = index.search(q_emb, 1)
        retrieved_context = documents[indices[0][0]]
        
        # 2. Baseline Prompting
        if model_type == "causal":
            baseline_prompt = f"Question: {user_q}\n\nAnswer:"
        else:
            baseline_prompt = f"Question: {user_q}\n\nAnswer:"
        
        baseline_ans = generate_answer(tokenizer, model, model_type, baseline_prompt)
        
        # 3. RAG Prompting
        if model_type == "causal":
            rag_prompt = f"Context: {retrieved_context}\n\nQuestion: {user_q}\n\nAnswer in a complete, detailed sentence:"
        else:
            rag_prompt = f"Context: {retrieved_context}\n\nQuestion: {user_q}\n\nAnswer in a complete, detailed sentence:"
            
        rag_ans = generate_answer(tokenizer, model, model_type, rag_prompt)
        
        # 4. SCE Calculation
        ans_emb = embedder.encode([rag_ans], convert_to_numpy=True)
        ctx_emb = embedder.encode([retrieved_context], convert_to_numpy=True)
        
        # Cosine similarity
        ans_norm = ans_emb / np.linalg.norm(ans_emb, axis=1, keepdims=True)
        ctx_norm = ctx_emb / np.linalg.norm(ctx_emb, axis=1, keepdims=True)
        sce_score = np.dot(ans_norm, ctx_norm.T)[0][0]
        
        is_hallucination = sce_score < tau
        
        # Save session state for feedback
        st.session_state['last_run'] = {
            "question": user_q,
            "model": model_choice,
            "retrieved_context": retrieved_context,
            "baseline": baseline_ans,
            "rag": rag_ans,
            "sce_score": float(sce_score),
            "flagged": is_hallucination
        }

# Display Results
if 'last_run' in st.session_state:
    res = st.session_state['last_run']
    
    st.markdown(f"**Retrieved Context:**\n> _{res['retrieved_context']}_")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Baseline LLM")
        st.info(res['baseline'])
        st.caption("Generated purely from parametric memory. Highly prone to guessing or generic answers.")
        
    with col2:
        st.subheader("2. RAG Enhanced")
        st.success(res['rag'])
        st.caption("Generated using the retrieved context. Factually grounded, but the model might still drift.")
        
    with col3:
        st.subheader("3. SCE Verification")
        st.metric("Semantic Consistency Score", f"{res['sce_score']:.3f}")
        if res['flagged']:
            st.error(f"🚨 **FLAGGED AS HALLUCINATION**\n\nScore is below threshold (τ={tau}). The model generated an answer unsupported by the context.")
        else:
            st.success(f"✅ **CONSISTENT**\n\nScore is above threshold (τ={tau}). The answer is heavily grounded in the context.")
            
    st.markdown("---")
    st.subheader("📝 User Feedback (Demo Data Collection)")
    st.write("Does the RAG answer look like a hallucination to you?")
    
    f_col1, f_col2 = st.columns([1, 4])
    with f_col1:
        if st.button("👍 Grounded / Correct"):
            feedback = "Consistent"
        elif st.button("👎 Hallucination / Wrong"):
            feedback = "Hallucinated"
        else:
            feedback = None
            
    if feedback:
        new_data = pd.DataFrame([{
            "timestamp": datetime.now().isoformat(),
            "model": res['model'],
            "question": res['question'],
            "rag_answer": res['rag'],
            "sce_score": res['sce_score'],
            "sce_flagged": res['flagged'],
            "user_feedback": feedback
        }])
        
        if os.path.exists(FEEDBACK_FILE):
            new_data.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
        else:
            new_data.to_csv(FEEDBACK_FILE, index=False)
        st.toast(f"Feedback recorded: {feedback}! Thank you.", icon="📝")

