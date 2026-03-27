# RAG-Powered Customer Support Chatbot (Streamlit Version)
# --------------------------------------------------------
# This app builds a simple Retrieval-Augmented Generation (RAG) chatbot
# for answering customer support questions using a CSV knowledge base.

# ---------------------------
# 1. Page Setup
# ---------------------------
import streamlit as st
import pandas as pd
import re
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Configure Streamlit page layout and title
st.set_page_config(page_title="RAG Chatbot (Basics)", page_icon="🤖")
st.title("🤖 RAG-Powered Customer Support Chatbot")

# ---------------------------
# 2. Load & Clean Dataset
# ---------------------------
# Function to clean text by removing placeholders like {{...}} and
# extra whitespace, and converting to lowercase

def clean(s):
    s = re.sub(r"\{\{.*?\}\}", "", str(s))            # Remove placeholders
    return re.sub(r"\s+", " ", s).strip().lower()        # Normalize spacing & case

# Load and cache dataset after cleaning instruction & response
@st.cache_resource
def load_data():
    df = pd.read_csv("Customer_Support_Training_Dataset.csv")
    df["instruction_clean"] = df["instruction"].apply(clean)
    df["response_clean"] = df["response"].apply(clean)
    return df

df = load_data()

# ---------------------------
# 3. Chunk & Embed
# ---------------------------
# Splits rows into smaller overlapping word chunks for better retrieval

def chunk_words(text, n=150, overlap=30):
    words = text.split()
    step = n - overlap
    return [" ".join(words[i:i+n]) for i in range(0, len(words), step)]

# Build and cache list of mini documents (chunks) with ID metadata
@st.cache_resource
def build_chunks(df):
    docs = []
    for rid, row in df.iterrows():
        text = f"instruction: {row['instruction_clean']} | response: {row['response_clean']}"
        for i, ch in enumerate(chunk_words(text)):
            docs.append({"rid": rid, "chunk_id": i, "text": ch})
    return docs

mini_docs = build_chunks(df)
chunk_texts = [d["text"] for d in mini_docs]  # Just the raw text for embeddings

# Load sentence-transformer model and compute normalized embeddings
@st.cache_resource
def get_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = model.encode(chunk_texts, normalize_embeddings=True).astype("float32")
    return model, X

embedder, X = get_embeddings()

# Build FAISS index for fast similarity search (dot-product based)
index = faiss.IndexFlatIP(X.shape[1])
index.add(X)

# ---------------------------
# 4. OpenAI Setup
# ---------------------------
# Load your API key and set model name for chat completion
api_key_path = Path("api_key.txt")
client = OpenAI(api_key=api_key_path.read_text().strip())
LLM_MODEL = "gpt-4o-mini"

# ---------------------------
# 5. RAG Function
# ---------------------------
# Retrieve top-k relevant chunks using embeddings and pass to LLM

def answer_with_rag(query, k=3, pool=200, temp=0.7, max_tokens=500):
    # Step 1: Embed the question and search top "pool" candidates
    qv = embedder.encode([query], normalize_embeddings=True).astype("float32")
    _, I = index.search(qv, pool)  # I = indices of top matches

    # Step 2: Pick first k chunks (with min 20 words for quality)
    hits = []
    for idx in I[0]:
        doc = mini_docs[int(idx)]
        if len(doc["text"].split()) >= 20:
            hits.append(doc)
        if len(hits) >= k:
            break

    if not hits:
        return "Sorry, no relevant info found.", []

    # Step 3: Build context string with document previews
    context = "\n\n".join([
        f"[Doc {i+1}] (rid={h['rid']}, chunk={h['chunk_id']})\n{h['text']}"
        for i, h in enumerate(hits)
    ])

    # Step 4: Send to OpenAI chat model using a grounded prompt
    prompt = f"""Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful customer support assistant. Use only the context. Be clear and polite. Cite sources like [Doc 1]."},
            {"role": "user", "content": prompt}
        ],
        temperature=temp,
        max_tokens=max_tokens,
    )

    answer = response.choices[0].message.content.strip()
    return answer, hits

# ---------------------------
# 6. Streamlit Chat Interface
# ---------------------------
# Enables a chat-style UI with memory of user/assistant messages

# Initialize chat history in session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history (alternating roles)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box for user's next question
query = st.chat_input("Ask a support question...")
if query:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": query})

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(query)

    # LLM response box with spinner during search
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            answer, hits = answer_with_rag(query)
            st.markdown(answer)

        # Expandable section to show source chunks
        with st.expander("Context Chunks Used"):
            for i, h in enumerate(hits, 1):
                st.markdown(f"**[Doc {i}] rid={h['rid']} | chunk={h['chunk_id']}**")
                st.markdown(h["text"])

    # Store assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
