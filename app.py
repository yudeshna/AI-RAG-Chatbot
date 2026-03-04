import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import requests

# -----------------------------
# Settings
# -----------------------------

DB_DIR = "./chroma_db"
COLLECTION_NAME = "rag_docs"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# -----------------------------
# Load Models
# -----------------------------

@st.cache_resource
def load_embedder():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(
        path=DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(name=COLLECTION_NAME)

# -----------------------------
# Read PDF
# -----------------------------

def read_pdf(file):

    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content

    return text

# -----------------------------
# Chunk Text
# -----------------------------

def split_text(text, size=700, overlap=120):

    chunks = []
    start = 0

    while start < len(text):

        end = start + size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += size - overlap

    return chunks

# -----------------------------
# Store Documents
# -----------------------------

def store_docs(collection, embedder, text, filename):

    chunks = split_text(text)

    if not chunks:
        return 0

    embeddings = embedder.encode(chunks).tolist()

    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    metadata = [{"source": filename} for _ in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadata
    )

    return len(chunks)

# -----------------------------
# Retrieve Context
# -----------------------------

def retrieve(collection, embedder, query, k=4):

    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )

    return results["documents"][0]

# -----------------------------
# LLM Call
# -----------------------------

def ask_llm(api_key, model, question, context):

    prompt = f"""
Answer using ONLY the context below.

Context:
{context}

Question:
{question}
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=data
    )

    result = response.json()

    if "choices" not in result:
        return f"API Error: {result}"

    return result["choices"][0]["message"]["content"]

# -----------------------------
# App UI
# -----------------------------

def main():

    st.set_page_config(page_title="Simple Local RAG", layout="wide")
    st.title("Simple Local RAG Chat (Streamlit + ChromaDB)")

    embedder = load_embedder()
    collection = load_collection()

    # -----------------------------
    # Sidebar
    # -----------------------------

    st.sidebar.subheader("LLM Settings")

    api_key = st.sidebar.text_input("OpenRouter API Key", type="password")

    model_name = st.sidebar.text_input(
        "Model",
        "mistralai/mistral-7b-instruct:free"
    )

    top_k = st.sidebar.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=8,
        value=4
    )

    st.sidebar.divider()

    st.sidebar.subheader("Ingest Documents")

    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if st.sidebar.button("Index Uploaded File"):

        if uploaded_file:

            text = read_pdf(uploaded_file)

            with st.spinner("Embedding and storing chunks..."):

                count = store_docs(
                    collection,
                    embedder,
                    text,
                    uploaded_file.name
                )

            st.sidebar.success(f"Indexed {count} chunks")

        else:
            st.sidebar.warning("Upload a PDF first.")

    if st.sidebar.button("Reset Vector DB"):

        client = chromadb.PersistentClient(
            path=DB_DIR,
            settings=Settings(anonymized_telemetry=False)
        )

        client.delete_collection(COLLECTION_NAME)
        client.get_or_create_collection(COLLECTION_NAME)

        st.sidebar.success("Vector database reset.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.success("Chat cleared.")

    # -----------------------------
    # Chat Interface
    # -----------------------------

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask about your docs...")

    if question:

        if not api_key:
            st.warning("Add your OpenRouter API key.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Retrieving context..."):

            docs = retrieve(collection, embedder, question, k=top_k)

        context = "\n".join(docs)

        with st.chat_message("assistant"):

            with st.spinner("Generating answer..."):

                answer = ask_llm(
                    api_key,
                    model_name,
                    question,
                    context
                )

            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":

    os.makedirs(DB_DIR, exist_ok=True)

    main()