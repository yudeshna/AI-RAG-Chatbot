import warnings
warnings.filterwarnings("ignore")

import unicodedata
import os
import streamlit as st
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import requests


# -----------------------------
# Settings
# -----------------------------

DB_DIR = "/tmp/chroma_db"
COLLECTION_NAME = "rag_docs"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# -----------------------------
# Load Models
# -----------------------------

@st.cache_resource
def load_embedder():
    return SentenceTransformer(MODEL_NAME)


# -----------------------------
# ChromaDB Client
# -----------------------------

@st.cache_resource
def get_chroma_client():
    """
    Create one ChromaDB client with consistent settings.
    """
    from chromadb.config import Settings

    return chromadb.PersistentClient(
        path=DB_DIR,
        settings=Settings(
            anonymized_telemetry=False,
            is_persistent=True
        )
    )

def get_collection():
    """
    Get or create the RAG document collection.
    """
    client = get_chroma_client()

    return client.get_or_create_collection(
        name=COLLECTION_NAME
    )


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

    clean_chunks = []

    for i, c in enumerate(chunks):

        if not isinstance(c, str):
            st.warning(
                f"Chunk {i} is not a string: "
                f"{type(c)} — {repr(c)}"
            )
            continue

        cleaned = c.strip()

        if not cleaned:
            continue

        # Remove problematic characters
        cleaned = cleaned.replace("\x00", "")
        cleaned = cleaned.replace("\ufffd", "")

        # Normalize unicode
        cleaned = unicodedata.normalize(
            "NFKD",
            cleaned
        )

        # Convert to ASCII-safe text
        cleaned = (
            cleaned
            .encode("ascii", errors="ignore")
            .decode("ascii")
        )

        cleaned = cleaned.strip()

        if cleaned:
            clean_chunks.append(cleaned)

    if not clean_chunks:

        st.error(
            "No valid chunks after cleaning!"
        )

        return 0

    st.sidebar.info(
        f"🔍 Encoding {len(clean_chunks)} chunks..."
    )

    embeddings = []

    for i, chunk in enumerate(clean_chunks):

        try:

            emb = embedder.encode(
                [chunk]
            ).tolist()[0]

            embeddings.append(emb)

        except Exception as e:

            st.error(
                f"❌ Failed on chunk {i}: "
                f"{repr(chunk[:100])}"
            )

            st.error(
                f"Error: {e}"
            )

            return 0

    ids = [
        f"{filename}_{i}"
        for i in range(len(clean_chunks))
    ]

    metadata = [
        {"source": filename}
        for _ in clean_chunks
    ]

    collection.add(
        documents=clean_chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadata
    )

    return len(clean_chunks)


# -----------------------------
# Retrieve Context
# -----------------------------

def retrieve(collection, embedder, query, k=4):

    is_summary = any(
        word in query.lower()
        for word in [
            "summarize",
            "summary",
            "overview",
            "what is this about",
            "what does this say",
            "explain the document",
            "brief"
        ]
    )

    if is_summary:

        search_query = (
            "main topics key points "
            "overview introduction"
        )

    else:

        search_query = query

    query_embedding = embedder.encode(
        [search_query]
    ).tolist()

    collection_count = collection.count()

    if collection_count == 0:
        return []

    if is_summary:

        n = min(
            collection_count,
            8
        )

    else:

        n = min(
            k,
            collection_count
        )

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n
    )

    docs = results["documents"][0]

    docs = [
        str(d)
        for d in docs
        if d and str(d).strip()
    ]

    return docs


# -----------------------------
# LLM Call
# -----------------------------

def ask_llm(api_key, model, question, context):

    is_summary = any(
        word in question.lower()
        for word in [
            "summarize",
            "summary",
            "overview",
            "what is this about",
            "what does this say",
            "explain the document",
            "brief"
        ]
    )

    if is_summary:

        instruction = (
            "You are a helpful assistant. "
            "Give a clear, detailed summary "
            "of the context below. "
            "Cover the main topics, key points, "
            "and important details."
        )

    else:

        instruction = (
            "You are a helpful assistant. "
            "Answer the question using ONLY "
            "the context below. "
            "If the answer is not in the context, "
            "say 'I could not find that in the document.'"
        )

    prompt = f"""{instruction}

Context:
{context}

Question:
{question}

Answer:"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Simple Local RAG"
    }

    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:

        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=data,
            timeout=60
        )

        result = response.json()

        if "choices" not in result:

            return (
                f"❌ API Error: "
                f"{result.get('error', result)}"
            )

        return result["choices"][0]["message"]["content"]

    except requests.exceptions.Timeout:

        return (
            "❌ Request timed out. "
            "Please try again."
        )

    except Exception as e:

        return f"❌ Error: {str(e)}"


# -----------------------------
# App UI
# -----------------------------

def main():

    st.set_page_config(
        page_title="Simple Local RAG",
        layout="wide"
    )

    st.title(
        "🤖 Simple Local RAG Chat"
    )

    st.caption(
        "Powered by Streamlit + ChromaDB + OpenRouter"
    )

    # Load models and database
    embedder = load_embedder()

    collection = get_collection()

    # -----------------------------
    # Sidebar - LLM Settings
    # -----------------------------

    st.sidebar.subheader(
        "⚙️ LLM Settings"
    )

    api_key = st.sidebar.text_input(
        "OpenRouter API Key",
        type="password"
    )

    model_name = st.sidebar.text_input(
        "Model",
        "stepfun/step-3.5-flash:free"
    )

    top_k = st.sidebar.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=8,
        value=4
    )

    # -----------------------------
    # Upload PDF
    # -----------------------------

    st.sidebar.divider()

    st.sidebar.subheader(
        "📄 Ingest Documents"
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if st.sidebar.button(
        "📥 Index Uploaded File"
    ):

        if uploaded_file:

            text = read_pdf(
                uploaded_file
            )

            if not text.strip():

                st.sidebar.error(
                    "Could not extract text from PDF. "
                    "It may be scanned/image-based."
                )

            else:

                collection = get_collection()

                with st.spinner(
                    "Embedding and storing chunks..."
                ):

                    count = store_docs(
                        collection,
                        embedder,
                        text,
                        uploaded_file.name
                    )

                if count > 0:

                    st.sidebar.success(
                        f"✅ Indexed {count} chunks"
                    )

                    st.sidebar.info(
                        f"📦 Total chunks in DB: "
                        f"{collection.count()}"
                    )

                else:

                    st.sidebar.error(
                        "❌ No chunks were indexed."
                    )

        else:

            st.sidebar.warning(
                "Upload a PDF first."
            )

    # -----------------------------
    # Database Status
    # -----------------------------

    st.sidebar.divider()

    try:

        total_chunks = collection.count()

    except Exception:

        collection = get_collection()

        try:
            total_chunks = collection.count()
        except Exception:
            total_chunks = 0

    if total_chunks > 0:

        st.sidebar.success(
            f"📦 DB has {total_chunks} chunks ready"
        )

    else:

        st.sidebar.warning(
            "⚠️ DB is empty — "
            "please index a document"
        )

    # -----------------------------
    # Reset Vector DB
    # -----------------------------

    st.sidebar.divider()

    if st.sidebar.button(
        "🗑️ Reset Vector DB"
    ):

        client = get_chroma_client()

        try:

            client.delete_collection(
                COLLECTION_NAME
            )

        except Exception:

            pass

        client.get_or_create_collection(
            name=COLLECTION_NAME
        )

        st.sidebar.success(
            "✅ Vector database reset."
        )

        st.rerun()

    # -----------------------------
    # Chat State
    # -----------------------------

    if "messages" not in st.session_state:

        st.session_state.messages = []

    # -----------------------------
    # Clear Chat
    # -----------------------------

    if st.sidebar.button(
        "🧹 Clear Chat"
    ):

        st.session_state.messages = []

        st.rerun()

    # -----------------------------
    # Chat Interface
    # -----------------------------

    if total_chunks == 0:

        st.info(
            "👆 Upload a PDF and click "
            "**Index Uploaded File** "
            "to get started."
        )

    # Display previous messages

    for msg in st.session_state.messages:

        with st.chat_message(
            msg["role"]
        ):

            st.write(
                msg["content"]
            )

    # Chat input

    question = st.chat_input(
        "Ask about your docs..."
    )

    if question:

        # Check API key

        if not api_key:

            st.warning(
                "⚠️ Add your OpenRouter API key "
                "in the sidebar."
            )

            st.stop()

        # Check documents

        if total_chunks == 0:

            st.warning(
                "⚠️ No documents indexed yet. "
                "Please upload and index a PDF first."
            )

            st.stop()

        # Add user message

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.write(question)

        # Retrieve relevant chunks

        with st.spinner(
            "🔍 Retrieving context..."
        ):

            collection = get_collection()

            docs = retrieve(
                collection,
                embedder,
                question,
                k=top_k
            )

        if not docs:

            answer = (
                "I could not find any relevant "
                "information in the document."
            )

        else:

            context = "\n\n".join(docs)

            # Generate answer

            with st.chat_message("assistant"):

                with st.spinner(
                    "💬 Generating answer..."
                ):

                    answer = ask_llm(
                        api_key,
                        model_name,
                        question,
                        context
                    )

                st.write(answer)

        # Save assistant response

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


# -----------------------------
# Run App
# -----------------------------

if __name__ == "__main__":

    os.makedirs(
        DB_DIR,
        exist_ok=True
    )

    main()
