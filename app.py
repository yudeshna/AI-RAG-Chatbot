import warnings
warnings.filterwarnings("ignore")

import os
import re
import json
import unicodedata

import streamlit as st
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import requests


# ============================================================
# SETTINGS
# ============================================================

DB_DIR = "/tmp/chroma_db"
COLLECTION_NAME = "rag_docs"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedder():
    return SentenceTransformer(MODEL_NAME)


# ============================================================
# CHROMA CLIENT
# ============================================================

@st.cache_resource
def get_chroma_client():
    """
    Create exactly one ChromaDB client and reuse it.
    This prevents the 'different settings' error.
    """
    os.makedirs(DB_DIR, exist_ok=True)

    return chromadb.PersistentClient(
        path=DB_DIR,
        settings=chromadb.config.Settings(
            anonymized_telemetry=False
        )
    )


# ============================================================
# GET COLLECTION
# ============================================================

def get_collection():
    client = get_chroma_client()

    return client.get_or_create_collection(
        name=COLLECTION_NAME
    )


# ============================================================
# READ PDF
# ============================================================

def read_pdf(file):

    reader = PdfReader(file)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        content = page.extract_text()

        if content:

            pages.append({
                "page": page_number,
                "text": content
            })

    return pages


# ============================================================
# SPLIT TEXT
# ============================================================

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


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.replace("\x00", "")
    text = text.replace("\ufffd", "")

    text = unicodedata.normalize("NFKD", text)

    text = text.encode(
        "ascii",
        errors="ignore"
    ).decode("ascii")

    return text.strip()


# ============================================================
# STORE DOCUMENTS
# ============================================================

def store_docs(collection, embedder, pages, filename):

    all_chunks = []
    all_metadata = []

    for page in pages:

        page_number = page["page"]
        page_text = page["text"]

        chunks = split_text(page_text)

        for chunk in chunks:

            cleaned = clean_text(chunk)

            if cleaned:

                all_chunks.append(cleaned)

                all_metadata.append({
                    "source": filename,
                    "page": str(page_number)
                })

    if not all_chunks:
        return 0

    st.sidebar.info(
        f"🔍 Creating embeddings for {len(all_chunks)} chunks..."
    )

    embeddings = []

    for i, chunk in enumerate(all_chunks):

        try:

            embedding = embedder.encode(
                [chunk]
            ).tolist()[0]

            embeddings.append(embedding)

        except Exception as e:

            st.error(
                f"❌ Embedding failed for chunk {i}: {e}"
            )

            return 0

    ids = [
        f"{filename}_{i}"
        for i in range(len(all_chunks))
    ]

    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=all_metadata
    )

    return len(all_chunks)


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve(collection, embedder, query, k=4):

    if collection.count() == 0:
        return [], []

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
            "main topics key points overview "
            "introduction important concepts"
        )

        n_results = min(
            collection.count(),
            8
        )

    else:

        search_query = query

        n_results = min(
            collection.count(),
            k
        )

    query_embedding = embedder.encode(
        [search_query]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    clean_documents = []
    sources = []

    for doc, metadata in zip(
        documents,
        metadatas
    ):

        if doc and str(doc).strip():

            clean_documents.append(
                str(doc)
            )

            sources.append({
                "source": metadata.get(
                    "source",
                    "Unknown"
                ),
                "page": metadata.get(
                    "page",
                    "Unknown"
                )
            })

    return clean_documents, sources


# ============================================================
# FORMAT CONTEXT
# ============================================================

def format_context(documents, sources):

    parts = []

    for i, (doc, source) in enumerate(
        zip(documents, sources),
        start=1
    ):

        parts.append(
            f"""
SOURCE {i}
File: {source['source']}
Page: {source['page']}

{doc}
"""
        )

    return "\n".join(parts)


# ============================================================
# SHOW SOURCES
# ============================================================

def show_sources(sources):

    if not sources:
        return

    with st.expander(
        "📚 Retrieved Sources",
        expanded=False
    ):

        seen = set()

        for source in sources:

            key = (
                source["source"],
                source["page"]
            )

            if key in seen:
                continue

            seen.add(key)

            st.write(
                f"📄 **{source['source']}** — "
                f"Page {source['page']}"
            )


# ============================================================
# CALL OPENROUTER
# ============================================================

def call_llm(
    api_key,
    model,
    prompt
):

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.io",
        "X-Title": "AI Study Assistant"
    }

    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2
    }

    try:

        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=data,
            timeout=90
        )

        result = response.json()

        if "choices" not in result:

            error = result.get(
                "error",
                result
            )

            return None, f"API Error: {error}"

        answer = result[
            "choices"
        ][0][
            "message"
        ][
            "content"
        ]

        return answer, None

    except requests.exceptions.Timeout:

        return None, (
            "Request timed out. "
            "Please try again."
        )

    except Exception as e:

        return None, str(e)


# ============================================================
# ANSWER PDF QUESTION
# ============================================================

def answer_question(
    api_key,
    model,
    question,
    context
):

    prompt = f"""
You are an AI study assistant.

Answer the user's question using ONLY the
provided document context.

Rules:

1. Do not invent information.
2. Use the terminology from the document.
3. Give a clear and student-friendly explanation.
4. If the question requires steps, provide numbered steps.
5. If the question requires comparison, use a table when useful.
6. If the question requires an example, provide an example only
   when supported by the context.
7. If the answer cannot be found in the context, say:

"I could not find that in the uploaded document."

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    return call_llm(
        api_key,
        model,
        prompt
    )


# ============================================================
# GENERATE QUIZ
# ============================================================

def generate_quiz(
    api_key,
    model,
    context,
    difficulty,
    number
):

    prompt = f"""
You are an expert university exam question generator.

Create exactly {number} questions based ONLY on
the document context below.

Difficulty: {difficulty}

Requirements:

- Questions must be directly related to the document.
- Do not invent topics that are not present.
- For Easy questions, test definitions and basic concepts.
- For Medium questions, test explanations, comparisons,
  processes and applications.
- For Hard questions, test analysis, derivations,
  problem-solving or deeper understanding.
- Questions should be suitable for university examinations.
- Do not provide answers.
- Number the questions from 1 to {number}.

DOCUMENT CONTEXT:

{context}

Generate the questions now.
"""

    answer, error = call_llm(
        api_key,
        model,
        prompt
    )

    if error:
        return None, error

    return answer, None


# ============================================================
# PARSE QUESTIONS
# ============================================================

def parse_questions(text):

    if not text:
        return []

    lines = text.splitlines()

    questions = []

    current = ""

    for line in lines:

        line = line.strip()

        if not line:
            continue

        match = re.match(
            r"^(?:Q(?:uestion)?\s*)?(\d+)[\.\):\-]\s*(.*)",
            line,
            re.IGNORECASE
        )

        if match:

            if current:
                questions.append(
                    current.strip()
                )

            current = match.group(2).strip()

        else:

            if current:
                current += " " + line

    if current:
        questions.append(
            current.strip()
        )

    return questions


# ============================================================
# EVALUATE STUDENT ANSWER
# ============================================================

def evaluate_answer(
    api_key,
    model,
    question,
    student_answer,
    context
):

    prompt = f"""
You are a strict but helpful university examiner.

Evaluate the student's answer using ONLY the
provided document context.

QUESTION:
{question}

STUDENT ANSWER:
{student_answer}

DOCUMENT CONTEXT:
{context}

Evaluate the answer.

Return your evaluation using exactly this structure:

SCORE: X/10

VERDICT:
Correct / Mostly Correct / Partially Correct / Incorrect

WHAT YOU DID WELL:
- point 1
- point 2

WHAT YOU MISSED:
- point 1
- point 2

IMPROVEMENT:
Explain exactly how the student can improve.

MODEL ANSWER:
Give a clear exam-ready answer based ONLY on the document.

IMPORTANT:
- Do not give marks for information not supported by the document.
- Do not invent facts.
- Be fair.
- A partially correct answer should receive partial marks.
"""

    return call_llm(
        api_key,
        model,
        prompt
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    st.title("🤖 AI Study Assistant")

    st.caption(
        "Upload your study PDF, ask questions, "
        "generate quizzes, and improve your answers."
    )

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    embedder = load_embedder()

    collection = get_collection()

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    st.sidebar.header("⚙️ LLM Settings")

    api_key = st.sidebar.text_input(
        "OpenRouter API Key",
        type="password"
    )

    model_name = st.sidebar.text_input(
        "Model",
        value="stepfun/step-3.5-flash"
    )

    top_k = st.sidebar.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=8,
        value=4
    )

    st.sidebar.divider()

    # --------------------------------------------------------
    # PDF UPLOAD
    # --------------------------------------------------------

    st.sidebar.header("📄 Ingest Documents")

    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if st.sidebar.button(
        "📥 Index Uploaded File",
        use_container_width=True
    ):

        if uploaded_file:

            try:

                pages = read_pdf(
                    uploaded_file
                )

                if not pages:

                    st.sidebar.error(
                        "Could not extract text from PDF. "
                        "It may be scanned/image-based."
                    )

                else:

                    with st.spinner(
                        "Embedding and storing document..."
                    ):

                        count = store_docs(
                            collection,
                            embedder,
                            pages,
                            uploaded_file.name
                        )

                    if count > 0:

                        st.sidebar.success(
                            f"✅ Indexed {count} chunks"
                        )

                        st.sidebar.info(
                            f"📦 Total chunks: "
                            f"{collection.count()}"
                        )

                    else:

                        st.sidebar.error(
                            "No usable text was found."
                        )

            except Exception as e:

                st.sidebar.error(
                    f"❌ Error: {e}"
                )

        else:

            st.sidebar.warning(
                "Upload a PDF first."
            )

    # --------------------------------------------------------
    # DATABASE STATUS
    # --------------------------------------------------------

    st.sidebar.divider()

    try:

        total_chunks = collection.count()

    except Exception:

        total_chunks = 0

    if total_chunks > 0:

        st.sidebar.success(
            f"📦 Database: {total_chunks} chunks"
        )

    else:

        st.sidebar.warning(
            "⚠️ Database is empty"
        )

    # --------------------------------------------------------
    # RESET DATABASE
    # --------------------------------------------------------

    st.sidebar.divider()

    if st.sidebar.button(
        "🗑️ Reset Vector DB",
        use_container_width=True
    ):

        try:

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

        except Exception as e:

            st.sidebar.error(
                f"Reset failed: {e}"
            )

    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

    if st.sidebar.button(
        "🧹 Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    # --------------------------------------------------------
    # SESSION STATE
    # --------------------------------------------------------

    if "messages" not in st.session_state:

        st.session_state.messages = []

    if "quiz_questions" not in st.session_state:

        st.session_state.quiz_questions = []

    if "quiz_answers" not in st.session_state:

        st.session_state.quiz_answers = {}

    if "quiz_results" not in st.session_state:

        st.session_state.quiz_results = {}

    if "quiz_started" not in st.session_state:

        st.session_state.quiz_started = False

    # ========================================================
    # TABS
    # ========================================================

    chat_tab, quiz_tab = st.tabs([
        "💬 Chat with PDF",
        "🧠 Quiz Mode"
    ])

    # ========================================================
    # CHAT TAB
    # ========================================================

    with chat_tab:

        if total_chunks == 0:

            st.info(
                "👆 Upload a PDF from the sidebar "
                "and click **Index Uploaded File** "
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

                if (
                    msg["role"] == "assistant"
                    and msg.get("sources")
                ):

                    show_sources(
                        msg["sources"]
                    )

        question = st.chat_input(
            "Ask about your uploaded documents..."
        )

        if question:

            if not api_key:

                st.warning(
                    "⚠️ Add your OpenRouter API key "
                    "in the sidebar."
                )

                st.stop()

            if total_chunks == 0:

                st.warning(
                    "⚠️ Please upload and index "
                    "a PDF first."
                )

                st.stop()

            # USER MESSAGE

            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            with st.chat_message("user"):

                st.write(question)

            # RETRIEVE

            with st.spinner(
                "🔍 Searching your document..."
            ):

                collection = get_collection()

                documents, sources = retrieve(
                    collection,
                    embedder,
                    question,
                    k=top_k
                )

            context = format_context(
                documents,
                sources
            )

            # ANSWER

            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "💬 Generating answer..."
                ):

                    answer, error = answer_question(
                        api_key,
                        model_name,
                        question,
                        context
                    )

                if error:

                    st.error(
                        f"❌ {error}"
                    )

                    answer = (
                        f"❌ {error}"
                    )

                else:

                    st.write(answer)

                    show_sources(
                        sources
                    )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

    # ========================================================
    # QUIZ TAB
    # ========================================================

    with quiz_tab:

        st.header("🧠 Quiz Mode")

        st.write(
            "Generate exam-style questions from "
            "your uploaded PDF and test yourself."
        )

        if total_chunks == 0:

            st.info(
                "📄 Upload and index a PDF first "
                "to use Quiz Mode."
            )

        else:

            # ------------------------------------------------
            # QUIZ SETTINGS
            # ------------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                difficulty = st.selectbox(
                    "Difficulty",
                    [
                        "Easy",
                        "Medium",
                        "Hard"
                    ]
                )

            with col2:

                number = st.selectbox(
                    "Number of questions",
                    [3, 5, 10]
                )

            st.divider()

            # ------------------------------------------------
            # GENERATE QUIZ BUTTON
            # ------------------------------------------------

            if st.button(
                "✨ Generate Quiz",
                type="primary",
                use_container_width=True
            ):

                if not api_key:

                    st.warning(
                        "⚠️ Add your OpenRouter API key "
                        "in the sidebar."
                    )

                else:

                    with st.spinner(
                        "🧠 Creating questions from your PDF..."
                    ):

                        # Retrieve broad document context

                        documents, sources = retrieve(
                            collection,
                            embedder,
                            "main topics concepts definitions "
                            "important principles algorithms "
                            "applications examples",
                            k=8
                        )

                        context = format_context(
                            documents,
                            sources
                        )

                        quiz_text, error = generate_quiz(
                            api_key,
                            model_name,
                            context,
                            difficulty,
                            number
                        )

                    if error:

                        st.error(
                            f"❌ {error}"
                        )

                    else:

                        questions = parse_questions(
                            quiz_text
                        )

                        if questions:

                            st.session_state.quiz_questions = (
                                questions[:number]
                            )

                            st.session_state.quiz_answers = {}

                            st.session_state.quiz_results = {}

                            st.session_state.quiz_started = True

                            st.rerun()

                        else:

                            st.error(
                                "Could not generate questions. "
                                "Please try again."
                            )

            # ------------------------------------------------
            # DISPLAY QUIZ
            # ------------------------------------------------

            if st.session_state.quiz_started:

                questions = (
                    st.session_state.quiz_questions
                )

                st.subheader(
                    f"📝 {difficulty} Quiz"
                )

                st.caption(
                    f"{len(questions)} questions"
                )

                for i, question in enumerate(
                    questions
                ):

                    st.markdown(
                        f"### Question {i + 1}"
                    )

                    st.write(question)

                    answer_key = (
                        f"answer_{i}"
                    )

                    st.session_state.quiz_answers[
                        answer_key
                    ] = st.text_area(
                        "Your answer:",
                        value=st.session_state.quiz_answers.get(
                            answer_key,
                            ""
                        ),
                        key=f"text_{i}",
                        height=150
                    )

                    # Existing result

                    result_key = (
                        f"result_{i}"
                    )

                    if result_key in (
                        st.session_state.quiz_results
                    ):

                        result = (
                            st.session_state.quiz_results[
                                result_key
                            ]
                        )

                        st.markdown(
                            "---"
                        )

                        st.markdown(
                            "### 🤖 Evaluation"
                        )

                        st.write(result)

                    else:

                        if st.button(
                            f"🤖 Evaluate Answer {i + 1}",
                            key=f"evaluate_{i}"
                        ):

                            student_answer = (
                                st.session_state.quiz_answers[
                                    answer_key
                                ]
                            )

                            if not student_answer.strip():

                                st.warning(
                                    "Please write an answer first."
                                )

                            elif not api_key:

                                st.warning(
                                    "Add your OpenRouter API key."
                                )

                            else:

                                with st.spinner(
                                    "Checking your answer..."
                                ):

                                    documents, sources = retrieve(
                                        collection,
                                        embedder,
                                        question,
                                        k=top_k
                                    )

                                    context = format_context(
                                        documents,
                                        sources
                                    )

                                    result, error = evaluate_answer(
                                        api_key,
                                        model_name,
                                        question,
                                        student_answer,
                                        context
                                    )

                                if error:

                                    st.error(
                                        f"❌ {error}"
                                    )

                                else:

                                    st.session_state.quiz_results[
                                        result_key
                                    ] = result

                                    st.rerun()

                    st.divider()

                # ------------------------------------------------
                # FINAL SCORE
                # ------------------------------------------------

                evaluated = (
                    len(
                        st.session_state.quiz_results
                    )
                )

                if evaluated == len(
                    questions
                ):

                    st.success(
                        "🎉 You have completed the quiz!"
                    )

                    scores = []

                    for result in (
                        st.session_state.quiz_results.values()
                    ):

                        match = re.search(
                            r"SCORE:\s*(\d+)\s*/\s*10",
                            result,
                            re.IGNORECASE
                        )

                        if match:

                            scores.append(
                                int(match.group(1))
                            )

                    if scores:

                        total_score = sum(
                            scores
                        )

                        max_score = (
                            len(scores) * 10
                        )

                        percentage = (
                            total_score /
                            max_score
                        ) * 100

                        st.metric(
                            "Your Score",
                            f"{total_score}/{max_score}"
                        )

                        st.progress(
                            percentage / 100
                        )

                        st.write(
                            f"📊 **Percentage: "
                            f"{percentage:.0f}%**"
                        )

                        if percentage >= 80:

                            st.success(
                                "🔥 Excellent! "
                                "You are well prepared."
                            )

                        elif percentage >= 60:

                            st.info(
                                "👍 Good job! "
                                "Review the topics you missed."
                            )

                        else:

                            st.warning(
                                "📚 Keep practicing. "
                                "Review the PDF and try again."
                            )


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":

    os.makedirs(
        DB_DIR,
        exist_ok=True
    )

    main()
