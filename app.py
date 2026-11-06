import os
import uuid
import requests
import streamlit as st
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# -------------------- SETTINGS --------------------
APP_TITLE = "StudySphere AI"
DB_DIR = "chroma_db"
COLLECTION_NAME = "studysphere_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"

st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")

# Cache heavy objects so they are not reloaded on every interaction.
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def load_vector_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

def extract_pdf_pages(uploaded_file):
    """Return one text record per PDF page, preserving page numbers."""
    reader = PdfReader(uploaded_file)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({
                "text": text.strip(),
                "page": page_number,
                "source": uploaded_file.name
            })
    return pages

def split_into_chunks(text, chunk_size=900, overlap=150):
    """Split text into overlapping character chunks."""
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

def index_uploaded_pdfs(files):
    model = load_embedding_model()
    collection = load_vector_collection()
    added = 0
    skipped = []

    for uploaded_file in files:
        try:
            pages = extract_pdf_pages(uploaded_file)
            if not pages:
                skipped.append(f"{uploaded_file.name}: no extractable text")
                continue

            # Replace earlier indexed content for this filename.
            try:
                collection.delete(where={"source": uploaded_file.name})
            except Exception:
                pass

            docs, ids, metadatas = [], [], []
            for page_record in pages:
                for chunk in split_into_chunks(page_record["text"]):
                    docs.append(chunk)
                    ids.append(str(uuid.uuid4()))
                    metadatas.append({
                        "source": page_record["source"],
                        "page": int(page_record["page"])
                    })

            if docs:
                vectors = model.encode(docs, normalize_embeddings=True).tolist()
                collection.add(
                    ids=ids,
                    documents=docs,
                    metadatas=metadatas,
                    embeddings=vectors
                )
                added += len(docs)
        except Exception as exc:
            skipped.append(f"{uploaded_file.name}: {exc}")

    return added, skipped

def retrieve_context(question, top_k=5):
    collection = load_vector_collection()
    model = load_embedding_model()
    if collection.count() == 0:
        return [], []

    query_vector = model.encode([question], normalize_embeddings=True).tolist()[0]
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    context_parts = []
    sources = []

    for document, metadata in zip(documents, metadatas):
        source = metadata.get("source", "Unknown file")
        page = metadata.get("page", "?")
        context_parts.append(f"[Source: {source}, page {page}]\n{document}")
        sources.append({"source": source, "page": page, "excerpt": document[:240]})

    return context_parts, sources

def ask_ollama(question, context, mode="answer"):
    if mode == "eli5":
        instruction = (
            "Explain the answer in very simple English, as if teaching a beginner. "
            "Use a small everyday example if useful."
        )
    elif mode == "mcq":
        instruction = (
            "Create 5 multiple-choice questions using only the supplied context. "
            "Give four options (A-D), mark the correct answer, and briefly explain it."
        )
    elif mode == "short":
        instruction = (
            "Create 5 useful short-answer exam questions from the supplied context. "
            "Provide a concise model answer for each."
        )
    elif mode == "summary":
        instruction = (
            "Write a clear, structured summary of the supplied context. "
            "Use headings and bullet points where useful."
        )
    else:
        instruction = (
            "Answer the user's question accurately using only the supplied context. "
            "If the context does not contain the answer, say: "
            "'I couldn't find this information in the uploaded documents.'"
        )

    prompt = f"""
You are StudySphere AI, a document-grounded study assistant.
Rules:
- Use only the context below. Do not invent facts.
- If the context is insufficient, clearly say so.
- When making factual claims, cite the source filename and page number shown in the context.
- Be clear and student-friendly.

TASK:
{instruction}

DOCUMENT CONTEXT:
{chr(10).join(context)}

USER REQUEST:
{question}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        },
        timeout=180
    )
    response.raise_for_status()
    return response.json()["message"]["content"]

# -------------------- USER INTERFACE --------------------
st.title("📚 StudySphere AI")
st.caption("Your personal RAG-powered study assistant. Ask questions from your own PDFs.")

with st.sidebar:
    st.header("1. Add your study PDFs")
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )
    if st.button("Process PDFs", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        else:
            with st.spinner("Reading PDFs, creating embeddings and saving them..."):
                count, skipped = index_uploaded_pdfs(uploaded_files)
            if count:
                st.success(f"Indexed {count} text chunks.")
            for message in skipped:
                st.warning(message)

    st.divider()
    st.header("2. Knowledge base")
    try:
        current_count = load_vector_collection().count()
        st.metric("Stored text chunks", current_count)
    except Exception:
        st.info("The knowledge base will be created when you process a PDF.")
    st.caption("First use downloads the embedding model. Keep internet connected during that download.")

tab_chat, tab_quiz, tab_summary = st.tabs(
    ["💬 Ask your PDFs", "📝 Generate questions", "📖 Summarise"]
)

with tab_chat:
    st.subheader("Ask a question")
    mode = st.radio(
        "Answer style",
        ["Normal", "ELI5 (simple English)"],
        horizontal=True
    )
    question = st.text_input(
        "What would you like to know?",
        placeholder="Example: Explain the main idea of Chapter 2."
    )
    if st.button("Get answer", key="answer_button"):
        if not question.strip():
            st.warning("Type a question first.")
        else:
            with st.spinner("Searching your PDFs and asking the local AI model..."):
                try:
                    context, sources = retrieve_context(question)
                    if not context:
                        st.warning("No indexed text found. Upload and process a PDF first.")
                    else:
                        answer_mode = "eli5" if mode.startswith("ELI5") else "answer"
                        answer = ask_ollama(question, context, answer_mode)
                        st.markdown("### Answer")
                        st.write(answer)
                        st.markdown("### Sources retrieved")
                        for item in sources:
                            st.markdown(
                                f"- **{item['source']}**, page **{item['page']}** — "
                                f"{item['excerpt']}..."
                            )
                except requests.exceptions.ConnectionError:
                    st.error("Ollama is not running. Open Ollama and make sure it is running.")
                except Exception as exc:
                    st.error(f"Something went wrong: {exc}")

with tab_quiz:
    st.subheader("Create practice questions")
    quiz_type = st.selectbox(
        "Choose question type",
        ["Multiple-choice questions (MCQs)", "Short-answer exam questions"]
    )
    topic = st.text_input(
        "Optional topic (leave blank for the whole indexed collection)",
        placeholder="Example: neural networks"
    )
    if st.button("Generate questions"):
        try:
            query = topic.strip() or "important concepts, definitions and key ideas"
            context, sources = retrieve_context(query, top_k=8)
            if not context:
                st.warning("Upload and process PDFs first.")
            else:
                mode = "mcq" if quiz_type.startswith("Multiple") else "short"
                with st.spinner("Creating questions from your documents..."):
                    output = ask_ollama(query, context, mode)
                st.markdown(output)
                st.caption(
                    "Questions are generated from retrieved document sections; "
                    "check the cited pages against your notes."
                )
                st.markdown("**Retrieved pages:** " + ", ".join(
                    sorted({f"{s['source']} p.{s['page']}" for s in sources})
                ))
        except requests.exceptions.ConnectionError:
            st.error("Ollama is not running. Open Ollama and try again.")
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")

with tab_summary:
    st.subheader("Summarise your study material")
    summary_topic = st.text_input(
        "Optional topic or chapter name",
        placeholder="Example: Chapter 1 introduction"
    )
    if st.button("Create summary"):
        try:
            query = summary_topic.strip() or "main topics and important ideas"
            context, sources = retrieve_context(query, top_k=8)
            if not context:
                st.warning("Upload and process PDFs first.")
            else:
                with st.spinner("Preparing your summary..."):
                    summary = ask_ollama(query, context, "summary")
                st.markdown(summary)
                st.markdown("**Retrieved pages:** " + ", ".join(
                    sorted({f"{s['source']} p.{s['page']}" for s in sources})
                ))
        except requests.exceptions.ConnectionError:
            st.error("Ollama is not running. Open Ollama and try again.")
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")

st.divider()
st.caption(
    "StudySphere AI uses retrieval-augmented generation. "
    "Answers can still contain mistakes, so verify important exam information in your notes."
)
