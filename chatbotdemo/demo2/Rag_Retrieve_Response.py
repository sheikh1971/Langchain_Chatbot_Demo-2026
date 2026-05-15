import os
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load variables from .env into process env.
# Normal: lets the app read your API keys without hardcoding them.
load_dotenv()


# Project paths used by loaders and vector DB persistence.
# Technical: these are resolved from the script location for portability.
BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR.parent / "public"
TXT_PATH = PUBLIC_DIR / "demo_file.txt"
PDF_PATH = PUBLIC_DIR / "AI-Based_Mental_Health_Chatbot_using_LangChain_and.pdf"
CHROMA_DIR = BASE_DIR / "chroma_db"


def read_env(name: str, required: bool = True) -> Optional[str]:
    # Small helper to validate env configuration early.
    value = os.getenv(name)
    if required and not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def extract_text_response(content) -> str:
    # Gemini/LangChain can return either a plain string or structured parts.
    # This normalizes both formats into one clean text answer for UI display.
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part).strip()

    return str(content)


@st.cache_resource(show_spinner=False)
def build_pipeline() -> Tuple:
    # Cached pipeline = faster repeated questions in Streamlit.
    # Technical: cache key is tied to function code + input state.
    google_api_key = read_env("GOOGLE_API_KEY", required=True)
    nvidia_embed_key = read_env("NVIDIA_API_KEY", required=True)
    nvidia_rerank_key = read_env("NVIDIA_RETRIEVER", required=False)

    docs: List[Document] = []

    # Load source documents from public folder.
    if TXT_PATH.exists():
        docs.extend(TextLoader(str(TXT_PATH), encoding="utf-8").load())

    if PDF_PATH.exists():
        docs.extend(PyPDFLoader(str(PDF_PATH)).load())

    if not docs:
        raise ValueError("No source documents found in public folder.")

    # Chunking improves retrieval granularity and keeps context window efficient.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # NVIDIA embeddings convert text into vectors for semantic similarity search.
    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embed-v1",
        api_key=nvidia_embed_key,
        truncate="NONE",
    )

    # Chroma stores vectors locally in chroma_db for retrieval.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # Retrieve top-k semantically similar chunks.
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Optional reranker: improves ordering quality after retrieval.
    reranker = None
    if nvidia_rerank_key:
        reranker = NVIDIARerank(
            model="nv-rerank-qa-mistral-4b:1",
            api_key=nvidia_rerank_key,
        )

    # Gemini is used only for answer generation (not embeddings/reranking).
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=google_api_key,
        temperature=0.3,
    )

    # Prompt includes explicit context grounding to reduce hallucinations.
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use only the context to answer. "
                "If the context does not contain the answer, say that clearly.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    return retriever, reranker, llm, prompt_template


def maybe_rerank(
    question: str, docs: List[Document], reranker: Optional[NVIDIARerank]
) -> List[Document]:
    # No docs = nothing to rerank.
    if not docs:
        return []

    # Graceful fallback when rerank API key is not configured.
    if reranker is None:
        return docs

    # Copy docs to keep original retrieval output untouched.
    base_docs = [Document(page_content=d.page_content, metadata=d.metadata) for d in docs]
    reranked = reranker.compress_documents(query=question, documents=base_docs)

    # Keep metadata if reranker strips it.
    for i, doc in enumerate(reranked):
        if not doc.metadata and i < len(base_docs):
            doc.metadata = base_docs[i].metadata

    return reranked


def ask_rag(question: str) -> Tuple[str, str, List[Document]]:
    # End-to-end RAG flow: retrieve -> rerank -> build context -> generate answer.
    retriever, reranker, llm, prompt_template = build_pipeline()

    retrieved_docs = retriever.invoke(question)
    ranked_docs = maybe_rerank(question, retrieved_docs, reranker)

    # Keep top documents to control prompt size and latency.
    top_docs = ranked_docs[:3]
    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in top_docs
        ]
    )

    if not context.strip():
        return "No relevant context was found.", "", top_docs

    prompt_input = {"question": question, "context": context}
    response = llm.invoke(prompt_template.format_prompt(**prompt_input).to_string())
    answer = extract_text_response(response.content)

    return answer, context, top_docs


def main() -> None:
    # Streamlit UI layer.
    st.set_page_config(page_title="RAG Retrieve Response", page_icon="📚", layout="wide")
    st.title("RAG Retrieve Response")
    st.caption("NVIDIA Embeddings + NVIDIA Rerank + Gemini LLM")

    with st.sidebar:
        # Quick diagnostics for beginners.
        st.subheader("Status")
        st.write(f"Text file: {'Found' if TXT_PATH.exists() else 'Missing'}")
        st.write(f"PDF file: {'Found' if PDF_PATH.exists() else 'Missing'}")
        st.write(f"Chroma DB folder: {CHROMA_DIR}")
        if st.button("Rebuild Index"):
            # Clears cached pipeline so embeddings/vector index can rebuild.
            build_pipeline.clear()
            st.success("Index cache cleared. It will rebuild on the next question.")

    question = st.text_input(
        "Ask a question from your PDF/text documents:",
        value="What is the main objective of this paper?",
    )

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        try:
            # Single action path for user query execution.
            with st.spinner("Retrieving, reranking, and generating answer..."):
                answer, context, docs = ask_rag(question)

            st.subheader("Answer")
            st.write(answer)

            with st.expander("Context sent to LLM"):
                st.text(context if context else "No context available")

            with st.expander("Top retrieved documents"):
                if not docs:
                    st.write("No documents returned.")
                for idx, doc in enumerate(docs, start=1):
                    source = doc.metadata.get("source", "Unknown")
                    st.markdown(f"**Doc {idx} Source:** {source}")
                    st.write(doc.page_content[:1200])
                    st.divider()

        except Exception as exc:
            st.error(f"Error: {exc}")


if __name__ == "__main__":
    main()
