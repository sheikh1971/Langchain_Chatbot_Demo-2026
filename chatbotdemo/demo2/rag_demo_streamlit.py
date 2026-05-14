import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
import os
from openai import OpenAI

# Set NVIDIA API key directly (do not display in UI)
NVIDIA_API_KEY = "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"

# --- NVIDIA Embeddings using OpenAI client with NVIDIA endpoint ---
client = OpenAI(
    api_key=NVIDIA_API_KEY,
    base_url="https://integrate.api.nvidia.com/v1"
)

def get_nvidia_embeddings(texts, input_type="document"):
    response = client.embeddings.create(
        input=texts,
        model="nvidia/nv-embed-v1",
        encoding_format="float",
        extra_body={"input_type": input_type, "truncate": "NONE"}
    )
    return [d.embedding for d in response.data]

class NvidiaEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return get_nvidia_embeddings(texts, input_type="document")
    def embed_query(self, text):
        return get_nvidia_embeddings([text], input_type="query")[0]

# --- Streamlit UI ---
st.title("RAG Demo with NVIDIA Embeddings & Chroma")

query = st.text_input("Enter your question:")

# Load Chroma vector store (assumes persistence in ./chroma_db)
if os.path.exists("chroma_db"):  # Adjust path if needed
    db = Chroma(persist_directory="chroma_db", embedding_function=NvidiaEmbeddings())
else:
    st.warning("Chroma vector store not found. Please run the notebook to create it first.")
    db = None

if query and db:
    with st.spinner("Searching..."):
        results = db.similarity_search(query, k=3)
    st.subheader("Top Results:")
    for i, doc in enumerate(results, 1):
        st.markdown(f"**Result {i}:**")
        st.write(doc.page_content)
        if doc.metadata:
            st.caption(str(doc.metadata))
