import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import streamlit as st
from sentence_transformers import CrossEncoder

import preprocesser
import retriever
import generator
embedding_model="BAAI/bge-small-en-v1.5"
chunk_size = 1500
chunk_overlap = 300
candidate_k = 50
top_k = 5
UPLOAD_DIR = "uploaded_pdfs"
CHUNKS_DIR = "chunks_cache"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

st.set_page_config(page_title="RAG Chat", page_icon="📄", layout="wide")

@st.cache_resource(show_spinner=False)
def load_reranker():
    # CPU is fine and safer for small GPUs (avoids OOM on MiniLM cross-encoder)
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")


@st.cache_resource(show_spinner="Processing document, this may take a minute...")
def process_pdf(pdf_path: str, chunks_file: str):
    chunks, collection = preprocesser.preprocess(
        chunk_size, chunk_overlap, embedding_model, chunks_file, pdf_path
    )
    return chunks, collection


if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "collection" not in st.session_state:
    st.session_state.collection = None
if "active_doc" not in st.session_state:
    st.session_state.active_doc = None

reranker = load_reranker()

with st.sidebar:
    st.header("📄 Document status")
    if st.session_state.active_doc:
        st.success(f"Loaded: {st.session_state.active_doc}")
        if st.button("Clear document"):
            st.session_state.chunks = None
            st.session_state.collection = None
            st.session_state.active_doc = None
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("No document loaded yet. Attach a PDF using the + icon below.")

    st.divider()
    st.caption("Settings")
    top_k_ui = st.slider("Chunks to use as context (top_k)", 1, 10, top_k)
    candidate_k_ui = st.slider("Candidates before rerank", 10, 100, candidate_k)

st.title("💬 Chat with your PDF")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt_data = st.chat_input(
    "Ask a question, or attach a PDF to get started...",
    accept_file=True,
    file_type=["pdf"],
)

if prompt_data:
    user_text = prompt_data.text or ""
    uploaded_files = prompt_data.get("files", []) if hasattr(prompt_data, "get") else prompt_data.files

    if uploaded_files:
        uploaded_file = uploaded_files[0]
        pdf_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        chunks_file = os.path.join(CHUNKS_DIR, f"{uploaded_file.name}.json")

        with st.chat_message("user"):
            st.markdown(f"📎 Uploaded **{uploaded_file.name}**" + (f"\n\n{user_text}" if user_text else ""))
        st.session_state.messages.append(
            {"role": "user", "content": f"📎 Uploaded **{uploaded_file.name}**" + (f"\n\n{user_text}" if user_text else "")}
        )

        chunks, collection = process_pdf(pdf_path, chunks_file)
        st.session_state.chunks = chunks
        st.session_state.collection = collection
        st.session_state.active_doc = uploaded_file.name

        with st.chat_message("assistant"):
            st.markdown(f"Got it — I've processed **{uploaded_file.name}** ({len(chunks)} chunks). Ask me anything about it.")
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Got it — I've processed **{uploaded_file.name}** ({len(chunks)} chunks). Ask me anything about it."}
        )

    elif user_text:
        with st.chat_message("user"):
            st.markdown(user_text)
        st.session_state.messages.append({"role": "user", "content": user_text})

        if st.session_state.chunks is None:
            answer = "Please attach a PDF first using the + icon so I have something to search."
        else:
            with st.spinner("Retrieving relevant chunks..."):
                results = retriever.retrieve(
                    reranker,
                    st.session_state.chunks,
                    user_text,
                    st.session_state.collection,
                    candidate_k_ui,
                    top_k_ui,
                )
                context = "\n\n".join(f"[{doc_id}]: {text}" for doc_id, text, _ in results)

                rag_prompt = f"""Answer the question using the context below.
                If the context doesn't contain enough information to answer, say "I don't have enough information to answer this."
                Cite the chunk ID(s) you used in your answer, like [chunk_2].
                
                Context:
                {context}
                
                Question: {user_text}
                
                Answer:"""

            with st.spinner("Generating answer..."):
                answer = generator.generate_answer(rag_prompt)

            with st.expander("Retrieved context"):
                for doc_id, text, score in results:
                    st.markdown(f"**{doc_id}** (score: {score:.4f})")
                    st.text(text[:500] + ("..." if len(text) > 500 else ""))

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})