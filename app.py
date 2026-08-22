import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag.pipeline import RAGPipeline  # noqa: E402

load_dotenv()

UPLOAD_DIR = Path("data/pdf/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="PDF Q&A Assistant", page_icon="📄")
st.title("📄 Custom PDF Q&A Assistant")
st.caption("Upload one or more PDFs, then ask questions grounded in their content.")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()
    st.session_state.pipeline.load_existing()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

pipeline: RAGPipeline = st.session_state.pipeline

with st.sidebar:
    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Process PDFs", disabled=not uploaded_files):
        new_paths = []
        for uploaded_file in uploaded_files:
            dest = UPLOAD_DIR / uploaded_file.name
            dest.write_bytes(uploaded_file.getbuffer())
            new_paths.append(str(dest))

        with st.spinner(f"Indexing {len(new_paths)} PDF(s)..."):
            chunk_count = pipeline.ingest_files(new_paths)

        st.session_state.indexed_files.update(p.name for p in map(Path, new_paths))
        st.success(f"Indexed {chunk_count} chunks from {len(new_paths)} file(s).")

    if st.session_state.indexed_files:
        st.subheader("Indexed this session")
        for name in sorted(st.session_state.indexed_files):
            st.text(f"• {name}")

    if pipeline.qa_chain is None:
        st.info("No documents indexed yet. Upload PDFs above, or add them to data/pdf/ and re-run ingestion.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for doc in message["sources"]:
                    page = doc.metadata.get("page", "?")
                    source = doc.metadata.get("source_file", "unknown")
                    st.markdown(f"**{source}** (page {page})")
                    st.text(doc.page_content[:300])

question = st.chat_input("Ask a question about your PDFs...")

if question:
    if pipeline.qa_chain is None:
        st.error("Please upload and process at least one PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = pipeline.ask(question)
            st.markdown(result["answer"])
            with st.expander("Sources"):
                for doc in result["sources"]:
                    page = doc.metadata.get("page", "?")
                    source = doc.metadata.get("source_file", "unknown")
                    st.markdown(f"**{source}** (page {page})")
                    st.text(doc.page_content[:300])

        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
        )
