from langchain_core.documents import Document

from rag.chunker import chunk_documents
from rag.loader import process_all_pdfs, process_pdf_files
from rag.qa_chain import QAChain
from rag.vectorstore import VectorStore


class RAGPipeline:
    """Ties PDF loading, chunking, vector storage, and Q&A together for multiple PDFs."""

    def __init__(
        self,
        persist_dir: str = "data/vectorstore",
        collection_name: str = "rag_collection",
        groq_model: str = "openai/gpt-oss-120b",
    ):
        self.vector_store = VectorStore(persist_dir=persist_dir, collection_name=collection_name)
        self.groq_model = groq_model
        self.qa_chain: QAChain | None = None

    def load_existing(self) -> bool:
        """Load a previously persisted vector store, if it already has content."""
        self.vector_store.load()
        if self.vector_store.is_empty():
            return False
        self._build_qa_chain()
        return True

    def ingest_directory(self, data_dir: str) -> int:
        """Load and index every PDF found recursively under data_dir. Returns chunk count."""
        documents = process_all_pdfs(data_dir)
        return self._ingest_documents(documents)

    def ingest_files(self, pdf_paths: list[str]) -> int:
        """Load and index a specific list of PDF file paths. Returns chunk count."""
        documents = process_pdf_files(pdf_paths)
        return self._ingest_documents(documents)

    def _ingest_documents(self, documents: list[Document]) -> int:
        if not documents:
            return 0

        chunks = chunk_documents(documents)
        if self.vector_store.store is None:
            self.vector_store.build(chunks)
        else:
            self.vector_store.add_documents(chunks)

        self._build_qa_chain()
        return len(chunks)

    def _build_qa_chain(self) -> None:
        retriever = self.vector_store.as_retriever()
        self.qa_chain = QAChain(retriever, model=self.groq_model)

    def ask(self, question: str) -> dict:
        if self.qa_chain is None:
            raise ValueError("No documents ingested yet — call ingest_directory() or ingest_files() first.")
        return self.qa_chain.ask(question)
