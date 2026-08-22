from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStore:
    """Wraps a Chroma collection with a local sentence-transformers embedding model."""

    def __init__(
        self,
        persist_dir: str = "data/vectorstore",
        collection_name: str = "rag_collection",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.store: Chroma | None = None

    def build(self, chunks: list[Document]) -> "VectorStore":
        """Embed chunks and create a fresh, persisted Chroma collection from them."""
        self.store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
        )
        return self

    def load(self) -> "VectorStore":
        """Load an existing persisted Chroma collection instead of rebuilding it."""
        self.store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )
        return self

    def add_documents(self, chunks: list[Document]) -> None:
        """Embed and add more chunks to an existing collection."""
        if self.store is None:
            raise ValueError("Vector store not initialized. Call build() or load() first.")
        self.store.add_documents(chunks)

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        if self.store is None:
            raise ValueError("Vector store not initialized. Call build() or load() first.")
        return self.store.similarity_search(query, k=k)

    def as_retriever(self, k: int = 4):
        if self.store is None:
            raise ValueError("Vector store not initialized. Call build() or load() first.")
        return self.store.as_retriever(search_kwargs={"k": k})

    def is_empty(self) -> bool:
        if self.store is None:
            return True
        return self.store._collection.count() == 0
