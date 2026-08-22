from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


def process_all_pdfs(data_dir: str) -> list[Document]:
    """Recursively find every PDF under data_dir, load it, and tag each doc with source metadata."""
    data_path = Path(data_dir)
    pdf_files = sorted(data_path.rglob("*.pdf"))

    all_documents: list[Document] = []
    for pdf_file in pdf_files:
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            documents = loader.load()

            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"

            all_documents.extend(documents)
        except Exception as e:
            print(f"Error loading {pdf_file.name}: {e}")

    return all_documents


def process_pdf_files(pdf_paths: list[str]) -> list[Document]:
    """Load a specific list of PDF file paths (e.g. freshly uploaded files) and tag with source metadata."""
    all_documents: list[Document] = []
    for pdf_path in pdf_paths:
        path = Path(pdf_path)
        try:
            loader = PyMuPDFLoader(str(path))
            documents = loader.load()

            for doc in documents:
                doc.metadata["source_file"] = path.name
                doc.metadata["file_type"] = "pdf"

            all_documents.extend(documents)
        except Exception as e:
            print(f"Error loading {path.name}: {e}")

    return all_documents
