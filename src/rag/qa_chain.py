from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a set of PDF documents.
Answer the question using ONLY the context below. If the answer isn't in the context, say you don't know.

Context:
{context}
"""


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('source_file', 'unknown')} p.{d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


class QAChain:
    """Retrieves relevant chunks for a question and asks Groq to answer from them."""

    def __init__(self, retriever, model: str = "openai/gpt-oss-120b", temperature: float = 0.0):
        self.retriever = retriever
        self.llm = ChatGroq(model=model, temperature=temperature)
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "{question}")]
        )

    def ask(self, question: str) -> dict:
        docs = self.retriever.invoke(question)
        messages = self.prompt.format_messages(context=format_docs(docs), question=question)
        response = self.llm.invoke(messages)
        return {"answer": response.content, "sources": docs}
