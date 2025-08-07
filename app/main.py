"""Main application for the PDF Question Answering system.

This module initializes and runs a chatbot that answers questions based on
a collection of PDF documents. It uses a vector store for document retrieval
and a large language model to generate answers.
"""

from typing import List

from config import CHAT_TEMPLATE, Config
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from vector import synchronize_vector_store
from vector_store import VectorStoreManager


class PDFChatBot:
    """A chatbot that answers questions based on PDF documents.

    This class encapsulates the logic for retrieving relevant document
    snippets from a vector store and using a large language model to generate
    answers to user questions.
    """

    def __init__(self) -> None:
        """Initialize the PDFChatBot.

        This method sets up the language model, the prompt template, and the
        retriever for the chatbot.
        """
        self.model = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.LLM_MODEL,
        )

        self.prompt = ChatPromptTemplate.from_template(CHAT_TEMPLATE)
        self.chain = self.prompt | self.model

        # Initialize vector store manager and get retriever
        vector_manager = VectorStoreManager()
        self.retriever = vector_manager.get_retriever()

    def format_docs(self, docs: List[Document]) -> str:
        """Format a list of documents into a single string.

        Args:
            docs: A list of documents to be formatted.

        Returns:
            A string containing the formatted documents, separated by
            double newlines.
        """
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            content = doc.page_content.strip()

            # Increment page number for display
            page_display = page + 1 if isinstance(page, int) else "Unknown"

            formatted.append(
                f"[Source: {source}, Page: {page_display}]\n{content}"
            )
        return "\n\n".join(formatted)

    def get_unique_sources(self, docs: List[Document]) -> List[str]:
        """Extract unique source information from a list of documents.

        Args:
            docs: A list of documents from which to extract sources.

        Returns:
            A sorted list of unique source strings.
        """
        sources = set()
        for doc in docs:
            source_file = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "Unknown")

            # Increment page number for display
            page_display = page + 1 if isinstance(page, int) else "Unknown"

            sources.add(f"- {source_file} (Page {page_display})")
        return sorted(sources)

    def answer_question(self, question: str) -> tuple[str, List[str]]:
        """Answer a question based on the content of the PDF documents.

        Args:
            question: The question to be answered.

        Returns:
            A tuple containing the answer and a list of sources.
        """
        # Retrieve relevant documents
        docs = self.retriever.invoke(question)
        context = self.format_docs(docs)

        # Generate response
        result = self.chain.invoke({"context": context, "question": question})

        # Get sources
        sources = self.get_unique_sources(docs)

        return result.content, sources

    def run_chat_loop(self) -> None:
        """Run the main chat loop for the PDF Question Answering system.

        This method continuously prompts the user for questions and displays
        the answers until the user decides to quit.
        """
        print("PDF Question Answering System")
        print("Type 'q' to quit")

        while True:
            print("\n" + "-" * 80)
            question = input("Ask your question about the PDFs (q to quit): ")
            print()

            if question.lower() == "q":
                break

            answer, sources = self.answer_question(question)

            print("Answer:")
            print(answer)

            if sources:
                print("\nSources:")
                for source in sources:
                    print(source)
            print("\n" + "-" * 80)


if __name__ == "__main__":
    # Synchronize the vector store at startup
    synchronize_vector_store()

    # Start the chatbot
    chatbot = PDFChatBot()
    chatbot.run_chat_loop()
