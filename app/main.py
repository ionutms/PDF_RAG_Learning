"""Main application for the PDF Question Answering system.

This module initializes and runs a chatbot that answers questions based on
a collection of PDF documents. It uses a vector store for document retrieval
and a large language model to generate answers.
"""

import re
from typing import List

from colorama import Fore, Style, init
from config import CHAT_TEMPLATE, Config
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from vector import synchronize_vector_store
from vector_store import VectorStoreManager

# Initialize colorama for cross-platform colored output
init(autoreset=True)


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

    def format_docs_colored(self, docs: List[Document]) -> str:
        """Format documents with colored output for console display.

        Args:
            docs: A list of documents to be formatted.

        Returns:
            A string with colored formatting for console display.
        """
        formatted = []
        # Use cyan for all context chunks
        color = Fore.CYAN

        for doc in docs:
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            content = doc.page_content.strip()

            # Clean up excessive whitespace and newlines
            content = self._clean_content(content)

            # Increment page number for display
            page_display = page + 1 if isinstance(page, int) else "Unknown"

            formatted_chunk = (
                f"{color}[Source: {source}, "
                f"Page: {page_display}]{Style.RESET_ALL}\n"
                f"{color}{content}{Style.RESET_ALL}"
            )
            formatted.append(formatted_chunk)

        # Join chunks with delimiter
        delimiter = f"\n{Fore.YELLOW}{'*' * 80}{Style.RESET_ALL}\n"
        return delimiter.join(formatted)

    def _clean_content(self, content: str) -> str:
        """Clean excessive whitespace and newlines from content.

        Args:
            content: The raw content string to clean

        Returns:
            Cleaned content string
        """
        # Replace multiple consecutive newlines with maximum of 2
        content = re.sub(r"\n{3,}", "\n\n", content)

        # Replace multiple consecutive spaces with single space
        content = re.sub(r" {2,}", " ", content)

        # Remove trailing whitespace from each line
        content = "\n".join(line.rstrip() for line in content.split("\n"))

        # Remove leading/trailing whitespace from entire content
        content = content.strip()

        return content

    def get_unique_sources(self, docs: List[Document]) -> List[str]:
        """Extract unique source information from a list of documents.

        Preserves the original retrieval order while removing duplicates.

        Args:
            docs: A list of documents from which to extract sources.

        Returns:
            A list of unique source strings in retrieval order.
        """
        seen_sources = set()
        unique_sources = []

        for doc in docs:
            source_file = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "Unknown")

            # Increment page number for display
            page_display = page + 1 if isinstance(page, int) else "Unknown"

            source_str = f"- {source_file} (Page {page_display})"

            # Only add if we haven't seen this source before
            if source_str not in seen_sources:
                seen_sources.add(source_str)
                unique_sources.append(source_str)

        return unique_sources

    def answer_question(self, question: str) -> tuple[str, List[str]]:
        """Answer a question based on the content of the PDF documents.

        Args:
            question: The question to be answered.

        Returns:
            A tuple containing the answer and a list of sources.
        """
        # Retrieve relevant documents
        docs = self.retriever.invoke(question)

        # Format context for LLM (plain text)
        context = self.format_docs(docs)

        # Generate response
        result = self.chain.invoke({"context": context, "question": question})

        # Get sources
        sources = self.get_unique_sources(docs)

        return result.content, sources, docs

    def run_chat_loop(self) -> None:
        """Run the main chat loop for the PDF Question Answering system.

        This method continuously prompts the user for questions and displays
        the answers until the user decides to quit.
        """
        print(f"{Fore.BLUE}PDF Question Answering System{Style.RESET_ALL}")
        print(
            f"{Fore.YELLOW}Type 'q' to quit, 'context' to show last "
            f"retrieved context{Style.RESET_ALL}"
        )

        last_docs = []

        while True:
            print("\n" + "-" * 80)
            question = input(
                f"{Fore.YELLOW}Ask your question about the PDFs "
                f"(q to quit): {Style.RESET_ALL}\n"
            )
            print()

            if question.lower() == "q":
                break

            if question.lower() == "context":
                if last_docs:
                    print(
                        f"{Fore.BLUE}Last Retrieved Context:{Style.RESET_ALL}"
                    )
                    print(self.format_docs_colored(last_docs))
                else:
                    print(
                        f"{Fore.RED}No context available. "
                        f"Ask a question first.{Style.RESET_ALL}"
                    )
                continue

            answer, sources, docs = self.answer_question(question)
            last_docs = docs

            print(f"{Fore.YELLOW}Answer:{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{answer}{Style.RESET_ALL}")

            # if sources:
            #     print(f"\n{Fore.YELLOW}Sources:{Style.RESET_ALL}")
            #     for source in sources:
            #         print(f"{Fore.CYAN}{source}{Style.RESET_ALL}")

            # Show retrieved context in color
            # print(f"\n{Fore.YELLOW}Retrieved Context:{Style.RESET_ALL}")
            # print(self.format_docs_colored(docs))
            # print("\n" + "-" * 80)


if __name__ == "__main__":
    # Synchronize the vector store at startup
    synchronize_vector_store()

    # Start the chatbot
    chatbot = PDFChatBot()
    chatbot.run_chat_loop()
