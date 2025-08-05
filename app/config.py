"""Configuration settings for the PDF RAG system.

This module defines configuration constants, file paths, and model settings
for the PDF question-answering application. It also includes a utility
function to create necessary directories.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from typing_extensions import TypedDict

load_dotenv()


class FileInfo(TypedDict):
    """A dictionary to hold file metadata for tracking changes.

    Attributes:
        modified_time: The last modification time of the file.
        size: The size of the file in bytes.
    """

    modified_time: float
    size: int


class Config:
    """Stores configuration constants for the application.

    This class centralizes all configuration variables, making them easy to
    manage and access throughout the application.
    """

    SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    PDF_DIRECTORY = SCRIPT_DIR.joinpath("pdfs")
    DB_LOCATION = SCRIPT_DIR.joinpath("chroma_langchain_db")
    METADATA_FILE = SCRIPT_DIR.joinpath("processed_files.json")

    # Text processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 200
    RETRIEVAL_K = 20

    # API keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

    # Models
    LLM_MODEL = "llama-3.3-70b-versatile"

    # EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    # EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

    @classmethod
    def create_directories(cls) -> None:
        """Creates necessary application directories if they don't exist.

        This function ensures that the directories for PDFs, the vector
        database, and metadata files are present before the application
        attempts to use them.
        """
        cls.PDF_DIRECTORY.mkdir(parents=True, exist_ok=True)
        cls.DB_LOCATION.parent.mkdir(parents=True, exist_ok=True)
        cls.METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)


# Chat template
CHAT_TEMPLATE = """
You are an expert assistant that answers questions based on the provided
PDF documents.

Here are relevant excerpts from the documents: {context}

Question: {question}

Please provide a comprehensive answer based on the context above.
If the answer cannot be found in the provided context, please say so.
"""