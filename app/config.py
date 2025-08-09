"""Configuration settings for the PDF RAG system with semantic chunking."""

import os
from pathlib import Path

from dotenv import load_dotenv
from typing_extensions import TypedDict

load_dotenv()


class FileInfo(TypedDict):
    """File metadata for tracking processed files."""

    modified_time: float
    size: int


class Config:
    """Holds application-wide configuration constants."""

    SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    PDF_DIRECTORY = SCRIPT_DIR.joinpath("pdfs")
    MARKDOWN_DIRECTORY = SCRIPT_DIR.joinpath("markdowns")
    DB_LOCATION = SCRIPT_DIR.joinpath("chroma_DB/chroma_langchain_db")
    METADATA_FILE = SCRIPT_DIR.joinpath("chroma_DB/processed_files.json")

    # Chunking & retrieval
    CHUNK_SIZE = 1500
    RETRIEVAL_K = 30

    SEMANTIC_SIMILARITY_THRESHOLD = 0.9
    MIN_CHUNK_SIZE = 200
    SENTENCE_OVERLAP = 2

    # pymupdf4llm extraction options: no images, preserve tables
    PYMUPDF_EXTRACT_OPTIONS = {
        "page_chunks": True,
        "write_images": False,
        "embed_images": False,
        "table_strategy": "lines_strict",
        "margins": (0, 0, 0, 0),
        "force_text": True,
    }

    TABLE_CHUNK_MIN_SIZE = 100
    TABLE_PRESERVE_STRUCTURE = True

    # API keys
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

    # Models
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

    @classmethod
    def create_directories(cls) -> None:
        """Create filesystem directories required by the app."""
        cls.PDF_DIRECTORY.mkdir(parents=True, exist_ok=True)
        cls.MARKDOWN_DIRECTORY.mkdir(parents=True, exist_ok=True)
        cls.DB_LOCATION.parent.mkdir(parents=True, exist_ok=True)
        cls.METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def print_semantic_config(cls) -> None:
        """Print current semantic chunking configuration."""
        print("=== Semantic Chunking Configuration ===")
        print(f"Similarity Threshold: {cls.SEMANTIC_SIMILARITY_THRESHOLD}")
        print(f"Max Chunk Size: {cls.CHUNK_SIZE}")
        print(f"Min Chunk Size: {cls.MIN_CHUNK_SIZE}")
        print(f"Sentence Overlap: {cls.SENTENCE_OVERLAP}")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print("=" * 40)
