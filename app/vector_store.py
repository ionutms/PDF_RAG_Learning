"""Manages vector store operations for PDF documents with semantic chunking.

This module provides the `VectorStoreManager` class, which handles the
entire lifecycle of PDF documents in the vector store, including embedding,
semantic chunking, adding, and removing documents using pymupdf4llm for
markdown extraction and semantic text splitting for intelligent chunking.
"""

from pathlib import Path
from typing import Dict, List, Set

import pymupdf4llm
from config import Config, FileInfo
from file_manager import FileManager
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from semantic_splitter import SemanticTextSplitter


class VectorStoreManager:
    """Handles all vector store operations for PDF documents with chunking.

    This class is responsible for initializing the embedding model and vector
    store, processing and adding new PDFs using pymupdf4llm for markdown
    extraction, semantic chunking for intelligent text splitting, removing
    outdated documents, and providing a retriever for querying.

    Attributes:
        embeddings: The sentence-transformer model for creating embeddings.
        vector_store: The ChromaDB instance for storing document vectors.
        text_splitter: The semantic splitter for dividing documents into
            contextually coherent chunks.
        file_manager: An instance of `FileManager` for metadata handling.
    """

    def __init__(self) -> None:
        """Initializes the VectorStoreManager with semantic chunking."""
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model=Config.EMBEDDING_MODEL,
            huggingfacehub_api_token=Config.HUGGINGFACE_API_KEY,
        )

        self.vector_store = Chroma(
            collection_name="pdf_documents",
            persist_directory=str(Config.DB_LOCATION),
            embedding_function=self.embeddings,
        )

        # Initialize semantic text splitter
        self.text_splitter = SemanticTextSplitter(
            embeddings=self.embeddings,
            similarity_threshold=getattr(
                Config, "SEMANTIC_SIMILARITY_THRESHOLD", 0.7
            ),
            min_chunk_size=getattr(Config, "MIN_CHUNK_SIZE", 200),
            max_chunk_size=Config.CHUNK_SIZE,
            sentence_overlap=getattr(Config, "SENTENCE_OVERLAP", 1),
        )

        self.file_manager = FileManager()

    def remove_deleted_files(self, deleted_files: Set[str]) -> None:
        """Removes document chunks from the vector store for deleted files.

        Args:
            deleted_files: A set of file paths for the deleted files.
        """
        if not deleted_files:
            return

        print(
            f"Removing {len(deleted_files)} deleted files from vector store:"
        )
        for deleted_file in deleted_files:
            filename = Path(deleted_file).name
            print(f"  - {filename}")

        all_docs = self.vector_store.get()

        if not all_docs or not all_docs["metadatas"]:
            return

        ids_to_delete = self._find_ids_by_filenames(
            all_docs, {Path(f).name for f in deleted_files}
        )

        if ids_to_delete:
            self.vector_store.delete(ids=ids_to_delete)
            print(f"Removed {len(ids_to_delete)} chunks from deleted files")

    def remove_old_chunks(self, filename: str) -> None:
        """Removes all document chunks for a specific file.

        This is used to clear out old versions of a file before adding the
        updated version.

        Args:
            filename: The name of the file whose chunks should be removed.
        """
        print(f"Removing old chunks for modified file: {filename}")

        all_docs = self.vector_store.get()
        if not all_docs or not all_docs["metadatas"]:
            return

        ids_to_delete = self._find_ids_by_filenames(all_docs, {filename})

        if ids_to_delete:
            self.vector_store.delete(ids=ids_to_delete)
            print(f"Removed {len(ids_to_delete)} old chunks")

    def process_and_add_pdfs(
        self,
        pdf_paths: List[Path],
        processed_files: Dict[str, FileInfo],
    ) -> None:
        """Processes and adds a list of PDF files to the vector store.

        This method handles loading, converting to markdown,
        semantic splitting, and embedding the documents,
        then updates the tracking metadata.

        Args:
            pdf_paths: A list of paths to the PDF files to process.
            processed_files: The dictionary of processed file metadata, which
                will be updated.
        """
        documents = []

        for pdf_path in pdf_paths:
            file_key = str(pdf_path)

            if file_key in processed_files:
                self.remove_old_chunks(pdf_path.name)

            print(f"Processing: {pdf_path.name}")

            pdf_documents = self._load_pdf_as_markdown(pdf_path)
            documents.extend(pdf_documents)

            processed_files[file_key] = self.file_manager.get_file_info(
                pdf_path
            )

        self._split_and_add_documents(documents)

        self.file_manager.save_processed_files(processed_files)
        print("Documents updated successfully!")

    def get_retriever(self):
        """Returns a retriever for the vector store.

        The retriever is configured to fetch a specific number of documents.

        Returns:
            A retriever object for querying the vector store.
        """
        return self.vector_store.as_retriever(
            search_kwargs={"k": Config.RETRIEVAL_K}
        )

    def _find_ids_by_filenames(
        self,
        all_docs: dict,
        filenames: Set[str],
    ) -> List[str]:
        """Finds the internal document IDs associated with filenames.

        Args:
            all_docs: The dictionary of all documents from the vector store.
            filenames: A set of filenames to search for.

        Returns:
            A list of document IDs to be deleted.
        """
        ids_to_delete = []

        for i, metadata in enumerate(all_docs["metadatas"]):
            if (
                metadata
                and "source_file" in metadata
                and metadata["source_file"] in filenames
            ):
                ids_to_delete.append(all_docs["ids"][i])

        return ids_to_delete

    def _load_pdf_as_markdown(self, pdf_path: Path) -> List[Document]:
        """Loads a PDF file and converts it to markdown using pymupdf4llm.

        Args:
            pdf_path: The path to the PDF file.

        Returns:
            A list of `Document` objects, each representing a page with
            markdown content and metadata.
        """
        # Extract markdown from PDF using pymupdf4llm
        md_result = pymupdf4llm.to_markdown(
            str(pdf_path), **Config.PYMUPDF_EXTRACT_OPTIONS
        )

        documents = []

        # Handle both list (page_chunks=True) and string (page_chunks=False)
        if isinstance(md_result, list):
            # page_chunks=True returns a list of page dictionaries
            for page_num, page_data in enumerate(md_result):
                if isinstance(page_data, dict):
                    # Extract text from dictionary
                    page_content = page_data.get("text", "")
                    if page_content and page_content.strip():
                        doc = Document(
                            page_content=page_content.strip(),
                            metadata={
                                "source_file": pdf_path.name,
                                "file_path": str(pdf_path),
                                "page": page_num,
                                "content_type": "markdown",
                            },
                        )
                        documents.append(doc)
                elif isinstance(page_data, str):
                    # Handle case where it's a string
                    if page_data.strip():
                        doc = Document(
                            page_content=page_data.strip(),
                            metadata={
                                "source_file": pdf_path.name,
                                "file_path": str(pdf_path),
                                "page": page_num,
                                "content_type": "markdown",
                            },
                        )
                        documents.append(doc)
        else:
            # page_chunks=False returns a single string
            if md_result and md_result.strip():
                doc = Document(
                    page_content=md_result.strip(),
                    metadata={
                        "source_file": pdf_path.name,
                        "file_path": str(pdf_path),
                        "page": 0,
                        "content_type": "markdown",
                    },
                )
                documents.append(doc)

        return documents

    def _split_and_add_documents(self, documents: List[Document]) -> None:
        """Splits documents into chunks and adds them to the vector store.

        Args:
            documents: A list of documents to be split and added.
        """
        print("Splitting markdown documents into semantic chunks...")

        # Use semantic splitter instead of fixed-size splitter
        split_documents = self.text_splitter.split_documents(documents)

        # Add chunking strategy info to metadata
        for doc in split_documents:
            doc.metadata["chunking_strategy"] = "semantic"
            doc.metadata["similarity_threshold"] = (
                self.text_splitter.similarity_threshold
            )

        print(f"Created {len(split_documents)} semantic document chunks")

        # Show some statistics about chunk sizes
        chunk_sizes = [len(doc.page_content) for doc in split_documents]
        if chunk_sizes:
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            min_size = min(chunk_sizes)
            max_size = max(chunk_sizes)
            print(
                "Chunk size stats - "
                f"Min: {min_size}, Max: {max_size}, Avg: {avg_size:.0f}"
            )

        print("Adding new/updated documents to vector store...")
        self.vector_store.add_documents(documents=split_documents)
