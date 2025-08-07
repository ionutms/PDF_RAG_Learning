"""Manages vector store operations for PDF documents.

This module provides the `VectorStoreManager` class, which handles the
entire lifecycle of PDF documents in the vector store, including embedding,
chunking, adding, and removing documents.
"""

from pathlib import Path
from typing import Dict, List, Set

from config import Config, FileInfo
from file_manager import FileManager
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorStoreManager:
    """Handles all vector store operations for PDF documents.

    This class is responsible for initializing the embedding model and vector
    store, processing and adding new PDFs, removing outdated documents, and
    providing a retriever for querying.

    Attributes:
        embeddings: The sentence-transformer model for creating embeddings.
        vector_store: The ChromaDB instance for storing document vectors.
        text_splitter: The splitter for dividing documents into chunks.
        file_manager: An instance of `FileManager` for metadata handling.
    """

    def __init__(self) -> None:
        """Initializes the VectorStoreManager."""
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model=Config.EMBEDDING_MODEL,
            huggingfacehub_api_token=Config.HUGGINGFACE_API_KEY,
        )

        self.vector_store = Chroma(
            collection_name="pdf_documents",
            persist_directory=str(Config.DB_LOCATION),
            embedding_function=self.embeddings,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
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

        This method handles loading, splitting, and embedding the documents,
        and then updates the tracking metadata.

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

            pdf_documents = self._load_pdf(pdf_path)
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
        """Finds the internal document IDs associated with a set of filenames.

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

    def _load_pdf(self, pdf_path: Path) -> List[Document]:
        """Loads a PDF file and enriches its documents with metadata.

        Args:
            pdf_path: The path to the PDF file.

        Returns:
            A list of `Document` objects, each with added metadata.
        """
        loader = PyPDFLoader(str(pdf_path))
        pdf_documents = loader.load()

        for doc in pdf_documents:
            doc.metadata["source_file"] = pdf_path.name
            doc.metadata["file_path"] = str(pdf_path)

        return pdf_documents

    def _split_and_add_documents(self, documents: List[Document]) -> None:
        """Splits documents into chunks and adds them to the vector store.

        Args:
            documents: A list of documents to be split and added.
        """
        print("Splitting documents into chunks...")
        split_documents = self.text_splitter.split_documents(documents)
        print(f"Created {len(split_documents)} document chunks")

        print("Adding new/updated documents to vector store...")
        self.vector_store.add_documents(documents=split_documents)
