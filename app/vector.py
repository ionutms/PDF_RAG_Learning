"""Vector store synchronization logic.

This module provides the core function for synchronizing the vector store
with the PDF directory. It is designed to be run at application startup to
ensure the vector store is up-to-date.
"""

from config import Config
from file_manager import FileManager
from vector_store import VectorStoreManager


def synchronize_vector_store():
    """Synchronizes the vector store with the PDF directory.

    This function scans the PDF directory for new, modified, or deleted
    files and updates the vector store accordingly. It ensures that the
    RAG system's knowledge base is current.
    """
    print("Synchronizing vector store with PDF directory...")
    Config.create_directories()

    file_manager = FileManager()
    vector_manager = VectorStoreManager()

    new_or_modified_pdfs, processed_files = (
        file_manager.find_new_or_modified_pdfs()
    )
    deleted_files = file_manager.find_deleted_files(processed_files)

    if deleted_files:
        vector_manager.remove_deleted_files(deleted_files)
        for deleted_file in deleted_files:
            processed_files.pop(deleted_file, None)
        file_manager.save_processed_files(processed_files)

    if new_or_modified_pdfs:
        print(
            f"Found {len(new_or_modified_pdfs)} new or modified PDF files to "
            "process:"
        )
        for pdf in new_or_modified_pdfs:
            print(f"  - {pdf.name}")
        vector_manager.process_and_add_pdfs(
            new_or_modified_pdfs, processed_files
        )

    if not new_or_modified_pdfs and not deleted_files:
        print("Vector store is already up-to-date.")

    current_pdfs = list(Config.PDF_DIRECTORY.glob("*.pdf"))
    print(f"\nVector store contains {len(current_pdfs)} PDF files.")
