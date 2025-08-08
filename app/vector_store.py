"""Vector store manager updated to persist markdowns and ingest them.

- Converts PDFs to markdown files (no images).
- Saves markdowns under <project>/markdowns.
- Vector ingestion is performed from the markdown files.
- Deletion of PDFs removes associated markdown files.
"""

from pathlib import Path
from typing import List, Set

import pymupdf4llm
from config import Config, FileInfo
from file_manager import FileManager
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from semantic_splitter import TableAwareSemanticSplitter


class VectorStoreManager:
    """Handles vector store lifecycle and markdown-based ingestion."""

    def __init__(self) -> None:
        """Initialize embeddings, vector store, splitter and file manager."""
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model=Config.EMBEDDING_MODEL,
            huggingfacehub_api_token=Config.HUGGINGFACE_API_KEY,
        )

        self.vector_store = Chroma(
            collection_name="pdf_documents",
            persist_directory=str(Config.DB_LOCATION),
            embedding_function=self.embeddings,
        )

        self.text_splitter = TableAwareSemanticSplitter(
            embeddings=self.embeddings,
            similarity_threshold=getattr(
                Config, "SEMANTIC_SIMILARITY_THRESHOLD", 0.7
            ),
            min_chunk_size=getattr(Config, "MIN_CHUNK_SIZE", 200),
            max_chunk_size=Config.CHUNK_SIZE,
            sentence_overlap=getattr(Config, "SENTENCE_OVERLAP", 1),
            table_preserve_structure=getattr(
                Config, "TABLE_PRESERVE_STRUCTURE", True
            ),
            table_min_chunk_size=getattr(Config, "TABLE_CHUNK_MIN_SIZE", 100),
        )

        self.file_manager = FileManager()

    def remove_deleted_files(self, deleted_files: Set[str]) -> None:
        """Remove deleted PDFs' chunks and their markdown files."""
        if not deleted_files:
            return

        print(
            "Removing %d deleted files from vector store and markdown "
            "directory:" % len(deleted_files)
        )
        for deleted_file in deleted_files:
            filename = Path(deleted_file).name
            print(f"  - {filename}")

        all_docs = self.vector_store.get()
        if all_docs and all_docs.get("metadatas"):
            ids_to_delete = self._find_ids_by_filenames(
                all_docs, {Path(f).name for f in deleted_files}
            )
            if ids_to_delete:
                self.vector_store.delete(ids=ids_to_delete)
                print(
                    f"Removed {len(ids_to_delete)} chunks from deleted files"
                )

        # Remove generated markdown files for deleted PDFs
        for pdf_path_str in deleted_files:
            stem = Path(pdf_path_str).stem
            for md in Config.MARKDOWN_DIRECTORY.glob(f"{stem}*.md"):
                try:
                    md.unlink()
                    print(f"Deleted markdown: {md.name}")
                except Exception:
                    print(f"Could not delete markdown: {md}")

    def remove_old_chunks(self, filename: str) -> None:
        """Remove all vector chunks for the filename and its markdowns."""
        print(f"Removing old chunks for modified file: {filename}")

        all_docs = self.vector_store.get()
        if not all_docs or not all_docs.get("metadatas"):
            return

        ids_to_delete = self._find_ids_by_filenames(all_docs, {filename})

        if ids_to_delete:
            self.vector_store.delete(ids=ids_to_delete)
            print(f"Removed {len(ids_to_delete)} old chunks")

        # Also remove old markdowns for this file
        stem = Path(filename).stem
        for md in Config.MARKDOWN_DIRECTORY.glob(f"{stem}*.md"):
            try:
                md.unlink()
                print(f"Removed old markdown: {md.name}")
            except Exception:
                print(f"Could not remove old markdown: {md}")

    def process_and_add_pdfs(
        self,
        pdf_paths: List[Path],
        processed_files: dict[str, FileInfo],
    ) -> None:
        """Process PDFs to markdowns, then split markdowns and add vectors."""
        # First convert each PDF to markdown file(s)
        for pdf_path in pdf_paths:
            file_key = str(pdf_path)

            if file_key in processed_files:
                self.remove_old_chunks(pdf_path.name)

            print(f"Processing and saving markdown for: {pdf_path.name}")
            # write markdown(s) to disk
            self._save_pdf_as_markdown(pdf_path)

            # Update processed_files metadata for the original PDF path
            processed_files[file_key] = self.file_manager.get_file_info(
                pdf_path
            )

        # Load markdown files for all created/updated pdfs
        markdown_docs: List[Document] = []
        for md_path in Config.MARKDOWN_DIRECTORY.rglob("*.md"):
            # read content
            try:
                content = md_path.read_text(encoding="utf-8")
            except Exception:
                continue

            if not content.strip():
                continue

            doc = Document(
                page_content=content.strip(),
                metadata={
                    "source_file": md_path.name,
                    "file_path": str(md_path),
                    "page": 0,
                    "content_type": "markdown_file",
                },
            )
            markdown_docs.append(doc)

        self._split_and_add_documents(markdown_docs)

        self.file_manager.save_processed_files(processed_files)
        print("Documents updated successfully!")

    def get_retriever(self):
        """Return a retriever for queries."""
        return self.vector_store.as_retriever(
            search_kwargs={"k": Config.RETRIEVAL_K}
        )

    def _find_ids_by_filenames(
        self,
        all_docs: dict,
        filenames: Set[str],
    ) -> List[str]:
        """Return internal ids for given filenames."""
        ids_to_delete: List[str] = []

        for i, metadata in enumerate(all_docs["metadatas"]):
            if (
                metadata
                and "source_file" in metadata
                and metadata["source_file"] in filenames
            ):
                ids_to_delete.append(all_docs["ids"][i])

        return ids_to_delete

    def _save_pdf_as_markdown(self, pdf_path: Path) -> List[Path]:
        """Convert a PDF to markdown files and save them in a subfolder.

        Returns a list of saved markdown Path objects. Images are skipped.
        Each page becomes a separate markdown file when pymupdf4llm
        returns page chunks.
        """
        md_result = pymupdf4llm.to_markdown(
            str(pdf_path), **Config.PYMUPDF_EXTRACT_OPTIONS
        )

        saved: List[Path] = []

        stem = pdf_path.stem
        pdf_md_dir = Config.MARKDOWN_DIRECTORY.joinpath(stem)
        pdf_md_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(md_result, list):
            for page_num, page_data in enumerate(md_result, start=1):
                if isinstance(page_data, dict):
                    page_content = page_data.get("text", "")
                else:
                    page_content = page_data

                if not page_content or not page_content.strip():
                    continue

                md_name = f"{stem}_page_{page_num}.md"
                md_path = pdf_md_dir.joinpath(md_name)
                md_path.write_text(page_content.strip(), encoding="utf-8")
                saved.append(md_path)

        elif isinstance(md_result, str):
            if md_result.strip():
                md_name = f"{stem}_page_1.md"
                md_path = pdf_md_dir.joinpath(md_name)
                md_path.write_text(md_result.strip(), encoding="utf-8")
                saved.append(md_path)

        return saved

    def _split_and_add_documents(self, documents: List[Document]) -> None:
        """Split documents into chunks and add them to the vector store."""
        print("Splitting markdown documents into table-aware chunks...")

        split_documents = self.text_splitter.split_documents(documents)

        for doc in split_documents:
            doc.metadata["chunking_strategy"] = "table_aware_semantic"
            doc.metadata["similarity_threshold"] = (
                self.text_splitter.similarity_threshold
            )

        print(f"Created {len(split_documents)} document chunks")

        chunk_sizes = [len(doc.page_content) for doc in split_documents]
        table_chunks = [
            doc
            for doc in split_documents
            if doc.metadata.get("contains_table", False)
        ]

        if chunk_sizes:
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            min_size = min(chunk_sizes)
            max_size = max(chunk_sizes)
            print(
                "Chunk size stats - "
                f"Min: {min_size}, Max: {max_size}, Avg: {avg_size:.0f}"
            )
            print(
                "Table-containing chunks: "
                f"{len(table_chunks)}/{len(split_documents)}"
            )

        print("Adding new/updated documents to vector store...")
        if split_documents:
            self.vector_store.add_documents(documents=split_documents)
