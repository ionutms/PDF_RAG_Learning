"""File management utilities for tracking PDF files.

This module provides the `FileManager` class, which is responsible for
monitoring a directory of PDF files for additions, modifications, and
deletions. It uses a metadata file to track the state of processed files.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set

from config import Config, FileInfo


class FileManager:
    """Manages PDF file tracking and metadata persistence.

    This class handles loading and saving the metadata of processed files,
    detecting changes in the PDF directory, and identifying new, modified,
    or deleted files.

    Attributes:
        metadata_file: The path to the JSON file storing processed file
            metadata.
        pdf_directory: The path to the directory containing PDF files.
    """

    def __init__(self) -> None:
        """Initializes the FileManager with paths from the configuration."""
        self.metadata_file = Config.METADATA_FILE
        self.pdf_directory = Config.PDF_DIRECTORY

    def load_processed_files(self) -> Dict[str, FileInfo]:
        """Loads the dictionary of processed files from the metadata file.

        If the metadata file does not exist, it returns an empty dictionary.

        Returns:
            A dictionary mapping file paths to their `FileInfo` metadata.
        """
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}

    def save_processed_files(
        self, processed_files: Dict[str, FileInfo]
    ) -> None:
        """Saves the dictionary of processed files to the metadata file.

        Args:
            processed_files: A dictionary mapping file paths to their
                `FileInfo` metadata.
        """
        with open(self.metadata_file, "w") as f:
            json.dump(processed_files, f, indent=2)

    def get_file_info(self, file_path: Path) -> FileInfo:
        """Retrieves the modification time and size of a file.

        Args:
            file_path: The path to the file.

        Returns:
            A `FileInfo` dictionary containing the file's metadata.
        """
        stat = os.stat(file_path)
        return FileInfo(modified_time=stat.st_mtime, size=stat.st_size)

    def find_new_or_modified_pdfs(
        self,
    ) -> tuple[List[Path], Dict[str, FileInfo]]:
        """Finds new or modified PDF files in the directory.

        Compares the current state of PDF files against the stored metadata
        to identify changes.

        Returns:
            A tuple containing:
            - A list of paths to new or modified PDF files.
            - The dictionary of all previously processed files.
        """
        processed_files = self.load_processed_files()
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        new_or_modified = []

        for pdf_path in pdf_files:
            file_key = str(pdf_path)
            current_info = self.get_file_info(pdf_path)

            if self._is_file_changed(file_key, current_info, processed_files):
                new_or_modified.append(pdf_path)

        return new_or_modified, processed_files

    def find_deleted_files(
        self, processed_files: Dict[str, FileInfo]
    ) -> Set[str]:
        """Finds PDF files that have been deleted since the last check.

        Args:
            processed_files: A dictionary of previously processed files.

        Returns:
            A set of file paths for the deleted files.
        """
        current_pdf_files = {
            str(pdf) for pdf in self.pdf_directory.glob("*.pdf")
        }
        processed_file_paths = set(processed_files.keys())
        return processed_file_paths - current_pdf_files

    def _is_file_changed(
        self,
        file_key: str,
        current_info: FileInfo,
        processed_files: Dict[str, FileInfo],
    ) -> bool:
        """Checks if a file has been modified since it was last processed.

        A file is considered changed if it is new or if its modification
        time or size has changed.

        Args:
            file_key: The path of the file to check.
            current_info: The current `FileInfo` of the file.
            processed_files: The dictionary of processed file metadata.

        Returns:
            True if the file is new or has been modified, False otherwise.
        """
        if file_key not in processed_files:
            return True

        old_info = processed_files[file_key]
        return (
            old_info["modified_time"] != current_info["modified_time"]
            or old_info["size"] != current_info["size"]
        )
