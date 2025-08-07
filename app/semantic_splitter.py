"""Semantic text splitter for creating context-aware document chunks.

This module provides a SemanticTextSplitter that groups text based on
semantic similarity rather than fixed chunk sizes. It uses embeddings
to determine when content changes topic significantly enough to warrant
a new chunk.
"""

import re
from typing import List

import numpy as np
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


class SemanticTextSplitter:
    """A text splitter that creates chunks based on semantic similarity.

    This splitter analyzes the semantic content of sentences and groups
    them into chunks where sentences within each chunk are semantically
    similar. It uses cosine similarity between sentence embeddings to
    determine chunk boundaries.

    Attributes:
        embeddings: The embedding model used to encode sentences.
        similarity_threshold: The minimum similarity required to keep
            sentences in the same chunk.
        min_chunk_size: The minimum number of characters per chunk.
        max_chunk_size: The maximum number of characters per chunk.
        sentence_overlap: Number of sentences to overlap between chunks.
    """

    def __init__(
        self,
        embeddings: HuggingFaceEndpointEmbeddings,
        similarity_threshold: float = 0.7,
        min_chunk_size: int = 200,
        max_chunk_size: int = 1500,
        sentence_overlap: int = 1,
    ):
        """Initialize the SemanticTextSplitter.

        Args:
            embeddings: The embedding model to use for semantic analysis.
            similarity_threshold: Similarity threshold for grouping sentences
                (0.0 to 1.0, higher means more similar content required).
            min_chunk_size: Minimum chunk size in characters.
            max_chunk_size: Maximum chunk size in characters.
            sentence_overlap: Number of sentences to overlap between chunks.
        """
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.sentence_overlap = sentence_overlap

    def split_text(self, text: str) -> List[str]:
        """Split text into semantically coherent chunks.

        Args:
            text: The input text to be split.

        Returns:
            A list of text chunks.
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)

        if len(sentences) <= 1:
            return [text]

        # Get embeddings for all sentences
        try:
            embeddings = self._get_sentence_embeddings(sentences)
        except Exception as e:
            print(
                f"Warning: Embedding failed ({e}), "
                "falling back to simple splitting"
            )
            return self._fallback_split(text)

        # Group sentences into semantic chunks
        chunks = self._create_semantic_chunks(sentences, embeddings)

        # Post-process chunks to respect size constraints
        final_chunks = self._post_process_chunks(chunks)

        return final_chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split a list of documents into smaller chunks.

        Args:
            documents: List of documents to split.

        Returns:
            List of split documents with preserved metadata.
        """
        split_docs = []

        for doc in documents:
            chunks = self.split_text(doc.page_content)

            for i, chunk in enumerate(chunks):
                chunk_metadata = doc.metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(chunks)

                split_doc = Document(
                    page_content=chunk, metadata=chunk_metadata
                )
                split_docs.append(split_doc)

        return split_docs

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns.

        This method handles common sentence boundaries while being careful
        about abbreviations and other edge cases.

        Args:
            text: The input text.

        Returns:
            List of sentences.
        """
        # Clean up the text first
        text = re.sub(r"\s+", " ", text.strip())

        # More sophisticated sentence splitting pattern
        # This pattern looks for sentence-ending punctuation followed by
        # whitespace and a capital letter, but avoids common abbreviations
        sentence_pattern = r"""
            (?<!\w\.\w.)
            (?<![A-Z][a-z]\.)
            (?<!\d\.)
            [.!?]+
            \s+
            (?=[A-Z])
        """

        sentences = re.split(sentence_pattern, text, flags=re.VERBOSE)

        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        # If no sentences found (no proper sentence boundaries),
        # split by line breaks as fallback
        if len(sentences) <= 1:
            sentences = [s.strip() for s in text.split("\n") if s.strip()]

        return sentences

    def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Get embeddings for a list of sentences.

        Args:
            sentences: List of sentences to embed.

        Returns:
            NumPy array of embeddings.
        """
        # Filter out very short sentences that might not be meaningful
        meaningful_sentences = [
            s
            for s in sentences
            if len(s.split()) >= 3  # At least 3 words
        ]

        if not meaningful_sentences:
            meaningful_sentences = sentences  # Fallback to all sentences

        # Get embeddings from HuggingFace
        embeddings = self.embeddings.embed_documents(meaningful_sentences)
        return np.array(embeddings)

    def _create_semantic_chunks(
        self, sentences: List[str], embeddings: np.ndarray
    ) -> List[str]:
        """Group sentences into semantic chunks based on similarity.

        Args:
            sentences: List of sentences.
            embeddings: Corresponding embeddings for sentences.

        Returns:
            List of text chunks.
        """
        if len(sentences) != len(embeddings):
            # Fallback to simple grouping
            return self._simple_group_sentences(sentences)

        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_chunk_embeddings = [embeddings[0]]

        for i in range(1, len(sentences)):
            # Calculate similarity between current sentence and chunk
            current_embedding = embeddings[i].reshape(1, -1)

            # Use mean of current chunk embeddings for comparison
            chunk_embedding = np.mean(
                current_chunk_embeddings, axis=0
            ).reshape(1, -1)

            similarity = cosine_similarity(
                current_embedding, chunk_embedding
            )[0][0]

            # Check if current chunk would exceed max size with new sentence
            current_chunk_text = " ".join(
                current_chunk_sentences + [sentences[i]]
            )
            would_exceed_max = len(current_chunk_text) > self.max_chunk_size

            # Decide whether to add to current chunk or start new one
            if (
                similarity >= self.similarity_threshold
                and not would_exceed_max
            ):
                # Add to current chunk
                current_chunk_sentences.append(sentences[i])
                current_chunk_embeddings.append(embeddings[i])
            else:
                # Start new chunk
                if current_chunk_sentences:
                    chunks.append(" ".join(current_chunk_sentences))

                # Start new chunk with overlap if specified
                overlap_sentences = current_chunk_sentences[
                    -self.sentence_overlap :
                ]
                overlap_embeddings = current_chunk_embeddings[
                    -self.sentence_overlap :
                ]

                current_chunk_sentences = overlap_sentences + [sentences[i]]
                current_chunk_embeddings = overlap_embeddings + [
                    embeddings[i]
                ]

        # Add the last chunk
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks

    def _simple_group_sentences(self, sentences: List[str]) -> List[str]:
        """Fallback method to group sentences without semantic analysis.

        Args:
            sentences: List of sentences to group.

        Returns:
            List of text chunks.
        """
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # Check if adding this sentence would exceed max chunk size
            if (
                current_length + sentence_length > self.max_chunk_size
                and current_chunk
            ):
                # Finalize current chunk
                chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap
                overlap_sentences = (
                    current_chunk[-self.sentence_overlap :]
                    if self.sentence_overlap > 0
                    else []
                )
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _post_process_chunks(self, chunks: List[str]) -> List[str]:
        """Post-process chunks to ensure they meet size requirements.

        Args:
            chunks: List of initial chunks.

        Returns:
            List of processed chunks.
        """
        processed_chunks = []

        for chunk in chunks:
            # Skip very small chunks unless they're the only content
            if len(chunk) < self.min_chunk_size and len(chunks) > 1:
                # Try to merge with previous chunk if possible
                if (
                    processed_chunks
                    and len(processed_chunks[-1] + " " + chunk)
                    <= self.max_chunk_size
                ):
                    processed_chunks[-1] = processed_chunks[-1] + " " + chunk
                    continue
                # Otherwise skip this chunk
                continue

            # Split chunks that are too large
            if len(chunk) > self.max_chunk_size:
                sub_chunks = self._split_large_chunk(chunk)
                processed_chunks.extend(sub_chunks)
            else:
                processed_chunks.append(chunk)

        return processed_chunks

    def _split_large_chunk(self, chunk: str) -> List[str]:
        """Split a chunk that exceeds maximum size.

        Args:
            chunk: The chunk to split.

        Returns:
            List of smaller chunks.
        """
        # Simple word-based splitting for oversized chunks
        words = chunk.split()
        sub_chunks = []
        current_sub_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space

            if (
                current_length + word_length > self.max_chunk_size
                and current_sub_chunk
            ):
                sub_chunks.append(" ".join(current_sub_chunk))
                current_sub_chunk = [word]
                current_length = len(word)
            else:
                current_sub_chunk.append(word)
                current_length += word_length

        if current_sub_chunk:
            sub_chunks.append(" ".join(current_sub_chunk))

        return sub_chunks

    def _fallback_split(self, text: str) -> List[str]:
        """Fallback splitting method when embeddings fail.

        Args:
            text: Text to split.

        Returns:
            List of text chunks.
        """
        # Simple character-based splitting
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.max_chunk_size

            if end >= len(text):
                chunks.append(text[start:])
                break

            # Try to break at word boundary
            while end > start and text[end] != " ":
                end -= 1

            if end == start:  # No word boundary found
                end = start + self.max_chunk_size

            chunks.append(text[start:end])
            start = end - (self.max_chunk_size // 10)  # Small overlap

        return chunks
