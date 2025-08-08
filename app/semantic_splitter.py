"""Enhanced semantic text splitter with table-aware chunking.

This enhanced version of the semantic splitter can detect and handle
tables specially, preserving table structure and context.
"""

import re
from typing import List

import numpy as np
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


class TableAwareSemanticSplitter:
    """Enhanced semantic text splitter with table awareness.

    This splitter can identify tables in markdown content and handle them
    specially to preserve their structure and ensure they remain intact
    within chunks when possible.
    """

    def __init__(
        self,
        embeddings: HuggingFaceEndpointEmbeddings,
        similarity_threshold: float = 0.85,
        min_chunk_size: int = 200,
        max_chunk_size: int = 1500,
        sentence_overlap: int = 1,
        table_preserve_structure: bool = True,
        table_min_chunk_size: int = 100,
    ):
        """Initialize the TableAwareSemanticSplitter.

        Args:
            embeddings: The embedding model to use for semantic analysis.
            similarity_threshold: Similarity threshold for grouping sentences.
            min_chunk_size: Minimum chunk size in characters.
            max_chunk_size: Maximum chunk size in characters.
            sentence_overlap: Number of sentences to overlap between chunks.
            table_preserve_structure:
                Whether to keep tables intact when possible.
            table_min_chunk_size:
                Minimum chunk size for table-containing chunks.
        """
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.sentence_overlap = sentence_overlap
        self.table_preserve_structure = table_preserve_structure
        self.table_min_chunk_size = table_min_chunk_size

    def split_text(self, text: str) -> List[str]:
        """Split text into semantically coherent chunks with table awareness.

        Args:
            text: The input text to be split.

        Returns:
            A list of text chunks.
        """
        text_segments = self._segment_text_with_tables(text)

        chunks = []
        for segment in text_segments:
            if segment["type"] == "table":
                table_chunks = self._process_table_segment(segment)
                chunks.extend(table_chunks)
            else:
                regular_chunks = self._process_regular_text(
                    segment["content"]
                )
                chunks.extend(regular_chunks)

        return chunks

    def _segment_text_with_tables(self, text: str) -> List[dict]:
        """Segment text into table and non-table parts.

        Args:
            text: Input text to segment.

        Returns:
            List of segments with type and content.
        """
        segments = []

        table_pattern = (
            r"(\|.*\|.*\n(?:\|.*\|.*\n)*(?:\|[-:\s|]*\|.*\n)?(?:\|.*\|.*\n)*)"
        )

        matches = list(re.finditer(table_pattern, text, re.MULTILINE))

        if not matches:
            return [{"type": "text", "content": text}]

        last_end = 0

        for match in matches:
            if match.start() > last_end:
                pre_table_text = text[last_end : match.start()].strip()
                if pre_table_text:
                    segments.append({
                        "type": "text",
                        "content": pre_table_text,
                    })

            table_content = match.group(1).strip()
            segments.append({
                "type": "table",
                "content": table_content,
                "start": match.start(),
                "end": match.end(),
            })

            last_end = match.end()

        if last_end < len(text):
            post_table_text = text[last_end:].strip()
            if post_table_text:
                segments.append({"type": "text", "content": post_table_text})

        return segments

    def _process_table_segment(self, segment: dict) -> List[str]:
        """Process a table segment.

        Args:
            segment: Dictionary containing table information.

        Returns:
            List of chunks containing the table.
        """
        table_content = segment["content"]

        if self.table_preserve_structure:
            if len(table_content) <= self.max_chunk_size:
                return [table_content]

            return self._split_large_table(table_content)
        else:
            return self._process_regular_text(table_content)

    def _split_large_table(self, table_content: str) -> List[str]:
        """Split a large table while preserving structure.

        Args:
            table_content: The table content to split.

        Returns:
            List of table chunks.
        """
        lines = table_content.split("\n")

        header_lines = []
        separator_line = None
        data_lines = []

        in_data = False
        for i, line in enumerate(lines):
            if "|" in line:
                if not in_data and ("---" in line or "::" in line):
                    separator_line = line
                    in_data = True
                elif not in_data:
                    header_lines.append(line)
                else:
                    data_lines.append(line)

        if not header_lines or not separator_line:
            return self._fallback_split(table_content)

        chunks = []
        header_block = "\n".join(header_lines + [separator_line])

        current_chunk_lines = []
        current_length = len(header_block)

        for line in data_lines:
            line_length = len(line) + 1

            if (
                current_length + line_length > self.max_chunk_size
                and current_chunk_lines
            ):
                chunk = header_block + "\n" + "\n".join(current_chunk_lines)
                chunks.append(chunk)

                current_chunk_lines = [line]
                current_length = len(header_block) + line_length
            else:
                current_chunk_lines.append(line)
                current_length += line_length

        if current_chunk_lines:
            chunk = header_block + "\n" + "\n".join(current_chunk_lines)
            chunks.append(chunk)

        return chunks

    def _process_regular_text(self, text: str) -> List[str]:
        """Process regular (non-table) text with semantic splitting.

        Args:
            text: Regular text content.

        Returns:
            List of text chunks.
        """
        sentences = self._split_into_sentences(text)

        if len(sentences) <= 1:
            return [text]

        try:
            embeddings = self._get_sentence_embeddings(sentences)
        except Exception as e:
            print(
                f"Warning: Embedding failed ({e}), "
                f"falling back to simple splitting"
            )
            return self._fallback_split(text)

        chunks = self._create_semantic_chunks(sentences, embeddings)

        final_chunks = self._post_process_chunks(chunks)

        return final_chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        text = re.sub(r"\s+", " ", text.strip())

        sentence_pattern = r"""
            (?<!\w\.\w.)
            (?<![A-Z][a-z]\.)
            (?<!\d\.)
            [.!?]+
            \s+
            (?=[A-Z])
        """

        sentences = re.split(sentence_pattern, text, flags=re.VERBOSE)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            sentences = [s.strip() for s in text.split("\n") if s.strip()]

        return sentences

    def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Get embeddings for a list of sentences."""
        meaningful_sentences = [s for s in sentences if len(s.split()) >= 3]

        if not meaningful_sentences:
            meaningful_sentences = sentences

        embeddings = self.embeddings.embed_documents(meaningful_sentences)
        return np.array(embeddings)

    def _create_semantic_chunks(
        self, sentences: List[str], embeddings: np.ndarray
    ) -> List[str]:
        """Group sentences into semantic chunks based on similarity."""
        if len(sentences) != len(embeddings):
            return self._simple_group_sentences(sentences)

        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_chunk_embeddings = [embeddings[0]]

        for i in range(1, len(sentences)):
            current_embedding = embeddings[i].reshape(1, -1)
            chunk_embedding = np.mean(
                current_chunk_embeddings, axis=0
            ).reshape(1, -1)
            similarity = cosine_similarity(
                current_embedding, chunk_embedding
            )[0][0]

            current_chunk_text = " ".join(
                current_chunk_sentences + [sentences[i]]
            )
            would_exceed_max = len(current_chunk_text) > self.max_chunk_size

            if (
                similarity >= self.similarity_threshold
                and not would_exceed_max
            ):
                current_chunk_sentences.append(sentences[i])
                current_chunk_embeddings.append(embeddings[i])
            else:
                if current_chunk_sentences:
                    chunks.append(" ".join(current_chunk_sentences))

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

        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks

    def _simple_group_sentences(self, sentences: List[str]) -> List[str]:
        """Fallback method to group sentences without semantic analysis."""
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if (
                current_length + sentence_length > self.max_chunk_size
                and current_chunk
            ):
                chunks.append(" ".join(current_chunk))

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

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _post_process_chunks(self, chunks: List[str]) -> List[str]:
        """Post-process chunks to ensure they meet size requirements."""
        processed_chunks = []

        for chunk in chunks:
            if len(chunk) < self.min_chunk_size and len(chunks) > 1:
                if (
                    processed_chunks
                    and len(processed_chunks[-1] + " " + chunk)
                    <= self.max_chunk_size
                ):
                    processed_chunks[-1] = processed_chunks[-1] + " " + chunk
                    continue
                continue

            if len(chunk) > self.max_chunk_size:
                sub_chunks = self._split_large_chunk(chunk)
                processed_chunks.extend(sub_chunks)
            else:
                processed_chunks.append(chunk)

        return processed_chunks

    def _split_large_chunk(self, chunk: str) -> List[str]:
        """Split a chunk that exceeds maximum size."""
        words = chunk.split()
        sub_chunks = []
        current_sub_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1

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
        """Fallback splitting method when embeddings fail."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.max_chunk_size

            if end >= len(text):
                chunks.append(text[start:])
                break

            while end > start and text[end] != " ":
                end -= 1

            if end == start:
                end = start + self.max_chunk_size

            chunks.append(text[start:end])
            start = end - (self.max_chunk_size // 10)

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split a list of documents into smaller chunks."""
        split_docs = []

        for doc in documents:
            chunks = self.split_text(doc.page_content)

            for i, chunk in enumerate(chunks):
                chunk_metadata = doc.metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(chunks)

                has_table = "|" in chunk and ("---" in chunk or "::" in chunk)
                chunk_metadata["contains_table"] = has_table

                split_doc = Document(
                    page_content=chunk, metadata=chunk_metadata
                )
                split_docs.append(split_doc)

        return split_docs
