"""
Chunker

Provides document chunking functionality including text cleaning, token estimation, and sentence splitting.

The implementation supports:
    - Cleaning whitespace and newlines from text
    - Estimating token counts based on character length
    - Splitting text into sentences
    - Chunking documents with token limits and overlap

Key classes / functions:
    - clean: Cleans whitespace and newlines from a string.
    - estimate_tokens: Estimates the number of tokens in a string.
    - split_sentences: Splits a string into a list of sentences.
    - chunk_document: Chunks a document into smaller segments with token limits and overlap.

Version:
    - 05-Jun-2026 (Version 1.0): Initial implementation and documentation.
"""
import re
from typing import List, Dict


def clean(text: str) -> str:
    """
    Cleans whitespace and newlines from a string.

    This function removes extra spaces, replaces multiple newlines with a double
    newline, and strips leading and trailing whitespace from each line.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.

    Example:
        >>> clean("Hello   world!\\n\\n\\nHow are you?")
        'Hello world!\\n\\nHow are you?'
    """
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


def estimate_tokens(text: str) -> int:
    """
    Estimates the number of tokens in a string.

    This is a simple estimation based on the character length of the text,
    assuming an average of 4 characters per token.

    Args:
        text (str): The input text to estimate tokens for.

    Returns:
        int: The estimated number of tokens.

    Example:
        >>> estimate_tokens("Hello, world!")
        3
    """
    return len(text) // 4


def split_sentences(text: str) -> List[str]:
    """
    Splits a string into a list of sentences.

    Uses a regular expression to find sentence boundaries based on punctuation
    and capitalization rules.

    Args:
        text (str): The input text to split.

    Returns:
        List[str]: A list of sentences.

    Example:
        >>> split_sentences("Hello! How are you? I am fine.")
        ['Hello!', 'How are you?', 'I am fine.']
    """
    pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+"
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(
    doc: dict,
    max_tokens: int = 500,
    overlap: int = 50,
    min_tokens: int = 500,
    clean_text: bool = True,
) -> List[Dict]:
    """
    Chunks a document into smaller segments with token limits and overlap.

    This function processes a document dictionary containing text and optionally
    a token count. It splits the text into sentences and groups them into chunks
    that respect the maximum token limit while providing the specified overlap.

    Args:
        doc (dict): The document to chunk. Must contain a 'text' key.
        max_tokens (int): The maximum number of tokens per chunk. Default is 500.
        overlap (int): The number of overlapping tokens between chunks. Default is 50.
        min_tokens (int): The minimum number of tokens for the final chunk. Default is 500.
        clean_text (bool): Whether to clean the text before chunking. Default is True.

    Returns:
        List[Dict]: A list of chunk dictionaries, each containing the chunk text,
        token count, character range, sentence count, and document metadata.

    Example:
        >>> doc = {"id": "1", "text": "This is a long document. It has many sentences."}
        >>> chunks = chunk_document(doc, max_tokens=10)
        >>> len(chunks) > 0
        True
    """
    text = doc.get("text", "")
    if clean_text:
        text = clean(text)

    token_count = doc.get("token_count")
    if token_count is None:
        token_count = estimate_tokens(text)

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_tokens = 0
    char_pos = 0

    for sentence in sentences:
        sent_tokens = estimate_tokens(sentence)

        if current_tokens + sent_tokens > max_tokens and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_start = char_pos - len(chunk_text)

            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "text": chunk_text,
                    "token_count": current_tokens,
                    "char_range": (chunk_start, char_pos),
                    "sentence_count": len(current_chunk),
                    "doc_id": doc.get("id"),
                    "url": doc.get("url"),
                }
            )

            overlap_buffer = []
            overlap_tokens = 0

            for sent in reversed(current_chunk):
                sent_tokens = estimate_tokens(sent)
                if overlap_tokens + sent_tokens <= overlap:
                    overlap_buffer.insert(0, sent)
                    overlap_tokens += sent_tokens
                else:
                    break

            current_chunk = overlap_buffer
            current_tokens = overlap_tokens

        current_chunk.append(sentence)
        current_tokens += sent_tokens
        char_pos += len(sentence) + 1

    if current_chunk and current_tokens >= min_tokens:
        chunk_text = " ".join(current_chunk)
        chunk_start = char_pos - len(chunk_text)

        chunks.append(
            {
                "chunk_id": len(chunks),
                "text": chunk_text,
                "token_count": current_tokens,
                "char_range": (chunk_start, char_pos),
                "sentence_count": len(current_chunk),
                "doc_id": doc.get("id"),
                "url": doc.get("url"),
            }
        )
    elif current_chunk and chunks:
        last_chunk = chunks[-1]
        merged_text = last_chunk["text"] + " " + " ".join(current_chunk)
        last_chunk["text"] = merged_text
        last_chunk["token_count"] = estimate_tokens(merged_text)
        last_chunk["sentence_count"] += len(current_chunk)
        last_chunk["char_range"] = (last_chunk["char_range"][0], char_pos)

    return chunks
