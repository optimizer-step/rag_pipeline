"""Split document pages into smaller retrievable chunks."""

from pathlib import Path
from typing import Any

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from .config import CHUNK_OVERLAP, CHUNK_SIZE


def create_text_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Create and configure the recursive text splitter."""

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )


def chunk_pages(
    pages: list[dict[str, Any]],
    splitter: RecursiveCharacterTextSplitter,
) -> list[dict[str, Any]]:
    """Split every page while preserving its metadata."""

    if not pages:
        raise ValueError(
            "The pages list cannot be empty."
        )

    all_chunks = []

    for page in pages:
        page_documents = splitter.create_documents(
            texts=[page["text"]],
            metadatas=[page["metadata"]],
        )

        for chunk_index, document in enumerate(
            page_documents,
            start=1,
        ):
            page_number = int(
                document.metadata["page_number"]
            )

            source_name = document.metadata["source"]
            source_stem = Path(source_name).stem

            chunk_id = (
                f"{source_stem}"
                f"-p{page_number:02d}"
                f"-c{chunk_index:02d}"
            )

            metadata = dict(document.metadata)

            metadata["chunk_index_on_page"] = (
                chunk_index
            )
            metadata["chunk_id"] = chunk_id
            metadata["character_count"] = len(
                document.page_content
            )

            all_chunks.append(
                {
                    "text": (
                        document.page_content.strip()
                    ),
                    "metadata": metadata,
                }
            )

    if not all_chunks:
        raise ValueError(
            "No chunks were created."
        )

    return all_chunks
