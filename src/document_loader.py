"""Load a PDF page by page while preserving metadata."""

from pathlib import Path
from typing import Any

import pymupdf


def load_pdf_pages(
    pdf_path: str | Path,
) -> list[dict[str, Any]]:
    """Extract readable text and metadata from every PDF page."""

    pdf_path = Path(pdf_path).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            "The supplied file must be a PDF."
        )

    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(
            document,
            start=1,
        ):
            page_text = page.get_text(
                "text",
                sort=True,
            ).strip()

            if not page_text:
                continue

            pages.append(
                {
                    "text": page_text,
                    "metadata": {
                        "source": pdf_path.name,
                        "page_number": page_index,
                    },
                }
            )

    if not pages:
        raise ValueError(
            "No readable text was extracted. "
            "The PDF may require OCR."
        )

    return pages
