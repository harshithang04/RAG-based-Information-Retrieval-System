from pathlib import Path
from typing import Callable, Optional

import pdfplumber
import pymupdf

import config
import openrouter_client


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            boundary = text.rfind("\n", start, end)
            if boundary == -1 or boundary <= start + chunk_size // 2:
                boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + chunk_size // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def table_row_to_line(row: list[Optional[str]]) -> str:
    """Compact a table row to 'Label: v1 | v2 | ...', dropping empty and '$'/'%' filler cells."""
    cells = [(c or "").replace("\xa0", " ").strip() for c in row]
    cells = [c for c in cells if c and c not in ("$", "%", ")")]
    if not cells:
        return ""
    label, values = cells[0], cells[1:]
    return f"{label}: " + " | ".join(values) if values else label


def table_to_chunks(
    rows: list[list[Optional[str]]], context: str, page_num: int, max_chars: int
) -> list[str]:
    """Split a table into row-group chunks that each fit the embedder's window.

    Every chunk is prefixed with the text above the table (statement title, column
    headers such as 'Three Months Ended June 25, 2022') so a row is interpretable alone.
    """
    lines = [line for line in (table_row_to_line(r) for r in rows) if line]
    header = f"Table on page {page_num}. {context}".strip()
    chunks, current, size = [], [], 0
    for line in lines:
        if current and size + len(line) > max_chars:
            chunks.append(header + "\n" + "\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append(header + "\n" + "\n".join(current))
    return chunks


def table_context(plumber_page, table) -> str:
    """Whitespace-collapsed text just above a table, cut on a word boundary."""
    top = max(table.bbox[1], 1)
    above = plumber_page.crop((0, 0, plumber_page.width, top)).extract_text() or ""
    context = " ".join(above.split())
    limit = config.TABLE_CONTEXT_CHARS
    if len(context) > limit:
        context = context[-limit:]
        context = context[context.find(" ") + 1 :]
    return context


def find_table_chunks(plumber_page, tables: list, page_num: int) -> list[str]:
    chunks: list[str] = []
    for table in tables:
        chunks.extend(
            table_to_chunks(
                table.extract(), table_context(plumber_page, table), page_num, config.TABLE_CHUNK_CHARS
            )
        )
    return chunks


def ingest_pdf(
    pdf_path: Path,
    api_key: str,
    vision_model: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """Parse a PDF into text/table/figure chunks with page-level metadata."""
    pdf_path = Path(pdf_path)
    source_name = pdf_path.name
    docs: list[dict] = []

    doc = pymupdf.open(pdf_path)
    plumber_pdf = pdfplumber.open(pdf_path)
    num_pages = len(doc)

    for page_index in range(num_pages):
        page = doc[page_index]
        page_num = page_index + 1
        if progress_cb:
            progress_cb(f"Reading page {page_num}/{num_pages}")

        text = page.get_text("text")
        for chunk in chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
            docs.append(
                {
                    "type": "text",
                    "page": page_num,
                    "source": source_name,
                    "content": chunk,
                }
            )

        plumber_page = plumber_pdf.pages[page_index]
        try:
            table_chunks = find_table_chunks(plumber_page, plumber_page.find_tables(), page_num)
        except Exception:
            table_chunks = []
        for content in table_chunks:
            docs.append(
                {
                    "type": "table",
                    "page": page_num,
                    "source": source_name,
                    "content": content,
                }
            )

        seen_xrefs = set()
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue
            width, height = base_image.get("width", 0), base_image.get("height", 0)
            if width < config.MIN_FIGURE_DIM or height < config.MIN_FIGURE_DIM:
                continue

            ext = base_image.get("ext", "png")
            fname = f"{pdf_path.stem}_p{page_num}_img{img_idx}.{ext}"
            fpath = config.FIGURES_DIR / fname
            fpath.write_bytes(base_image["image"])

            if api_key:
                if progress_cb:
                    progress_cb(f"Captioning figure on page {page_num}")
                try:
                    caption = openrouter_client.caption_image(
                        fpath, vision_model, api_key, context_hint=text
                    )
                except Exception as e:
                    caption = f"Figure on page {page_num} (captioning failed: {e})"
            else:
                caption = f"Figure on page {page_num} (no API key set - caption unavailable)"

            docs.append(
                {
                    "type": "figure",
                    "page": page_num,
                    "source": source_name,
                    "content": caption,
                    "image_path": str(fpath),
                }
            )

    plumber_pdf.close()
    doc.close()
    return docs
