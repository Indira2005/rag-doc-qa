"""Structure-aware chunking: split on section headings, keep tables whole, track source and page."""
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.pdf_extract import extract_pdf_with_tables

# Numbered headings such as "3 Model Architecture" or "3.2.1 Scaled Dot-Product Attention",
# plus a few common unnumbered ones. A heuristic: it will not catch every style.
HEADING = re.compile(
    r"^(?:\d{1,2}(?:\.\d{1,2}){0,2}\.?\s+[A-Z][^\n]{2,70}(?<![.,:;])|Abstract|References|Conclusion"
    r"|(?:[A-Z]+ )*(?:EDUCATION|SKILLS|PROJECTS|EXPERIENCE|CERTIFICATIONS|ACHIEVEMENTS|SUMMARY|PUBLICATIONS)"
    r"(?: (?:AND|&) [A-Z]+)?)$"
)


def split_sections(text: str, current: str):
    """Return ([(section, body)], last_section). The section carries over to the next page."""
    parts, lines = [], []
    for line in text.splitlines():
        if HEADING.match(line.strip()):
            if lines:
                parts.append((current, "\n".join(lines)))
            current, lines = line.strip(), [line]
        else:
            lines.append(line)
    if lines:
        parts.append((current, "\n".join(lines)))
    return parts, current


def chunk_pages(pages, source: str, size: int, overlap: int):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks, section = [], ""

    def add(text, page, sec, kind):
        if sec and not text.startswith(sec):
            text = f"{sec}\n{text}"  # so search can see which section this belongs to
        chunks.append({"id": f"{source}:{len(chunks)}", "text": text, "source": source,
                       "page": page, "section": sec, "type": kind})

    for page in pages:
        if page["chunk_type"] == "table":
            add(page["text"], page["page"], section, "table")
            continue
        parts, section = split_sections(page["text"], section)
        for sec, body in parts:
            for piece in splitter.split_text(body):
                add(piece, page["page"], sec, "text")
    return chunks


def chunk_pdf(path: str, source: str, size: int, overlap: int):
    return chunk_pages(extract_pdf_with_tables(path), source, size, overlap)
