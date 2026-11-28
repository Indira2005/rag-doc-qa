from rag.chunking import chunk_pages
from rag.fusion import rrf_merge
from rag.generate import citations, should_refuse


def test_chunks_keep_page_and_section():
    pages = [{"page": 1, "chunk_type": "text", "text": "3 Model Architecture\nSome text."},
             {"page": 2, "chunk_type": "text", "text": "More text."},
             {"page": 2, "chunk_type": "table", "text": "| a |\n| --- |"}]
    chunks = chunk_pages(pages, "x.pdf", 500, 50)
    assert [c["page"] for c in chunks] == [1, 2, 2]
    assert all(c["section"] == "3 Model Architecture" for c in chunks)
    assert chunks[-1]["type"] == "table"


def test_rrf_prefers_chunk_found_by_both():
    assert rrf_merge([{"id": "a"}, {"id": "b"}], [{"id": "b"}])[0]["id"] == "b"


def test_refusal_and_citations(monkeypatch):
    monkeypatch.setattr("rag.config.MIN_SCORE", 0.05)
    hit = {"source": "x.pdf", "page": 4, "section": "", "score": 0.9}
    assert should_refuse([]) and should_refuse([{**hit, "score": 0.001}])
    assert not should_refuse([hit])
    assert citations([hit])[0]["page"] == 4


def test_resume_headings_split_sections():
    text = "Name\nSKILLS\nPython\nPROJECTS\nBuilt a bot.\nEDUCATION\nB.Tech"
    chunks = chunk_pages([{"page": 1, "chunk_type": "text", "text": text}], "r.pdf", 500, 50)
    assert [c["section"] for c in chunks] == ["", "SKILLS", "PROJECTS", "EDUCATION"]
    assert chunks[-1]["text"].startswith("EDUCATION")


def test_tightly_spaced_words_stay_separate(tmp_path):
    from reportlab.pdfgen import canvas
    from rag.pdf_extract import extract_pdf_with_tables
    path = str(tmp_path / "tight.pdf")
    pdf = canvas.Canvas(path)
    pdf.setFont("Times-Roman", 10)
    x = 72
    for word in ["Encoder", "and", "Decoder", "Stacks"]:
        pdf.drawString(x, 700, word)
        x += pdf.stringWidth(word, "Times-Roman", 10) + 2.4  # narrow gap, as in LaTeX PDFs
    pdf.save()
    assert "Encoder and Decoder Stacks" in extract_pdf_with_tables(path)[0]["text"]
