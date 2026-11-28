from reportlab.pdfgen import canvas

from rag.pdf_extract import extract_pdf_with_tables


def test_two_columns_under_three_author_blocks(tmp_path):
    path = str(tmp_path / "paper.pdf")
    c = canvas.Canvas(path, pagesize=(612, 792))
    for row in range(8):  # three author blocks create false column boundaries
        for x in (70, 255, 440):
            c.drawString(x, 770 - row * 12, f"author {row}")
    for row in range(30):  # two columns of body text
        y = 650 - row * 14
        c.setFont("Helvetica", 10)
        c.drawString(63, y, f"LEFT{row:02d} " + "alpha beta gamma delta epsilon zeta eta")
        c.drawString(320, y, f"RIGHT{row:02d} " + "one two three four five six seven")
    c.save()
    text = " ".join(p["text"] for p in extract_pdf_with_tables(path))
    assert text.index("LEFT29") < text.index("RIGHT00")
