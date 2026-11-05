import unicodedata
from pypdf import PdfReader 

def load_pdf(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        cleaned_text = unicodedata.normalize("NFKC", text)
        pages.append({"page": i+1, "text": cleaned_text})

    return pages 