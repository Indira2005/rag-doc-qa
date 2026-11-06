from load_pdf import load_pdf 

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size 
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap 
    return chunks

def chunk_pages(pages, source):
    all_chunks = []
    for page in pages:
        pieces = chunk_text(page["text"])
        for piece in pieces:
            all_chunks.append({
                "text": piece,
                "page": page["page"],
                "source": source 
            })
    return all_chunks

if __name__ == "__main__":
    pages = load_pdf("data/attention_is_all_you_need.pdf")
    chunks = chunk_pages(pages, "data/attention_is_all_you_need.pdf")
    print(len(chunks))
    print(chunks[0])
    print(chunks[-1])
    