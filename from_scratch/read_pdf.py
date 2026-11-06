from load_pdf import load_pdf

pages = load_pdf("data/attention_is_all_you_need.pdf")
print(len(pages))
print(pages[4]["text"][:200])