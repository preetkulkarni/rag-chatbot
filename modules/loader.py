import fitz
from modules import config

def extract_text_from_pdf(pdf_path = config.PDF_PATH):
    # pdf_path (path): path to pdf file
    # returns string with cleaned full text from PDF

    doc = fitz.open(pdf_path)
    all_text = []

    for page_num, page in enumerate(doc, start=1):
        raw_text = page.get_text("text")
        cleaned_text = clean_text(raw_text)
        all_text.append(cleaned_text)
    
    return "\n".join(all_text)

def clean_text(text):
    lines = text.splitlines()
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    return " ".join(cleaned_lines)
