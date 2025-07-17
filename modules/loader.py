import fitz
import re
from collections import Counter
import unicodedata
from modules import config

# v1 functions: extract_text_from_pdf(pdf_path = config.PDF_PATH) ; clean_text(text)

# V2 FUNCTIONS:
# NORMALIZATION
def normalize_ligatures(text: str) -> str:
    # replaces ligatures w their equivalent character
    ligatures = {
        'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
        'ﬅ': 'ft', 'ﬆ': 'st',
    }
    
    for ligature, replacement in ligatures.items():
        text = text.replace(ligature, replacement)
    return text

def rejoin_hyphenated_words(text: str) -> str:
    # rejoin broken words eg: docu- ment -> document
    return re.sub(r'([a-z])-\n([a-z])', r'\1\2', text)

def normalize_whitespace(text: str) -> str:
    # cleans up whitespace / newlines
    
    # text converted into single line
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_redundant_newlines(text: str) -> str:
    # removes multiple consecutive newlines
    return re.sub(r'\n\s*\n', '\n', text)

def normalize_unicode(text: str) -> str:
    # normalizes unicode to standard form: NKFC
    return unicodedata.normalize('NFKC', text)

# DOCUMENT STRUCTURE
def remove_headers_footers(pages_text: list[str], header_size: int = 70, footer_size: int = 70) -> list[str]:
    # identifies common headers and footers and removes them

    header_lines = []
    footer_lines = []
    for text in pages_text:
        header = text[:header_size].strip()
        footer = text[-footer_size:].strip()
        if header:
            header_lines.extend(line for line in header.split('\n') if line.strip())
        if footer:
            footer_lines.extend(line for line in footer.split('\n') if line.strip())

    # frequency of each line
    header_counts = Counter(header_lines)
    footer_counts = Counter(footer_lines)

    # if lines appear on atleast 25% of pages, consider them common
    min_occurrence = max(2, int(len(pages_text) * 0.25)) 
    common_headers = {line for line, count in header_counts.items() if count >= min_occurrence}
    common_footers = {line for line, count in footer_counts.items() if count >= min_occurrence}

    if not common_headers and not common_footers:
        return pages_text

    # clean pages
    cleaned_pages = []
    for text in pages_text:
        lines = text.split('\n')
        
        # Remove header lines
        cleaned_lines = []
        in_header = True
        for line in lines:
            if in_header and line.strip() in common_headers:
                continue
            else:
                in_header = False
                cleaned_lines.append(line)

        final_lines = []
        in_footer = True
        for line in reversed(cleaned_lines):
            if in_footer and line.strip() in common_footers:
                continue
            else:
                in_footer = False
                final_lines.append(line)

        cleaned_pages.append('\n'.join(reversed(final_lines)))

    return cleaned_pages

# PREPROCESSING PIPELINE

def clean_text_pipeline(text: str) -> str:
    text = normalize_ligatures(text)
    text = rejoin_hyphenated_words(text)
    text = normalize_unicode(text)
    text = remove_redundant_newlines(text)
    return text

def extract_and_clean_pdf(pdf_path = config.PDF_PATH) -> str:
    # main function
    print(f"Starting processing for: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        return ""
    
    # extract raw text
    raw_pages_text = [page.get_text("text") for page in doc]
    print(f"Extracted {len(raw_pages_text)} pages.")

    # clean structure: remove common headers / footers
    structurally_cleaned_pages = remove_headers_footers(raw_pages_text)
    print("Attempted to remove common headers and footers.")

    # apply final cleaning pipeline
    final_cleaned_pages = []
    for i, page_text in enumerate(structurally_cleaned_pages):
        # apply the series of cleaning functions
        processed_text = clean_text_pipeline(page_text)
        # final whitespace cleanup
        normalized_page_text = normalize_whitespace(processed_text)
        
        if normalized_page_text:
            final_cleaned_pages.append(normalized_page_text)
    
    print("Applied text normalization and cleaning pipeline to all pages.")

    # join all cleaned pages into single text block
    full_text = "\n\n".join(final_cleaned_pages)

    print("✅ Processing complete.\n")
    return full_text

