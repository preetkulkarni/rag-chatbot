import fitz
import re
from collections import Counter
import unicodedata
from typing import List, Dict, Any, Tuple
from . import config

# NORMALIZATION
def normalize_ligatures(text: str) -> str:
    ligatures = {'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl', 'ﬅ': 'ft', 'ﬆ': 'st'}
    for ligature, replacement in ligatures.items():
        text = text.replace(ligature, replacement)
    return text

def rejoin_hyphenated_words(text: str) -> str:
    return re.sub(r'([a-z])-\n([a-z])', r'\1\2', text)

def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def remove_redundant_newlines(text: str) -> str:
    return re.sub(r'\n\s*\n', '\n', text)

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize('NFKC', text)

# DOCUMENT STRUCTURE    
def remove_headers_footers(pages_text: List[str]) -> Tuple[List[str], str]:
    """
    Identifies common headers/footers using a weighted scoring model,
    removes them, and returns the cleaned pages and document context.
    """
    def normalize_line_for_comparison(line: str) -> str:
        line_no_digits = re.sub(r'\d+', '', line)
        return line_no_digits.lower().strip()

    # --- NEW: Define weights based on line position ---
    # The first line is most likely a header, the 5th line is less likely.
    header_weights = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    footer_weights = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # Reversed for footers
    
    # --- Step 1: Identify potential common lines ---
    # We store the line and its position (weight index)
    potential_headers = []
    potential_footers = []
    for text in pages_text:
        lines = [line for line in text.split('\n') if line.strip()]
        if not lines:
            continue
            
        # Check top lines for headers
        for i in range(min(len(lines), len(header_weights))):
            normalized_line = normalize_line_for_comparison(lines[i])
            potential_headers.append((normalized_line, i)) # Store (line, position_index)
            
        # Check bottom lines for footers
        for i in range(min(len(lines), len(footer_weights))):
            line_index_from_bottom = len(lines) - 1 - i
            normalized_line = normalize_line_for_comparison(lines[line_index_from_bottom])
            potential_footers.append((normalized_line, i)) # Store (line, position_index)

    header_counts = Counter(potential_headers)
    footer_counts = Counter(potential_footers)

    # --- Step 2: Determine common lines using weighted threshold ---
    base_min_occurrence = int(len(pages_text) * 0.40) # Base threshold: 40%
    if base_min_occurrence < 2 and len(pages_text) > 1:
        base_min_occurrence = 2

    common_headers = set()
    for (line, pos_index), count in header_counts.items():
        weight = header_weights[pos_index]
        # A line further down the page (lower weight) needs to appear more often
        if count >= (base_min_occurrence / weight):
            common_headers.add(line)

    common_footers = set()
    for (line, pos_index), count in footer_counts.items():
        weight = footer_weights[pos_index]
        if count >= (base_min_occurrence / weight):
            common_footers.add(line)

    # --- Step 3: Extract first header and clean all pages ---
    first_header_instance = []
    cleaned_pages = []
    is_first_header_captured = False

    for text in pages_text:
        lines = text.split('\n')
        
        # Find end of header section
        header_end_index = 0
        current_header_lines = []
        for i, line in enumerate(lines):
            normalized_line = normalize_line_for_comparison(line)
            if line.strip() and normalized_line in common_headers:
                current_header_lines.append(line)
            # Stop if we hit a line that is not a common header
            elif line.strip() and normalized_line not in common_headers:
                header_end_index = i
                break
        
        if not is_first_header_captured and current_header_lines:
            first_header_instance = current_header_lines
            is_first_header_captured = True

        # Find start of footer section (from the bottom up)
        footer_start_index = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            normalized_line = normalize_line_for_comparison(lines[i])
            if line.strip() and normalized_line not in common_footers:
                footer_start_index = i + 1
                break
        
        # Combine the lines that are between the header and footer
        if header_end_index < footer_start_index:
            cleaned_page_lines = lines[header_end_index:footer_start_index]
            cleaned_pages.append('\n'.join(cleaned_page_lines))
        else:
            cleaned_pages.append(text) # Failsafe
        
    doc_context = '\n'.join(first_header_instance)
    return cleaned_pages, doc_context


def clean_text_pipeline(text: str) -> str:
    text = normalize_ligatures(text)
    text = rejoin_hyphenated_words(text)
    text = normalize_unicode(text)
    text = remove_redundant_newlines(text)
    return text

def extract_and_clean_pdf(pdf_path) -> Dict[str, Any]:
    print(f"\nStarting processing for: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        return {}
    
    raw_pages_text = [page.get_text("text") for page in doc]
    print(f"✅ Extracted {len(raw_pages_text)} pages.")

    structurally_cleaned_pages, doc_context = remove_headers_footers(raw_pages_text)
    print("✅ Identified document context and removed headers/footers using weighted scoring.")

    final_pages_data = []
    for i, page_text in enumerate(structurally_cleaned_pages):
        page_num = i + 1
        processed_text = clean_text_pipeline(page_text)
        normalized_page_text = normalize_whitespace(processed_text)
        
        if normalized_page_text:
            final_pages_data.append({
                "page_number": page_num,
                "text": normalized_page_text
            })
    
    print("✅ Applied text normalization and cleaning pipeline to all pages.")
    
    result = {
        "doc_context": clean_text_pipeline(doc_context),
        "pages": final_pages_data
    }
    
    print("✅ Processing complete.\n")
    return result
