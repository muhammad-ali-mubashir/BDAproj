import PyPDF2
import re
from typing import List, Dict


def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """Extracts text from PDF and returns a list of pages with text."""
    pages_content = []
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            if text.strip():
                pages_content.append({
                    "page_number": page_num + 1,
                    "text": text
                })
    return pages_content



#Basic text cleaning.

def clean_text(text: str) -> str:
    
    # Remove excessive whitespaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()







    """Splits text into meaningful chunks with overlap."""

def chunk_text(pages_content: List[Dict], chunk_size: int = 300, overlap: int = 50) -> List[Dict]:
    chunks = []
    
    # Combine all text first to handle chunks spanning across pages
    # But keep track of page numbers
    all_text_with_meta = []
    for p in pages_content:
        words = p["text"].split()
        for word in words:
            all_text_with_meta.append({"word": word, "page": p["page_number"]})
            
    i = 0
    while i < len(all_text_with_meta):
        # Take chunk_size words
        current_chunk_data = all_text_with_meta[i : i + chunk_size]
        if not current_chunk_data:
            break
            
        chunk_text = " ".join([d["word"] for d in current_chunk_data])
        # Find the primary page number (the one that appears most or the first one)
        # For simplicity, we'll take the range of pages
        pages = sorted(list(set([d["page"] for d in current_chunk_data])))
        page_range = f"{pages[0]}" if len(pages) == 1 else f"{pages[0]}-{pages[-1]}"
        
        chunks.append({
            "content": clean_text(chunk_text),
            "metadata": {
                "source": "Undergraduate-Handbook.pdf",
                "page": page_range
            }
        })
        
        # Move forward by chunk_size - overlap
        i += (chunk_size - overlap)
        
    return chunks

if __name__ == "__main__":
    pdf_path = "Undergraduate-Handbook.pdf"
    print(f"Extracting text from {pdf_path}...")
    pages = extract_text_from_pdf(pdf_path)
    print(f"Total pages extracted: {len(pages)}")
    
    print("Chunking text...")
    chunks = chunk_text(pages)
    print(f"Total chunks created: {len(chunks)}")
    
    if chunks:
        print("\nSample Chunk (first 100 characters):")
        print(chunks[0]["content"][:100] + "...")
        print(f"Metadata: {chunks[0]['metadata']}")
