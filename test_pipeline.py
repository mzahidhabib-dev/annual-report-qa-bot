import os
from src.ingestion.text_extractor import extract_text_by_page
from src.embeddings.chunker import chunk_text

def run_test():
    pdf_path = os.path.join("sample_docs", "Northbridge_Dynamics_2025_Annual_Report.pdf")
    
    print(f"==========================================")
    print(f"STEP 1.1: Extracting Text")
    print(f"==========================================")
    pages = extract_text_by_page(pdf_path)
    print(f"✅ Successfully read {len(pages)} pages of text.\n")
    
    if pages:
        print("Here is a snippet from Page 1:")
        print("------------------------------------------")
        print(pages[0]["text"][:500] + "...")
        print("------------------------------------------\n")
    
    print(f"==========================================")
    print(f"STEP 1.2: Chunking Text")
    print(f"==========================================")
    chunks = chunk_text(pages, chunk_size=800, overlap=150)
    print(f"✅ Successfully split the document into {len(chunks)} overlapping chunks.\n")
    
    if chunks:
        print("Here are the first two chunks generated:")
        for i, chunk in enumerate(chunks[:2]):
            print(f"\n--- CHUNK {i+1} (Page {chunk['page_number']}, Size: {len(chunk['chunk_text'])} chars) ---")
            print(chunk["chunk_text"])
            
if __name__ == "__main__":
    run_test()
