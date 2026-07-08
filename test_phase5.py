import os
from dotenv import load_dotenv
from src.db.database import SessionLocal
from src.retrieval.hybrid_search import search_with_self_query

load_dotenv()

def run_test():
    db = SessionLocal()
    
    print("==================================================")
    print("PHASE 5: SELF-QUERYING RETRIEVAL TEST")
    print("==================================================")
    
    from src.db.models import DocumentChunk
    latest_chunk = db.query(DocumentChunk).order_by(DocumentChunk.created_at.desc()).first()
    if not latest_chunk:
        print("No documents found in database.")
        return
    document_id = str(latest_chunk.document_id)
    
    print(f"\n[INFO] Testing with Document ID: {document_id}")
    
    # Test 1: Trend question
    q1 = "what was the revenue trend"
    print(f"\n[TEST 1] Asking: {q1}")
    results1 = search_with_self_query(q1, document_id, top_k=3)
    
    if results1:
        print("Top results:")
        for i, r in enumerate(results1, 1):
            print(f" {i}. Type: {r['chunk_type']} | Score: {r['score']:.4f} | Page: {r['page_number']}")
            
    # Test 2: General question
    q2 = "How many employees does the company have?"
    print(f"\n[TEST 2] Asking: {q2}")
    results2 = search_with_self_query(q2, document_id, top_k=3)
    
    if results2:
        print("Top results:")
        for i, r in enumerate(results2, 1):
            print(f" {i}. Type: {r['chunk_type']} | Score: {r['score']:.4f} | Page: {r['page_number']}")
            
    db.close()

if __name__ == "__main__":
    run_test()
