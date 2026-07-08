import os
from dotenv import load_dotenv
from src.db.database import SessionLocal
from src.retrieval.searcher import search_with_self_query

load_dotenv()

def run_test():
    db = SessionLocal()
    
    print("==================================================")
    print("PHASE 5: SELF-QUERYING RETRIEVAL TEST")
    print("==================================================")
    
    # Test 1: Trend question
    # Deliverable check: Asking "what was the revenue trend" returns table/image chunks ranked higher than plain text chunks
    q1 = "what was the revenue trend"
    print(f"\n[TEST 1] Asking: {q1}")
    results1 = search_with_self_query(q1, db, top_k=3)
    
    if results1:
        print("Top results:")
        for i, r in enumerate(results1, 1):
            print(f" {i}. Type: {r['chunk_type']} | Score: {r['score']:.4f} | Page: {r['page_number']}")
            
    # Test 2: General question
    # Deliverable check: Asking a general question still returns reasonable results
    q2 = "How many employees does the company have?"
    print(f"\n[TEST 2] Asking: {q2}")
    results2 = search_with_self_query(q2, db, top_k=3)
    
    if results2:
        print("Top results:")
        for i, r in enumerate(results2, 1):
            print(f" {i}. Type: {r['chunk_type']} | Score: {r['score']:.4f} | Page: {r['page_number']}")
            
    db.close()

if __name__ == "__main__":
    run_test()
