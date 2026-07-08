import os
from sqlalchemy.orm import Session
from src.db.models import DocumentChunk
from google import genai
from rank_bm25 import BM25Okapi

# We must cache the client to avoid re-initializing it on every search query
_client = None
def get_genai_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _client

def embed_query(query: str) -> list[float]:
    """Converts a user query into a 768-d vector using Gemini."""
    client = get_genai_client()
    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=query,
        config={"task_type": "RETRIEVAL_QUERY", "output_dimensionality": 768}
    )
    return result.embeddings[0].values

def semantic_search(query: str, db_session: Session, document_id: str = None, top_k: int = 5) -> list[DocumentChunk]:
    """Finds chunks using cosine similarity via pgvector."""
    query_vector = embed_query(query)
    
    q = db_session.query(DocumentChunk)
    if document_id:
        q = q.filter(DocumentChunk.document_id == document_id)
        
    results = q.order_by(
        DocumentChunk.embedding.cosine_distance(query_vector)
    ).limit(top_k).all()
    
    return results

def bm25_search(query: str, db_session: Session, document_id: str = None, top_k: int = 5) -> list[DocumentChunk]:
    """Finds chunks using exact keyword frequency (BM25 algorithm)."""
    q = db_session.query(DocumentChunk)
    if document_id:
        q = q.filter(DocumentChunk.document_id == document_id)
    all_chunks = q.all()
    
    if not all_chunks:
        return []
        
    corpus = [chunk.content.lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(corpus)
    
    tokenized_query = query.lower().split()
    doc_scores = bm25.get_scores(tokenized_query)
    
    scored_chunks = [(chunk, score) for chunk, score in zip(all_chunks, doc_scores) if score > 0]
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    return [chunk for chunk, score in scored_chunks[:top_k]]

def hybrid_search(query: str, db_session: Session, document_id: str = None, top_k: int = 5) -> list[DocumentChunk]:
    """Combines semantic search and BM25 search mathematically using Reciprocal Rank Fusion."""
    pool_size = max(top_k * 3, 20)
    
    semantic_results = semantic_search(query, db_session, document_id, top_k=pool_size)
    bm25_results = bm25_search(query, db_session, document_id, top_k=pool_size)
    
    k = 60
    rrf_scores = {}
    
    for rank, chunk in enumerate(semantic_results, 1):
        if chunk.id not in rrf_scores:
            rrf_scores[chunk.id] = {"chunk": chunk, "score": 0.0}
        rrf_scores[chunk.id]["score"] += 1.0 / (k + rank)
        
    for rank, chunk in enumerate(bm25_results, 1):
        if chunk.id not in rrf_scores:
            rrf_scores[chunk.id] = {"chunk": chunk, "score": 0.0}
        rrf_scores[chunk.id]["score"] += 1.0 / (k + rank)
        
    sorted_chunks = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    
    # Return list of dictionaries as required by the roadmap
    final_results = []
    for item in sorted_chunks[:top_k]:
        chunk = item["chunk"]
        final_results.append({
            "chunk_id": str(chunk.id),
            "content": chunk.content,
            "page_number": chunk.page_number,
            "chunk_type": chunk.chunk_type.value,
            "score": item["score"]
        })
        
    return final_results

from src.retrieval.self_query import rewrite_query

def search_with_self_query(user_question: str, db_session: Session, document_id: str = None, top_k: int = 10) -> list[dict]:
    """
    Rewrites the question via Gemini, runs hybrid search, and boosts scores of the preferred chunk type.
    """
    # 1. Call rewrite_query
    query_data = rewrite_query(user_question)
    search_query = query_data.get("search_query", user_question)
    preferred_type = query_data.get("preferred_chunk_type", "any")
    
    print(f"\n[DEBUG] Original: '{user_question}'\n[DEBUG] Rewritten: '{search_query}' (Prefers: {preferred_type})")
    
    # 2. Call hybrid_search
    results = hybrid_search(search_query, db_session, document_id, top_k=20)
    
    # 3. Boost scores if preferred_chunk_type != "any"
    if preferred_type != "any":
        for result in results:
            if result["chunk_type"] == preferred_type:
                result["score"] *= 1.3
                
    # 4. Re-sort by adjusted score, return top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
