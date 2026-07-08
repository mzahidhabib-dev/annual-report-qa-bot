from sqlalchemy.orm import Session
from src.retrieval.searcher import search_with_self_query, get_genai_client
from src.db.models import QueryLog, RetrievalMethod

def generate_answer(query: str, db_session: Session) -> str:
    """
    Retrieves relevant chunks using hybrid search, passes them to Gemini,
    and returns a conversational answer. Logs the transaction to the database.
    """
    # 1. Retrieve the top 5 chunks using self_query (Phase 5)
    chunks = search_with_self_query(query, db_session, top_k=5)
    
    if not chunks:
        return "I could not find any relevant information in the annual report to answer your question."
        
    # 2. Combine the context
    context_blocks = []
    chunk_ids = []
    for i, chunk_dict in enumerate(chunks, 1):
        context_blocks.append(f"--- Document Excerpt {i} (Page {chunk_dict['page_number']}) ---\n{chunk_dict['content']}")
        chunk_ids.append(chunk_dict['chunk_id'])

        
    context_str = "\n\n".join(context_blocks)
    
    # 3. Prompting Gemini
    client = get_genai_client()
    
    prompt = f"""You are a professional financial AI assistant analyzing an annual report.
    
Your task is to answer the user's question using strictly the provided context below.
If the answer is not contained within the context, you must explicitly state: "I don't know based on the provided document." Do not hallucinate external information.

Always cite your sources by referencing the excerpt or page number from the context (e.g., "According to Excerpt 1 on Page 5...").

CONTEXT:
{context_str}

USER QUESTION:
{query}
"""
    
    # 4. Generate Answer
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )
    
    answer_text = response.text
    
    # 5. Log the query for future dashboard accuracy tracking
    try:
        log_entry = QueryLog(
            question=query,
            retrieved_chunk_ids=chunk_ids,
            answer=answer_text,
            retrieval_method=RetrievalMethod.hybrid
        )
        db_session.add(log_entry)
        db_session.commit()
    except Exception as e:
        print(f"Warning: Failed to log query to database: {e}")
        db_session.rollback()
        
    return answer_text
