import time
import os
from dotenv import load_dotenv
from google import genai
from sqlalchemy.orm import Session
from src.db.models import DocumentChunk
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

api_key = os.environ.get("GEMINI_API_KEY", "dummy_key_for_testing")
client = genai.Client(api_key=api_key)

def embed_and_save_chunks(chunks: list[dict], document_id: str, db_session: Session):
    """
    Generates embeddings for a list of text chunks using Gemini and saves them to the database.
    Includes a 4-second delay to respect the free-tier rate limits (15 RPM).
    """
    saved_count = 0
    
    for i, chunk in enumerate(chunks, 1):
        text = chunk["chunk_text"]
        page_num = chunk["page_number"]
        chunk_type = chunk.get("chunk_type", "text")
        
        try:
            # Generate embedding using the new official google-genai SDK
            result = client.models.embed_content(
                model="gemini-embedding-2",
                contents=text,
                config={"task_type": "RETRIEVAL_DOCUMENT", "output_dimensionality": 768}
            )
            embedding_vector = result.embeddings[0].values
            
            # Create DB model
            db_chunk = DocumentChunk(
                document_id=document_id,
                content=text,
                chunk_type=chunk_type,
                page_number=page_num,
                embedding=embedding_vector
            )
            
            db_session.add(db_chunk)
            saved_count += 1
            
            logger.info(f"Embedded chunk {i}/{len(chunks)} from page {page_num}.")
            
            # Commit periodically to avoid massive transactions
            if i % 50 == 0:
                db_session.commit()
                
            # Rate limit protection for Free Tier (15 RPM -> 4 seconds per request)
            if i < len(chunks):
                time.sleep(4.0)
                
        except Exception as e:
            logger.error(f"Failed to embed/save chunk {i} on page {page_num}: {e}")
            db_session.rollback()
            raise
            
    # Final commit
    db_session.commit()
    logger.info(f"Successfully embedded and saved {saved_count} chunks to the database.")
    return saved_count
