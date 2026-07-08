import os
import time
from dotenv import load_dotenv
from google import genai
from src.db.database import SessionLocal
from src.db.models import DocumentChunk, ExtractedImage
from src.ingestion.image_extractor import extract_images, describe_image

load_dotenv()

def ingest_images(pdf_path: str, document_id: str):
    """
    Extracts images from the PDF, describes them using Gemini Vision,
    embeds the descriptions, and saves both raw descriptions and embedded chunks to the database.
    """
    output_dir = "temp_images"
    
    # 1. Call extract_images
    images = extract_images(pdf_path, output_dir)
    
    if not images:
        print(f"Found 0 charts/images out of 0 pages checked")
        return
        
    db_session = SessionLocal()
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    saved_count = 0
    
    try:
        # 2. For each image: call describe_image
        for img_data in images:
            page_num = img_data["page_number"]
            image_path = img_data["image_path"]
            
            description = describe_image(image_path)
            
            # 3. If description is "NO_CHART_FOUND" or "DESCRIPTION_FAILED", skip — do not save
            if description in ["NO_CHART_FOUND", "DESCRIPTION_FAILED"]:
                continue
                
            # 4. Otherwise save to ExtractedImage
            db_image = ExtractedImage(
                document_id=document_id,
                page_number=page_num,
                image_description=description
            )
            db_session.add(db_image)
            
            # 5. Call embed_text (inline using gemini-embedding-2)
            result = client.models.embed_content(
                model="gemini-embedding-2",
                contents=description,
                config={"task_type": "RETRIEVAL_DOCUMENT", "output_dimensionality": 768}
            )
            embedding_vector = result.embeddings[0].values
            
            # 6. Save to DocumentChunk
            db_chunk = DocumentChunk(
                document_id=document_id,
                content=description,
                chunk_type="image_caption",
                page_number=page_num,
                embedding=embedding_vector
            )
            db_session.add(db_chunk)
            
            saved_count += 1
            
            # Delay to avoid embedding rate limit
            time.sleep(4.0)
            
        db_session.commit()
        
        # 7. Print summary
        print(f"Found {saved_count} charts/images out of {len(images)} pages checked")
        
    except Exception as e:
        print(f"[ERROR] Failed during image pipeline execution: {e}")
        db_session.rollback()
    finally:
        db_session.close()
