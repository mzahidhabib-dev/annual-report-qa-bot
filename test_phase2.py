import os
import uuid
import time
from dotenv import load_dotenv
from google import genai
from src.db.database import SessionLocal
from src.db.models import ExtractedTable
from src.ingestion.table_extractor import extract_tables_from_pdf
from src.embeddings.table_summarizer import summarize_tables

load_dotenv()

def run_phase2():
    pdf_path = os.path.join("sample_docs", "Northbridge_Dynamics_2025_Annual_Report.pdf")
    document_id = str(uuid.uuid4())
    
    print(f"==========================================")
    print(f"PHASE 2 - STEP 2.1: Extracting Tables")
    print(f"==========================================")
    tables = extract_tables_from_pdf(pdf_path)
    print(f"[SUCCESS] Found {len(tables)} tables.\n")
    
    if not tables:
        print("No tables to process. Exiting.")
        return
        
    print(f"==========================================")
    print(f"PHASE 2 - STEP 2.2: Summarizing Tables")
    print(f"==========================================")
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    # For testing purposes, only process the first 3 tables to save time
    test_tables = tables[:3]
    print(f"[INFO] To save time, this test will only summarize the first {len(test_tables)} tables.\n")
    
    summarized_tables = summarize_tables(test_tables, client)
    
    print(f"\n==========================================")
    print(f"PHASE 2 - STEP 2.3: Embedding & Saving to DB")
    print(f"==========================================")
    
    from src.db.models import ExtractedTable, DocumentChunk
    
    db_session = SessionLocal()
    try:
        saved_table_count = 0
        for i, table_dict in enumerate(summarized_tables, 1):
            summary_text = table_dict["summary"]
            markdown_table = table_dict["markdown_table"]
            page_num = table_dict["page_number"]
            
            # 1. Save the raw table data for reference
            db_table = ExtractedTable(
                document_id=document_id,
                table_summary=summary_text,
                table_data={"markdown": markdown_table},
                page_number=page_num
            )
            db_session.add(db_table)
            
            # 2. Embed the summary so it can be searched
            result = client.models.embed_content(
                model="gemini-embedding-2",
                contents=summary_text,
                config={"task_type": "RETRIEVAL_DOCUMENT", "output_dimensionality": 768}
            )
            embedding_vector = result.embeddings[0].values
            
            # 3. Add the summary to the chunk database so the AI can retrieve it
            db_chunk = DocumentChunk(
                document_id=document_id,
                content=summary_text,
                chunk_type="table",
                page_number=page_num,
                embedding=embedding_vector
            )
            db_session.add(db_chunk)
            
            saved_table_count += 1
            print(f"[SUCCESS] Embedded and staged table {i}/{len(summarized_tables)}")
            
            if i < len(summarized_tables):
                time.sleep(4.0)
                
        db_session.commit()
        print(f"\n[SUCCESS] {saved_table_count} tables have been permanently saved and embedded!")
        print("\n[INFO] You can view the raw tables in 'extracted_tables' and the searchable summaries in 'document_chunks'.")
    except Exception as e:
        print(f"[ERROR] Failed to save tables: {e}")
        db_session.rollback()
    finally:
        db_session.close()

if __name__ == "__main__":
    run_phase2()
