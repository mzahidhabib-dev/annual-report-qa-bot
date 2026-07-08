import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel

from src.db.database import SessionLocal
from src.db.models import Document, ChatSession, DocumentChunk
from src.ingestion.pipeline import ingest_text, ingest_tables, ingest_images
from src.retrieval.hybrid_search import search_with_self_query
from src.retrieval.answer_generator import generate_answer

app = FastAPI(title="Annual Report QA Bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("API_KEY", "dev-secret-key")

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Endpoints ---

@app.post("/documents/upload", dependencies=[Depends(verify_api_key)])
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Roadmap 6.2: Uploads a PDF and strictly runs the Phase 1-3 ingestion pipeline sequentially."""
    os.makedirs("temp_pdfs", exist_ok=True)
    file_path = os.path.join("temp_pdfs", file.filename)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    doc_id = str(uuid.uuid4())
    db_doc = Document(id=doc_id, filename=file.filename, status="ingesting")
    db.add(db_doc)
    db.commit()
    
    try:
        ingest_text(file_path, doc_id)
        ingest_tables(file_path, doc_id)
        ingest_images(file_path, doc_id)
        
        db_doc.status = "ready"
        db.commit()
        
        chunks_created = db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).count()
        
        return {
            "document_id": doc_id,
            "status": "ingested",
            "chunks_created": chunks_created
        }
    except Exception as e:
        db_doc.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.get("/documents", dependencies=[Depends(verify_api_key)])
def list_documents(db: Session = Depends(get_db)):
    """Blueprint req: List available documents for the frontend."""
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    return [{"id": str(d.id), "filename": d.filename, "status": d.status} for d in docs]

@app.get("/documents/{document_id}/status", dependencies=[Depends(verify_api_key)])
def get_status(document_id: str, db: Session = Depends(get_db)):
    """Roadmap 6.2: Returns statistics about the processed document."""
    counts = db.query(DocumentChunk.chunk_type, func.count(DocumentChunk.id)).filter(
        DocumentChunk.document_id == document_id
    ).group_by(DocumentChunk.chunk_type).all()
    
    result = {str(c[0].value): c[1] for c in counts}
    return result

class AskRequest(BaseModel):
    question: str
    session_id: str = None

@app.post("/documents/{document_id}/ask", dependencies=[Depends(verify_api_key)])
def ask_question(document_id: str, req: AskRequest, db: Session = Depends(get_db)):
    """Roadmap 6.2: Accepts a question, self-queries the document, and returns a cited answer."""
    session_id = req.session_id
    if not session_id:
        # Blueprint req: Auto-name sessions based on first question
        session_id = str(uuid.uuid4())
        session_name = req.question[:50] + "..." if len(req.question) > 50 else req.question
        new_session = ChatSession(id=session_id, session_name=session_name)
        db.add(new_session)
        db.commit()
        
    retrieved_chunks = search_with_self_query(req.question, document_id, top_k=5)
    
    if not retrieved_chunks:
        return {
            "answer": "I don't know. The document does not contain enough information to answer this question.",
            "sources": [],
            "session_id": session_id,
            "tokens_used": 0
        }
        
    result = generate_answer(req.question, retrieved_chunks, session_id, db)
    result["session_id"] = session_id
    return result

@app.get("/sessions", dependencies=[Depends(verify_api_key)])
def list_sessions(db: Session = Depends(get_db)):
    """Blueprint req: Sidebar displays 5 most recent sessions."""
    sessions = db.query(ChatSession).order_by(ChatSession.created_at.desc()).limit(5).all()
    return [{"id": str(s.id), "name": s.session_name, "created_at": s.created_at} for s in sessions]
