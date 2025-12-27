from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pdfplumber
import ollama
import os
import tempfile
import uuid
import threading
from typing import Optional

app = FastAPI(title="PDF Summarizer API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".pdf"}
OLLAMA_MODEL = "mistral"

# Storage untuk task results
tasks_storage = {}

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file"""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        return None

def summarize_text(text: str) -> str:
    """Summarize text using Ollama"""
    try:
        prompt = f"""Buatlah ringkasan komprehensif dari teks berikut dalam bahasa Indonesia. 
Ringkasan harus mencakup poin-poin utama dan informasi penting:

{text[:4000]}

Berikan ringkasan yang jelas dan terstruktur."""

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{
                'role': 'user',
                'content': prompt
            }]
        )
        
        return response['message']['content']
    except Exception as e:
        return None

def process_pdf_thread(job_id: str, content: bytes, filename: str):
    """Process PDF in separate thread"""
    temp_file_path = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(content)
            temp_file.flush()
            temp_file_path = temp_file.name
        
        # Update status to extracting
        tasks_storage[job_id]["status"] = "extracting"
        
        # Extract text from PDF
        text = extract_text_from_pdf(temp_file_path)
        
        if not text:
            tasks_storage[job_id] = {
                "status": "error",
                "error": "No text found in PDF. The file might be empty or contain only images."
            }
            return
        
        # Update status to summarizing
        tasks_storage[job_id]["status"] = "summarizing"
        tasks_storage[job_id]["text_length"] = len(text)
        
        # Summarize text
        summary = summarize_text(text)
        
        if not summary:
            tasks_storage[job_id] = {
                "status": "error",
                "error": "Failed to generate summary. Ollama might be unavailable."
            }
            return
        
        # Store completed result
        tasks_storage[job_id] = {
            "status": "completed",
            "filename": filename,
            "text_length": len(text),
            "summary": summary
        }
        
    except Exception as e:
        tasks_storage[job_id] = {
            "status": "error",
            "error": str(e)
        }
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass

@app.get("/")
async def root():
    return {"message": "PDF Summarizer API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        ollama.list()
        return {
            "status": "healthy",
            "ollama": "connected",
            "model": OLLAMA_MODEL
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/summarize")
async def summarize_pdf(file: UploadFile = File(...)):
    """Upload PDF and get job ID - returns immediately"""
    
    # Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Only PDF files are allowed."
        )
    
    # Read file content
    content = await file.read()
    
    # Check file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is 10MB."
        )
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Initialize task status
    tasks_storage[job_id] = {
        "status": "processing",
        "filename": file.filename
    }
    
    # Start processing in separate thread (non-blocking!)
    thread = threading.Thread(
        target=process_pdf_thread,
        args=(job_id, content, file.filename),
        daemon=True
    )
    thread.start()
    
    # Return immediately without waiting for thread
    return JSONResponse(content={
        "job_id": job_id,
        "status": "processing",
        "message": "Your PDF is being processed. Use the job_id to check status."
    })

@app.get("/summarize/{job_id}")
async def get_summary_status(job_id: str):
    """Check processing status and get result"""
    if job_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JSONResponse(content=tasks_storage[job_id])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
