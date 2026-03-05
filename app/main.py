"""
FastAPI deployment for multilingual PDF summarizer.
Includes health checks, async processing, and error handling.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import uuid
import os
import shutil
from pathlib import Path
import logging
import time
from datetime import datetime

from app.models.summarizer import MultilingualSummarizer
from app.utils.pdf_extractor import PDFExtractor
from app.utils.text_processor import TextProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multilingual PDF Summarizer API",
    description="Extract and summarize text from PDFs in English, French, and Arabic",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
summarizer = MultilingualSummarizer()
pdf_extractor = PDFExtractor()
text_processor = TextProcessor()

# Create temp directory for uploads
TEMP_DIR = Path("/tmp/pdf_summarizer")
TEMP_DIR.mkdir(exist_ok=True)

# Request/Response models
class SummaryRequest(BaseModel):
    text: str = Field(..., description="Text to summarize", min_length=100)
    language: str = Field("en", description="Language code (en/fr/ar)")
    max_length: Optional[int] = Field(300, description="Maximum summary length")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Long text to summarize...",
                "language": "en",
                "max_length": 300
            }
        }

class SummaryResponse(BaseModel):
    summary: str
    processing_time: float
    language: str
    model_used: str

class PDFSummaryResponse(BaseModel):
    task_id: str
    status: str
    summary: Optional[str] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

# Store for async tasks
tasks = {}

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Multilingual PDF Summarizer API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/docs": "API documentation",
            "/summarize/text": "Summarize text directly",
            "/summarize/pdf": "Upload and summarize PDF",
            "/task/{task_id}": "Get task status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with model status."""
    return {
        "status": "healthy",
        "device": summarizer.device,
        "model": "mT5_multilingual_XLSum",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/summarize/text", response_model=SummaryResponse)
async def summarize_text(request: SummaryRequest):
    """
    Summarize provided text directly.
    """
    try:
        start_time = time.time()
        
        # Validate language
        if request.language not in ["en", "fr", "ar"]:
            raise HTTPException(status_code=400, detail="Language must be en, fr, or ar")
        
        # Generate summary
        summary = summarizer.summarize(request.text, request.language)
        
        processing_time = time.time() - start_time
        
        return SummaryResponse(
            summary=summary,
            processing_time=processing_time,
            language=request.language,
            model_used="csebuetnlp/mT5_multilingual_XLSum"
        )
        
    except Exception as e:
        logger.error(f"Error in summarize_text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize/pdf", response_model=PDFSummaryResponse)
async def summarize_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = "en"
):
    """
    Upload PDF and get summary asynchronously.
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = TEMP_DIR / f"{task_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Initialize task status
        tasks[task_id] = {
            "status": "processing",
            "file_path": str(file_path),
            "language": language
        }
        
        # Add background task
        background_tasks.add_task(process_pdf_task, task_id, str(file_path), language)
        
        return PDFSummaryResponse(
            task_id=task_id,
            status="processing"
        )
        
    except Exception as e:
        logger.error(f"Error in summarize_pdf: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task/{task_id}", response_model=PDFSummaryResponse)
async def get_task_status(task_id: str):
    """Get status of an async task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    return PDFSummaryResponse(**task)

async def process_pdf_task(task_id: str, file_path: str, language: str):
    """Background task for PDF processing."""
    try:
        start_time = time.time()
        
        # Extract text from PDF
        text = pdf_extractor.extract(file_path)
        
        # Clean and process text
        cleaned_text = text_processor.clean(text)
        
        # Detect language if not specified
        if language == "auto":
            language = text_processor.detect_language(cleaned_text)
        
        # Generate summary
        summary = summarizer.summarize(cleaned_text, language)
        
        # Update task status
        tasks[task_id].update({
            "status": "completed",
            "summary": summary,
            "processing_time": time.time() - start_time
        })
        
        # Clean up file
        os.unlink(file_path)
        
    except Exception as e:
        tasks[task_id].update({
            "status": "failed",
            "error": str(e)
        })
        logger.error(f"Task {task_id} failed: {str(e)}")