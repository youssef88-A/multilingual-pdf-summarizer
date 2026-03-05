# Multilingual PDF Summarizer

A production-ready multilingual PDF summarization service supporting English, French, and Arabic.

##  Features

- **Multilingual Support**: English, French, Arabic text extraction and summarization
- **Hybrid PDF Extraction**: Combines pdfplumber with OCR fallback for scanned documents
- **FastAPI Backend**: Async processing with background tasks
- **Docker Deployment**: Containerized with GPU support
- **CI/CD Pipeline**: Automated testing and deployment
- **Model Optimization**: ONNX/TensorRT conversion for faster inference

##  Quick Start

### Using Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/multilingual-pdf-summarizer
cd multilingual-pdf-summarizer

# Build and run with Docker Compose
docker-compose up --build

# API will be available at http://localhost:8000

## Requirements
- Python 3.9+
- PyTorch 2.5.1
- Tesseract OCR (for PDF processing)
- 4GB RAM minimum (8GB recommended)

## License
MIT License - see LICENSE file
