# DHARMA Legal Analysis System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![AI Powered](https://img.shields.io/badge/AI-Powered-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Intelligent First Information Report Processing with Multilingual Support**

*AI-Driven Legal Document Analysis | English & Telugu*

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [API Documentation](#api-documentation) • [Support](#support)

</div>

---

## Overview

DHARMA (Digital Hub for Automated Research and Management of Allegations) is an advanced legal analysis system designed to automate and enhance the processing of First Information Reports (FIRs). Leveraging state-of-the-art natural language processing and machine learning technologies, the system provides intelligent document analysis, entity extraction, and case categorization with bilingual support for English and Telugu.

### Key Objectives

Modern law enforcement agencies face challenges in efficiently processing and analyzing large volumes of FIRs. DHARMA addresses these challenges by providing automated intelligence extraction, semantic search capabilities, and structured case analysis—enabling faster response times and more informed decision-making.

---

## Features

### Core Capabilities

**Intelligent FIR Processing**
- Automated text analysis and entity extraction
- Case categorization and severity assessment
- Structured information extraction from unstructured reports
- Metadata generation for efficient case management

**Multilingual Support**
- Native English language processing
- Telugu language support for regional accessibility
- Cross-language semantic understanding
- Bilingual report generation

**AI-Powered Analysis**
- Qwen 2.5 large language model integration
- Context-aware information extraction
- Semantic similarity search across case database
- Automated case classification

**Document Management**
- Support for PDF, DOCX, and plain text formats
- Automated document parsing and preprocessing
- Structured report generation in PDF format
- Secure document storage and retrieval

**Vector Search Engine**
- ChromaDB-powered semantic search
- Efficient case similarity detection
- Cross-document information retrieval
- Scalable vector indexing

---

## Installation

### System Requirements

**Minimum Specifications**
- Python 3.10 or higher
- 8GB RAM
- 10GB available storage
- Windows 10/11, Linux, or macOS

**Recommended Specifications**
- Python 3.11+
- 16GB RAM
- 20GB available storage
- GPU support for enhanced performance

### Installation Steps

**Step 1: Clone Repository**
```bash
git clone https://github.com/leslyvj/DHARMA-legal-analysis-system.git
cd DHARMA-legal-analysis-system
```

**Step 2: Create Virtual Environment**
```bash
# Create isolated environment
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

**Step 3: Install Python Dependencies**
```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

**Step 4: Install Ollama Framework**

Download and install Ollama from the official website: [https://ollama.ai/](https://ollama.ai/)

```bash
# Start Ollama service
ollama serve

# In a new terminal, pull the Qwen model
ollama pull qwen2.5:latest
```

**Step 5: Configure Environment**

Create a `.env` file in the project root directory:

```env
# LLM Configuration
USE_LOCAL_LLM=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:latest

# Database Configuration
CHROMA_PERSIST_DIR=./chroma_db

# Server Configuration
HOST=127.0.0.1
PORT=8000
```

**Step 6: Launch Application**
```bash
# Navigate to backend directory
cd dharma_project/backend

# Start the server
python app.py
```

The application will be available at:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

---

## Usage

### Processing FIRs via API

**Using cURL**
```bash
curl -X POST "http://localhost:8000/process_fir" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I, Ram Kumar, hereby report the theft of my mobile phone valued at Rs 25,000 that occurred on October 15, 2025, at approximately 3:00 PM near the Central Market.",
    "language": "en"
  }'
```

**Using Python Requests**
```python
import requests

url = "http://localhost:8000/process_fir"
payload = {
    "text": "FIR text content here",
    "language": "en"
}

response = requests.post(url, json=payload)
result = response.json()
print(result)
```

**Using JavaScript/Fetch**
```javascript
fetch('http://localhost:8000/process_fir', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'FIR text content here',
    language: 'en'
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

### Interactive API Documentation

Navigate to http://localhost:8000/docs to access the Swagger UI interface, where you can:
- Explore all available endpoints
- Test API calls directly in the browser
- View request/response schemas
- Download API specifications

---

## API Documentation

### Endpoints

**POST /process_fir**

Process and analyze an FIR document.

*Request Body:*
```json
{
  "text": "string",
  "language": "en"
}
```

*Response:*
```json
{
  "analysis": {
    "entities": [...],
    "case_type": "string",
    "severity": "string",
    "summary": "string"
  },
  "metadata": {
    "processed_at": "timestamp",
    "model_version": "string"
  }
}
```

**GET /search**

Search for similar cases in the database.

*Query Parameters:*
- `query`: Search text
- `limit`: Number of results (default: 10)

**POST /upload_document**

Upload FIR document files (PDF, DOCX, TXT).

*Request:* Multipart form data with file attachment

### Authentication

Currently, the API operates in development mode without authentication. For production deployment, implement appropriate authentication mechanisms such as:
- API key authentication
- OAuth 2.0
- JWT tokens

---

## Architecture

### System Components

```
Application Layer
├── FastAPI Server (app.py)
│   ├── REST API Endpoints
│   ├── Request Validation
│   └── Response Formatting
│
Processing Layer
├── Document Parser
├── Text Preprocessing
├── Entity Extraction
└── Case Classification
│
Intelligence Layer
├── LLM Interface (Qwen 2.5)
├── Semantic Embedding
├── Vector Search Engine
└── Analysis Orchestrator
│
Data Layer
├── ChromaDB Vector Store
├── Document Storage
└── Metadata Repository
```

### Technology Stack

| Component | Technology |
|-----------|-----------|
| Web Framework | FastAPI |
| Language Model | Qwen 2.5 |
| LLM Runtime | Ollama |
| Vector Database | ChromaDB |
| Embeddings | Sentence Transformers |
| Document Processing | PyPDF2, python-docx |
| Report Generation | ReportLab |

---

## Dependencies

### Core Packages

```
fastapi>=0.104.0          # Modern web framework
uvicorn>=0.24.0           # ASGI server
pydantic>=2.4.0           # Data validation
chromadb>=0.4.15          # Vector database
sentence-transformers>=2.2.0  # Text embeddings
requests>=2.31.0          # HTTP client
python-multipart>=0.0.6   # File upload support
python-dotenv>=1.0.0      # Environment management
PyPDF2>=3.0.0             # PDF processing
python-docx>=1.0.0        # DOCX processing
tenacity>=8.2.0           # Retry logic
reportlab>=4.0.0          # PDF generation
```

### Installation

All dependencies are specified in `requirements.txt` and can be installed with:

```bash
pip install -r requirements.txt
```

---

## Configuration

### Environment Variables

The system can be configured through environment variables or a `.env` file:

**LLM Settings**
```env
USE_LOCAL_LLM=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:latest
```

**Database Settings**
```env
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=fir_collection
```

**Server Settings**
```env
HOST=127.0.0.1
PORT=8000
DEBUG=false
```

**Processing Settings**
```env
MAX_UPLOAD_SIZE=10485760  # 10MB
SUPPORTED_LANGUAGES=en,te
DEFAULT_LANGUAGE=en
```

---

## Troubleshooting

### Common Issues and Solutions

**Issue: Ollama Connection Failed**
```bash
# Verify Ollama is running
ollama serve

# Check if model is available
ollama list

# Test model directly
ollama run qwen2.5:latest "Test message"
```

**Issue: Module Import Errors**
```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall

# Verify virtual environment is activated
which python  # Linux/macOS
where python  # Windows
```

**Issue: Model Not Found**
```bash
# Pull the required model
ollama pull qwen2.5:latest

# Verify model is downloaded
ollama list
```

**Issue: Port Already in Use**
```bash
# Check what's using port 8000
# Linux/macOS
lsof -i :8000

# Windows
netstat -ano | findstr :8000

# Change port in .env file
PORT=8001
```

**Issue: ChromaDB Initialization Error**
```bash
# Delete existing database and restart
rm -rf chroma_db/  # Linux/macOS
rmdir /s chroma_db\  # Windows
```

**Issue: Slow Processing Performance**
- Ensure sufficient RAM is available
- Close unnecessary applications
- Consider upgrading to GPU-accelerated processing
- Verify Ollama is using appropriate hardware acceleration

---

## Development

### Running Tests

```bash
# Install testing dependencies
pip install pytest pytest-cov

# Run test suite
pytest tests/

# Run with coverage report
pytest --cov=dharma_project tests/
```

### Code Quality

```bash
# Format code
black dharma_project/

# Check linting
flake8 dharma_project/

# Type checking
mypy dharma_project/
```

### Contributing

We welcome contributions to improve DHARMA. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Make your changes with appropriate tests
4. Ensure code passes all quality checks
5. Commit with clear messages (`git commit -m 'Add: feature description'`)
6. Push to your branch (`git push origin feature/enhancement`)
7. Submit a Pull Request with detailed description

Please review our contributing guidelines before submitting PRs.

---

## Deployment

### Production Considerations

**Security**
- Implement authentication and authorization
- Enable HTTPS/TLS encryption
- Configure CORS policies appropriately
- Sanitize user inputs
- Implement rate limiting

**Performance**
- Use production ASGI server (Gunicorn + Uvicorn workers)
- Configure database connection pooling
- Enable caching mechanisms
- Implement load balancing for high traffic

**Monitoring**
- Set up application logging
- Configure error tracking (e.g., Sentry)
- Monitor system resources
- Track API metrics and performance

### Docker Deployment

```dockerfile
# Dockerfile example
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dharma_project/ .
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Roadmap

### Planned Features

- [ ] Additional language support (Hindi, Tamil, Kannada)
- [ ] Advanced case analytics dashboard
- [ ] Real-time collaboration features
- [ ] Mobile application interface
- [ ] Integration with existing legal management systems
- [ ] Enhanced security and audit logging
- [ ] Batch processing capabilities
- [ ] Export to multiple formats

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for complete terms and conditions.

---

## Acknowledgments

This system integrates several open-source technologies and frameworks:

- **Qwen**: Advanced language model by Alibaba Cloud
- **Ollama**: Local LLM deployment framework
- **FastAPI**: Modern, high-performance web framework
- **ChromaDB**: Efficient vector database
- **Sentence Transformers**: Text embedding models
- **PyPDF2 & python-docx**: Document processing libraries

---

## Support

### Getting Help

**Documentation**: Comprehensive guides available in the `/docs` directory

**GitHub Issues**: Report bugs or request features at [GitHub Issues](https://github.com/leslyvj/DHARMA-legal-analysis-system/issues)

**Community**: Join discussions and share feedback in our community forums

### Contact

For technical support, feature requests, or collaboration inquiries, please open an issue on GitHub or contact the development team through the repository.

---

<div align="center">

**DHARMA Legal Analysis System**

*Empowering Justice Through Artificial Intelligence*

Made with ⚖️ for the Legal Community

</div>
