# 🏛️ DHARMA Legal Analysis System

AI-powered FIR processing system with bilingual support (English + Telugu)

---

## 🚀 Quick Start

### 1. Clone Repository

`bash
git clone https://github.com/yourusername/dharma-legal-analysis.git
cd dharma-legal-analysis
`

---

### 2. Install Requirements

#### A. Install Python Dependencies

`bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
`

#### B. Install Ollama

Download and install from: **https://ollama.ai/**

`bash
# Start Ollama
ollama serve

# Pull Qwen model (in new terminal)
ollama pull qwen2.5:latest
`

---

### 3. Setup Environment

Create .env file:

`env
USE_LOCAL_LLM=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:latest
`

---

### 4. Run Application

`bash
cd dharma_project/backend
python app.py
`

**Server runs at:** http://localhost:8000

**API Docs:** http://localhost:8000/docs

---

## 📦 Requirements

### System Requirements
- Python 3.10+
- 8GB RAM minimum
- 10GB storage

### Dependencies (requirements.txt)
`txt
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.4.0
chromadb>=0.4.15
sentence-transformers>=2.2.0
requests>=2.31.0
python-multipart>=0.0.6
python-dotenv>=1.0.0
PyPDF2>=3.0.0
python-docx>=1.0.0
tenacity>=8.2.0
reportlab>=4.0.0
`

---

## 🎯 Basic Usage

### Test API with cURL

`bash
curl -X POST "http://localhost:8000/process_fir" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I Ram Kumar report theft of mobile phone worth Rs 25000 on 15/10/2025"
  }'
`

### Or use Swagger UI

Open http://localhost:8000/docs and test directly in browser

---

## ❓ Troubleshooting

**Ollama not connecting?**
`bash
ollama serve
`

**Module not found?**
`bash
pip install -r requirements.txt --force-reinstall
`

**Model missing?**
`bash
ollama pull qwen2.5:latest
`

---

## 📧 Support

Issues: [GitHub Issues](https://github.com/yourusername/dharma-legal-analysis/issues)

---

**That's it! You're ready to process FIRs with AI** 🚀
