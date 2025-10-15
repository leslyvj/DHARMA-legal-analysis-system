"""
=============================================================================
DHARMA LEGAL ANALYSIS SYSTEM - COMPLETE SINGLE FILE BACKEND
=============================================================================

REQUIREMENTS:
- FastAPI backend with comprehensive FIR analysis
- Bilingual support (English + Telugu Unicode detection)
- Legal section mapping with RAG grounding
- ChromaDB for knowledge base
- Local LLM embeddings (sentence-transformers)
- Ollama/Qwen for LLM generation (instead of Gemini)
- Structured JSON output with ≥90% extraction accuracy
- Response time ≤10 seconds

DEPENDENCIES (requirements.txt):
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

EXTERNAL REQUIREMENTS:
- Ollama installed and running (ollama serve)
- Qwen model pulled (ollama pull qwen2.5:latest)

=============================================================================
"""

import os
import sys
import uuid
import json
import logging
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ------------------ CONFIGURATION ------------------

# Local LLM Configuration (Ollama)
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:latest")

# Fallback to Gemini if needed
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-1.5-flash-latest")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_LLM_MODEL}:generateContent"

# Database and Storage
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./kb_processed/chroma_db")
LOGS_DIR = os.getenv("LOGS_DIR", "./logs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Performance
MAX_FIR_PROCESS_TIME = int(os.getenv("MAX_FIR_PROCESS_TIME", "10"))

# Create directories
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ------------------ LOGGING SETUP ------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(f"{LOGS_DIR}/app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DHARMA")

# ------------------ CUSTOM EXCEPTIONS ------------------

class ExtractionError(Exception):
    """Raised when FIR extraction fails"""
    pass

class IngestionError(Exception):
    """Raised when KB ingestion fails"""
    pass

class LegalMappingError(Exception):
    """Raised when legal mapping fails"""
    pass

# ------------------ PYDANTIC MODELS ------------------

class Community(str, Enum):
    SC = "SC"
    ST = "ST"
    OBC = "OBC"
    GENERAL = "General"
    UNKNOWN = "Unknown"

class ComplainantInfo(BaseModel):
    name: Optional[str] = None
    father: Optional[str] = None
    age: Optional[int] = None
    community: Optional[str] = None
    occupation: Optional[str] = None
    address: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

class AccusedInfo(BaseModel):
    name: Optional[str] = "Unknown"
    age: Optional[int] = None
    relation: Optional[str] = None
    address: Optional[str] = None
    criminal_history: Optional[str] = None
    description: Optional[str] = None
    identification_status: str = "Unknown"

class VehicleInfo(BaseModel):
    registration: Optional[str] = None
    vehicle_type: Optional[str] = None
    model: Optional[str] = None
    color: Optional[str] = None

class WeaponInfo(BaseModel):
    weapon_type: Optional[str] = None
    usage: Optional[str] = None
    legal_status: Optional[str] = "Unknown"

class PropertyLoss(BaseModel):
    item: str = Field(default="Unknown")  # Non-null default
    value: float = Field(default=0.0, ge=0.0)  # Non-negative
    currency: str = Field(default="INR")

class InjuryInfo(BaseModel):
    injury_type: Optional[str] = None
    location: Optional[str] = None
    severity: Optional[str] = None
    medical_treatment: Optional[str] = None

class LegalGrounding(BaseModel):
    primary_source: Optional[str] = None
    supporting_documents: List[str] = []
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)

class LegalSection(BaseModel):
    act: str
    section: str
    description: str
    reasoning: str
    severity: Optional[str] = None
    keywords_found: List[str] = []
    legal_grounding: Optional[LegalGrounding] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

class ExtractedData(BaseModel):
    complainant: Optional[ComplainantInfo] = None
    datetime: Optional[str] = None
    place: Optional[str] = None
    location_type: Optional[str] = None
    accused: List[AccusedInfo] = []
    vehicles: List[VehicleInfo] = []
    weapons: List[WeaponInfo] = []
    offences: List[str] = []
    injuries: List[InjuryInfo] = []
    property_loss: List[PropertyLoss] = []
    threats: List[str] = []
    witnesses: List[str] = []
    impact: Optional[str] = None
    emotional_tone: Optional[str] = None
    overall_confidence: float = 0.0

class FIRRequest(BaseModel):
    text: str = Field(..., min_length=50)
    filename: Optional[str] = None
    preserve_original: bool = True

class FIRResponse(BaseModel):
    case_id: str
    language: str
    original_text: Optional[str] = None
    translated_text: Optional[str] = None
    extracted_data: ExtractedData
    legal_sections: List[LegalSection]
    summary: str
    processing_time: float
    timestamp: str
    partial: bool = False
    missing_fields: List[str] = []

class QueryRequest(BaseModel):
    query: str
    k: int = 3
# ------------------ FASTAPI APP ------------------

app = FastAPI(
    title="DHARMA Legal Analysis System",
    description="AI-powered bilingual FIR processing with local LLM (Qwen)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ GLOBAL INITIALIZATION ------------------

embedding_model: Optional[SentenceTransformer] = None
chroma_client: Optional[chromadb.Client] = None
kb_collection: Optional[chromadb.Collection] = None

@app.on_event("startup")
async def startup_event():
    """Initialize models and databases on startup"""
    global embedding_model, chroma_client, kb_collection
    
    try:
        logger.info("🚀 Starting DHARMA Backend...")
        
        # Check LLM availability
        if USE_LOCAL_LLM:
            logger.info(f"🤖 Using local LLM: Ollama ({OLLAMA_MODEL})")
            try:
                test_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
                if test_response.status_code == 200:
                    logger.info("✅ Ollama is running and accessible")
                else:
                    logger.warning("⚠️ Ollama might not be running properly")
            except:
                logger.error("❌ Cannot connect to Ollama. Make sure 'ollama serve' is running!")
                logger.info("   Run: ollama serve")
                logger.info(f"   Check model: ollama list | grep {OLLAMA_MODEL}")
        else:
            logger.info("🌐 Using Gemini API")
            if not GEMINI_API_KEY:
                logger.warning("⚠️ GEMINI_API_KEY not set!")
        
        # Initialize embedding model
        logger.info(f"📦 Loading embedding model: {EMBEDDING_MODEL}")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("✅ Embedding model loaded")
        
        # Initialize ChromaDB
        logger.info(f"🗄️ Initializing ChromaDB at {CHROMA_DB_DIR}")
        chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_DIR,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        kb_collection = chroma_client.get_or_create_collection(
            name="dharma_kb",
            metadata={"description": "Legal knowledge base for DHARMA"}
        )
        logger.info("✅ ChromaDB initialized")
        
        logger.info("✅ DHARMA Backend ready!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

# ------------------ HELPER FUNCTIONS ------------------

def detect_telugu(text: str) -> bool:
    """Detect Telugu Unicode characters (0C00-0C7F)"""
    telugu_pattern = re.compile(r'[\u0C00-\u0C7F]+')
    return bool(telugu_pattern.search(text))

def extract_telugu_segments(text: str) -> List[str]:
    """Extract all Telugu text segments"""
    telugu_pattern = re.compile(r'[\u0C00-\u0C7F\s]+')
    segments = telugu_pattern.findall(text)
    return [s.strip() for s in segments if s.strip()]

def detect_language(text: str) -> str:
    """Detect document language"""
    has_telugu = detect_telugu(text)
    telugu_ratio = len(''.join(extract_telugu_segments(text))) / len(text) if text else 0
    
    if telugu_ratio > 0.5:
        return "Telugu"
    elif telugu_ratio > 0.1:
        return "Mixed"
    else:
        return "English"

def extract_vehicle_numbers(text: str) -> List[str]:
    """Extract Indian vehicle registration numbers"""
    pattern = r'\b[A-Z]{2}[-\s]?\d{2}[-\s]?[A-Z]{1,2}[-\s]?\d{4}\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return [m.upper().replace(' ', '-') for m in matches]

def extract_currency_amounts(text: str) -> List[float]:
    """Extract currency amounts"""
    pattern = r'(?:₹|Rs\.?|INR)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
    matches = re.findall(pattern, text)
    return [float(m.replace(',', '')) for m in matches]

def create_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using sentence-transformers"""
    if not embedding_model:
        raise RuntimeError("Embedding model not initialized")
    
    embeddings = embedding_model.encode(texts, convert_to_tensor=False)
    return embeddings.tolist()

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    
    return chunks

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_ollama(prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.1) -> str:
    """Call Ollama API with Qwen model"""
    try:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        
        # Build messages
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 8192,
            }
        }
        
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error calling Ollama: {e}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_gemini(prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.1) -> str:
    """Call Gemini API (fallback)"""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not configured")
    
    headers = {
        "Content-Type": "application/json"
    }
    
    contents = []
    if system_prompt:
        contents.append({
            "role": "user",
            "parts": [{"text": system_prompt}]
        })
    
    contents.append({
        "role": "user",
        "parts": [{"text": prompt}]
    })
    
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 8192,
        }
    }
    
    url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if len(parts) > 0 and "text" in parts[0]:
                    return parts[0]["text"]
        
        logger.warning("No text found in Gemini response")
        return ""
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Gemini API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error calling Gemini: {e}")
        raise

def call_llm(prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.1) -> str:
    """Universal LLM caller - uses Ollama/Qwen or Gemini based on config"""
    if USE_LOCAL_LLM:
        return call_ollama(prompt, system_prompt, temperature)
    else:
        return call_gemini(prompt, system_prompt, temperature)

# Find semantic_search function (around line 300-350) and replace with:

def semantic_search(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Perform semantic search on knowledge base"""
    if not kb_collection:
        return []
    
    try:
        # Generate query embedding
        query_embedding = create_embeddings([query])[0]
        
        # Query ChromaDB
        results = kb_collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Format results
        contexts = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                raw_distance = results['distances'][0][i] if results['distances'] else 0.5
                
                # FIX: Normalize distance to valid confidence score (0.0 to 1.0)
                # ChromaDB can return negative distances for cosine similarity
                if raw_distance < 0:
                    # Negative distance = high similarity
                    normalized_distance = abs(raw_distance)
                else:
                    # Positive distance, keep as is
                    normalized_distance = raw_distance
                
                # Ensure it's in valid range
                normalized_distance = max(0.0, min(1.0, normalized_distance))
                
                contexts.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': normalized_distance  # Now guaranteed to be 0.0-1.0
                })
        
        return contexts
        
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        return []
    
# ------------------ FIR EXTRACTION ------------------

def extract_fir_data(fir_text: str) -> ExtractedData:
    """Extract structured data from FIR using LLM"""
    try:
        logger.info("🔍 Extracting FIR data...")
        
        system_prompt = """You are an expert AI assistant analyzing police FIRs (First Information Reports) in India.
You must extract structured information accurately and return valid JSON only."""
        
        prompt = f"""
Extract ALL information from this FIR and return ONLY valid JSON (no markdown, no explanations).

**FIR TEXT:**
{fir_text}

**EXTRACTION RULES:**
1. Extract from both English and Telugu text
2. Preserve Telugu text exactly
3. Use null for missing data
4. Extract approximate ages (e.g., "about 28" → 28)
5. Identify community: SC/ST/OBC/General/Unknown

**OUTPUT FORMAT (VALID JSON ONLY):**
{{
  "complainant": {{
    "name": "string or null",
    "father": "string or null",
    "age": number or null,
    "community": "SC/ST/OBC/General/Unknown",
    "occupation": "string or null",
    "address": "string or null",
    "confidence": 0.0 to 1.0
  }},
  "datetime": "DD-MM-YYYY HH:MM AM/PM or null",
  "place": "string or null",
  "location_type": "Rural/Urban/Highway or null",
  "accused": [
    {{
      "name": "string or Unknown",
      "age": number or null,
      "relation": "string or null",
      "address": "string or null",
      "criminal_history": "string or null",
      "description": "string or null",
      "identification_status": "Identified/Unidentified/Partially_Identified"
    }}
  ],
  "vehicles": [
    {{
      "registration": "string or null",
      "vehicle_type": "string or null",
      "model": "string or null",
      "color": "string or null"
    }}
  ],
  "weapons": [
    {{
      "weapon_type": "string or null",
      "usage": "string or null",
      "legal_status": "Legal/Illegal/Unknown"
    }}
  ],
  "offences": ["list of offence descriptions"],
  "injuries": [
    {{
      "injury_type": "string or null",
      "location": "body part",
      "severity": "Minor/Moderate/Severe/Grievous",
      "medical_treatment": "string or null"
    }}
  ],
  "property_loss": [
    {{
      "item": "string",
      "value": number,
      "currency": "INR"
    }}
  ],
  "threats": ["specific threats made"],
  "witnesses": ["witness names"],
  "impact": "impact description",
  "emotional_tone": "Fear/Anger/Distress/Trauma/Neutral"
}}

Return ONLY the JSON object.
"""
        
        # Call LLM
        response = call_llm(prompt, system_prompt, temperature=0.1)
        
        # Parse JSON
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        # Clean response
        response = response.strip()
        
        data = json.loads(response)
        
        # Convert to ExtractedData model with validation
        extracted = ExtractedData(
            complainant=ComplainantInfo(**data.get('complainant', {})) if data.get('complainant') else None,
            datetime=data.get('datetime'),
            place=data.get('place'),
            location_type=data.get('location_type'),
            accused=[AccusedInfo(**a) for a in data.get('accused', [])],
            vehicles=[VehicleInfo(**v) for v in data.get('vehicles', [])],
            weapons=[WeaponInfo(**w) for w in data.get('weapons', [])],
            offences=data.get('offences', []),
            injuries=[InjuryInfo(**i) for i in data.get('injuries', [])],
            # FIX: Validate property loss items with proper null handling
            property_loss=[
                PropertyLoss(
                    item=str(p.get('item') or 'Unknown'),  # Ensure non-null string
                    value=float(p.get('value') or 0.0),     # Ensure float
                    currency=str(p.get('currency') or 'INR') # Ensure non-null string
                ) 
                for p in data.get('property_loss', [])
                if isinstance(p, dict) and p  # Ensure it's a non-empty dictionary
            ],
            threats=data.get('threats', []),
            witnesses=data.get('witnesses', []),
            impact=data.get('impact'),
            emotional_tone=data.get('emotional_tone')
        )        
        # Calculate confidence
        extracted.overall_confidence = calculate_extraction_confidence(extracted)
        
        logger.info(f"✅ Extraction complete (confidence: {extracted.overall_confidence:.2f})")
        return extracted
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        logger.error(f"Response: {response[:500]}")
        raise ExtractionError(f"Failed to parse extraction response: {str(e)}")
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        logger.error(traceback.format_exc())
        raise ExtractionError(f"Extraction failed: {str(e)}")

def calculate_extraction_confidence(data: ExtractedData) -> float:
    """Calculate overall extraction confidence score"""
    scores = []
    
    if data.complainant and data.complainant.confidence > 0:
        scores.append(data.complainant.confidence)
    
    if data.complainant and data.complainant.name:
        scores.append(0.9)
    if data.datetime:
        scores.append(0.85)
    if data.place:
        scores.append(0.85)
    if len(data.accused) > 0:
        scores.append(0.9)
    if len(data.offences) > 0:
        scores.append(0.95)
    
    return sum(scores) / len(scores) if scores else 0.5

# ------------------ LEGAL MAPPING ------------------

# Find the map_legal_sections function (around line 550-600) and replace it with this:

def map_legal_sections(extracted_data: ExtractedData, fir_text: str) -> List[LegalSection]:
    """Map extracted data to applicable legal sections"""
    try:
        logger.info("⚖️ Mapping legal sections...")
        
        # Build KB query
        query_parts = []
        if extracted_data.offences:
            query_parts.extend(extracted_data.offences[:3])
        if extracted_data.weapons:
            query_parts.extend([w.weapon_type for w in extracted_data.weapons if w.weapon_type])
        if extracted_data.complainant and extracted_data.complainant.community in ["SC", "ST"]:
            query_parts.append("SC ST atrocities caste abuse")
        
        kb_query = " ".join(query_parts)
        
        # Retrieve KB context
        kb_contexts = semantic_search(kb_query, k=5) if kb_query else []
        
        # Format KB context
        kb_text = "\n\n".join([
            f"Source: {ctx['metadata'].get('filename', 'Unknown')}\n{ctx['text']}"
            for ctx in kb_contexts[:3]
        ]) if kb_contexts else "No specific legal references retrieved. Use general knowledge of Indian laws."
        
        system_prompt = """You are a legal expert in Indian criminal law with expertise in BNS 2023, SC/ST Act 1989, Arms Act 1959, and Motor Vehicles Act 1988."""
        
        prompt = f"""
Map this FIR to applicable Indian legal sections.

**KNOWLEDGE BASE:**
{kb_text}

**FIR DATA:**
- Complainant Community: {extracted_data.complainant.community if extracted_data.complainant else 'Unknown'}
- Offences: {', '.join(extracted_data.offences)}
- Weapons: {', '.join([w.weapon_type for w in extracted_data.weapons if w.weapon_type]) or 'None'}
- Injuries: {len(extracted_data.injuries)} injuries
- Property Loss: ₹{sum([p.value for p in extracted_data.property_loss])}
- Threats: {len(extracted_data.threats)} threats
- Accused: {len(extracted_data.accused)} person(s)

**LEGAL FRAMEWORK:**

1. **BNS 2023:**
   - Sec 115(2): Voluntarily causing hurt
   - Sec 118: Grievous hurt
   - Sec 309: Robbery
   - Sec 351: Criminal intimidation
   - Sec 303: Theft

2. **SC/ST Act 1989:**
   - Sec 3(1)(r): Caste-based insult/abuse
   - Sec 3(2)(v): Atrocity on SC/ST grounds
   - **Apply if:** Community = SC/ST AND caste abuse

3. **Arms Act 1959:**
   - Sec 25: Illegal possession
   - Sec 27: Use in offence

4. **Motor Vehicles Act 1988:**
   - Sec 66: Unauthorized use

**OUTPUT (VALID JSON ARRAY ONLY):**
[
  {{
    "act": "BNS 2023",
    "section": "309",
    "description": "Robbery",
    "reasoning": "Evidence-based reasoning here",
    "severity": "Cognizable, Non-bailable",
    "keywords_found": ["force", "theft"],
    "confidence": 0.95
  }}
]

Return ONLY JSON array.
"""
        
        response = call_llm(prompt, system_prompt, temperature=0.1)
        
        # Parse JSON
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        sections_data = json.loads(response.strip())
        
        # Convert to LegalSection models with grounding
        legal_sections = []
        for section_dict in sections_data:
            grounding = None
            if kb_contexts:
                # FIX: Ensure confidence_score is between 0 and 1
                raw_distance = kb_contexts[0].get('distance', 0.5)
                
                # Convert distance to similarity score (0 to 1)
                # ChromaDB uses cosine distance, so we need to normalize
                # Distance can be negative for cosine similarity
                if raw_distance < 0:
                    # Negative distance means high similarity
                    confidence_score = min(1.0, abs(raw_distance))
                else:
                    # Positive distance, convert to similarity
                    confidence_score = max(0.0, 1.0 - raw_distance)
                
                # Clamp to valid range
                confidence_score = max(0.0, min(1.0, confidence_score))
                
                grounding = LegalGrounding(
                    primary_source=kb_contexts[0]['metadata'].get('filename') if kb_contexts else None,
                    supporting_documents=[ctx['metadata'].get('filename', 'Unknown') for ctx in kb_contexts[:3]],
                    confidence_score=confidence_score
                )
            
            section = LegalSection(
                **section_dict,
                legal_grounding=grounding
            )
            legal_sections.append(section)
        
        logger.info(f"✅ Mapped {len(legal_sections)} legal sections")
        return legal_sections
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        raise LegalMappingError(f"Failed to parse mapping response: {str(e)}")
    except Exception as e:
        logger.error(f"Legal mapping error: {e}")
        logger.error(traceback.format_exc())
        raise LegalMappingError(f"Legal mapping failed: {str(e)}")
    
    
# ------------------ SUMMARY GENERATION ------------------

def generate_summary(extracted_data: ExtractedData, legal_sections: List[LegalSection]) -> str:
    """Generate comprehensive legal summary"""
    try:
        logger.info("📝 Generating summary...")
        
        system_prompt = """You are a legal professional writing formal case summaries for police investigations."""
        
        prompt = f"""
Write a professional 3-5 paragraph legal summary of this FIR.

**INCIDENT:**
- Complainant: {extracted_data.complainant.name if extracted_data.complainant else 'Unknown'}
- Date/Time: {extracted_data.datetime or 'Not specified'}
- Location: {extracted_data.place or 'Not specified'}
- Accused: {len(extracted_data.accused)} person(s)
- Offences: {', '.join(extracted_data.offences[:5])}
- Property Loss: ₹{sum([p.value for p in extracted_data.property_loss])}
- Injuries: {len(extracted_data.injuries)}
- Emotional Impact: {extracted_data.emotional_tone or 'Not assessed'}

**LEGAL SECTIONS:**
{chr(10).join([f"- {s.act} Sec {s.section}: {s.description} ({s.severity})" for s in legal_sections])}

**FORMAT:**
Paragraph 1: Incident overview (who, what, when, where)
Paragraph 2: Nature of offences and evidence
Paragraph 3: Legal framework and severity
Paragraph 4: Investigation considerations

Use formal legal language. Be comprehensive but concise.
"""
        
        summary = call_llm(prompt, system_prompt, temperature=0.3)
        logger.info("✅ Summary generated")
        return summary.strip()
        
    except Exception as e:
        logger.error(f"Summary error: {e}")
        return "Summary generation failed. Please review extracted data manually."

# ------------------ API ENDPOINTS ------------------

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "DHARMA Legal Analysis System",
        "version": "1.0.0",
        "llm": f"Ollama ({OLLAMA_MODEL})" if USE_LOCAL_LLM else "Gemini",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "process_fir": "/process_fir",
            "kb_ingest": "/kb/ingest",
            "kb_query": "/query"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "llm_backend": "Ollama" if USE_LOCAL_LLM else "Gemini",
            "components": {
                "embedding_model": embedding_model is not None,
                "chromadb": chroma_client is not None,
                "kb_collection": kb_collection is not None,
                "ollama": False,
                "gemini": bool(GEMINI_API_KEY)
            }
        }
        
        # Check Ollama
        if USE_LOCAL_LLM:
            try:
                resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
                health["components"]["ollama"] = resp.status_code == 200
            except:
                health["components"]["ollama"] = False
        
        # Check KB size
        if kb_collection:
            try:
                health["kb_document_count"] = kb_collection.count()
            except:
                health["kb_document_count"] = 0
        
        return health
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/kb/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Ingest document into knowledge base"""
    try:
        logger.info(f"📥 Ingesting: {file.filename}")
        
        content = (await file.read()).decode("utf-8", errors="ignore")
        chunks = chunk_text(content, chunk_size=800, overlap=100)
        embeddings = create_embeddings(chunks)
        
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {
                "filename": file.filename,
                "chunk_id": i,
                "total_chunks": len(chunks),
                "ingested_at": datetime.now().isoformat()
            }
            for i in range(len(chunks))
        ]
        
        kb_collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        
        logger.info(f"✅ Ingested {len(chunks)} chunks")
        
        return {
            "filename": file.filename,
            "chunks_stored": len(chunks),
            "total_kb_documents": kb_collection.count()
        }
        
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query_kb(request: QueryRequest):
    """Query knowledge base with RAG"""
    try:
        logger.info(f"🔍 Query: {request.query[:50]}...")
        
        contexts = semantic_search(request.query, k=request.k)
        context_text = "\n\n".join([ctx['text'] for ctx in contexts])
        
        prompt = f"""
Context:
{context_text}

Question: {request.query}

Answer based on context:
"""
        
        answer = call_llm(prompt, temperature=0.2)
        
        return {
            "query": request.query,
            "answer": answer,
            "sources": [
                {
                    "filename": ctx['metadata'].get('filename', 'Unknown'),
                    "relevance": 1.0 - ctx['distance']
                }
                for ctx in contexts
            ]
        }
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_fir", response_model=FIRResponse)
def process_fir(request: FIRRequest):
    """Main FIR processing endpoint"""
    start_time = datetime.now()
    case_id = str(uuid.uuid4())
    
    try:
        logger.info("=" * 80)
        logger.info(f"📋 PROCESSING FIR (Case: {case_id})")
        logger.info("=" * 80)
        
        fir_text = request.text.strip()
        language = detect_language(fir_text)
        logger.info(f"🌐 Language: {language}")
        
        # Translation if needed
        original_text = fir_text if request.preserve_original else None
        translated_text = None
        
        if language == "Telugu":
            logger.info("🔄 Translating Telugu...")
            translation_prompt = f"Translate this Telugu text to English, preserving all details:\n\n{fir_text}"
            translated_text = call_llm(translation_prompt, temperature=0.1)
            fir_text_for_processing = translated_text
        else:
            fir_text_for_processing = fir_text
        
        # Extract data
        extracted_data = extract_fir_data(fir_text_for_processing)
        
        # Map legal sections
        legal_sections = map_legal_sections(extracted_data, fir_text_for_processing)
        
        # Generate summary
        summary = generate_summary(extracted_data, legal_sections)
        
        # Check missing fields
        missing_fields = []
        if not extracted_data.complainant or not extracted_data.complainant.name:
            missing_fields.append("complainant_name")
        if not extracted_data.datetime:
            missing_fields.append("datetime")
        if not extracted_data.place:
            missing_fields.append("place")
        if len(extracted_data.accused) == 0:
            missing_fields.append("accused")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = FIRResponse(
            case_id=case_id,
            language=language,
            original_text=original_text,
            translated_text=translated_text,
            extracted_data=extracted_data,
            legal_sections=legal_sections,
            summary=summary,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            partial=len(missing_fields) > 0,
            missing_fields=missing_fields
        )
        
        logger.info(f"✅ Complete in {processing_time:.2f}s (Confidence: {extracted_data.overall_confidence:.2f})")
        logger.info("=" * 80)
        
        return response
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ MAIN ------------------

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 DHARMA Backend Server")
    logger.info(f"📍 URL: http://0.0.0.0:{os.getenv('PORT', 8080)}")
    logger.info(f"🤖 LLM: {'Ollama ('+OLLAMA_MODEL+')' if USE_LOCAL_LLM else 'Gemini'}")
    logger.info(f"📊 Embeddings: {EMBEDDING_MODEL}")
    logger.info(f"🗄️ ChromaDB: {CHROMA_DB_DIR}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        log_level="info"
    )