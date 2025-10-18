"""
=============================================================================
DHARMA LEGAL ANALYSIS SYSTEM - STREAMLIT FRONTEND
=============================================================================

Features:
- FIR Processing with bilingual support
- Knowledge Base Ingestion
- KB Query Interface
- Case History Viewer
- Export Functionality

=============================================================================
"""

import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import time
import os

# ------------------ CONFIGURATION ------------------

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

# Page config
st.set_page_config(
    page_title="DHARMA Legal Analysis",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ HELPER FUNCTIONS ------------------

def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except:
        return False, None

def process_fir(fir_text, filename=None, preserve_original=True):
    """Process FIR through backend API"""
    try:
        payload = {
            "text": fir_text,
            "filename": filename,
            "preserve_original": preserve_original
        }
        
        with st.spinner("🔍 Analyzing FIR... This may take up to 10 seconds..."):
            response = requests.post(
                f"{BACKEND_URL}/process_fir",
                json=payload,
                timeout=120
            )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json()
    except requests.exceptions.Timeout:
        return False, {"detail": "Request timeout. Backend processing took too long."}
    except Exception as e:
        return False, {"detail": str(e)}

def ingest_document(uploaded_file):
    """Ingest document into knowledge base"""
    try:
        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        
        with st.spinner(f"📥 Ingesting {uploaded_file.name}..."):
            response = requests.post(
                f"{BACKEND_URL}/kb/ingest",
                files=files,
                timeout=120
            )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.text
    except Exception as e:
        return False, str(e)

def query_kb(query, k=3):
    """Query knowledge base"""
    try:
        payload = {"query": query, "k": k}
        
        with st.spinner("🔍 Searching knowledge base..."):
            response = requests.post(
                f"{BACKEND_URL}/query",
                json=payload,
                timeout=30
            )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.text
    except Exception as e:
        return False, str(e)

# ------------------ SIDEBAR ------------------

with st.sidebar:
    st.markdown("## ⚖️ DHARMA System")
    st.markdown("**Legal Analysis & Processing**")
    st.markdown("---")
    
    # Backend health check
    st.markdown("### 🔧 System Status")
    is_healthy, health_data = check_backend_health()
    
    if is_healthy:
        st.success("✅ Backend Online")
        if health_data:
            st.metric("LLM Backend", health_data.get('llm_backend', 'Unknown'))
            st.metric("KB Documents", health_data.get('kb_document_count', 0))
            
            with st.expander("Component Status"):
                components = health_data.get('components', {})
                for comp, status in components.items():
                    icon = "✅" if status else "❌"
                    st.text(f"{icon} {comp.replace('_', ' ').title()}")
    else:
        st.error("❌ Backend Offline")
        st.warning("Please start the backend:\n```bash\npython app.py\n```")
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("### 📊 Session Stats")
    if 'processed_cases' not in st.session_state:
        st.session_state.processed_cases = []
    
    st.metric("Cases Processed", len(st.session_state.processed_cases))
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.info("""
    **DHARMA** analyzes FIRs with:
    - Bilingual support (English + Telugu)
    - Legal section mapping
    - RAG-based grounding
    - 90%+ accuracy target
    """)

# ------------------ MAIN CONTENT ------------------

st.markdown('<div class="main-header">⚖️ DHARMA Legal Analysis System</div>', unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Process FIR", "📚 Knowledge Base", "🔍 Query KB", "📂 Case History"])

# ------------------ TAB 1: PROCESS FIR ------------------

with tab1:
    st.header("📋 FIR Processing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input FIR Text")
        
        # Input method
        input_method = st.radio(
            "Choose input method:",
            ["Paste Text", "Upload File"],
            horizontal=True
        )
        
        fir_text = ""
        filename = None
        
        if input_method == "Paste Text":
            fir_text = st.text_area(
                "Enter FIR text:",
                height=300,
                placeholder="Paste the FIR content here (minimum 50 characters)...\n\nSupports:\n- English text\n- Telugu Unicode (తెలుగు)\n- Mixed bilingual content"
            )
        else:
            uploaded_fir = st.file_uploader(
                "Upload FIR document",
                type=['txt', 'pdf', 'docx'],
                help="Supported formats: TXT, PDF, DOCX"
            )
            
            if uploaded_fir:
                filename = uploaded_fir.name
                
                if uploaded_fir.type == "text/plain":
                    fir_text = uploaded_fir.read().decode('utf-8', errors='ignore')
                elif uploaded_fir.type == "application/pdf":
                    st.warning("PDF processing: text will be extracted automatically")
                    import PyPDF2
                    from io import BytesIO
                    pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_fir.read()))
                    fir_text = ""
                    for page in pdf_reader.pages:
                        fir_text += page.extract_text()
                else:
                    st.error("Unsupported file type")
        
        # Display preview
        if fir_text:
            with st.expander("📄 Preview Input"):
                st.text(fir_text[:500] + ("..." if len(fir_text) > 500 else ""))
                st.caption(f"Total characters: {len(fir_text)}")
    
    with col2:
        st.subheader("Processing Options")
        
        preserve_original = st.checkbox(
            "Preserve Original Text",
            value=True,
            help="Keep original text for evidence purposes"
        )
        
        st.markdown("---")
        
        # Process button
        if st.button("🚀 Process FIR", type="primary", use_container_width=True):
            if not fir_text or len(fir_text) < 50:
                st.error("❌ Please enter at least 50 characters of FIR text")
            elif not is_healthy:
                st.error("❌ Backend is offline. Please start the backend service.")
            else:
                # Process FIR
                start_time = time.time()
                success, result = process_fir(fir_text, filename, preserve_original)
                processing_time = time.time() - start_time
                
                if success:
                    # Store in session
                    st.session_state.processed_cases.append({
                        'timestamp': datetime.now(),
                        'case_id': result['case_id'],
                        'result': result
                    })
                    st.session_state.current_result = result
                    
                    st.success(f"✅ Processing complete in {processing_time:.2f}s")
                    st.balloons()
                else:
                    st.error(f"❌ Processing failed: {result.get('detail', 'Unknown error')}")
    
    # Display results
    if 'current_result' in st.session_state:
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        result = st.session_state.current_result
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Case ID", result['case_id'][:8] + "...")
        with col2:
            st.metric("Language", result['language'])
        with col3:
            st.metric("Confidence", f"{result['extracted_data']['overall_confidence']:.1%}")
        with col4:
            st.metric("Processing Time", f"{result['processing_time']:.2f}s")
        
        # Warning for partial extraction
        if result.get('partial'):
            st.warning(f"⚠️ Partial extraction. Missing: {', '.join(result['missing_fields'])}")
        
        # Tabs for results
        result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs([
            "📝 Summary", "👤 Extracted Data", "⚖️ Legal Sections", "📄 Raw Data"
        ])
        
        with result_tab1:
            st.subheader("Case Summary")
            st.markdown(result['summary'])
            
            if result.get('translated_text'):
                with st.expander("🔄 Translated Text"):
                    st.text(result['translated_text'])
            
            if result.get('original_text') and preserve_original:
                with st.expander("📜 Original Text"):
                    st.text(result['original_text'])
        
        with result_tab2:
            st.subheader("Extracted Information")
            
            data = result['extracted_data']
            
            # Complainant
            if data.get('complainant'):
                st.markdown("### 👤 Complainant")
                comp = data['complainant']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.text(f"Name: {comp.get('name', 'N/A')}")
                    st.text(f"Age: {comp.get('age', 'N/A')}")
                with col2:
                    st.text(f"Father: {comp.get('father', 'N/A')}")
                    st.text(f"Community: {comp.get('community', 'N/A')}")
                with col3:
                    st.text(f"Occupation: {comp.get('occupation', 'N/A')}")
                    st.metric("Confidence", f"{comp.get('confidence', 0):.1%}")
            
            # Incident Details
            st.markdown("### 📅 Incident Details")
            col1, col2 = st.columns(2)
            with col1:
                st.text(f"Date/Time: {data.get('datetime', 'N/A')}")
                st.text(f"Place: {data.get('place', 'N/A')}")
            with col2:
                st.text(f"Location Type: {data.get('location_type', 'N/A')}")
                st.text(f"Emotional Tone: {data.get('emotional_tone', 'N/A')}")
            
            # Accused
            if data.get('accused'):
                st.markdown("### 🔴 Accused")
                for i, accused in enumerate(data['accused'], 1):
                    with st.expander(f"Accused {i}: {accused.get('name', 'Unknown')}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.text(f"Age: {accused.get('age', 'N/A')}")
                            st.text(f"Relation: {accused.get('relation', 'N/A')}")
                        with col2:
                            st.text(f"Status: {accused.get('identification_status', 'N/A')}")
                            st.text(f"Description: {accused.get('description', 'N/A')}")
            
            # Offences
            if data.get('offences'):
                st.markdown("### ⚠️ Offences")
                for offence in data['offences']:
                    st.markdown(f"- {offence}")
            
            # Vehicles
            if data.get('vehicles'):
                st.markdown("### 🚗 Vehicles")
                vehicle_data = []
                for v in data['vehicles']:
                    vehicle_data.append({
                        'Registration': v.get('registration', 'N/A'),
                        'Type': v.get('vehicle_type', 'N/A'),
                        'Color': v.get('color', 'N/A')
                    })
                st.dataframe(vehicle_data, use_container_width=True)
            
            # Weapons
            if data.get('weapons'):
                st.markdown("### 🔫 Weapons")
                for w in data['weapons']:
                    st.text(f"- {w.get('weapon_type', 'Unknown')} ({w.get('legal_status', 'Unknown')})")
            
            # Property Loss
            if data.get('property_loss'):
                st.markdown("### 💰 Property Loss")
                total_loss = sum(p.get('value', 0) for p in data['property_loss'])
                st.metric("Total Loss", f"₹{total_loss:,.2f}")
                
                loss_data = []
                for p in data['property_loss']:
                    loss_data.append({
                        'Item': p.get('item', 'Unknown'),
                        'Value': f"₹{p.get('value', 0):,.2f}"
                    })
                st.dataframe(loss_data, use_container_width=True)
            
            # Injuries
            if data.get('injuries'):
                st.markdown("### 🩹 Injuries")
                for inj in data['injuries']:
                    st.text(f"- {inj.get('injury_type', 'Unknown')} ({inj.get('severity', 'Unknown')})")
            
            # Witnesses
            if data.get('witnesses'):
                st.markdown("### 👁️ Witnesses")
                for witness in data['witnesses']:
                    st.text(f"- {witness}")
            
            # Threats
            if data.get('threats'):
                st.markdown("### ⚠️ Threats")
                for threat in data['threats']:
                    st.text(f"- {threat}")
        
        with result_tab3:
            st.subheader("⚖️ Legal Sections Applied")
            
            for section in result['legal_sections']:
                with st.expander(f"**{section['act']} - Section {section['section']}** ({section.get('severity', 'N/A')})"):
                    st.markdown(f"**Description:** {section['description']}")
                    st.markdown(f"**Reasoning:** {section['reasoning']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Confidence", f"{section.get('confidence', 0):.1%}")
                    with col2:
                        if section.get('keywords_found'):
                            st.text("Keywords: " + ", ".join(section['keywords_found']))
                    
                    # Legal grounding
                    if section.get('legal_grounding'):
                        grounding = section['legal_grounding']
                        st.markdown("**Knowledge Base Grounding:**")
                        st.text(f"Primary Source: {grounding.get('primary_source', 'N/A')}")
                        st.metric("KB Confidence", f"{grounding.get('confidence_score', 0):.1%}")
        
        with result_tab4:
            st.subheader("📄 Raw JSON")
            st.json(result)
            
            # Download button
            json_str = json.dumps(result, indent=2)
            st.download_button(
                "💾 Download JSON",
                data=json_str,
                file_name=f"case_{result['case_id'][:8]}.json",
                mime="application/json"
            )

# ------------------ TAB 2: KNOWLEDGE BASE INGESTION ------------------

with tab2:
    st.header("📚 Knowledge Base Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Legal Documents")
        
        uploaded_files = st.file_uploader(
            "Select files to ingest",
            type=['pdf', 'txt', 'docx', 'md'],
            accept_multiple_files=True,
            help="Upload legal documents, acts, sections, or case laws"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} file(s) selected")
            
            for file in uploaded_files:
                st.text(f"- {file.name} ({file.size / 1024:.1f} KB)")
    
    with col2:
        st.subheader("Ingestion Options")
        
        st.markdown("**Supported Formats:**")
        st.text("✅ PDF")
        st.text("✅ TXT")
        st.text("✅ DOCX")
        st.text("✅ Markdown")
        
        st.markdown("---")
        
        if st.button("📥 Ingest Files", type="primary", use_container_width=True):
            if not uploaded_files:
                st.error("❌ Please select files to ingest")
            elif not is_healthy:
                st.error("❌ Backend is offline")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    success, result = ingest_document(file)
                    
                    results.append({
                        'file': file.name,
                        'success': success,
                        'result': result
                    })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.empty()
                progress_bar.empty()
                
                # Display results
                st.markdown("### Ingestion Results")
                
                success_count = sum(1 for r in results if r['success'])
                st.metric("Success Rate", f"{success_count}/{len(results)}")
                
                for r in results:
                    if r['success']:
                        st.success(f"✅ {r['file']}: {r['result']['chunks_stored']} chunks")
                    else:
                        st.error(f"❌ {r['file']}: {r['result']}")
    
    st.markdown("---")
    
    # KB Statistics
    if is_healthy and health_data:
        st.subheader("📊 Knowledge Base Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Documents", health_data.get('kb_document_count', 0))
        with col2:
            st.metric("Embedding Model", health_data.get('components', {}).get('embedding_model', 'N/A'))
        with col3:
            st.metric("Vector DB", "ChromaDB" if health_data.get('components', {}).get('chromadb') else "Offline")

# ------------------ TAB 3: QUERY KB ------------------

with tab3:
    st.header("🔍 Query Knowledge Base")
    
    st.info("💡 Ask questions about legal sections, acts, or procedures stored in the knowledge base")
    
    query_text = st.text_input(
        "Enter your query:",
        placeholder="e.g., What are the provisions for illegal weapons under Arms Act?",
        key="kb_query"
    )
    
    num_results = st.slider("Number of results:", 1, 10, 3)
    
    if st.button("🔍 Search", type="primary"):
        if not query_text:
            st.error("❌ Please enter a query")
        elif not is_healthy:
            st.error("❌ Backend is offline")
        else:
            success, result = query_kb(query_text, num_results)
            
            if success:
                st.success("✅ Query completed")
                
                # Display answer
                st.markdown("### 💬 Answer")
                st.markdown(result['answer'])
                
                # Display sources
                if result.get('sources'):
                    st.markdown("### 📚 Sources")
                    
                    for i, source in enumerate(result['sources'], 1):
                        with st.expander(f"Source {i}: {source['filename']} (Relevance: {source['relevance']:.1%})"):
                            st.text(f"Relevance Score: {source['relevance']:.3f}")
            else:
                st.error(f"❌ Query failed: {result}")

# ------------------ TAB 4: CASE HISTORY ------------------

with tab4:
    st.header("📂 Case History")
    
    if st.session_state.processed_cases:
        st.info(f"📊 Total cases processed in this session: {len(st.session_state.processed_cases)}")
        
        # Display cases
        for i, case in enumerate(reversed(st.session_state.processed_cases), 1):
            with st.expander(f"Case {i}: {case['case_id'][:8]}... ({case['timestamp'].strftime('%Y-%m-%d %H:%M:%S')})"):
                result = case['result']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Language", result['language'])
                with col2:
                    st.metric("Confidence", f"{result['extracted_data']['overall_confidence']:.1%}")
                with col3:
                    st.metric("Legal Sections", len(result['legal_sections']))
                
                # View full result button
                if st.button(f"View Full Analysis", key=f"view_{i}"):
                    st.session_state.current_result = result
                    st.rerun()
                
                # Download button
                json_str = json.dumps(result, indent=2)
                st.download_button(
                    "💾 Download",
                    data=json_str,
                    file_name=f"case_{result['case_id'][:8]}.json",
                    mime="application/json",
                    key=f"download_{i}"
                )
    else:
        st.info("📭 No cases processed yet. Go to 'Process FIR' tab to analyze your first case.")
        
        # Example FIR
        with st.expander("📋 Try with Example FIR"):
            st.code("""
I, Ramesh Kumar S/o Venkat Rao, aged about 35 years, residing at H.No. 12-34, 
Gandhi Nagar, Hyderabad, hereby state that on 15-10-2024 at about 10:30 PM, 
while I was returning home from work, three unknown persons stopped me near 
the bus stop at Gandhi Nagar. They forcibly snatched my mobile phone worth 
₹15,000 and cash ₹12,500 from my pocket. One of them was carrying a country-made 
pistol and threatened me saying "మళ్లీ పోలీసులకు చెబితే చంపేస్తాం" 
(If you tell police again, we will kill you). They also abused me with caste names. 
I belong to Scheduled Caste community. I sustained minor injuries on my hand 
while trying to resist. Two witnesses, Mr. Suresh and Mr. Kumar, were present 
at the scene.
            """)
            st.caption("Copy this text and paste it in the 'Process FIR' tab")

# ------------------ FOOTER ------------------

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>⚖️ <strong>DHARMA Legal Analysis System v1.0</strong></p>
    <p>Powered by Qwen LLM | ChromaDB | Sentence Transformers</p>
</div>
""", unsafe_allow_html=True)