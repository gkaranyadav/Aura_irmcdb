# app.py - IRMC AskPro ⚡ with Voice Features
import streamlit as st
import tempfile
import os
from datetime import datetime
import uuid
import logging
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from databricks.sdk import WorkspaceClient
import mlflow
import time
import requests
import json
import base64
import io

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    DATABRICKS_HOST = "https://dbc-484c2988-d6e6.cloud.databricks.com"
    DATABRICKS_TOKEN = "dapiaa126510ff360ca569ec6d125bc727d1"
    HF_TOKEN = "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"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
    TOP_K_CHUNKS = 3
    CHUNK_SIZE = 500

# =============================================================================
# VOICE SERVICE (TEXT-TO-SPEECH)
# =============================================================================
class VoiceService:
    def __init__(self):
        self.voice_enabled = True
    
    def text_to_speech_html(self, text):
        """Generate HTML for text-to-speech using browser's speech synthesis"""
        clean_text = text.replace('"', '\\"').replace("'", "\\'")
        html = f"""
        <script>
        function speakText() {{
            if ('speechSynthesis' in window) {{
                const utterance = new SpeechSynthesisUtterance();
                utterance.text = "{clean_text}";
                utterance.rate = 1.0;
                utterance.pitch = 1.0;
                utterance.volume = 0.8;
                window.speechSynthesis.speak(utterance);
            }} else {{
                alert("Text-to-speech not supported in your browser.");
            }}
        }}
        speakText();
        </script>
        """
        return html
    
    def speech_to_text_html(self):
        """Generate HTML for speech-to-text"""
        html = """
        <script>
        function startSpeechToText() {
            if (!('webkitSpeechRecognition' in window)) {
                alert("Speech recognition not supported in your browser.");
                return;
            }
            
            const recognition = new webkitSpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-US';
            
            recognition.start();
            
            recognition.onresult = function(event) {
                const transcript = event.results[0][0].transcript;
                // Send transcript to Streamlit
                const input = document.querySelector('input[data-testid="stChatInput"]');
                if (input) {
                    input.value = transcript;
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                }
            };
            
            recognition.onerror = function(event) {
                console.error('Speech recognition error', event.error);
            };
        }
        </script>
        """
        return html

# =============================================================================
# DATABRICKS BACKEND
# =============================================================================
class DatabricksBackend:
    def __init__(self):
        try:
            self.ws = WorkspaceClient(host=Config.DATABRICKS_HOST, token=Config.DATABRICKS_TOKEN)
            self.session_id = str(uuid.uuid4())
            st.success("✅ Connected to Databricks")
        except Exception as e:
            st.error(f"❌ Databricks connection failed: {e}")
    
    def log_query(self, question, answer, confidence, sources, response_time):
        try:
            # Simple logging - you can enhance this with MLflow
            logging.info(f"Query: {question} | Confidence: {confidence} | Time: {response_time}ms")
            return True
        except Exception as e:
            logging.error(f"Logging error: {e}")
            return False

# =============================================================================
# IMPROVED DOCUMENT PROCESSOR
# =============================================================================
class DocumentProcessor:
    def __init__(self):
        try:
            self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
            self.index = None
            self.chunks = []
            self.chunk_metadata = []
            st.success("✅ Document processor ready")
        except Exception as e:
            st.error(f"❌ Document processor failed: {e}")
    
    def process_pdf(self, pdf_file):
        try:
            # Show processing status
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_path = tmp_file.name
            
            # Read PDF
            status_text.text("📖 Reading PDF...")
            reader = PdfReader(tmp_path)
            
            self.chunks = []
            self.chunk_metadata = []
            total_pages = len(reader.pages)
            
            # Process each page
            for page_num, page in enumerate(reader.pages, 1):
                status_text.text(f"🔍 Processing page {page_num}/{total_pages}...")
                progress_bar.progress(page_num / total_pages)
                
                text = page.extract_text() or ""
                
                if text.strip():
                    # Clean and chunk text
                    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
                    
                    for para in paragraphs:
                        if len(para) > 50:  # Only keep substantial paragraphs
                            self.chunks.append(para)
                            self.chunk_metadata.append({
                                "page": page_num,
                                "chunk_id": f"page_{page_num}_para_{len(self.chunks)}",
                                "type": "paragraph"
                            })
            
            # Create embeddings if we have chunks
            if self.chunks:
                status_text.text("🧠 Creating embeddings...")
                embeddings = self.embedder.encode(self.chunks)
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
                self.index.add(np.array(embeddings))
                
                status_text.text("✅ Complete!")
                progress_bar.progress(100)
                
                st.success(f"✅ Processed {len(self.chunks)} text chunks from {total_pages} pages")
                return len(self.chunks)
            else:
                st.error("❌ No readable text found in PDF. The PDF might be scanned or image-based.")
                return 0
                
        except Exception as e:
            st.error(f"❌ PDF processing failed: {str(e)}")
            return 0
        finally:
            # Cleanup
            try:
                os.unlink(tmp_path)
            except:
                pass
            try:
                progress_bar.empty()
                status_text.empty()
            except:
                pass
    
    def search_similar(self, query, top_k=3):
        if self.index is None or len(self.chunks) == 0:
            return []
        
        try:
            query_embedding = self.embedder.encode([query])
            distances, indices = self.index.search(np.array(query_embedding), top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunks):
                    similarity = 1 / (1 + distances[0][i])
                    results.append({
                        "content": self.chunks[idx],
                        "metadata": self.chunk_metadata[idx],
                        "similarity": round(similarity, 3)
                    })
            
            # Sort by similarity (highest first)
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results
        except Exception as e:
            st.error(f"🔍 Search error: {e}")
            return []

# =============================================================================
# IMPROVED LLM SERVICE
# =============================================================================
class LLMService:
    def generate_answer(self, question, context_chunks):
        try:
            if not context_chunks:
                return "I couldn't find relevant information in the document to answer your question. \n\n**Tips:**\n- Try different keywords\n- Ask about general topics in the document\n- Check if the PDF contains readable text", 0.0
            
            # Calculate overall confidence
            avg_confidence = sum(chunk['similarity'] for chunk in context_chunks) / len(context_chunks)
            
            # Prepare detailed answer
            answer = f"""**🤔 Your Question:** {question}

**📊 Search Results:** Found {len(context_chunks)} relevant sections with {avg_confidence:.1%} average confidence.

**💡 Answer Based on Document:**\n"""
            
            # Extract key information from each chunk
            for i, chunk in enumerate(context_chunks):
                content_preview = chunk['content'][:150] + "..." if len(chunk['content']) > 150 else chunk['content']
                answer += f"\n**{i+1}. Page {chunk['metadata']['page']}** (Confidence: {chunk['similarity']:.1%})\n"
                answer += f"   {content_preview}\n"
            
            answer += f"\n**🎯 Overall Confidence Score:** {avg_confidence:.1%}"
            
            # Add suggestions based on confidence
            if avg_confidence > 0.7:
                answer += "\n\n✅ **High confidence** - This information is very relevant to your question."
            elif avg_confidence > 0.4:
                answer += "\n\n⚠️ **Medium confidence** - This information is somewhat relevant."
            else:
                answer += "\n\n🔍 **Low confidence** - Consider rephrasing your question for better results."
            
            return answer, avg_confidence
            
        except Exception as e:
            return f"❌ Error generating answer: {str(e)}", 0.0

# =============================================================================
# ENHANCED STREAMLIT APP
# =============================================================================
class AuraPDFQAApp:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.llm_service = LLMService()
        self.db_backend = DatabricksBackend()
        self.voice_service = VoiceService()
        self.setup_ui()
    
    def setup_ui(self):
        st.set_page_config(
            page_title="IRMC AskPro ⚡ - Smart Document QA",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .feature-card {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f0f2f6;
            margin: 0.5rem 0;
        }
        .voice-btn {
            background-color: #ff4b4b;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            cursor: pointer;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        with st.sidebar:
            st.markdown('<div class="main-header">📚 IRMC AskPro</div>', unsafe_allow_html=True)
            st.markdown("---")
            
            # File upload section
            st.subheader("📄 Upload Document")
            uploaded_file = st.file_uploader(
                "Choose PDF file", 
                type="pdf",
                help="Upload a PDF document to analyze",
                key="pdf_uploader"
            )
            
            if uploaded_file is not None:
                col1, col2 = st.columns([2, 1])
                with col1:
                    file_size = len(uploaded_file.getvalue()) / 1024
                    st.info(f"**File:** {uploaded_file.name}\n**Size:** {file_size:.1f} KB")
                with col2:
                    if st.button("🔄 Process", use_container_width=True, type="primary"):
                        with st.spinner("Processing..."):
                            chunk_count = self.doc_processor.process_pdf(uploaded_file)
                            if chunk_count > 0:
                                st.session_state.pdf_processed = True
                                st.session_state.pdf_name = uploaded_file.name
                                st.session_state.messages = []
                                st.rerun()
                
                if st.session_state.get('pdf_processed', False):
                    st.success(f"✅ **Ready:** {st.session_state.get('pdf_name', 'Document')}")
            
            st.markdown("---")
            
            # Settings
            st.subheader("⚙️ Settings")
            top_k = st.slider("Sources to retrieve", 1, 5, 3)
            enable_voice = st.checkbox("Enable Voice Features", value=True)
            
            st.markdown("---")
            
            # Quick actions
            st.subheader("🚀 Actions")
            if st.button("🆕 New Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
            
            if st.button("🗑️ Clear Document", use_container_width=True):
                st.session_state.pdf_processed = False
                self.doc_processor = DocumentProcessor()
                st.rerun()
            
            return top_k, enable_voice
    
    def render_voice_features(self, enable_voice):
        """Render voice input/output features"""
        if enable_voice:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🎤 Voice Input", use_container_width=True):
                    # Inject JavaScript for speech-to-text
                    js_code = self.voice_service.speech_to_text_html()
                    st.components.v1.html(js_code, height=0)
                    st.info("🎤 Click the microphone icon in your browser and speak...")
            
            with col2:
                if st.session_state.get('last_answer'):
                    if st.button("🔊 Speak Answer", use_container_width=True):
                        # Inject JavaScript for text-to-speech
                        js_code = self.voice_service.text_to_speech_html(st.session_state.last_answer)
                        st.components.v1.html(js_code, height=0)
                        st.success("🔊 Speaking answer...")
    
    def render_chat_interface(self, top_k, enable_voice):
        st.markdown('<div class="main-header">IRMC AskPro ⚡</div>', unsafe_allow_html=True)
        st.markdown("### Ask questions about your documents and get AI-powered answers")
        
        # Initialize session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'pdf_processed' not in st.session_state:
            st.session_state.pdf_processed = False
        if 'last_answer' not in st.session_state:
            st.session_state.last_answer = ""
        
        # Show upload reminder
        if not st.session_state.pdf_processed:
            st.markdown("---")
            st.info("""
            **📋 To get started:**
            1. **Upload a PDF document** in the sidebar
            2. **Click 'Process'** to analyze the content  
            3. **Ask questions** about the document below
            4. **Get instant answers** with source references
            """)
            
            # Example questions
            st.markdown("**💡 Example questions to try:**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("What is this document about?"):
                    st.session_state.example_question = "What is this document about?"
            with col2:
                if st.button("List the main topics"):
                    st.session_state.example_question = "List the main topics"
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Voice features
        self.render_voice_features(enable_voice)
        
        # Chat input
        question = st.chat_input("Ask a question about your document...")
        
        # Use example question if set
        if hasattr(st.session_state, 'example_question'):
            question = st.session_state.example_question
            del st.session_state.example_question
        
        # Process question
        if question and st.session_state.pdf_processed:
            self.process_question(question, top_k, enable_voice)
        elif question and not st.session_state.pdf_processed:
            st.error("❌ Please upload and process a PDF document first!")
    
    def process_question(self, question, top_k, enable_voice):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching document..."):
                start_time = time.time()
                
                # Search for relevant content
                context_chunks = self.doc_processor.search_similar(question, top_k)
                
                # Generate answer
                answer, confidence = self.llm_service.generate_answer(question, context_chunks)
                
                # Calculate response time
                response_time = (time.time() - start_time) * 1000
            
            # Display confidence badge
            if confidence > 0.7:
                st.success(f"✅ High Confidence: {confidence:.1%}")
            elif confidence > 0.4:
                st.warning(f"⚠️ Medium Confidence: {confidence:.1%}")
            else:
                st.error(f"🔍 Low Confidence: {confidence:.1%}")
            
            # Display answer
            st.markdown(answer)
            
            # Show response time
            st.caption(f"⏱️ Response time: {response_time:.0f}ms")
            
            # Display sources if available
            if context_chunks:
                with st.expander(f"📋 View Source Documents ({len(context_chunks)} found)"):
                    for i, chunk in enumerate(context_chunks):
                        st.markdown(f"**Source {i+1}** | Page {chunk['metadata']['page']} | Confidence: `{chunk['similarity']:.1%}`")
                        st.text_area(
                            f"Content {i+1}",
                            chunk['content'],
                            height=120,
                            key=f"source_{i}_{int(time.time())}",
                            label_visibility="collapsed"
                        )
                        st.markdown("---")
            
            # Store last answer for voice features
            st.session_state.last_answer = answer
            
            # Log the query
            self.db_backend.log_query(question, answer, confidence, context_chunks, response_time)
        
        # Add to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # Auto-speak if voice enabled and high confidence
        if enable_voice and confidence > 0.6:
            js_code = self.voice_service.text_to_speech_html("Here's what I found in the document.")
            st.components.v1.html(js_code, height=0)
    
    def run(self):
        top_k, enable_voice = self.render_sidebar()
        self.render_chat_interface(top_k, enable_voice)

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    app = AuraPDFQAApp()
    app.run()
