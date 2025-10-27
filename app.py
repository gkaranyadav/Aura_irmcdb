# app.py - IRMC AskPro ⚡ with OCR & Voice Features
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
import pytesseract
from PIL import Image
import pdf2image

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
# IMPROVED DOCUMENT PROCESSOR WITH OCR
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
    
    def extract_text_with_ocr(self, pdf_path):
        """Extract text using OCR for scanned PDFs"""
        try:
            st.info("🔍 PDF appears to be scanned/image-based. Using OCR...")
            
            # Convert PDF to images
            images = pdf2image.convert_from_path(pdf_path, dpi=200)
            extracted_text = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, image in enumerate(images):
                status_text.text(f"📷 OCR processing page {i+1}/{len(images)}...")
                progress_bar.progress((i + 1) / len(images))
                
                # Use pytesseract to do OCR on the image
                text = pytesseract.image_to_string(image)
                if text.strip():
                    extracted_text.append({
                        'page': i + 1,
                        'text': text.strip()
                    })
            
            status_text.text("✅ OCR completed!")
            return extracted_text
            
        except Exception as e:
            st.error(f"❌ OCR failed: {e}")
            return []
    
    def extract_text_direct(self, pdf_path):
        """Extract text directly from PDF (for text-based PDFs)"""
        try:
            reader = PdfReader(pdf_path)
            extracted_text = []
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    extracted_text.append({
                        'page': page_num,
                        'text': text.strip()
                    })
            
            return extracted_text
        except Exception as e:
            st.error(f"❌ Direct text extraction failed: {e}")
            return []
    
    def analyze_pdf_structure(self, pdf_path):
        """Analyze PDF to determine if it's text-based or scanned"""
        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            text_pages = 0
            
            for page in reader.pages:
                text = page.extract_text() or ""
                if len(text.strip()) > 100:  # Substantial text found
                    text_pages += 1
            
            text_ratio = text_pages / total_pages if total_pages > 0 else 0
            
            if text_ratio > 0.7:
                return "text_based", text_ratio
            else:
                return "scanned", text_ratio
                
        except Exception as e:
            return "unknown", 0
    
    def process_pdf(self, pdf_file):
        try:
            # Show processing status
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_path = tmp_file.name
            
            # Analyze PDF type
            status_text.text("🔍 Analyzing PDF structure...")
            pdf_type, text_ratio = self.analyze_pdf_structure(tmp_path)
            
            st.info(f"📊 PDF Analysis: {pdf_type.upper()} (Text ratio: {text_ratio:.1%})")
            
            # Extract text based on PDF type
            if pdf_type == "text_based":
                status_text.text("📖 Extracting text directly...")
                extracted_data = self.extract_text_direct(tmp_path)
                method = "direct_extraction"
            else:
                status_text.text("📷 Using OCR for scanned PDF...")
                extracted_data = self.extract_text_with_ocr(tmp_path)
                method = "ocr"
            
            # Process extracted text
            self.chunks = []
            self.chunk_metadata = []
            
            if extracted_data:
                status_text.text("✂️ Chunking text...")
                
                for item in extracted_data:
                    text = item['text']
                    page_num = item['page']
                    
                    # Split into paragraphs and clean
                    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
                    
                    for para in paragraphs:
                        # Further split long paragraphs
                        if len(para) > 500:
                            sentences = para.split('. ')
                            current_chunk = ""
                            for sentence in sentences:
                                if len(current_chunk + sentence) < 500:
                                    current_chunk += sentence + ". "
                                else:
                                    if current_chunk:
                                        self.chunks.append(current_chunk.strip())
                                        self.chunk_metadata.append({
                                            "page": page_num,
                                            "chunk_id": f"page_{page_num}_chunk_{len(self.chunks)}",
                                            "method": method
                                        })
                                    current_chunk = sentence + ". "
                            if current_chunk:
                                self.chunks.append(current_chunk.strip())
                                self.chunk_metadata.append({
                                    "page": page_num,
                                    "chunk_id": f"page_{page_num}_chunk_{len(self.chunks)}",
                                    "method": method
                                })
                        else:
                            self.chunks.append(para)
                            self.chunk_metadata.append({
                                "page": page_num,
                                "chunk_id": f"page_{page_num}_chunk_{len(self.chunks)}",
                                "method": method
                            })
            
            # Create embeddings if we have chunks
            if self.chunks:
                status_text.text("🧠 Creating embeddings...")
                embeddings = self.embedder.encode(self.chunks)
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
                self.index.add(np.array(embeddings))
                
                status_text.text("✅ Complete!")
                progress_bar.progress(100)
                
                # Show detailed processing results
                st.success(f"""
                ✅ **Processing Complete!**
                - **Pages processed:** {len(extracted_data)}
                - **Text chunks created:** {len(self.chunks)}
                - **Extraction method:** {method}
                - **Vector database:** Ready with {embeddings.shape[1]} dimensions
                """)
                
                # Debug information
                with st.expander("🔍 Debug Information"):
                    st.write(f"**First 3 chunks preview:**")
                    for i, chunk in enumerate(self.chunks[:3]):
                        st.write(f"**Chunk {i+1}** (Page {self.chunk_metadata[i]['page']}):")
                        st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                
                return len(self.chunks)
            else:
                st.error("""
                ❌ **No readable text found!**
                
                **Possible reasons:**
                1. PDF is password protected
                2. PDF contains only images (no text layer)
                3. PDF is corrupted
                4. Text is in unsupported language
                
                **Try:**
                - Different PDF file
                - PDF with selectable text
                - Simple text document instead
                """)
                return 0
                
        except Exception as e:
            st.error(f"❌ PDF processing failed: {str(e)}")
            
            # Detailed error information
            with st.expander("🐛 Error Details"):
                st.write("**Error type:**", type(e).__name__)
                st.write("**Error message:**", str(e))
                st.write("**File info:**", f"{pdf_file.name} ({len(pdf_file.getvalue())} bytes)")
            
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
            # Show search in progress
            search_status = st.empty()
            search_status.text("🔍 Searching for relevant content...")
            
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
            
            search_status.empty()
            
            # Debug search results
            if results:
                st.info(f"🎯 Found {len(results)} relevant sections (Best match: {results[0]['similarity']:.1%})")
            else:
                st.warning("🔍 No relevant content found for your query")
            
            return results
        except Exception as e:
            st.error(f"🔍 Search error: {e}")
            return []

# =============================================================================
# IMPROVED LLM SERVICE WITH BETTER ANSWERS
# =============================================================================
class LLMService:
    def generate_answer(self, question, context_chunks):
        try:
            if not context_chunks:
                return self._get_no_results_response(question), 0.0
            
            # Calculate overall confidence
            avg_confidence = sum(chunk['similarity'] for chunk in context_chunks) / len(context_chunks)
            
            # Generate comprehensive answer
            answer = self._format_comprehensive_answer(question, context_chunks, avg_confidence)
            
            return answer, avg_confidence
            
        except Exception as e:
            return f"❌ Error generating answer: {str(e)}", 0.0
    
    def _get_no_results_response(self, question):
        """Generate helpful response when no results found"""
        return f"""
**🤔 Your Question:** {question}

**🔍 Search Results:** No relevant information found in the document.

**💡 Suggestions:**
1. **Try different keywords** - Use specific terms from the document
2. **Rephrase your question** - Be more specific or general
3. **Check document type** - Make sure the PDF contains readable text
4. **Try these example questions:**
   - "What is this document about?"
   - "List the main topics"
   - "Summarize the key points"

**Need help?** Upload a different PDF or contact support.
"""
    
    def _format_comprehensive_answer(self, question, context_chunks, avg_confidence):
        """Format a comprehensive answer with extracted information"""
        
        # Extract key information
        summary = self._extract_key_information(context_chunks)
        
        answer = f"""
**🤔 Your Question:** {question}

**📊 Search Results:** Found {len(context_chunks)} relevant sections with {avg_confidence:.1%} average confidence.

**💡 Comprehensive Answer:**

{summary}

**🔍 Source Details:**
"""
        
        # Add detailed source information
        for i, chunk in enumerate(context_chunks):
            content_preview = chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
            extraction_method = "📖 Text" if chunk['metadata'].get('method') == 'direct_extraction' else "📷 OCR"
            
            answer += f"""
**{i+1}. Page {chunk['metadata']['page']}** {extraction_method} | Confidence: {chunk['similarity']:.1%}
{content_preview}
"""
        
        # Add confidence assessment
        answer += f"""
**🎯 Confidence Assessment:** {self._get_confidence_assessment(avg_confidence)}
"""
        
        return answer
    
    def _extract_key_information(self, context_chunks):
        """Extract and summarize key information from chunks"""
        if not context_chunks:
            return "No specific information extracted from the document."
        
        # Combine all content for analysis
        all_content = " ".join([chunk['content'] for chunk in context_chunks])
        
        # Simple extraction of key sentences (you can enhance this)
        sentences = [s.strip() for s in all_content.split('.') if len(s.strip()) > 20]
        key_sentences = sentences[:5]  # Take first 5 substantial sentences
        
        summary = "Based on the document content, here are the key points:\n\n"
        for i, sentence in enumerate(key_sentences, 1):
            summary += f"• {sentence}.\n"
        
        return summary
    
    def _get_confidence_assessment(self, confidence):
        """Get confidence level description"""
        if confidence > 0.8:
            return "✅ **Very High Confidence** - Information is highly relevant and accurate"
        elif confidence > 0.6:
            return "✅ **High Confidence** - Information is relevant and reliable"
        elif confidence > 0.4:
            return "⚠️ **Medium Confidence** - Information is somewhat relevant"
        elif confidence > 0.2:
            return "🔍 **Low Confidence** - Limited relevant information found"
        else:
            return "❌ **Very Low Confidence** - Information may not be relevant"

# =============================================================================
# VOICE SERVICE (KEEPING THE SAME)
# =============================================================================
class VoiceService:
    def __init__(self):
        self.voice_enabled = True
    
    def text_to_speech_html(self, text):
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
                const input = document.querySelector('input[data-testid="stChatInput"]');
                if (input) {
                    input.value = transcript;
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                }
            };
        }
        startSpeechToText();
        </script>
        """
        return html

# =============================================================================
# DATABRICKS BACKEND (KEEPING THE SAME)
# =============================================================================
class DatabricksBackend:
    def __init__(self):
        try:
            self.ws = WorkspaceClient(host=Config.DATABRICKS_HOST, token=Config.DATABRICKS_TOKEN)
            self.session_id = str(uuid.uuid4())
        except Exception as e:
            st.error(f"❌ Databricks connection failed: {e}")
    
    def log_query(self, question, answer, confidence, sources, response_time):
        try:
            logging.info(f"Query: {question} | Confidence: {confidence} | Time: {response_time}ms")
            return True
        except Exception as e:
            logging.error(f"Logging error: {e}")
            return False

# =============================================================================
# STREAMLIT APP (KEEPING THE SAME STRUCTURE)
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
    
    def render_sidebar(self):
        with st.sidebar:
            st.title("📚 IRMC AskPro")
            st.markdown("---")
            
            uploaded_file = st.file_uploader("Upload PDF", type="pdf")
            if uploaded_file:
                if st.button("🔄 Process Document", type="primary", use_container_width=True):
                    with st.spinner("Processing..."):
                        chunk_count = self.doc_processor.process_pdf(uploaded_file)
                        if chunk_count > 0:
                            st.session_state.pdf_processed = True
                            st.session_state.pdf_name = uploaded_file.name
                            st.rerun()
            
            st.markdown("---")
            top_k = st.slider("Sources to retrieve", 1, 5, 3)
            enable_voice = st.checkbox("Enable Voice Features", value=True)
            
            return top_k, enable_voice
    
    def render_chat_interface(self, top_k, enable_voice):
        st.title("IRMC AskPro ⚡")
        st.markdown("Ask questions about your documents and get AI-powered answers")
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'pdf_processed' not in st.session_state:
            st.session_state.pdf_processed = False
        
        if not st.session_state.pdf_processed:
            st.info("📄 Upload and process a PDF in the sidebar to get started")
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        question = st.chat_input("Ask a question...")
        
        if question and st.session_state.pdf_processed:
            self.process_question(question, top_k, enable_voice)
        elif question:
            st.error("Please upload and process a PDF first!")
    
    def process_question(self, question, top_k, enable_voice):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching..."):
                start_time = time.time()
                context_chunks = self.doc_processor.search_similar(question, top_k)
                answer, confidence = self.llm_service.generate_answer(question, context_chunks)
                response_time = (time.time() - start_time) * 1000
            
            st.markdown(answer)
            st.caption(f"⏱️ {response_time:.0f}ms | Confidence: {confidence:.1%}")
            
            if enable_voice and confidence > 0.5:
                js_code = self.voice_service.text_to_speech_html("I found some information in the document.")
                st.components.v1.html(js_code, height=0)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
    
    def run(self):
        top_k, enable_voice = self.render_sidebar()
        self.render_chat_interface(top_k, enable_voice)

if __name__ == "__main__":
    app = AuraPDFQAApp()
    app.run()
