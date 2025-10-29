# app.py - Aura PDF QA ⚡
import streamlit as st
import tempfile
import os
import time
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pytesseract
from PIL import Image
import pdf2image
from gtts import gTTS

# =============================================================================
# CONFIG
# =============================================================================
class Config:
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    TOP_K_CHUNKS = 3
    CHUNK_SIZE = 500
    MIN_PARAGRAPH_LENGTH = 20

# =============================================================================
# DOCUMENT PROCESSOR
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

    def extract_text_direct(self, pdf_path):
        """Extract text directly from PDF"""
        try:
            reader = PdfReader(pdf_path)
            extracted = []
            for i, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    extracted.append({"page": i, "text": text.strip()})
            return extracted
        except Exception as e:
            st.error(f"Error in direct text extraction: {e}")
            return []

    def extract_text_ocr(self, pdf_path):
        """Extract text using OCR for scanned PDFs"""
        try:
            images = pdf2image.convert_from_path(pdf_path, dpi=200)
            extracted = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, image in enumerate(images):
                status_text.text(f"📷 OCR page {i+1}/{len(images)}")
                text = pytesseract.image_to_string(image)
                if text.strip():
                    extracted.append({"page": i + 1, "text": text.strip()})
                progress_bar.progress((i + 1) / len(images))
            
            status_text.text("✅ OCR complete")
            return extracted
        except Exception as e:
            st.error(f"Error in OCR extraction: {e}")
            return []

    def analyze_pdf_type(self, pdf_path):
        """Analyze if PDF is text-based or scanned"""
        try:
            reader = PdfReader(pdf_path)
            if len(reader.pages) == 0:
                return "scanned"
            
            text_pages = 0
            for page in reader.pages:
                text = page.extract_text() or ""
                if len(text.strip()) > 50:
                    text_pages += 1
            
            total_pages = len(reader.pages)
            return "text_based" if text_pages / total_pages > 0.5 else "scanned"
        except Exception as e:
            st.error(f"Error analyzing PDF type: {e}")
            return "scanned"  # Default to OCR for safety

    def chunk_text(self, extracted_text, method):
        """Split text into manageable chunks"""
        chunks = []
        metadata = []
        
        for item in extracted_text:
            page = item["page"]
            text = item["text"]
            
            # Split into paragraphs first
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= Config.MIN_PARAGRAPH_LENGTH]
            
            for para in paragraphs:
                if len(para) <= Config.CHUNK_SIZE:
                    # Paragraph is small enough
                    chunks.append(para)
                    metadata.append({"page": page, "method": method})
                else:
                    # Split large paragraphs into sentences
                    sentences = [s.strip() + "." for s in para.split(". ") if s.strip()]
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) <= Config.CHUNK_SIZE:
                            current_chunk += " " + sentence
                        else:
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                                metadata.append({"page": page, "method": method})
                            current_chunk = sentence
                    
                    # Add the last chunk
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                        metadata.append({"page": page, "method": method})
        
        return chunks, metadata

    def process_pdf(self, uploaded_file):
        """Main PDF processing pipeline"""
        # Clear previous data
        self.chunks = []
        self.chunk_metadata = []
        self.index = None
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        try:
            st.info("🔍 Analyzing PDF type...")
            pdf_type = self.analyze_pdf_type(pdf_path)
            st.info(f"📄 PDF type detected: {pdf_type}")

            # Extract text based on PDF type
            if pdf_type == "text_based":
                extracted = self.extract_text_direct(pdf_path)
                method = "text"
            else:
                extracted = self.extract_text_ocr(pdf_path)
                method = "ocr"

            if not extracted:
                st.error("❌ No text could be extracted from the PDF")
                return 0

            st.info(f"✅ Extracted text from {len(extracted)} pages using {method.upper()}")

            # Chunk the text
            self.chunks, self.chunk_metadata = self.chunk_text(extracted, method)
            st.info(f"📊 Created {len(self.chunks)} text chunks")

            if not self.chunks:
                st.error("❌ No valid text chunks created from PDF")
                return 0

            # Create embeddings and FAISS index
            st.info("🧠 Creating embeddings...")
            embeddings = self.embedder.encode(self.chunks)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(np.array(embeddings))
            
            st.success(f"✅ PDF processed successfully: {len(self.chunks)} chunks indexed")
            return len(self.chunks)

        except Exception as e:
            st.error(f"❌ PDF processing failed: {str(e)}")
            return 0
        finally:
            # Clean up temporary file
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def search_similar(self, query, top_k=3):
        """Search for similar text chunks"""
        if not self.chunks or self.index is None:
            return []

        try:
            query_vec = self.embedder.encode([query])
            distances, indices = self.index.search(np.array(query_vec), min(top_k, len(self.chunks)))
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunks):
                    # Convert distance to similarity score (0-1)
                    similarity = 1 / (1 + distances[0][i])
                    results.append({
                        "content": self.chunks[idx],
                        "metadata": self.chunk_metadata[idx],
                        "similarity": round(similarity, 3)
                    })
            
            # Sort by similarity (highest first)
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            st.error(f"Search error: {e}")
            return []

# =============================================================================
# LLM SERVICE
# =============================================================================
class LLMService:
    def generate_answer(self, question, chunks):
        """Generate answer from relevant chunks"""
        if not chunks:
            return "❌ No relevant information found for your question. Please try rephrasing or ask about a different topic.", 0.0
        
        try:
            # Calculate average confidence
            avg_conf = sum(c["similarity"] for c in chunks) / len(chunks)
            
            # Generate summary from chunks
            summary = self._summarize_chunks(chunks)
            
            # Format answer
            answer = f"**Question:** {question}\n\n**Answer:**\n{summary}\n\n**Sources:**\n"
            
            for i, chunk in enumerate(chunks, 1):
                preview = chunk['content'][:150] + "..." if len(chunk['content']) > 150 else chunk['content']
                answer += f"{i}. Page {chunk['metadata']['page']} | Confidence: {chunk['similarity']:.1%}\n   {preview}\n\n"
            
            return answer, avg_conf
            
        except Exception as e:
            return f"❌ Error generating answer: {str(e)}", 0.0

    def _summarize_chunks(self, chunks):
        """Simple summarization of relevant chunks"""
        try:
            all_text = " ".join([c['content'] for c in chunks])
            sentences = [s.strip() for s in all_text.split(". ") if len(s.strip()) > 20]
            
            # Take most relevant sentences (simplified approach)
            key_sentences = sentences[:4]
            summary = " ".join([f"{s}." for s in key_sentences])
            
            return summary if summary else "Based on the document, the relevant information is contained in the sources above."
            
        except Exception as e:
            return "Relevant information found in the sources above."

# =============================================================================
# VOICE SERVICE
# =============================================================================
class VoiceService:
    def speak_text(self, text):
        """Convert text to speech"""
        try:
            # Extract only the answer part for TTS (remove sources)
            if "**Answer:**" in text:
                answer_part = text.split("**Answer:**")[1].split("**Sources:**")[0].strip()
            else:
                answer_part = text
                
            # Limit text length for TTS
            if len(answer_part) > 1000:
                answer_part = answer_part[:1000] + "..."
                
            tts = gTTS(text=answer_part, lang='en', slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tts.save(tmp_file.name)
                st.audio(tmp_file.name, format="audio/mp3")
                
        except Exception as e:
            st.error(f"Text-to-speech error: {e}")

# =============================================================================
# STREAMLIT APP
# =============================================================================
class AuraPDFQAApp:
    def __init__(self):
        self.setup_session_state()
        self.llm_service = LLMService()
        self.voice_service = VoiceService()
        self.setup_ui()

    def setup_session_state(self):
        """Initialize session state variables"""
        if 'pdf_processed' not in st.session_state:
            st.session_state.pdf_processed = False
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'doc_processor' not in st.session_state:
            st.session_state.doc_processor = DocumentProcessor()
        if 'processing_done' not in st.session_state:
            st.session_state.processing_done = False

    def setup_ui(self):
        """Setup Streamlit UI configuration"""
        st.set_page_config(
            page_title="Aura PDF QA ⚡",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def render_sidebar(self):
        """Render the sidebar with controls"""
        st.sidebar.title("📚 Aura PDF QA")
        st.sidebar.markdown("---")
        
        # File upload
        uploaded_file = st.sidebar.file_uploader(
            "Upload PDF Document", 
            type="pdf",
            help="Upload a PDF file to analyze"
        )
        
        # Status display
        st.sidebar.markdown("### Status")
        if st.session_state.pdf_processed:
            st.sidebar.success("✅ PDF Ready for Questions!")
            if hasattr(st.session_state, 'pdf_name'):
                st.sidebar.info(f"**Document:** {st.session_state.pdf_name}")
        else:
            st.sidebar.warning("⚠️ Upload PDF to Start")
        
        # Processing button
        if uploaded_file and not st.session_state.processing_done:
            if st.sidebar.button("🚀 Process Document", type="primary", use_container_width=True):
                with st.spinner("Processing PDF... This may take a few moments"):
                    count = st.session_state.doc_processor.process_pdf(uploaded_file)
                    if count > 0:
                        st.session_state.pdf_processed = True
                        st.session_state.pdf_name = uploaded_file.name
                        st.session_state.processing_done = True
                        st.rerun()
                    else:
                        st.session_state.pdf_processed = False
                        st.session_state.processing_done = False
        
        # Configuration
        st.sidebar.markdown("### Configuration")
        top_k = st.sidebar.slider(
            "Number of sources", 
            min_value=1, 
            max_value=5, 
            value=3,
            help="How many text chunks to use for answering"
        )
        
        enable_voice = st.sidebar.checkbox(
            "Enable Voice Output", 
            value=True,
            help="Convert answers to speech"
        )
        
        # Reset button
        if st.sidebar.button("🔄 Reset Session", use_container_width=True):
            self.reset_session()
            
        return top_k, enable_voice

    def reset_session(self):
        """Reset the session state"""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    def render_chat(self, top_k, enable_voice):
        """Render the main chat interface"""
        st.title("Aura PDF QA ⚡")
        st.markdown("Ask questions about your document and get AI-powered answers with source references")
        st.markdown("---")
        
        # Show processing status
        if not st.session_state.pdf_processed:
            self.render_welcome_screen()
            return

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your document..."):
            self.handle_user_query(prompt, top_k, enable_voice)

    def render_welcome_screen(self):
        """Render welcome screen when no PDF is processed"""
        st.info("👋 Welcome to Aura PDF QA!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📤 Upload")
            st.write("1. Go to the sidebar")
            st.write("2. Upload your PDF file")
            
        with col2:
            st.markdown("### ⚙️ Process")
            st.write("3. Click 'Process Document'")
            st.write("4. Wait for processing to complete")
            
        with col3:
            st.markdown("### 💬 Chat")
            st.write("5. Start asking questions!")
            st.write("6. Get answers with sources")
        
        st.markdown("---")
        st.markdown("### 💡 Tips for better results:")
        st.write("- Use clear, specific questions")
        st.write("- Ask about topics likely covered in your document")
        st.write("- For large documents, processing may take a minute")

    def handle_user_query(self, question, top_k, enable_voice):
        """Handle user question and generate response"""
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching document..."):
                try:
                    start_time = time.time()
                    
                    # Search for relevant chunks
                    chunks = st.session_state.doc_processor.search_similar(question, top_k)
                    
                    # Generate answer
                    answer, confidence = self.llm_service.generate_answer(question, chunks)
                    
                    # Calculate response time
                    response_time = (time.time() - start_time) * 1000
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Show metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"⏱️ {response_time:.0f}ms")
                    with col2:
                        st.caption(f"🎯 Confidence: {confidence:.1%}")
                    
                    # Voice output if enabled
                    if enable_voice and confidence > 0.2:
                        with st.spinner("🎵 Generating audio..."):
                            self.voice_service.speak_text(answer)
                            
                except Exception as e:
                    error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    answer = error_msg
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

    def run(self):
        """Main application runner"""
        try:
            top_k, enable_voice = self.render_sidebar()
            self.render_chat(top_k, enable_voice)
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            st.info("Please refresh the page and try again.")

if __name__ == "__main__":
    app = AuraPDFQAApp()
    app.run()
