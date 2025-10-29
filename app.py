# app.py - Aura PDF QA ⚡ (Enhanced with LLM & Voice)
import streamlit as st
import tempfile
import os
import time
import io
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pytesseract
from PIL import Image
import pdf2image
from gtts import gTTS
import speech_recognition as sr
import groq

# =============================================================================
# CONFIG
# =============================================================================
class Config:
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    TOP_K_CHUNKS = 5
    CHUNK_SIZE = 1000
    MIN_PARAGRAPH_LENGTH = 50
    GROQ_MODEL = "llama3-70b-8192"  # or "mixtral-8x7b-32768"

# =============================================================================
# LLM SERVICE (Enhanced with Groq)
# =============================================================================
class LLMService:
    def __init__(self):
        self.client = None
        try:
            # Initialize Groq client
            self.client = groq.Groq(api_key=st.secrets.get("GROQ_API_KEY", ""))
            st.success("✅ LLM Service Ready (Groq)")
        except Exception as e:
            st.warning(f"⚠️ LLM Service: {e}")

    def generate_analytical_answer(self, question, context_chunks, document_context=""):
        """Generate analytical answer using LLM"""
        if not context_chunks:
            return "I couldn't find relevant information in the document to answer your question. Please try asking about specific content that might be in the document.", 0.0
        
        try:
            # Calculate average confidence
            avg_conf = sum(c["similarity"] for c in context_chunks) / len(context_chunks)
            
            # Prepare context for LLM
            context_text = "\n\n".join([f"Source {i+1} (Page {c['metadata']['page']}, Confidence: {c['similarity']:.1%}):\n{c['content']}" 
                                      for i, c in enumerate(context_chunks)])
            
            # Enhanced prompt for better analysis
            system_prompt = """You are an expert document analyst. Based on the provided document excerpts, provide a comprehensive, analytical answer to the user's question.

Guidelines:
1. SYNTHESIZE information from multiple sources to form a complete answer
2. ANALYZE patterns, themes, and key insights from the document
3. BE SPECIFIC and reference document content directly
4. If the document doesn't contain the answer, clearly state this
5. Focus on the MOST RELEVANT information
6. Provide meaningful insights beyond just repeating text"""

            user_prompt = f"""DOCUMENT CONTEXT:
{document_context}

RELEVANT EXCERPTS:
{context_text}

QUESTION: {question}

Please provide a comprehensive answer analyzing the document content:"""

            if self.client:
                # Use Groq LLM for analysis
                response = self.client.chat.completions.create(
                    model=Config.GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1024
                )
                answer = response.choices[0].message.content
            else:
                # Fallback to simple summarization
                answer = self._fallback_summarize(question, context_chunks)
            
            # Add sources
            answer += "\n\n**📚 Sources:**\n"
            for i, chunk in enumerate(context_chunks, 1):
                preview = chunk['content'][:120] + "..." if len(chunk['content']) > 120 else chunk['content']
                answer += f"{i}. Page {chunk['metadata']['page']} | Confidence: {chunk['similarity']:.1%}\n"
            
            return answer, avg_conf
            
        except Exception as e:
            return f"❌ Error generating answer: {str(e)}", 0.0

    def analyze_document_overview(self, chunks, top_n=10):
        """Generate document overview using LLM"""
        try:
            # Get top chunks for overview
            top_chunks = chunks[:top_n]
            context_text = "\n\n".join([f"Page {c['metadata']['page']}: {c['content']}" for c in top_chunks])
            
            prompt = f"""Based on the following document excerpts, provide a comprehensive overview:

{context_text}

Please analyze and provide:
1. Main topics and themes
2. Document type and purpose
3. Key findings or main points
4. Overall tone and style

Provide a structured overview:"""
            
            if self.client:
                response = self.client.chat.completions.create(
                    model=Config.GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=800
                )
                return response.choices[0].message.content
            else:
                return self._fallback_overview(top_chunks)
                
        except Exception as e:
            return f"Unable to generate overview: {str(e)}"

    def _fallback_summarize(self, question, chunks):
        """Fallback summarization without LLM"""
        all_text = " ".join([c['content'] for c in chunks])
        sentences = [s.strip() for s in all_text.split(". ") if len(s.strip()) > 30]
        
        # Simple extraction of key sentences
        key_sentences = sentences[:6]
        summary = " ".join([f"{s}." for s in key_sentences])
        
        return f"**Based on the document:**\n\n{summary}"

    def _fallback_overview(self, chunks):
        """Fallback overview without LLM"""
        pages_covered = list(set([c['metadata']['page'] for c in chunks]))
        content_samples = [c['content'][:100] for c in chunks[:3]]
        
        overview = f"""
**Document Overview:**
- Pages analyzed: {min(pages_covered)}-{max(pages_covered)}
- Key content samples:
"""
        for i, sample in enumerate(content_samples, 1):
            overview += f"  {i}. {sample}...\n"
            
        return overview

# =============================================================================
# VOICE SERVICE (Enhanced)
# =============================================================================
class VoiceService:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        
    def speech_to_text(self):
        """Convert speech to text using microphone"""
        try:
            with sr.Microphone() as source:
                st.info("🎤 Listening... Speak now!")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=15)
                
            st.info("🔄 Processing audio...")
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            st.error("❌ Could not understand audio")
            return None
        except Exception as e:
            st.error(f"❌ Voice input error: {e}")
            return None

    def text_to_speech(self, text):
        """Convert text to speech and play automatically"""
        try:
            # Clean text for TTS (remove markdown, sources)
            clean_text = self._clean_text_for_tts(text)
            
            if len(clean_text) > 1000:
                clean_text = clean_text[:1000] + "..."
                
            tts = gTTS(text=clean_text, lang='en', slow=False)
            
            # Create audio file and auto-play
            audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(audio_file.name)
            
            # Auto-play the audio
            audio_bytes = open(audio_file.name, 'rb').read()
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            
            # Cleanup
            os.unlink(audio_file.name)
            
        except Exception as e:
            st.error(f"❌ Text-to-speech error: {e}")

    def _clean_text_for_tts(self, text):
        """Clean text for better TTS output"""
        # Remove markdown formatting
        import re
        clean_text = re.sub(r'\*\*.*?\*\*', '', text)  # Remove bold
        clean_text = re.sub(r'\*.*?\*', '', clean_text)  # Remove italic
        clean_text = re.sub(r'#+', '', clean_text)  # Remove headers
        clean_text = re.sub(r'\[.*?\]\(.*?\)', '', clean_text)  # Remove links
        clean_text = re.sub(r'📚 Sources:.*', '', clean_text)  # Remove sources section
        clean_text = re.sub(r'\n+', ' ', clean_text)  # Replace newlines with spaces
        clean_text = re.sub(r'\s+', ' ', clean_text)  # Remove extra spaces
        return clean_text.strip()

# =============================================================================
# DOCUMENT PROCESSOR (Same as before with improvements)
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
        reader = PdfReader(pdf_path)
        extracted = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                extracted.append({"page": i, "text": text.strip()})
        return extracted

    def extract_text_ocr(self, pdf_path):
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

    def analyze_pdf_type(self, pdf_path):
        reader = PdfReader(pdf_path)
        text_pages = sum(1 for page in reader.pages if len((page.extract_text() or "").strip()) > 50)
        total_pages = len(reader.pages)
        return "text_based" if total_pages == 0 or text_pages / total_pages > 0.5 else "scanned"

    def process_pdf(self, uploaded_file):
        self.chunks = []
        self.chunk_metadata = []
        self.index = None
        
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_file.write(uploaded_file.getvalue())
        tmp_file.close()
        pdf_path = tmp_file.name

        try:
            st.info("🔍 Analyzing PDF type...")
            pdf_type = self.analyze_pdf_type(pdf_path)
            st.info(f"PDF type detected: {pdf_type}")

            if pdf_type == "text_based":
                extracted = self.extract_text_direct(pdf_path)
                method = "text"
                st.info(f"📄 Extracted text from {len(extracted)} pages")
            else:
                extracted = self.extract_text_ocr(pdf_path)
                method = "ocr"
                st.info(f"📷 OCR processed {len(extracted)} pages")

            # Chunking
            self.chunks, self.chunk_metadata = [], []
            for item in extracted:
                page = item["page"]
                paragraphs = [p.strip() for p in item["text"].split("\n\n") if len(p.strip()) >= Config.MIN_PARAGRAPH_LENGTH]
                for para in paragraphs:
                    if len(para) > Config.CHUNK_SIZE:
                        sentences = para.split(". ")
                        chunk = ""
                        for s in sentences:
                            if len(chunk + s) < Config.CHUNK_SIZE:
                                chunk += s + ". "
                            else:
                                if chunk.strip():
                                    self.chunks.append(chunk.strip())
                                    self.chunk_metadata.append({"page": page, "method": method})
                                chunk = s + ". "
                        if chunk.strip():
                            self.chunks.append(chunk.strip())
                            self.chunk_metadata.append({"page": page, "method": method})
                    else:
                        if para.strip():
                            self.chunks.append(para)
                            self.chunk_metadata.append({"page": page, "method": method})

            st.info(f"📊 Created {len(self.chunks)} text chunks")

            if not self.chunks:
                st.error("❌ No text extracted from PDF.")
                return 0

            # Embeddings + FAISS
            st.info("🧠 Creating embeddings...")
            embeddings = self.embedder.encode(self.chunks)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(np.array(embeddings))
            st.success(f"✅ PDF processed: {len(self.chunks)} chunks created")
            return len(self.chunks)

        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            return 0
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def search_similar(self, query, top_k=3):
        if not self.chunks or self.index is None:
            return []

        query_vec = self.embedder.encode([query])
        distances, indices = self.index.search(np.array(query_vec), top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                sim = 1 / (1 + distances[0][i])
                results.append({"content": self.chunks[idx], "metadata": self.chunk_metadata[idx], "similarity": round(sim, 3)})
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

# =============================================================================
# STREAMLIT APP (Enhanced with Voice & LLM)
# =============================================================================
class AuraPDFQAApp:
    def __init__(self):
        self.setup_session_state()
        self.llm_service = LLMService()
        self.voice_service = VoiceService()
        self.doc_processor = DocumentProcessor()
        self.setup_ui()

    def setup_session_state(self):
        if 'pdf_processed' not in st.session_state:
            st.session_state.pdf_processed = False
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'auto_voice' not in st.session_state:
            st.session_state.auto_voice = True

    def setup_ui(self):
        st.set_page_config(
            page_title="Aura PDF QA ⚡", 
            layout="wide",
            page_icon="🎯"
        )

    def render_sidebar(self):
        st.sidebar.title("🎯 Aura PDF QA")
        st.sidebar.markdown("---")
        
        # File upload
        uploaded_file = st.sidebar.file_uploader("📁 Upload PDF", type="pdf")
        
        # Status
        if st.session_state.pdf_processed:
            st.sidebar.success("✅ PDF Ready!")
        else:
            st.sidebar.warning("⚠️ Upload PDF to Start")
        
        # Process button
        if uploaded_file and not st.session_state.pdf_processed:
            if st.sidebar.button("🚀 Process Document", type="primary", use_container_width=True):
                with st.spinner("Processing PDF..."):
                    count = self.doc_processor.process_pdf(uploaded_file)
                    if count > 0:
                        st.session_state.pdf_processed = True
                        st.session_state.pdf_name = uploaded_file.name
                        st.rerun()
        
        # Configuration
        st.sidebar.markdown("### ⚙️ Settings")
        top_k = st.sidebar.slider("Sources to use", 1, 5, 3)
        st.session_state.auto_voice = st.sidebar.checkbox("🔊 Auto Voice Response", True)
        enable_voice_input = st.sidebar.checkbox("🎤 Voice Input", True)
        
        # Document overview
        if st.session_state.pdf_processed:
            st.sidebar.markdown("---")
            if st.sidebar.button("📊 Document Overview", use_container_width=True):
                with st.spinner("Generating overview..."):
                    overview = self.llm_service.analyze_document_overview(
                        self.doc_processor.chunks[:15]  # Use top chunks for overview
                    )
                    st.session_state.messages.append({"role": "assistant", "content": overview})
                    st.rerun()
        
        return top_k, enable_voice_input

    def render_voice_input(self):
        """Render voice input interface"""
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🎤 Click to Speak", use_container_width=True, type="secondary"):
                question = self.voice_service.speech_to_text()
                if question:
                    # Add to chat and process
                    st.session_state.messages.append({"role": "user", "content": question})
                    return question
        return None

    def render_chat(self, top_k, enable_voice_input):
        st.title("Aura PDF QA ⚡")
        st.markdown("Ask questions and get **AI-powered analysis** of your documents")
        
        if not st.session_state.pdf_processed:
            self.render_welcome()
            return

        # Display chat
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Voice input
        voice_question = None
        if enable_voice_input:
            voice_question = self.render_voice_input()

        # Text input
        text_question = st.chat_input("Ask a question about your document...")
        
        question = voice_question or text_question
        if question:
            self.process_question(question, top_k)

    def render_welcome(self):
        st.info("👋 **Welcome to Aura PDF QA!**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🚀 How to Start:
            1. **Upload** PDF in sidebar
            2. **Process** the document  
            3. **Ask questions** by text or voice
            4. **Get AI-powered answers** with sources
            """)
        
        with col2:
            st.markdown("""
            ### 🎯 Features:
            - **AI Document Analysis** using Groq LLM
            - **Voice Input & Output** 
            - **Smart Source Retrieval**
            - **Automatic Speech**
            - **Document Overview**
            """)
        
        st.markdown("---")
        st.markdown("💡 **Pro Tip:** Use 'Document Overview' in sidebar to get a summary after processing!")

    def process_question(self, question, top_k):
        """Process user question and generate response"""
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing document..."):
                start_time = time.time()
                
                # Search for relevant content
                chunks = self.doc_processor.search_similar(question, top_k)
                
                # Generate analytical answer
                document_context = f"Document: {getattr(st.session_state, 'pdf_name', 'Unknown')}"
                answer, confidence = self.llm_service.generate_analytical_answer(
                    question, chunks, document_context
                )
                
                response_time = (time.time() - start_time) * 1000
                
                # Display answer
                st.markdown(answer)
                
                # Show metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"⏱️ {response_time:.0f}ms")
                with col2:
                    st.caption(f"🎯 {confidence:.1%} confidence")
                with col3:
                    st.caption(f"📚 {len(chunks)} sources")
                
                # Auto voice response
                if st.session_state.auto_voice and confidence > 0.3:
                    with st.spinner("🔊 Generating voice..."):
                        self.voice_service.text_to_speech(answer)
        
        # Add to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

    def run(self):
        top_k, enable_voice_input = self.render_sidebar()
        self.render_chat(top_k, enable_voice_input)

if __name__ == "__main__":
    app = AuraPDFQAApp()
    app.run()
