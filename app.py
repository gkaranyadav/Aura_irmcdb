# app.py - Aura PDF QA ⚡ (Fixed Full Version)
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
    TOP_K_CHUNKS = 5
    CHUNK_SIZE = 1000
    MIN_PARAGRAPH_LENGTH = 50

# =============================================================================
# LLM SERVICE (Groq/Fallback)
# =============================================================================
class LLMService:
    def __init__(self):
        self.client = None
        try:
            import groq
            api_key = st.secrets.get("GROQ_API_KEY", "")
            if api_key:
                self.client = groq.Groq(api_key=api_key)
                st.sidebar.success("✅ Groq LLM Ready")
            else:
                st.sidebar.info("ℹ️ Add GROQ_API_KEY to secrets for enhanced AI")
        except ImportError:
            st.sidebar.info("ℹ️ Install 'groq' package for enhanced AI")
        except Exception as e:
            st.sidebar.info(f"ℹ️ Using fallback mode: {e}")

    def generate_analytical_answer(self, question, context_chunks, document_context=""):
        if not context_chunks:
            return self._no_info_response(question), 0.0
        
        try:
            avg_conf = sum(c["similarity"] for c in context_chunks) / len(context_chunks)
            context_text = "\n\n".join([f"Source {i+1} (Page {c['metadata']['page']}, Confidence: {c['similarity']:.1%}):\n{c['content']}" 
                                      for i, c in enumerate(context_chunks)])
            if self.client:
                return self._groq_analysis(question, context_text, context_chunks, avg_conf)
            else:
                return self._enhanced_fallback(question, context_chunks, avg_conf)
        except Exception as e:
            return f"❌ Error generating answer: {str(e)}", 0.0

    def _groq_analysis(self, question, context_text, chunks, avg_conf):
        system_prompt = """You are an expert document analyst. Based on the provided document excerpts, provide a comprehensive, analytical answer to the user's question.

Guidelines:
1. SYNTHESIZE information from multiple sources
2. ANALYZE patterns and key insights  
3. BE SPECIFIC and reference document content
4. If information is insufficient, state this clearly
5. Provide meaningful insights beyond just repeating text"""

        user_prompt = f"""RELEVANT DOCUMENT EXCERPTS:
{context_text}

QUESTION: {question}

Please provide a comprehensive answer analyzing the document content:"""

        response = self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        answer = response.choices[0].message.content
        answer += self._format_sources(chunks)
        return answer, avg_conf

    def _enhanced_fallback(self, question, chunks, avg_conf):
        page_content = {}
        for chunk in chunks:
            page = chunk['metadata']['page']
            if page not in page_content:
                page_content[page] = []
            page_content[page].append(chunk['content'])
        answer = f"**Based on the document analysis:**\n\n"
        for page, contents in list(page_content.items())[:3]:
            answer += f"**Page {page}:**\n"
            for content in contents[:2]:
                sentences = [s.strip() for s in content.split('. ') if len(s.strip()) > 20]
                if sentences:
                    answer += f"• {sentences[0]}.\n"
            answer += "\n"
        answer += self._format_sources(chunks)
        return answer, avg_conf

    def _format_sources(self, chunks):
        sources = "\n\n**📚 Sources:**\n"
        for i, chunk in enumerate(chunks, 1):
            preview = chunk['content'][:100] + "..." if len(chunk['content']) > 100 else chunk['content']
            sources += f"{i}. Page {chunk['metadata']['page']} | Confidence: {chunk['similarity']:.1%}\n"
        return sources

    def _no_info_response(self, question):
        responses = [
            f"I couldn't find specific information about '{question}' in the document.",
            f"The document doesn't appear to contain information about '{question}'.",
            f"No relevant content found for '{question}'. Consider rephrasing."
        ]
        import random
        return random.choice(responses)

    def analyze_document_overview(self, chunks, doc_name="Document"):
        try:
            top_chunks = chunks[:10]
            if self.client:
                context_text = "\n\n".join([f"Page {c['metadata']['page']}: {c['content']}" for c in top_chunks])
                prompt = f"""Based on these document excerpts, provide a comprehensive overview:

{context_text}

Please analyze:
1. Main topics and themes
2. Document type and purpose  
3. Key findings or main points
4. Overall structure

Provide a structured overview:"""
                response = self.client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=800
                )
                return response.choices[0].message.content
            else:
                return self._fallback_overview(top_chunks, doc_name)
        except Exception as e:
            return f"Overview generation failed: {str(e)}"

    def _fallback_overview(self, chunks, doc_name):
        pages = list(set([c['metadata']['page'] for c in chunks]))
        content_samples = [c['content'] for c in chunks[:5]]
        overview = f"""
**📊 Document Overview: {doc_name}**

**Pages Analyzed:** {min(pages)}-{max(pages)}
**Content Samples:**\n
"""
        for i, sample in enumerate(content_samples, 1):
            preview = sample[:150] + "..." if len(sample) > 150 else sample
            overview += f"{i}. {preview}\n\n"
        overview += "\n**💡 Tip:** Ask specific questions about the content above for detailed answers."
        return overview

# =============================================================================
# VOICE SERVICE
# =============================================================================
class VoiceService:
    def __init__(self):
        self.has_voice_input = False
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.has_voice_input = True
        except ImportError:
            st.sidebar.info("🎤 Install 'speechrecognition' and 'pyaudio' for voice input")

    def speech_to_text(self):
        if not self.has_voice_input:
            st.warning("Voice input not available. Please install: pip install speechrecognition pyaudio")
            return None
        try:
            import speech_recognition as sr
            with sr.Microphone() as source:
                st.info("🎤 Listening...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=15)
            text = self.recognizer.recognize_google(audio)
            return text
        except Exception as e:
            st.error(f"❌ Voice input error: {e}")
            return None

    def text_to_speech(self, text, autoplay=True):
        try:
            clean_text = self._clean_text_for_tts(text)
            if len(clean_text) > 1000:
                clean_text = clean_text[:1000] + "..."
            tts = gTTS(text=clean_text, lang='en', slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tts.save(tmp_file.name)
                audio_bytes = open(tmp_file.name, 'rb').read()
            st.audio(audio_bytes, format="audio/mp3")
            os.unlink(tmp_file.name)
        except Exception as e:
            st.error(f"❌ Text-to-speech error: {e}")

    def _clean_text_for_tts(self, text):
        import re
        clean_text = re.sub(r'\*\*.*?\*\*', '', text)
        clean_text = re.sub(r'\*.*?\*', '', clean_text)
        clean_text = re.sub(r'#+', '', clean_text)
        clean_text = re.sub(r'\[.*?\]\(.*?\)', '', clean_text)
        clean_text = re.sub(r'📚 Sources:.*', '', clean_text)
        clean_text = re.sub(r'\n+', ' ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        return clean_text.strip()

# =============================================================================
# DOCUMENT PROCESSOR
# =============================================================================
class DocumentProcessor:
    def __init__(self):
        self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.index = None
        self.chunks = []
        self.chunk_metadata = []

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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        try:
            with st.status("🔄 Processing PDF...", expanded=True) as status:
                pdf_type = self.analyze_pdf_type(pdf_path)
                status.write(f"📄 PDF type: {pdf_type}")
                extracted = self.extract_text_direct(pdf_path) if pdf_type == "text_based" else self.extract_text_ocr(pdf_path)
                method = "text" if pdf_type == "text_based" else "ocr"
                status.write(f"✅ Extracted text from {len(extracted)} pages")

                # Chunking
                for item in extracted:
                    page = item["page"]
                    text = item["text"]
                    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= Config.MIN_PARAGRAPH_LENGTH]
                    for para in paragraphs:
                        if len(para) > Config.CHUNK_SIZE:
                            sentences = para.split(". ")
                            current_chunk = ""
                            for sentence in sentences:
                                if len(current_chunk + sentence) < Config.CHUNK_SIZE:
                                    current_chunk += sentence + ". "
                                else:
                                    if current_chunk.strip():
                                        self.chunks.append(current_chunk.strip())
                                        self.chunk_metadata.append({"page": page, "method": method})
                                    current_chunk = sentence + ". "
                            if current_chunk.strip():
                                self.chunks.append(current_chunk.strip())
                                self.chunk_metadata.append({"page": page, "method": method})
                        else:
                            self.chunks.append(para)
                            self.chunk_metadata.append({"page": page, "method": method})
                status.write(f"📊 Created {len(self.chunks)} text chunks")

                if not self.chunks:
                    st.error("❌ No text extracted from PDF.")
                    return 0

                # Create embeddings
                status.write("🧠 Creating embeddings...")
                embeddings = self.embedder.encode(self.chunks)
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
                self.index.add(np.array(embeddings))
                status.update(label="✅ PDF Processing Complete", state="complete", expanded=False)
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
        distances, indices = self.index.search(np.array(query_vec), min(top_k, len(self.chunks)))
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                sim = 1 / (1 + distances[0][i])
                results.append({"content": self.chunks[idx], "metadata": self.chunk_metadata[idx], "similarity": round(sim, 3)})
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

# =============================================================================
# STREAMLIT APP
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
        uploaded_file = st.sidebar.file_uploader("📁 Upload PDF", type="pdf")
        if uploaded_file:
            st.session_state['uploaded_file'] = uploaded_file

        if st.session_state.get('pdf_processed', False):
            st.sidebar.success("✅ PDF Ready!")
            if hasattr(st.session_state, 'pdf_name'):
                st.sidebar.info(f"**Document:** {st.session_state.pdf_name}")
        else:
            st.sidebar.warning("⚠️ Upload PDF to Start")

        if 'uploaded_file' in st.session_state:
            if st.sidebar.button("🚀 Process Document"):
                if not st.session_state.get('pdf_processed', False):
                    count = self.doc_processor.process_pdf(st.session_state['uploaded_file'])
                    if count > 0:
                        st.session_state.pdf_processed = True
                        st.session_state.pdf_name = st.session_state['uploaded_file'].name
                        st.sidebar.success("✅ PDF Processed!")

        top_k = st.sidebar.slider("Sources to use", 1, 5, 3)
        st.session_state.auto_voice = st.sidebar.checkbox("🔊 Auto Voice Response", True)

        if st.session_state.get('pdf_processed', False):
            st.sidebar.markdown("---")
            if st.sidebar.button("📊 Document Overview"):
                with st.spinner("Generating overview..."):
                    overview = self.llm_service.analyze_document_overview(
                        self.doc_processor.chunks[:15],
                        getattr(st.session_state, 'pdf_name', 'Document')
                    )
                    st.session_state.messages.append({"role": "assistant", "content": overview})

        return top_k

    def render_voice_input_safe(self):
        st.markdown("---")
        with st.expander("🎤 Voice Input (Optional)"):
            st.info("Voice input requires additional packages. You can always type your questions!")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Try Voice Input"):
                    if self.voice_service.has_voice_input:
                        question = self.voice_service.speech_to_text()
                        if question:
                            return question
                    else:
                        st.error("Voice input not available. Install speechrecognition + pyaudio")
            with col2:
                st.markdown("**Or type your question below:**")
        return None

    def render_chat(self, top_k):
        st.title("Aura PDF QA ⚡")
        st.markdown("Ask questions and get **AI-powered analysis** of your documents")
        if not st.session_state.pdf_processed:
            self.render_welcome()
            return

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        voice_question = self.render_voice_input_safe()
        text_question = st.chat_input("💬 Ask a question about your document...")
        question = voice_question or text_question
        if question:
            self.process_question(question, top_k)

    def render_welcome(self):
        st.info("👋 **Welcome to Aura PDF QA!**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🚀 Quick Start:
            1. **Upload** PDF in sidebar
            2. **Process** the document  
            3. **Ask questions** by text
            4. **Get AI-powered answers**
            """)
        with col2:
            st.markdown("""
            ### 🎯 Features:
            - **AI Document Analysis**
            - **Smart Source Retrieval**
            - **Voice Responses** 
            - **Document Overview**
            - **OCR Support**
            """)
        st.markdown("---")
        st.markdown("""
        **💡 Pro Tips:**
        - Use 'Document Overview' after processing to understand the document
        - Ask specific questions for best results
        - Voice output works automatically for answers
        """)

    def process_question(self, question, top_k):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing document..."):
                start_time = time.time()
                chunks = self.doc_processor.search_similar(question, top_k)
                document_context = f"Document: {getattr(st.session_state, 'pdf_name', 'Unknown')}"
                answer, confidence = self.llm_service.generate_analytical_answer(
                    question, chunks, document_context
                )
                response_time = (time.time() - start_time) * 1000
                st.markdown(answer)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"⏱️ {response_time:.0f}ms")
                with col2:
                    st.caption(f"🎯 {confidence:.1%} confidence")
                with col3:
                    st.caption(f"📚 {len(chunks)} sources")
                if st.session_state.auto_voice and confidence > 0.2:
                    with st.spinner("🔊 Generating voice response..."):
                        self.voice_service.text_to_speech(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    def run(self):
        top_k = self.render_sidebar()
        self.render_chat(top_k)

if __name__ == "__main__":
    app = AuraPDFQAApp()
    app.run()
