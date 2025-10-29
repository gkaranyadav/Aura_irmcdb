# app.py - Aura PDF QA ⚡
import streamlit as st
import tempfile, os, time
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pytesseract
from PIL import Image
import pdf2image
from gtts import gTTS
from groq import Groq
import speech_recognition as sr
import io

# =============================================================================
# CONFIG
# =============================================================================
class Config:
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    GROQ_MODEL = "llama-3.3-70b-versatile"
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
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            if text.strip():
                extracted.append({"page": i + 1, "text": text.strip()})
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
            with st.spinner("📄 Processing PDF..."):
                pdf_type = self.analyze_pdf_type(pdf_path)

                if pdf_type == "text_based":
                    extracted = self.extract_text_direct(pdf_path)
                else:
                    extracted = self.extract_text_ocr(pdf_path)

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
                                        self.chunk_metadata.append({"page": page})
                                    chunk = s + ". "
                            if chunk.strip():
                                self.chunks.append(chunk.strip())
                                self.chunk_metadata.append({"page": page})
                        else:
                            if para.strip():
                                self.chunks.append(para)
                                self.chunk_metadata.append({"page": page})

                if not self.chunks:
                    st.error("❌ No text extracted from PDF.")
                    return 0

                embeddings = self.embedder.encode(self.chunks)
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
                self.index.add(np.array(embeddings))
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
# LLM SERVICE - Simplified
# =============================================================================
class LLMService:
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        except:
            self.client = None

    def generate_answer(self, question, chunks):
        if not chunks:
            return "I couldn't find relevant information in the document to answer your question.", []

        if self.client:
            return self._generate_llm_answer(question, chunks)
        else:
            return self._simple_answer(chunks)

    def _generate_llm_answer(self, question, chunks):
        try:
            context = "\n\n".join([f"Page {c['metadata']['page']}: {c['content']}" for c in chunks])
            
            messages = [
                {
                    "role": "system",
                    "content": "Provide a direct, concise answer based on the document. Include page numbers for key facts. Keep it simple and clear."
                },
                {
                    "role": "user", 
                    "content": f"Question: {question}\n\nDocument Context:\n{context}\n\nAnswer:"
                }
            ]
            
            response = self.client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            pages = list(set([c['metadata']['page'] for c in chunks]))
            return answer, pages
            
        except Exception as e:
            return self._simple_answer(chunks)

    def _simple_answer(self, chunks):
        key_sentences = []
        for chunk in chunks:
            sentences = [s.strip() for s in chunk['content'].split('.') if s.strip()]
            key_sentences.extend(sentences[:2])
        
        unique_sentences = []
        for sentence in key_sentences:
            if sentence not in unique_sentences and len(sentence) > 20:
                unique_sentences.append(sentence)
        
        summary = " ".join(unique_sentences[:4])
        pages = list(set([c['metadata']['page'] for c in chunks]))
        return summary, pages

# =============================================================================
# VOICE SERVICE
# =============================================================================
class VoiceService:
    def text_to_speech(self, text):
        try:
            tts = gTTS(text)
            audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(audio_file.name)
            return audio_file.name
        except Exception as e:
            st.error(f"❌ Voice generation failed: {e}")
            return None

    def speech_to_text(self):
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("🎤 Listening... Speak now!")
                audio = r.listen(source, timeout=10)
                st.info("🔄 Processing speech...")
                text = r.recognize_google(audio)
                return text
        except sr.WaitTimeoutError:
            st.error("❌ No speech detected")
            return None
        except sr.UnknownValueError:
            st.error("❌ Could not understand audio")
            return None
        except Exception as e:
            st.error(f"❌ Voice input failed: {e}")
            return None

# =============================================================================
# STREAMLIT APP - Simplified
# =============================================================================
class AuraPDFQAApp:
    def __init__(self):
        if 'pdf_processed' not in st.session_state:
            st.session_state.pdf_processed = False
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'doc_processor' not in st.session_state:
            st.session_state.doc_processor = DocumentProcessor()
            
        self.llm_service = LLMService()
        self.voice_service = VoiceService()
        self.setup_ui()

    def setup_ui(self):
        st.set_page_config(page_title="Aura PDF QA", layout="wide")

    def render_sidebar(self):
        st.sidebar.title("📚 Aura PDF QA")
        
        uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
        
        if uploaded_file:
            st.sidebar.write(f"**File:** {uploaded_file.name}")
            if st.sidebar.button("🚀 Process Document", use_container_width=True):
                with st.spinner("Processing PDF..."):
                    count = st.session_state.doc_processor.process_pdf(uploaded_file)
                    if count > 0:
                        st.session_state.pdf_processed = True
                        st.session_state.pdf_name = uploaded_file.name
                        st.sidebar.success(f"✅ PDF processed!")
                        st.rerun()
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎤 Voice Input")
        if st.sidebar.button("🎤 Ask by Voice", use_container_width=True):
            voice_question = self.voice_service.speech_to_text()
            if voice_question:
                st.session_state.voice_question = voice_question
                st.rerun()

    def render_chat(self):
        st.title("Aura PDF QA")
        
        if not st.session_state.pdf_processed:
            st.info("📝 Upload a PDF and click 'Process Document' to start")
            return

        # Display chat messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("audio"):
                    st.audio(msg["audio"], format="audio/mp3")

        # Handle voice question
        if 'voice_question' in st.session_state:
            question = st.session_state.voice_question
            del st.session_state.voice_question
        else:
            question = st.chat_input("Ask a question or use voice...")

        if question:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Generate answer
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching..."):
                    chunks = st.session_state.doc_processor.search_similar(question, 3)
                    answer, pages = self.llm_service.generate_answer(question, chunks)
                    
                    # Display answer
                    if pages:
                        st.write(f"{answer}")
                        st.caption(f"📄 Pages: {', '.join(map(str, pages))}")
                    else:
                        st.write(answer)
                    
                    # Generate and play audio
                    audio_file = self.voice_service.text_to_speech(answer)
                    if audio_file:
                        st.audio(audio_file, format="audio/mp3")
                        # Store audio in message
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "audio": audio_file
                        })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer
                        })

    def run(self):
        self.render_sidebar()
        self.render_chat()

if __name__ == "__main__":
    app = AuraPDFQAApp()
    app.run()
