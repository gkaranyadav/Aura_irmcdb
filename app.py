# app.py - Aura PDF QA ⚡
# Streamlit PDF Q&A app with auto TTS (Voice output), minimal UI messages

import streamlit as st
import tempfile, os, time, base64, re
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pytesseract
import pdf2image
from gtts import gTTS
from groq import Groq

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Embedding model
    GROQ_MODEL = "llama-3.3-70b-versatile"                        # LLM model
    CHUNK_SIZE = 500                                              # Max chars per chunk
    MIN_PARAGRAPH_LENGTH = 20                                     # Ignore tiny paragraphs

# =============================================================================
# DOCUMENT PROCESSOR
# =============================================================================
class DocumentProcessor:
    def __init__(self):
        try:
            self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)  # Load embeddings
            self.index = None
            self.chunks = []
            self.chunk_metadata = []
            st.success("✅ Document processor ready")
        except Exception as e:
            st.error(f"❌ Document processor failed: {e}")

    # Extract text directly from text-based PDF
    def extract_text_direct(self, pdf_path):
        reader = PdfReader(pdf_path)
        extracted = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                extracted.append({"page": i, "text": text.strip()})
        return extracted

    # Extract text via OCR for scanned PDFs
    def extract_text_ocr(self, pdf_path):
        images = pdf2image.convert_from_path(pdf_path, dpi=200)
        extracted = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i, image in enumerate(images):
            status_text.text(f"Knowledge at your command page {i+1}/{len(images)}")
            text = pytesseract.image_to_string(image)
            if text.strip():
                extracted.append({"page": i + 1, "text": text.strip()})
            progress_bar.progress((i + 1) / len(images))
        status_text.text("✅ PDF processed")
        return extracted

    # Determine if PDF is text-based or scanned
    def analyze_pdf_type(self, pdf_path):
        reader = PdfReader(pdf_path)
        text_pages = sum(1 for page in reader.pages if len((page.extract_text() or "").strip()) > 50)
        total_pages = len(reader.pages)
        return "text_based" if total_pages == 0 or text_pages / total_pages > 0.5 else "scanned"

    # Process PDF: extract, chunk, and embed
    def process_pdf(self, uploaded_file):
        self.chunks, self.chunk_metadata, self.index = [], [], None
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_file.write(uploaded_file.getvalue())
        tmp_file.close()
        pdf_path = tmp_file.name

        try:
            st.info("Processing PDF...")  # Short message
            pdf_type = self.analyze_pdf_type(pdf_path)
            if pdf_type == "text_based":
                extracted = self.extract_text_direct(pdf_path)
                method = "text"
            else:
                extracted = self.extract_text_ocr(pdf_path)
                method = "ocr"

            # Chunk text into manageable sizes
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

            # Create embeddings
            embeddings = self.embedder.encode(self.chunks)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(np.array(embeddings))
            st.success("✅ PDF processed")  # Minimal success message
            return len(self.chunks)
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    # Search for relevant chunks
    def search_similar(self, query, top_k=3):
        if not self.chunks or self.index is None:
            st.warning("⚠️ No document processed yet")
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
# LLM SERVICE
# =============================================================================
class LLMService:
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            st.sidebar.success("✅ Groq LLM connected")
        except Exception as e:
            st.error(f"❌ Groq initialization failed: {e}")
            self.client = None

    # Generate answer from chunks
    def generate_answer(self, question, chunks):
        if not chunks:
            return f"❌ No relevant info found for '{question}'", 0.0
        avg_conf = sum(c["similarity"] for c in chunks)/len(chunks)
        if self.client:
            return self._generate_llm_answer(question, chunks, avg_conf)
        else:
            return self._simple_answer(question, chunks, avg_conf)

    # Use LLM
    def _generate_llm_answer(self, question, chunks, avg_conf):
        try:
            context = "\n\n".join([f"Page {c['metadata']['page']}: {c['content']}" for c in chunks])
            messages = [
                {"role": "system", "content": "You are an expert document analyst. Only use the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
            response = self.client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=1024
            )
            ai_answer = response.choices[0].message.content
            return ai_answer, avg_conf
        except:
            return self._simple_answer(question, chunks, avg_conf)

    # Simple fallback answer
    def _simple_answer(self, question, chunks, avg_conf):
        key_sentences = []
        for chunk in chunks:
            sentences = [s.strip() for s in chunk['content'].split('.') if s.strip()]
            key_sentences.extend(sentences[:2])
        summary = ' '.join(key_sentences[:6])
        return summary, avg_conf

# =============================================================================
# VOICE SERVICE
# =============================================================================
class VoiceService:
    # Clean text to avoid reading weird symbols
    def clean_text_for_tts(self, text):
        text = re.sub(r'[^\w\s.,?-]', '', text)
        replacements = {"%": " percent", "$": " dollars", "°": " degrees", "&": " and "}
        for k, v in replacements.items():
            text = text.replace(k, v)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Speak answer automatically
    def speak_text(self, text):
        text = self.clean_text_for_tts(text)
        tts = gTTS(text)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp_file.name)
        with open(tmp_file.name, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        html = f"""<audio autoplay><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>"""
        st.components.v1.html(html, height=50)

# =============================================================================
# STREAMLIT APP
# =============================================================================
class AuraPDFQAApp:
    def __init__(self):
        # Session state
        if 'pdf_processed' not in st.session_state:
            st.session_state.pdf_processed = False
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'doc_processor' not in st.session_state:
            st.session_state.doc_processor = DocumentProcessor()
        self.llm_service = LLMService()
        self.voice_service = VoiceService()
        st.set_page_config(page_title="📚 IRMC Aura", layout="wide")

    # Sidebar UI
    def render_sidebar(self):
        st.sidebar.title("Fast & reliable ⚡")  # Only heading here
        uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
        if st.session_state.pdf_processed:
            st.sidebar.success("✅ PDF ready")
        else:
            st.sidebar.warning("⚠️ Upload PDF first")
        if uploaded_file:
            st.sidebar.write(f"**File:** {uploaded_file.name}")
            if st.sidebar.button("🚀 Process Document"):
                with st.spinner("Processing....."):
                    count = st.session_state.doc_processor.process_pdf(uploaded_file)
                    if count > 0:
                        st.session_state.pdf_processed = True
                        st.session_state.pdf_name = uploaded_file.name
                        st.sidebar.success(f"✅ PDF processed ({count} chunks)")
        top_k = st.sidebar.slider("Sources to retrieve", 1, 5, 3)
        enable_voice = st.sidebar.checkbox("Enable Voice", True)
        return top_k, enable_voice

    # Main chat
    def render_chat(self, top_k, enable_voice):
        st.title("IRMC Aura 📚")  # Only one main heading
        if not st.session_state.pdf_processed:
            st.error("❌ Upload PDF and click Process first!")
            return
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        question = st.chat_input("Ask a question about your document...")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("assistant"):
                with st.spinner("Just a sec........."):  # Minimal message
                    chunks = st.session_state.doc_processor.search_similar(question, top_k)
                    answer, conf = self.llm_service.generate_answer(question, chunks)
                    st.markdown(answer)
                    if enable_voice:
                        self.voice_service.speak_text(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    # Run app
    def run(self):
        top_k, enable_voice = self.render_sidebar()
        self.render_chat(top_k, enable_voice)

if __name__ == "__main__":
    app = AuraPDFQAApp()
    app.run()
