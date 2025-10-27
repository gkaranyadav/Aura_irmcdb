# app.py - IRMC AskPro ⚡ Fully Improved
import streamlit as st
import tempfile, os, uuid, time
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
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_file.write(uploaded_file.getvalue())
        tmp_file.close()
        pdf_path = tmp_file.name

        try:
            st.info("🔍 Analyzing PDF type...")
            pdf_type = self.analyze_pdf_type(pdf_path)
            st.info(f"PDF type detected: {pdf_type}")

            # Extract text
            if pdf_type == "text_based":
                extracted = self.extract_text_direct(pdf_path)
                method = "text"
            else:
                extracted = self.extract_text_ocr(pdf_path)
                method = "ocr"

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
                                self.chunks.append(chunk.strip())
                                self.chunk_metadata.append({"page": page, "method": method})
                                chunk = s + ". "
                        if chunk:
                            self.chunks.append(chunk.strip())
                            self.chunk_metadata.append({"page": page, "method": method})
                    else:
                        self.chunks.append(para)
                        self.chunk_metadata.append({"page": page, "method": method})

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

        finally:
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
# LLM SERVICE
# =============================================================================
class LLMService:
    def generate_answer(self, question, chunks):
        if not chunks:
            return f"❌ No relevant info found for '{question}'", 0.0
        avg_conf = sum(c["similarity"] for c in chunks)/len(chunks)
        summary = self._summarize(chunks)
        answer = f"**Question:** {question}\n\n**Answer:**\n{summary}\n\n**Sources:**\n"
        for i, c in enumerate(chunks):
            answer += f"Page {c['metadata']['page']} ({c['metadata']['method']}) | Conf: {c['similarity']:.1%}\n{c['content'][:200]}...\n\n"
        return answer, avg_conf

    def _summarize(self, chunks):
        all_text = " ".join([c['content'] for c in chunks])
        sentences = [s.strip() for s in all_text.split(".") if len(s.strip()) > 20]
        summary = "\n".join([f"• {s}." for s in sentences[:5]])
        return summary

# =============================================================================
# VOICE SERVICE
# =============================================================================
class VoiceService:
    def speak_text(self, text):
        tts = gTTS(text)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp_file.name)
        st.audio(tmp_file.name, format="audio/mp3")

# =============================================================================
# STREAMLIT APP
# =============================================================================
class AuraPDFQAApp:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.llm_service = LLMService()
        self.voice_service = VoiceService()
        self.setup_ui()

    def setup_ui(self):
        st.set_page_config(page_title="IRMC AskPro ⚡", layout="wide")

    def render_sidebar(self):
        st.sidebar.title("📚 IRMC AskPro")
        uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
        if uploaded_file and st.sidebar.button("Process Document"):
            count = self.doc_processor.process_pdf(uploaded_file)
            if count > 0:
                st.session_state.pdf_processed = True
                st.session_state.pdf_name = uploaded_file.name
                st.success(f"PDF processed with {count} chunks.")
        top_k = st.sidebar.slider("Sources to retrieve", 1, 5, 3)
        enable_voice = st.sidebar.checkbox("Enable Voice", True)
        return top_k, enable_voice

    def render_chat(self, top_k, enable_voice):
        st.title("IRMC AskPro ⚡")
        st.markdown("Ask questions about your document and get AI-powered answers")
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'pdf_processed' not in st.session_state:
            st.session_state.pdf_processed = False

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("Ask a question...")
        if question:
            if not st.session_state.pdf_processed:
                st.error("Upload and process a PDF first!")
                return
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("assistant"):
                with st.spinner("Searching..."):
                    start = time.time()
                    chunks = self.doc_processor.search_similar(question, top_k)
                    answer, conf = self.llm_service.generate_answer(question, chunks)
                    elapsed = (time.time() - start) * 1000
                    st.markdown(answer)
                    st.caption(f"⏱️ {elapsed:.0f}ms | Confidence: {conf:.1%}")
                    if enable_voice and conf > 0.1:
                        self.voice_service.speak_text(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    def run(self):
        top_k, enable_voice = self.render_sidebar()
        self.render_chat(top_k, enable_voice)

if __name__ == "__main__":
    app = AuraPDFQAApp()
    app.run()
