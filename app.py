# app.py - Aura PDF QA ⚡ (Stable Version, No Voice)
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

# =============================================================================
# CONFIG
# =============================================================================
class Config:
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    TOP_K_CHUNKS = 5
    CHUNK_SIZE = 1000
    MIN_PARAGRAPH_LENGTH = 50

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
            st.info("🔄 Processing PDF...")
            pdf_type = self.analyze_pdf_type(pdf_path)
            st.info(f"📄 PDF type: {pdf_type}")

            extracted = self.extract_text_direct(pdf_path) if pdf_type == "text_based" else self.extract_text_ocr(pdf_path)

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
                                    self.chunk_metadata.append({"page": page})
                                current_chunk = sentence + ". "
                        if current_chunk.strip():
                            self.chunks.append(current_chunk.strip())
                            self.chunk_metadata.append({"page": page})
                    else:
                        self.chunks.append(para)
                        self.chunk_metadata.append({"page": page})

            st.success(f"📊 Created {len(self.chunks)} text chunks")

            # Create embeddings
            embeddings = self.embedder.encode(self.chunks)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(np.array(embeddings))

        except Exception as e:
            st.error(f"❌ PDF processing failed: {e}")
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
                results.append({
                    "content": self.chunks[idx],
                    "metadata": self.chunk_metadata[idx],
                    "similarity": round(sim, 3)
                })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

# =============================================================================
# STREAMLIT APP
# =============================================================================
class AuraPDFQAApp:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.setup_session_state()
        st.set_page_config(page_title="Aura PDF QA ⚡", layout="wide", page_icon="🎯")

    def setup_session_state(self):
        if 'pdf_processed' not in st.session_state:
            st.session_state.pdf_processed = False
        if 'messages' not in st.session_state:
            st.session_state.messages = []

    def render_sidebar(self):
        st.sidebar.title("🎯 Aura PDF QA")
        uploaded_file = st.sidebar.file_uploader("📁 Upload PDF", type="pdf")

        if uploaded_file and st.sidebar.button("🚀 Process Document"):
            self.doc_processor.process_pdf(uploaded_file)
            st.session_state.pdf_processed = True
            st.session_state.pdf_name = uploaded_file.name
            st.experimental_rerun()

        top_k = st.sidebar.slider("Sources to use", 1, 5, 3)
        return top_k

    def render_chat(self, top_k):
        st.title("Aura PDF QA ⚡")
        st.markdown("Ask questions and get **AI-powered analysis** of your documents")

        if not st.session_state.pdf_processed:
            st.info("👋 Upload and process a PDF to start")
            return

        # Display previous messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("💬 Ask a question about your document...")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching for relevant content..."):
                    chunks = self.doc_processor.search_similar(question, top_k)
                    if chunks:
                        answer = "\n\n".join([f"**Page {c['metadata']['page']}**: {c['content']}" for c in chunks])
                    else:
                        answer = "❌ No relevant content found in the document."
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

    def run(self):
        top_k = self.render_sidebar()
        self.render_chat(top_k)

# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    app = AuraPDFQAApp()
    app.run()
