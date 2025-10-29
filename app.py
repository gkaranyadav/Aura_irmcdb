# app.py - Aura PDF QA ⚡ Cloud-Friendly Voice Version
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
from streamlit.components.v1 import html

# =============================================================================
# CONFIG
# =============================================================================
class Config:
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    TOP_K_CHUNKS = 5
    CHUNK_SIZE = 1000
    MIN_PARAGRAPH_LENGTH = 50

# =============================================================================
# LLM SERVICE (Groq / Fallback)
# =============================================================================
class LLMService:
    def __init__(self):
        self.client = None
        try:
            import groq
            api_key = st.secrets.get("GROQ_API_KEY", "")
            if api_key:
                self.client = groq.Groq(api_key=api_key)
        except:
            pass

    def generate_answer(self, question, context_chunks):
        if not context_chunks:
            return self._no_info_response(question), 0.0
        avg_conf = sum(c["similarity"] for c in context_chunks) / len(context_chunks)
        context_text = "\n\n".join([f"Page {c['metadata']['page']}:\n{c['content']}" for c in context_chunks])
        if self.client:
            return self._groq_analysis(question, context_text, context_chunks, avg_conf)
        else:
            return self._fallback_analysis(context_chunks, avg_conf)

    def _groq_analysis(self, question, context_text, chunks, avg_conf):
        system_prompt = """You are a professional document analyst. Based on the given document excerpts, provide a concise, analytical, and well-structured answer. Reference page numbers and confidence scores when relevant."""
        user_prompt = f"QUESTION: {question}\n\nCONTEXT:\n{context_text}\n\nProvide a detailed answer:"
        try:
            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "system", "content": system_prompt},{"role": "user", "content": user_prompt}],
                temperature=0.3,
                max_tokens=1024
            )
            answer = response.choices[0].message.content
            return answer + self._format_sources(chunks), avg_conf
        except:
            return self._fallback_analysis(chunks, avg_conf)

    def _fallback_analysis(self, chunks, avg_conf):
        answer = "**Based on the document analysis:**\n\n"
        for chunk in chunks[:3]:
            sentences = [s.strip() for s in chunk['content'].split('. ') if len(s.strip()) > 20]
            if sentences:
                answer += f"• {sentences[0]} (Page {chunk['metadata']['page']})\n"
        return answer + self._format_sources(chunks), avg_conf

    def _format_sources(self, chunks):
        sources = "\n\n**📚 Sources:**\n"
        for i, chunk in enumerate(chunks, 1):
            preview = chunk['content'][:100] + "..." if len(chunk['content']) > 100 else chunk['content']
            sources += f"{i}. Page {chunk['metadata']['page']} | Confidence: {chunk['similarity']:.1%}\n"
        return sources

    def _no_info_response(self, question):
        return f"No relevant information found for '{question}'.", 0.0

    def analyze_document_overview(self, chunks, doc_name="Document"):
        overview = f"**📊 Overview of {doc_name}:**\n\n"
        for i, chunk in enumerate(chunks[:5], 1):
            preview = chunk['content'][:150] + "..." if len(chunk['content']) > 150 else chunk['content']
            overview += f"{i}. Page {chunk['metadata']['page']}: {preview}\n"
        return overview

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
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            if text.strip():
                extracted.append({"page": i+1, "text": text.strip()})
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
        pdf_type = self.analyze_pdf_type(pdf_path)
        if pdf_type == "text_based":
            extracted = self.extract_text_direct(pdf_path)
            method = "text"
        else:
            extracted = self.extract_text_ocr(pdf_path)
            method = "ocr"
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
        if not self.chunks:
            st.error("❌ No text extracted from PDF.")
            return 0
        embeddings = self.embedder.encode(self.chunks)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)
        return len(self.chunks)

    def search_similar(self, query, top_k=3):
        if not self.chunks or self.index is None:
            return []
        query_vec = self.embedder.encode([query])
        distances, indices = self.index.search(np.array(query_vec), min(top_k, len(self.chunks)))
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                sim = 1 / (1 + distances[0][i])
                results.append({"content": self.chunks[idx], "metadata": self.chunk_metadata[idx], "similarity": round(sim,3)})
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

# =============================================================================
# VOICE SERVICE (Browser Native)
# =============================================================================
def speak_text_js(text):
    escaped_text = text.replace('"', '\\"')
    html_code = f"""
    <script>
    const msg = new SpeechSynthesisUtterance("{escaped_text}");
    msg.lang = "en-US";
    msg.rate = 1;
    window.speechSynthesis.speak(msg);
    </script>
    """
    html(html_code)

def get_voice_input():
    js_code = """
    <script>
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.start();
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        document.dispatchEvent(new CustomEvent("voice_input", {detail: transcript}));
    };
    </script>
    """
    html(js_code)
    voice_text = st.text_input("Voice Input (Speak now):", key="voice_text")
    return voice_text

# =============================================================================
# STREAMLIT APP
# =============================================================================
class AuraPDFQAApp:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.llm_service = LLMService()
        self.setup_session_state()
        st.set_page_config(page_title="Aura PDF QA ⚡", layout="wide")

    def setup_session_state(self):
        if 'pdf_processed' not in st.session_state:
            st.session_state.pdf_processed = False
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'input_mode' not in st.session_state:
            st.session_state.input_mode = "Text"

    def render_sidebar(self):
        st.sidebar.title("🎯 Aura PDF QA")
        uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
        if uploaded_file and not st.session_state.pdf_processed:
            if st.sidebar.button("Process Document"):
                count = self.doc_processor.process_pdf(uploaded_file)
                if count > 0:
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_name = uploaded_file.name
                    st.experimental_rerun()
        st.sidebar.markdown("### Settings")
        st.session_state.input_mode = st.sidebar.radio("Input mode", ["Text", "Voice"], index=0, label_visibility="collapsed")
        top_k = st.sidebar.slider("Sources to use", 1, 5, 3)
        auto_voice = st.sidebar.checkbox("Auto Voice Response", True)
        st.session_state.auto_voice = auto_voice
        if st.session_state.pdf_processed:
            if st.sidebar.button("Document Overview"):
                overview = self.llm_service.analyze_document_overview(self.doc_processor.chunks[:15], st.session_state.pdf_name)
                st.session_state.messages.append({"role":"assistant","content":overview})
                st.experimental_rerun()
        return top_k

    def render_chat(self, top_k):
        st.title("Aura PDF QA ⚡")
        if not st.session_state.pdf_processed:
            st.info("Upload PDF from sidebar to start.")
            return
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if st.session_state.input_mode == "Voice":
            question = get_voice_input()
        else:
            question = st.chat_input("Ask a question about your document...")
        if question:
            self.process_question(question, top_k)

    def process_question(self, question, top_k):
        st.session_state.messages.append({"role":"user","content":question})
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document..."):
                chunks = self.doc_processor.search_similar(question, top_k)
                answer, confidence = self.llm_service.generate_answer(question, chunks)
                st.markdown(answer)
                if st.session_state.auto_voice:
                    speak_text_js(answer)
        st.session_state.messages.append({"role":"assistant","content":answer})

    def run(self):
        top_k = self.render_sidebar()
        self.render_chat(top_k)

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    app = AuraPDFQAApp()
    app.run()
