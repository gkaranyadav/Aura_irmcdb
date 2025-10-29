# app.py - Aura PDF QA ⚡
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
from groq import Groq

# =============================================================================
# CONFIG
# =============================================================================
class Config:
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    GROQ_MODEL = "llama-3.3-70b-versatile"  # Updated to current model
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
        # Clear previous data
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

            # Extract text
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
# GROQ LLM SERVICE - Fixed with current models
# =============================================================================
class LLMService:
    def __init__(self):
        try:
            # Initialize Groq client
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            st.sidebar.success("✅ Groq LLM connected")
        except Exception as e:
            st.error(f"❌ Groq initialization failed: {e}")
            self.client = None

    def generate_answer(self, question, chunks):
        if not chunks:
            return f"❌ No relevant info found for '{question}'", 0.0
        
        avg_conf = sum(c["similarity"] for c in chunks)/len(chunks)
        
        # If Groq is available, use it for intelligent analysis
        if self.client:
            return self._generate_llm_answer(question, chunks, avg_conf)
        else:
            # Fallback to simple summary
            return self._simple_answer(question, chunks, avg_conf)

    def _generate_llm_answer(self, question, chunks, avg_conf):
        """Use Groq LLM to analyze and provide intelligent answers"""
        try:
            # Prepare context from relevant chunks
            context = "\n\n".join([
                f"From Page {c['metadata']['page']}: {c['content']}" 
                for c in chunks
            ])
            
            # Create intelligent prompt for analysis
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert document analyst. Based EXCLUSIVELY on the provided document context, provide:

1. A direct, concise answer to the question
2. Key evidence and quotes from the document to support your answer
3. Specific page numbers where information was found
4. Clear analysis connecting the evidence to the answer

IMPORTANT: Only use information from the provided document. If the answer isn't in the document, say so clearly."""
                },
                {
                    "role": "user", 
                    "content": f"""DOCUMENT CONTEXT:
{context}

USER QUESTION: {question}

Based ONLY on the document context above, please provide a comprehensive answer with specific evidence and page references."""
                }
            ]
            
            # Get response from Groq
            response = self.client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=messages,
                temperature=0.1,  # Lower temperature for more factual responses
                max_tokens=1024,
                top_p=0.9
            )
            
            ai_answer = response.choices[0].message.content
            
            # Format the final answer
            answer = f"**🤔 Question:** {question}\n\n"
            answer += f"**💡 AI Analysis:**\n{ai_answer}\n\n"
            answer += f"**📚 Source References:**\n"
            for c in chunks:
                answer += f"• Page {c['metadata']['page']} (Relevance: {c['similarity']:.1%})\n"
            
            return answer, avg_conf
            
        except Exception as e:
            st.error(f"❌ LLM analysis failed: {e}")
            # Try alternative models if the first one fails
            return self._try_alternative_models(question, chunks, avg_conf)

    def _try_alternative_models(self, question, chunks, avg_conf):
        """Try alternative Groq models if the primary one fails"""
        alternative_models = [
            "llama-3.1-8b-instant",  # Fast alternative
            "mixtral-8x7b-32768",    # High quality alternative
            "llama-3.2-1b-preview"   # Lightweight alternative
        ]
        
        for model in alternative_models:
            try:
                st.info(f"🔄 Trying alternative model: {model}")
                context = "\n\n".join([
                    f"From Page {c['metadata']['page']}: {c['content']}" 
                    for c in chunks
                ])
                
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant that answers questions based on the provided document context. Provide accurate answers with page references."
                    },
                    {
                        "role": "user", 
                        "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer based on the context:"
                    }
                ]
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=800
                )
                
                ai_answer = response.choices[0].message.content
                answer = f"**🤔 Question:** {question}\n\n"
                answer += f"**💡 AI Analysis:**\n{ai_answer}\n\n"
                answer += f"**📚 Source References:**\n"
                for c in chunks:
                    answer += f"• Page {c['metadata']['page']} (Relevance: {c['similarity']:.1%})\n"
                
                st.success(f"✅ Using model: {model}")
                return answer, avg_conf
                
            except Exception as e:
                continue
        
        # If all models fail, use simple method
        st.error("❌ All LLM models failed, using basic analysis")
        return self._simple_answer(question, chunks, avg_conf)

    def _simple_answer(self, question, chunks, avg_conf):
        """Fallback method without LLM"""
        # Extract key sentences and create a better summary
        key_sentences = []
        for chunk in chunks:
            sentences = [s.strip() for s in chunk['content'].split('.') if s.strip()]
            # Take first 2 sentences from each relevant chunk
            key_sentences.extend(sentences[:2])
        
        # Remove duplicates and create summary
        unique_sentences = []
        for sentence in key_sentences:
            if sentence not in unique_sentences and len(sentence) > 20:
                unique_sentences.append(sentence)
        
        summary = "\n".join([f"• {sentence}." for sentence in unique_sentences[:6]])
        
        answer = f"**🤔 Question:** {question}\n\n"
        answer += f"**📊 Document Summary:**\n{summary}\n\n"
        answer += f"**📚 Sources:**\n"
        for c in chunks:
            answer += f"• Page {c['metadata']['page']} | Confidence: {c['similarity']:.1%}\n"
        
        return answer, avg_conf

# =============================================================================
# VOICE SERVICE
# =============================================================================
class VoiceService:
    def speak_text(self, text):
        try:
            tts = gTTS(text)
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(tmp_file.name)
            st.audio(tmp_file.name, format="audio/mp3")
        except Exception as e:
            st.error(f"❌ Voice generation failed: {e}")

# =============================================================================
# STREAMLIT APP
# =============================================================================
class AuraPDFQAApp:
    def __init__(self):
        # Initialize session state
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
        st.set_page_config(page_title="Aura PDF QA ⚡", layout="wide")

    def render_sidebar(self):
        st.sidebar.title("📚 Aura PDF QA")
        
        # Groq status
        if hasattr(self.llm_service, 'client') and self.llm_service.client:
            st.sidebar.success("🦙 Groq LLM: Connected")
        else:
            st.sidebar.warning("🦙 Groq LLM: Not connected")
            st.sidebar.info("Add GROQ_API_KEY to Streamlit secrets")
        
        uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
        
        # Show current status
        if st.session_state.pdf_processed:
            st.sidebar.success("✅ PDF is ready for questions!")
        else:
            st.sidebar.warning("⚠️ Upload and process a PDF to start")
        
        if uploaded_file:
            st.sidebar.write(f"**File:** {uploaded_file.name}")
            if st.sidebar.button("🚀 Process Document", type="primary", use_container_width=True):
                with st.spinner("Processing PDF... This may take a few seconds"):
                    count = st.session_state.doc_processor.process_pdf(uploaded_file)
                    if count > 0:
                        st.session_state.pdf_processed = True
                        st.session_state.pdf_name = uploaded_file.name
                        st.sidebar.success(f"✅ PDF processed with {count} chunks!")
                        st.rerun()
                    else:
                        st.session_state.pdf_processed = False
                        st.sidebar.error("❌ Failed to process PDF")
        
        top_k = st.sidebar.slider("Sources to retrieve", 1, 5, 3)
        enable_voice = st.sidebar.checkbox("Enable Voice", True)
        return top_k, enable_voice

    def render_chat(self, top_k, enable_voice):
        st.title("Aura PDF QA ⚡")
        st.markdown("Ask questions about your document and get AI-powered answers")
        
        # Show processing status clearly
        if not st.session_state.pdf_processed:
            st.error("❌ Please upload a PDF and click 'Process Document' first!")
            st.info("📝 Steps to use:")
            st.write("1. Upload PDF in sidebar")
            st.write("2. Click 'Process Document' button")  
            st.write("3. Wait for processing to complete")
            st.write("4. Start asking questions!")
            return

        # Display chat messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        question = st.chat_input("Ask a question about your document...")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("assistant"):
                with st.spinner("🔍 Analyzing document with AI..."):
                    start = time.time()
                    chunks = st.session_state.doc_processor.search_similar(question, top_k)
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
