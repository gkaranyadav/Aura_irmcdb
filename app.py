# app.py - Aura PDF QA ⚡
# Import all necessary libraries
import streamlit as st  # For creating web interface
import tempfile, os, uuid, time  # For file handling and utilities
from PyPDF2 import PdfReader  # For reading PDF files
from sentence_transformers import SentenceTransformer  # For text embeddings
import faiss  # For similarity search
import numpy as np  # For numerical operations
import pytesseract  # For OCR text extraction
from PIL import Image  # For image processing
import pdf2image  # For converting PDF to images
from gtts import gTTS  # For text-to-speech
from groq import Groq  # For Groq LLM API

# =============================================================================
# CONFIGURATION SETTINGS
# =============================================================================
class Config:
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Model for text embeddings
    GROQ_MODEL = "llama-3.1-70b-versatile"  # Groq model for answering questions
    TOP_K_CHUNKS = 3  # Number of text chunks to retrieve
    CHUNK_SIZE = 500  # Maximum size of each text chunk
    MIN_PARAGRAPH_LENGTH = 20  # Minimum length for paragraphs

# =============================================================================
# DOCUMENT PROCESSOR - Handles PDF reading and text extraction
# =============================================================================
class DocumentProcessor:
    def __init__(self):
        try:
            # Load the embedding model for converting text to vectors
            self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
            self.index = None  # Will store the search index
            self.chunks = []  # Will store text chunks from PDF
            self.chunk_metadata = []  # Will store info about each chunk
            st.success("✅ Document processor ready")  # Show success message
        except Exception as e:
            st.error(f"❌ Document processor failed: {e}")  # Show error if failed

    def extract_text_direct(self, pdf_path):
        # Extract text directly from PDF (for text-based PDFs)
        reader = PdfReader(pdf_path)  # Create PDF reader object
        extracted = []  # List to store extracted text
        # Loop through each page in PDF
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""  # Extract text, use empty string if None
            if text.strip():  # If text is not empty
                extracted.append({"page": i, "text": text.strip()})  # Store page number and text
        return extracted  # Return all extracted text

    def extract_text_ocr(self, pdf_path):
        # Extract text using OCR (for scanned PDFs)
        images = pdf2image.convert_from_path(pdf_path, dpi=200)  # Convert PDF to images
        extracted = []  # List to store OCR results
        progress_bar = st.progress(0)  # Create progress bar
        status_text = st.empty()  # Create empty text for status updates
        # Process each image
        for i, image in enumerate(images):
            status_text.text(f"📷 OCR page {i+1}/{len(images)}")  # Update status
            text = pytesseract.image_to_string(image)  # Extract text from image using OCR
            if text.strip():  # If text found
                extracted.append({"page": i + 1, "text": text.strip()})  # Store result
            progress_bar.progress((i + 1) / len(images))  # Update progress bar
        status_text.text("✅ OCR complete")  # Show completion message
        return extracted  # Return OCR results

    def analyze_pdf_type(self, pdf_path):
        # Determine if PDF is text-based or scanned
        reader = PdfReader(pdf_path)  # Create PDF reader
        # Count pages with substantial text
        text_pages = sum(1 for page in reader.pages if len((page.extract_text() or "").strip()) > 50)
        total_pages = len(reader.pages)  # Total number of pages
        # Return "text_based" if more than 50% pages have text, else "scanned"
        return "text_based" if total_pages == 0 or text_pages / total_pages > 0.5 else "scanned"

    def process_pdf(self, uploaded_file):
        # Main method to process uploaded PDF
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")  # Create temp file
        tmp_file.write(uploaded_file.getvalue())  # Write uploaded content to temp file
        tmp_file.close()  # Close file
        pdf_path = tmp_file.name  # Get temp file path

        try:
            st.info("🔍 Analyzing PDF type...")  # Show analysis message
            pdf_type = self.analyze_pdf_type(pdf_path)  # Determine PDF type
            st.info(f"PDF type detected: {pdf_type}")  # Show detected type

            # Extract text based on PDF type
            if pdf_type == "text_based":
                extracted = self.extract_text_direct(pdf_path)  # Use direct text extraction
                method = "text"  # Set method flag
            else:
                extracted = self.extract_text_ocr(pdf_path)  # Use OCR extraction
                method = "ocr"  # Set method flag

            # Split text into chunks for processing
            self.chunks, self.chunk_metadata = [], []  # Reset chunks and metadata
            for item in extracted:  # Process each extracted page
                page = item["page"]  # Get page number
                # Split text into paragraphs
                paragraphs = [p.strip() for p in item["text"].split("\n\n") if len(p.strip()) >= Config.MIN_PARAGRAPH_LENGTH]
                for para in paragraphs:  # Process each paragraph
                    if len(para) > Config.CHUNK_SIZE:  # If paragraph is too long
                        sentences = para.split(". ")  # Split into sentences
                        chunk = ""  # Initialize chunk
                        for s in sentences:  # Process each sentence
                            if len(chunk + s) < Config.CHUNK_SIZE:  # If chunk not full
                                chunk += s + ". "  # Add sentence to chunk
                            else:  # If chunk is full
                                self.chunks.append(chunk.strip())  # Save current chunk
                                self.chunk_metadata.append({"page": page, "method": method})  # Save metadata
                                chunk = s + ". "  # Start new chunk
                        if chunk:  # If remaining text
                            self.chunks.append(chunk.strip())  # Save last chunk
                            self.chunk_metadata.append({"page": page, "method": method})  # Save metadata
                    else:  # If paragraph fits in one chunk
                        self.chunks.append(para)  # Save paragraph as chunk
                        self.chunk_metadata.append({"page": page, "method": method})  # Save metadata

            if not self.chunks:  # If no text extracted
                st.error("❌ No text extracted from PDF.")  # Show error
                return 0  # Return 0 chunks

            # Create embeddings and search index
            st.info("🧠 Creating embeddings...")  # Show embedding message
            embeddings = self.embedder.encode(self.chunks)  # Convert chunks to vectors
            self.index = faiss.IndexFlatL2(embeddings.shape[1])  # Create FAISS index
            self.index.add(np.array(embeddings))  # Add embeddings to index
            st.success(f"✅ PDF processed: {len(self.chunks)} chunks created")  # Show success
            return len(self.chunks)  # Return number of chunks

        finally:
            os.unlink(pdf_path)  # Clean up temp file

    def search_similar(self, query, top_k=3):
        # Search for similar text chunks
        if not self.chunks or self.index is None:  # If no data processed
            return []  # Return empty list

        query_vec = self.embedder.encode([query])  # Convert query to vector
        distances, indices = self.index.search(np.array(query_vec), top_k)  # Search index
        results = []  # List for results
        for i, idx in enumerate(indices[0]):  # Process each result
            if idx < len(self.chunks):  # If valid index
                sim = 1 / (1 + distances[0][i])  # Calculate similarity score
                # Add result to list
                results.append({"content": self.chunks[idx], "metadata": self.chunk_metadata[idx], "similarity": round(sim, 3)})
        results.sort(key=lambda x: x["similarity"], reverse=True)  # Sort by similarity
        return results  # Return sorted results

# =============================================================================
# GROQ LLM SERVICE - Handles AI responses using Groq
# =============================================================================
class LLMService:
    def __init__(self):
        # Initialize Groq client with API key
        self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    def generate_answer(self, question, chunks):
        if not chunks:  # If no relevant chunks found
            return f"❌ No relevant info found for '{question}'", 0.0  # Return error
        
        # Calculate average confidence from chunks
        avg_conf = sum(c["similarity"] for c in chunks)/len(chunks)
        
        # Prepare context from chunks for the AI
        context = "\n\n".join([f"Page {c['metadata']['page']}: {c['content']}" for c in chunks])
        
        # Create prompt for Groq
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that answers questions based on the provided document context. Provide accurate, concise answers and cite page numbers."
            },
            {
                "role": "user", 
                "content": f"Context from document:\n{context}\n\nQuestion: {question}\n\nAnswer based on the context above:"
            }
        ]
        
        try:
            # Get response from Groq API
            response = self.client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=1024
            )
            
            # Extract AI response
            ai_answer = response.choices[0].message.content
            
            # Format final answer with sources
            answer = f"**Question:** {question}\n\n**Answer:**\n{ai_answer}\n\n**Sources:**\n"
            for i, c in enumerate(chunks):
                answer += f"• Page {c['metadata']['page']} (Confidence: {c['similarity']:.1%})\n"
            
            return answer, avg_conf  # Return answer and confidence
            
        except Exception as e:
            return f"❌ Error generating answer: {str(e)}", 0.0  # Return error

# =============================================================================
# VOICE SERVICE - Handles text-to-speech
# =============================================================================
class VoiceService:
    def speak_text(self, text):
        tts = gTTS(text)  # Create text-to-speech object
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")  # Create temp file
        tts.save(tmp_file.name)  # Save audio to file
        st.audio(tmp_file.name, format="audio/mp3")  # Display audio player

# =============================================================================
# STREAMLIT APP - Main application class
# =============================================================================
class AuraPDFQAApp:
    def __init__(self):
        self.doc_processor = DocumentProcessor()  # Create document processor
        self.llm_service = LLMService()  # Create LLM service
        self.voice_service = VoiceService()  # Create voice service
        self.setup_ui()  # Setup user interface

    def setup_ui(self):
        st.set_page_config(page_title="Aura PDF QA ⚡", layout="wide")  # Configure page

    def render_sidebar(self):
        st.sidebar.title("📚 Aura PDF QA")  # Sidebar title
        
        # File uploader for PDF
        uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
        
        # Process button
        if uploaded_file and st.sidebar.button("Process Document"):
            count = self.doc_processor.process_pdf(uploaded_file)  # Process PDF
            if count > 0:  # If successful
                st.session_state.pdf_processed = True  # Set flag
                st.session_state.pdf_name = uploaded_file.name  # Store PDF name
                st.success(f"PDF processed with {count} chunks.")  # Show success
        
        # Settings
        top_k = st.sidebar.slider("Sources to retrieve", 1, 5, 3)  # Number of sources
        enable_voice = st.sidebar.checkbox("Enable Voice", True)  # Voice toggle
        
        return top_k, enable_voice  # Return settings

    def render_chat(self, top_k, enable_voice):
        st.title("Aura PDF QA ⚡")  # Main title
        st.markdown("Ask questions about your document and get AI-powered answers")  # Subtitle
        
        # Initialize session state for messages
        if 'messages' not in st.session_state:
            st.session_state.messages = []  # Empty message history
        
        # Initialize PDF processed flag
        if 'pdf_processed' not in st.session_state:
            st.session_state.pdf_processed = False  # Not processed initially

        # Display chat messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):  # Create chat bubble
                st.markdown(msg["content"])  # Display message content

        # Chat input
        question = st.chat_input("Ask a question...")
        if question:
            if not st.session_state.pdf_processed:  # If no PDF processed
                st.error("Upload and process a PDF first!")  # Show error
                return
            
            # Add user question to history
            st.session_state.messages.append({"role": "user", "content": question})
            
            with st.chat_message("assistant"):  # Assistant response
                with st.spinner("Searching..."):  # Show loading spinner
                    start = time.time()  # Start timer
                    chunks = self.doc_processor.search_similar(question, top_k)  # Search for relevant chunks
                    answer, conf = self.llm_service.generate_answer(question, chunks)  # Generate answer
                    elapsed = (time.time() - start) * 1000  # Calculate response time
                    
                    st.markdown(answer)  # Display answer
                    st.caption(f"⏱️ {elapsed:.0f}ms | Confidence: {conf:.1%}")  # Show metrics
                    
                    # Play voice if enabled and confident
                    if enable_voice and conf > 0.1:
                        self.voice_service.speak_text(answer)  # Convert to speech
            
            # Add assistant answer to history
            st.session_state.messages.append({"role": "assistant", "content": answer})

    def run(self):
        top_k, enable_voice = self.render_sidebar()  # Render sidebar and get settings
        self.render_chat(top_k, enable_voice)  # Render main chat interface

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    app = AuraPDFQAApp()  # Create app instance
    app.run()  # Run the app
