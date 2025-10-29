# app.py - Aura PDF QA ⚡
import streamlit as st  # For creating web app
import tempfile, os, uuid, time  # For file operations and timing
from PyPDF2 import PdfReader  # For reading PDF files
from sentence_transformers import SentenceTransformer  # For text embeddings
import faiss  # For similarity search
import numpy as np  # For numerical operations
import pytesseract  # For OCR text extraction
from PIL import Image  # For image processing
import pdf2image  # For converting PDF to images
from gtts import gTTS  # For text-to-speech conversion

# =============================================================================
# CONFIGURATION SETTINGS
# =============================================================================
class Config:
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Model for creating text embeddings
    TOP_K_CHUNKS = 3  # Number of similar chunks to retrieve
    CHUNK_SIZE = 500  # Maximum size of each text chunk
    MIN_PARAGRAPH_LENGTH = 20  # Minimum length for paragraphs

# =============================================================================
# DOCUMENT PROCESSOR - Handles PDF processing and text extraction
# =============================================================================
class DocumentProcessor:
    def __init__(self):
        try:
            # Initialize the embedding model
            self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
            self.index = None  # Will store the search index
            self.chunks = []  # Will store text chunks from PDF
            self.chunk_metadata = []  # Will store information about each chunk
            st.success("✅ Document processor ready")  # Show success message
        except Exception as e:
            st.error(f"❌ Document processor failed: {e}")  # Show error if initialization fails

    def extract_text_direct(self, pdf_path):
        # Extract text directly from PDF (for text-based PDFs)
        reader = PdfReader(pdf_path)  # Create PDF reader object
        extracted = []  # List to store extracted text
        # Loop through each page in the PDF
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""  # Extract text from page
            if text.strip():  # If text is not empty
                extracted.append({"page": i, "text": text.strip()})  # Store page number and text
        return extracted  # Return all extracted text

    def extract_text_ocr(self, pdf_path):
        # Extract text using OCR (for scanned PDFs)
        images = pdf2image.convert_from_path(pdf_path, dpi=200)  # Convert PDF pages to images
        extracted = []  # List to store OCR results
        progress_bar = st.progress(0)  # Create progress bar
        status_text = st.empty()  # Create empty text for status updates
        # Process each image
        for i, image in enumerate(images):
            status_text.text(f"📷 OCR page {i+1}/{len(images)}")  # Update status message
            text = pytesseract.image_to_string(image)  # Extract text from image using OCR
            if text.strip():  # If text was found
                extracted.append({"page": i + 1, "text": text.strip()})  # Store page number and text
            progress_bar.progress((i + 1) / len(images))  # Update progress bar
        status_text.text("✅ OCR complete")  # Show completion message
        return extracted  # Return OCR results

    def analyze_pdf_type(self, pdf_path):
        # Determine if PDF is text-based or scanned
        reader = PdfReader(pdf_path)  # Create PDF reader
        # Count pages with substantial text (more than 50 characters)
        text_pages = sum(1 for page in reader.pages if len((page.extract_text() or "").strip()) > 50)
        total_pages = len(reader.pages)  # Total number of pages
        # Return "text_based" if more than 50% of pages have text, otherwise "scanned"
        return "text_based" if total_pages == 0 or text_pages / total_pages > 0.5 else "scanned"

    def process_pdf(self, uploaded_file):
        # Create temporary file to store uploaded PDF
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_file.write(uploaded_file.getvalue())  # Write uploaded content to temp file
        tmp_file.close()  # Close the file
        pdf_path = tmp_file.name  # Get the path to the temporary file

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
                # Split text into paragraphs (separated by double newlines)
                paragraphs = [p.strip() for p in item["text"].split("\n\n") if len(p.strip()) >= Config.MIN_PARAGRAPH_LENGTH]
                for para in paragraphs:  # Process each paragraph
                    if len(para) > Config.CHUNK_SIZE:  # If paragraph is too long, split it
                        sentences = para.split(". ")  # Split into sentences
                        chunk = ""  # Initialize current chunk
                        for s in sentences:  # Process each sentence
                            if len(chunk + s) < Config.CHUNK_SIZE:  # If chunk not full
                                chunk += s + ". "  # Add sentence to chunk
                            else:  # If chunk is full
                                self.chunks.append(chunk.strip())  # Save current chunk
                                self.chunk_metadata.append({"page": page, "method": method})  # Save metadata
                                chunk = s + ". "  # Start new chunk with current sentence
                        if chunk:  # If there's remaining text
                            self.chunks.append(chunk.strip())  # Save last chunk
                            self.chunk_metadata.append({"page": page, "method": method})  # Save metadata
                    else:  # If paragraph fits in one chunk
                        self.chunks.append(para)  # Save paragraph as chunk
                        self.chunk_metadata.append({"page": page, "method": method})  # Save metadata

            if not self.chunks:  # If no text was extracted
                st.error("❌ No text extracted from PDF.")  # Show error message
                return 0  # Return 0 chunks

            # Create embeddings and search index
            st.info("🧠 Creating embeddings...")  # Show embedding message
            embeddings = self.embedder.encode(self.chunks)  # Convert text chunks to vectors
            self.index = faiss.IndexFlatL2(embeddings.shape[1])  # Create FAISS search index
            self.index.add(np.array(embeddings))  # Add embeddings to the index
            st.success(f"✅ PDF processed: {len(self.chunks)} chunks created")  # Show success message
            return len(self.chunks)  # Return number of chunks created

        finally:
            os.unlink(pdf_path)  # Clean up temporary file

    def search_similar(self, query, top_k=3):
        # Search for text chunks similar to the query
        if not self.chunks or self.index is None:  # If no data has been processed
            return []  # Return empty list

        query_vec = self.embedder.encode([query])  # Convert query to vector
        distances, indices = self.index.search(np.array(query_vec), top_k)  # Search for similar chunks
        results = []  # List to store results
        for i, idx in enumerate(indices[0]):  # Process each result
            if idx < len(self.chunks):  # If index is valid
                sim = 1 / (1 + distances[0][i])  # Calculate similarity score
                # Add result to list with content, metadata, and similarity
                results.append({"content": self.chunks[idx], "metadata": self.chunk_metadata[idx], "similarity": round(sim, 3)})
        results.sort(key=lambda x: x["similarity"], reverse=True)  # Sort by similarity (highest first)
        return results  # Return sorted results

# =============================================================================
# LLM SERVICE - Generates answers based on retrieved chunks
# =============================================================================
class LLMService:
    def generate_answer(self, question, chunks):
        if not chunks:  # If no relevant chunks found
            return f"❌ No relevant info found for '{question}'", 0.0  # Return error message
        avg_conf = sum(c["similarity"] for c in chunks)/len(chunks)  # Calculate average confidence
        summary = self._summarize(chunks)  # Generate summary from chunks
        # Format the answer with question, answer, and sources
        answer = f"**Question:** {question}\n\n**Answer:**\n{summary}\n\n**Sources:**\n"
        for i, c in enumerate(chunks):
            # Add source information for each chunk
            answer += f"Page {c['metadata']['page']} ({c['metadata']['method']}) | Conf: {c['similarity']:.1%}\n{c['content'][:200]}...\n\n"
        return answer, avg_conf  # Return formatted answer and confidence

    def _summarize(self, chunks):
        # Create a summary from the chunks
        all_text = " ".join([c['content'] for c in chunks])  # Combine all chunk content
        sentences = [s.strip() for s in all_text.split(".") if len(s.strip()) > 20]  # Split into sentences
        summary = "\n".join([f"• {s}." for s in sentences[:5]])  # Create bullet points from first 5 sentences
        return summary  # Return the summary

# =============================================================================
# VOICE SERVICE - Handles text-to-speech functionality
# =============================================================================
class VoiceService:
    def speak_text(self, text):
        tts = gTTS(text)  # Create text-to-speech object
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")  # Create temporary audio file
        tts.save(tmp_file.name)  # Save speech to file
        st.audio(tmp_file.name, format="audio/mp3")  # Display audio player in Streamlit

# =============================================================================
# STREAMLIT APP - Main application class
# =============================================================================
class AuraPDFQAApp:
    def __init__(self):
        self.doc_processor = DocumentProcessor()  # Initialize document processor
        self.llm_service = LLMService()  # Initialize LLM service
        self.voice_service = VoiceService()  # Initialize voice service
        self.setup_ui()  # Setup user interface

    def setup_ui(self):
        st.set_page_config(page_title="Aura PDF QA ⚡", layout="wide")  # Configure Streamlit page

    def render_sidebar(self):
        st.sidebar.title("📚 Aura PDF QA")  # Sidebar title
        uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")  # PDF uploader
        if uploaded_file and st.sidebar.button("Process Document"):  # If file uploaded and process button clicked
            count = self.doc_processor.process_pdf(uploaded_file)  # Process the PDF
            if count > 0:  # If processing successful
                st.session_state.pdf_processed = True  # Set processed flag
                st.session_state.pdf_name = uploaded_file.name  # Store PDF name
                st.success(f"PDF processed with {count} chunks.")  # Show success message
        top_k = st.sidebar.slider("Sources to retrieve", 1, 5, 3)  # Slider for number of sources
        enable_voice = st.sidebar.checkbox("Enable Voice", True)  # Checkbox for voice feature
        return top_k, enable_voice  # Return settings

    def render_chat(self, top_k, enable_voice):
        st.title("Aura PDF QA ⚡")  # Main title
        st.markdown("Ask questions about your document and get AI-powered answers")  # Subtitle
        
        # Initialize chat messages in session state if not exists
        if 'messages' not in st.session_state:
            st.session_state.messages = []  # Empty message history
        
        # Initialize PDF processed flag if not exists
        if 'pdf_processed' not in st.session_state:
            st.session_state.pdf_processed = False  # Not processed initially

        # Display all previous messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):  # Create chat bubble for each message
                st.markdown(msg["content"])  # Display message content

        # Chat input for new questions
        question = st.chat_input("Ask a question...")  # Chat input widget
        if question:  # If user entered a question
            if not st.session_state.pdf_processed:  # If no PDF has been processed
                st.error("Upload and process a PDF first!")  # Show error message
                return
            
            # Add user question to message history
            st.session_state.messages.append({"role": "user", "content": question})
            
            with st.chat_message("assistant"):  # Assistant response bubble
                with st.spinner("Searching..."):  # Show loading spinner
                    start = time.time()  # Start timer
                    chunks = self.doc_processor.search_similar(question, top_k)  # Search for relevant chunks
                    answer, conf = self.llm_service.generate_answer(question, chunks)  # Generate answer
                    elapsed = (time.time() - start) * 1000  # Calculate response time
                    
                    st.markdown(answer)  # Display the answer
                    st.caption(f"⏱️ {elapsed:.0f}ms | Confidence: {conf:.1%}")  # Show performance metrics
                    
                    # Play voice if enabled and confidence is good
                    if enable_voice and conf > 0.1:
                        self.voice_service.speak_text(answer)  # Convert answer to speech
            
            # Add assistant answer to message history
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
