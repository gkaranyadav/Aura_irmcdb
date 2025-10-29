# app.py - Aura PDF QA ⚡
# Streamlit PDF Q&A app with auto TTS (Voice output), minimal UI messages

import streamlit as st  # Import Streamlit for web app interface
import tempfile, os, time, base64, re  # Import utilities for file handling, time, encoding, and regex
from PyPDF2 import PdfReader  # Import PDF reader for text extraction
from sentence_transformers import SentenceTransformer  # Import for text embeddings
import faiss  # Import for similarity search
import numpy as np  # Import for numerical operations
import pytesseract  # Import for OCR text extraction
import pdf2image  # Import for converting PDF to images
from gtts import gTTS  # Import for text-to-speech
from groq import Groq  # Import Groq LLM API

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Model for creating text embeddings
    GROQ_MODEL = "llama-3.3-70b-versatile"  # Groq model for answering questions
    CHUNK_SIZE = 500  # Maximum size of each text chunk
    MIN_PARAGRAPH_LENGTH = 20  # Minimum length for paragraphs

# =============================================================================
# DOCUMENT PROCESSOR
# =============================================================================
class DocumentProcessor:
    def __init__(self):
        try:
            self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)  # Load embedding model
            self.index = None  # Initialize search index as None
            self.chunks = []  # Initialize empty list for text chunks
            self.chunk_metadata = []  # Initialize empty list for chunk metadata
            st.success("✅ Document processor ready")  # Show success message
        except Exception as e:
            st.error(f"❌ Document processor failed: {e}")  # Show error if initialization fails

    # Extract text directly from text-based PDF
    def extract_text_direct(self, pdf_path):
        reader = PdfReader(pdf_path)  # Create PDF reader object
        extracted = []  # Initialize empty list for extracted text
        for i, page in enumerate(reader.pages, 1):  # Loop through each page starting from 1
            text = page.extract_text() or ""  # Extract text from page, use empty string if None
            if text.strip():  # Check if text is not empty
                extracted.append({"page": i, "text": text.strip()})  # Add page number and text to list
        return extracted  # Return extracted text

    # Extract text via OCR for scanned PDFs
    def extract_text_ocr(self, pdf_path):
        images = pdf2image.convert_from_path(pdf_path, dpi=200)  # Convert PDF pages to images
        extracted = []  # Initialize empty list for OCR results
        progress_bar = st.progress(0)  # Create progress bar
        status_text = st.empty()  # Create empty text for status updates
        for i, image in enumerate(images):  # Loop through each image
            status_text.text(f"Knowledge at your command page {i+1}/{len(images)}")  # Update status message
            text = pytesseract.image_to_string(image)  # Extract text from image using OCR
            if text.strip():  # Check if text was found
                extracted.append({"page": i + 1, "text": text.strip()})  # Add page number and text to list
            progress_bar.progress((i + 1) / len(images))  # Update progress bar
        status_text.text("✅ PDF processed")  # Show completion message
        return extracted  # Return OCR results

    # Determine if PDF is text-based or scanned
    def analyze_pdf_type(self, pdf_path):
        reader = PdfReader(pdf_path)  # Create PDF reader
        # Count pages with substantial text (more than 50 characters)
        text_pages = sum(1 for page in reader.pages if len((page.extract_text() or "").strip()) > 50)
        total_pages = len(reader.pages)  # Get total number of pages
        # Return "text_based" if more than 50% of pages have text, otherwise "scanned"
        return "text_based" if total_pages == 0 or text_pages / total_pages > 0.5 else "scanned"

    # Process PDF: extract, chunk, and embed
    def process_pdf(self, uploaded_file):
        self.chunks, self.chunk_metadata, self.index = [], [], None  # Reset chunks, metadata, and index
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")  # Create temporary file
        tmp_file.write(uploaded_file.getvalue())  # Write uploaded content to temp file
        tmp_file.close()  # Close the file
        pdf_path = tmp_file.name  # Get path to temporary file

        try:
            st.info("Knowledge at your command!")  # Show processing message
            pdf_type = self.analyze_pdf_type(pdf_path)  # Determine PDF type
            if pdf_type == "text_based":  # If PDF is text-based
                extracted = self.extract_text_direct(pdf_path)  # Use direct text extraction
                method = "text"  # Set method to text
            else:  # If PDF is scanned
                extracted = self.extract_text_ocr(pdf_path)  # Use OCR extraction
                method = "ocr"  # Set method to OCR

            # Chunk text into manageable sizes
            for item in extracted:  # Process each extracted page
                page = item["page"]  # Get page number
                # Split text into paragraphs (separated by double newlines)
                paragraphs = [p.strip() for p in item["text"].split("\n\n") if len(p.strip()) >= Config.MIN_PARAGRAPH_LENGTH]
                for para in paragraphs:  # Process each paragraph
                    if len(para) > Config.CHUNK_SIZE:  # If paragraph is too long
                        sentences = para.split(". ")  # Split into sentences
                        chunk = ""  # Initialize current chunk
                        for s in sentences:  # Process each sentence
                            if len(chunk + s) < Config.CHUNK_SIZE:  # If chunk not full
                                chunk += s + ". "  # Add sentence to chunk
                            else:  # If chunk is full
                                if chunk.strip():  # If chunk has content
                                    self.chunks.append(chunk.strip())  # Save current chunk
                                    self.chunk_metadata.append({"page": page, "method": method})  # Save metadata
                                chunk = s + ". "  # Start new chunk with current sentence
                        if chunk.strip():  # If there's remaining text
                            self.chunks.append(chunk.strip())  # Save last chunk
                            self.chunk_metadata.append({"page": page, "method": method})  # Save metadata
                    else:  # If paragraph fits in one chunk
                        if para.strip():  # If paragraph has content
                            self.chunks.append(para)  # Save paragraph as chunk
                            self.chunk_metadata.append({"page": page, "method": method})  # Save metadata

            # Create embeddings
            embeddings = self.embedder.encode(self.chunks)  # Convert text chunks to vectors
            self.index = faiss.IndexFlatL2(embeddings.shape[1])  # Create FAISS search index
            self.index.add(np.array(embeddings))  # Add embeddings to the index
            st.success("✅ PDF processed")  # Show success message
            return len(self.chunks)  # Return number of chunks created
        finally:
            if os.path.exists(pdf_path):  # Check if temp file exists
                os.unlink(pdf_path)  # Delete temporary file

    # Search for relevant chunks
    def search_similar(self, query, top_k=3):
        if not self.chunks or self.index is None:  # If no data has been processed
            st.warning("⚠️ No document processed yet")  # Show warning
            return []  # Return empty list
        query_vec = self.embedder.encode([query])  # Convert query to vector
        distances, indices = self.index.search(np.array(query_vec), top_k)  # Search for similar chunks
        results = []  # Initialize empty list for results
        for i, idx in enumerate(indices[0]):  # Process each result
            if idx < len(self.chunks):  # If index is valid
                sim = 1 / (1 + distances[0][i])  # Calculate similarity score
                # Add result to list with content, metadata, and similarity
                results.append({"content": self.chunks[idx], "metadata": self.chunk_metadata[idx], "similarity": round(sim, 3)})
        results.sort(key=lambda x: x["similarity"], reverse=True)  # Sort by similarity (highest first)
        return results  # Return sorted results

# =============================================================================
# LLM SERVICE
# =============================================================================
class LLMService:
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])  # Initialize Groq client with API key
            st.sidebar.success("✅ Groq LLM connected")  # Show success message in sidebar
        except Exception as e:
            st.error(f"❌ Groq initialization failed: {e}")  # Show error if initialization fails
            self.client = None  # Set client to None

    # Generate answer from chunks
    def generate_answer(self, question, chunks):
        if not chunks:  # If no relevant chunks found
            return f"❌ No relevant info found for '{question}'", 0.0  # Return error message
        avg_conf = sum(c["similarity"] for c in chunks)/len(chunks)  # Calculate average confidence
        if self.client:  # If Groq client is available
            return self._generate_llm_answer(question, chunks, avg_conf)  # Use LLM for answer
        else:  # If Groq client is not available
            return self._simple_answer(question, chunks, avg_conf)  # Use simple answer

    # Use LLM
    def _generate_llm_answer(self, question, chunks, avg_conf):
        try:
            context = "\n\n".join([f"Page {c['metadata']['page']}: {c['content']}" for c in chunks])  # Prepare context from chunks
            messages = [
                {"role": "system", "content": "You are an expert document analyst. Only use the provided context."},  # System prompt
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}  # User question with context
            ]
            response = self.client.chat.completions.create(  # Send request to Groq API
                model=Config.GROQ_MODEL,  # Use configured model
                messages=messages,  # Pass messages
                temperature=0.1,  # Low temperature for consistent responses
                max_tokens=1024  # Maximum tokens in response
            )
            ai_answer = response.choices[0].message.content  # Extract AI response
            return ai_answer, avg_conf  # Return answer and confidence
        except:  # If LLM fails
            return self._simple_answer(question, chunks, avg_conf)  # Fall back to simple answer

    # Simple fallback answer
    def _simple_answer(self, question, chunks, avg_conf):
        key_sentences = []  # Initialize empty list for key sentences
        for chunk in chunks:  # Process each chunk
            sentences = [s.strip() for s in chunk['content'].split('.') if s.strip()]  # Split chunk into sentences
            key_sentences.extend(sentences[:2])  # Add first 2 sentences to key sentences
        summary = ' '.join(key_sentences[:6])  # Join first 6 sentences into summary
        return summary, avg_conf  # Return summary and confidence

# =============================================================================
# VOICE SERVICE
# =============================================================================
class VoiceService:
    # Clean text to avoid reading weird symbols
    def clean_text_for_tts(self, text):
        text = re.sub(r'[^\w\s.,?-]', '', text)  # Remove special characters except basic punctuation
        replacements = {"%": " percent", "$": " dollars", "°": " degrees", "&": " and "}  # Define symbol replacements
        for k, v in replacements.items():  # Loop through replacements
            text = text.replace(k, v)  # Replace symbols with words
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        return text  # Return cleaned text

    # Speak answer automatically
    def speak_text(self, text):
        text = self.clean_text_for_tts(text)  # Clean text for TTS
        tts = gTTS(text)  # Create text-to-speech object
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")  # Create temporary audio file
        tts.save(tmp_file.name)  # Save speech to file
        with open(tmp_file.name, "rb") as f:  # Open audio file in binary mode
            audio_bytes = f.read()  # Read audio bytes
        audio_base64 = base64.b64encode(audio_bytes).decode()  # Encode audio to base64
        html = f"""<audio autoplay><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>"""  # Create HTML audio element
        st.components.v1.html(html, height=50)  # Display auto-playing audio

# =============================================================================
# STREAMLIT APP
# =============================================================================
class AuraPDFQAApp:
    def __init__(self):
        # Session state initialization
        if 'pdf_processed' not in st.session_state:  # Check if PDF processed flag exists
            st.session_state.pdf_processed = False  # Initialize as not processed
        if 'messages' not in st.session_state:  # Check if messages exist
            st.session_state.messages = []  # Initialize empty message list
        if 'doc_processor' not in st.session_state:  # Check if document processor exists
            st.session_state.doc_processor = DocumentProcessor()  # Initialize document processor
        self.llm_service = LLMService()  # Initialize LLM service
        self.voice_service = VoiceService()  # Initialize voice service
        st.set_page_config(page_title="📚 IRMC Aura", layout="wide")  # Configure Streamlit page

    # Sidebar UI
    def render_sidebar(self):
        st.sidebar.title("Fast & reliable ⚡")  # Sidebar title
        uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")  # PDF uploader
        if st.session_state.pdf_processed:  # If PDF is processed
            st.sidebar.success("✅ PDF ready")  # Show ready message
        else:  # If PDF is not processed
            st.sidebar.warning("⚠️ Upload PDF first")  # Show warning
        if uploaded_file:  # If file is uploaded
            st.sidebar.write(f"**File:** {uploaded_file.name}")  # Show file name
            if st.sidebar.button("🚀 Process Document"):  # Process button
                with st.spinner("Processing....."):  # Show spinner
                    count = st.session_state.doc_processor.process_pdf(uploaded_file)  # Process PDF
                    if count > 0:  # If processing successful
                        st.session_state.pdf_processed = True  # Set processed flag
                        st.session_state.pdf_name = uploaded_file.name  # Store PDF name
                        st.sidebar.success(f"✅ PDF processed ({count} chunks)")  # Show success
        top_k = st.sidebar.slider("Sources to retrieve", 1, 5, 3)  # Slider for number of sources
        enable_voice = st.sidebar.checkbox("Enable Voice", True)  # Checkbox for voice feature
        return top_k, enable_voice  # Return settings

    # Main chat
    def render_chat(self, top_k, enable_voice):
        # Center the title using HTML/CSS
        st.markdown("""
            <style>
            .centered-title {
                text-align: center;
            }
            </style>
            <h1 class="centered-title">IRMC Aura 📚</h1>
        """, unsafe_allow_html=True)  # Apply centered title styling
        
        if not st.session_state.pdf_processed:  # If no PDF processed
            st.error("❌ Upload PDF and click Process first!")  # Show error
            return  # Exit function
        
        for msg in st.session_state.messages:  # Display all previous messages
            with st.chat_message(msg["role"]):  # Create chat bubble
                st.markdown(msg["content"])  # Display message content
        
        question = st.chat_input("Ask a question about your document...")  # Chat input widget
        if question:  # If user entered a question
            st.session_state.messages.append({"role": "user", "content": question})  # Add user message to history
            with st.chat_message("assistant"):  # Assistant response bubble
                with st.spinner("Just a sec........."):  # Show loading spinner
                    chunks = st.session_state.doc_processor.search_similar(question, top_k)  # Search for relevant chunks
                    answer, conf = self.llm_service.generate_answer(question, chunks)  # Generate answer
                    st.markdown(answer)  # Display the answer
                    if enable_voice:  # If voice is enabled
                        self.voice_service.speak_text(answer)  # Speak the answer
            st.session_state.messages.append({"role": "assistant", "content": answer})  # Add assistant answer to history

    # Run app
    def run(self):
        top_k, enable_voice = self.render_sidebar()  # Render sidebar and get settings
        self.render_chat(top_k, enable_voice)  # Render main chat interface

if __name__ == "__main__":
    app = AuraPDFQAApp()  # Create app instance
    app.run()  # Run the app
