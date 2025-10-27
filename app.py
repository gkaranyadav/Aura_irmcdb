# app.py - IRMC AskPro ⚡ SIMPLE WORKING VERSION
import streamlit as st
import tempfile
import os
from datetime import datetime
import uuid
import logging
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from databricks.sdk import WorkspaceClient
import time

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    DATABRICKS_HOST = "https://dbc-484c2988-d6e6.cloud.databricks.com"
    DATABRICKS_TOKEN = "dapiaa126510ff360ca569ec6d125bc727d1"
    HF_TOKEN = "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"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    TOP_K_CHUNKS = 3
    CHUNK_SIZE = 500

# =============================================================================
# SIMPLE DOCUMENT PROCESSOR (NO OCR)
# =============================================================================
class DocumentProcessor:
    def __init__(self):
        try:
            st.info("🔄 Loading AI model...")
            self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
            self.index = None
            self.chunks = []
            self.chunk_metadata = []
            st.success("✅ Document processor ready!")
        except Exception as e:
            st.error(f"❌ Model loading failed: {e}")
    
    def process_pdf(self, pdf_file):
        try:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_path = tmp_file.name
            
            # Read PDF
            status_text.text("📖 Reading PDF...")
            reader = PdfReader(tmp_path)
            total_pages = len(reader.pages)
            
            self.chunks = []
            self.chunk_metadata = []
            
            # Extract text from each page
            for page_num, page in enumerate(reader.pages, 1):
                status_text.text(f"📄 Processing page {page_num}/{total_pages}...")
                progress_bar.progress(page_num / total_pages)
                
                text = page.extract_text() or ""
                
                if text.strip():
                    # Simple text cleaning
                    clean_text = ' '.join(text.split())
                    
                    # Split into chunks
                    words = clean_text.split()
                    for i in range(0, len(words), Config.CHUNK_SIZE):
                        chunk = ' '.join(words[i:i + Config.CHUNK_SIZE])
                        if len(chunk) > 50:  # Only keep substantial chunks
                            self.chunks.append(chunk)
                            self.chunk_metadata.append({
                                "page": page_num,
                                "chunk_id": f"page_{page_num}_chunk_{len(self.chunks)}"
                            })
            
            # Create embeddings
            if self.chunks:
                status_text.text("🧠 Creating search index...")
                embeddings = self.embedder.encode(self.chunks)
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
                self.index.add(np.array(embeddings))
                
                status_text.text("✅ Complete!")
                progress_bar.progress(100)
                
                st.success(f"✅ Success! Processed {len(self.chunks)} text chunks from {total_pages} pages")
                
                # Show sample of extracted text
                with st.expander("🔍 View extracted text sample"):
                    for i in range(min(3, len(self.chunks))):
                        st.write(f"**Chunk {i+1} (Page {self.chunk_metadata[i]['page']}):**")
                        st.text(self.chunks[i][:200] + "..." if len(self.chunks[i]) > 200 else self.chunks[i])
                
                return len(self.chunks)
            else:
                st.error("""
                ❌ No text found in PDF!
                
                **Possible reasons:**
                - PDF is scanned (image-based)
                - PDF is password protected  
                - PDF is corrupted
                - Text is in unsupported format
                
                **Try uploading a PDF with selectable text.**
                """)
                return 0
                
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            return 0
        finally:
            # Cleanup
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    def search_similar(self, query, top_k=3):
        if self.index is None or len(self.chunks) == 0:
            return []
        
        try:
            with st.spinner("🔍 Searching document..."):
                query_embedding = self.embedder.encode([query])
                distances, indices = self.index.search(np.array(query_embedding), top_k)
                
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx < len(self.chunks):
                        similarity = 1 / (1 + distances[0][i])
                        results.append({
                            "content": self.chunks[idx],
                            "metadata": self.chunk_metadata[idx],
                            "similarity": round(similarity, 3)
                        })
                
                # Sort by similarity
                results.sort(key=lambda x: x['similarity'], reverse=True)
                return results
        except Exception as e:
            st.error(f"Search error: {e}")
            return []

# =============================================================================
# SIMPLE LLM SERVICE
# =============================================================================
class LLMService:
    def generate_answer(self, question, context_chunks):
        try:
            if not context_chunks:
                return self._no_results_template(question), 0.0
            
            # Calculate confidence
            avg_confidence = sum(chunk['similarity'] for chunk in context_chunks) / len(context_chunks)
            
            # Generate answer
            answer = self._create_answer(question, context_chunks, avg_confidence)
            
            return answer, avg_confidence
            
        except Exception as e:
            return f"Error: {str(e)}", 0.0
    
    def _no_results_template(self, question):
        return f"""
**❌ No relevant information found for:** "{question}"

**💡 Try these solutions:**
1. **Use different keywords** from the document
2. **Ask about general topics** like "summary" or "main points"  
3. **Rephrase your question** to be more specific
4. **Check if PDF has selectable text** (not scanned images)

**Example questions that usually work:**
- "What is this document about?"
- "Summarize the main topics"
- "What are the key points?"
"""
    
    def _create_answer(self, question, context_chunks, confidence):
        # Extract key information
        key_points = []
        for chunk in context_chunks[:3]:  # Use top 3 chunks
            sentences = chunk['content'].split('. ')
            if sentences:
                key_points.append(sentences[0] + ".")  # First sentence
        
        answer = f"""
**🤔 Question:** {question}

**📊 Found {len(context_chunks)} relevant sections** (Confidence: {confidence:.1%})

**💡 Answer:**

Based on the document, here's what I found:

"""
        
        # Add key points
        for i, point in enumerate(key_points, 1):
            answer += f"{i}. {point}\n"
        
        # Add source references
        answer += f"""
**🔍 Sources referenced:**
"""
        for i, chunk in enumerate(context_chunks, 1):
            answer += f"- Page {chunk['metadata']['page']} (Confidence: {chunk['similarity']:.1%})\n"
        
        return answer

# =============================================================================
# SIMPLE VOICE SERVICE
# =============================================================================
class VoiceService:
    def text_to_speech_html(self, text):
        clean_text = text.replace('"', '\\"').replace("'", "\\'")[:500]  # Limit length
        html = f"""
        <script>
        function speak() {{
            if ('speechSynthesis' in window) {{
                var msg = new SpeechSynthesisUtterance();
                msg.text = "{clean_text}";
                msg.rate = 0.8;
                window.speechSynthesis.speak(msg);
            }}
        }}
        speak();
        </script>
        """
        return html

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    st.set_page_config(
        page_title="IRMC AskPro ⚡",
        page_icon="📚",
        layout="wide"
    )
    
    # Initialize services
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    if 'llm' not in st.session_state:
        st.session_state.llm = LLMService()
    if 'voice' not in st.session_state:
        st.session_state.voice = VoiceService()
    
    # Sidebar
    with st.sidebar:
        st.title("📚 IRMC AskPro")
        st.markdown("---")
        
        # File upload
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file:
            st.info(f"**File:** {uploaded_file.name}")
            
            if st.button("🔄 Process Document", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    chunk_count = st.session_state.processor.process_pdf(uploaded_file)
                    if chunk_count > 0:
                        st.session_state.pdf_processed = True
                        st.session_state.pdf_name = uploaded_file.name
                        st.rerun()
        
        if st.session_state.get('pdf_processed'):
            st.success(f"✅ **Ready:** {st.session_state.pdf_name}")
        
        st.markdown("---")
        st.markdown("**💡 Tips:**")
        st.markdown("- Upload PDFs with **selectable text**")
        st.markdown("- Ask **specific questions**")
        st.markdown("- Try **'What is this about?'** first")
    
    # Main area
    st.title("IRMC AskPro ⚡")
    st.markdown("Ask questions about your PDF documents")
    
    # Initialize chat
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    
    # Show upload reminder
    if not st.session_state.pdf_processed:
        st.info("📄 **Please upload and process a PDF document in the sidebar**")
        st.markdown("""
        **How to get started:**
        1. 📤 Upload a PDF file (left sidebar)
        2. 🔄 Click 'Process Document' 
        3. 💬 Ask questions below
        4. 🎯 Get instant answers with sources
        """)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    question = st.chat_input("Ask a question about your document...")
    
    # Process question
    if question and st.session_state.pdf_processed:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching document..."):
                start_time = time.time()
                
                # Search for relevant content
                context_chunks = st.session_state.processor.search_similar(question, top_k=3)
                
                # Generate answer
                answer, confidence = st.session_state.llm.generate_answer(question, context_chunks)
                
                response_time = time.time() - start_time
            
            # Display answer
            st.markdown(answer)
            
            # Show confidence
            if confidence > 0:
                if confidence > 0.7:
                    st.success(f"✅ High confidence: {confidence:.1%}")
                elif confidence > 0.4:
                    st.warning(f"⚠️ Medium confidence: {confidence:.1%}")
                else:
                    st.error(f"🔍 Low confidence: {confidence:.1%}")
            
            # Show response time
            st.caption(f"⏱️ Response time: {response_time:.2f}s")
            
            # Voice button
            if st.button("🔊 Speak Answer", key="speak"):
                js_code = st.session_state.voice.text_to_speech_html(answer)
                st.components.v1.html(js_code, height=0)
        
        # Add to history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
    elif question and not st.session_state.pdf_processed:
        st.error("❌ Please upload and process a PDF document first!")

if __name__ == "__main__":
    main()
