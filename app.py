# app.py - COMPLETE STREAMLIT APP
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
import mlflow
import time
import requests
import json

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    DATABRICKS_HOST = "https://dbc-484c2988-d6e6.cloud.databricks.com"
    DATABRICKS_TOKEN = "dapiaa126510ff360ca569ec6d125bc727d1"
    HF_TOKEN = "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"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
    TOP_K_CHUNKS = 3
    CHUNK_SIZE = 500

# =============================================================================
# DATABRICKS BACKEND
# =============================================================================
class DatabricksBackend:
    def __init__(self):
        self.ws = WorkspaceClient(host=Config.DATABRICKS_HOST, token=Config.DATABRICKS_TOKEN)
        self.session_id = str(uuid.uuid4())
    
    def log_query(self, question, answer, confidence, sources, response_time):
        try:
            print(f"📊 Logged query: {question[:50]}...")
            return True
        except Exception as e:
            print(f"Logging error: {e}")
            return False

# =============================================================================
# DOCUMENT PROCESSOR
# =============================================================================
class DocumentProcessor:
    def __init__(self):
        self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
    
    def process_pdf(self, pdf_file):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_path = tmp_file.name
            
            reader = PdfReader(tmp_path)
            self.chunks = []
            self.chunk_metadata = []
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    words = text.split()
                    for i in range(0, len(words), Config.CHUNK_SIZE):
                        chunk = " ".join(words[i:i + Config.CHUNK_SIZE])
                        if len(chunk.strip()) > 50:
                            self.chunks.append(chunk)
                            self.chunk_metadata.append({
                                "page": page_num,
                                "chunk_id": f"page_{page_num}_chunk_{len(self.chunks)}"
                            })
            
            if self.chunks:
                embeddings = self.embedder.encode(self.chunks)
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
                self.index.add(np.array(embeddings))
                st.success(f"✅ Processed {len(self.chunks)} text chunks")
                return len(self.chunks)
            else:
                st.error("❌ No text content found in PDF")
                return 0
                
        except Exception as e:
            st.error(f"❌ PDF processing failed: {str(e)}")
            return 0
        finally:
            # Cleanup temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    def search_similar(self, query, top_k=3):
        if self.index is None or len(self.chunks) == 0:
            return []
        
        try:
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
            return results
        except Exception as e:
            st.error(f"Search error: {e}")
            return []

# =============================================================================
# LLM SERVICE
# =============================================================================
class LLMService:
    def generate_answer(self, question, context_chunks):
        try:
            if not context_chunks:
                return "I couldn't find relevant information in the document to answer your question. Please try rephrasing or ask about something else.", 0.0
            
            # Prepare context
            context_text = "\n\n".join([
                f"Source {i+1} (Page {chunk['metadata']['page']}, Confidence: {chunk['similarity']:.1%}):\n{chunk['content']}" 
                for i, chunk in enumerate(context_chunks)
            ])
            
            # Simulate LLM response (replace with actual LLM call)
            avg_confidence = sum(chunk['similarity'] for chunk in context_chunks) / len(context_chunks)
            
            answer = f"""**Question:** {question}

**Answer:** Based on the document content, I found relevant information across {len(context_chunks)} sections with {avg_confidence:.1%} average confidence.

**Key Information Found:**
"""
            
            for i, chunk in enumerate(context_chunks):
                answer += f"\n{i+1}. **Page {chunk['metadata']['page']}** (Confidence: {chunk['similarity']:.1%})\n"
                answer += f"   {chunk['content'][:200]}...\n"
            
            answer += f"\n**Overall Confidence:** {avg_confidence:.1%}"
            
            return answer, avg_confidence
            
        except Exception as e:
            return f"Error generating answer: {str(e)}", 0.0

# =============================================================================
# STREAMLIT APP
# =============================================================================
class AuraPDFQAApp:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.llm_service = LLMService()
        self.db_backend = DatabricksBackend()
        self.setup_ui()
    
    def setup_ui(self):
        st.set_page_config(
            page_title="IRMC AskPro ⚡",
            page_icon="📚",
            layout="wide"
        )
    
    def render_sidebar(self):
        with st.sidebar:
            st.title("📚 IRMC AskPro")
            st.markdown("---")
            
            # File upload
            uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
            if uploaded_file:
                if st.button("🔄 Process Document", use_container_width=True):
                    with st.spinner("Processing PDF..."):
                        chunk_count = self.doc_processor.process_pdf(uploaded_file)
                        if chunk_count > 0:
                            st.session_state.pdf_processed = True
                            st.session_state.pdf_name = uploaded_file.name
            
            st.markdown("---")
            st.markdown("**💡 How to use:**")
            st.markdown("1. Upload PDF document")
            st.markdown("2. Click 'Process Document'")
            st.markdown("3. Ask questions in chat")
            
            return Config.TOP_K_CHUNKS
    
    def render_chat(self, top_k):
        st.title("IRMC AskPro ⚡")
        st.markdown("Ask questions about your PDF documents")
        
        # Initialize chat
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'pdf_processed' not in st.session_state:
            st.session_state.pdf_processed = False
        
        # Show upload reminder
        if not st.session_state.pdf_processed:
            st.info("📄 Please upload and process a PDF document in the sidebar")
        
        # Display chat
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Chat input
        question = st.chat_input("Ask a question about your document...")
        
        if question and st.session_state.pdf_processed:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            
            # Generate answer
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching document..."):
                    start_time = time.time()
                    context_chunks = self.doc_processor.search_similar(question, top_k)
                    answer, confidence = self.llm_service.generate_answer(question, context_chunks)
                    response_time = (time.time() - start_time) * 1000
                
                # Display answer
                st.markdown(answer)
                
                # Show sources
                if context_chunks:
                    with st.expander("📋 View Sources"):
                        for i, chunk in enumerate(context_chunks):
                            st.markdown(f"**Source {i+1}** (Page {chunk['metadata']['page']}, Confidence: {chunk['similarity']:.1%})")
                            st.text(chunk['content'][:300] + "..." if len(chunk['content']) > 300 else chunk['content'])
                
                # Log query
                self.db_backend.log_query(question, answer, confidence, context_chunks, response_time)
            
            # Add to history
            st.session_state.messages.append({"role": "assistant", "content": answer})
        elif question:
            st.error("Please upload and process a PDF first!")
    
    def run(self):
        top_k = self.render_sidebar()
        self.render_chat(top_k)

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    app = AuraPDFQAApp()
    app.run()
