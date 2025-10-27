# app.py - IRMC AskPro ⚡ WITH LLM ANALYSIS & BETTER CHUNKING
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
import re
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
    
    # LLM API Configuration (Using Hugging Face Inference API)
    HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    LLM_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# =============================================================================
# IMPROVED DOCUMENT PROCESSOR WITH BETTER CHUNKING
# =============================================================================
class DocumentProcessor:
    def __init__(self):
        try:
            st.info("🔄 Loading AI models...")
            self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
            self.index = None
            self.chunks = []
            self.chunk_metadata = []
            st.success("✅ Document processor ready!")
        except Exception as e:
            st.error(f"❌ Model loading failed: {e}")
    
    def smart_chunking(self, text, page_num):
        """Intelligent chunking that preserves context"""
        chunks = []
        
        # First, split by paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        for para in paragraphs:
            if len(para) <= 300:
                # Short paragraph - keep as is
                chunks.append(para)
                self.chunk_metadata.append({
                    "page": page_num,
                    "type": "paragraph",
                    "length": len(para)
                })
            else:
                # Long paragraph - split by sentences but preserve context
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence) < 400:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            self.chunk_metadata.append({
                                "page": page_num,
                                "type": "logical_chunk",
                                "length": len(current_chunk)
                            })
                        current_chunk = sentence + " "
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    self.chunk_metadata.append({
                        "page": page_num,
                        "type": "logical_chunk", 
                        "length": len(current_chunk)
                    })
        
        return chunks
    
    def process_pdf(self, pdf_file):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_path = tmp_file.name
            
            status_text.text("📖 Reading PDF...")
            reader = PdfReader(tmp_path)
            total_pages = len(reader.pages)
            
            self.chunks = []
            self.chunk_metadata = []
            
            for page_num, page in enumerate(reader.pages, 1):
                status_text.text(f"📄 Processing page {page_num}/{total_pages}...")
                progress_bar.progress(page_num / total_pages)
                
                text = page.extract_text() or ""
                
                if text.strip():
                    # Clean text
                    clean_text = re.sub(r'\s+', ' ', text).strip()
                    
                    # Use smart chunking
                    page_chunks = self.smart_chunking(clean_text, page_num)
                    self.chunks.extend(page_chunks)
            
            # Create embeddings
            if self.chunks:
                status_text.text("🧠 Creating search index...")
                embeddings = self.embedder.encode(self.chunks)
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
                self.index.add(np.array(embeddings))
                
                status_text.text("✅ Complete!")
                progress_bar.progress(100)
                
                st.success(f"✅ Success! Processed {len(self.chunks)} intelligent chunks from {total_pages} pages")
                return len(self.chunks)
            else:
                st.error("❌ No text found in PDF!")
                return 0
                
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            return 0
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    def search_similar(self, query, top_k=5):
        if self.index is None or len(self.chunks) == 0:
            return []
        
        try:
            with st.spinner("🔍 Finding relevant content..."):
                query_embedding = self.embedder.encode([query])
                distances, indices = self.index.search(np.array(query_embedding), top_k)
                
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx < len(self.chunks):
                        similarity = 1 / (1 + distances[0][i])
                        if similarity > 0.1:  # Filter out very low similarity
                            results.append({
                                "content": self.chunks[idx],
                                "metadata": self.chunk_metadata[idx],
                                "similarity": round(similarity, 3)
                            })
                
                results.sort(key=lambda x: x['similarity'], reverse=True)
                return results
        except Exception as e:
            st.error(f"Search error: {e}")
            return []

# =============================================================================
# ADVANCED LLM SERVICE WITH PROPER ANALYSIS
# =============================================================================
class LLMService:
    def __init__(self):
        self.api_url = Config.HF_API_URL
        self.headers = Config.LLM_HEADERS
    
    def call_llm_api(self, prompt, max_length=500):
        """Call Hugging Face Inference API"""
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_length,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0]['generated_text']
                else:
                    return "I couldn't generate a proper response. Please try again."
            else:
                return f"API Error: {response.status_code}"
                
        except Exception as e:
            return f"LLM Error: {str(e)}"
    
    def generate_answer(self, question, context_chunks):
        try:
            if not context_chunks:
                return self._no_results_template(question), 0.0
            
            avg_confidence = sum(chunk['similarity'] for chunk in context_chunks) / len(context_chunks)
            
            # Use LLM to analyze and generate intelligent answer
            answer = self._analyze_with_llm(question, context_chunks, avg_confidence)
            
            return answer, avg_confidence
            
        except Exception as e:
            return f"Error: {str(e)}", 0.0
    
    def _no_results_template(self, question):
        return f"""
**❌ No relevant information found for:** "{question}"

**💡 Suggestions:**
- Try different keywords or rephrase your question
- Ask about general topics in the document
- Check if the PDF contains relevant information
"""
    
    def _analyze_with_llm(self, question, context_chunks, confidence):
        """Use LLM to analyze context and generate intelligent answer"""
        
        # Prepare context for LLM
        context_text = "\n\n".join([
            f"[Source {i+1}, Page {chunk['metadata']['page']}, Confidence: {chunk['similarity']:.1%}]: {chunk['content']}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Create intelligent prompt
        prompt = f"""Based EXACTLY on the following document excerpts, answer the question clearly and accurately.

DOCUMENT CONTEXT:
{context_text}

QUESTION: {question}

INSTRUCTIONS:
1. Answer using ONLY information from the provided context
2. If the exact answer isn't in the context, say "The document doesn't contain specific information about this"
3. Be precise and cite sources when possible
4. If numbers or statistics are asked for, provide the exact figures from the context
5. Keep the answer focused and relevant to the question

ANSWER:"""
        
        # Get LLM response
        llm_response = self.call_llm_api(prompt)
        
        # Format the final answer
        answer = f"""
**🤔 Your Question:** {question}

**📊 Analysis Results:** Found {len(context_chunks)} relevant sections with {confidence:.1%} confidence

**💡 Intelligent Answer:**

{llm_response}

**🔍 Source Analysis:** Based on information from pages {', '.join(set(str(chunk['metadata']['page']) for chunk in context_chunks))}
"""
        
        return answer
    
    def improve_chunking(self, large_text):
        """Use LLM to improve chunking for complex documents"""
        prompt = f"""Split the following text into logical, self-contained chunks that preserve context. Each chunk should be 200-400 words and make sense on its own.

TEXT:
{large_text}

CHUNKS:"""
        
        improved_chunks = self.call_llm_api(prompt, max_length=800)
        return improved_chunks

# =============================================================================
# VOICE SERVICE
# =============================================================================
class VoiceService:
    def text_to_speech_html(self, text):
        clean_text = text.replace('"', '\\"').replace("'", "\\'")[:500]
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
                with st.spinner("Processing with AI..."):
                    chunk_count = st.session_state.processor.process_pdf(uploaded_file)
                    if chunk_count > 0:
                        st.session_state.pdf_processed = True
                        st.session_state.pdf_name = uploaded_file.name
                        st.rerun()
        
        if st.session_state.get('pdf_processed'):
            st.success(f"✅ **Ready:** {st.session_state.pdf_name}")
        
        st.markdown("---")
        st.markdown("**🚀 Enhanced Features:**")
        st.markdown("- 🤖 LLM-Powered Analysis")
        st.markdown("- 🧠 Intelligent Chunking") 
        st.markdown("- 📊 Context-Aware Answers")
        st.markdown("- 🎯 Precise Information")
    
    # Main area
    st.title("IRMC AskPro ⚡")
    st.markdown("**AI-Powered Document Analysis with LLM Intelligence**")
    
    # Initialize chat
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    
    # Show upload reminder
    if not st.session_state.pdf_processed:
        st.info("📄 **Upload a PDF to start intelligent document analysis**")
        st.markdown("""
        **How it works:**
        1. 📤 Upload any PDF document
        2. 🔄 AI processes with intelligent chunking  
        3. 💬 Ask complex, specific questions
        4. 🧠 LLM analyzes context and provides precise answers
        
        **Ask about:**
        - Specific numbers and statistics
        - Complex concepts and relationships
        - Detailed technical information
        - Financial data and projections
        """)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    question = st.chat_input("Ask complex questions about your document...")
    
    # Process question
    if question and st.session_state.pdf_processed:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("🧠 Analyzing document with AI..."):
                start_time = time.time()
                
                # Search for relevant content
                context_chunks = st.session_state.processor.search_similar(question, top_k=5)
                
                # Generate intelligent answer using LLM
                answer, confidence = st.session_state.llm.generate_answer(question, context_chunks)
                
                response_time = time.time() - start_time
            
            # Display answer
            st.markdown(answer)
            
            # Show detailed sources
            if context_chunks:
                with st.expander("📋 View Source Context"):
                    for i, chunk in enumerate(context_chunks):
                        st.write(f"**Source {i+1}** (Page {chunk['metadata']['page']}, Confidence: {chunk['similarity']:.1%})")
                        st.text(chunk['content'][:400] + "..." if len(chunk['content']) > 400 else chunk['content'])
                        st.markdown("---")
            
            # Response time and confidence
            st.caption(f"⏱️ AI Analysis Time: {response_time:.2f}s | Confidence: {confidence:.1%}")
            
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
