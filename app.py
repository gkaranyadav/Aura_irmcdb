# app.py - IRMC AskPro ⚡ GENERAL PURPOSE FOR ANY DOCUMENT
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
# DOCUMENT PROCESSOR (SAME - IT'S GOOD)
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
                    clean_text = re.sub(r'\s+', ' ', text).strip()
                    
                    sentences = re.split(r'[.!?]+', clean_text)
                    current_chunk = ""
                    
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                            
                        if len(current_chunk + sentence) < Config.CHUNK_SIZE:
                            current_chunk += sentence + ". "
                        else:
                            if current_chunk:
                                self.chunks.append(current_chunk.strip())
                                self.chunk_metadata.append({
                                    "page": page_num,
                                    "chunk_id": f"page_{page_num}_chunk_{len(self.chunks)}"
                                })
                            current_chunk = sentence + ". "
                    
                    if current_chunk:
                        self.chunks.append(current_chunk.strip())
                        self.chunk_metadata.append({
                            "page": page_num,
                            "chunk_id": f"page_{page_num}_chunk_{len(self.chunks)}"
                        })
            
            if self.chunks:
                status_text.text("🧠 Creating search index...")
                embeddings = self.embedder.encode(self.chunks)
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
                self.index.add(np.array(embeddings))
                
                status_text.text("✅ Complete!")
                progress_bar.progress(100)
                
                st.success(f"✅ Success! Processed {len(self.chunks)} text chunks from {total_pages} pages")
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
                        if similarity > 0.1:
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
# SIMPLE & SMART LLM SERVICE - WORKS WITH ANY DOCUMENT
# =============================================================================
class LLMService:
    def generate_answer(self, question, context_chunks):
        try:
            if not context_chunks:
                return self._no_results_template(question), 0.0
            
            avg_confidence = sum(chunk['similarity'] for chunk in context_chunks) / len(context_chunks)
            
            # SIMPLE LOGIC: Just extract the most relevant information
            answer = self._extract_relevant_info(question, context_chunks, avg_confidence)
            
            return answer, avg_confidence
            
        except Exception as e:
            return f"Error: {str(e)}", 0.0
    
    def _no_results_template(self, question):
        return f"""
**❌ No relevant information found for:** "{question}"

**💡 Suggestions:**
- Try different keywords from the document
- Ask more general questions
- Check if the PDF contains relevant information
"""
    
    def _extract_relevant_info(self, question, context_chunks, confidence):
        """Extract and present the most relevant information from chunks"""
        
        # Combine and clean the content
        all_content = " ".join([chunk['content'] for chunk in context_chunks])
        
        # Remove duplicate sentences
        sentences = list(set([s.strip() for s in all_content.split('. ') if len(s.strip()) > 10]))
        
        # Sort sentences by relevance (simple heuristic - longer sentences often have more info)
        sentences.sort(key=len, reverse=True)
        
        answer = f"""
**🤔 Your Question:** {question}

**📊 Search Results:** Found {len(context_chunks)} relevant sections with {confidence:.1%} confidence

**💡 Relevant Information from Document:**

"""
        
        # Add the most informative sentences
        for i, sentence in enumerate(sentences[:5], 1):  # Show top 5 unique sentences
            if len(sentence) > 15:  # Only substantial sentences
                answer += f"• {sentence}.\n"
        
        # Add source information
        unique_pages = set(str(chunk['metadata']['page']) for chunk in context_chunks)
        answer += f"\n**🔍 Source Pages:** {', '.join(unique_pages)}"
        
        # Add confidence indicator
        if confidence > 0.7:
            answer += f"\n\n✅ **High Confidence** - This information is very relevant to your question"
        elif confidence > 0.4:
            answer += f"\n\n⚠️ **Medium Confidence** - This information is somewhat relevant"
        else:
            answer += f"\n\n🔍 **Low Confidence** - Consider rephrasing your question"
        
        return answer

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
                with st.spinner("Processing..."):
                    chunk_count = st.session_state.processor.process_pdf(uploaded_file)
                    if chunk_count > 0:
                        st.session_state.pdf_processed = True
                        st.session_state.pdf_name = uploaded_file.name
                        st.rerun()
        
        if st.session_state.get('pdf_processed'):
            st.success(f"✅ **Ready:** {st.session_state.pdf_name}")
        
        st.markdown("---")
        st.markdown("**💡 Works with:**")
        st.markdown("- Resumes/CVs")
        st.markdown("- Research papers")
        st.markdown("- Reports & documents")
        st.markdown("- Manuals & guides")
        st.markdown("- Any text-based PDF")
    
    # Main area
    st.title("IRMC AskPro ⚡")
    st.markdown("Ask questions about **ANY** PDF document")
    
    # Initialize chat
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    
    # Show upload reminder
    if not st.session_state.pdf_processed:
        st.info("📄 **Please upload and process a PDF document in the sidebar**")
        st.markdown("""
        **How it works:**
        1. 📤 Upload any PDF file
        2. 🔄 Click 'Process Document' 
        3. 💬 Ask questions in natural language
        4. 🎯 Get instant answers with sources
        
        **Example questions:**
        - "What is this document about?"
        - "Tell me about [specific topic]"
        - "Find information about [keyword]"
        - "Summarize the main points"
        """)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    question = st.chat_input("Ask any question about your document...")
    
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
            
            # Show detailed sources if available
            if context_chunks:
                with st.expander("📋 View Source Details"):
                    for i, chunk in enumerate(context_chunks):
                        st.write(f"**Source {i+1}** (Page {chunk['metadata']['page']}, Confidence: {chunk['similarity']:.1%})")
                        st.text(chunk['content'][:300] + "..." if len(chunk['content']) > 300 else chunk['content'])
                        st.markdown("---")
            
            # Response time
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
