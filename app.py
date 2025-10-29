# app.py - Aura PDF QA ⚡ (Browser voice recording + upload fallback)
import streamlit as st
import tempfile, os, time, random, base64, wave
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, pdf2image, pytesseract
from gtts import gTTS

# --------------------
# Config
# --------------------
class Config:
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    CHUNK_SIZE = 1000
    MIN_PARAGRAPH_LENGTH = 50
    TOP_K = 3

# --------------------
# LLM fallback (keeps behavior minimal & deterministic)
# --------------------
class LLMService:
    def generate(self, question, chunks):
        if not chunks:
            return f"No relevant info found for '{question}'.", 0.0
        avg_conf = sum(c['similarity'] for c in chunks) / len(chunks)
        ans = "**Answer (summary):**\n\n"
        for c in chunks[:3]:
            snippet = c['content'][:250].replace("\n", " ")
            ans += f"• Page {c['metadata']['page']}: {snippet}...\n\n"
        ans += "\n**Sources:**\n" + "\n".join(
            [f"Page {c['metadata']['page']} | Confidence: {c['similarity']:.1%}" for c in chunks]
        )
        return ans, avg_conf

# --------------------
# Voice service: handles TTS and transcription (via uploaded audio or browser stream)
# --------------------
class VoiceService:
    def __init__(self):
        # Note: we try to import speech_recognition; if unavailable we will still allow upload fallback
        try:
            import speech_recognition as sr  # type: ignore
            self.sr = sr
            self.has_speech_recognition = True
        except Exception:
            self.sr = None
            self.has_speech_recognition = False

        # try to import streamlit-webrtc for browser-based recording
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode  # type: ignore
            self.webrtc_available = True
            self.webrtc = webrtc_streamer
            self.WebRtcMode = WebRtcMode
        except Exception:
            self.webrtc_available = False
            self.webrtc = None
            self.WebRtcMode = None

    def transcribe_from_wav_bytes(self, wav_bytes):
        """Transcribe a WAV bytes object using speech_recognition (if available)."""
        if not self.has_speech_recognition:
            st.error("SpeechRecognition not installed on backend. Upload a short audio file and enable transcript on local environment.")
            return None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(wav_bytes)
                tmp_path = tmp.name

            r = self.sr.Recognizer()
            with self.sr.AudioFile(tmp_path) as src:
                audio = r.record(src)
                text = r.recognize_google(audio)
            os.unlink(tmp_path)
            return text
        except Exception as e:
            st.error(f"Transcription error: {e}")
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            return None

    def text_to_speech(self, text, autoplay=True):
        """Convert text (string) to speech and play via Streamlit audio player."""
        try:
            clean = " ".join(text.split())  # minimal cleaning
            if len(clean) > 1200:  # limit TTS length
                clean = clean[:1200] + "..."
            tts = gTTS(text=clean, lang="en", slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tts.save(tmp.name)
                audio_path = tmp.name
            # stream audio file bytes to Streamlit
            audio_bytes = open(audio_path, "rb").read()
            st.audio(audio_bytes, format="audio/mp3", start_time=0)
            os.unlink(audio_path)
        except Exception as e:
            st.error(f"TTS error: {e}")

    # helper to convert frames from streamlit-webrtc -> WAV bytes (if used)
    def save_webrtc_audio_to_wav(self, audio_frames):
        """
        audio_frames: list of frames from streamlit-webrtc (each frame is a numpy array)
        We'll write a 16-bit PCM WAV file.
        """
        # audio_frames expected shape: list of (ndarray, sample_rate)
        try:
            if not audio_frames:
                return None
            # first frame sampling info
            frames = audio_frames
            raw = b"".join([f.tobytes() for f in frames])
            # We'll write a WAV file by guessing sample width and channels
            # NOTE: This helper expects frames as numpy.int16 arrays (default from webrtc)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                wf = wave.open(tmp.name, 'wb')
                # guess channels = 1, sampwidth = 2 (16-bit), framerate = 48000
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(raw)
                wf.close()
                wav_bytes = open(tmp.name, "rb").read()
            os.unlink(tmp.name)
            return wav_bytes
        except Exception:
            return None


# --------------------
# Document Processor: same architecture (text/ocr + chunking + embeddings + FAISS)
# --------------------
class DocumentProcessor:
    def __init__(self):
        try:
            self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
        except Exception as e:
            st.error(f"Embedder init error: {e}")
            raise
        self.chunks = []
        self.chunk_metadata = []
        self.index = None

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
        progress = st.progress(0)
        status = st.empty()
        for i, image in enumerate(images):
            status.text(f"OCR: page {i+1}/{len(images)}")
            text = pytesseract.image_to_string(image)
            if text.strip():
                extracted.append({"page": i + 1, "text": text.strip()})
            progress.progress((i + 1) / len(images))
        status.text("OCR done")
        return extracted

    def analyze_pdf_type(self, pdf_path):
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        text_pages = sum(1 for page in reader.pages if len((page.extract_text() or "").strip()) > 50)
        # if no pages, treat as scanned
        if total_pages == 0:
            return "scanned"
        return "text_based" if (text_pages / total_pages) > 0.5 else "scanned"

    def process_pdf(self, uploaded_file):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(uploaded_file.getvalue())
        tmp.close()
        pdf_path = tmp.name

        try:
            st.info("Analyzing PDF...")
            pdf_type = self.analyze_pdf_type(pdf_path)
            st.info(f"Detected PDF type: {pdf_type}")

            if pdf_type == "text_based":
                extracted = self.extract_text_direct(pdf_path)
                method = "text"
            else:
                extracted = self.extract_text_ocr(pdf_path)
                method = "ocr"

            # chunking
            self.chunks = []
            self.chunk_metadata = []
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
                        self.chunks.append(para)
                        self.chunk_metadata.append({"page": page, "method": method})

            if not self.chunks:
                st.error("No text extracted.")
                return 0

            st.info("Creating embeddings...")
            embeddings = self.embedder.encode(self.chunks)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(np.array(embeddings))
            st.success(f"Processed: {len(self.chunks)} chunks")
            return len(self.chunks)
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def search_similar(self, query, top_k=3):
        if not self.chunks or self.index is None:
            return []
        qvec = self.embedder.encode([query])
        D, I = self.index.search(np.array(qvec), min(top_k, len(self.chunks)))
        results = []
        for i, idx in enumerate(I[0]):
            if idx < len(self.chunks):
                sim = 1 / (1 + D[0][i])
                results.append({"content": self.chunks[idx], "metadata": self.chunk_metadata[idx], "similarity": round(sim, 3)})
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

# --------------------
# Streamlit App (main)
# --------------------
class AuraPDFQAApp:
    def __init__(self):
        st.set_page_config(page_title="Aura PDF QA ⚡", layout="wide", page_icon="🎯")
        self.doc_processor = DocumentProcessor()
        self.llm = LLMService()
        self.voice = VoiceService()
        self.setup_state()

    def setup_state(self):
        if 'pdf_processed' not in st.session_state:
            st.session_state.pdf_processed = False
        if 'messages' not in st.session_state:
            st.session_state.messages = []

    def render_sidebar(self):
        st.sidebar.title("Aura PDF QA ⚡")
        uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
        if uploaded_file and not st.session_state.pdf_processed:
            if st.sidebar.button("Process Document"):
                count = self.doc_processor.process_pdf(uploaded_file)
                if count > 0:
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_name = uploaded_file.name
                    st.experimental_rerun()
        st.sidebar.markdown("---")
        st.sidebar.markdown("Input mode:")
        mode = st.sidebar.radio("Input mode", ["Text", "Voice"], index=0, label_visibility="collapsed")
        st.sidebar.checkbox("Auto voice responses", value=True, key="auto_voice")
        return mode

    def browser_record_component(self):
        """
        Try to use streamlit-webrtc for browser recording.
        If not available, return None and caller should show a file_uploader fallback.
        """
        if not self.voice.webrtc_available:
            return None

        from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings  # type: ignore
        import av  # type: ignore

        ctx = webrtc_streamer(
            key="webrtc-audio",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={},
            media_stream_constraints={"audio": True, "video": False},
            async_processing=False,
            # this app only needs audio frames; default settings usually suffice
        )
        return ctx

    def run(self):
        mode = self.render_sidebar()
        st.title("Aura PDF QA ⚡")
        st.markdown("Ask questions about your document — use Text or Voice input.")

        # Show chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if not st.session_state.pdf_processed:
            st.info("Upload and process a PDF to start.")
            return

        question = None

        if mode == "Text":
            question = st.chat_input("Type your question...")
        else:
            # Voice mode UI:
            st.markdown("**Voice input** — record in the browser or upload an audio file.")
            ctx = None
            if self.voice.webrtc_available:
                st.caption("Click the 'Start' button in the component to enable microphone (if prompted by browser).")
                ctx = self.browser_record_component()
            else:
                st.info("Browser recording not available in this environment. Use audio upload below.")

            col1, col2 = st.columns([1, 1])
            with col1:
                if ctx and ctx.state.playing:
                    st.write("Recording active. Click 'Stop' in the component when done.")
                    # webrtc_streamer stores audio frames in ctx.audio_receiver if any
                    if st.button("Transcribe recorded audio"):
                        # pull available frames (best-effort)
                        audio_frames = []
                        try:
                            audio_receiver = ctx.audio_receiver
                            if audio_receiver:
                                # drain available frames
                                frames = []
                                while True:
                                    frame = audio_receiver.get_frame(timeout=0.1)
                                    if frame is None:
                                        break
                                    frames.append(frame.to_ndarray())
                                audio_frames = frames
                        except Exception:
                            audio_frames = []
                        wav_bytes = self.voice.save_webrtc_audio_to_wav(audio_frames) if audio_frames else None
                        if wav_bytes:
                            text = self.voice.transcribe_from_wav_bytes(wav_bytes)
                            if text:
                                question = text
                else:
                    st.write("Record using the component above, then click 'Transcribe recorded audio'.")

            with col2:
                st.markdown("**Or upload an audio file (wav/mp3)**")
                audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])
                if audio_file is not None:
                    # convert to wav bytes if mp3/m4a via pydub fallback if available
                    file_bytes = audio_file.read()
                    # If it's mp3/m4a, speech_recognition can usually handle mp3 if ffmpeg is available.
                    # We'll attempt to transcribe directly
                    if self.voice.has_speech_recognition:
                        # If mp3/m4a, write to temp and let recognizer handle it via AudioFile
                        try:
                            text = self.voice.transcribe_from_wav_bytes(file_bytes)
                            if text:
                                question = text
                        except Exception:
                            # Try saving bytes to file and using AudioFile
                            try:
                                tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                                tmpf.write(file_bytes)
                                tmpf.close()
                                r = self.voice.sr.Recognizer()
                                with self.voice.sr.AudioFile(tmpf.name) as src:
                                    audio = r.record(src)
                                    text = r.recognize_google(audio)
                                    question = text
                                os.unlink(tmpf.name)
                            except Exception as e:
                                st.error(f"Upload transcription failed: {e}")
                    else:
                        st.warning("Backend transcription library not available. Install `speechrecognition` to transcribe uploads on server.")

        # If question obtained:
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("assistant"):
                with st.spinner("Finding answers..."):
                    start = time.time()
                    chunks = self.doc_processor.search_similar(question, Config.TOP_K)
                    answer, conf = self.llm.generate(question, chunks)
                    elapsed = (time.time() - start) * 1000
                    st.markdown(answer)
                    st.caption(f"⏱ {elapsed:.0f} ms | Confidence: {conf:.1%}")

                    # Auto voice reply if enabled
                    if st.session_state.get("auto_voice", True):
                        self.voice.text_to_speech(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    app = AuraPDFQAApp()
    app.run()
