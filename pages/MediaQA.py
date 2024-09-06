import streamlit as st
import sys
import os
from io import BytesIO
from groq import Groq
from audiorecorder import audiorecorder
from streamlit_extras.stylable_container import stylable_container
from dotenv import load_dotenv
import subprocess
import tempfile
from pydub import AudioSegment

# Add the app1 directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
MediaQA_dir = os.path.join(current_dir, 'MediaQA')
sys.path.append(current_dir)
from MediaQA import config
from MediaQA import utils
from MediaQA import styles
from MediaQA.styles import button_css, selectbox_css, file_uploader_css, header_container_css, transcript_container
from MediaQA.utils import read_from_url, prerecorded, chat_stream, create_vectorstore, read_from_youtube
from MediaQA.config import GROQ_CLIENT, VECTOR_INDEX

avatar_path = os.path.join(current_dir, 'MediaQA', 'static', 'ai_avatar.png')
groq_api_key = st.secrets["GROQ_API_KEY"]

st.title("Media QA")
st.caption("Audio transcription, summarization, & QA.")



VECTOR_INDEX = VECTOR_INDEX
groqClient = Groq(api_key=groq_api_key)

st.markdown("<a href='https://wow.groq.com/groq-labs/'><img src='app/static/logo.png' width='200'></a>", unsafe_allow_html=True)
st.write("---")
header_container = stylable_container(
    key="header",
    css_styles=header_container_css
)

ASR_MODELS = {"Whisper V3 large": "whisper-large-v3", "Whisper V3 large simplified": 'distil-whisper-large-v3-en'}
GROModelOptions = ["llama-3.1-70b-versatile", "llGuidId-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"]
GROQ_MODELS = {model: model for model in GROModelOptions}
LANGUAGES = {
    "Automatic Language Detection": None,
}

def get_audio_duration(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        audio = AudioSegment.from_file(tmp_file_path)
        duration_seconds = len(audio) / 1000
        minutes, seconds = divmod(int(duration_seconds), 60)
        return f"{minutes}:{seconds:02d}"
    finally:
        os.unlink(tmp_file_path)

# Dropdowns with styling
dropdown_container = stylable_container(
    key="dropdown_container",
    css_styles=selectbox_css
)

# Columns for horizontal layout
col1, col2, col3 = st.columns(3)

with col1:
    language = st.selectbox(
        "Language",
        options=list(LANGUAGES.keys()),
    )
    lang_options = {
        "detect_language" if language == "Automatic Language Detection" else "language": True if language == "Automatic Language Detection" else LANGUAGES[language]
    }

with col2:
    asr_model = st.selectbox("Speech Recognition", options=list(ASR_MODELS.keys()))

with col3:
    groq_model = st.selectbox("Language Models", options=list(GROQ_MODELS.keys()))


audio_source = st.radio(
    "Choose audio source",
    options=["Record audio", "Upload media file", "Load media from URL"],
    horizontal=True,
)

# Then, use it in the conditional statements
if audio_source == "Upload media file":
    file_uploader = stylable_container(
    key="file_uploader",
    css_styles=file_uploader_css
    )
    audio_file = file_uploader.file_uploader(
        label="Upload media file",
        type=["mp3", "wav", "webm"],
        label_visibility="collapsed",
    )
    print(f"Audio uploaded: {audio_file}")
    if audio_file:
        st.session_state['result'] = None
        original_file_size = audio_file.size / (1024 * 1024)  # Convert to MB
        original_mimetype = audio_file.type  # Store the original mimetype
        audio_file_contents = audio_file.getvalue()
        # Preprocess the uploaded audio
        result = subprocess.run([
            'ffmpeg',
            '-i', 'pipe:0',
            '-ar', '16000',
            '-ac', '1',
            '-map', '0:a',
            '-f', 'mp3',
            '-'
        ], input=audio_file_contents, capture_output=True, check=True)
        processed_audio = BytesIO(result.stdout)
        processed_audio.name = "processed_audio.mp3"  # Set a new file name
        processed_file_size = processed_audio.getbuffer().nbytes / (1024 * 1024)  # Convert to MB
        duration = get_audio_duration(processed_audio)
        st.write(f"File: {processed_audio.name} ({original_file_size:.2f} MB --> {processed_file_size:.2f} MB, Duration: {duration})")

        st.session_state['audio'] = processed_audio
        st.session_state['mimetype'] = "audio/mp3"  # Set mimetype to mp3 since we converted it
        st.audio(st.session_state['audio'])  # Show audio player for uploaded file
    else:
        st.session_state['audio'] = None
        st.session_state['mimetype'] = None

elif audio_source == "Load media from URL":
    url = st.text_input(
        "URL",
        key="url",
        value="https://static.deepgram.com/examples/interview_speech-analytics.wav",
    )

    if url != "":
        st.session_state["audio"] = None
        try:
            if "youtube.com" in url or "youtu.be" in url:
                print("Reading audio from YouTube")
                with st.spinner("Loading Youtube video..."):
                    st.session_state['result'] = None
                    st.video(url)
                    original_audio, original_mimetype = read_from_youtube(url)
                    original_filename = "youtube_audio.mp3"  # Default name for YouTube audio
            else:
                print("Reading audio from URL")
                with st.spinner("Loading audio URL..."):
                    st.session_state['result'] = None
                    original_audio = read_from_url(url)
                    original_mimetype = "audio/wav"  # Assuming WAV format for non-YouTube URLs
                    original_filename = url.split("/")[-1]  # Extract filename from URL

            # Get original file size
            original_file_size = len(original_audio.getvalue()) / (1024 * 1024)  # Convert to MB

            # Preprocess the audio
            with st.spinner("Processing audio..."):
                result = subprocess.run([
                    'ffmpeg',
                    '-i', 'pipe:0',
                    '-ar', '16000',
                    '-ac', '1',
                    '-map', '0:a',
                    '-f', 'mp3',
                    '-'
                ], input=original_audio.getvalue(), capture_output=True, check=True)

                processed_audio = BytesIO(result.stdout)
                processed_audio.name = original_filename  # Use original filename

            # Get processed file size
            processed_file_size = len(processed_audio.getvalue()) / (1024 * 1024)  # Convert to MB

            # Get duration
            duration = get_audio_duration(processed_audio)

            # Display file information
            st.write(f"File: {original_filename} ({original_file_size:.2f} MB --> {processed_file_size:.2f} MB, Duration: {duration})")

            # Store processed audio in session state
            st.session_state["audio"] = processed_audio
            st.session_state['mimetype'] = "audio/mp3"

            # Display audio player
            st.audio(st.session_state["audio"])

            print(f"Audio bytes: {len(st.session_state['audio'].getvalue())} bytes")

        except Exception as e:
            st.error(str(e))
            st.error("Invalid URL entered.")

elif audio_source == "Record audio":
    audio = audiorecorder("Click to record", "Click to stop recording", show_visualizer=True, key="audio-recorder")
    if len(audio) != 0:
        print(f"Audio recorded: {audio}, length {len(audio)}")
        st.session_state["result"] = None
        with st.spinner("Processing audio..."):
            audio_bytes = BytesIO()
            audio.export(audio_bytes, format="wav")
            st.session_state["audio"] = audio_bytes
            st.audio(audio_bytes)
            st.session_state['mimetype'] = "audio/wav"
            st.session_state["audio"].seek(0)
    else:
        st.session_state['audio'] = None
        st.session_state['mimetype'] = None

options = {
    "model": ASR_MODELS[asr_model],
    list(lang_options.keys())[0]: list(lang_options.values())[0],
}


@st.experimental_fragment
def transcribe_container():
    global transcribe_button_container, transcribe_status, transcribe_button, VECTOR_INDEX
    transcribe_button_container = stylable_container(
        key="transcribe_button",
        css_styles=button_css
    )
    transcribe_status = stylable_container(key="details",css_styles=transcript_container).empty()
    user_input = ""
    # Buttons with styling
    transcribe_button = transcribe_button_container.button("Transcribe", use_container_width=True, type="primary")
    if st.session_state['audio']:
        if transcribe_button:
            try:
                with transcribe_status.status("Transcribing", expanded=True) as transcribe_status:
                    output = prerecorded({"buffer": st.session_state["audio"], "mimetype": st.session_state.get("mimetype", "audio/wav")}, options['model'], options)
                    st.session_state.result = output['text']
                    transcribe_button_container.download_button("Download Transcript", data=st.session_state.result, type="primary", file_name="transcript.txt")
                    time_taken = output['time_taken']
                    transcribe_status.update(label=f"_Completed in {round(time_taken, 2)}s_", state='complete')
                    if st.session_state.result:
                        st.write(st.session_state.result)
                    with st.spinner("Indexing documents..."):
                        print(f"Indexing transcript to vectorstore...")
                        VECTOR_INDEX = create_vectorstore(st.session_state.result)
            except Exception as e:
                raise e
                transcribe_status.update(label="Error", state='error')
                st.error("Something went wrong :/")

@st.experimental_fragment
def chat_container():
    global user_input, transcribe_status, VECTOR_INDEX
    if st.session_state.get('audio'):
        user_input = st.chat_input(placeholder="Ask a question about the transcript:")
    else:
        user_input = ""

    groq_m = GROQ_MODELS[groq_model]
    if user_input:
        if not st.session_state.get("result"):
            try:
                with transcribe_status.status("Transcribing", expanded=True) as transcribe_status:
                    output = prerecorded({"buffer": st.session_state["audio"], "mimetype": st.session_state.get("mimetype", "audio/wav")}, options['model'], options)
                    st.session_state.result = output['text']
                    #TODO: download button does not work if user chats multiple times
                    transcribe_button_container.download_button("Download Transcript", data=st.session_state.result, type="primary", file_name="transcript.txt")
                    time_taken = output['time_taken']
                    transcribe_status.update(label=f"_Completed in {round(time_taken, 2)}s_", state='complete')
                    if st.session_state.result:
                        st.write(st.session_state.result)
                    with st.spinner("Indexing documents..."):
                        VECTOR_INDEX = create_vectorstore(st.session_state.result)
            except Exception as e:
                raise e
                transcribe_status.update(label="Error", state='error')
                st.error("Something went wrong :/")
        
        # Chat
        if len(st.session_state.result) <= 2000:
            print("Stuffing whole transcript into system prompt")
            context = st.session_state.result
        else:
            # Find most similar documents
            print("Using RAG pipeline")
            retriever = VECTOR_INDEX.as_retriever(similarity_top_k=3)
            nodes = retriever.retrieve(user_input)
            context = ""
            for node in nodes:
                context += node.text + "\n"

        try:
            prompt = f"""
            {user_input}
            """
            messages=[
                {"role": "system", "content": f"""\
You are helpful assistant that answers questions based on this transcript:
```
{context}
```
Answer questions that the user asks only about the transcript and nothing else. \
Do not include the user's question in your response, only respond with your answer. \
Your responses should be in markdown. \
"""},
                {"role": "user", "content": prompt},
            ]
            model=groq_m
            gen = chat_stream(model, messages)
            if transcribe_status:
                transcribe_status.update(expanded=False)
            with st.chat_message("ai", avatar=avatar_path):
                st.write_stream(gen)
        except Exception as e:
            raise e
            st.error("Something went wrong:/")
    return

transcribe_container()
chat_container()