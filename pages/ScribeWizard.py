import streamlit as st
import sys
import os

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


st.title("Scribe Wizard")




import streamlit as st
from groq import Groq
import json
import os
import subprocess
from io import BytesIO
from md2pdf.core import md2pdf
from dotenv import load_dotenv
from download import download_video_audio, delete_download
from pydub import AudioSegment
import tempfile

#st.title("Main App")

GROQ_MODELS = {
    model.id.replace("-", " ").title(): model.id
    for model in Groq().models.list().data
    if not ("whisper" in model.id or model.id.startswith("llama-guard"))
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

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", None)  ### see if it fixes it

MAX_FILE_SIZE = 100 * 1024 * 1024  # 25 MB
FILE_TOO_LARGE_MESSAGE = "The audio file is too large for the current size and rate limits using Whisper. If you used a YouTube link, please try a shorter video clip. If you uploaded an audio file, try trimming or compressing the audio to under 25 MB."

if 'api_key' not in st.session_state:
    st.session_state.api_key = GROQ_API_KEY

if 'groq' not in st.session_state:
    if GROQ_API_KEY:
        st.session_state.groq = Groq()



class GenerationStatistics:
    def __init__(self, input_time=0,output_time=0,input_tokens=0,output_tokens=0,total_time=0,model_name="llama-3.1-8b-instant"):
        self.input_time = input_time
        self.output_time = output_time
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_time = total_time # Sum of queue, prompt (input), and completion (output) times
        self.model_name = model_name

    def get_input_speed(self):
        """ 
        Tokens per second calculation for input
        """
        if self.input_time != 0:
            return self.input_tokens / self.input_time
        else:
            return 0
    
    def get_output_speed(self):
        """ 
        Tokens per second calculation for output
        """
        if self.output_time != 0:
            return self.output_tokens / self.output_time
        else:
            return 0
    
    def add(self, other):
        """
        Add statistics from another GenerationStatistics object to this one.
        """
        if not isinstance(other, GenerationStatistics):
            raise TypeError("Can only add GenerationStatistics objects")
        
        self.input_time += other.input_time
        self.output_time += other.output_time
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_time += other.total_time

    def __str__(self):
        return (f"\n## {self.get_output_speed():.2f} T/s ⚡\nRound trip time: {self.total_time:.2f}s  Model: {self.model_name}\n\n"
                f"| Metric          | Input          | Output          | Total          |\n"
                f"|-----------------|----------------|-----------------|----------------|\n"
                f"| Speed (T/s)     | {self.get_input_speed():.2f}            | {self.get_output_speed():.2f}            | {(self.input_tokens + self.output_tokens) / self.total_time if self.total_time != 0 else 0:.2f}            |\n"
                f"| Tokens          | {self.input_tokens}            | {self.output_tokens}            | {self.input_tokens + self.output_tokens}            |\n"
                f"| Inference Time (s) | {self.input_time:.2f}            | {self.output_time:.2f}            | {self.total_time:.2f}            |")

class NoteSection:
    def __init__(self, structure, transcript):
        self.structure = structure
        self.contents = {title: "" for title in self.flatten_structure(structure)}
        self.placeholders = {title: st.empty() for title in self.flatten_structure(structure)}

        st.markdown("## Raw transcript:")
        st.markdown(transcript)
        st.markdown("---")

    def flatten_structure(self, structure):
        sections = []
        for title, content in structure.items():
            sections.append(title)
            if isinstance(content, dict):
                sections.extend(self.flatten_structure(content))
        return sections

    def update_content(self, title, new_content):
        try:
            self.contents[title] += new_content
            self.display_content(title)
        except TypeError as e:
            pass

    def display_content(self, title):
        if self.contents[title].strip():
            self.placeholders[title].markdown(f"## {title}\n{self.contents[title]}")

    def return_existing_contents(self, level=1) -> str:
        existing_content = ""
        for title, content in self.structure.items():
            if self.contents[title].strip():  # Only include title if there is content
                existing_content += f"{'#' * level} {title}\n{self.contents[title]}.\n\n"
            if isinstance(content, dict):
                existing_content += self.get_markdown_content(content, level + 1)
        return existing_content

    def display_structure(self, structure=None, level=1):
        if structure is None:
            structure = self.structure
        
        for title, content in structure.items():
            if self.contents[title].strip():  # Only display title if there is content
                st.markdown(f"{'#' * level} {title}")
                self.placeholders[title].markdown(self.contents[title])
            if isinstance(content, dict):
                self.display_structure(content, level + 1)

    def display_toc(self, structure, columns, level=1, col_index=0):
        for title, content in structure.items():
            with columns[col_index % len(columns)]:
                st.markdown(f"{' ' * (level-1) * 2}- {title}")
            col_index += 1
            if isinstance(content, dict):
                col_index = self.display_toc(content, columns, level + 1, col_index)
        return col_index

    def get_markdown_content(self, structure=None, level=1):
        """
        Returns the markdown styled pure string with the contents.
        """
        if structure is None:
            structure = self.structure
        
        markdown_content = ""
        for title, content in structure.items():
            if self.contents[title].strip():  # Only include title if there is content
                markdown_content += f"{'#' * level} {title}\n{self.contents[title]}.\n\n"
            if isinstance(content, dict):
                markdown_content += self.get_markdown_content(content, level + 1)
        return markdown_content

def create_markdown_file(content: str) -> BytesIO:
    """
    Create a Markdown file from the provided content.
    """
    markdown_file = BytesIO()
    markdown_file.write(content.encode('utf-8'))
    markdown_file.seek(0)
    return markdown_file

def create_pdf_file(content: str):
    """
    Create a PDF file from the provided content.
    """
    pdf_buffer = BytesIO()
    md2pdf(pdf_buffer, md_content=content)
    pdf_buffer.seek(0)
    return pdf_buffer

def transcribe_audio(audio_file):
    """
    Transcribes audio using Groq's Whisper API.
    """
    transcription = st.session_state.groq.audio.transcriptions.create(
      file=audio_file,
      model="whisper-large-v3",
      prompt="",
      response_format="json",
      language="en",
      temperature=0.0 
    )

    results = transcription.text
    return results

def generate_notes_structure(transcript: str, model: str = "llama-3.1-70b-versatile"):
    """
    Returns notes structure content as well as total tokens and total time for generation.
    """

    shot_example = """
"Introduction": "Introduction to the AMA session, including the topic of Groq scaling architecture and the panelists",
"Panelist Introductions": "Brief introductions from Igor, Andrew, and Omar, covering their backgrounds and roles at Groq",
"Groq Scaling Architecture Overview": "High-level overview of Groq's scaling architecture, covering hardware, software, and cloud components",
"Hardware Perspective": "Igor's overview of Groq's hardware approach, using an analogy of city traffic management to explain the traditional compute approach and Groq's innovative approach",
"Traditional Compute": "Description of traditional compute approach, including asynchronous nature, queues, and poor utilization of infrastructure",
"Groq's Approach": "Description of Groq's approach, including pre-orchestrated movement of data, low latency, high energy efficiency, and high utilization of resources",
"Hardware Implementation": "Igor's explanation of the hardware implementation, including a comparison of GPU and LPU architectures"
}"""
    completion = st.session_state.groq.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Write in JSON format:\n\n{\"Title of section goes here\":\"Description of section goes here\",\"Title of section goes here\":\"Description of section goes here\",\"Title of section goes here\":\"Description of section goes here\"}"
            },
            {
                "role": "user",
                "content": f"### Transcript {transcript}\n\n### Example\n\n{shot_example}### Instructions\n\nCreate a structure for comprehensive notes on the above transcribed audio. Section titles and content descriptions must be comprehensive. Quality over quantity."
            }
        ],
        temperature=0.3,
        max_tokens=8000,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )

    usage = completion.usage
    statistics_to_return = GenerationStatistics(input_time=usage.prompt_time, output_time=usage.completion_time, input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens, total_time=usage.total_time, model_name=model)

    return statistics_to_return, completion.choices[0].message.content

def generate_section(transcript: str, existing_notes: str, section: str, model: str = "llama-3.1-8b-instant"):
    stream = st.session_state.groq.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert writer. Generate a comprehensive note for the section provided based factually on the transcript provided. Do *not* repeat any content from previous sections."
            },
            {
                "role": "user",
                "content": f"### Transcript\n\n{transcript}\n\n### Existing Notes\n\n{existing_notes}\n\n### Instructions\n\nGenerate comprehensive notes for this section only based on the transcript: \n\n{section}"
            }
        ],
        temperature=0.3,
        max_tokens=8000,
        top_p=1,
        stream=True,
        stop=None,
    )

    for chunk in stream:
        tokens = chunk.choices[0].delta.content
        if tokens:
            yield tokens
        if x_groq := chunk.x_groq:
            if not x_groq.usage:
                continue
            usage = x_groq.usage
            statistics_to_return = GenerationStatistics(input_time=usage.prompt_time, output_time=usage.completion_time, input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens, total_time=usage.total_time, model_name=model)
            yield statistics_to_return

# Initialize
if 'button_disabled' not in st.session_state:
    st.session_state.button_disabled = False

if 'button_text' not in st.session_state:
    st.session_state.button_text = "Generate Notes"

if 'statistics_text' not in st.session_state:
    st.session_state.statistics_text = ""

st.write("""
# ScribeWizard: Create structured notes from audio 
""")

def disable():
    st.session_state.button_disabled = True

def enable():
    st.session_state.button_disabled = False

def empty_st():
    st.empty()

try:
    with st.sidebar:
        st.write(f"# 🧙‍♂️ ScribeWizard \n## Generate notes from audio using Groq, Whisper, and Llama 3.1")
        #st.markdown(f"[Github Repository](https://github.com/bklieger/scribewizard)\n\nAs with all generative AI, content may include inaccurate or placeholder information. ScribeWizard is in beta and all feedback is welcome!")

        #st.write(f"---") 
        st.write("# Customization Settings\n🧪 These settings are experimental.\n")
        st.write(f"By default, ScribeWizard uses Llama3.1-70b for generating the notes outline and Llama3.1-8b for the content. This balances quality with speed and rate limit usage. You can customize these selections below.")
        outline_model_options = ["llama-3.1-70b-versatile","llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"]
        #outline_model_options = list(GROQ_MODELS.keys())
        outline_selected_model = st.selectbox("Outline generation:", outline_model_options)
        content_model_options = ["llama-3.1-8b-instant","llama-3.1-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"]
        #content_model_options = list(GROQ_MODELS.keys())
        content_selected_model = st.selectbox("Content generation:", content_model_options)
        st.write(f"---")



        
        # Add note about rate limits
        st.info("Important: Different models have different token and rate limits which may cause runtime errors.")
    

    if 'button_step' not in st.session_state:
        st.session_state.button_step = 0

    if 'button_text' not in st.session_state:
        st.session_state.button_text = "Check File"

    if 'file_info' not in st.session_state:
        st.session_state.file_info = None

    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None

    if st.button('End Generation and Download Notes'):
        if "notes" in st.session_state:

            # Create markdown file
            markdown_file = create_markdown_file(st.session_state.notes.get_markdown_content())
            st.download_button(
                label='Download Text',
                data=markdown_file,
                file_name='generated_notes.txt',
                mime='text/plain'
            )

            # Create pdf file (styled)
            pdf_file = create_pdf_file(st.session_state.notes.get_markdown_content())
            st.download_button(
                label='Download PDF',
                data=pdf_file,
                file_name='generated_notes.pdf',
                mime='application/pdf'
            )
            st.session_state.button_disabled = False
        else:
            raise ValueError("Please generate content first before downloading the notes.")

    input_method = st.radio("Choose input method:", ["Upload audio file", "YouTube link"])
    if input_method == "Upload audio file":
        audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
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
            st.write(f"File: {audio_file.name} ({original_file_size:.2f} MB --> {processed_file_size:.2f} MB, Duration: {duration})")

            st.session_state['audio'] = processed_audio
            st.session_state['mimetype'] = "audio/mp3"  # Set mimetype to mp3 since we converted it
            st.audio(st.session_state['audio'])  # Show audio player for uploaded file

            st.session_state.audio_file = processed_audio.read()
            st.session_state.button_step = 1
            st.session_state.button_text = "Generate Notes"   
    else:
        youtube_link = st.text_input("Enter YouTube link:", "")

    with st.form("groqform"):
        if not GROQ_API_KEY:
            groq_input_key = st.text_input("Enter your Groq API Key (gsk_yA...):", "", type="password")

        submitted = st.form_submit_button(st.session_state.button_text)

        if submitted:
            if st.session_state.button_step == 0:
                if input_method == "Upload audio file" and st.session_state.audio_file is None:
                    st.error("Please upload an audio file")
                elif input_method == "YouTube link" and not youtube_link:
                    st.error("Please enter a YouTube link")
                else:
                    st.session_state.button_step = 1
                    st.session_state.button_text = "Generate Notes"
            elif st.session_state.button_step == 1:
                #processing status
                status_text = st.empty()
                def display_status(text):
                    status_text.write(text)

                def clear_status():
                    status_text.empty()

                download_status_text = st.empty()
                def display_download_status(text:str):
                    download_status_text.write(text)    

                def clear_download_status():
                    download_status_text.empty()
                
                # Statistics
                placeholder = st.empty()
                def display_statistics():
                    with placeholder.container():
                        if st.session_state.statistics_text:
                            if "Transcribing audio in background" not in st.session_state.statistics_text:
                                st.markdown(st.session_state.statistics_text + "\n\n---\n")  # Format with line if showing statistics
                            else:
                                st.markdown(st.session_state.statistics_text)
                        else:
                            placeholder.empty()

                if submitted:
                    if input_method == "Upload audio file" and st.session_state.audio_file is None:
                        st.error("Please upload an audio file")
                    elif input_method == "YouTube link" and not youtube_link:
                        st.error("Please enter a YouTube link")
                    else:
                        st.session_state.button_disabled = True
                        # Show temporary message before transcription is generated and statistics show
                
                        audio_file_path = None

                        if input_method == "YouTube link":
                            display_status("Downloading audio from YouTube link ....")
                            audio_file_path = download_video_audio(youtube_link, display_download_status)
                            if audio_file_path is None:
                                st.error("Failed to download audio from YouTube link. Please try again.")
                                enable()
                                clear_status()
                            else:
                                # Read the downloaded file and create a file-like object
                                display_status("Processing Youtube audio ....")
                                with open(audio_file_path, 'rb') as f:
                                    file_contents = f.read()
                                audio_file = BytesIO(file_contents)

                                # Check size first to ensure will work with Whisper
                                if os.path.getsize(audio_file_path) > MAX_FILE_SIZE:
                                    raise ValueError(FILE_TOO_LARGE_MESSAGE)

                                audio_file.name = os.path.basename(audio_file_path)  # Set the file name
                                delete_download(audio_file_path)
                            clear_download_status()

                        if not GROQ_API_KEY:
                            st.session_state.groq = Groq(api_key=groq_input_key)

                        if input_method == "Upload audio file":
                            # Preprocess the uploaded audio
                            display_status("Preprocessing audio file ....")
                            audio_file_contents = st.session_state.audio_file
                            try:
                                result = subprocess.run([
                                    'ffmpeg',
                                    '-i', 'pipe:0',
                                    '-ar', '16000',
                                    '-ac', '1',
                                    '-map', '0:a',
                                    '-f', 'mp3',
                                    '-'
                                ], input=audio_file_contents, capture_output=True, check=True)
                                
                                audio_file = BytesIO(result.stdout)
                                audio_file.name = "processed_audio.mp3"  # Set a new file name
                                
                                display_status("Audio preprocessing completed successfully.")
                            except subprocess.CalledProcessError as e:
                                display_status(f"Error preprocessing audio: {e}")
                                st.error("Failed to preprocess the audio file. Please try again with a different file.")
                                st.stop()
                            except Exception as e:
                                display_status(f"Unexpected error during audio preprocessing: {e}")
                                st.error("An unexpected error occurred. Please try again.")
                                st.stop()
                            
                        print(f"Preprocessed file size: {audio_file.getbuffer().nbytes / (1024 * 1024):.2f} MB")
                        display_status("Transcribing audio in background....")
                        transcription_text = transcribe_audio(audio_file)

                        display_statistics()

                        display_status("Generating notes structure....")
                        large_model_generation_statistics, notes_structure = generate_notes_structure(transcription_text, model=str(outline_selected_model))
                        print("Structure: ",notes_structure)

                        display_status("Generating notes ...")
                        total_generation_statistics = GenerationStatistics(model_name=str(content_selected_model))
                        clear_status()


                        try:
                            notes_structure_json = json.loads(notes_structure)
                            notes = NoteSection(structure=notes_structure_json,transcript=transcription_text)
                            
                            if 'notes' not in st.session_state:
                                st.session_state.notes = notes

                            st.session_state.notes.display_structure()

                            def stream_section_content(sections):
                                for title, content in sections.items():
                                    if isinstance(content, str):
                                        content_stream = generate_section(transcript=transcription_text, existing_notes=notes.return_existing_contents(), section=(title + ": " + content),model=str(content_selected_model))
                                        for chunk in content_stream:
                                            # Check if GenerationStatistics data is returned instead of str tokens
                                            chunk_data = chunk
                                            if type(chunk_data) == GenerationStatistics:
                                                total_generation_statistics.add(chunk_data)
                                                
                                                st.session_state.statistics_text = str(total_generation_statistics)
                                                display_statistics()
                                            elif chunk is not None:
                                                st.session_state.notes.update_content(title, chunk)
                                    elif isinstance(content, dict):
                                        stream_section_content(content)

                            stream_section_content(notes_structure_json)
                        except json.JSONDecodeError:
                            st.error("Failed to decode the notes structure. Please try again.")

                        enable()

except Exception as e:
    st.session_state.button_disabled = False

    if hasattr(e, 'status_code') and e.status_code == 413:
        # In the future, this limitation will be fixed as ScribeWizard will automatically split the audio file and transcribe each part.
        st.error(FILE_TOO_LARGE_MESSAGE)
    else:
        st.error(e)

    if st.button("Clear"):
        st.rerun()
    
    # Remove audio after exception to prevent data storage leak
    if audio_file_path is not None:
        delete_download(audio_file_path)