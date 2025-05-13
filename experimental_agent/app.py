import streamlit as st
from .agent import Agent
import asyncio
import nest_asyncio
import base64
from PIL import Image
import io
import fitz  # PyMuPDF
import tempfile
import os
import pygments
from pygments import lexers
from pygments.formatters import HtmlFormatter
import librosa
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
import noisereduce as nr
import time
from datetime import datetime

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()


def extract_images_from_pdf(pdf_file):
    """Extract images from a PDF file."""
    try:
        # Create a temporary file to store the PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf.write(pdf_file.getvalue())
            temp_pdf.flush()

            # Open the PDF
            pdf_document = fitz.open(temp_pdf.name)
            images = []

            # Iterate through each page
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]

                # Get image list
                image_list = page.get_images(full=True)

                # Iterate through images on the page
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]

                        # Convert to PIL Image
                        image = Image.open(io.BytesIO(image_bytes))
                        images.append(
                            {
                                "image": image,
                                "page": page_num + 1,
                                "index": img_index + 1,
                            }
                        )
                    except Exception as e:
                        st.warning(
                            f"Could not extract image {img_index + 1} from page {page_num + 1}: {str(e)}"
                        )
                        continue

            pdf_document.close()
            os.unlink(temp_pdf.name)
            return images
    except Exception as e:
        st.error(f"Error extracting images from PDF: {str(e)}")
        return []


def display_graph(graph_data):
    if graph_data and graph_data.startswith("data:image/png;base64,"):
        # Extract the base64 data
        base64_data = graph_data.split(",")[1]
        # Convert base64 to image
        image_data = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_data))
        # Display the image
        st.image(image, use_column_width=True)


def display_pdf_images():
    """Display all extracted PDF images."""
    if not st.session_state.pdf_images:
        st.info(
            "No images have been extracted from PDFs yet. Please upload a PDF file first."
        )
        return False

    images_found = False
    for pdf_name, images in st.session_state.pdf_images.items():
        if images:
            st.subheader(f"Images from {pdf_name}")
            for img_data in images:
                try:
                    st.image(
                        img_data["image"],
                        caption=f"Page {img_data['page']}, Image {img_data['index']}",
                        use_column_width=True,
                    )
                    images_found = True
                except Exception as e:
                    st.warning(
                        f"Could not display image from page {img_data['page']}: {str(e)}"
                    )

    if not images_found:
        st.info("No images could be displayed from the PDFs.")
        return False

    return True


def get_file_extension(filename):
    """Get the file extension from filename."""
    return os.path.splitext(filename)[1].lower()


def get_language_from_extension(extension):
    """Get programming language from file extension."""
    extension_map = {
        ".py": "Python",
        ".js": "JavaScript",
        ".jsx": "React",
        ".ts": "TypeScript",
        ".tsx": "React TypeScript",
        ".java": "Java",
        ".cpp": "C++",
        ".c": "C",
        ".cs": "C#",
        ".php": "PHP",
        ".rb": "Ruby",
        ".go": "Go",
        ".rs": "Rust",
        ".swift": "Swift",
        ".kt": "Kotlin",
        ".scala": "Scala",
        ".html": "HTML",
        ".css": "CSS",
        ".sql": "SQL",
        ".sh": "Shell",
        ".md": "Markdown",
        ".json": "JSON",
        ".xml": "XML",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".toml": "TOML",
        ".ini": "INI",
        ".env": "Environment Variables",
        ".txt": "Text",
        ".ipynb": "Jupyter Notebook",
    }
    return extension_map.get(extension, "Unknown")


def analyze_code_file(file):
    """Analyze a code file and return relevant information."""
    try:
        content = file.getvalue().decode("utf-8")
        extension = get_file_extension(file.name)
        language = get_language_from_extension(extension)

        # Get lexer for syntax highlighting
        try:
            lexer = lexers.get_lexer_for_filename(file.name)
        except:
            lexer = lexers.get_lexer_by_name("text")

        # Format code with syntax highlighting
        formatter = HtmlFormatter(style="monokai")
        highlighted_code = pygments.highlight(content, lexer, formatter)

        # Basic code analysis
        lines = content.split("\n")
        total_lines = len(lines)
        non_empty_lines = len([line for line in lines if line.strip()])
        comment_lines = len(
            [
                line
                for line in lines
                if line.strip().startswith(("#", "//", "/*", "*", "--"))
            ]
        )

        analysis = {
            "language": language,
            "total_lines": total_lines,
            "non_empty_lines": non_empty_lines,
            "comment_lines": comment_lines,
            "highlighted_code": highlighted_code,
            "raw_content": content,
        }

        return analysis
    except Exception as e:
        st.error(f"Error analyzing code file: {str(e)}")
        return None


def display_code_analysis(analysis):
    """Display code analysis results."""
    if not analysis:
        return

    # Only display analysis results without metrics
    if "imports" in analysis and analysis["imports"]:
        st.write("Imports:", ", ".join(analysis["imports"]))

    if "functions" in analysis and analysis["functions"]:
        st.write("Functions:", len(analysis["functions"]))
        for func in analysis["functions"]:
            st.write(f"- {func}")

    if "classes" in analysis and analysis["classes"]:
        st.write("Classes:", len(analysis["classes"]))
        for cls in analysis["classes"]:
            st.write(f"- {cls}")

    if "variables" in analysis and analysis["variables"]:
        st.write("Variables:", len(analysis["variables"]))
        for var in analysis["variables"]:
            st.write(f"- {var}")


def process_audio(audio_file, operations=None):
    """Process audio file with various operations."""
    if operations is None:
        operations = ["text", "noise_reduction", "normalize", "trim_silence"]

    results = {}

    try:
        # Load audio file
        audio = AudioSegment.from_file(audio_file)

        # Convert to numpy array for processing
        samples = np.array(audio.get_array_of_samples())
        sample_rate = audio.frame_rate

        # Perform requested operations
        if "noise_reduction" in operations:
            # Convert to float for noise reduction
            samples_float = samples.astype(np.float32) / np.iinfo(samples.dtype).max
            # Apply noise reduction
            reduced_noise = nr.reduce_noise(y=samples_float, sr=sample_rate)
            # Convert back to original format
            samples = (reduced_noise * np.iinfo(samples.dtype).max).astype(
                samples.dtype
            )
            results["noise_reduction"] = samples

        if "normalize" in operations:
            # Normalize audio
            samples = librosa.util.normalize(samples)
            results["normalize"] = samples

        if "trim_silence" in operations:
            # Trim silence from beginning and end
            trimmed_samples, _ = librosa.effects.trim(samples, top_db=20)
            results["trim_silence"] = trimmed_samples

        if "text" in operations:
            # Convert audio to text
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_file) as source:
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data)
                    results["text"] = text
                except sr.UnknownValueError:
                    results["text"] = "Could not understand audio"
                except sr.RequestError as e:
                    results["text"] = f"Error with speech recognition service: {str(e)}"

        # Save processed audio if any processing was done
        if any(
            op in operations for op in ["noise_reduction", "normalize", "trim_silence"]
        ):
            processed_audio = AudioSegment(
                samples.tobytes(),
                frame_rate=sample_rate,
                sample_width=samples.dtype.itemsize,
                channels=1,
            )
            results["processed_audio"] = processed_audio

        return results

    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None


def display_audio_processing_results(results):
    """Display the results of audio processing."""
    if not results:
        return

    st.subheader("Audio Processing Results")

    # Display transcribed text if available
    if "text" in results:
        st.write("Transcribed Text:")
        st.write(results["text"])

    # Display processed audio if available
    if "processed_audio" in results:
        st.write("Processed Audio:")
        # Convert to bytes for display
        audio_bytes = results["processed_audio"].export(format="wav").read()
        st.audio(audio_bytes, format="audio/wav")

    # Display audio analysis if available
    if any(key in results for key in ["noise_reduction", "normalize", "trim_silence"]):
        st.write("Audio Analysis:")
        if "noise_reduction" in results:
            st.write("✅ Noise reduction applied")
        if "normalize" in results:
            st.write("✅ Audio normalized")
        if "trim_silence" in results:
            st.write("✅ Silence trimmed")


def convert_audio_to_wav(audio_file):
    """Convert uploaded audio file to WAV format."""
    try:
        # Create a temporary directory to store the audio file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file to temporary directory
            temp_input = os.path.join(temp_dir, "input_audio")
            with open(temp_input, "wb") as f:
                f.write(audio_file.getvalue())

            # Convert to WAV using pydub
            audio = AudioSegment.from_file(temp_input)
            temp_output = os.path.join(temp_dir, "output.wav")
            audio.export(temp_output, format="wav")

            # Read the converted WAV file
            with open(temp_output, "rb") as f:
                wav_data = f.read()

            return wav_data
    except Exception as e:
        raise Exception(f"Error converting audio format: {str(e)}")


def remove_background_noise(audio_data, sr):
    """Remove background noise from audio using spectral gating."""
    try:
        # Convert to float32
        audio_float = audio_data.astype(np.float32)

        # Estimate noise from the first 1000ms
        noise_sample = audio_float[: int(sr)]

        # Apply noise reduction with more aggressive parameters for background noise
        reduced = nr.reduce_noise(
            y=audio_float,
            sr=sr,
            prop_decrease=1.0,  # More aggressive noise reduction
            n_std_thresh_stationary=1.5,  # Lower threshold for noise detection
            stationary=True,  # Assume stationary noise (background)
            n_fft=2048,  # Larger FFT window for better frequency resolution
            win_length=1024,  # Window length for analysis
            n_jobs=-1,  # Use all available CPU cores
        )

        return reduced
    except Exception as e:
        raise Exception(f"Error removing background noise: {str(e)}")


def process_and_transcribe(audio_segment):
    """Process audio segment and return transcription."""
    try:
        # Export to temporary WAV file for transcription
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            audio_segment.export(temp_wav.name, format="wav")

            # Transcribe using speech recognition
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_wav.name) as source:
                audio_data = recognizer.record(source)
                transcription = recognizer.recognize_google(audio_data)

            # Clean up temporary file
            os.unlink(temp_wav.name)
            return transcription
    except Exception as e:
        raise Exception(f"Error transcribing audio: {str(e)}")


def format_user_bubble(query, timestamp):
    """Format user message as a chat bubble on the right side of the screen"""
    return f"""
        <style>
        .user-bubble {{
            background-color: #E9F5FE;
            border-radius: 18px 18px 0px 18px;
            padding: 12px 18px;
            margin: 5px 0px;
            max-width: 80%;
            display: inline-block;
            float: right;
            clear: both;
            color: #0A2540;
            box-shadow: 0px 1px 2px rgba(0, 0, 0, 0.1);
            word-wrap: break-word;
        }}
        .user-bubble-container {{
            display: flex;
            justify-content: flex-end;
            width: 100%;
            margin-bottom: 5px;
            padding-right: 12%;
        }}
        </style>
        
        <div class="user-bubble-container">
            <div class="user-bubble">
                {query}
            </div>
        </div>
        """


def format_assistant_bubble(answer, timestamp):
    """Format assistant message as a chat bubble on the left side of the screen"""
    return f"""
    <div style="display: flex; justify-content: flex-start; margin-left:12%;">
        <div style="padding:10px;border-radius:10px;margin-bottom:5px;max-width:88%;align-self:flex-start;">
            <p style="margin:0;color: var(--text-color);">{answer}</p>
        </div>
    </div>
    """


def format_assistant_bubble_typewrite(answer: str, typewriter: bool = False):
    """Display assistant's response with optional typewriter effect"""
    container = st.empty()
    bubble_start = """
    <div style="display: flex; justify-content: flex-start; margin-left:12%;">
        <div style="color: var(--text-color);padding: 5px; border-radius: 10px;
                    margin-bottom: 5px; max-width: 88%; align-self: flex-start;">
    """
    bubble_end = """
        </div>
    </div>
    """

    if typewriter:
        current = ""
        for char in answer:
            current += char
            container.markdown(
                bubble_start + current + bubble_end,
                unsafe_allow_html=True,
            )
            time.sleep(0.007)
    else:
        container.markdown(bubble_start + answer + bubble_end, unsafe_allow_html=True)


def main():
    # Only set page config if running directly
    # if __name__ == "__main__":
    #     st.set_page_config("Experimental Agent")

    st.title("Experimental Agent")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent" not in st.session_state:
        try:
            st.session_state.agent = Agent()
        except Exception as e:
            st.error(f"🔴 {str(e)}")
            st.stop()
    if "content_processed" not in st.session_state:
        st.session_state.content_processed = False
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    # Add custom CSS for better styling
    st.markdown(
        """
        <style>
        .stChatInput {
            width: 100%;
            max-width: 950px;
            margin: 0 auto;
        }
        .stTextInput {
            width: 100%;
            max-width: 600px;
            margin: 0 auto;
        }
        .stTextInput input {
            font-size: 18px;
            padding: 20px;
        }
        .loading-box {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 50vh;
            margin-top:-10px;
        }
        .loading-card {
            background-color: #ffffff;
            padding: 40px 60px;
            border-radius: 16px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            font-size: 22px;
            font-weight: 500;
            color: #0A2540;
            text-align: center;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar for file upload and website input
    with st.sidebar:
        st.title("Menu:")

        # File upload section
        st.subheader("Upload Files")
        st.info("ℹ️ Maximum file size: 4MB")
        uploaded_files = st.file_uploader(
            "",  # Empty label to remove default text
            accept_multiple_files=True,
            type=[
                "pdf",
                "txt",
                "xlsx",
                "xls",
                "csv",
                "jpg",
                "jpeg",
                "png",
                "py",
                "js",
                "jsx",
                "ts",
                "tsx",
                "java",
                "cpp",
                "c",
                "cs",
                "php",
                "rb",
                "go",
                "rs",
                "swift",
                "kt",
                "scala",
                "html",
                "css",
                "sql",
                "sh",
                "md",
                "json",
                "xml",
                "yaml",
                "yml",
                "toml",
                "ini",
                "env",
                "ipynb",
                "wav",
                "mp3",
                "ogg",
                "flac",
            ],
            label_visibility="collapsed",
        )
        # Store uploaded files in session state
        st.session_state.uploaded_files = uploaded_files

        # Initialize preview states in session state if not exists
        if "preview_states" not in st.session_state:
            st.session_state.preview_states = {}

        # Add image preview functionality
        if uploaded_files:
            for file in uploaded_files:
                if file.type.startswith("image/"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"📸 {file.name}")

                    # Initialize preview state for this file if not exists
                    if file.name not in st.session_state.preview_states:
                        st.session_state.preview_states[file.name] = False

                    # Create a unique key for this file's button
                    button_key = f"preview_{file.name}"

                    # Toggle button for preview
                    with col2:
                        if st.button(
                            (
                                "👁️ Show"
                                if not st.session_state.preview_states[file.name]
                                else "👁️ Hide"
                            ),
                            key=button_key,
                            use_container_width=True,
                        ):
                            # Toggle the state
                            st.session_state.preview_states[file.name] = (
                                not st.session_state.preview_states[file.name]
                            )
                            st.rerun()

                    # Show image if preview is enabled
                    if st.session_state.preview_states[file.name]:
                        image = Image.open(file)
                        st.image(image, caption=file.name, use_container_width=True)

        # Website input section
        st.subheader("Add Website Content")
        website_url = st.text_input("Enter website URL")

        # Check if any content is available
        has_content = bool(uploaded_files) or bool(website_url.strip())

        # Process button
        process_button = st.button(
            "Process Content",
            disabled=not has_content,
            help="Upload files or enter a website URL to enable processing",
        )

        if not has_content:
            st.info("ℹ️ Please upload files or enter a website URL to begin")
            st.session_state.content_processed = False

        # Only process content when the button is clicked
        if process_button:
            with st.spinner("Processing..."):
                all_text = ""

                # Process uploaded files
                if uploaded_files:
                    for file in uploaded_files:
                        try:
                            # For non-audio files, process normally
                            if not file.type.startswith("audio/"):
                                file_text = st.session_state.agent.get_text_from_file(
                                    file
                                )
                                if file_text:
                                    all_text += file_text + "\n\n"
                                else:
                                    st.warning(
                                        f"⚠️ No text could be extracted from {file.name}"
                                    )
                        except Exception as e:
                            st.error(f"🔴 Error processing {file.name}: {str(e)}")

                # Process website content
                if website_url:
                    try:
                        website_text = asyncio.run(
                            st.session_state.agent.extract_text_from_website_async(
                                website_url
                            )
                        )
                        if website_text:
                            all_text += website_text + "\n\n"
                        else:
                            st.warning("⚠️ No text could be extracted from the website")
                    except Exception as e:
                        st.error(f"🔴 Error processing website: {str(e)}")

                if not all_text.strip() and not any(
                    f.type.startswith("audio/") for f in uploaded_files
                ):
                    st.error(
                        "🔴 No content could be extracted from the provided sources."
                    )
                    st.session_state.content_processed = False
                    return

                try:
                    if all_text.strip():
                        text_chunks = st.session_state.agent.get_text_chunks(all_text)
                        if not text_chunks:
                            st.error("🔴 The content might be empty or unreadable.")
                            st.session_state.content_processed = False
                            return

                        vector_store = st.session_state.agent.get_vector_store(
                            text_chunks
                        )
                        if not vector_store:
                            st.error(
                                "🔴 Failed to create vector store from the content."
                            )
                            st.session_state.content_processed = False
                            return

                    st.session_state.content_processed = True
                    st.success("✅ Done")
                except Exception as e:
                    st.error(f"🔴 Error creating vector store: {str(e)}")
                    st.session_state.content_processed = False
                    return

    # Display chat history with styled bubbles
    for message in st.session_state.chat_history:
        timestamp = datetime.now().strftime("%H:%M:%S")
        if message["role"] == "user":
            user_html = format_user_bubble(message["content"], timestamp)
            st.markdown(user_html, unsafe_allow_html=True)
        else:
            assistant_html = format_assistant_bubble(message["content"], timestamp)
            st.markdown(assistant_html, unsafe_allow_html=True)
            if "model_used" in message:
                st.caption(f"Model used: {message['model_used']}")
            if "audio" in message:
                st.audio(message["audio"])

    # Chat input
    if st.session_state.content_processed or any(
        f.type.startswith("audio/") for f in st.session_state.uploaded_files
    ):
        # Show help message for audio files
        if any(f.type.startswith("audio/") for f in st.session_state.uploaded_files):
            st.info(
                """
            📢 I see you've uploaded audio file(s). You can ask me to:
            - 'transcribe the audio'
            - 'reduce noise in the audio'
            - 'remove background noise'
            - 'normalize the audio'

            Just let me know what you'd like to do!
            """
            )

        user_question = st.chat_input("What would you like me to do?")
        if user_question:
            # Add user message to chat history
            st.session_state.chat_history.append(
                {"role": "user", "content": user_question}
            )

            # Display user message with styled bubble
            timestamp = datetime.now().strftime("%H:%M:%S")
            user_html = format_user_bubble(user_question, timestamp)
            st.markdown(user_html, unsafe_allow_html=True)

            try:
                # Get information about the most recently uploaded file
                recent_file_info = None
                if st.session_state.uploaded_files:
                    recent_file = st.session_state.uploaded_files[-1]
                    recent_file_info = f"{recent_file.name} ({recent_file.type})"

                # Get current model configuration
                model_config = st.session_state.agent.get_model_config()

                # Handle audio-related queries
                if any(
                    f.type.startswith("audio/") for f in st.session_state.uploaded_files
                ):
                    audio_file = next(
                        f
                        for f in st.session_state.uploaded_files
                        if f.type.startswith("audio/")
                    )

                    try:
                        # Convert audio to WAV format first
                        with st.spinner("Converting audio format..."):
                            wav_data = convert_audio_to_wav(audio_file)

                        # Create a temporary WAV file for processing
                        with tempfile.NamedTemporaryFile(
                            suffix=".wav", delete=False
                        ) as temp_wav:
                            temp_wav.write(wav_data)
                            temp_wav.flush()

                            # Determine operations from user query
                            operations = []
                            if "transcribe" in user_question.lower():
                                operations.append("transcribe")
                            if (
                                "background" in user_question.lower()
                                and "noise" in user_question.lower()
                            ):
                                operations.append("background_noise")
                            elif (
                                "noise" in user_question.lower()
                                and "reduce" in user_question.lower()
                            ):
                                operations.append("noise_reduction")
                            if "normalize" in user_question.lower():
                                operations.append("normalize")

                            if operations:
                                with st.spinner(f"Processing audio..."):
                                    processed_audio = None
                                    transcription = None

                                    for operation in operations:
                                        if operation == "background_noise":
                                            # Load audio file
                                            y, sr = librosa.load(temp_wav.name)
                                            # Remove background noise
                                            processed_y = remove_background_noise(y, sr)
                                            # Convert back to int16
                                            processed_audio_data = (
                                                processed_y * 32767
                                            ).astype(np.int16)
                                            # Create AudioSegment
                                            processed_audio = AudioSegment(
                                                processed_audio_data.tobytes(),
                                                frame_rate=sr,
                                                sample_width=2,
                                                channels=1,
                                            )
                                            response_parts = [
                                                "Background noise has been removed"
                                            ]

                                        elif operation == "transcribe":
                                            # If we have processed audio, transcribe that, otherwise transcribe original
                                            audio_to_transcribe = (
                                                processed_audio
                                                if processed_audio
                                                else AudioSegment.from_wav(
                                                    temp_wav.name
                                                )
                                            )
                                            transcription = process_and_transcribe(
                                                audio_to_transcribe
                                            )
                                            response_parts = [
                                                "Here's the transcription:",
                                                transcription,
                                            ]

                                        elif operation == "noise_reduction":
                                            audio_results = st.session_state.agent.process_audio_input(
                                                temp_wav.name, operation
                                            )
                                            if "processed_audio" in audio_results:
                                                processed_audio = audio_results[
                                                    "processed_audio"
                                                ]
                                                response_parts = [
                                                    "Noise has been reduced"
                                                ]

                                        elif operation == "normalize":
                                            audio_results = st.session_state.agent.process_audio_input(
                                                temp_wav.name, operation
                                            )
                                            if "processed_audio" in audio_results:
                                                processed_audio = audio_results[
                                                    "processed_audio"
                                                ]
                                                response_parts = [
                                                    "Audio has been normalized"
                                                ]

                                    # Prepare final response
                                    response = "\n\n".join(response_parts)

                                    # Add results to chat history
                                    message = {
                                        "role": "assistant",
                                        "content": response,
                                        "model_used": model_config["model"],
                                    }

                                    # Add processed audio if available
                                    if (
                                        processed_audio
                                        and "transcribe" not in operations
                                    ):
                                        message["audio"] = (
                                            processed_audio.export().read()
                                        )

                                    st.session_state.chat_history.append(message)

                                    # Display response with styled bubble
                                    assistant_html = format_assistant_bubble(
                                        response, timestamp
                                    )
                                    st.markdown(assistant_html, unsafe_allow_html=True)
                                    if (
                                        processed_audio
                                        and "transcribe" not in operations
                                    ):
                                        st.audio(processed_audio.export().read())
                                    st.caption(f"Model used: {model_config['model']}")

                                    # Clean up temporary file
                                    os.unlink(temp_wav.name)
                                    return
                            else:
                                response = """
                                I can help you with audio processing. Please specify what you'd like to do:
                                - 'transcribe the audio'
                                - 'reduce noise in the audio'
                                - 'remove background noise'
                                - 'normalize the audio'
                                """

                            # Clean up temporary file
                            os.unlink(temp_wav.name)
                    except Exception as e:
                        response = f"Error processing audio: {str(e)}"
                else:
                    # Process regular queries
                    response = asyncio.run(
                        st.session_state.agent.process_user_input_async(
                            user_question, recent_file_info
                        )
                    )

                # Add response to chat history
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "model_used": model_config["model"],
                    }
                )

                # Display response with styled bubble
                assistant_html = format_assistant_bubble(response, timestamp)
                st.markdown(assistant_html, unsafe_allow_html=True)
                st.caption(f"Model used: {model_config['model']}")

                st.rerun()
            except Exception as e:
                error_message = f"🔴 Error: {str(e)}"
                st.error(error_message)
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "model_used": "Error occurred",
                    }
                )
    else:
        st.info("📝 Please upload and process content first to start asking questions.")
        st.chat_input("Upload and process content first", disabled=True)

    # Auto-scroll to latest
    st.markdown(
        """
    <script>
    const chatContainer = window.parent.document.querySelector('.main');
    chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
    </script>
    """,
        unsafe_allow_html=True,
    )


# if __name__ == "__main__":
#     main()
