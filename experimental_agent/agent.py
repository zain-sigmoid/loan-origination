import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
from bs4 import BeautifulSoup
import io
from duckduckgo_search import DDGS
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import datetime
import concurrent.futures
import asyncio
import aiohttp
from functools import lru_cache
import atexit
import matplotlib.pyplot as plt
import json
import base64
from langchain_community.embeddings import HuggingFaceEmbeddings

# from youtube_transcript_api import YouTubeTranscriptApi
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
from gtts import gTTS
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
import streamlit as st

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("LLM_API_KEY")

# Define available Gemini models
GEMINI_MODELS = {
    "gemini-1.5-flash": {
        "model": "gemini-1.5-flash",
        "temperature": 0.3,
        "max_retries": 3,
        "chunk_size": 10000,
        "chunk_overlap": 1000,
    },
    "gemini-1.5-pro": {
        "model": "gemini-1.5-pro",
        "temperature": 0.3,
        "max_retries": 3,
        "chunk_size": 15000,
        "chunk_overlap": 1500,
    },
    "gemini-pro": {
        "model": "gemini-pro",
        "temperature": 0.3,
        "max_retries": 3,
        "chunk_size": 12000,
        "chunk_overlap": 1200,
    },
}


class ModelManager:
    def __init__(self):
        self.current_model_index = 0
        self.models = list(GEMINI_MODELS.keys())
        self.model_usage = {model: 0 for model in self.models}
        self.max_requests_per_model = 60  # Adjust this based on your API limits
        self._embeddings_cache = {}

    def get_next_model(self):
        current_model = self.models[self.current_model_index]
        self.model_usage[current_model] += 1

        # Check if current model has reached its limit
        if self.model_usage[current_model] >= self.max_requests_per_model:
            # Move to next model
            self.current_model_index = (self.current_model_index + 1) % len(self.models)
            current_model = self.models[self.current_model_index]
            # Reset usage for the new model
            self.model_usage[current_model] = 1

        return GEMINI_MODELS[current_model]

    @lru_cache(maxsize=100)
    def get_embeddings(self, text):
        if text not in self._embeddings_cache:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", google_api_key=GOOGLE_API_KEY
            )
            self._embeddings_cache[text] = embeddings.embed_query(text)
        return self._embeddings_cache[text]


class Agent:
    def __init__(self):
        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY not found. Please set it in your .env file or environment variables."
            )

        genai.configure(api_key=GOOGLE_API_KEY)
        self.memory = {
            "conversations": [],
            "context": {},
            "last_interaction": None,
            "document_context": {},
        }
        self.model_manager = ModelManager()
        self.vision_model = genai.GenerativeModel("gemini-1.5-flash")
        self.session = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.max_file_size = 4 * 1024 * 1024  # 4MB in bytes

        # Initialize prompts for different file types
        self.file_prompts = {
            "application/pdf": {
                "summary": """
                You are an expert document analyst. Please provide a comprehensive summary of this PDF document.

                Guidelines for Summary:
                1. Document Overview:
                   - Main topic and purpose
                   - Document type and format
                   - Target audience
                   - Key objectives

                2. Content Analysis:
                   - Main arguments and key points
                   - Supporting evidence and examples
                   - Important findings or conclusions
                   - Any significant data or statistics
                   - Methodologies used (if applicable)

                3. Critical Elements:
                   - Key takeaways
                   - Recommendations or implications
                   - Limitations or constraints
                   - Future considerations

                4. Structure and Organization:
                   - Document layout
                   - Section organization
                   - Flow of information
                   - Visual elements (if any)

                Please ensure the summary is:
                - Clear and well-structured
                - Concise but comprehensive
                - Focused on the most important information
                - Easy to understand
                - Properly formatted with sections and bullet points

                Content:
                {content}

                Summary:
                """,
                "analysis": """
                You are an expert document analyst. Please perform a detailed analysis of this PDF document.

                Guidelines for Analysis:
                1. Document Structure:
                   - Organization and layout
                   - Section hierarchy
                   - Flow and coherence
                   - Visual elements and formatting
                   - Navigation and accessibility

                2. Content Evaluation:
                   - Main arguments and their validity
                   - Supporting evidence quality
                   - Data analysis and interpretation
                   - Methodology assessment
                   - Conclusion reliability

                3. Writing Style:
                   - Tone and voice
                   - Language complexity
                   - Clarity and precision
                   - Technical accuracy
                   - Professional standards

                4. Critical Assessment:
                   - Strengths and weaknesses
                   - Bias and objectivity
                   - Completeness and gaps
                   - Credibility and reliability
                   - Potential improvements

                5. Practical Implications:
                   - Real-world applications
                   - Industry relevance
                   - Future implications
                   - Risk considerations
                   - Implementation challenges

                Please provide:
                - Detailed analysis of each aspect
                - Specific examples and evidence
                - Clear recommendations
                - Actionable insights
                - Professional assessment

                Content:
                {content}

                Analysis:
                """,
            },
            "text/plain": {
                "summary": """
                You are an expert text analyst. Please provide a comprehensive summary of this text document.

                Guidelines for Summary:
                1. Document Overview:
                   - Main topic and purpose
                   - Document type and format
                   - Target audience
                   - Key objectives

                2. Content Analysis:
                   - Main ideas and themes
                   - Key points and arguments
                   - Supporting evidence
                   - Important details
                   - Examples and illustrations

                3. Critical Elements:
                   - Key takeaways
                   - Main conclusions
                   - Important implications
                   - Notable insights
                   - Action items (if any)

                4. Structure and Flow:
                   - Organization of ideas
                   - Logical flow
                   - Section relationships
                   - Transition points

                Please ensure the summary is:
                - Clear and well-structured
                - Concise but comprehensive
                - Focused on key information
                - Easy to understand
                - Properly formatted

                Content:
                {content}

                Summary:
                """,
                "analysis": """
                You are an expert text analyst. Please perform a detailed analysis of this text document.

                Guidelines for Analysis:
                1. Content Evaluation:
                   - Main arguments and their validity
                   - Supporting evidence quality
                   - Information accuracy
                   - Completeness of coverage
                   - Depth of analysis

                2. Writing Style:
                   - Tone and voice
                   - Language complexity
                   - Clarity and precision
                   - Technical accuracy
                   - Professional standards

                3. Structure and Organization:
                   - Logical flow
                   - Section coherence
                   - Transition effectiveness
                   - Information hierarchy
                   - Formatting impact

                4. Critical Assessment:
                   - Strengths and weaknesses
                   - Bias and objectivity
                   - Completeness and gaps
                   - Credibility and reliability
                   - Potential improvements

                5. Practical Implications:
                   - Real-world applications
                   - Industry relevance
                   - Future implications
                   - Risk considerations
                   - Implementation challenges

                Please provide:
                - Detailed analysis of each aspect
                - Specific examples and evidence
                - Clear recommendations
                - Actionable insights
                - Professional assessment

                Content:
                {content}

                Analysis:
                """,
            },
            "application/vnd.ms-excel": {
                "summary": """
                You are an expert data analyst. Please provide a comprehensive summary of this spreadsheet data.

                Guidelines for Summary:
                1. Data Overview:
                   - Dataset structure and organization
                   - Key variables and metrics
                   - Data types and formats
                   - Time period covered
                   - Data sources

                2. Statistical Analysis:
                   - Key metrics and statistics
                   - Central tendencies
                   - Distribution patterns
                   - Correlation insights
                   - Significant findings

                3. Trends and Patterns:
                   - Notable trends
                   - Seasonal patterns
                   - Anomalies or outliers
                   - Growth or decline indicators
                   - Comparative insights

                4. Data Quality:
                   - Completeness assessment
                   - Accuracy evaluation
                   - Consistency check
                   - Reliability indicators
                   - Potential issues

                Please ensure the summary is:
                - Clear and well-structured
                - Data-driven and precise
                - Focused on key insights
                - Supported by evidence
                - Actionable and practical

                Content:
                {content}

                Summary:
                """,
                "analysis": """
                You are an expert data analyst. Please perform a detailed analysis of this spreadsheet data.

                Guidelines for Analysis:
                1. Data Quality Assessment:
                   - Completeness and coverage
                   - Accuracy and precision
                   - Consistency and reliability
                   - Data integrity
                   - Potential biases

                2. Statistical Analysis:
                   - Descriptive statistics
                   - Inferential statistics
                   - Correlation analysis
                   - Regression insights
                   - Significance testing

                3. Pattern Recognition:
                   - Trend analysis
                   - Seasonal patterns
                   - Cyclical behavior
                   - Anomaly detection
                   - Predictive indicators

                4. Business Intelligence:
                   - Key performance indicators
                   - Business impact
                   - Risk assessment
                   - Opportunity identification
                   - Strategic implications

                5. Technical Evaluation:
                   - Data structure efficiency
                   - Formula accuracy
                   - Calculation reliability
                   - Visualization effectiveness
                   - Technical limitations

                Please provide:
                - Detailed analysis of each aspect
                - Statistical evidence
                - Clear recommendations
                - Actionable insights
                - Professional assessment

                Content:
                {content}

                Analysis:
                """,
            },
            "text/csv": {
                "summary": """
                You are an expert data analyst. Please provide a comprehensive summary of this CSV data.

                Guidelines for Summary:
                1. Data Overview:
                   - Dataset structure and organization
                   - Key variables and metrics
                   - Data types and formats
                   - Time period covered
                   - Data sources

                2. Statistical Analysis:
                   - Key metrics and statistics
                   - Central tendencies
                   - Distribution patterns
                   - Correlation insights
                   - Significant findings

                3. Trends and Patterns:
                   - Notable trends
                   - Seasonal patterns
                   - Anomalies or outliers
                   - Growth or decline indicators
                   - Comparative insights

                4. Data Quality:
                   - Completeness assessment
                   - Accuracy evaluation
                   - Consistency check
                   - Reliability indicators
                   - Potential issues

                Please ensure the summary is:
                - Clear and well-structured
                - Data-driven and precise
                - Focused on key insights
                - Supported by evidence
                - Actionable and practical

                Content:
                {content}

                Summary:
                """,
                "analysis": """
                You are an expert data analyst. Please perform a detailed analysis of this CSV data.

                Guidelines for Analysis:
                1. Data Quality Assessment:
                   - Completeness and coverage
                   - Accuracy and precision
                   - Consistency and reliability
                   - Data integrity
                   - Potential biases

                2. Statistical Analysis:
                   - Descriptive statistics
                   - Inferential statistics
                   - Correlation analysis
                   - Regression insights
                   - Significance testing

                3. Pattern Recognition:
                   - Trend analysis
                   - Seasonal patterns
                   - Cyclical behavior
                   - Anomaly detection
                   - Predictive indicators

                4. Business Intelligence:
                   - Key performance indicators
                   - Business impact
                   - Risk assessment
                   - Opportunity identification
                   - Strategic implications

                5. Technical Evaluation:
                   - Data structure efficiency
                   - Format consistency
                   - Processing requirements
                   - Integration capabilities
                   - Technical limitations

                Please provide:
                - Detailed analysis of each aspect
                - Statistical evidence
                - Clear recommendations
                - Actionable insights
                - Professional assessment

                Content:
                {content}

                Analysis:
                """,
            },
            "image/": {
                "summary": """
                You are an expert image analyst. Please provide a comprehensive description of this image.

                Guidelines for Description:
                1. Visual Elements:
                   - Main subjects and objects
                   - Composition and layout
                   - Colors and lighting
                   - Visual hierarchy
                   - Key focal points

                2. Technical Aspects:
                   - Image quality
                   - Resolution and clarity
                   - Lighting conditions
                   - Focus and depth
                   - Technical execution

                3. Context and Setting:
                   - Environment and location
                   - Time period or era
                   - Cultural context
                   - Historical significance
                   - Environmental factors

                4. Notable Details:
                   - Important features
                   - Unique characteristics
                   - Significant elements
                   - Hidden details
                   - Special effects

                Please ensure the description is:
                - Clear and well-structured
                - Detailed but concise
                - Focused on key elements
                - Technically accurate
                - Contextually relevant

                Image Analysis:
                """,
                "analysis": """
                You are an expert image analyst. Please perform a detailed analysis of this image.

                Guidelines for Analysis:
                1. Visual Composition:
                   - Layout and structure
                   - Balance and symmetry
                   - Visual hierarchy
                   - Focal points
                   - Negative space

                2. Technical Evaluation:
                   - Image quality
                   - Resolution and clarity
                   - Lighting and exposure
                   - Focus and depth
                   - Technical execution

                3. Artistic Elements:
                   - Color theory
                   - Composition rules
                   - Style and technique
                   - Creative elements
                   - Artistic intent

                4. Content Analysis:
                   - Subject matter
                   - Symbolism and meaning
                   - Cultural context
                   - Historical significance
                   - Emotional impact

                5. Professional Assessment:
                   - Technical proficiency
                   - Artistic merit
                   - Communication effectiveness
                   - Target audience appeal
                   - Overall impact

                Please provide:
                - Detailed analysis of each aspect
                - Technical insights
                - Artistic evaluation
                - Cultural context
                - Professional assessment

                Image Analysis:
                """,
            },
        }

        # Universal prompt for any file type
        self.universal_prompt = """
        You are an expert content analyst. Please provide a comprehensive analysis of this content.

        Guidelines for Analysis:
        1. Content Overview:
           - Main topic and purpose
           - Content type and format
           - Target audience
           - Key objectives
           - Overall structure

        2. Detailed Analysis:
           - Main points and arguments
           - Supporting evidence
           - Key findings
           - Important details
           - Technical aspects

        3. Critical Evaluation:
           - Strengths and weaknesses
           - Completeness and gaps
           - Accuracy and reliability
           - Bias and objectivity
           - Potential improvements

        4. Practical Implications:
           - Real-world applications
           - Industry relevance
           - Future implications
           - Risk considerations
           - Implementation challenges

        Please ensure the analysis is:
        - Clear and well-structured
        - Comprehensive and detailed
        - Evidence-based
        - Actionable
        - Professional

        Content:
        {content}

        Analysis:
        """

        # Register cleanup
        atexit.register(self.cleanup)

        # Initialize the model
        self.model = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048,
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize vector store
        self.vector_store = None

        # Initialize code analysis tools
        self.code_analysis_tools = {
            "python": self._analyze_python_code,
            "javascript": self._analyze_javascript_code,
            "java": self._analyze_java_code,
            "cpp": self._analyze_cpp_code,
            "csharp": self._analyze_csharp_code,
            "php": self._analyze_php_code,
            "ruby": self._analyze_ruby_code,
            "go": self._analyze_go_code,
            "rust": self._analyze_rust_code,
            "swift": self._analyze_swift_code,
            "kotlin": self._analyze_kotlin_code,
            "scala": self._analyze_scala_code,
            "html": self._analyze_html_code,
            "css": self._analyze_css_code,
            "sql": self._analyze_sql_code,
            "shell": self._analyze_shell_code,
            "markdown": self._analyze_markdown_code,
            "json": self._analyze_json_code,
            "xml": self._analyze_xml_code,
            "yaml": self._analyze_yaml_code,
            "toml": self._analyze_toml_code,
            "ini": self._analyze_ini_code,
            "env": self._analyze_env_code,
        }

        # Initialize audio components
        self.recognizer = sr.Recognizer()

    def cleanup(self):
        if self.session:
            asyncio.run(self.session.close())
        self.executor.shutdown(wait=False)

    async def ensure_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    def get_model_config(self):
        return self.model_manager.get_next_model()

    async def search_web_async(self, query, num_results=3):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))
                return results
        except Exception as e:
            raise Exception(f"Error performing web search: {e}")

    async def extract_text_from_website_async(self, url):
        try:
            session = await self.ensure_session()
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    raise Exception(f"Failed to fetch website: HTTP {response.status}")

                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                # Remove unwanted elements
                for element in soup(
                    ["script", "style", "nav", "footer", "header", "aside"]
                ):
                    element.decompose()

                # Extract main content
                main_content = (
                    soup.find("main")
                    or soup.find("article")
                    or soup.find("div", class_="content")
                    or soup.find("div", id="content")
                )

                if main_content:
                    text = main_content.get_text(separator="\n", strip=True)
                else:
                    text = soup.get_text(separator="\n", strip=True)

                # Clean up the text
                lines = (line.strip() for line in text.splitlines())
                chunks = (
                    phrase.strip() for line in lines for phrase in line.split("  ")
                )
                text = "\n".join(chunk for chunk in chunks if chunk)

                if not text.strip():
                    raise Exception("No content could be extracted from the website")

                return text
        except Exception as e:
            raise Exception(f"Error extracting text from website: {e}")

    def process_image(self, image_file):
        try:
            # Read image data
            image_bytes = io.BytesIO(image_file.getvalue())

            # Verify image data is valid
            try:
                image = Image.open(image_bytes)
                image.verify()  # Verify it's a valid image
                image_bytes.seek(0)  # Reset buffer position
            except Exception as e:
                raise Exception(f"Invalid image file: {str(e)}")

            if image_file.type == "application/pdf":
                try:
                    images = convert_from_bytes(image_bytes.getvalue())
                    text = ""
                    for idx, img in enumerate(images):
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format="PNG")
                        img_byte_arr = img_byte_arr.getvalue()

                        image_parts = [{"mime_type": "image/png", "data": img_byte_arr}]

                        prompt = """
                        Analyze this image in detail and provide a comprehensive description. 
                        Focus on:
                        1. Main subjects and objects
                        2. Colors and visual elements
                        3. Text content (if any)
                        4. Context and setting
                        5. Any notable features or patterns
                        
                        Provide the information in a clear, structured format.
                        """

                        try:
                            response = self.vision_model.generate_content(
                                [prompt, image_parts[0]]
                            )
                            if response and response.text:
                                text += f"Page {idx + 1} Analysis:\n{response.text}\n\n"
                            else:
                                text += f"Page {idx + 1}: Vision model could not analyze. Trying OCR...\n"
                                ocr_text = pytesseract.image_to_string(img)
                                if ocr_text.strip():
                                    text += (
                                        f"OCR Text from Page {idx + 1}:\n{ocr_text}\n\n"
                                    )
                                else:
                                    text += f"Page {idx + 1}: No text could be extracted.\n\n"
                        except Exception as e:
                            text += (
                                f"Page {idx + 1}: Error in vision analysis: {str(e)}\n"
                            )
                            try:
                                ocr_text = pytesseract.image_to_string(img)
                                if ocr_text.strip():
                                    text += (
                                        f"OCR Text from Page {idx + 1}:\n{ocr_text}\n\n"
                                    )
                            except Exception as ocr_error:
                                text += f"Page {idx + 1}: OCR also failed: {str(ocr_error)}\n\n"
                except Exception as e:
                    raise Exception(f"Error processing PDF: {str(e)}")
            else:
                # For single images
                image_parts = [
                    {"mime_type": image_file.type, "data": image_bytes.getvalue()}
                ]

                prompt = """
                Analyze this image in detail and provide a comprehensive description. 
                Focus on:
                1. Main subjects and objects
                2. Colors and visual elements
                3. Text content (if any)
                4. Context and setting
                5. Any notable features or patterns
                
                Provide the information in a clear, structured format.
                """

                try:
                    response = self.vision_model.generate_content(
                        [prompt, image_parts[0]]
                    )
                    if response and response.text:
                        text = response.text
                    else:
                        raise Exception("Vision model returned empty response")
                except Exception as e:
                    # If vision model fails, try OCR
                    try:
                        image = Image.open(image_bytes)
                        text = pytesseract.image_to_string(image)
                        if not text.strip():
                            raise Exception("OCR produced no text")
                        text = f"Vision model analysis failed. OCR Text:\n{text}"
                    except Exception as ocr_error:
                        raise Exception(
                            f"Both vision model and OCR failed. Vision error: {str(e)}, OCR error: {str(ocr_error)}"
                        )

            if not text.strip():
                raise Exception(
                    "No content could be extracted from the image using any method"
                )

            return text
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            if "API key" in str(e):
                error_msg = "Error: Invalid or missing Google API key for vision model"
            elif "quota" in str(e).lower():
                error_msg = "Error: API quota exceeded for vision model"
            elif "permission" in str(e).lower():
                error_msg = "Error: Permission denied for vision model"
            elif "invalid image" in str(e).lower():
                error_msg = (
                    "Error: The uploaded file is not a valid image or is corrupted"
                )
            raise Exception(error_msg)

    def check_file_size(self, file):
        """Check if the file size is within the allowed limit."""
        file_size = len(file.getvalue())
        if file_size > self.max_file_size:
            raise Exception(
                f"File size ({file_size / (1024 * 1024):.2f}MB) exceeds the maximum allowed size of 4MB"
            )
        file.seek(0)  # Reset file pointer to beginning
        return True

    def get_text_from_file(self, file):
        text = ""
        try:
            # Check file size before processing
            self.check_file_size(file)

            if file.type == "application/pdf":
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted
            elif file.type == "text/plain":
                text = file.getvalue().decode("utf-8")
            elif file.type in [
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "text/csv",
            ]:
                df = (
                    pd.read_excel(file)
                    if file.type != "text/csv"
                    else pd.read_csv(file)
                )
                text = df.to_string()
                # Store the DataFrame in memory for later use
                self.current_dataframe = df
                # Add column information to the text
                text += "\n\nColumn Information:\n"
                for col in df.columns:
                    text += f"{col}: {df[col].dtype}\n"
            elif file.type.startswith("image/"):
                text = self.process_image(file)
            else:
                # Handle code files
                content = file.getvalue().decode("utf-8")
                extension = os.path.splitext(file.name)[1].lower()
                language = self._get_language_from_extension(extension)

                # Analyze code if it's a supported language
                if language in self.code_analysis_tools:
                    analysis = self.analyze_code(content, language)
                    return f"Code Analysis for {file.name}:\n{json.dumps(analysis, indent=2)}"
                else:
                    return content
        except Exception as e:
            raise Exception(f"Error processing file: {e}")
        return text

    def get_text_chunks(self, text):
        model_config = self.get_model_config()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=model_config["chunk_size"],
            chunk_overlap=model_config["chunk_overlap"],
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vector_store(self, text_chunks):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", google_api_key=GOOGLE_API_KEY
            )
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            vector_store.save_local("faiss_index")
            return vector_store
        except Exception as e:
            raise Exception(f"Error creating vector store: {e}")

    def load_vector_store(self):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", google_api_key=GOOGLE_API_KEY
            )
            if os.path.exists("faiss_index"):
                return FAISS.load_local(
                    "faiss_index", embeddings, allow_dangerous_deserialization=True
                )
            return None
        except Exception as e:
            raise Exception(f"Error loading vector store: {e}")

    def update_memory(self, user_input, assistant_response, context=None):
        timestamp = datetime.datetime.now()

        conversation = {
            "timestamp": timestamp,
            "user_input": user_input,
            "assistant_response": assistant_response,
            "context": context or {},
        }

        self.memory["conversations"].append(conversation)
        self.memory["last_interaction"] = timestamp

        if context and "document_info" in context:
            doc_key = f"doc_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            self.memory["document_context"][doc_key] = context["document_info"]

        if len(self.memory["conversations"]) > 10:
            self.memory["conversations"] = self.memory["conversations"][-10:]

        if len(self.memory["document_context"]) > 10:
            recent_docs = dict(
                sorted(
                    self.memory["document_context"].items(),
                    key=lambda x: x[0],
                    reverse=True,
                )[:10]
            )
            self.memory["document_context"] = recent_docs

    def get_relevant_context(self, user_question):
        relevant_context = []

        for conv in reversed(self.memory["conversations"]):
            if any(
                keyword in user_question.lower()
                for keyword in conv["user_input"].lower().split()
            ):
                relevant_context.append(
                    {
                        "timestamp": conv["timestamp"],
                        "context": conv["context"],
                        "response": conv["assistant_response"],
                    }
                )

        return relevant_context

    def generate_graph(self, data, columns):
        """Generate appropriate graph based on data types and relationships."""
        try:
            # Convert data to DataFrame if it's not already
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data, columns=columns)

            # Determine the best graph type based on data
            graph_type = self._determine_graph_type(data, columns)

            # Create the graph
            plt.figure(figsize=(10, 6))

            if graph_type == "line":
                self._create_line_graph(data, columns)
            elif graph_type == "bar":
                self._create_bar_graph(data, columns)
            elif graph_type == "scatter":
                self._create_scatter_plot(data, columns)
            elif graph_type == "pie":
                self._create_pie_chart(data, columns)
            elif graph_type == "histogram":
                self._create_histogram(data, columns)

            plt.tight_layout()

            # Convert plot to base64
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            plt.close()

            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            print(f"Error generating graph: {str(e)}")
            return None

    def _determine_graph_type(self, data, columns):
        """Determine the most appropriate graph type based on data characteristics."""
        if len(columns) < 2:
            return "histogram"  # Default to histogram for single column

        # Get data types
        dtypes = data.dtypes

        # Check for time series data
        if any(
            col.lower() in ["date", "time", "year", "month", "day"] for col in columns
        ):
            return "line"

        # Check for categorical vs numerical relationship
        categorical_cols = [
            col
            for col in columns
            if dtypes[col] == "object" or dtypes[col] == "category"
        ]
        numerical_cols = [
            col for col in columns if pd.api.types.is_numeric_dtype(dtypes[col])
        ]

        if len(categorical_cols) == 1 and len(numerical_cols) == 1:
            return "bar"
        elif len(numerical_cols) >= 2:
            return "scatter"
        elif len(categorical_cols) == 1 and len(data) <= 10:
            return "pie"

        return "bar"  # Default to bar graph

    def _create_line_graph(self, data, columns):
        """Create a line graph."""
        x_col = columns[0]
        y_col = columns[1]
        plt.plot(data[x_col], data[y_col], marker="o")
        plt.title(f"{y_col} vs {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)

    def _create_bar_graph(self, data, columns):
        """Create a bar graph."""
        x_col = columns[0]
        y_col = columns[1]
        plt.bar(data[x_col], data[y_col])
        plt.title(f"{y_col} by {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.xticks(rotation=45)
        plt.grid(True)

    def _create_scatter_plot(self, data, columns):
        """Create a scatter plot."""
        x_col = columns[0]
        y_col = columns[1]
        plt.scatter(data[x_col], data[y_col])
        plt.title(f"{y_col} vs {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)

    def _create_pie_chart(self, data, columns):
        """Create a pie chart."""
        x_col = columns[0]
        y_col = columns[1]
        plt.pie(data[y_col], labels=data[x_col], autopct="%1.1f%%")
        plt.title(f"Distribution of {y_col}")

    def _create_histogram(self, data, columns):
        """Create a histogram."""
        col = columns[0]
        plt.hist(data[col], bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True)

    def get_conversational_chain(self):
        prompt_template = """
        You are an advanced AI assistant with expertise in analyzing and understanding various types of content including documents, websites, and data files. You can handle complex queries and provide detailed, accurate responses.

        Previous Context:
        {memory_context}
        
        Your task is to provide comprehensive and accurate answers based on the provided context. Follow these guidelines:
        
        1. For Complex Queries:
           - Break down complex questions into sub-questions
           - Analyze relationships between different pieces of information
           - Consider temporal aspects (if time-related)
           - Handle multi-step reasoning
           - Provide step-by-step explanations when needed
           - Consider edge cases and exceptions
        
        2. For Data Analysis:
           - Identify patterns and trends
           - Perform statistical analysis when relevant
           - Compare and contrast different data points
           - Make predictions based on historical data
           - Identify anomalies or outliers
           - Provide confidence levels for predictions
        
        3. For Document Analysis:
           - Extract key information and themes
           - Identify relationships between different documents
           - Compare information across multiple sources
           - Highlight contradictions or inconsistencies
           - Provide source-based evidence
           - Consider document context and purpose
        
        4. For Visualization Requests:
           - Choose appropriate visualization types based on:
             * Data types and relationships
             * Number of variables
             * Purpose of visualization
             * Audience needs
           - Consider multiple visualization options
           - Explain visualization choices
           - Provide insights from the visualization
        
        5. For Comparative Analysis:
           - Compare multiple entities or concepts
           - Identify similarities and differences
           - Provide quantitative and qualitative comparisons
           - Consider multiple dimensions of comparison
           - Highlight significant differences
        
        6. For Predictive Analysis:
           - Consider historical patterns
           - Account for relevant variables
           - Provide confidence intervals
           - Explain assumptions made
           - Consider potential biases
        
        7. For Complex Search:
           - Use multiple search strategies
           - Consider synonyms and related terms
           - Handle ambiguous queries
           - Provide alternative interpretations
           - Rank results by relevance
        
        8. Response Format:
           - Start with a clear, direct answer
           - Provide detailed explanation
           - Include relevant examples
           - Use bullet points for clarity
           - Highlight key findings
           - Provide confidence levels
           - Suggest follow-up questions
        
        9. Error Handling:
           - Acknowledge limitations
           - Provide alternative approaches
           - Explain why certain information is unavailable
           - Suggest related information
           - Offer to refine the query
        
        10. Context Management:
            - Maintain conversation context
            - Reference previous interactions
            - Build on previous insights
            - Update understanding based on new information
            - Track changes in user's focus
        
        Context: {context}
        Web Search Results: {web_search}
        Question: {question}
        
        Answer:
        """

        try:
            model_config = self.get_model_config()
            model = ChatGoogleGenerativeAI(
                model=model_config["model"],
                temperature=model_config["temperature"],
                google_api_key=GOOGLE_API_KEY,
            )

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["memory_context", "context", "web_search", "question"],
            )
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            return chain
        except Exception as e:
            raise Exception(f"Error creating conversational chain: {e}")

    def get_prompt_for_file(self, file_type, analysis_type="summary"):
        """Get the appropriate prompt for a given file type and analysis type."""
        # Check for exact file type match
        if file_type in self.file_prompts:
            return self.file_prompts[file_type].get(
                analysis_type, self.universal_prompt
            )

        # Check for partial match (e.g., for image types)
        for key in self.file_prompts:
            if file_type.startswith(key):
                return self.file_prompts[key].get(analysis_type, self.universal_prompt)

        # Return universal prompt if no specific prompt is found
        return self.universal_prompt

    async def process_user_input_async(self, user_question, recent_file_info=None):
        try:
            memory_context = self.get_relevant_context(user_question)

            # First, check if we have uploaded files to analyze
            if st.session_state.uploaded_files:
                # Get the most recent file
                recent_file = st.session_state.uploaded_files[-1]

                # Handle image analysis
                if recent_file.type.startswith("image/"):
                    try:
                        analysis = self.process_image(recent_file)
                        return analysis
                    except Exception as e:
                        st.warning(f"Could not analyze image: {str(e)}")

                # Handle document analysis
                elif recent_file.type in [
                    "application/pdf",
                    "text/plain",
                    "application/vnd.ms-excel",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "text/csv",
                ]:
                    try:
                        # Get the content from the file
                        content = self.get_text_from_file(recent_file)

                        # Determine if this is a summary or analysis request
                        analysis_type = (
                            "analysis"
                            if any(
                                word in user_question.lower()
                                for word in [
                                    "analyze",
                                    "analysis",
                                    "examine",
                                    "evaluate",
                                    "assess",
                                ]
                            )
                            else "summary"
                        )

                        # Get the appropriate prompt for the file type
                        prompt_template = self.get_prompt_for_file(
                            recent_file.type, analysis_type
                        )

                        # Format the prompt with the content
                        prompt = prompt_template.format(content=content)

                        # Get model configuration
                        model_config = self.get_model_config()
                        model = ChatGoogleGenerativeAI(
                            model=model_config["model"],
                            temperature=0.3,  # Lower temperature for more focused responses
                            google_api_key=GOOGLE_API_KEY,
                        )

                        # Generate response
                        response = model.invoke(prompt)
                        return response.content

                    except Exception as e:
                        st.warning(f"Could not analyze document: {str(e)}")

            # Enhanced query processing for other cases
            query_type = self._analyze_query_type(user_question)

            # Handle different types of queries
            if query_type == "visualization":
                return await self._handle_visualization_query(
                    user_question, recent_file_info
                )
            elif query_type == "comparison":
                return await self._handle_comparison_query(
                    user_question, recent_file_info
                )
            elif query_type == "prediction":
                return await self._handle_prediction_query(
                    user_question, recent_file_info
                )
            elif query_type == "analysis":
                return await self._handle_analysis_query(
                    user_question, recent_file_info
                )

            # Regular processing for other queries
            vector_store = self.load_vector_store()
            if not vector_store:
                # Only perform web search if no local content is available
                try:
                    web_results = await self.search_web_async(user_question)
                    web_context = "\n".join(
                        [
                            f"Source {i+1}: {result['body']}"
                            for i, result in enumerate(web_results)
                        ]
                    )

                    if recent_file_info:
                        web_context += (
                            f"\nMost recently uploaded file: {recent_file_info}"
                        )

                    chain = self.get_conversational_chain()
                    if not chain:
                        raise Exception("Failed to create conversational chain")

                    memory_context_str = "\n".join(
                        [
                            f"Previous interaction at {ctx['timestamp']}: {ctx['response']}"
                            for ctx in memory_context
                        ]
                    )

                    response = chain(
                        {
                            "input_documents": [],
                            "memory_context": memory_context_str,
                            "web_search": web_context,
                            "question": user_question,
                        },
                        return_only_outputs=True,
                    )

                    context = {
                        "document_info": recent_file_info,
                        "web_search": web_context,
                        "timestamp": datetime.datetime.now(),
                    }
                    self.update_memory(user_question, response["output_text"], context)

                    return response["output_text"]
                except Exception as e:
                    raise Exception(f"Error during web search: {str(e)}")

            docs = vector_store.similarity_search(user_question)
            if not docs:
                raise Exception("No relevant content found in the processed files.")

            chain = self.get_conversational_chain()
            if not chain:
                raise Exception("Failed to create conversational chain")

            memory_context_str = "\n".join(
                [
                    f"Previous interaction at {ctx['timestamp']}: {ctx['response']}"
                    for ctx in memory_context
                ]
            )

            response = chain(
                {
                    "input_documents": docs,
                    "memory_context": memory_context_str,
                    "web_search": "",  # Don't use web search for local content
                    "question": user_question,
                },
                return_only_outputs=True,
            )

            context = {
                "document_info": recent_file_info,
                "timestamp": datetime.datetime.now(),
            }
            self.update_memory(user_question, response["output_text"], context)

            return response["output_text"]

        except Exception as e:
            raise Exception(f"Error during user input processing: {e}")

    def _analyze_query_type(self, question):
        """Analyze the type of query to determine appropriate processing."""
        question = question.lower()

        # Visualization queries
        if any(
            word in question
            for word in [
                "graph",
                "plot",
                "chart",
                "visualize",
                "visualization",
                "show",
                "display",
            ]
        ):
            return "visualization"

        # Comparison queries
        if any(
            word in question
            for word in [
                "compare",
                "difference",
                "similar",
                "versus",
                "vs",
                "versus",
                "contrast",
            ]
        ):
            return "comparison"

        # Prediction queries
        if any(
            word in question
            for word in ["predict", "forecast", "future", "trend", "will", "going to"]
        ):
            return "prediction"

        # Analysis queries
        if any(
            word in question
            for word in [
                "analyze",
                "analysis",
                "pattern",
                "trend",
                "correlation",
                "relationship",
            ]
        ):
            return "analysis"

        return "general"

    async def _handle_visualization_query(self, question, recent_file_info):
        """Handle visualization-specific queries."""
        if hasattr(self, "current_dataframe") and self.current_dataframe is not None:
            df = self.current_dataframe
            columns = df.columns.tolist()

            # Extract visualization parameters from the question
            viz_params = self._extract_visualization_params(question, columns)

            # Generate the graph
            graph_data = self.generate_graph(df, viz_params["columns"])

            if graph_data:
                context = {
                    "document_info": recent_file_info,
                    "graph_data": graph_data,
                    "timestamp": datetime.datetime.now(),
                }

                response = f"I've created a {viz_params['type']} graph showing {viz_params['description']}. The graph is displayed below."

                self.update_memory(question, response, context)
                return response
            else:
                raise Exception("Failed to generate graph")
        else:
            raise Exception("No data available for visualization")

    def _extract_visualization_params(self, question, available_columns):
        """Extract visualization parameters from the question."""
        question = question.lower()

        # Determine graph type
        if "line" in question:
            graph_type = "line"
        elif "bar" in question:
            graph_type = "bar"
        elif "scatter" in question:
            graph_type = "scatter"
        elif "pie" in question:
            graph_type = "pie"
        elif "histogram" in question:
            graph_type = "histogram"
        else:
            graph_type = "auto"  # Will be determined by data characteristics

        # Extract column mentions
        mentioned_columns = [
            col for col in available_columns if col.lower() in question
        ]

        # If no columns mentioned, use default columns
        if not mentioned_columns:
            mentioned_columns = (
                available_columns[:2]
                if len(available_columns) >= 2
                else available_columns
            )

        return {
            "type": graph_type,
            "columns": mentioned_columns,
            "description": f"the relationship between {', '.join(mentioned_columns)}",
        }

    async def _handle_comparison_query(self, question, recent_file_info):
        """Handle comparison-specific queries."""
        # Implementation for comparison queries
        pass

    async def _handle_prediction_query(self, question, recent_file_info):
        """Handle prediction-specific queries."""
        # Implementation for prediction queries
        pass

    async def _handle_analysis_query(self, question, recent_file_info):
        """Handle analysis-specific queries."""
        # Implementation for analysis queries
        pass

    def process_user_input(self, user_question, recent_file_info=None):
        return asyncio.run(
            self.process_user_input_async(user_question, recent_file_info)
        )

    def _analyze_python_code(self, code):
        """Analyze Python code."""
        try:
            # Basic Python code analysis
            analysis = {
                "imports": [],
                "functions": [],
                "classes": [],
                "variables": [],
                "docstrings": [],
            }

            # Parse imports
            for line in code.split("\n"):
                if line.strip().startswith("import ") or line.strip().startswith(
                    "from "
                ):
                    analysis["imports"].append(line.strip())
                elif line.strip().startswith("def "):
                    analysis["functions"].append(line.strip())
                elif line.strip().startswith("class "):
                    analysis["classes"].append(line.strip())
                elif "=" in line and not line.strip().startswith(
                    ("#", "def ", "class ")
                ):
                    analysis["variables"].append(line.strip())
                elif '"""' in line or "'''" in line:
                    analysis["docstrings"].append(line.strip())

            return analysis
        except Exception as e:
            return {"error": str(e)}

    def _analyze_javascript_code(self, code):
        """Analyze JavaScript code."""
        try:
            analysis = {
                "imports": [],
                "functions": [],
                "classes": [],
                "variables": [],
                "comments": [],
            }

            for line in code.split("\n"):
                if line.strip().startswith("import ") or line.strip().startswith(
                    "require("
                ):
                    analysis["imports"].append(line.strip())
                elif "function" in line or "=>" in line:
                    analysis["functions"].append(line.strip())
                elif line.strip().startswith("class "):
                    analysis["classes"].append(line.strip())
                elif "=" in line and not line.strip().startswith(
                    ("//", "/*", "function", "class")
                ):
                    analysis["variables"].append(line.strip())
                elif line.strip().startswith("//") or line.strip().startswith("/*"):
                    analysis["comments"].append(line.strip())

            return analysis
        except Exception as e:
            return {"error": str(e)}

    # Add similar analysis methods for other languages
    def _analyze_java_code(self, code):
        """Analyze Java code."""
        try:
            analysis = {
                "imports": [],
                "classes": [],
                "methods": [],
                "variables": [],
                "comments": [],
            }

            for line in code.split("\n"):
                if line.strip().startswith("import "):
                    analysis["imports"].append(line.strip())
                elif line.strip().startswith(
                    "public class "
                ) or line.strip().startswith("class "):
                    analysis["classes"].append(line.strip())
                elif (
                    "(" in line
                    and ")" in line
                    and "{" in line
                    and not line.strip().startswith(("//", "/*", "import", "class"))
                ):
                    analysis["methods"].append(line.strip())
                elif "=" in line and not line.strip().startswith(
                    ("//", "/*", "import", "class")
                ):
                    analysis["variables"].append(line.strip())
                elif line.strip().startswith("//") or line.strip().startswith("/*"):
                    analysis["comments"].append(line.strip())

            return analysis
        except Exception as e:
            return {"error": str(e)}

    # Add placeholder methods for other languages
    def _analyze_cpp_code(self, code):
        return {"error": "Not implemented"}

    def _analyze_csharp_code(self, code):
        return {"error": "Not implemented"}

    def _analyze_php_code(self, code):
        return {"error": "Not implemented"}

    def _analyze_ruby_code(self, code):
        return {"error": "Not implemented"}

    def _analyze_go_code(self, code):
        return {"error": "Not implemented"}

    def _analyze_rust_code(self, code):
        return {"error": "Not implemented"}

    def _analyze_swift_code(self, code):
        return {"error": "Not implemented"}

    def _analyze_kotlin_code(self, code):
        return {"error": "Not implemented"}

    def _analyze_scala_code(self, code):
        return {"error": "Not implemented"}

    def _analyze_html_code(self, code):
        return {"error": "Not implemented"}

    def _analyze_css_code(self, code):
        return {"error": "Not implemented"}

    def _analyze_sql_code(self, code):
        return {"error": "Not implemented"}

    def _analyze_shell_code(self, code):
        return {"error": "Not implemented"}

    def _analyze_markdown_code(self, code):
        return {"error": "Not implemented"}

    def _analyze_json_code(self, code):
        return {"error": "Not implemented"}

    def _analyze_xml_code(self, code):
        return {"error": "Not implemented"}

    def _analyze_yaml_code(self, code):
        return {"error": "Not implemented"}

    def _analyze_toml_code(self, code):
        return {"error": "Not implemented"}

    def _analyze_ini_code(self, code):
        return {"error": "Not implemented"}

    def _analyze_env_code(self, code):
        return {"error": "Not implemented"}

    def analyze_code(self, code, language):
        """Analyze code based on language."""
        try:
            # Get the appropriate analysis function
            analysis_func = self.code_analysis_tools.get(language.lower())
            if analysis_func:
                return analysis_func(code)
            else:
                return {"error": f"Language {language} not supported"}
        except Exception as e:
            return {"error": str(e)}

    def _get_language_from_extension(self, extension):
        """Get programming language from file extension."""
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "cpp",
            ".cs": "csharp",
            ".php": "php",
            ".rb": "ruby",
            ".go": "go",
            ".rs": "rust",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".html": "html",
            ".css": "css",
            ".sql": "sql",
            ".sh": "shell",
            ".md": "markdown",
            ".json": "json",
            ".xml": "xml",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".ini": "ini",
            ".env": "env",
        }
        return extension_map.get(extension, "unknown")

    def process_audio_input(self, audio_file, operation=None):
        """Process audio input based on user's chosen operation."""
        try:
            if operation not in ["transcribe", "noise_reduction", "normalize"]:
                return {
                    "error": "Please choose one of the following operations:\n1. transcribe\n2. noise_reduction\n3. normalize"
                }

            # Load audio file
            audio = AudioSegment.from_file(audio_file)
            results = {}

            if operation == "transcribe":
                # Transcribe audio to text
                with sr.AudioFile(audio_file) as source:
                    audio_data = self.recognizer.record(source)
                    try:
                        text = self.recognizer.recognize_google(audio_data)
                        results["transcription"] = text
                        # Store transcribed text for future use
                        self.last_transcription = text
                    except sr.UnknownValueError:
                        results["error"] = "Could not understand audio"
                    except sr.RequestError as e:
                        results["error"] = (
                            f"Error with speech recognition service: {str(e)}"
                        )

            elif operation == "noise_reduction":
                # Convert to numpy array for processing
                samples = np.array(audio.get_array_of_samples())
                sample_rate = audio.frame_rate
                samples_float = samples.astype(np.float32) / np.iinfo(samples.dtype).max
                reduced_noise = nr.reduce_noise(y=samples_float, sr=sample_rate)
                samples = (reduced_noise * np.iinfo(samples.dtype).max).astype(
                    samples.dtype
                )
                processed_audio = AudioSegment(
                    samples.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=samples.dtype.itemsize,
                    channels=1,
                )
                results["processed_audio"] = processed_audio
                results["message"] = "Audio noise has been reduced"

            elif operation == "normalize":
                # Normalize audio volume
                normalized_audio = audio.normalize()
                results["processed_audio"] = normalized_audio
                results["message"] = "Audio has been normalized"

            return results

        except Exception as e:
            return {"error": f"Error processing audio: {str(e)}"}

    def text_to_speech(self, text, language="en", output_file="response.mp3"):
        """Convert text to speech."""
        try:
            # Create gTTS object
            tts = gTTS(text=text, lang=language, slow=False)

            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_file.name)

            # Process audio (optional enhancements)
            audio = AudioSegment.from_mp3(temp_file.name)

            # Apply audio enhancements
            audio = audio.normalize()  # Normalize volume
            audio = audio.set_frame_rate(44100)  # High-quality sample rate

            # Export enhanced audio
            audio.export(output_file, format="mp3")

            # Clean up temporary file
            os.unlink(temp_file.name)

            return output_file

        except Exception as e:
            print(f"Error converting text to speech: {str(e)}")
            return None

    def should_respond_with_audio(self, user_input):
        """Determine if response should be in audio format."""
        # Keywords that might indicate audio response is preferred
        audio_keywords = [
            "speak",
            "say",
            "tell me",
            "pronounce",
            "read",
            "narrate",
            "voice",
            "audio",
            "sound",
            "listen",
            "hear",
        ]

        # Check if any audio keywords are in the user input
        return any(keyword in user_input.lower() for keyword in audio_keywords)

    def process_user_query(self, query, audio_input=None, audio_operation=None):
        """Process user query with optional audio input."""
        try:
            # If audio input is provided, process it first
            if audio_input:
                if not audio_operation:
                    return {
                        "text": "Please choose one of the following operations for the audio:\n1. transcribe\n2. noise_reduction\n3. normalize",
                        "format": "text",
                    }

                audio_results = self.process_audio_input(audio_input, audio_operation)

                if "error" in audio_results:
                    return {"text": audio_results["error"], "format": "text"}

                if audio_operation == "transcribe":
                    if "transcription" in audio_results:
                        # Use transcribed text for further processing
                        query = audio_results["transcription"]
                        response = self.model.invoke(query)
                        return {
                            "text": f"Transcribed Text: {query}\n\nResponse: {response.content}",
                            "format": "text",
                        }
                    else:
                        return {"text": "Failed to transcribe audio", "format": "text"}

                elif audio_operation in ["noise_reduction", "normalize"]:
                    if "processed_audio" in audio_results:
                        return {
                            "text": audio_results["message"],
                            "audio": audio_results["processed_audio"],
                            "format": "audio",
                        }
                    else:
                        return {
                            "text": f"Failed to {audio_operation} audio",
                            "format": "text",
                        }

            # If no audio input, process as regular text query
            response = self.model.invoke(query)
            return {"text": response.content, "format": "text"}

        except Exception as e:
            return {"text": f"Error processing query: {str(e)}", "format": "text"}

    def analyze_audio(self, audio_file):
        """Analyze audio file and extract features."""
        try:
            # Load audio file
            y, sr = librosa.load(audio_file)

            analysis = {
                "duration": librosa.get_duration(y=y, sr=sr),
                "sample_rate": sr,
                "tempo": librosa.beat.tempo(y=y, sr=sr)[0],
                "mean_amplitude": np.mean(np.abs(y)),
                "max_amplitude": np.max(np.abs(y)),
                "zero_crossings": librosa.zero_crossings(y).sum(),
            }

            # Extract pitch
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            analysis["pitch_mean"] = np.mean(pitches[pitches > 0])

            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            analysis["spectral_centroid_mean"] = np.mean(spectral_centroids)

            return analysis

        except Exception as e:
            print(f"Error analyzing audio: {str(e)}")
            return None
