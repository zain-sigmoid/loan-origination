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
from scipy.stats import gaussian_kde
from .prompts import generate_dynamic_prompt, FILE_PROMPTS, CONVERSATIONAL_CHAIN_PROMPT
import random

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("LLM_API_KEY")

# Define available Gemini models
GEMINI_MODELS = {
    'gemini-2.0-flash': {
        'model': 'gemini-2.0-flash',
        'temperature': 0.3,
        'max_retries': 3,
        'chunk_size': 12000,
        'chunk_overlap': 1200
    },
    'gemini-1.5-flash': {
        'model': 'gemini-1.5-flash',
        'temperature': 0.3,
        'max_retries': 3,
        'chunk_size': 10000,
        'chunk_overlap': 1000
    },
    'gemini-1.5-pro': {
        'model': 'gemini-1.5-pro',
        'temperature': 0.3,
        'max_retries': 3,
        'chunk_size': 15000,
        'chunk_overlap': 1500
    },
    'gemini-pro': {
        'model': 'gemini-pro',
        'temperature': 0.3,
        'max_retries': 3,
        'chunk_size': 12000,
        'chunk_overlap': 1200
    }
}

# Add constants at the top of the file after imports
SUPPORTED_LANGUAGES = {
    'python': '_analyze_python_code',
    'javascript': '_analyze_javascript_code',
    'java': '_analyze_java_code',
    'cpp': '_analyze_cpp_code',
    'csharp': '_analyze_csharp_code',
    'php': '_analyze_php_code',
    'ruby': '_analyze_ruby_code',
    'go': '_analyze_go_code',
    'rust': '_analyze_rust_code',
    'swift': '_analyze_swift_code',
    'kotlin': '_analyze_kotlin_code',
    'scala': '_analyze_scala_code',
    'html': '_analyze_html_code',
    'css': '_analyze_css_code',
    'sql': '_analyze_sql_code',
    'shell': '_analyze_shell_code',
    'markdown': '_analyze_markdown_code',
    'json': '_analyze_json_code',
    'xml': '_analyze_xml_code',
    'yaml': '_analyze_yaml_code',
    'toml': '_analyze_toml_code',
    'ini': '_analyze_ini_code',
    'env': '_analyze_env_code'
}

QUERY_TYPES = {
    'visualization': ['graph', 'plot', 'chart', 'visualize', 'visualization', 'show', 'display'],
    'comparison': ['compare', 'difference', 'similar', 'versus', 'vs', 'contrast'],
    'prediction': ['predict', 'forecast', 'future', 'trend', 'will', 'going to'],
    'analysis': ['analyze', 'analysis', 'pattern', 'trend', 'correlation', 'relationship']
}

GRAPH_TYPES = {
    'line': ['line'],
    'bar': ['bar'],
    'scatter': ['scatter'],
    'pie': ['pie'],
    'histogram': ['histogram'],
    'box': ['box', 'boxplot'],
    'violin': ['violin'],
    'heatmap': ['heat', 'correlation']
}

# Add more constants at the top after the existing ones
FILE_TYPE_HANDLERS = {
    'application/pdf': 'handle_pdf',
    'text/plain': 'handle_text',
    'application/vnd.ms-excel': 'handle_excel',
    'text/csv': 'handle_csv',
    'image/': 'handle_image'
}

ERROR_MESSAGES = {
    'api_key': "Error: Invalid or missing Google API key for vision model",
    'quota': "Error: API quota exceeded for vision model",
    'permission': "Error: Permission denied for vision model",
    'invalid_image': "Error: The uploaded file is not a valid image or is corrupted",
    'file_size': lambda size: f"File size ({size / (1024 * 1024):.2f}MB) exceeds the maximum allowed size of 4MB",
    'chain_creation': "Failed to create conversational chain",
    'no_content': "No content could be extracted from the provided sources.",
    'empty_content': "The content might be empty or unreadable.",
    'vector_store': "Failed to create vector store from the content."
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

        # Initialize code analysis tools using the SUPPORTED_LANGUAGES constant
        self.code_analysis_tools = {
            lang: getattr(self, method_name)
            for lang, method_name in SUPPORTED_LANGUAGES.items()
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

    def process_image(self, image_file, question=None):
        """Process an image file and analyze it based on the question asked."""
        try:
            # Handle both file-like objects and PIL Image objects
            if hasattr(image_file, 'getvalue'):
                # If it's a file-like object (e.g., from Streamlit)
                image_bytes = io.BytesIO(image_file.getvalue())
            elif isinstance(image_file, Image.Image):
                # If it's already a PIL Image
                image_bytes = io.BytesIO()
                image_file.save(image_bytes, format=image_file.format or 'PNG')
                image_bytes.seek(0)
            else:
                # If it's raw bytes
                image_bytes = io.BytesIO(image_file)

            # Verify image data is valid
            try:
                image = Image.open(image_bytes)
                image.verify()  # Verify it's a valid image
                image_bytes.seek(0)  # Reset buffer position
                image = Image.open(image_bytes)  # Reopen after verify
            except Exception as e:
                raise Exception(f"Invalid image file: {str(e)}")

            if getattr(image_file, 'type', '').startswith('application/pdf'):
                try:
                    images = convert_from_bytes(image_bytes.getvalue())
                    text = ""
                    for idx, img in enumerate(images):
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        img_byte_arr.seek(0)

                        image_parts = [{"mime_type": "image/png", "data": img_byte_arr.getvalue()}]

                        # Generate a focused prompt based on the user's question
                        prompt = self._generate_image_analysis_prompt(question)

                        try:
                            response = self.vision_model.generate_content([prompt, image_parts[0]])
                            if response and response.text:
                                text += f"Page {idx + 1} Analysis:\n{response.text}\n\n"
                            else:
                                text += f"Page {idx + 1}: Vision model could not analyze. Trying OCR...\n"
                                ocr_text = pytesseract.image_to_string(img)
                                if ocr_text.strip():
                                    text += f"OCR Text from Page {idx + 1}:\n{ocr_text}\n\n"
                                else:
                                    text += f"Page {idx + 1}: No text could be extracted.\n\n"
                        except Exception as e:
                            text += f"Page {idx + 1}: Error in vision analysis: {str(e)}\n"
                            try:
                                ocr_text = pytesseract.image_to_string(img)
                                if ocr_text.strip():
                                    text += f"OCR Text from Page {idx + 1}:\n{ocr_text}\n\n"
                            except Exception as ocr_error:
                                text += f"Page {idx + 1}: OCR also failed: {str(ocr_error)}\n\n"
                except Exception as e:
                    raise Exception(f"Error processing PDF: {str(e)}")
            else:
                # For single images
                # Convert to bytes for the vision model
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format or 'PNG')
                img_byte_arr.seek(0)

                image_parts = [{
                    "mime_type": getattr(image_file, 'type', 'image/png'),
                    "data": img_byte_arr.getvalue()
                }]

                # Generate a focused prompt based on the user's question
                prompt = self._generate_image_analysis_prompt(question)

                try:
                    response = self.vision_model.generate_content([prompt, image_parts[0]])
                    if response and response.text:
                        text = response.text
                    else:
                        raise Exception("Vision model returned empty response")
                except Exception as e:
                    # If vision model fails, try OCR
                    try:
                        text = pytesseract.image_to_string(image)
                        if not text.strip():
                            raise Exception("OCR produced no text")
                        text = f"Vision model analysis failed. OCR Text:\n{text}"
                    except Exception as ocr_error:
                        raise Exception(
                            f"Both vision model and OCR failed. Vision error: {str(e)}, OCR error: {str(ocr_error)}"
                        )

            if not text.strip():
                raise Exception("No content could be extracted from the image using any method")

            return text
        except Exception as e:
            error_msg = str(e)
            if "API key" in error_msg:
                error_msg = ERROR_MESSAGES['api_key']
            elif "quota" in error_msg.lower():
                error_msg = ERROR_MESSAGES['quota']
            elif "permission" in error_msg.lower():
                error_msg = ERROR_MESSAGES['permission']
            elif "invalid image" in error_msg.lower():
                error_msg = ERROR_MESSAGES['invalid_image']
            raise Exception(f"Error processing image: {error_msg}")

    def _generate_image_analysis_prompt(self, question):
        """Generate a focused prompt for image analysis based on the user's question."""
        if not question:
            return "Describe what you see in this image."
        
        question = question.lower().strip()
        return f"Look at this image and answer this specific question: {question}"

    def check_file_size(self, file):
        """Check if the file size is within the allowed limit."""
        file_size = len(file.getvalue())
        if file_size > self.max_file_size:
            raise Exception(
                f"File size ({file_size / (1024 * 1024):.2f}MB) exceeds the maximum allowed size of 4MB"
            )
        file.seek(0)  # Reset file pointer to beginning
        return True

    def handle_file_type(self, file_type, file_content):
        """Centralized file type handling method."""
        # Check for exact file type match
        handler = None
        for type_pattern, handler_name in FILE_TYPE_HANDLERS.items():
            if file_type.startswith(type_pattern):
                handler = getattr(self, handler_name, None)
                break
        
        if handler:
            return handler(file_content)
        
        # If no specific handler found, try to handle as text
        try:
            return file_content.decode('utf-8')
        except Exception as e:
            return self.handle_error(e)

    def get_text_from_file(self, file):
        """Get text content from a file."""
        try:
            # Check file size
            self.check_file_size(file)
            
            return self.handle_file_type(file.type, file.getvalue())
            
        except Exception as e:
            raise Exception(self.handle_error(e))

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
            
            return self._create_graph(graph_type, data, columns)
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
        if any(col.lower() in ['date', 'time', 'year', 'month', 'day'] for col in columns):
            return "line"
            
        # Check for categorical vs numerical relationship
        categorical_cols = [col for col in columns if dtypes[col] == 'object' or dtypes[col] == 'category']
        numerical_cols = [col for col in columns if pd.api.types.is_numeric_dtype(dtypes[col])]
        
        # Enhanced graph type selection
        if len(numerical_cols) >= 2:
            # Check correlation for scatter plot
            if len(numerical_cols) == 2:
                correlation = abs(data[numerical_cols].corr().iloc[0, 1])
                if correlation > 0.3:  # If there's meaningful correlation
                    return "scatter"
            return "scatter"
        elif len(categorical_cols) == 1 and len(numerical_cols) == 1:
            if len(data[categorical_cols[0]].unique()) <= 10:
                return "bar"
            else:
                return "box"  # Use box plot for many categories
        elif len(categorical_cols) == 1:
            if len(data[categorical_cols[0]].unique()) <= 10:
                return "pie"
            else:
                return "bar"
        elif len(numerical_cols) == 1:
            return "histogram"
        
        return "bar"  # Default to bar graph

    def _create_graph(self, graph_type, data, columns):
        """Create a graph based on the specified type."""
        plt.figure(figsize=(12, 8))
        plt.style.use('seaborn')  # Use seaborn style for better aesthetics
        
        graph_creators = {
            'line': self._create_line_graph,
            'bar': self._create_bar_graph,
            'scatter': self._create_scatter_plot,
            'pie': self._create_pie_chart,
            'histogram': self._create_histogram,
            'box': self._create_box_plot,
            'violin': self._create_violin_plot,
            'heatmap': self._create_heatmap
        }
        
        creator = graph_creators.get(graph_type)
        if creator:
            creator(data, columns)
        else:
            plt.close()
            return None
        
        plt.tight_layout()
        return self._save_plot_to_base64()

    def _save_plot_to_base64(self):
        """Save the current plot to base64 string."""
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        return f"data:image/png;base64,{img_str}"

    def _create_line_graph(self, data, columns):
        """Create an enhanced line graph."""
        x_col = columns[0]
        y_col = columns[1]
        
        # Create line plot with markers and styling
        plt.plot(data[x_col], data[y_col], marker='o', linestyle='-', linewidth=2, markersize=6)
        
        # Add trend line
        z = np.polyfit(range(len(data[x_col])), data[y_col], 1)
        p = np.poly1d(z)
        plt.plot(data[x_col], p(range(len(data[x_col]))), "r--", alpha=0.8, label='Trend')
        
        plt.title(f"{y_col} vs {x_col}", pad=20, fontsize=14, fontweight='bold')
        plt.xlabel(x_col, fontsize=12)
        plt.ylabel(y_col, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Rotate x-axis labels if they're too long
        if max([len(str(x)) for x in data[x_col]]) > 10:
            plt.xticks(rotation=45, ha='right')

    def _create_bar_graph(self, data, columns):
        """Create an enhanced bar graph."""
        x_col = columns[0]
        y_col = columns[1]
        
        # Create bar plot with styling
        bars = plt.bar(data[x_col], data[y_col], color='skyblue', edgecolor='black')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.0f}',
                    ha='center', va='bottom')
        
        plt.title(f"{y_col} by {x_col}", pad=20, fontsize=14, fontweight='bold')
        plt.xlabel(x_col, fontsize=12)
        plt.ylabel(y_col, fontsize=12)
        
        # Rotate x-axis labels if they're too long
        if max([len(str(x)) for x in data[x_col]]) > 10:
            plt.xticks(rotation=45, ha='right')

    def _create_scatter_plot(self, data, columns):
        """Create an enhanced scatter plot."""
        x_col = columns[0]
        y_col = columns[1]
        
        # Create scatter plot
        plt.scatter(data[x_col], data[y_col], alpha=0.6, c='blue', edgecolor='white')
        
        # Add trend line
        z = np.polyfit(data[x_col], data[y_col], 1)
        p = np.poly1d(z)
        plt.plot(data[x_col], p(data[x_col]), "r--", alpha=0.8, label='Trend')
        
        # Add correlation coefficient
        corr = data[columns].corr().iloc[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {corr:.2f}', 
                transform=plt.gca().transAxes, fontsize=10)
        
        plt.title(f"{y_col} vs {x_col}", pad=20, fontsize=14, fontweight='bold')
        plt.xlabel(x_col, fontsize=12)
        plt.ylabel(y_col, fontsize=12)
        plt.legend()

    def _create_pie_chart(self, data, columns):
        """Create an enhanced pie chart."""
        x_col = columns[0]
        y_col = columns[1]
        
        # Calculate percentages
        total = data[y_col].sum()
        percentages = [f'{(val/total)*100:.1f}%' for val in data[y_col]]
        
        # Create pie chart with styling
        plt.pie(data[y_col], labels=data[x_col], autopct='%1.1f%%',
                colors=plt.cm.Pastel1(np.linspace(0, 1, len(data[x_col]))),
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        
        plt.title(f"Distribution of {y_col}", pad=20, fontsize=14, fontweight='bold')
        
        # Add legend if there are many categories
        if len(data[x_col]) > 5:
            plt.legend(data[x_col], title=x_col, bbox_to_anchor=(1.05, 1), loc='upper left')

    def _create_histogram(self, data, columns):
        """Create an enhanced histogram."""
        col = columns[0]
        
        # Create histogram with KDE
        plt.hist(data[col], bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add KDE plot
        density = gaussian_kde(data[col])
        xs = np.linspace(data[col].min(), data[col].max(), 200)
        plt.plot(xs, density(xs), 'r-', lw=2, label='KDE')
        
        plt.title(f"Distribution of {col}", pad=20, fontsize=14, fontweight='bold')
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        
        # Add summary statistics
        mean = data[col].mean()
        median = data[col].median()
        std = data[col].std()
        stats_text = f'Mean: {mean:.2f}\nMedian: {median:.2f}\nStd: {std:.2f}'
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def _create_box_plot(self, data, columns):
        """Create a box plot."""
        plt.boxplot([data[col] for col in columns], labels=columns)
        plt.title("Box Plot of " + " vs ".join(columns), pad=20, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45 if max([len(col) for col in columns]) > 10 else 0)
        plt.grid(True, linestyle='--', alpha=0.7)

    def _create_violin_plot(self, data, columns):
        """Create a violin plot."""
        plt.violinplot([data[col] for col in columns])
        plt.title("Violin Plot of " + " vs ".join(columns), pad=20, fontsize=14, fontweight='bold')
        plt.xticks(range(1, len(columns) + 1), columns,
                  rotation=45 if max([len(col) for col in columns]) > 10 else 0)
        plt.grid(True, linestyle='--', alpha=0.7)

    def _create_heatmap(self, data, columns):
        """Create a correlation heatmap."""
        corr = data[columns].corr()
        plt.imshow(corr, cmap='coolwarm', aspect='auto')
        plt.colorbar(label='Correlation coefficient')
        plt.xticks(range(len(columns)), columns, rotation=45, ha='right')
        plt.yticks(range(len(columns)), columns)
        plt.title("Correlation Heatmap", pad=20, fontsize=14, fontweight='bold')
        
        # Add correlation values
        for i in range(len(columns)):
            for j in range(len(columns)):
                plt.text(j, i, f'{corr.iloc[i, j]:.2f}',
                        ha='center', va='center')

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

    def get_prompt_for_file(self, file_type, analysis_type="summary", question=None):
        """Get the appropriate prompt for a given file type and analysis type."""
        # Determine content type based on the question and file type
        content_type = self._determine_content_type(question, file_type)
        
        # Generate dynamic prompt based on question and content type
        dynamic_prompt = generate_dynamic_prompt(
            question=question or "Analyze this content",
            file_type=file_type,
            content_type=content_type
        )
        
        # For specific file types that need specialized prompts, combine with file-specific prompt
        if file_type in FILE_PROMPTS:
            file_prompt = FILE_PROMPTS[file_type].get(analysis_type, "")
            if file_prompt:
                # Extract guidelines from file prompt and add to dynamic prompt
                guidelines = self._extract_guidelines(file_prompt)
                dynamic_prompt = self._merge_prompts(dynamic_prompt, guidelines)
        
        return dynamic_prompt

    def _determine_content_type(self, question, file_type):
        """Determine the content type based on question and file type."""
        if not question:
            return None
        
        question = question.lower()
        
        # Technical content indicators
        technical_keywords = [
            "technical", "specification", "performance", "implementation",
            "code", "algorithm", "system", "architecture", "framework",
            "api", "database", "protocol", "configuration"
        ]
        
        # Business content indicators
        business_keywords = [
            "business", "market", "cost", "revenue", "profit", "strategy",
            "customer", "sales", "roi", "investment", "stakeholder",
            "competition", "pricing", "growth"
        ]
        
        # Check file type first
        if file_type in ["application/vnd.ms-excel", "text/csv"]:
            return "business" if any(word in question for word in business_keywords) else "technical"
        
        # Check question content
        if any(word in question for word in technical_keywords):
            return "technical"
        if any(word in question for word in business_keywords):
            return "business"
        
        return None

    def _extract_guidelines(self, file_prompt):
        """Extract guidelines from file-specific prompt."""
        guidelines = []
        lines = file_prompt.split("\n")
        in_guidelines = False
        
        for line in lines:
            if "Guidelines" in line:
                in_guidelines = True
                continue
            if in_guidelines and line.strip() and not line.strip().startswith("Please") and not line.strip().startswith("Content"):
                guidelines.append(line.strip())
        
        return guidelines

    def _merge_prompts(self, dynamic_prompt, guidelines):
        """Merge dynamic prompt with file-specific guidelines."""
        # Find the position to insert guidelines (before the response format section)
        insert_pos = dynamic_prompt.find("Please structure your response")
        if insert_pos == -1:
            insert_pos = len(dynamic_prompt)
        
        # Add file-specific guidelines
        if guidelines:
            guidelines_text = "\nAdditional Guidelines:\n" + "\n".join(guidelines) + "\n"
            merged_prompt = dynamic_prompt[:insert_pos] + guidelines_text + dynamic_prompt[insert_pos:]
            return merged_prompt
        
        return dynamic_prompt

    def _extract_topics(self, question):
        """Extract key topics from the question."""
        if not question:
            return []
        
        topics = []
        question = question.lower()
        
        # Check for specific analysis requests
        if "trend" in question or "pattern" in question:
            topics.append("Trend and pattern analysis")
        if "compare" in question or "comparison" in question:
            topics.append("Comparative analysis")
        if "risk" in question or "impact" in question:
            topics.append("Risk and impact assessment")
        if "recommend" in question or "suggest" in question:
            topics.append("Recommendations and suggestions")
        if "cost" in question or "price" in question or "value" in question:
            topics.append("Cost and value analysis")
        if "performance" in question or "efficiency" in question:
            topics.append("Performance metrics")
        if "future" in question or "predict" in question or "forecast" in question:
            topics.append("Future predictions and forecasts")
        if "problem" in question or "issue" in question or "challenge" in question:
            topics.append("Problem identification and solutions")
        
        return topics

    def _is_greeting(self, text):
        """Check if the input text is a greeting."""
        greetings = {
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
            'howdy', 'hola', 'namaste', 'greetings', 'sup', "what's up",
            'yo', 'hiya', 'morning', 'evening', 'afternoon'
        }
        
        # Clean and tokenize the input text
        text = text.lower().strip()
        words = set(text.split())
        
        # Check if any word is a greeting
        return bool(words & greetings)

    def _get_greeting_response(self):
        """Generate an appropriate greeting response based on time of day."""
        current_hour = datetime.datetime.now().hour
        
        # Get user's name if available in session state
        user_name = getattr(st.session_state, 'user_name', '')
        name_suffix = f", {user_name}" if user_name else ''
        
        if 5 <= current_hour < 12:
            responses = [
                f"Good morning{name_suffix}! How can I help you today?",
                f"Morning{name_suffix}! What can I assist you with?",
                f"Hello{name_suffix}! Hope you're having a great morning. How may I help?"
            ]
        elif 12 <= current_hour < 17:
            responses = [
                f"Good afternoon{name_suffix}! How can I assist you today?",
                f"Hi{name_suffix}! What can I help you with this afternoon?",
                f"Hello{name_suffix}! Hope you're having a good afternoon. What brings you here?"
            ]
        elif 17 <= current_hour < 22:
            responses = [
                f"Good evening{name_suffix}! How can I help you?",
                f"Evening{name_suffix}! What can I assist you with?",
                f"Hi{name_suffix}! How may I help you this evening?"
            ]
        else:
            responses = [
                f"Hello{name_suffix}! How can I help you?",
                f"Hi{name_suffix}! What can I assist you with?",
                f"Greetings{name_suffix}! How may I help you?"
            ]
        
        return random.choice(responses)

    async def process_user_input_async(self, user_question, recent_file_info=None):
        try:
            # Check if the input is a greeting
            if self._is_greeting(user_question):
                return self._get_greeting_response()

            memory_context = self.get_relevant_context(user_question)

            # First, check if we have uploaded files to analyze
            if st.session_state.uploaded_files:
                # Get the most recent file
                recent_file = st.session_state.uploaded_files[-1]

                # Handle image analysis
                if recent_file.type.startswith("image/"):
                    try:
                        analysis = self.process_image(recent_file, user_question)
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

                        # Get the appropriate prompt for the file type and question
                        prompt_template = self.get_prompt_for_file(
                            recent_file.type, 
                            "analysis" if self._is_analysis_question(user_question) else "summary",
                            user_question
                        )

                        # Format the prompt with the content and question
                        prompt = prompt_template.format(
                            content=content,
                            question=user_question
                        )

                        # Get model configuration with adjusted parameters for more dynamic responses
                        model_config = self.get_model_config()
                        model = ChatGoogleGenerativeAI(
                            model=model_config["model"],
                            temperature=0.7,  # Higher temperature for more creative responses
                            top_p=0.9,       # More diversity in responses
                            top_k=50,        # Consider more token options
                            google_api_key=GOOGLE_API_KEY,
                        )

                        # Generate response
                        response = model.invoke(prompt)
                        
                        # Post-process the response
                        processed_response = self._post_process_response(response.content, user_question)
                        return processed_response

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
            docs = None
            
            if vector_store:
                docs = vector_store.similarity_search(user_question)
            
            # Perform web search if no vector store or no relevant docs found
            if not vector_store or not docs:
                try:
                    # Inform user that we're searching the web
                    web_search_msg = "No relevant information found in the uploaded documents. Searching the web for answers..."
                    
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

                    # Prepend the web search message to the response
                    return web_search_msg + "\n\n" + response["output_text"]
                except Exception as e:
                    raise Exception(f"Error during web search: {str(e)}")

            # If we have relevant docs from vector store, use them
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

    def _post_process_response(self, response, question):
        """Post-process the response to ensure it's relevant to the question."""
        # Return the response directly without any prefix
        return response.strip()

    def _get_question_topic(self, question):
        """Extract the main topic from the question."""
        # Remove common question words
        common_words = ["what", "how", "why", "when", "where", "who", "can", "could", "would", "should", "is", "are", "was", "were"]
        words = question.lower().split()
        topic_words = [w for w in words if w not in common_words]
        
        # Return the first few meaningful words
        return " ".join(topic_words[:3]) + "..."

    def _analyze_query_type(self, question):
        """Analyze the type of query to determine appropriate processing."""
        question = question.lower()
        
        # Check each query type using the QUERY_TYPES constant
        for query_type, keywords in QUERY_TYPES.items():
            if any(word in question for word in keywords):
                return query_type
        
        return "general"

    async def _handle_visualization_query(self, question, recent_file_info):
        """Handle visualization-specific queries."""
        if hasattr(self, "current_dataframe") and self.current_dataframe is not None:
            df = self.current_dataframe
            columns = df.columns.tolist()
            
            # Extract visualization parameters from the question
            viz_params = self._extract_visualization_params(question, columns)
            
            # Validate column selection
            if not viz_params['columns']:
                return "I couldn't determine which columns to visualize. Please specify the columns you'd like to see in the graph."
            
            # Generate the graph
            graph_data = self.generate_graph(df, viz_params['columns'])
            
            if graph_data:
                context = {
                    'document_info': recent_file_info,
                    'graph_data': graph_data,
                    'timestamp': datetime.datetime.now()
                }
                
                # Enhanced response with more details
                response_parts = [
                    f"I've created a {viz_params['type']} graph showing {viz_params['description']}.",
                    "The graph is displayed below."
                ]
                
                # Add statistical insights based on graph type
                if viz_params['type'] == 'scatter':
                    corr = df[viz_params['columns']].corr().iloc[0, 1]
                    response_parts.append(f"\nThe correlation coefficient between the variables is {corr:.2f}.")
                elif viz_params['type'] in ['histogram', 'box', 'violin']:
                    for col in viz_params['columns']:
                        mean = df[col].mean()
                        median = df[col].median()
                        std = df[col].std()
                        response_parts.append(f"\nFor {col}:")
                        response_parts.append(f"- Mean: {mean:.2f}")
                        response_parts.append(f"- Median: {median:.2f}")
                        response_parts.append(f"- Standard Deviation: {std:.2f}")
                elif viz_params['type'] == 'pie':
                    total = df[viz_params['columns'][1]].sum()
                    largest_category = df.loc[df[viz_params['columns'][1]].idxmax(), viz_params['columns'][0]]
                    largest_value = df[viz_params['columns'][1]].max()
                    response_parts.append(f"\nThe largest category is '{largest_category}' with {(largest_value/total)*100:.1f}% of the total.")
                
                response = "\n".join(response_parts)
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
        if 'line' in question:
            graph_type = 'line'
        elif 'bar' in question:
            graph_type = 'bar'
        elif 'scatter' in question:
            graph_type = 'scatter'
        elif 'pie' in question:
            graph_type = 'pie'
        elif 'histogram' in question:
            graph_type = 'histogram'
        elif 'box' in question or 'boxplot' in question:
            graph_type = 'box'
        elif 'violin' in question:
            graph_type = 'violin'
        elif 'heat' in question or 'correlation' in question:
            graph_type = 'heatmap'
        else:
            graph_type = 'auto'  # Will be determined by data characteristics
        
        # Extract column mentions
        mentioned_columns = []
        for col in available_columns:
            # Check for exact matches
            if col.lower() in question:
                mentioned_columns.append(col)
            # Check for partial matches (e.g., "sales" matches "total_sales")
            elif any(word in col.lower() for word in question.split()):
                mentioned_columns.append(col)
        
        # If no columns mentioned, use intelligent defaults based on graph type
        if not mentioned_columns:
            if graph_type == 'heatmap':
                # For heatmap, use all numeric columns
                mentioned_columns = [col for col in available_columns 
                                  if pd.api.types.is_numeric_dtype(self.current_dataframe[col].dtype)]
                # Limit to top 10 columns if too many
                if len(mentioned_columns) > 10:
                    mentioned_columns = mentioned_columns[:10]
            elif graph_type in ['box', 'violin']:
                # For box and violin plots, use all numeric columns
                mentioned_columns = [col for col in available_columns 
                                  if pd.api.types.is_numeric_dtype(self.current_dataframe[col].dtype)]
                # Limit to top 5 columns if too many
                if len(mentioned_columns) > 5:
                    mentioned_columns = mentioned_columns[:5]
            else:
                # For other plots, use default behavior
                mentioned_columns = (available_columns[:2] 
                                  if len(available_columns) >= 2 
                                  else available_columns)
        
        # Determine description based on graph type
        if graph_type == 'heatmap':
            description = f"correlation between {', '.join(mentioned_columns)}"
        elif graph_type in ['box', 'violin']:
            description = f"distribution of {', '.join(mentioned_columns)}"
        elif graph_type == 'histogram':
            description = f"distribution of {mentioned_columns[0]}"
        elif graph_type == 'pie':
            description = f"proportion of {mentioned_columns[0]}"
        else:
            description = f"relationship between {', '.join(mentioned_columns)}"
        
        return {
            'type': graph_type,
            'columns': mentioned_columns,
            'description': description
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

    def _analyze_code_base(self, code, analysis_type='python'):
        """Base method for code analysis."""
        try:
            analysis = self._create_analysis_template()
            
            for line in code.split('\n'):
                line = line.strip()
                if line.startswith(('import ', 'from ', 'require(')):
                    analysis['imports'].append(line)
                elif line.startswith('def ') or 'function' in line or '=>' in line:
                    analysis['functions'].append(line)
                elif line.startswith('class '):
                    analysis['classes'].append(line)
                elif '=' in line and not any(line.startswith(prefix) for prefix in ['#', '//', '/*', 'def ', 'class ', 'function']):
                    analysis['variables'].append(line)
                elif any(comment in line for comment in ['#', '//', '/*', '"""', "'''"]):
                    analysis['comments' if not any(doc in line for doc in ['"""', "'''"]) else 'docstrings'].append(line)
            
            return analysis
        except Exception as e:
            return {'error': str(e)}

    def _analyze_python_code(self, code):
        """Analyze Python code using the base analyzer."""
        return self._analyze_code_base(code, 'python')

    def _analyze_javascript_code(self, code):
        """Analyze JavaScript code using the base analyzer."""
        return self._analyze_code_base(code, 'javascript')

    def _analyze_java_code(self, code):
        """Analyze Java code using the base analyzer."""
        return self._analyze_code_base(code, 'java')

    # Replace all the individual placeholder methods with a single method
    def __getattr__(self, name):
        """Handle all unimplemented language analysis methods."""
        if name.startswith('_analyze_') and name.endswith('_code'):
            return lambda code: {'error': 'Not implemented'}
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

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

    def handle_error(self, error, error_type=None):
        """Centralized error handling method."""
        error_str = str(error)
        
        if error_type in ERROR_MESSAGES:
            return ERROR_MESSAGES[error_type]
        
        # Check for known error patterns
        if "API key" in error_str:
            return ERROR_MESSAGES['api_key']
        elif "quota" in error_str.lower():
            return ERROR_MESSAGES['quota']
        elif "permission" in error_str.lower():
            return ERROR_MESSAGES['permission']
        elif "invalid image" in error_str.lower():
            return ERROR_MESSAGES['invalid_image']
        
        # Default error message
        return f"Error: {error_str}"

    def handle_pdf(self, file_content):
        """Handle PDF file content."""
        text = ""
        pdf_reader = PdfReader(io.BytesIO(file_content))
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
        return text

    def handle_text(self, file_content):
        """Handle plain text file content."""
        return file_content.decode("utf-8")

    def handle_excel(self, file_content):
        """Handle Excel file content."""
        df = pd.read_excel(io.BytesIO(file_content))
        return self._process_dataframe(df)

    def handle_csv(self, file_content):
        """Handle CSV file content."""
        df = pd.read_csv(io.BytesIO(file_content))
        return self._process_dataframe(df)

    def handle_image(self, file_content):
        """Handle image file content."""
        try:
            # Convert bytes to image for processing
            image = Image.open(io.BytesIO(file_content))
            return self.process_image(image)
        except Exception as e:
            return self.handle_error(e)

    def _process_dataframe(self, df):
        """Process DataFrame and return formatted text."""
        self.current_dataframe = df  # Store for later use
        text = df.to_string()
        
        # Add column information
        text += "\n\nColumn Information:\n"
        for col in df.columns:
            text += f"{col}: {df[col].dtype}\n"
        
        return text

    def _is_analysis_question(self, question):
        """Determine if the question requires analysis rather than summary."""
        analysis_keywords = [
            "analyze", "analysis", "examine", "evaluate", "assess",
            "investigate", "explore", "explain", "why", "how",
            "what causes", "what factors", "in depth", "detailed"
        ]
        return any(keyword in question.lower() for keyword in analysis_keywords)
