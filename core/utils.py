import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from IPython.display import display, Image
import re
import uuid
import yaml
import ast
from dotenv import load_dotenv
from termcolor import colored
import traceback
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
import warnings

warnings.filterwarnings("ignore")

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("LLM_API_KEY")
memory = MemorySaver()


class Utility:
    def load_prompts(filepath="prompts.yml"):
        base_path = os.path.dirname(__file__)  # folder where utils.py is
        full_path = os.path.join(base_path, filepath)
        with open(full_path, "r") as f:
            return yaml.safe_load(f)

    def load_config(filepath="config.yml"):
        base_path = os.path.dirname(__file__)  # folder where utils.py is
        full_path = os.path.join(base_path, filepath)
        with open(full_path, "r") as f:
            return yaml.safe_load(f)

    def llm():
        try:
            # llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, convert_system_message_to_human=True)
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",  # Using flash for potentially faster responses
                # model="gemini-1.5-pro",
                temperature=0.2,
                convert_system_message_to_human=True,  # Important for some models
            )
            return llm
        except Exception as e:
            print(f"Error initializing Google Generative AI model: {e}")
            print(
                "Please ensure your API key is valid and you have the necessary permissions."
            )
            return None

    def ood_llm():
        try:
            ood_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",  # Using flash for potentially faster responses
                temperature=0.5,
                convert_system_message_to_human=True,  # Important for some models
            )
            return ood_llm
        except Exception as e:
            print(f"Error initializing Google Generative AI model: {e}")
            return None


class Helper:
    def extract_code_segments(response_text: str) -> list:
        """
        Extract Python code segments from a text response.

        Args:
            response_text (str): The full LLM response text.

        Returns:
            list: A list of extracted code strings.
        """
        # This matches code blocks in markdown format ```python ... ```
        pattern = r"```(?:python)?(.*?)```"
        code_blocks = re.findall(pattern, response_text, re.DOTALL)
        return [block.strip() for block in code_blocks]

    def display_saved_plot(plot_path: str):
        """
        Display a saved image plot using IPython.

        Args:
            plot_path (str): The path to the image file.
        """
        if os.path.exists(plot_path):
            display(Image(filename=plot_path))
        else:
            print(f"[Warning] Plot not found at: {plot_path}")


class Tools:
    def __init__(self):
        os.makedirs("plots", exist_ok=True)
        self.PLOT_DIR = "plots"
        os.makedirs(self.PLOT_DIR, exist_ok=True)

    def extract_code_segments(self, response_text):
        """
        Extracts segments wrapped in <approach>, <code>, <chart>, <answer> tags.
        Returns a dictionary with keys: 'approach', 'code', 'chart', 'answer'
        """
        import re

        segments = {}
        tags = ["approach", "code", "chart", "answer"]
        for tag in tags:
            match = re.search(f"<{tag}>(.*?)</{tag}>", response_text, re.DOTALL)
            if match:
                segments[tag] = match.group(1).strip()
        return segments

    # Helper to clean indentation
    def _dedent_code(self, code_str):
        lines = code_str.strip().split("\n")
        min_indent = float("inf")
        for line in lines:
            if line.strip():
                min_indent = min(min_indent, len(line) - len(line.lstrip()))
        return "\n".join(line[min_indent:] if line.strip() else "" for line in lines)


class Formatting:
    @staticmethod
    def format_response(prompt, answer, table, llm):
        try:
            FORMATTER_PROMPT = PromptTemplate.from_template(prompt)
            formatted_response = llm.invoke(
                FORMATTER_PROMPT.format(original_answer=answer, table_data=table)
            )

            respons = formatted_response.content
            # cleaned = re.sub(r"^```(?:python)?\s*", "", respons.strip())
            # cleaned = re.sub(r"```$", "", cleaned.strip())

            # # Safely evaluate to a dictionary
            # ans = ast.literal_eval(cleaned)
            # respons = Formatting.insert_newlines_for_markdown(respons)
            return respons
        except Exception as e:
            print(colored(f"Exception in formatting: {str(e)}", "light_yellow"))
            return answer

    @staticmethod
    def normalize_markdown(text: str) -> str:
        # Ensure there's exactly one line break before checkmarks (✓), dashes (-), asterisks (*)
        text = re.sub(r"\n*([✓*-])", r"\n\1", text)

        # Remove double/triple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Trim leading/trailing newlines
        text = text.strip()

        return text

    def insert_newlines_for_markdown(text: str) -> str:
        # Add newline before ✓ or ✗ if not already on a new line
        text = re.sub(r"(?<!\n)([✓✗])", r"\n\1", text)

        # Add newline before asterisks for markdown bullets or emphasis (but not mid-word italics)
        text = re.sub(r"(?<!\n)(\*{1,2})(?=\s)", r"\n\1", text)

        # Remove accidental multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()


@tool
def execute_analysis(df, response_text):
    """
    Execute the extracted code segments on the provided dataframe and return formatted output.

    Args:
        df (pd.DataFrame): The input dataframe.
        response_text (str): The LLM response containing code blocks and tags.

    Returns:
        dict: A dictionary with keys: 'approach', 'answer', 'figure', 'code', 'chart_code'
    """
    tools = Tools()
    print(colored("inside bi execute analysis", "cyan"))
    results = {
        "approach": None,
        "answer": None,
        "figure": None,
        "code": None,
        "chart_code": None,
    }

    try:
        segments = tools.extract_code_segments(response_text)
        if not segments:
            print("No segments extracted.")
            results["answer"] = (
                "⚠️ Sorry, I couldn't extract any analysis from the response."
            )
            results["error"] = True
            return results

        # Store textual segments
        results.update(
            {
                "approach": segments.get("approach", "No approach provided."),
                "code": segments.get("code", ""),
                "chart_code": segments.get("chart", ""),
                "answer": segments.get("answer", ""),
            }
        )

        # Setup a shared namespace
        namespace = {"df": df.copy(), "pd": pd, "plt": plt, "sns": sns}
        success = True
        error_val = ""

        # === Execute Analysis Code + Compute Answer ===
        if "code" in segments and "answer" in segments:
            raw_code = segments["code"]
            raw_code = re.sub(r"^```(?:python)?\s*", "", raw_code.strip())
            code = tools._dedent_code(raw_code)
            # code += f"\n\nanswer_text = f'''{segments['answer']}'''"
            safe_answer = re.sub(r"\{\s*\}", "[missing value]", segments["answer"])
            combined_code = f"""
{code}

# Now format the answer using actual results
answer_text = f'''{safe_answer}'''
"""
            try:
                exec(combined_code, namespace)
                results["answer"] = namespace.get(
                    "answer_text", "Sorry!, No specific answer was generated."
                )
                # results["output_df"] = namespace.get("output_df")
                if "output_df" in namespace:
                    df = namespace["output_df"]
                    if isinstance(df, pd.Series):
                        df = df.to_frame()
                    if isinstance(df, pd.DataFrame):
                        results["table"] = df
            except Exception as e:
                print(
                    colored("Code execution failed:", "light_red"),
                    traceback.format_exc(),
                )
                success = False
                error_val = str(e)

        # === Execute Chart Code and Save Plot ===
        if "chart" in segments and "no chart" not in segments["chart"].lower():
            raw_chart_code = segments["chart"]
            if raw_chart_code.strip().startswith("```"):
                raw_chart_code = re.sub(
                    r"^```(?:python)?\s*", "", raw_chart_code.strip()
                )
                raw_chart_code = re.sub(r"```$", "", raw_chart_code)
            raw_chart_code = re.sub(r"</?(code|pre)>", "", raw_chart_code.strip())
            chart_code = tools._dedent_code(raw_chart_code)
            chart_code = "\n".join(
                line for line in chart_code.split("\n") if "plt.show" not in line
            )

            plot_path = os.path.join(tools.PLOT_DIR, f"plot_{uuid.uuid4().hex}.png")
            if "plt.savefig" not in chart_code:
                chart_code += f"\nplt.savefig('{plot_path}', bbox_inches='tight')"

            try:
                exec(chart_code, namespace)
                results["figure"] = plot_path
            except Exception as e:
                print(
                    colored("Chart generation failed:", "light_red"),
                    traceback.format_exc(),
                )
                success = False
                error_val = str(e)

        if not success:
            results["answer"] = (
                f"⚠️ Something went wrong while analyzing the data. Please try again or rephrase your question."
            )
            results["error"] = True
            results["figure"] = None
            results["table"] = None
            results["error_val"] = error_val

        return results

    except Exception as e:
        print(
            colored("Critical failure in execute_analysis:", "red"),
            traceback.format_exc(),
        )
        return {
            "approach": "Error occurred during analysis.",
            "answer": "⚠️ There was a problem while processing your request. Please try again.",
            "figure": None,
            "code": str(e),
            "chart_code": None,
            "error": True,
            "error_val": str(e),
        }
