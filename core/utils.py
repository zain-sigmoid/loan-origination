import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from IPython.display import display, Image
import re
import uuid
import yaml
from dotenv import load_dotenv

from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
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

    def llm():
        try:
            # llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, convert_system_message_to_human=True)
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",  # Using flash for potentially faster responses
                temperature=0,
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
                temperature=0,
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
    ans_generated = False
    chart_generated = False
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
            return results

        # Store textual segments
        results = {
            "approach": segments.get("approach", "No approach provided."),
            "code": segments.get("code", ""),
            "chart_code": segments.get("chart", ""),
            "answer": None,
            "figure": None,
            "table": None,
            "error": False,
        }

        # Setup a shared namespace
        namespace = {"df": df.copy(), "pd": pd, "plt": plt, "sns": sns}

        # === Execute Analysis Code + Compute Answer ===
        if "code" in segments and "answer" in segments:
            code = tools._dedent_code(segments["code"])
            # code += f"\n\nanswer_text = f'''{segments['answer']}'''"
            combined_code = f"""
{code}

# Now format the answer using actual results
answer_text = f'''{segments['answer']}'''
"""
            try:
                exec(combined_code, namespace)
                ans_generated = True
                results["answer"] = namespace.get(
                    "answer_text", "Answer could not be computed."
                )
                # results["output_df"] = namespace.get("output_df")
                if "output_df" in namespace:
                    df = namespace["output_df"]
                    if isinstance(df, pd.Series):
                        df = df.to_frame()
                    if isinstance(df, pd.DataFrame):
                        results["table"] = df
            except Exception as e:
                ans_generated = False
                results["answer"] = "There was an error in executing the code."

        # === Execute Chart Code and Save Plot ===
        if "chart" in segments and "no chart" not in segments["chart"].lower():
            raw_chart_code = segments["chart"]
            if raw_chart_code.strip().startswith("```"):
                raw_chart_code = re.sub(
                    r"^```(?:python)?\s*", "", raw_chart_code.strip()
                )
                raw_chart_code = re.sub(r"```$", "", raw_chart_code)
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
                chart_generated = True
            except Exception as e:
                chart_generated = False
                results["figure"] = "There was an error in executing the chart code."

        return results

    except Exception as e:
        error_msg = f"[Agent Execution Error] {str(e)}"
        print(error_msg)
        if not chart_generated and not ans_generated:
            return {
                "approach": "Error occurred during analysis.",
                "answer": "An Error Occurred during analysis, Please try again.",
                "figure": None,
                "code": str(e),
                "chart_code": None,
                "error": True,
            }
        elif ans_generated and not chart_generated:
            return {
                "approach": results["approach"],
                "answer": results["answer"],
                "figure": None,
                "code": results["code"],
                "chart_code": results["chart_code"],
            }
        elif chart_generated and not ans_generated:
            return {
                "approach": results["approach"],
                "answer": "The chart was generated successfully, but no textual answer could be computed.",
                "figure": results["figure"],
                "code": results["code"],
                "chart_code": results["chart_code"],
            }
        else:
            return results
