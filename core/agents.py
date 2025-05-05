import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from IPython.display import display, Markdown, Image
from typing import TypedDict, Optional
from termcolor import colored
import re
import uuid
import yaml
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
from .utils import Utility, Helper, Tools

load_dotenv()


os.environ["GOOGLE_API_KEY"] = os.getenv("LLM_API_KEY")

## LangChain related imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
    AIMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_core.tools import tool
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()


class Supervisor:
    def chain():
        prompt = Utility.load_prompts()
        llm = Utility.llm()
        supervisor_prompt = prompt["prompts"]["Supervisor"]
        members = [
            {
                "agent_name": "BI Agent",
                "description": """BI Agent (Business Intelligence Agent) is responsible for analyzing Home Mortgage Disclosure Act. data to generate insights. It performs exploratory data analysis (EDA), calculates summary statistics, identifies trends, and compares metrics across different borrower or loan characteristics 
        (e.g., ethnicity, loan purpose, state). It helps users understand patterns in loan applications, 
        approvals, and other dimensions by generating descriptive insights.""",
            },
            {
                "agent_name": "Fair Lending Compliance Agent",
                "description": """Fair Lending Compliance Agent is responsible for identifying potential bias or 
        discrimination in the Home Mortgage Disclosure Act. dataset. It performs fairness checks related to protected groups such as race, ethnicity, gender, or age, and determines whether lending practices meet compliance standards. This agent is activated when the user asks about fair lending, bias detection, or legal compliance 
        in the context of mortgage data.""",
            },
            {
                "agent_name": "Risk Evaluation Agent",
                "description": """Risk Evaluation Agent is responsible for assessing financial risks and 
        evaluating cost-related aspects of mortgage lending. It analyzes variables such as loan amount, 
        interest rate, debt-to-income ratio, and loan-to-value ratio to estimate financial health and risk. 
        This agent helps answer questions related to affordability, loan performance, and financial stress 
        among applicants.""",
            },
            {
                "agent_name": "General Scenario Agent",
                "description": """Generate Scenario Agent is responsible for creating and analyzing "what-if" scenarios based on user-defined parameters. This agent helps compare the outcomes of various decisions or actions, such 
        as the impact of increasing truck capacity, changing shipment consolidation strategies, or exploring 
        different operational scenarios. It can model changes in the system and assess the consequences of 
        those changes to support decision-making and optimization. This agent is called when the user asks 
        about scenario generation, comparisons of different outcomes, or analysis of hypothetical situations.""",
            },
        ]

        options = ["FINISH"] + [mem["agent_name"] for mem in members]

        # Generate members information for the prompt
        formatted_descriptions = []
        for member in members:
            cleaned_desc = re.sub(
                r"\s+", " ", member["description"].replace("\n", " ")
            ).strip()
            formatted_descriptions.append(f"{member['agent_name']}: {cleaned_desc}")

        members_info = "\n\n".join(formatted_descriptions)

        route_tool_schema = {
            "name": "route",
            "description": "Select the next agent based on the user's intent.",
            "parameters": {
                "title": "RouteSchema",
                "type": "object",
                "properties": {
                    "next": {
                        "title": "Next",
                        "type": "string",
                        "enum": options,
                        "description": "The agent to handle the next step in the conversation.",
                    }
                },
                "required": ["next"],
            },
        }
        formatted_prompt = supervisor_prompt.format(
            members_info=members_info,
            options=", ".join(options),  # list of agent names
        )
        prompt_s = ChatPromptTemplate.from_messages(
            [
                ("system", formatted_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        supervisor_chain = (
            prompt_s
            | llm.bind_tools(tools=[route_tool_schema], tool_choice="route")
            | JsonOutputFunctionsParser(key_name="next")  # Extract the 'next' value
        )
        return supervisor_chain


class BI_Agent:
    def __init__(self, llm, tools, dataset, data_description, helper_functions=None):
        """
        Initialize the Agent.

        Args:
            llm: The language model instance (e.g., ChatGoogleGenerativeAI).
            agent_name (str): The name of the agent (e.g., "BI Agent").
            tools (list): Tools available to the agent.
            dataset (pd.DataFrame): Data the agent will work with.
            helper_functions (dict): Optional dictionary of helper functions.
        """
        prompt_s = Utility.load_prompts()
        self.llm = llm
        # self.agent_name = agent_name
        self.tools = tools
        self.dataset = dataset
        self.helper_functions = helper_functions or {}
        self.prompt = prompt_s["prompts"]["BI_Agent"]
        self.data_description = data_description

    def add_helper_function(self, name, func):
        """
        Add a helper function to the agent.

        Args:
            name (str): Function name.
            func (callable): Function implementation.
        """
        self.helper_functions[name] = func

    def run(self, question):
        """
        Run the prompt and get LLM response.

        Args:
            question (str): User question.

        Returns:
            LLM output object.
        """
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompt.strip()),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        return self.llm.invoke(
            prompt_template.invoke(
                {
                    "data_description": self.data_description,
                    "question": question,
                    "messages": [HumanMessage(content=question)],
                }
            )
        )

    def generate_response(self, question):
        """
        Generate a final response from the agent.

        Args:
            question (str): The user input.

        Returns:
            str or dict: Output from helper function (e.g., executed analysis).
        """
        llm_response = self.run(question)
        print("llm response", llm_response)
        if "execute_analysis" in self.helper_functions:
            return self.helper_functions["execute_analysis"](
                df=self.dataset, response_text=llm_response.content
            )
        else:
            return llm_response.content

    def __repr__(self):
        """
        String representation of the agent.
        """
        return (
            f"Agent(name={self.agent_name}, "
            f"tools={[tool.name for tool in self.tools]}, "
            f"dataset_shape={self.dataset.shape})"
        )
