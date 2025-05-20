import os
import re
import warnings
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_core.messages import AIMessage
from termcolor import colored


warnings.filterwarnings("ignore")
from .utils import Utility, Formatting

load_dotenv()
memory = MemorySaver()


class Supervisor:
    def chain(history=""):
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
                "agent_name": "Scenario Simulation Agent",
                "description": """Generate Scenario Agent is responsible for creating and analyzing "what-if" scenarios based on user-defined parameters. This agent helps compare the outcomes of various decisions or actions, such 
        as the impact of increasing truck capacity, changing shipment consolidation strategies, or exploring 
        different operational scenarios. It can model changes in the system and assess the consequences of 
        those changes to support decision-making and optimization. This agent is called when the user asks 
        about scenario generation, comparisons of different outcomes, or analysis of hypothetical situations.""",
            },
            {
                "agent_name": "OOD Agent",
                "description": """The Out of Domain Agent is responsible for handling user queries that are unrelated to mortgage data analytics. It manages casual conversation (e.g., greetings), generic questions about the assistant's capabilities, or any off-topic queries such as weather, jokes, or general knowledge. 
                This agent ensures a friendly and helpful response while clearly communicating that the system specializes in mortgage loan data. 
                It is triggered when the user's message does not align with the analytic, compliance, risk, or scenario-based goals of the other agents.""",
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
                "type": "object",
                "properties": {
                    "next": {
                        "type": "string",
                        "enum": options,
                        "description": "The agent to handle the next step in the conversation.",
                    }
                },
                "required": ["next"],
            },
        }
        formatted_history = "\n".join(f"{m.type}: {m.content}" for m in history)
        formatted_prompt = supervisor_prompt.format(
            members_info=members_info,
            conversation_history=formatted_history,
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
    def __init__(
        self, llm, tools, dataset, data_description, data_sample, helper_functions=None
    ):
        """
        Initialize the Agent.

        Args:
            llm: The language model instance (e.g., ChatGoogleGenerativeAI).
            agent_name (str): The name of the agent (e.g., "BI Agent").
            tools (list): Tools available to the agent.
            dataset (pd.DataFrame): Data the agent will work with.
            helper_functions (dict): Optional dictionary of helper functions.
        """
        self.prompt_s = Utility.load_prompts()
        self.llm = llm
        # self.agent_name = agent_name
        self.tools = tools
        self.dataset = dataset
        self.helper_functions = helper_functions or {}
        self.prompt = self.prompt_s["prompts"]["BI_Agent"]
        self.data_description = data_description
        self.data_sample = data_sample

    def add_helper_function(self, name, func):
        """
        Add a helper function to the agent.

        Args:
            name (str): Function name.
            func (callable): Function implementation.
        """
        self.helper_functions[name] = func

    def format_messages(self, messages):
        return "\n".join(f"{m.type}: {m.content}" for m in messages)

    def run(self, question, history=""):
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
                    "data_sample": self.data_sample,
                    "question": question,
                    "messages": [HumanMessage(content=question)],
                    "conversation_history": self.format_messages(history),
                }
            )
        )

    def generate_response(self, question, history=""):
        """
        Generate a final response from the agent.

        Args:
            question (str): The user input.

        Returns:
            str or dict: Output from helper function (e.g., executed analysis).
        """
        llm_response = self.run(question, history)
        # print(llm_response)
        print(colored("inside bi agent class", "green"))
        response = self.helper_functions["execute_analysis"].invoke(
            {"df": self.dataset, "response_text": llm_response.content}
        )
        return response
        # if response.get("error"):
        #     # here we can call llm when there is error, else we can format as of now returning response
        #     return response
        # else:
        #     print(colored("formatting", "cyan"))
        #     formatting_prompt = self.prompt_s["prompts"]["Formatting_Prompt"]
        #     formatted_response = Formatting.format_response(
        #         prompt=formatting_prompt, response=response, llm=self.llm
        #     )
        #     return formatted_response

    #
    # else:
    #     return llm_response.content

    def __repr__(self):
        """
        String representation of the agent.
        """
        return (
            f"Agent(name={self.agent_name}, "
            f"tools={[tool.name for tool in self.tools]}, "
            f"dataset_shape={self.dataset.shape})"
        )


class FairLendingAgent:
    def __init__(self, llm, tools, dataset, data_description, helper_functions=None):
        prompt_s = Utility.load_prompts()
        self.llm = llm
        self.agent_name = "Fair Lending Compliance Agent"
        self.dataset = dataset
        self.tools = tools or []
        self.data_description = data_description
        self.helper_functions = helper_functions or {}
        self.prompt = prompt_s["prompts"]["Fair_Lending_Compliance_Agent"]

    def add_helper_function(self, name, func):
        self.helper_functions[name] = func

    def format_messages(self, messages):
        return "\n".join(f"{m.type}: {m.content}" for m in messages)

    def run(self, question, history=""):
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
                    "conversation_history": self.format_messages(history),
                }
            )
        )

    def generate_response(self, question, history=""):
        result = self.run(question, history)
        content = result.content

        if "[SEARCH_REQUIRED]" in content:
            # Call DuckDuckGo
            try:
                search_result = self.tools._run(question)
            except Exception as e:
                print(f"Error in search: {e}")
                search_result = "This question is out of domain. Please ask a question related to mortgage data analysis. I am unable results at this time."
            result = {
                "approach": "internet search",
                "answer": search_result,
                "figure": None,
            }
            return result
        else:
            # Execute analysis if applicable
            return self.helper_functions["execute_analysis"].invoke(
                {"df": self.dataset, "response_text": content}
            )

    def __repr__(self):
        return (
            f"Agent(name={self.agent_name}, "
            f"tools={[tool.name for tool in self.tools]}, "
            f"dataset_shape={self.dataset.shape})"
        )


class RiskEvaluationAgent:
    def __init__(self, llm, dataset, data_description, helper_functions=None):
        prompt_s = Utility.load_prompts()
        self.llm = llm
        self.agent_name = "Risk and Cost Evaluation Agent"
        self.dataset = dataset
        self.prompt = prompt_s["prompts"]["Risk_Evaluation_Agent"]
        self.helper_functions = helper_functions or {}
        self.data_description = data_description

    def add_helper_function(self, name, func):
        self.helper_functions[name] = func

    def format_messages(self, messages):
        return "\n".join(f"{m.type}: {m.content}" for m in messages)

    def run(self, question, history=""):
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
                    "conversation_history": self.format_messages(history),
                }
            )
        )

    def generate_response(self, question, history=""):
        result = self.run(question, history)
        response = self.helper_functions["execute_analysis"].invoke(
            {"df": self.dataset, "response_text": result.content}
        )
        return response

    def __repr__(self):
        return (
            f"Agent(name={self.agent_name}, "
            f"tools={[tool.name for tool in self.tools]}, "
            f"dataset_shape={self.dataset.shape})"
        )


class ScenarioSimulationAgent:
    def __init__(self, llm, dataset, data_description, helper_functions=None):
        prompt_s = Utility.load_prompts()
        self.llm = llm
        self.agent_name = "Scenario Simulation Agent"
        self.dataset = dataset
        self.prompt = prompt_s["prompts"]["General_Scenario_Agent"]
        self.data_description = data_description
        self.helper_functions = helper_functions or {}

    def add_helper_function(self, name, func):
        self.helper_functions[name] = func

    def format_messages(self, messages):
        return "\n".join(f"{m.type}: {m.content}" for m in messages)

    def run(self, question, history=""):
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
                    "conversation_history": self.format_messages(history),
                }
            )
        )

    def generate_response(self, question, history=""):
        result = self.run(question, history)
        response = self.helper_functions["execute_analysis"].invoke(
            {"df": self.dataset, "response_text": result.content}
        )
        return response

    def __repr__(self):
        return (
            f"Agent(name={self.agent_name}, "
            f"tools={[tool.name for tool in self.tools]}, "
            f"dataset_shape={self.dataset.shape})"
        )


class OutOfDomainAgent:
    def __init__(self, llm, tools):
        prompt_s = Utility.load_prompts()
        self.llm = llm
        self.prompt = prompt_s["prompts"]["Out_Of_Domain"]
        self.tools = tools or []

    def format_bullet_output(self, raw_text: str) -> str:
        # Normalize bullets: replace "*" or unicode bullets with "-"
        cleaned = re.sub(r"^[\*\u2022]\s*", "- ", raw_text, flags=re.MULTILINE)

        # Ensure bullets start on a new line
        cleaned = re.sub(r"\n?[\*\u2022]\s*", r"\n- ", cleaned)

        # Ensure at least one newline between list items
        cleaned = re.sub(r"(?<!\n)\n-(?!\s)", r"\n\n- ", cleaned)

        # Fix any mixed bullet with colon issue like:
        # - **Date:** - Forecast...
        cleaned = re.sub(r"(- \*\*[^:\n]+:\*\*)\s*-\s*", r"\1 ", cleaned)

        return cleaned.strip()

    def generate_response(self, question: str) -> str:
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", self.prompt), ("human", "{question}")]
        )
        response = self.llm.invoke(prompt_template.invoke({"question": question}))
        if "[SEARCH_REQUIRED]" in response.content:
            try:
                search_result = self.tools._run(question)
            except Exception as e:
                print(f"Error in search: {e}")
                search_result = "This question is out of domain. Please ask a question related to mortgage data analysis. I am unable results at this time."
            return AIMessage(content=search_result)
        else:
            return response
