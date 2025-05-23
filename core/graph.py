from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
import streamlit as st
import pandas as pd
from typing import TypedDict, Optional
from termcolor import colored
from .agents import (
    Supervisor,
    BI_Agent,
    FairLendingAgent,
    RiskEvaluationAgent,
    ScenarioSimulationAgent,
    OutOfDomainAgent,
)
from langchain.tools import DuckDuckGoSearchRun
from .utils import Utility, Helper, Tools, execute_analysis
from langchain_tavily import TavilySearch
from langchain_community.utilities import GoogleSerperAPIWrapper
from exa_py import Exa
from .tools import ResilientSearchTool
import functools
from typing import TypedDict, Annotated, List
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
)
import os

try:
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API")
    SERP_API = os.getenv("SERP_API")
    EXA_API = os.getenv("EXA_API")
    tool = Tools()
    duckduckgo = DuckDuckGoSearchRun()
    tavily = TavilySearch(max_results=2, topic="general")
    serp = GoogleSerperAPIWrapper(serper_api_key=SERP_API)
    exa = Exa(api_key=EXA_API)
    resilient_search = ResilientSearchTool(
        duckduckgo=duckduckgo, tavily=tavily, serp=serp, exa=exa
    )
except Exception as e:
    st.error("⚠️ An unexpected error occurred.")

memory = MemorySaver()
max_retry = Utility.load_config()


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    next: str
    hop_count: Optional[int]
    agent_response: dict
    thread_id: str


class Graph:
    def __init__(self):
        self.data = None
        self.data_description = None
        self.data_sample = None

    def supervisor_node(self, state: AgentState):
        hop_count = state.get("hop_count", 0) + 1
        state["hop_count"] = hop_count
        agent_res = state.get("agent_response", "")
        # print(hop_count)
        if hop_count >= 4:
            return {
                "messages": state["messages"]
                + [
                    AIMessage(
                        content="Reached max reasoning steps. Ending conversation."
                    )
                ],
                "next": "FINISH",
                "hop_count": hop_count,
                "agent_res": agent_res,
                "thread_id": state["thread_id"],
            }
        history = state["messages"]
        supervisor_chain = Supervisor.chain(history=history)
        result = supervisor_chain.invoke({"messages": state["messages"]})
        agent_name = result["next"]
        last_message = state["messages"][-1].content.lower()

        if "[error]" in last_message:
            return {
                "messages": state["messages"]
                + [
                    AIMessage(
                        content="There was an error in the agent's analysis. Please rephrase your question or try again."
                    )
                ],
                "next": "FINISH",
                "hop_count": state.get("hop_count", 0),
            }

        print(colored(f"\n🧭 Supervisor routed to: {agent_name}", "yellow"))
        return {
            "messages": state["messages"]
            + [AIMessage(content=f"Calling {result['next']}.")],
            "next": agent_name,
            "hop_count": hop_count,
            "agent_res": agent_res,
            "thread_id": state["thread_id"],
        }

    def agent_node(self, state: AgentState, agent_func, name: str):
        response, message = agent_func(state)
        return {
            "messages": state["messages"] + message,
            "next": "supervisor",
            "agent_response": response,
            "thread_id": state["thread_id"],
        }

    def bi_agent(self, state: AgentState):
        llm = Utility.llm()
        BIAgent = BI_Agent(
            llm=llm,
            # prompt=bi_agent_prompt,
            tools=[],
            data_description=self.data_description,
            data_sample=self.data_sample,
            dataset=self.data,
            helper_functions={"execute_analysis": execute_analysis},
        )
        question = next(
            (
                m.content
                for m in reversed(state["messages"])
                if isinstance(m, HumanMessage)
            ),
            "",
        )
        history = state["messages"]
        max_retries = max_retry["llm"]["max_retry"]
        retry_no = 0
        print(colored(f"inside bi_agent,{retry_no, max_retries}", "yellow"))
        response = BIAgent.generate_response(question, history=history, formatting=True)
        while retry_no < max_retries:
            # Check if both are generated
            answer = response.get("answer", "").strip()
            print(f"answer:{answer}")
            res_error = response.get("error")
            if len(answer) == 0:
                res_error = True
            print(
                colored(
                    f"Retry Count : {retry_no} and success {res_error} and answer {answer!=''}",
                    "red",
                )
            )

            if not res_error:
                break
            retry_no += 1
            response = BIAgent.generate_response(question, history=history)
            print(f"{retry_no}, Error: {response.get('error')}")

        answer = response.get("answer", "").strip()
        res_error = response.get("error")
        if len(answer) == 0:
            res_error = True
        if retry_no == max_retries and res_error:
            print(colored("⚠️ All retries failed. Attempting recovery...", "light_blue"))

            # Step 1: Ask LLM to reword the original query
            reword_prompt = f"""
                The following user query repeatedly failed to generate a valid response: "{question}"
                can you use some context to reword the user query with same meaning and context to be asked to LLM
                Do not add any other word, just return the query as it will be passed to other LLM.
                """
            reworded_question = llm.invoke(reword_prompt).content.strip()

            print(colored("🔄 Reworded query:", "yellow"), reworded_question)
            response = BIAgent.generate_response(
                reworded_question, history=history, formatting=True
            )

            # Helper.display_saved_plot(response["figure"])
        if "table" in response and isinstance(response["table"], pd.DataFrame):
            print(colored("table found", "red"))
            response["table"] = response["table"].to_dict()

        approach = response.get("approach") or "[No approach provided]"
        answer = response.get("answer") or "[No answer returned]"

        message = approach + "\n\nSolution we got from this approach is:\n" + answer
        return response, [HumanMessage(content=message)]

    def fair_agent(self, state: AgentState):
        llm = Utility.llm()
        fair_agent = FairLendingAgent(
            llm=llm,
            tools=resilient_search,
            data_description=self.data_description,
            dataset=self.data,
            helper_functions={"execute_analysis": execute_analysis},
        )
        question = next(
            (
                m.content
                for m in reversed(state["messages"])
                if isinstance(m, HumanMessage)
            ),
            "",
        )
        history = state["messages"]
        max_retries = max_retry["llm"]["max_retry"]
        retry_no = 0
        response = fair_agent.generate_response(
            question, history=history, formatting=True
        )
        while retry_no < max_retries:
            # Check if both are generated
            answer = response.get("answer", "").strip()
            res_error = response.get("error")
            if len(answer) == 0:
                res_error = True
            print(colored(f"Retry Count : {retry_no}", "red"))

            if not res_error:
                break
            retry_no += 1
            response = fair_agent.generate_response(
                question, history=history, formatting=True
            )
        answer = response.get("answer", "").strip()
        res_error = response.get("error")
        if len(answer) == 0:
            res_error = True
        if retry_no == max_retries and res_error:
            print(colored("⚠️ All retries failed. Attempting recovery...", "light_blue"))

            # Step 1: Ask LLM to reword the original query
            reword_prompt = f"""
                The following user query repeatedly failed to generate a valid response: "{question}"
                can you use some context to reword the user query with same meaning and context to be asked to LLM
                Do not add any other word, just return the query as it will be passed to other LLM.
                """
            reworded_question = llm.invoke(reword_prompt).content.strip()

            print(colored("🔄 Reworded query:", "yellow"), reworded_question)
            response = fair_agent.generate_response(
                reworded_question, history=history, formatting=True
            )

        # Helper.display_saved_plot(response["figure"])
        if "table" in response and isinstance(response["table"], pd.DataFrame):
            response["table"] = response["table"].to_dict()

        approach = response.get("approach") or "[No approach provided]"
        answer = response.get("answer") or "[No answer returned]"

        message = approach + "\n\nSolution we got from this approach is:\n" + answer
        return response, [HumanMessage(content=message)]

    def risk_agent(self, state: AgentState):
        llm = Utility.llm()
        risk_agent = RiskEvaluationAgent(
            llm=llm,
            data_description=self.data_description,
            dataset=self.data,
            helper_functions={"execute_analysis": execute_analysis},
        )
        question = next(
            (
                m.content
                for m in reversed(state["messages"])
                if isinstance(m, HumanMessage)
            ),
            "",
        )
        history = state["messages"]
        max_retries = max_retry["llm"]["max_retry"]
        retry_no = 0
        response = risk_agent.generate_response(
            question, history=history, formatting=True
        )

        while retry_no < max_retries:
            # Check if both are generated
            answer = response.get("answer", "").strip()
            res_error = response.get("error")
            if len(answer) == 0:
                res_error = True
            print(colored(f"Retry Count : {retry_no}", "red"))

            if not res_error:
                break

            retry_no += 1
            response = risk_agent.generate_response(
                question, history=history, formatting=True
            )

        answer = response.get("answer", "").strip()
        res_error = response.get("error")
        if len(answer) == 0:
            res_error = True
        if retry_no == max_retries and res_error:
            print(colored("⚠️ All retries failed. Attempting recovery...", "light_blue"))

            # Step 1: Ask LLM to reword the original query
            reword_prompt = f"""
                The following user query repeatedly failed to generate a valid response: "{question}"
                can you use some context to reword the user query with same meaning and context to be asked to LLM
                Do not add any other word, just return the query as it will be passed to other LLM.
                """
            reworded_question = llm.invoke(reword_prompt).content.strip()

            print(colored("🔄 Reworded query:", "yellow"), reworded_question)
            response = risk_agent.generate_response(
                reworded_question, history=history, formatting=True
            )

        # if response.get("error"):
        #     return response, [
        #         AIMessage(content="[ERROR] BI Agent failed with: " + response["answer"])
        #     ]

        # Helper.display_saved_plot(response["figure"])
        if "table" in response and isinstance(response["table"], pd.DataFrame):
            response["table"] = response["table"].to_dict()

        if response.get("figure"):
            Helper.display_saved_plot(response["figure"])

        approach = response.get("approach") or "[No approach provided]"
        answer = response.get("answer") or "[No answer returned]"

        message = approach + "\n\nSolution we got from this approach is:\n" + answer
        return response, [HumanMessage(content=message)]

    def general_agent(self, state: AgentState):
        llm = Utility.llm()
        gen_agent = ScenarioSimulationAgent(
            llm=llm,
            data_description=self.data_description,
            dataset=self.data,
            helper_functions={"execute_analysis": execute_analysis},
        )
        question = next(
            (
                m.content
                for m in reversed(state["messages"])
                if isinstance(m, HumanMessage)
            ),
            "",
        )
        history = state["messages"]
        max_retries = max_retry["llm"]["max_retry"]
        retry_no = 0
        response = gen_agent.generate_response(
            question, history=history, formatting=True
        )
        while retry_no < max_retries:
            # Check if both are generated
            answer = response.get("answer", "").strip()
            res_error = response.get("error")
            if len(answer) == 0:
                res_error = True
            print(colored(f"Retry Count : {retry_no}", "red"))

            if not res_error:
                break
            print("ERROR", response.get("answer"))

            retry_no += 1
            response = gen_agent.generate_response(
                question, history=history, formatting=True
            )

        answer = response.get("answer", "").strip()
        res_error = response.get("error")
        if len(answer) == 0:
            res_error = True
        if retry_no == max_retries and res_error:
            print(colored("⚠️ All retries failed. Attempting recovery...", "light_blue"))

            # Step 1: Ask LLM to reword the original query
            reword_prompt = f"""
                The following user query repeatedly failed to generate a valid response: "{question}"
                can you use some context to reword the user query with same meaning and context to be asked to LLM
                Do not add any other word, just return the query as it will be passed to other LLM.
                """
            reworded_question = llm.invoke(reword_prompt).content.strip()

            print(colored("🔄 Reworded query:", "yellow"), reworded_question)
            response = gen_agent.generate_response(
                reworded_question, history=history, formatting=True
            )

        if response.get("error"):
            return response, [
                AIMessage(content="[ERROR] BI Agent failed with: " + response["answer"])
            ]

            # Helper.display_saved_plot(response["figure"])
        if "table" in response and isinstance(response["table"], pd.DataFrame):
            response["table"] = response["table"].to_dict()

        approach = response.get("approach") or "[No approach provided]"
        answer = response.get("answer") or "[No answer returned]"

        message = approach + "\n\nSolution we got from this approach is:\n" + answer
        return response, [HumanMessage(content=message)]

    def ood_agent(self, state: AgentState):
        OODAgent = OutOfDomainAgent(llm=Utility.ood_llm(), tools=resilient_search)
        question = next(
            (
                m.content
                for m in reversed(state["messages"])
                if isinstance(m, HumanMessage)
            ),
            "",
        )
        resp = OODAgent.generate_response(question)
        message = "answer we got from this agent is:\n" + resp.content
        response = {
            "answer": resp.content,
            "approach": "",
            "figure": "",
        }
        return response, [HumanMessage(content=message)]

    def finish(self, state: AgentState):
        print("✅ Conversation complete.")
        print("Final message:", state["messages"][-1].content)
        return state

    def build(
        self,
    ):
        bi_agent_node = functools.partial(
            self.agent_node, agent_func=self.bi_agent, name="BI Agent"
        )
        fair_lending_agent_node = functools.partial(
            self.agent_node,
            agent_func=self.fair_agent,
            name="Fair Lending Compliance Agent",
        )
        risk_eval_agent_node = functools.partial(
            self.agent_node, agent_func=self.risk_agent, name="Risk Evaluation Agent"
        )
        scenario_agent_node = functools.partial(
            self.agent_node,
            agent_func=self.general_agent,
            name="Scenario Simulation Agent",
        )
        ood_agent_node = functools.partial(
            self.agent_node, agent_func=self.ood_agent, name="OOD Agent"
        )

        graph = StateGraph(AgentState)

        # Register nodes
        graph.add_node("supervisor", self.supervisor_node)
        graph.add_node("BI Agent", bi_agent_node)
        graph.add_node("Fair Lending Compliance Agent", fair_lending_agent_node)
        graph.add_node("Risk Evaluation Agent", risk_eval_agent_node)
        graph.add_node("Scenario Simulation Agent", scenario_agent_node)
        graph.add_node("OOD Agent", ood_agent_node)
        graph.add_node("FINISH", self.finish)

        graph.add_edge("BI Agent", "supervisor")
        graph.add_edge("Fair Lending Compliance Agent", "supervisor")
        graph.add_edge("Risk Evaluation Agent", "supervisor")
        graph.add_edge("Scenario Simulation Agent", "supervisor")
        graph.add_edge("OOD Agent", "supervisor")

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            lambda state: state["next"],
            {
                "BI Agent": "BI Agent",
                "Fair Lending Compliance Agent": "Fair Lending Compliance Agent",
                "Risk Evaluation Agent": "Risk Evaluation Agent",
                "Scenario Simulation Agent": "Scenario Simulation Agent",
                "OOD Agent": "OOD Agent",
                "FINISH": "FINISH",
            },
        )
        return graph

    def app(
        self,
    ):
        print("creating graph")
        data_loader = st.session_state.data
        self.data = data_loader.data
        data_dictionary = data_loader.format_schema_from_excel(
            data_loader.data_description
        )
        self.data_description = data_dictionary
        self.data_sample = data_loader.data_sample
        # Build and compile the graph
        graph = self.build()
        app = graph.compile(checkpointer=memory)
        # removing graph image from prod
        # mermaid_png_bytes = app.get_graph().draw_mermaid_png()
        # img = Image.open(BytesIO(mermaid_png_bytes))
        return app
