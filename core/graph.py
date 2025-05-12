from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
import streamlit as st
import pandas as pd
from typing import TypedDict, Optional
from PIL import Image
from io import BytesIO
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

# from .tools import get_schema_inf
import functools
from typing import TypedDict, Annotated, List
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
)

tool = Tools()
memory = MemorySaver()
search_tool = DuckDuckGoSearchRun()


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
        BIAgent = BI_Agent(
            llm=Utility.llm(),
            # prompt=bi_agent_prompt,
            tools=[],
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
        response = BIAgent.generate_response(question, history=history)

        if response.get("figure"):
            print("got some figure here", response["figure"])

        if response.get("error"):
            return response, [
                AIMessage(content="[ERROR] BI Agent failed with: " + response["answer"])
            ]

            # Helper.display_saved_plot(response["figure"])
        if "table" in response and isinstance(response["table"], pd.DataFrame):
            response["table"] = response["table"].to_dict()

        message = (
            response["approach"]
            + "\n\nSolution we got from this approach is:\n"
            + response["answer"]
        )
        return response, [HumanMessage(content=message)]

    def fair_agent(self, state: AgentState):
        fair_agent = FairLendingAgent(
            llm=Utility.llm(),
            tools=[search_tool],
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
        response = fair_agent.generate_response(question, history=history)

        if response.get("figure"):
            Helper.display_saved_plot(response["figure"])

        if response.get("error"):
            return response, [
                AIMessage(content="[ERROR] BI Agent failed with: " + response["answer"])
            ]

            # Helper.display_saved_plot(response["figure"])
        if "table" in response and isinstance(response["table"], pd.DataFrame):
            response["table"] = response["table"].to_dict()

        message = (
            response["approach"]
            + "\n\nSolution we got from this approach is:\n"
            + response["answer"]
        )
        return response, [HumanMessage(content=message)]

    def risk_agent(self, state: AgentState):
        risk_agent = RiskEvaluationAgent(
            llm=Utility.llm(),
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
        response = risk_agent.generate_response(question, history=history)

        if response.get("error"):
            return response, [
                AIMessage(content="[ERROR] BI Agent failed with: " + response["answer"])
            ]

            # Helper.display_saved_plot(response["figure"])
        if "table" in response and isinstance(response["table"], pd.DataFrame):
            response["table"] = response["table"].to_dict()

        if response.get("figure"):
            Helper.display_saved_plot(response["figure"])

        message = (
            response["approach"]
            + "\n\nSolution we got from this approach is:\n"
            + response["answer"]
        )
        return response, [HumanMessage(content=message)]

    def general_agent(self, state: AgentState):
        gen_agent = ScenarioSimulationAgent(
            llm=Utility.llm(),
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
        response = gen_agent.generate_response(question, history=history)

        if response.get("figure"):
            Helper.display_saved_plot(response["figure"])

        if response.get("error"):
            return response, [
                AIMessage(content="[ERROR] BI Agent failed with: " + response["answer"])
            ]

            # Helper.display_saved_plot(response["figure"])
        if "table" in response and isinstance(response["table"], pd.DataFrame):
            response["table"] = response["table"].to_dict()

        message = (
            response["approach"]
            + "\n\nSolution we got from this approach is:\n"
            + response["answer"]
        )
        return response, [HumanMessage(content=message)]

    def ood_agent(self, state: AgentState):
        OODAgent = OutOfDomainAgent(llm=Utility.ood_llm(), tools=[search_tool])
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
        data_loader = st.session_state.data
        self.data = data_loader.data
        data_dictionary = data_loader.format_schema_from_excel(
            data_loader.data_description
        )
        self.data_description = data_dictionary

        # Build and compile the graph
        graph = self.build()
        app = graph.compile(checkpointer=memory)
        # removing graph image from prod
        # mermaid_png_bytes = app.get_graph().draw_mermaid_png()
        # img = Image.open(BytesIO(mermaid_png_bytes))
        return app
