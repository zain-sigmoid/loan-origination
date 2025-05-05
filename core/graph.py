from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from uuid import uuid4
import pandas as pd
import streamlit as st
import json
from typing import TypedDict, Optional
from PIL import Image
from io import BytesIO
from termcolor import colored
from .agents import Supervisor, BI_Agent
from .utils import Utility, Helper, Tools
import functools
from typing import TypedDict, Annotated, List
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
    AIMessage,
)


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    next: str
    hop_count: Optional[int]
    agent_response: dict


class Graph:
    def __init__(self):
        self.data = None
        self.data_description = None

    def supervisor_node(self, state: AgentState):
        hop_count = state.get("hop_count", 0) + 1
        state["hop_count"] = hop_count
        agent_res = state.get("agent_response", "")
        # print(hop_count)
        if hop_count >= 3:
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
            }
        supervisor_chain = Supervisor.chain()
        result = supervisor_chain.invoke({"messages": state["messages"]})
        agent_name = result["next"]

        print(colored(f"\n🧭 Supervisor routed to: {agent_name}", "yellow"))
        return {
            "messages": state["messages"]
            + [AIMessage(content=f"Calling {result['next']}.")],
            "next": agent_name,
            "hop_count": hop_count,
            "agent_res": agent_res,
        }

    def agent_node(self, state: AgentState, agent_func, name: str):
        response, message = agent_func(state)
        return {
            "messages": state["messages"] + message,
            "next": "supervisor",
            "agent_response": response,
        }

    def bi_agent(self, state: AgentState):
        tool = Tools()
        BIAgent = BI_Agent(
            llm=Utility.llm(),
            # prompt=bi_agent_prompt,
            tools=[],
            data_description=self.data_description,
            dataset=self.data,
            helper_functions={"execute_analysis": tool.execute_analysis},
        )
        question = next(
            (
                m.content
                for m in reversed(state["messages"])
                if isinstance(m, HumanMessage)
            ),
            "",
        )
        print("calling biagent.generate response")
        response = BIAgent.generate_response(question)
        print(response.get("figure", "no fig"))

        if response.get("figure"):
            print("got some figure here", response["figure"])
            Helper.display_saved_plot(response["figure"])

        message = (
            response["approach"]
            + "\n\nSolution we got from this approach is:\n"
            + response["answer"]
        )
        return response, [HumanMessage(content=message)]

    # def fair_agent(state:AgentState):
    #     fair_agent = FairLendingAgent(
    #         llm=llm,
    #         prompt=bi_agent_prompt,
    #         data_description=feat,
    #         dataset=data,
    #         helper_functions=helper_functions,
    #     )
    #     question = next(
    #         (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
    #         "",
    #     )
    #     response = fair_agent.generate_response(question)

    #     if response.get("figure"):
    #         display_saved_plot(response["figure"])

    #     message = (
    #         response["approach"]
    #         + "\n\nSolution we got from this approach is:\n"
    #         + response["answer"]
    #     )
    #     return [HumanMessage(content=message)]

    # def risk_agent(state:AgentState):
    #     risk_agent = RiskEvaluationAgent(
    #         llm=llm,
    #         prompt=bi_agent_prompt,
    #         data_description=feat,
    #         dataset=data,
    #         helper_functions=helper_functions,
    #     )
    #     question = next(
    #         (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
    #         "",
    #     )
    #     response = risk_agent.generate_response(question)

    #     if response.get("figure"):
    #         display_saved_plot(response["figure"])

    #     message = (
    #         response["approach"]
    #         + "\n\nSolution we got from this approach is:\n"
    #         + response["answer"]
    #     )
    #     return [HumanMessage(content=message)]

    # def general_agent(state:AgentState):
    #     gen_agent = ScenarioSimulationAgent(
    #         llm=llm,
    #         prompt=bi_agent_prompt,
    #         data_description=feat,
    #         dataset=data,
    #         helper_functions=helper_functions,
    #     )
    #     question = next(
    #         (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
    #         "",
    #     )
    #     response = gen_agent.generate_response(question)

    #     if response.get("figure"):
    #         display_saved_plot(response["figure"])

    #     message = (
    #         response["approach"]
    #         + "\n\nSolution we got from this approach is:\n"
    #         + response["answer"]
    #     )
    #     return [HumanMessage(content=message)]

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

        graph = StateGraph(AgentState)

        # Register nodes
        graph.add_node("supervisor", self.supervisor_node)
        graph.add_node("BI Agent", bi_agent_node)
        graph.add_node("FINISH", self.finish)

        graph.add_edge("BI Agent", "supervisor")

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            lambda state: state["next"],
            {
                "BI Agent": "BI Agent",
                "FINISH": "FINISH",
            },
        )
        return graph

    def app(
        self,
    ):
        data_loader = st.session_state.data
        self.data = data_loader.data
        self.data_description = data_loader.data_description

        # Build and compile the graph
        graph = self.build()
        app = graph.compile()
        mermaid_png_bytes = app.get_graph().draw_mermaid_png()
        img = Image.open(BytesIO(mermaid_png_bytes))
        return app, img
