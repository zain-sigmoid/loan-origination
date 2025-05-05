import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
import json
import os
from PIL import Image
from core.graph import Graph
import pandas as pd


def format_user_bubble(content, timestamp):
    return f"""
    <div style="background-color:#f1f1f1;padding:10px;border-radius:10px;margin-bottom:5px;max-width:80%;align-self:flex-end;color:black;">
        <p style="margin:0;">{content}</p>
        <p style="font-size:10px;text-align:right;color:gray;margin:0;">{timestamp}</p>
    </div>
    """


def format_assistant_bubble(answer, timestamp, chart_key=None):
    chart_button = ""
    if chart_key:
        chart_button = f"""
        <div style="margin-top:5px;">
            <button id="{chart_key}" style="padding:5px 10px;border:none;border-radius:5px;background:#007bff;color:white;cursor:pointer;">
                📊 Show Chart
            </button>
        </div>
        """
    return f"""
    <div style="background-color:#e8f0fe;padding:10px;border-radius:10px;margin-bottom:10px;max-width:80%;align-self:flex-start;">
        <p style="margin:0;color:black">{answer}</p>
        <p style="font-size:10px;text-align:right;color:gray;margin:0;">{timestamp}</p>
    </div>
    """


def show_agentic_chat_interface():
    st.title("🤖 Lending AI Chat – Agentic Mode")

    if "graph_app" not in st.session_state:
        graph = Graph()
        app, _ = graph.app()
        st.session_state.graph_app = app

    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    if "message_ids" not in st.session_state:
        st.session_state.message_ids = set()

    if "charts_to_show" not in st.session_state:
        st.session_state.charts_to_show = set()

    # Render chat history
    if "chat_log" in st.session_state:
        for entry in st.session_state.chat_log:
            st.markdown(entry["html"], unsafe_allow_html=True)

            chart_path = entry.get("chart_path")
            chart_key = entry.get("chart_key")
            table = entry.get("table")
            if chart_path and chart_key:
                toggle_key = f"show_chart_{chart_key}"
                show_chart = st.toggle("📊 Show Chart", key=toggle_key)
                if show_chart:
                    img = Image.open(chart_path).resize((512, 512))
                    st.image(img, caption="Generated Chart", use_container_width=False)

            if (
                table is not None
                and isinstance(table, pd.DataFrame)
                and not table.empty
            ):
                table_toggle_key = f"show_table_{chart_key}"
                show_table = st.toggle("🧾 Show Table", key=table_toggle_key)
                if show_table:
                    st.dataframe(table)

        # Always show chat bubble
        # st.markdown(entry["html"], unsafe_allow_html=True)

        # # Show chart if toggle is on
        # if chart_path and st.session_state.get(f"toggle_{chart_key}", False):
        #     img = Image.open(chart_path).resize((512, 512))
        #     st.image(img, caption="Generated Chart", use_container_width=False)

    # Chat input
    prompt = st.chat_input("Ask your question...")

    if prompt:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_html = format_user_bubble(prompt, timestamp)
        st.session_state.chat_log.append({"html": user_html})
        st.markdown(user_html, unsafe_allow_html=True)

        # LangGraph execution
        with st.spinner("Analysing..."):
            app = st.session_state.graph_app
            state = {"messages": [HumanMessage(content=prompt)], "next": ""}
            stream = app.stream(state)
            shown_agent_response = False

            for step in stream:
                node_name = list(step.keys())[0]
                node_data = step[node_name]
                agent_response = node_data.get("agent_response", {})
                if agent_response and not shown_agent_response:
                    shown_agent_response = True
                    answer = agent_response.get("answer", "")
                    approach = agent_response.get("approach", "")
                    chart_path = agent_response.get("figure")
                    table = agent_response.get("table")
                    chart_key = (
                        f"chart_{len(st.session_state.chat_log)}"
                        if chart_path
                        else None
                    )

                    assistant_html = format_assistant_bubble(
                        answer,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        chart_key,
                    )

                    st.session_state.chat_log.append(
                        {
                            "html": assistant_html,
                            "chart_path": chart_path,
                            "chart_key": chart_key,
                            "table": table,
                        }
                    )

                    st.markdown(assistant_html, unsafe_allow_html=True)

                    if chart_path and chart_key:
                        toggle_key = f"show_chart_{chart_key}"

                        if toggle_key not in st.session_state:
                            st.session_state[toggle_key] = False

                        # Toggle chart visibility
                        if st.toggle(
                            (
                                "📊 Show Chart"
                                if not st.session_state[toggle_key]
                                else "❌ Hide Chart"
                            ),
                            key=chart_key,
                        ):
                            st.session_state[toggle_key] = not st.session_state[
                                toggle_key
                            ]

                        if st.session_state[toggle_key]:
                            img = Image.open(chart_path)
                            st.image(
                                img,
                                caption="Generated Chart",
                                use_container_width=False,
                            )

                    if (
                        table is not None
                        and isinstance(table, pd.DataFrame)
                        and not table.empty
                    ):
                        table_toggle_key = f"show_table_{chart_key}"

                        if table_toggle_key not in st.session_state:
                            st.session_state[table_toggle_key] = False

                        if st.toggle(
                            (
                                "🧾 Show Table"
                                if not st.session_state[table_toggle_key]
                                else "❌ Hide Table"
                            ),
                            key=table_toggle_key + "_toggle",
                        ):
                            st.session_state[table_toggle_key] = not st.session_state[
                                table_toggle_key
                            ]

                        if st.session_state[table_toggle_key]:
                            st.dataframe(table)

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
