import streamlit as st
from langchain_core.messages import HumanMessage
from datetime import datetime
from PIL import Image
from core.graph import Graph
import pandas as pd
import uuid
import time


# def format_user_bubble(content, timestamp):
#     return f"""
#     <div style="display: flex; justify-content: flex-end; margin-bottom: 5px; padding-right:10%;">
#         <div style="background-color: #e1f5fe; padding: 10px; border-radius: 10px;
#                     max-width: 70%; color: black; text-align: left;">
#             <p style="margin: 0;">{content}</p>
#             <p style="font-size: 10px; text-align: right; color: gray; margin: 0;">{timestamp}</p>
#         </div>
#     </div>
#     """
def format_user_bubble(query, timestamp):
    """
    Format user message as a chat bubble on the right side of the screen

    Args:
        query: The user's message text

    Returns:
        None (displays the message directly using st.markdown)
    """
    return f"""
        <style>
        .user-bubble {{
            background-color: #E9F5FE;
            border-radius: 18px 18px 0px 18px;
            padding: 12px 18px;
            margin: 5px 0px;
            max-width: 80%;
            display: inline-block;
            float: right;
            clear: both;
            color: #0A2540;
            box-shadow: 0px 1px 2px rgba(0, 0, 0, 0.1);
            word-wrap: break-word;
        }}
        .user-bubble-container {{
            display: flex;
            justify-content: flex-end;
            width: 100%;
            margin-bottom: 5px;
            padding-right: 12%;
        }}
        </style>
        
        <div class="user-bubble-container">
            <div class="user-bubble">
                {query}
            </div>
        </div>
        """


def format_assistant_bubble(answer, timestamp, chart_key=None):
    return f"""
    <div style="display: flex; justify-content: flex-start; margin-left:12%;">
        <div style="padding:10px;border-radius:10px;margin-bottom:5px;max-width:88%;align-self:flex-start;">
            <p style="margin:0;color: var(--text-color);">{answer}</p>
        </div>
    </div>
    """


def format_assistant_bubble_typewrite(answer: str, typewriter: bool = False):
    """
    Displays the assistant's markdown-formatted response inside a styled left-aligned bubble.
    If typewriter=True, shows the response with a typewriter animation.
    """
    container = st.empty()

    bubble_start = """
    <div style="display: flex; justify-content: flex-start; margin-left:10%;">
        <div style="color: var(--text-color);padding: 5px; border-radius: 10px;
                    margin-bottom: 5px; max-width: 88%; align-self: flex-start;">
    """
    bubble_end = """
        </div>
    </div>
    """

    if typewriter:
        current = ""
        for char in answer:
            current += char
            container.markdown(
                bubble_start + current + bubble_end,
                unsafe_allow_html=True,
            )
            time.sleep(0.007)
    else:
        container.markdown(bubble_start + answer + bubble_end, unsafe_allow_html=True)


def show_agentic_chat_interface():

    if "graph_app" not in st.session_state:
        graph = Graph()
        app = graph.app()
        st.session_state.graph_app = app

    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    if "message_ids" not in st.session_state:
        st.session_state.message_ids = set()

    if "charts_to_show" not in st.session_state:
        st.session_state.charts_to_show = set()

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    col1, col2 = st.columns([6, 1])
    with col1:
        st.header("Origination Data Analytics Tool")
    with col2:
        st.markdown(
            """
        <div style="margin-top: 6px;">
        """,
            unsafe_allow_html=True,
        )
        if st.button(
            "🔁 Reset Chat", help="Reset Chat and Memory", use_container_width=True
        ):
            st.session_state.chat_log = []
            st.session_state.message_ids = set()
            st.session_state.charts_to_show = set()
            st.session_state.reset_successful = False

            # Optional: clear memory
            app = st.session_state.graph_app
            if hasattr(app, "checkpointer") and app.checkpointer is not None:
                try:
                    app.checkpointer.delete_thread(st.session_state.thread_id)
                    st.session_state.reset_successful = True
                except Exception as e:
                    st.session_state.reset_error = f" Could not clear memory: {str(e)}"

            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("reset_successful"):
        st.toast("✅ Chat reset successfully.")
        st.session_state.thread_id = str(uuid.uuid4())
        del st.session_state.reset_successful

    if st.session_state.get("reset_error"):
        st.toast(st.session_state.reset_error)
        del st.session_state.reset_error

    # Render chat history
    if "chat_log" in st.session_state:
        for entry in st.session_state.chat_log:
            st.markdown(entry["html"], unsafe_allow_html=True)

            chart_path = entry.get("chart_path")
            chart_key = entry.get("chart_key")
            table = entry.get("table")
            col1, col2, col3 = st.columns([1, 3, 4])
            # 📊 Chart toggle in center
            with col2:
                if chart_path and chart_key:
                    toggle_key = f"show_chart_{chart_key}"
                    show_chart = st.toggle("📊 Show Chart", key=toggle_key)
                    if show_chart:
                        img = Image.open(chart_path)
                        st.image(
                            img, caption="Generated Chart", use_container_width=False
                        )

            # 🧾 Table toggle aligned next to it
            with col3:
                if (
                    table is not None
                    and isinstance(table, pd.DataFrame)
                    and not table.empty
                ):
                    table_toggle_key = f"show_table_{chart_key}"
                    show_table = st.toggle("🧾 Show Table", key=table_toggle_key)
                    if show_table:
                        st.dataframe(table)

    prompt = st.chat_input("Ask your question...")

    if prompt:
        timestamp = datetime.now().strftime("%H:%M:%S")
        user_html = format_user_bubble(prompt, timestamp)
        st.session_state.chat_log.append({"html": user_html})
        st.markdown(user_html, unsafe_allow_html=True)

        # LangGraph execution
        with st.spinner("Analysing..."):
            app = st.session_state.graph_app
            state = {
                "messages": [HumanMessage(content=prompt)],
                "next": "",
                "hop_count": 0,
                "thread_id": st.session_state.thread_id,
            }
            stream = app.stream(state, config={"thread_id": st.session_state.thread_id})
            shown_agent_response = False

            for step in stream:
                node_name = list(step.keys())[0]
                node_data = step[node_name]
                agent_response = node_data.get("agent_response", {})
                # st.write(agent_response)
                if agent_response and not shown_agent_response:
                    shown_agent_response = True
                    answer = agent_response.get("answer", "")
                    approach = agent_response.get("approach", "")
                    chart_path = agent_response.get("figure")
                    # st.write(agent_response)
                    table = agent_response.get("table")
                    if agent_response.get("error"):
                        st.error(f"⚠️ Error: {agent_response.get('answer')}")
                        break
                    if table is not None and isinstance(table, dict):
                        table = pd.DataFrame.from_dict(table)
                    chart_key = (
                        f"chart_{len(st.session_state.chat_log)}"
                        if chart_path
                        else None
                    )
                    assistant_html = format_assistant_bubble(
                        answer, timestamp, chart_key
                    )

                    st.session_state.chat_log.append(
                        {
                            "html": assistant_html,
                            "chart_path": chart_path,
                            "chart_key": chart_key,
                            "table": table,
                        }
                    )

                    # st.markdown(assistant_html, unsafe_allow_html=True)
                    format_assistant_bubble_typewrite(answer, typewriter=True)
                    col1, col2, col3 = st.columns([1, 3, 4])
                    with col2:
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
                    with col3:
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
                                st.session_state[table_toggle_key] = (
                                    not st.session_state[table_toggle_key]
                                )

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
