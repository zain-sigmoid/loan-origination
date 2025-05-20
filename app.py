import streamlit as st
import traceback

# Set page configuration
st.set_page_config(
    page_title="Lending Risk Analysis & Approval Prediction",
    page_icon="🏦",
    layout="wide",
)

# Minimal session state init
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# show_agentic_chat_interface()
# 🚀 Directly run the main chat interface
try:
    from chat_page import show_agentic_chat_interface

    show_agentic_chat_interface()
except Exception as e:
    print(f"Error in chat interface: {e}")
    traceback.print_exc()
    st.error(
        "⚠️ Oops! Something went wrong. Please try your request again. If the issue continues, try refreshing or clearing the chat to reset things."
    )
