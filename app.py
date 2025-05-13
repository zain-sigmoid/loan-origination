import streamlit as st
import streamlit as st
from chat_page import show_agentic_chat_interface

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
    show_agentic_chat_interface()
except Exception as e:
    print(f"Error in chat interface: {e}")
    st.error("⚠️ An unexpected error occurred. Please try again.")
    st.expander("Error Details").markdown(f"```{str(e)}```")
