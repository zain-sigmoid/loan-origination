import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
from chat_page import show_agentic_chat_interface
from static.intro_page import show_intro_page
from core.data_loader import Data


# Set page configuration
st.set_page_config(
    page_title="Lending Risk Analysis & Approval Prediction",
    page_icon="🏦",
    layout="wide",
)


def get_logo_base64():
    logo_path = os.path.join(os.path.dirname(__file__), "./static/logo.png")
    with open(logo_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


def top_navbar(data_loaded=False):
    encoded_logo = get_logo_base64()
    data_loaded = st.session_state.get("data_loaded", False)
    status = "✅ Data Loaded" if data_loaded else "⏳ Loading Data..."
    st.markdown(
        '<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">',
        unsafe_allow_html=True,
    )
    # Style and layout
    navbar_html = f"""
    <nav class="navbar sticky-top navbar-light bg-light">
        <div class="container">
            <a class="navbar-brand" href="#">
            <img src="data:image/png;base64,{encoded_logo}" alt="logo" width="50" height="30">
            </a>
            <form class="d-flex">
                <button class="btn btn-secondary text-warning" type="button" disabled>
                    {status}
                </button>
            </form>
        </div>
    </nav>
    """

    st.markdown(navbar_html, unsafe_allow_html=True)
    # st.markdown(
    #     f"""
    #     <style>
    #     .top-navbar {{
    #         position: sticky;
    #         top: 0;
    #         z-index: 100;
    #         background-color: #ffffff;
    #         display: flex;
    #         justify-content: space-between;
    #         align-items: center;
    #         padding: 6px 20px;
    #         height: 50px;
    #         border-bottom: 1px solid #eee;
    #     }}
    #     .nav-left img {{
    #         height: 30px;
    #         vertical-align: middle;
    #     }}
    #     .nav-right {{
    #         display: flex;
    #         align-items: center;
    #     }}
    #     .status-button {{
    #         padding: 5px 12px;
    #         background-color: gray;
    #         color: white;
    #         border: none;
    #         border-radius: 5px;
    #         font-size: 13px;
    #         font-weight: 500;
    #         cursor: default;
    #     }}
    #     </style>

    #     <div class="top-navbar">
    #         <div class="nav-left">
    #             <img src="data:image/png;base64,{encoded_logo}" alt="Logo">
    #         </div>
    #         <div class="nav-right">
    #             <button class="status-button" disabled>{status}</button>
    #         </div>
    #     </div>
    #     """,
    #     unsafe_allow_html=True,
    # )


# Initialize session state
if "show_chat" not in st.session_state:
    st.session_state.show_chat = "intro"
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "messages" not in st.session_state:
    st.session_state.messages = []

top_navbar(st.session_state.data_loaded)

if st.session_state.show_chat == "intro":
    show_intro_page()
else:
    try:
        show_agentic_chat_interface()
    except Exception as e:
        print(f"Error in chat interface: {e}")
        st.error("⚠️ An unexpected error occurred. Please try again.")
        st.expander("Error Details").markdown(f"```{str(e)}```")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if not st.session_state.data_loaded:
        with st.spinner("🔄 Loading data in background..."):
            st.session_state.data = Data()
            st.session_state.data_loaded = True
            st.rerun()

# ✅ Handle Try Now button
if st.session_state.get("try_now") and st.session_state.data_loaded:
    st.session_state.show_chat = "agentic"
    st.rerun()

if st.session_state.get("back"):
    st.session_state.show_chat = "intro"
    st.rerun()
