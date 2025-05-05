import streamlit as st
import pandas as pd
import numpy as np
from chat_page import show_agentic_chat_interface


# Set page configuration
st.set_page_config(
    page_title="Lending Risk Analysis & Approval Prediction",
    page_icon="🏦",
    layout="wide",
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


def show_intro_page():
    # Custom CSS to center content and add styling
    st.markdown(
        """
        <style>
        .main {
            padding: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        .stButton>button {
            width: 200px;
            margin: 2rem auto;
            display: block;
        }
        .title-text {
            text-align: center;
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        .subtitle-text {
            text-align: center;
            font-size: 1.5rem;
            margin-bottom: 2rem;
            color: #666;
        }
        .section {
            margin: 2rem 0;
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 10px;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Title and Subtitle
    st.markdown(
        '<h1 class="title-text">🏦 Lending Risk Analysis & Approval Prediction</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h2 class="subtitle-text">Data-Driven Credit Decisioning System</h2>',
        unsafe_allow_html=True,
    )

    # Project Description
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown(
        """
    This project is a comprehensive data science solution designed to streamline and enhance the loan origination process for financial institutions. 
    Using a real-world loan-level dataset containing over 50 features—ranging from borrower income, credit scores, and demographics to underwriting 
    outcomes and property-level census data—we aim to predict the outcomes of loan applications and assist underwriters in making faster, fairer, 
    and more informed credit decisions.
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Project Objectives
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 🔍 Project Objectives")
    st.markdown(
        """
    - Automate loan approval predictions based on borrower profiles.
    - Estimate key credit metrics such as risk levels and loan eligibility.
    - Analyze demographic and credit score influences on lending decisions.
    - Build interpretable and robust models using credit-relevant features.
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Dataset Highlights
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 🧩 Dataset Highlights")
    st.markdown(
        """
    - Contains detailed loan application and borrower information.
    - Includes credit scores, DTI, income, ethnicity, race, gender, AUS results, and denial reasons.
    - Enriched with census tract-level demographics and housing statistics.
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Try Model Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Try Model", type="primary", use_container_width=True):
            st.session_state.show_chat = True
            st.rerun()


def show_chat_interface():
    # Add back button at the top
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("← Back to Introduction", use_container_width=True):
            st.session_state.show_chat = False
            st.rerun()

    st.title("Lending Risk Analysis Chat")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about loan risk analysis..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            response = generate_response(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


def generate_response(prompt):
    # This is a simple static response generator
    responses = {
        "loan approval": "Based on our analysis, loan approval depends on several key factors including credit score, income, and debt-to-income ratio. Would you like to know more about any specific factor?",
        "risk level": "Our system categorizes risk levels into three tiers: Low, Medium, and High. Each tier is determined by analyzing multiple factors including credit history, income stability, and existing debt obligations.",
        "credit score": "Credit scores are a crucial factor in loan approval. Generally, scores above 700 are considered good, while scores below 600 may face challenges in approval.",
        "income": "Income verification is essential for loan approval. We typically look for stable income sources and sufficient income to cover monthly payments.",
        "default": "I can help you understand various aspects of loan risk analysis, including loan approval criteria, risk assessment, credit scoring, and income requirements. What specific information are you looking for?",
    }

    # Check for keywords in the prompt
    prompt_lower = prompt.lower()
    for key in responses:
        if key in prompt_lower:
            return responses[key]

    return responses["default"]


# Main app logic
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False

if st.session_state.show_chat:
    show_chat_interface()
else:
    show_intro_page()
