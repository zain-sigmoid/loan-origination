import streamlit as st


def show_intro_page():
    # Intro Page Styles
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

    # Title and subtitle
    # st.markdown(
    #     """<h1 class="title-text">🏦 Lending Risk Analysis</h1>
    #     <h1 class="title-text">And </h1>
    #     <h1 class="title-text">Approval Prediction</h1>""",
    #     unsafe_allow_html=True,
    # )
    st.markdown(
        """
                    <div style="display: flex; justify-content: center; text-align: center;flex-direction: column;">
                        <p style="font-size: 50px; color: #171719; font-weight: bold; font-family: 'Graphik';">
                            <span style="color: #023E8A;">Lending <span style="color: red;">Risk</span> Analysis</span> 
                        </p>
                        <p style="font-size: 50px; color: #171719; font-weight: bold; font-family: 'Graphik';">AND</p>
                        <p style="font-size: 50px; color: #023E8A; font-weight: bold; font-family: 'Graphik';">
                            Approval Prediction
                        </p>
                    </div>
                    """,
        unsafe_allow_html=True,
    )
    # st.markdown(
    #     '<h2 class="subtitle-text">Data-Driven Credit Decisioning System</h2>',
    #     unsafe_allow_html=True,
    # )

    # Project Description
    # st.markdown('<div class="section">', unsafe_allow_html=True)
    # st.markdown(
    #     """
    # This project streamlines the loan origination process using data science. It leverages real-world datasets with over 50 features
    # to help underwriters predict loan approval outcomes and assess borrower risk more fairly and efficiently.
    # """
    # )
    # st.markdown("</div>", unsafe_allow_html=True)

    # # Objectives
    # st.markdown('<div class="section">', unsafe_allow_html=True)
    # st.markdown("### 🔍 Project Objectives")
    # st.markdown(
    #     """
    # - Predict loan approval based on applicant profile.
    # - Assess borrower risk levels and eligibility.
    # - Examine fairness across demographic groups.
    # - Generate interpretable, actionable insights.
    # """
    # )
    # st.markdown("</div>", unsafe_allow_html=True)

    # # Dataset Highlights
    # st.markdown('<div class="section">', unsafe_allow_html=True)
    # st.markdown("### 🧩 Dataset Highlights")
    # st.markdown(
    #     """
    # - Loan, applicant, and property details.
    # - Credit score, DTI, income, race, sex, and AUS results.
    # - Census-level housing and demographic indicators.
    # """
    # )
    st.markdown("</div>", unsafe_allow_html=True)

    # Try Model Button
    col1, col2, col3 = st.columns([1, 2, 1])
    # with col2:
    #     if st.button("🚀 Try the Model", use_container_width=True):
    #         st.session_state.show_chat = "agentic"
    #         st.rerun()
    with col2:
        if st.button(
            "🚀 Try the Model", disabled=not st.session_state.get("data_loaded", False)
        ):
            st.session_state.show_chat = "agentic"
            st.rerun()
