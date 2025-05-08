from typing import Dict

import streamlit as st

from app.utils.visualizations import (
    create_emotion_wheel,
    create_detailed_scores_chart
)


def render_sidebar() -> str:
    """Render the sidebar with input options."""
    st.sidebar.title("Input Options")

    input_type = st.sidebar.radio(
        "Choose input type:",
        ["Text Input", "File Upload"]
    )

    return input_type


def render_text_input() -> str:
    """Render the text input area."""
    st.subheader("Enter Text")
    return st.text_area("", height=200)


def render_file_upload() -> str:
    """Render the file upload area."""
    st.subheader("Upload File")
    uploaded_file = st.file_uploader("Choose a text file", type=['txt'])

    if uploaded_file is not None:
        return uploaded_file.getvalue().decode()
    return ""


def render_results(result: Dict):
    """Render the analysis results."""
    if not result:
        st.warning("No results to display.")
        return

    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Dominant Emotion",
            result['emotions']['dominant_emotion'],
            delta=None
        )
    with col2:
        st.metric(
            "Intensity",
            f"{result['emotions']['intensity']:.2f}",
            delta=None
        )
    with col3:
        st.metric(
            "Analysis",
            result['emotions']['analysis'],
            delta=None
        )

    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Emotion Distribution", "Detailed Scores"])

    with tab1:
        st.subheader("Emotion Distribution")
        fig = create_emotion_wheel(
            result['emotions'],
            f"{result['model']} Emotion Wheel"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Detailed Emotion Scores")
        fig = create_detailed_scores_chart(
            result['emotions'],
            f"{result['model']} Detailed Scores"
        )
        st.plotly_chart(fig, use_container_width=True)
