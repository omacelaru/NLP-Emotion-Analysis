import streamlit as st

from app.core.analyzer import EmotionAnalyzer
from app.utils.config import APP_TITLE, APP_DESCRIPTION
from app.utils.visualizations import create_sentiment_gauge, create_sentiment_breakdown

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="😊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 1rem;
    }
    .stTextArea>div>div>textarea {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        font-size: 1rem;
    }
    .result-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

# Main content
st.title(APP_TITLE)
st.markdown(APP_DESCRIPTION)

# Initialize analyzer
analyzer = EmotionAnalyzer()

# Text input
st.markdown("### Enter Text for Analysis")
text = st.text_area(
    "",
    height=200,
    placeholder="Type or paste your text here..."
)

# Analysis button
if st.button("Analyze Text", use_container_width=True):
    if text:
        with st.spinner("Analyzing..."):
            result = analyzer.analyze_text(text)

            # Display results
            st.markdown("### Analysis Results")

            # Key metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Dominant Emotion",
                    result['emotions']['dominant_emotion']
                )

            with col2:
                st.metric(
                    "Intensity",
                    f"{result['emotions']['intensity']:.2f}"
                )

            # Visualizations
            st.markdown("### Visual Analysis")
            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(
                    create_sentiment_gauge(result['emotions']['compound']),
                    use_container_width=True
                )

            with col2:
                st.plotly_chart(
                    create_sentiment_breakdown(
                        result['emotions']['pos'],
                        result['emotions']['neu'],
                        result['emotions']['neg']
                    ),
                    use_container_width=True
                )

            # Detailed scores
            st.markdown("### Detailed Scores")
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.write(result['emotions'])
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter some text for analysis.")
