import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.core.analyzer import EmotionAnalyzer
from app.utils.config import APP_TITLE, APP_DESCRIPTION
from app.utils.visualizations import (
    create_emotion_wheel,
    create_detailed_scores_chart,
    create_comparison_chart,
    create_sentiment_gauge,
    create_sentiment_breakdown
)

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
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextArea>div>div>textarea {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }
    .stTextArea>div>div>textarea:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 5px rgba(76, 175, 80, 0.3);
    }
    .result-box {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .model-title {
        color: #2c3e50;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #4CAF50;
    }
    .analysis-header {
        color: #2c3e50;
        font-size: 1.2rem;
        margin: 1rem 0;
        padding: 0.5rem;
        background-color: #e8f5e9;
        border-radius: 6px;
    }
    .metric-box {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    .analysis-text {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #2c3e50;
        padding: 1rem;
        background-color: white;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Main content
st.title(APP_TITLE)
st.markdown(APP_DESCRIPTION)

# Initialize analyzer
try:
    analyzer = EmotionAnalyzer()
except Exception as e:
    st.error(f"Error initializing emotion analyzer: {str(e)}")
    st.stop()

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
        with st.spinner("Analyzing text with both models..."):
            try:
                results = analyzer.analyze_text(text)
                
                # Display results for each model
                for model_key, result in results.items():
                    st.markdown(f'<div class="model-title">{result["model"]} Analysis</div>', unsafe_allow_html=True)
                    
                    # Create two columns for metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric(
                            "Dominant Emotion",
                            result['emotions']['dominant_emotion']
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric(
                            "Intensity",
                            f"{result['emotions']['intensity']:.2f}"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display analysis
                    st.markdown('<div class="analysis-header">Analysis</div>', unsafe_allow_html=True)
                    st.markdown('<div class="analysis-text">', unsafe_allow_html=True)
                    st.write(result['emotions']['analysis'])
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display visualizations
                    st.markdown('<div class="analysis-header">Visual Analysis</div>', unsafe_allow_html=True)
                    
                    if model_key == 'vader':
                        # VADER specific visualizations
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
                    else:
                        # DistilRoBERTa specific visualizations
                        st.plotly_chart(
                            create_emotion_wheel(
                                result['emotions']['scores'],
                                f"{result['model']} Emotion Distribution"
                            ),
                            use_container_width=True
                        )
                        
                        st.plotly_chart(
                            create_detailed_scores_chart(
                                result['emotions']['scores'],
                                f"{result['model']} Detailed Scores"
                            ),
                            use_container_width=True
                        )
                
                # Display comparison chart
                st.markdown('<div class="analysis-header">Model Comparison</div>', unsafe_allow_html=True)
                st.plotly_chart(
                    create_comparison_chart(list(results.values())),
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Error analyzing text: {str(e)}")
    else:
        st.warning("Please enter some text for analysis.")
