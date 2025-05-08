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

# Main content
st.title(APP_TITLE)
st.markdown(APP_DESCRIPTION)

# Initialize analyzer only when needed
@st.cache_resource
def get_analyzer():
    return EmotionAnalyzer()

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
        with st.spinner("Initializing models and analyzing text..."):
            try:
                # Get analyzer instance (will be cached)
                analyzer = get_analyzer()
                results = analyzer.analyze_text(text)
                
                # Display results for each model
                for model_key, result in results.items():
                    st.markdown(f"### {result['model']} Analysis")
                    
                    # Create two columns for metrics
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
                    
                    # Display analysis in a container
                    with st.container():
                        st.markdown("#### Analysis")
                        st.info(result['emotions']['analysis'])
                    
                    # Display visualizations
                    st.markdown("#### Visual Analysis")
                    
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
                st.markdown("### Model Comparison")
                st.plotly_chart(
                    create_comparison_chart(list(results.values())),
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Error analyzing text: {str(e)}")
    else:
        st.warning("Please enter some text for analysis.")
