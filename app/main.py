import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.core.analyzer import EmotionAnalyzer
from app.config import APP_TITLE, APP_DESCRIPTION
from app.utils.visualizations import (
    create_emotion_wheel,
    create_detailed_scores_chart,
    create_comparison_chart,
    create_sentiment_gauge,
    create_sentiment_breakdown
)

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title=APP_TITLE['en'],
    page_icon="😊",
    layout="wide"
)

# Language selection
language = st.sidebar.selectbox(
    "Select Language / Selectează Limba",
    options=['en', 'ro'],
    format_func=lambda x: 'English' if x == 'en' else 'Română'
)

# Main content
st.title(APP_TITLE[language])
st.markdown(APP_DESCRIPTION[language])

# Initialize analyzer only when needed
@st.cache_resource
def get_analyzer():
    return EmotionAnalyzer()

# Text input
st.markdown("### " + ("Enter Text for Analysis" if language == 'en' else "Introduceți Textul pentru Analiză"))
text = st.text_area(
    "",
    height=200,
    placeholder="Type or paste your text here..." if language == 'en' else "Scrieți sau lipiți textul aici..."
)

# Analysis button
if st.button("Analyze Text" if language == 'en' else "Analizează Textul", use_container_width=True):
    if text:
        with st.spinner("Initializing models and analyzing text..." if language == 'en' else "Se inițializează modelele și se analizează textul..."):
            try:
                # Get analyzer instance (will be cached)
                analyzer = get_analyzer()
                results = analyzer.analyze_text(text, language)
                
                # Display results for each model
                for model_key, result in results.items():
                    st.markdown(f"### {result['model']} " + ("Analysis" if language == 'en' else "Analiză"))
                    
                    # Create two columns for metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Dominant Emotion" if language == 'en' else "Emoție Dominantă",
                            result['emotions']['dominant_emotion']
                        )
                    
                    with col2:
                        st.metric(
                            "Intensity" if language == 'en' else "Intensitate",
                            f"{result['emotions']['intensity']:.2f}"
                        )
                    
                    # Display analysis in a container
                    with st.container():
                        st.markdown("#### " + ("Analysis" if language == 'en' else "Analiză"))
                        st.info(result['emotions']['analysis'])
                    
                    # Display visualizations
                    st.markdown("#### " + ("Visual Analysis" if language == 'en' else "Analiză Vizuală"))
                    
                    if model_key == 'vader':
                        # VADER specific visualizations
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(
                                create_sentiment_gauge(result['emotions']['compound']),
                                use_container_width=True,
                                key=f"gauge_{model_key}"
                            )
                        with col2:
                            st.plotly_chart(
                                create_sentiment_breakdown(
                                    result['emotions']['pos'],
                                    result['emotions']['neu'],
                                    result['emotions']['neg']
                                ),
                                use_container_width=True,
                                key=f"breakdown_{model_key}"
                            )
                    else:
                        # DistilRoBERTa specific visualizations
                        st.plotly_chart(
                            create_emotion_wheel(
                                result['emotions']['scores'],
                                f"{result['model']} " + ("Emotion Distribution" if language == 'en' else "Distribuția Emoțiilor")
                            ),
                            use_container_width=True,
                            key=f"wheel_{model_key}"
                        )
                        
                        st.plotly_chart(
                            create_detailed_scores_chart(
                                result['emotions']['scores'],
                                f"{result['model']} " + ("Detailed Scores" if language == 'en' else "Scoruri Detaliate")
                            ),
                            use_container_width=True,
                            key=f"scores_{model_key}"
                        )
                
                # Display comparison chart
                st.markdown("### " + ("Model Comparison" if language == 'en' else "Comparare Modele"))
                st.plotly_chart(
                    create_comparison_chart(list(results.values())),
                    use_container_width=True,
                    key="comparison_chart"
                )
                
            except Exception as e:
                st.error(f"Error analyzing text: {str(e)}" if language == 'en' else f"Eroare la analizarea textului: {str(e)}")
    else:
        st.warning("Please enter some text for analysis." if language == 'en' else "Vă rugăm să introduceți text pentru analiză.")
