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
# Page configuration
st.set_page_config(
    page_title=APP_TITLE['en'],
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .stTextArea>div>div>textarea {
        border-radius: 5px;
        border: 1px solid #ddd;
        padding: 1rem;
    }
    .stTextArea>div>div>textarea:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 0 1px #4CAF50;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    .stMarkdown {
        padding: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)



# Language selection with improved styling
st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h2 style='color: #4CAF50;'>🌍 Language / Limba</h2>
    </div>
""", unsafe_allow_html=True)

language = st.sidebar.selectbox(
    "",
    options=['en', 'ro'],
    format_func=lambda x: 'English' if x == 'en' else 'Română',
    label_visibility="collapsed"
)

# Main content with improved layout
st.markdown(f"<h1 style='color: #4CAF50;'>{APP_TITLE[language]}</h1>", unsafe_allow_html=True)
st.markdown(APP_DESCRIPTION[language])

# Initialize analyzer only when needed
@st.cache_resource
def get_analyzer():
    return EmotionAnalyzer()

# Text input with improved styling
st.markdown(f"<h2 style='color: #4CAF50; margin-bottom: 0.5rem;'>{'Enter Text for Analysis' if language == 'en' else 'Introduceți Textul pentru Analiză'}</h2>", unsafe_allow_html=True)
text = st.text_area(
    "",
    height=150,
    placeholder="Type or paste your text here..." if language == 'en' else "Scrieți sau lipiți textul aici...",
    help="Enter the text you want to analyze for emotions and sentiment.",
    label_visibility="collapsed"
)

# Analysis button with improved styling
if st.button(
    "🔍 " + ("Analyze Text" if language == 'en' else "Analizează Textul"),
    use_container_width=True,
    help="Click to analyze the text and view detailed emotion insights."
):
    if text:
        with st.spinner("🔄 " + ("Initializing models and analyzing text..." if language == 'en' else "Se inițializează modelele și se analizează textul...")):
            try:
                # Get analyzer instance (will be cached)
                analyzer = get_analyzer()
                results = analyzer.analyze_text(text, language)
                
                # Display results for each model
                for model_key, result in results.items():
                    analysis_text = "Analysis" if language == 'en' else "Analiză"
                    st.markdown(f"""
                        <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 5px; margin: 1rem 0;'>
                            <h2 style='color: #4CAF50; margin: 0; border-bottom: 2px solid #4CAF50; padding-bottom: 0.5rem;'>
                                {result['model']} <span style='font-size: 0.8em; color: #666;'>{analysis_text}</span>
                            </h2>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Create two columns for metrics with improved styling
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "🎯 " + ("Dominant Emotion" if language == 'en' else "Emoție Dominantă"),
                            result['emotions']['dominant_emotion']
                        )
                    
                    with col2:
                        st.metric(
                            "📊 " + ("Intensity" if language == 'en' else "Intensitate"),
                            f"{result['emotions']['intensity']:.2f}"
                        )
                    
                    # Display analysis in a container with improved styling
                    with st.container():
                        st.markdown(f"<h3 style='color: #4CAF50;'>" + ("Analysis" if language == 'en' else "Analiză") + "</h3>", unsafe_allow_html=True)
                        st.info(result['emotions']['analysis'])
                    
                    # Display visualizations with improved styling
                    st.markdown(f"<h3 style='color: #4CAF50;'>" + ("Visual Analysis" if language == 'en' else "Analiză Vizuală") + "</h3>", unsafe_allow_html=True)
                    
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
                
                # Display comparison chart with improved styling
                st.markdown(f"<h2 style='color: #4CAF50;'>" + ("Model Comparison" if language == 'en' else "Comparare Modele") + "</h2>", unsafe_allow_html=True)
                st.plotly_chart(
                    create_comparison_chart(list(results.values())),
                    use_container_width=True,
                    key="comparison_chart"
                )
                
            except Exception as e:
                st.error("❌ " + (f"Error analyzing text: {str(e)}" if language == 'en' else f"Eroare la analizarea textului: {str(e)}"))
    else:
        st.warning("⚠️ " + ("Please enter some text for analysis." if language == 'en' else "Vă rugăm să introduceți text pentru analiză."))
