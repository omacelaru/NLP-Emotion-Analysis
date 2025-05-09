import numpy as np
import plotly.graph_objects as go

# Model configurations
MODEL_CONFIGS = {
    'distilroberta_en': {
        'name': 'DistilRoBERTa (English)',
        'model_id': 'j-hartmann/emotion-english-distilroberta-base',
        'type': 'emotion',
        'language': 'en'
    },
    'distilroberta_ro': {
        'name': 'DistilRoBERTa (Romanian)',
        'model_id': 'dumitrescustefan/bert-base-romanian-cased-v1',
        'type': 'emotion',
        'language': 'ro'
    },
    'vader': {
        'name': 'VADER',
        'type': 'sentiment',
        'language': 'en'
    },
    'nicupiticu': {
        'name': 'Nicupiticu',
        'model_id': 'app/models/nicupiticu/nicupiticu_model.pt',
        'type': 'emotion',
        'language': 'ro'
    }
}

# Emotion labels
EMOTION_LABELS = {
    'en': {
        'joy': 'Joy',
        'sadness': 'Sadness',
        'anger': 'Anger',
        'fear': 'Fear',
        'surprise': 'Surprise',
        'neutral': 'Neutral',
        'love': 'Love',
        'disgust': 'Disgust',
        'optimism': 'Optimism',
        'pessimism': 'Pessimism',
        'trust': 'Trust',
        'anticipation': 'Anticipation'
    },
    'ro': {
        'bucurie': 'Bucurie',
        'tristete': 'Tristețe',
        'furie': 'Furie',
        'frica': 'Frică',
        'surpriza': 'Surpriză',
        'neutru': 'Neutru',
        'iubire': 'Iubire',
        'dezgust': 'Dezgust',
        'optimism': 'Optimism',
        'pesimism': 'Pesimism',
        'incredere': 'Încredere',
        'anticipare': 'Anticipare'
    }
}

# App settings
APP_TITLE = {
    'en': "Emotion Analysis",
    'ro': "Analiză Emoțională"
}

APP_DESCRIPTION = {
    'en': """
This platform provides a comprehensive emotion analysis using advanced natural language processing models. The analysis is performed using the following models:

- VADER: Sentiment analysis for English text
- DistilRoBERTa (English): Detailed emotion analysis for English text

Results are presented through interactive visualizations and detailed graphs for each model.
""",
    'ro': """
Această platformă oferă o analiză complexă a emoțiilor din text folosind modele avansate de procesare a limbajului natural. Analiza este realizată prin următoarele modele:

- DistilRoBERTa (Romanian): Analiză detaliată a emoțiilor pentru textul în română
- Nicupiticu: Analiză specializată pentru textul în română

Rezultatele sunt prezentate prin vizualizări interactive și grafice detaliate pentru fiecare model.
"""
}

# Visualization settings
CHART_HEIGHT = 400
CHART_WIDTH = 800


# Visualization functions
def create_emotion_wheel(emotions, title):
    """Create a polar plot for the emotion wheel."""
    fig = go.Figure()

    # Convert emotions to polar coordinates
    theta = np.linspace(0, 2 * np.pi, len(emotions), endpoint=False)
    r = [emotions[emotion] for emotion in emotions.keys()]

    fig.add_trace(go.Scatterpolar(
        r=r,
        theta=list(emotions.keys()),
        fill='toself',
        name=title
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title=title,
        height=CHART_HEIGHT,
        width=CHART_WIDTH
    )

    return fig


def create_detailed_scores_chart(emotions, title):
    """Create a bar chart for detailed emotion scores."""
    if isinstance(emotions, list):
        # DistilRoBERTa format
        labels = [e['label'] for e in emotions]
        scores = [e['score'] for e in emotions]
    else:
        # VADER format
        labels = list(emotions.keys())
        scores = list(emotions.values())

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=scores,
        text=[f"{score:.2f}" for score in scores],
        textposition='auto',
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Emotions",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        height=CHART_HEIGHT,
        width=CHART_WIDTH,
        showlegend=False
    )

    return fig


def create_comparison_chart(results):
    """Create a bar chart comparing multiple models."""
    fig = go.Figure()

    # Get all unique emotions across all models
    all_emotions = set()
    for result in results:
        if isinstance(result['emotions'], list):
            all_emotions.update(e['label'] for e in result['emotions'])
        else:
            all_emotions.update(result['emotions'].keys())

    # Create a bar for each model and emotion
    for result in results:
        model_name = result['model']
        emotions = result['emotions']

        if isinstance(emotions, list):
            # DistilRoBERTa format
            emotion_dict = {e['label']: e['score'] for e in emotions}
        else:
            # VADER format
            emotion_dict = emotions

        # Add scores for all emotions, using 0 for missing ones
        scores = [emotion_dict.get(emotion, 0) for emotion in all_emotions]

        fig.add_trace(go.Bar(
            name=model_name,
            x=list(all_emotions),
            y=scores,
            text=[f"{score:.2f}" for score in scores],
            textposition='auto',
        ))

    fig.update_layout(
        title="Model Comparison",
        xaxis_title="Emotions",
        yaxis_title="Score",
        barmode='group',
        showlegend=True,
        height=CHART_HEIGHT,
        width=CHART_WIDTH,
        yaxis=dict(range=[0, 1])
    )

    return fig


def create_summary_table(results):
    """Create a summary table of all results."""
    data = []
    for result in results:
        model_name = result['model']
        emotions = result['emotions']
        row = {'Model': model_name}

        if isinstance(emotions, list):
            for emotion in emotions:
                row[emotion['label']] = f"{emotion['score']:.2f}"
        else:
            row.update(emotions)

        data.append(row)

    return data
