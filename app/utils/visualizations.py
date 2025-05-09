from typing import Dict, List

import numpy as np
import plotly.graph_objects as go


def create_emotion_wheel(emotions: Dict[str, float], title: str) -> go.Figure:
    """Create a polar plot for the emotion wheel."""
    fig = go.Figure()

    # Convert emotions to polar coordinates
    theta = np.linspace(0, 2 * np.pi, len(emotions), endpoint=False)
    r = list(emotions.values())

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
        height=400,
        width=800
    )

    return fig


def create_detailed_scores_chart(emotions: Dict[str, float], title: str) -> go.Figure:
    """Create a bar chart for detailed emotion scores."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=list(emotions.keys()),
        y=list(emotions.values()),
        text=[f"{score:.2f}" for score in emotions.values()],
        textposition='auto',
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Emotions",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        height=400,
        width=800,
        showlegend=False
    )

    return fig


def create_comparison_chart(results: List[Dict]) -> go.Figure:
    """Create a bar chart comparing multiple models."""
    fig = go.Figure()

    # Get all unique emotions across all models
    all_emotions = set()
    for result in results:
        if 'scores' in result['emotions']:
            all_emotions.update(result['emotions']['scores'].keys())
        else:
            all_emotions.update(['pos', 'neg', 'neu'])

    # Create a bar for each model and emotion
    for result in results:
        model_name = result['model']
        emotions = result['emotions']

        if 'scores' in emotions:
            # DistilRoBERTa format
            emotion_dict = emotions['scores']
        else:
            # VADER format
            emotion_dict = {
                'pos': emotions['pos'],
                'neg': emotions['neg'],
                'neu': emotions['neu']
            }

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
        height=400,
        width=800,
        yaxis=dict(range=[0, 1])
    )

    return fig


def create_summary_table(results: List[Dict]) -> List[Dict]:
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


def create_sentiment_gauge(compound_score: float) -> go.Figure:
    """Create a gauge chart for sentiment score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=compound_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.5], 'color': "red"},
                {'range': [-0.5, 0], 'color': "orange"},
                {'range': [0, 0.5], 'color': "lightgreen"},
                {'range': [0.5, 1], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': compound_score
            }
        },
        title={'text': "Sentiment Score"}
    ))

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


def create_sentiment_breakdown(pos: float, neu: float, neg: float) -> go.Figure:
    """Create a pie chart for sentiment breakdown."""
    fig = go.Figure(data=[go.Pie(
        labels=['Positive', 'Neutral', 'Negative'],
        values=[pos, neu, neg],
        hole=.3,
        marker_colors=['green', 'gray', 'red']
    )])

    fig.update_layout(
        title="Sentiment Breakdown",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


def create_emotion_radar(scores):
    """Create a radar chart for emotion scores."""
    categories = ['Positive', 'Negative', 'Neutral']
    values = [scores['pos'], scores['neg'], scores['neu']]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Emotion Scores'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_emotion_trend(scores):
    """Create a line chart showing emotion trends."""
    categories = ['Positive', 'Negative', 'Neutral']
    values = [scores['pos'], scores['neg'], scores['neu']]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=categories,
        y=values,
        mode='lines+markers',
        name='Emotion Trend',
        line=dict(width=3),
        marker=dict(size=10)
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False,
            range=[0, 1]
        )
    )
    return fig
