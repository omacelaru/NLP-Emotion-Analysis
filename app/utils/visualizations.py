from typing import Dict, List

import plotly.graph_objects as go

from app.utils.config import CHART_HEIGHT, CHART_WIDTH


def create_emotion_wheel(emotions: Dict[str, float], title: str) -> go.Figure:
    """Create a polar plot for the emotion wheel."""
    # Define emotion categories and their colors
    emotion_categories = {
        'Positive': ['joy', 'trust', 'anticipation'],
        'Negative': ['anger', 'disgust', 'fear', 'sadness'],
        'Neutral': ['neutral']
    }

    colors = {
        'Positive': '#2ecc71',  # Green
        'Negative': '#e74c3c',  # Red
        'Neutral': '#3498db'  # Blue
    }

    fig = go.Figure()

    # Create a trace for each category
    for category, emotion_list in emotion_categories.items():
        # Calculate scores for this category
        category_scores = []
        for emotion in emotion_list:
            if emotion == 'joy':
                score = emotions.get('pos', 0)
            elif emotion == 'trust':
                score = (emotions.get('pos', 0) + emotions.get('compound', 0)) / 2
            elif emotion == 'anticipation':
                score = emotions.get('compound', 0)
            elif emotion == 'anger':
                score = emotions.get('neg', 0)
            elif emotion == 'disgust':
                score = emotions.get('neg', 0)
            elif emotion == 'fear':
                score = emotions.get('neg', 0)
            elif emotion == 'sadness':
                score = emotions.get('neg', 0)
            else:  # neutral
                score = emotions.get('neu', 0)
            category_scores.append(score)

        # Add trace for this category
        fig.add_trace(go.Scatterpolar(
            r=category_scores,
            theta=emotion_list,
            fill='toself',
            name=category,
            line_color=colors[category],
            fillcolor=colors[category]
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=True,
                ticks='outside',
                tickfont=dict(size=12),
                gridcolor='lightgray'
            ),
            angularaxis=dict(
                showticklabels=True,
                tickfont=dict(size=12),
                gridcolor='lightgray'
            ),
            bgcolor='white'
        ),
        showlegend=True,
        title=dict(
            text=title,
            font=dict(size=20, color='black'),
            x=0.5,
            y=0.95
        ),
        height=CHART_HEIGHT,
        width=CHART_WIDTH,
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    return fig


def create_detailed_scores_chart(emotions: Dict[str, float], title: str) -> go.Figure:
    """Create a bar chart for detailed emotion scores."""
    # Map VADER scores to detailed emotions
    detailed_emotions = {
        'Positive': emotions.get('pos', 0),
        'Negative': emotions.get('neg', 0),
        'Neutral': emotions.get('neu', 0),
        'Compound': emotions.get('compound', 0)
    }

    # Create color gradient based on values
    colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in detailed_emotions.values()]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=list(detailed_emotions.keys()),
        y=list(detailed_emotions.values()),
        text=[f"{score:.2f}" for score in detailed_emotions.values()],
        textposition='auto',
        marker_color=colors,
        marker_line_color='black',
        marker_line_width=1,
        opacity=0.8
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color='black'),
            x=0.5,
            y=0.95
        ),
        xaxis=dict(
            title="Emotions",
            titlefont=dict(size=14),
            tickfont=dict(size=12),
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title="Score",
            titlefont=dict(size=14),
            tickfont=dict(size=12),
            range=[-1, 1],
            gridcolor='lightgray'
        ),
        height=CHART_HEIGHT,
        width=CHART_WIDTH,
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        bargap=0.3
    )

    return fig


def create_comparison_chart(results: List[Dict]) -> go.Figure:
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


def create_sentiment_gauge(compound_score):
    """Create a gauge chart for the overall sentiment score."""
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
        title={'text': "Overall Sentiment Score"}
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_sentiment_breakdown(pos, neu, neg):
    """Create a stacked bar chart for sentiment breakdown."""
    fig = go.Figure()

    # Add bars for each sentiment
    fig.add_trace(go.Bar(
        name='Positive',
        y=['Sentiment'],
        x=[pos],
        orientation='h',
        marker_color='green'
    ))

    fig.add_trace(go.Bar(
        name='Neutral',
        y=['Sentiment'],
        x=[neu],
        orientation='h',
        marker_color='gray'
    ))

    fig.add_trace(go.Bar(
        name='Negative',
        y=['Sentiment'],
        x=[neg],
        orientation='h',
        marker_color='red'
    ))

    fig.update_layout(
        barmode='stack',
        height=150,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
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
