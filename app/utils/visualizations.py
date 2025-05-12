from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def create_emotion_complexity_heatmap(emotions: Dict[str, float], complexity: float) -> go.Figure:
    """Create a heatmap showing emotion complexity and distribution."""
    # Create a 2D grid of emotions
    emotions_list = list(emotions.keys())
    n_emotions = len(emotions_list)
    grid_size = int(np.ceil(np.sqrt(n_emotions)))
    
    # Create a grid of emotion scores
    grid = np.zeros((grid_size, grid_size))
    for i, emotion in enumerate(emotions_list):
        row = i // grid_size
        col = i % grid_size
        grid[row, col] = emotions[emotion]
    
    fig = go.Figure(data=go.Heatmap(
        z=grid,
        colorscale='Viridis',
        showscale=True,
        text=[[f"{grid[i,j]:.2f}" if grid[i,j] > 0 else "" for j in range(grid_size)] for i in range(grid_size)],
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title=f"Emotion Complexity Heatmap (Complexity: {complexity:.2f})",
        height=400,
        width=600,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_ensemble_analysis_chart(results: Dict) -> go.Figure:
    """Create a comprehensive visualization of ensemble analysis results."""
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Emotion Distribution",
            "Model Confidence",
            "Emotion Complexity",
            "Top Emotions Comparison"
        ),
        specs=[
            [{"type": "pie"}, {"type": "indicator"}],
            [{"type": "bar"}, {"type": "bar"}]
        ]
    )
    
    # Get ensemble scores
    ensemble_scores = results['ensemble']['emotions']['scores']
    sorted_emotions = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 1. Emotion Distribution (Pie Chart)
    fig.add_trace(
        go.Pie(
            labels=list(ensemble_scores.keys()),
            values=list(ensemble_scores.values()),
            hole=.3,
            name="Distribution"
        ),
        row=1, col=1
    )
    
    # 2. Model Confidence (Gauge)
    confidence = results['ensemble']['emotions']['intensity']
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=confidence,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "red"},
                    {'range': [0.3, 0.7], 'color': "orange"},
                    {'range': [0.7, 1], 'color': "green"}
                ]
            },
            title={'text': "Confidence"}
        ),
        row=1, col=2
    )
    
    # 3. Emotion Complexity (Bar Chart)
    if 'complexity' in results['ensemble']['emotions']:
        complexity = results['ensemble']['emotions']['complexity']
        fig.add_trace(
            go.Bar(
                x=['Complexity'],
                y=[complexity],
                name="Complexity",
                marker_color='purple'
            ),
            row=2, col=1
        )
    
    # 4. Top Emotions Comparison (Bar Chart)
    top_emotions = sorted_emotions[:5]
    fig.add_trace(
        go.Bar(
            x=[e[0] for e in top_emotions],
            y=[e[1] for e in top_emotions],
            name="Top Emotions",
            marker_color='lightblue'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        width=1000,
        showlegend=True,
        title_text="Ensemble Analysis Dashboard",
        title_x=0.5
    )
    
    return fig


def create_model_comparison_radar(results: List[Dict]) -> go.Figure:
    """Create a radar chart comparing multiple models' predictions."""
    fig = go.Figure()
    
    # Get all unique emotions across models
    all_emotions = set()
    for result in results:
        if 'scores' in result['emotions']:
            all_emotions.update(result['emotions']['scores'].keys())
    
    all_emotions = sorted(list(all_emotions))
    
    # Add a trace for each model
    for result in results:
        model_name = result['model']
        emotions = result['emotions']
        
        if 'scores' in emotions:
            scores = [emotions['scores'].get(emotion, 0) for emotion in all_emotions]
        else:
            # Convert VADER scores to match emotion format
            scores = [
                emotions['pos'] if emotion == 'positive' else
                emotions['neg'] if emotion == 'negative' else
                emotions['neu'] if emotion == 'neutral' else 0
                for emotion in all_emotions
            ]
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=all_emotions,
            fill='toself',
            name=model_name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Model Comparison Radar",
        height=600,
        width=800
    )
    
    return fig
