from typing import Dict

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.utils.config import MODEL_CONFIGS


class EmotionAnalyzer:
    """Class for analyzing emotions in text using VADER."""

    def __init__(self):
        """Initialize the emotion analyzer with VADER."""
        self.vader = SentimentIntensityAnalyzer()

    def analyze_text(self, text: str) -> Dict:
        """Analyze text using VADER model and provide detailed emotion analysis."""
        # Get base VADER scores
        base_scores = self.vader.polarity_scores(text)

        # Calculate additional emotion metrics
        compound_score = base_scores['compound']
        positive_score = base_scores['pos']
        negative_score = base_scores['neg']
        neutral_score = base_scores['neu']

        # Determine dominant emotion
        if compound_score >= 0.05:
            dominant_emotion = "Positive"
        elif compound_score <= -0.05:
            dominant_emotion = "Negative"
        else:
            dominant_emotion = "Neutral"

        # Calculate intensity
        intensity = abs(compound_score)

        # Create complete emotions dictionary
        emotions = {
            'pos': positive_score,
            'neg': negative_score,
            'neu': neutral_score,
            'compound': compound_score,
            'dominant_emotion': dominant_emotion,
            'intensity': intensity,
            'analysis': self._get_emotion_analysis(compound_score, positive_score, negative_score)
        }

        return {
            'model': MODEL_CONFIGS['vader']['name'],
            'emotions': emotions
        }

    def _get_emotion_analysis(self, compound: float, positive: float, negative: float) -> str:
        """Generate a detailed analysis of the emotions detected."""
        if compound >= 0.05:
            if positive > 0.5:
                return "Strong positive sentiment with high enthusiasm"
            else:
                return "Moderate positive sentiment"
        elif compound <= -0.05:
            if negative > 0.5:
                return "Strong negative sentiment with high intensity"
            else:
                return "Moderate negative sentiment"
        else:
            return "Neutral sentiment with balanced emotions"
