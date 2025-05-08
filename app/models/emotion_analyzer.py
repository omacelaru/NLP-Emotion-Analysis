from config import MODEL_CONFIGS
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class EmotionAnalyzer:
    def __init__(self):
        """Initialize the emotion analyzer with DistilRoBERTa and VADER models."""
        self.distilroberta = pipeline(
            "text-classification",
            model=MODEL_CONFIGS['distilroberta']['model_id'],
            return_all_scores=True
        )
        self.vader = SentimentIntensityAnalyzer()

    def analyze_distilroberta(self, text):
        """Analyze text using DistilRoBERTa model."""
        results = self.distilroberta(text)[0]
        return [{'label': r['label'], 'score': r['score']} for r in results]

    def analyze_vader(self, text):
        """Analyze text using VADER sentiment analyzer."""
        scores = self.vader.polarity_scores(text)
        return {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound']
        }

    def analyze_text(self, text, selected_models):
        """Analyze text using selected models and return results."""
        results = []

        for model_key in selected_models:
            if model_key == 'distilroberta':
                emotions = self.analyze_distilroberta(text)
                results.append({
                    'model': MODEL_CONFIGS[model_key]['name'],
                    'emotions': emotions
                })
            elif model_key == 'vader':
                emotions = self.analyze_vader(text)
                results.append({
                    'model': MODEL_CONFIGS[model_key]['name'],
                    'emotions': emotions
                })

        return results
