from typing import Dict, List, Union
import logging
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.config import MODEL_CONFIGS, EMOTION_LABELS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionAnalyzer:
    """Class for analyzing emotions in text using both VADER and DistilRoBERTa."""

    def __init__(self):
        """Initialize the emotion analyzers with VADER and DistilRoBERTa."""
        try:
            # Initialize VADER
            self.vader = SentimentIntensityAnalyzer()
            
            # Initialize DistilRoBERTa models for both languages
            self.distilroberta_en = pipeline(
                "text-classification",
                model=MODEL_CONFIGS['distilroberta_en']['model_id'],
                return_all_scores=True
            )
            
            self.distilroberta_ro = pipeline(
                "text-classification",
                model=MODEL_CONFIGS['distilroberta_ro']['model_id'],
                return_all_scores=True
            )
            
            logger.info("Successfully initialized all emotion analyzers")
        except Exception as e:
            logger.error(f"Error initializing emotion analyzers: {str(e)}")
            raise

    def analyze_text(self, text: str, language: str = 'en') -> Dict:
        """Analyze text using both VADER and DistilRoBERTa models."""
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")

        try:
            results = {}
            
            # Get VADER analysis (only for English)
            if language == 'en':
                results['vader'] = self._analyze_vader(text)
            
            # Get DistilRoBERTa analysis based on language
            if language == 'en':
                results['distilroberta'] = self._analyze_distilroberta(text, self.distilroberta_en, 'en')
            else:
                results['distilroberta'] = self._analyze_distilroberta(text, self.distilroberta_ro, 'ro')
            
            return results
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            raise

    def _analyze_vader(self, text: str) -> Dict:
        """Analyze text using VADER model."""
        try:
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

            return {
                'model': MODEL_CONFIGS['vader']['name'],
                'emotions': {
                    'pos': positive_score,
                    'neg': negative_score,
                    'neu': neutral_score,
                    'compound': compound_score,
                    'dominant_emotion': dominant_emotion,
                    'intensity': intensity,
                    'analysis': self._get_emotion_analysis(compound_score, positive_score, negative_score)
                }
            }
        except Exception as e:
            logger.error(f"Error in VADER analysis: {str(e)}")
            raise

    def _analyze_distilroberta(self, text: str, model, language: str) -> Dict:
        """Analyze text using DistilRoBERTa model."""
        try:
            # Get predictions from DistilRoBERTa
            predictions = model(text)[0]
            
            # Sort predictions by score
            sorted_predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
            
            # Get dominant emotion
            dominant_emotion = sorted_predictions[0]['label']
            
            # Calculate intensity (using the highest score)
            intensity = sorted_predictions[0]['score']
            
            # Create detailed emotion scores
            emotion_scores = {
                pred['label']: pred['score'] for pred in sorted_predictions
            }
            
            return {
                'model': MODEL_CONFIGS[f'distilroberta_{language}']['name'],
                'emotions': {
                    'scores': emotion_scores,
                    'dominant_emotion': dominant_emotion,
                    'intensity': intensity,
                    'analysis': self._get_distilroberta_analysis(sorted_predictions, language)
                }
            }
        except Exception as e:
            logger.error(f"Error in DistilRoBERTa analysis: {str(e)}")
            raise

    def _get_emotion_analysis(self, compound: float, positive: float, negative: float) -> str:
        """Generate a detailed analysis of the emotions detected by VADER."""
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

    def _get_distilroberta_analysis(self, predictions: List[Dict], language: str) -> str:
        """Generate a detailed analysis of the emotions detected by DistilRoBERTa."""
        top_emotions = predictions[:3]  # Get top 3 emotions
        
        if language == 'en':
            analysis = f"Primary emotion: {top_emotions[0]['label']} "
            analysis += f"({top_emotions[0]['score']:.2f})"
            
            if len(top_emotions) > 1:
                secondary_emotions = []
                for e in top_emotions[1:]:
                    secondary_emotions.append(f"{e['label']} ({e['score']:.2f})")
                analysis += f"\nSecondary emotions: {', '.join(secondary_emotions)}"
        else:
            analysis = f"Emoție principală: {top_emotions[0]['label']} "
            analysis += f"({top_emotions[0]['score']:.2f})"
            
            if len(top_emotions) > 1:
                secondary_emotions = []
                for e in top_emotions[1:]:
                    secondary_emotions.append(f"{e['label']} ({e['score']:.2f})")
                analysis += f"\nEmoții secundare: {', '.join(secondary_emotions)}"
        
        return analysis
