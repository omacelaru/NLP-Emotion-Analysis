from typing import Dict, List, Union
import logging
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from app.models.nicupiticu.training.predict import NicupiticuPredictor
from app.models.distilroberta_ro.model import RomanianDistilRoBERTaEmotionAnalyzer

from app.config import MODEL_CONFIGS, EMOTION_LABELS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionAnalyzer:
    """Class for analyzing emotions in text using VADER, DistilRoBERTa (English), Nicupiticu, and DistilRoBERTa Romanian models."""

    def __init__(self):
        """Initialize the emotion analyzers with VADER, DistilRoBERTa (EN), Nicupiticu, and DistilRoBERTa Romanian."""
        try:
            # Initialize VADER
            self.vader = SentimentIntensityAnalyzer()

            # Initialize DistilRoBERTa for English
            self.distilroberta_en = pipeline(
                "text-classification",
                model=MODEL_CONFIGS['distilroberta_en']['model_id'],
                return_all_scores=True
            )

            # Initialize Nicupiticu model
            try:
                self.nicupiticu = NicupiticuPredictor()
                logger.info("Successfully initialized Nicupiticu model")
            except Exception as e:
                logger.warning(f"Nicupiticu model not available: {str(e)}")
                self.nicupiticu = None

            # Initialize DistilRoBERTa Romanian model
            try:
                self.distilroberta_ro = RomanianDistilRoBERTaEmotionAnalyzer()
                logger.info("Successfully initialized DistilRoBERTa Romanian model")
            except Exception as e:
                logger.warning(f"DistilRoBERTa Romanian model not available: {str(e)}")
                self.distilroberta_ro = None

            logger.info("Successfully initialized all emotion analyzers")
        except Exception as e:
            logger.error(f"Error initializing emotion analyzers: {str(e)}")
            raise

    def analyze_text(self, text: str, language: str = 'en') -> Dict:
        """Analyze text using available models based on language."""
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")

        try:
            results = {}

            # Get VADER analysis (only for English)
            if language == 'en':
                results['vader'] = self._analyze_vader(text)
                # Get DistilRoBERTa analysis for English
                results['distilroberta'] = self._analyze_distilroberta(text, self.distilroberta_en, 'en')
            else:
                # For Romanian, use both Nicupiticu and DistilRoBERTa Romanian if available
                if self.nicupiticu is not None:
                    results['nicupiticu'] = self._analyze_nicupiticu(text)
                if self.distilroberta_ro is not None:
                    results['distilroberta_ro'] = self._analyze_distilroberta_ro(text)

            return results
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            raise

    def _analyze_vader(self, text: str) -> Dict:
        """Analyze text using VADER model."""
        try:
            logger.info("VADER: Starting sentiment analysis...")
            # Get base VADER scores
            base_scores = self.vader.polarity_scores(text)
            logger.info(f"VADER: Raw scores - {base_scores}")

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

            logger.info(f"VADER: Dominant emotion: {dominant_emotion}, Intensity: {intensity:.2f}")

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
            logger.info(f"DistilRoBERTa ({language}): Starting emotion analysis...")
            # Get predictions from DistilRoBERTa
            predictions = model(text)[0]
            logger.info(f"DistilRoBERTa: Raw predictions - {predictions}")

            # For English, use the predictions directly
            emotion_scores = {pred['label']: pred['score'] for pred in predictions}

            # Sort emotions by score
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)

            # Get dominant emotion
            dominant_emotion = sorted_emotions[0][0]

            # Calculate intensity (using the highest score)
            intensity = sorted_emotions[0][1]

            logger.info(f"DistilRoBERTa: Top 3 emotions - {sorted_emotions[:3]}")
            logger.info(f"DistilRoBERTa: Dominant emotion: {dominant_emotion}, Intensity: {intensity:.2f}")

            return {
                'model': MODEL_CONFIGS[f'distilroberta_{language}']['name'],
                'emotions': {
                    'scores': emotion_scores,
                    'dominant_emotion': dominant_emotion,
                    'intensity': intensity,
                    'analysis': self._get_distilroberta_analysis(sorted_emotions, language)
                }
            }
        except Exception as e:
            logger.error(f"Error in DistilRoBERTa analysis: {str(e)}")
            raise

    def _analyze_nicupiticu(self, text: str) -> Dict:
        """Analyze text using Nicupiticu model."""
        try:
            # Get predictions from Nicupiticu
            result = self.nicupiticu.predict(text)

            # Create emotion scores dictionary
            emotion_scores = result['probabilities']

            # Sort emotions by score
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)

            return {
                'model': MODEL_CONFIGS['nicupiticu']['name'],
                'emotions': {
                    'scores': emotion_scores,
                    'dominant_emotion': result['emotion'],
                    'intensity': result['confidence'],
                    'analysis': self._get_nicupiticu_analysis(sorted_emotions)
                }
            }
        except Exception as e:
            logger.error(f"Error in Nicupiticu analysis: {str(e)}")
            raise

    def _analyze_distilroberta_ro(self, text: str) -> Dict:
        """Analyze text using DistilRoBERTa Romanian model."""
        try:
            # Get predictions from DistilRoBERTa Romanian
            emotions, scores = self.distilroberta_ro.predict(text)

            # Create emotion scores dictionary
            emotion_scores = dict(zip(emotions, scores))

            # Sort emotions by score
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)

            return {
                'model': MODEL_CONFIGS['distilroberta_ro']['name'],
                'emotions': {
                    'scores': emotion_scores,
                    'dominant_emotion': sorted_emotions[0][0] if sorted_emotions else 'neutru',
                    'intensity': sorted_emotions[0][1] if sorted_emotions else 0.0,
                    'analysis': self._get_distilroberta_ro_analysis(sorted_emotions)
                }
            }
        except Exception as e:
            logger.error(f"Error in DistilRoBERTa Romanian analysis: {str(e)}")
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

    def _get_distilroberta_analysis(self, sorted_emotions: List[tuple], language: str) -> str:
        """Generate a detailed analysis of the emotions detected by DistilRoBERTa."""
        top_emotions = sorted_emotions[:3]  # Get top 3 emotions

        analysis = f"Primary emotion: {top_emotions[0][0]} "
        analysis += f"({top_emotions[0][1]:.2f})"

        if len(top_emotions) > 1:
            secondary_emotions = []
            for emotion, score in top_emotions[1:]:
                secondary_emotions.append(f"{emotion} ({score:.2f})")
            analysis += f"\nSecondary emotions: {', '.join(secondary_emotions)}"

        return analysis

    def _get_nicupiticu_analysis(self, sorted_emotions: List[tuple]) -> str:
        """Generate a detailed analysis of the emotions detected by Nicupiticu."""
        top_emotions = sorted_emotions[:3]  # Get top 3 emotions

        analysis = f"Emoție principală: {top_emotions[0][0]} "
        analysis += f"({top_emotions[0][1]:.2f})"

        if len(top_emotions) > 1:
            secondary_emotions = []
            for emotion, score in top_emotions[1:]:
                secondary_emotions.append(f"{emotion} ({score:.2f})")
            analysis += f"\nEmoții secundare: {', '.join(secondary_emotions)}"

        return analysis

    def _get_distilroberta_ro_analysis(self, sorted_emotions: List[tuple]) -> str:
        """Generate a detailed analysis of the emotions detected by DistilRoBERTa Romanian."""
        if not sorted_emotions:
            return "Nu s-au detectat emoții clare în text."

        top_emotions = sorted_emotions[:3]  # Get top 3 emotions

        analysis = f"Emoție principală: {top_emotions[0][0]} "
        analysis += f"({top_emotions[0][1]:.2f})"

        if len(top_emotions) > 1:
            secondary_emotions = []
            for emotion, score in top_emotions[1:]:
                secondary_emotions.append(f"{emotion} ({score:.2f})")
            analysis += f"\nEmoții secundare: {', '.join(secondary_emotions)}"

        return analysis