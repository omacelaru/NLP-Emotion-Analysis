from typing import Dict, List, Union
import logging
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from app.models.nicupiticu.training.predict import NicupiticuPredictor
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import re
from collections import defaultdict

from app.config import MODEL_CONFIGS, EMOTION_LABELS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced Romanian emotion label mapping with sub-emotions
RO_EMOTION_MAPPING = {
    'LABEL_0': {'primary': 'bucurie', 'sub_emotions': ['entuziasm', 'fericire', 'satisfactie']},
    'LABEL_1': {'primary': 'tristete', 'sub_emotions': ['melancolie', 'dezamagire', 'regret']},
    'LABEL_2': {'primary': 'furie', 'sub_emotions': ['iritare', 'frustrare', 'indignare']},
    'LABEL_3': {'primary': 'frica', 'sub_emotions': ['anxietate', 'panică', 'neliniște']},
    'LABEL_4': {'primary': 'surpriza', 'sub_emotions': ['uimire', 'curiozitate', 'confuzie']},
    'LABEL_5': {'primary': 'neutru', 'sub_emotions': ['calm', 'echilibru', 'indiferență']},
    'LABEL_6': {'primary': 'iubire', 'sub_emotions': ['afecțiune', 'tendernță', 'apreciere']},
    'LABEL_7': {'primary': 'dezgust', 'sub_emotions': ['aversiune', 'repulsie', 'dispreț']},
    'LABEL_8': {'primary': 'optimism', 'sub_emotions': ['speranță', 'încredere', 'anticipare']},
    'LABEL_9': {'primary': 'pesimism', 'sub_emotions': ['deziluzie', 'resemnare', 'îndoială']},
    'LABEL_10': {'primary': 'incredere', 'sub_emotions': ['siguranță', 'convingere', 'stabilitate']},
    'LABEL_11': {'primary': 'anticipare', 'sub_emotions': ['așteptare', 'curiozitate', 'excitare']}
}

# Custom VADER lexicon for enhanced emotion detection
CUSTOM_VADER_LEXICON = {
    'excellent': 2.5, 'exceptional': 2.5, 'outstanding': 2.5,
    'terrible': -2.5, 'horrible': -2.5, 'awful': -2.5,
    'love': 2.0, 'adore': 2.0, 'passion': 2.0,
    'hate': -2.0, 'despise': -2.0, 'loathe': -2.0,
    'joy': 1.8, 'delight': 1.8, 'ecstasy': 1.8,
    'sadness': -1.8, 'grief': -1.8, 'sorrow': -1.8,
    'anger': -1.5, 'rage': -1.5, 'fury': -1.5,
    'fear': -1.5, 'terror': -1.5, 'dread': -1.5,
    'surprise': 1.0, 'amazement': 1.0, 'astonishment': 1.0,
    'trust': 1.5, 'confidence': 1.5, 'reliance': 1.5,
    'anticipation': 1.0, 'expectation': 1.0, 'hope': 1.0
}

class EmotionAnalyzer:
    """Class for analyzing emotions in text using enhanced VADER, DistilRoBERTa, and Nicupiticu models."""

    def __init__(self):
        """Initialize the emotion analyzers with enhanced capabilities."""
        try:
            # Initialize VADER with custom lexicon
            self.vader = SentimentIntensityAnalyzer()
            self.vader.lexicon.update(CUSTOM_VADER_LEXICON)
            
            # Initialize DistilRoBERTa models with enhanced processing
            self.distilroberta_en = pipeline(
                "text-classification",
                model=MODEL_CONFIGS['distilroberta_en']['model_id'],
                return_all_scores=True,
                truncation=True,
                max_length=512
            )
            
            self.distilroberta_ro = pipeline(
                "text-classification",
                model=MODEL_CONFIGS['distilroberta_ro']['model_id'],
                return_all_scores=True,
                truncation=True,
                max_length=512
            )
            
            # Initialize Nicupiticu model
            try:
                self.nicupiticu = NicupiticuPredictor()
                logger.info("Successfully initialized Nicupiticu model")
            except Exception as e:
                logger.warning(f"Nicupiticu model not available: {str(e)}")
                self.nicupiticu = None
            
            # Initialize scaler for emotion intensity normalization
            self.scaler = MinMaxScaler()
            
            logger.info("Successfully initialized all emotion analyzers")
        except Exception as e:
            logger.error(f"Error initializing emotion analyzers: {str(e)}")
            raise

    def _preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing for better emotion detection."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep punctuation for emotion detection
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Handle repeated punctuation for emphasis
        text = re.sub(r'([!?.])\1+', r'\1', text)
        
        return text

    def _analyze_vader(self, text: str) -> Dict:
        """Enhanced VADER analysis with mixed emotions detection."""
        try:
            logger.info("VADER: Starting enhanced sentiment analysis...")
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Get base VADER scores
            base_scores = self.vader.polarity_scores(processed_text)
            logger.info(f"VADER: Raw scores - {base_scores}")

            # Calculate enhanced emotion metrics
            compound_score = base_scores['compound']
            positive_score = base_scores['pos']
            negative_score = base_scores['neg']
            neutral_score = base_scores['neu']

            # Enhanced emotion detection
            emotions = {
                'positive': positive_score,
                'negative': negative_score,
                'neutral': neutral_score
            }
            
            # Detect mixed emotions
            mixed_emotions = []
            if positive_score > 0.3 and negative_score > 0.3:
                mixed_emotions.append('mixed_positive_negative')
            if positive_score > 0.3 and neutral_score > 0.3:
                mixed_emotions.append('mixed_positive_neutral')
            if negative_score > 0.3 and neutral_score > 0.3:
                mixed_emotions.append('mixed_negative_neutral')

            # Determine dominant emotion with enhanced logic
            if mixed_emotions:
                dominant_emotion = 'Mixed: ' + ', '.join(mixed_emotions)
            elif compound_score >= 0.05:
                if positive_score > 0.5:
                    dominant_emotion = "Strong Positive"
                else:
                    dominant_emotion = "Moderate Positive"
            elif compound_score <= -0.05:
                if negative_score > 0.5:
                    dominant_emotion = "Strong Negative"
                else:
                    dominant_emotion = "Moderate Negative"
            else:
                dominant_emotion = "Neutral"

            # Calculate enhanced intensity
            intensity = abs(compound_score)
            if mixed_emotions:
                intensity = max(positive_score, negative_score, neutral_score)

            logger.info(f"VADER: Enhanced analysis - Dominant: {dominant_emotion}, Intensity: {intensity:.2f}")

            return {
                'model': MODEL_CONFIGS['vader']['name'],
                'emotions': {
                    'pos': positive_score,
                    'neg': negative_score,
                    'neu': neutral_score,
                    'compound': compound_score,
                    'dominant_emotion': dominant_emotion,
                    'intensity': intensity,
                    'mixed_emotions': mixed_emotions,
                    'analysis': self._get_enhanced_emotion_analysis(compound_score, positive_score, negative_score, mixed_emotions)
                }
            }
        except Exception as e:
            logger.error(f"Error in enhanced VADER analysis: {str(e)}")
            raise

    def _analyze_distilroberta(self, text: str, model, language: str) -> Dict:
        """Enhanced DistilRoBERTa analysis with sub-emotions detection."""
        try:
            logger.info(f"DistilRoBERTa ({language}): Starting enhanced emotion analysis...")
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Get predictions with enhanced processing
            predictions = model(processed_text)[0]
            logger.info(f"DistilRoBERTa: Raw predictions - {predictions}")
            
            # Create enhanced emotion dictionary
            emotion_scores = {}
            sub_emotions = defaultdict(list)
            
            if language == 'ro':
                # Initialize all Romanian emotions with 0
                for emotion in EMOTION_LABELS['ro'].values():
                    emotion_scores[emotion] = 0.0
                
                # Update with actual predictions and sub-emotions
                for pred in predictions:
                    emotion_data = RO_EMOTION_MAPPING.get(pred['label'], {'primary': pred['label'], 'sub_emotions': []})
                    primary_emotion = emotion_data['primary']
                    emotion_scores[primary_emotion] = pred['score']
                    
                    # Add sub-emotions if score is significant
                    if pred['score'] > 0.3:
                        for sub_emotion in emotion_data['sub_emotions']:
                            sub_emotions[primary_emotion].append({
                                'emotion': sub_emotion,
                                'score': pred['score'] * 0.8  # Sub-emotions have slightly lower scores
                            })
            else:
                # For English, use the predictions directly
                emotion_scores = {pred['label']: pred['score'] for pred in predictions}
            
            # Sort emotions by score
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Get dominant emotion
            dominant_emotion = sorted_emotions[0][0]
            
            # Calculate enhanced intensity
            intensity = sorted_emotions[0][1]
            
            # Calculate emotion complexity
            complexity = self._calculate_emotion_complexity(emotion_scores)
            
            logger.info(f"DistilRoBERTa: Enhanced analysis - Top 3 emotions: {sorted_emotions[:3]}")
            
            return {
                'model': MODEL_CONFIGS[f'distilroberta_{language}']['name'],
                'emotions': {
                    'scores': emotion_scores,
                    'sub_emotions': dict(sub_emotions),
                    'dominant_emotion': dominant_emotion,
                    'intensity': intensity,
                    'complexity': complexity,
                    'analysis': self._get_enhanced_distilroberta_analysis(sorted_emotions, sub_emotions, complexity, language)
                }
            }
        except Exception as e:
            logger.error(f"Error in enhanced DistilRoBERTa analysis: {str(e)}")
            raise

    def _get_enhanced_emotion_analysis(self, compound: float, positive: float, negative: float, mixed_emotions: List[str]) -> str:
        """Generate enhanced detailed analysis of emotions detected by VADER."""
        analysis = []
        
        if mixed_emotions:
            analysis.append("The text shows mixed emotions:")
            for emotion in mixed_emotions:
                if emotion == 'mixed_positive_negative':
                    analysis.append("- Strong contrast between positive and negative feelings")
                elif emotion == 'mixed_positive_neutral':
                    analysis.append("- Combination of positive and neutral tones")
                elif emotion == 'mixed_negative_neutral':
                    analysis.append("- Mix of negative and neutral sentiments")
        else:
            if compound >= 0.05:
                if positive > 0.5:
                    analysis.append("Strong positive sentiment with high enthusiasm")
                    if positive > 0.7:
                        analysis.append("The text expresses very strong positive emotions")
                else:
                    analysis.append("Moderate positive sentiment")
            elif compound <= -0.05:
                if negative > 0.5:
                    analysis.append("Strong negative sentiment with high intensity")
                    if negative > 0.7:
                        analysis.append("The text expresses very strong negative emotions")
                else:
                    analysis.append("Moderate negative sentiment")
            else:
                analysis.append("Neutral sentiment with balanced emotions")
        
        return "\n".join(analysis)

    def _get_enhanced_distilroberta_analysis(self, sorted_emotions: List[tuple], sub_emotions: Dict, complexity: float, language: str) -> str:
        """Generate enhanced detailed analysis of emotions detected by DistilRoBERTa."""
        analysis = []
        
        if language == 'en':
            # Primary emotion analysis
            analysis.append(f"Primary emotion: {sorted_emotions[0][0]} ({sorted_emotions[0][1]:.2f})")
            
            # Add sub-emotions if available
            if sorted_emotions[0][0] in sub_emotions:
                sub_emotion_list = [f"{se['emotion']} ({se['score']:.2f})" for se in sub_emotions[sorted_emotions[0][0]]]
                analysis.append(f"Sub-emotions: {', '.join(sub_emotion_list)}")
            
            # Add secondary emotions
            if len(sorted_emotions) > 1:
                secondary = []
                for emotion, score in sorted_emotions[1:3]:  # Get top 2 secondary emotions
                    secondary.append(f"{emotion} ({score:.2f})")
                analysis.append(f"Secondary emotions: {', '.join(secondary)}")
            
            # Add complexity analysis
            if complexity > 1.5:
                analysis.append("The text shows complex emotional patterns with multiple emotions present.")
            elif complexity > 1.0:
                analysis.append("The text shows moderate emotional complexity.")
            else:
                analysis.append("The text shows focused emotional expression.")
        else:
            # Romanian analysis
            analysis.append(f"Emoție principală: {sorted_emotions[0][0]} ({sorted_emotions[0][1]:.2f})")
            
            if sorted_emotions[0][0] in sub_emotions:
                sub_emotion_list = [f"{se['emotion']} ({se['score']:.2f})" for se in sub_emotions[sorted_emotions[0][0]]]
                analysis.append(f"Sub-emoții: {', '.join(sub_emotion_list)}")
            
            if len(sorted_emotions) > 1:
                secondary = []
                for emotion, score in sorted_emotions[1:3]:
                    secondary.append(f"{emotion} ({score:.2f})")
                analysis.append(f"Emoții secundare: {', '.join(secondary)}")
            
            if complexity > 1.5:
                analysis.append("Textul prezintă modele emoționale complexe cu multiple emoții.")
            elif complexity > 1.0:
                analysis.append("Textul prezintă o complexitate emoțională moderată.")
            else:
                analysis.append("Textul prezintă o expresie emoțională focalizată.")
        
        return "\n".join(analysis)

    def analyze_text(self, text: str, language: str = 'en') -> Dict:
        """Analyze text using VADER, DistilRoBERTa, and BERT models."""
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
                # Add BERT emotion analysis for English
                results['bert'] = self._analyze_bert(text)
            else:
                results['distilroberta'] = self._analyze_distilroberta(text, self.distilroberta_ro, 'ro')
                
                # Add Nicupiticu analysis if available
                if self.nicupiticu is not None:
                    results['nicupiticu'] = self._analyze_nicupiticu(text)
            
            # Add ensemble analysis
            results['ensemble'] = self._ensemble_analysis(results)
            
            return results
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            raise

    def _analyze_bert(self, text: str) -> Dict:
        """Analyze text using BERT emotion model."""
        try:
            logger.info("BERT: Starting emotion analysis...")
            predictions = self.bert_emotion(text)[0]
            
            # Create emotion scores dictionary
            emotion_scores = {pred['label']: pred['score'] for pred in predictions}
            
            # Sort emotions by score
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Get dominant emotion
            dominant_emotion = sorted_emotions[0][0]
            
            # Calculate intensity
            intensity = sorted_emotions[0][1]
            
            # Calculate emotion complexity (entropy of emotion distribution)
            emotion_complexity = self._calculate_emotion_complexity(emotion_scores)
            
            return {
                'model': 'BERT Emotion',
                'emotions': {
                    'scores': emotion_scores,
                    'dominant_emotion': dominant_emotion,
                    'intensity': intensity,
                    'complexity': emotion_complexity,
                    'analysis': self._get_bert_analysis(sorted_emotions, emotion_complexity)
                }
            }
        except Exception as e:
            logger.error(f"Error in BERT analysis: {str(e)}")
            raise

    def _calculate_emotion_complexity(self, emotion_scores: Dict[str, float]) -> float:
        """Calculate the complexity of emotions using entropy."""
        scores = np.array(list(emotion_scores.values()))
        scores = scores / scores.sum()  # Normalize to probabilities
        entropy = -np.sum(scores * np.log2(scores + 1e-10))
        return entropy

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
        
        if language == 'en':
            analysis = f"Primary emotion: {top_emotions[0][0]} "
            analysis += f"({top_emotions[0][1]:.2f})"
            
            if len(top_emotions) > 1:
                secondary_emotions = []
                for emotion, score in top_emotions[1:]:
                    secondary_emotions.append(f"{emotion} ({score:.2f})")
                analysis += f"\nSecondary emotions: {', '.join(secondary_emotions)}"
        else:
            analysis = f"Emoție principală: {top_emotions[0][0]} "
            analysis += f"({top_emotions[0][1]:.2f})"
            
            if len(top_emotions) > 1:
                secondary_emotions = []
                for emotion, score in top_emotions[1:]:
                    secondary_emotions.append(f"{emotion} ({score:.2f})")
                analysis += f"\nEmoții secundare: {', '.join(secondary_emotions)}"
        
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

    def _ensemble_analysis(self, results: Dict) -> Dict:
        """Combine results from multiple models for ensemble analysis."""
        try:
            # Collect all emotion scores
            all_scores = {}
            weights = {
                'vader': 0.2,
                'distilroberta': 0.3,
                'bert': 0.3,
                'nicupiticu': 0.2
            }
            
            for model_key, result in results.items():
                if model_key in weights:
                    if 'scores' in result['emotions']:
                        for emotion, score in result['emotions']['scores'].items():
                            if emotion not in all_scores:
                                all_scores[emotion] = 0
                            all_scores[emotion] += score * weights[model_key]
            
            # Normalize scores
            if all_scores:
                total = sum(all_scores.values())
                all_scores = {k: v/total for k, v in all_scores.items()}
            
            # Sort emotions by score
            sorted_emotions = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'model': 'Ensemble Analysis',
                'emotions': {
                    'scores': all_scores,
                    'dominant_emotion': sorted_emotions[0][0] if sorted_emotions else 'unknown',
                    'intensity': sorted_emotions[0][1] if sorted_emotions else 0.0,
                    'analysis': self._get_ensemble_analysis(sorted_emotions)
                }
            }
        except Exception as e:
            logger.error(f"Error in ensemble analysis: {str(e)}")
            raise

    def _get_bert_analysis(self, sorted_emotions: List[tuple], complexity: float) -> str:
        """Generate a detailed analysis of the emotions detected by BERT."""
        top_emotions = sorted_emotions[:3]
        
        analysis = f"Primary emotion: {top_emotions[0][0]} ({top_emotions[0][1]:.2f})\n"
        
        if len(top_emotions) > 1:
            secondary_emotions = []
            for emotion, score in top_emotions[1:]:
                secondary_emotions.append(f"{emotion} ({score:.2f})")
            analysis += f"Secondary emotions: {', '.join(secondary_emotions)}\n"
        
        # Add complexity analysis
        if complexity > 1.5:
            analysis += "The text shows complex emotional patterns with multiple emotions present."
        elif complexity > 1.0:
            analysis += "The text shows moderate emotional complexity."
        else:
            analysis += "The text shows focused emotional expression."
        
        return analysis

    def _get_ensemble_analysis(self, sorted_emotions: List[tuple]) -> str:
        """Generate a detailed analysis combining results from all models."""
        top_emotions = sorted_emotions[:3]
        
        analysis = "Ensemble Analysis Results:\n"
        analysis += f"Primary emotion: {top_emotions[0][0]} ({top_emotions[0][1]:.2f})\n"
        
        if len(top_emotions) > 1:
            secondary_emotions = []
            for emotion, score in top_emotions[1:]:
                secondary_emotions.append(f"{emotion} ({score:.2f})")
            analysis += f"Secondary emotions: {', '.join(secondary_emotions)}\n"
        
        # Add confidence analysis
        if top_emotions[0][1] > 0.7:
            analysis += "High confidence in emotion detection across models."
        elif top_emotions[0][1] > 0.5:
            analysis += "Moderate confidence in emotion detection."
        else:
            analysis += "Multiple emotions detected with varying confidence levels."
        
        return analysis
