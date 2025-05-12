import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple
import numpy as np

class RomanianBERTEmotionAnalyzer:
    def __init__(self):
        self.model_name = "dumitrescustefan/bert-base-romanian-uncased-v1"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=6,  # pentru cele 6 emoții de bază
            problem_type="multi_label_classification"
        )
        self.emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']
        
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Preprocesează textul pentru modelul BERT."""
        return self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
    
    def predict(self, text: str) -> Tuple[List[str], List[float]]:
        """Prezice emoțiile pentru un text dat."""
        inputs = self.preprocess_text(text)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.sigmoid(outputs.logits)
            
        # Convertim predicțiile în emoții și scoruri
        scores = predictions[0].numpy()
        emotions = []
        emotion_scores = []
        
        for emotion, score in zip(self.emotions, scores):
            if score > 0.5:  # prag pentru a considera o emoție ca fiind prezentă
                emotions.append(emotion)
                emotion_scores.append(float(score))
        
        return emotions, emotion_scores
    
    def analyze_batch(self, texts: List[str]) -> List[Tuple[List[str], List[float]]]:
        """Analizează o listă de texte."""
        return [self.predict(text) for text in texts] 