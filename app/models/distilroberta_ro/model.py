import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple
import numpy as np
import logging
import os
from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RomanianDistilRoBERTaEmotionAnalyzer:
    def __init__(self):
        self.model_name = "readerbench/RoBERT-base"
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "distilroberta_ro")
        
        try:
            # Verificăm dacă modelul există în cache
            if not os.path.exists(self.cache_dir):
                logger.info(f"Modelul {self.model_name} nu este în cache. Se încearcă descărcarea...")
                # Descărcăm modelul în cache
                snapshot_download(
                    repo_id=self.model_name,
                    cache_dir=self.cache_dir,
                    local_files_only=False
                )
            
            # Încărcăm tokenizer-ul și modelul
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=True
            )
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=6,  # pentru cele 6 emoții de bază
                problem_type="multi_label_classification",
                cache_dir=self.cache_dir,
                local_files_only=True
            )
            
            self.emotions = ['bucurie', 'tristete', 'furie', 'frica', 'surpriza', 'neutru']
            logger.info("Modelul DistilRoBERTa Romanian a fost inițializat cu succes")
            
        except Exception as e:
            error_msg = f"""
            Nu s-a putut inițializa modelul DistilRoBERTa Romanian. 
            Cauze posibile:
            1. Lipsă conexiune la internet
            2. Spațiu insuficient pe disc
            3. Permisiuni insuficiente pentru directorul cache
            
            Eroare detaliată: {str(e)}
            
            Pentru a rezolva:
            1. Verificați conexiunea la internet
            2. Asigurați-vă că aveți cel puțin 2GB spațiu liber
            3. Rulați cu permisiuni de administrator sau modificați permisiunile pentru directorul cache
            """
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Preprocesează textul pentru modelul DistilRoBERTa."""
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