import torch
import re
from app.models.nicupiticu.model import NicupiticuModel
from sklearn.preprocessing import LabelEncoder
import torch.serialization
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import logging

# Add LabelEncoder to safe globals
torch.serialization.add_safe_globals([LabelEncoder])

logger = logging.getLogger(__name__)

class NicupiticuPredictor:
    def __init__(self, model_path='app/models/nicupiticu/nicupiticu_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.vocab = None
        self.label_encoder = None
        self.load_model(model_path)
        
    def load_model(self, model_path):
        try:
            # Load saved model and vocabulary
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Initialize model with same architecture
            self.model = NicupiticuModel(
                vocab_size=len(checkpoint['vocab']),
                num_classes=len(checkpoint['label_encoder'].classes_)
            ).to(self.device)
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Load vocabulary and label encoder
            self.vocab = checkpoint['vocab']
            self.label_encoder = checkpoint['label_encoder']
            
            print("Model loaded successfully")
            print(f"Vocabulary size: {len(self.vocab)}")
            print(f"Number of classes: {len(self.label_encoder.classes_)}")
            print(f"Classes: {self.label_encoder.classes_}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        
    def preprocess_text(self, text):
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
        
    def text_to_sequence(self, text, max_len=50):
        words = self.preprocess_text(text).split()
        sequence = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        if len(sequence) < max_len:
            sequence.extend([self.vocab['<PAD>']] * (max_len - len(sequence)))
        return sequence[:max_len]
    
    def predict(self, text):
        if not self.model or not self.vocab or not self.label_encoder:
            raise ValueError("Model not properly initialized")
            
        # Convert text to sequence
        sequence = self.text_to_sequence(text)
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.LongTensor([sequence]).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            
        # Get emotion label and confidence
        emotion = self.label_encoder.inverse_transform([predicted_class.item()])[0]
        confidence = probabilities[0][predicted_class].item()
        
        # Create probabilities dictionary with all emotions
        probabilities_dict = {}
        for i, label in enumerate(self.label_encoder.classes_):
            probabilities_dict[label] = probabilities[0][i].item()
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': probabilities_dict
        }

def main():
    # Example usage
    predictor = NicupiticuPredictor()
    
    test_texts = [
        "Sunt atât de fericit că am primit vestea bună!",
        "Mi-e dor de tine și nu știu ce să fac...",
        "Cum îndrăznești să-mi vorbești așa!",
        "Nu știu ce o să se întâmple cu mine...",
        "Nu-mi vine să cred ce am văzut!",
        "Astăzi este o zi normală.",
        "Te iubesc din toată inima!",
        "Mi-e scârbă de comportamentul tău!",
        "Totul va fi bine!",
        "Nu mai am speranță...",
        "Am încredere în tine!",
        "Abia aștept să vedem rezultatele!"
    ]
    
    print("\nTesting model predictions:")
    print("-" * 50)
    for text in test_texts:
        result = predictor.predict(text)
        print(f"\nText: {text}")
        print(f"Predicted emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("All probabilities:")
        for emotion, prob in result['probabilities'].items():
            print(f"  {emotion}: {prob:.2%}")

if __name__ == '__main__':
    main()

class NicupiticuModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=200, hidden_dim=256, num_classes=12, num_layers=2, dropout=0.5):
        super(NicupiticuModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # Use both directions of LSTM
        last_hidden = torch.cat((lstm_out[:, -1, :self.hidden_dim], lstm_out[:, 0, self.hidden_dim:]), dim=1)
        x = self.dropout(last_hidden)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)
        return output

def predict_emotions(text: str) -> list:
    """Predict emotions using the Nicupiticu model."""
    try:
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
        
        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Load model
        model = NicupiticuModel(
            vocab_size=len(tokenizer.vocab),
            embedding_dim=tokenizer.embedding_dim,
            hidden_dim=256,
            num_classes=12,
            num_layers=2,
            dropout=0.5
        )
        model.load_state_dict(torch.load("app/models/nicupiticu/nicupiticu_model.pt"))
        model.eval()
        
        # Get predictions
        with torch.no_grad():
            outputs = model(inputs['input_ids'])
            probabilities = F.softmax(outputs, dim=1)[0]
        
        # Map labels
        labels = [
            'bucurie', 'tristete', 'furie', 'frica', 'surpriza',
            'neutru', 'iubire', 'dezgust', 'optimism', 'pesimism',
            'incredere', 'anticipare'
        ]
        
        # Create predictions list
        predictions = [
            {'label': label, 'score': float(prob)}
            for label, prob in zip(labels, probabilities)
        ]
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error in Nicupiticu prediction: {str(e)}")
        raise

__all__ = ['predict_emotions'] 