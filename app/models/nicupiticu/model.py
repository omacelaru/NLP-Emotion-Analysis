import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter

class RomanianEmotionDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class NicupiticuModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5):
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

class NicupiticuTrainer:
    def __init__(self, model_path='app/models/nicupiticu/data/training_data.csv'):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab = {}
        self.label_encoder = LabelEncoder()
        
    def preprocess_text(self, text):
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
        
    def create_vocabulary(self, texts):
        words = set()
        for text in texts:
            words.update(self.preprocess_text(text).split())
        self.vocab = {word: idx + 1 for idx, word in enumerate(words)}
        self.vocab['<PAD>'] = 0
        self.vocab['<UNK>'] = len(self.vocab)
        return len(self.vocab)
        
    def text_to_sequence(self, text, max_len=50):
        words = self.preprocess_text(text).split()
        sequence = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        if len(sequence) < max_len:
            sequence.extend([self.vocab['<PAD>']] * (max_len - len(sequence)))
        return sequence[:max_len]
    
    def prepare_data(self):
        # Load data
        df = pd.read_csv(self.model_path)
        texts = df['text'].values
        labels = df['emotion'].values
        
        # Print class distribution
        class_dist = Counter(labels)
        print("\nClass distribution in dataset:")
        for emotion, count in class_dist.items():
            print(f"{emotion}: {count} examples")
        
        # Create vocabulary
        vocab_size = self.create_vocabulary(texts)
        
        # Encode labels
        labels = self.label_encoder.fit_transform(labels)
        num_classes = len(self.label_encoder.classes_)
        
        # Convert texts to sequences
        sequences = [self.text_to_sequence(text) for text in texts]
        
        # Split data without stratification
        X_train, X_val, y_train, y_val = train_test_split(
            sequences, labels, test_size=0.2, random_state=42
        )
        
        # Print split distribution
        print("\nTraining set class distribution:")
        train_dist = Counter(y_train)
        for label, count in train_dist.items():
            emotion = self.label_encoder.inverse_transform([label])[0]
            print(f"{emotion}: {count} examples")
            
        print("\nValidation set class distribution:")
        val_dist = Counter(y_val)
        for label, count in val_dist.items():
            emotion = self.label_encoder.inverse_transform([label])[0]
            print(f"{emotion}: {count} examples")
        
        # Create datasets
        train_dataset = RomanianEmotionDataset(
            torch.LongTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = RomanianEmotionDataset(
            torch.LongTensor(X_val),
            torch.LongTensor(y_val)
        )
        
        return train_dataset, val_dataset, vocab_size, num_classes
    
    def train(self, epochs=50, batch_size=16, learning_rate=0.001):
        # Prepare data
        train_dataset, val_dataset, vocab_size, num_classes = self.prepare_data()
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        model = NicupiticuModel(
            vocab_size=vocab_size,
            embedding_dim=200,
            hidden_dim=256,
            num_classes=num_classes,
            num_layers=2,
            dropout=0.5
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        best_val_acc = 0
        patience = 10
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch_texts, batch_labels in train_loader:
                batch_texts, batch_labels = batch_texts.to(self.device), batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_texts)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            val_loss = 0
            class_correct = [0] * num_classes
            class_total = [0] * num_classes
            
            with torch.no_grad():
                for batch_texts, batch_labels in val_loader:
                    batch_texts, batch_labels = batch_texts.to(self.device), batch_labels.to(self.device)
                    outputs = model(batch_texts)
                    val_loss += criterion(outputs, batch_labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
                    
                    # Calculate per-class accuracy
                    for i in range(batch_labels.size(0)):
                        label = batch_labels[i]
                        pred = predicted[i]
                        if label == pred:
                            class_correct[label] += 1
                        class_total[label] += 1
            
            val_acc = 100 * correct / total
            scheduler.step(val_acc)
            
            print(f'\nEpoch {epoch+1}/{epochs}:')
            print(f'Training Loss: {total_loss/len(train_loader):.4f}')
            print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
            print(f'Validation Accuracy: {val_acc:.2f}%')
            
            # Print per-class accuracy
            print("\nPer-class validation accuracy:")
            for i in range(num_classes):
                if class_total[i] > 0:
                    accuracy = 100 * class_correct[i] / class_total[i]
                    emotion = self.label_encoder.inverse_transform([i])[0]
                    print(f"{emotion}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'vocab': self.vocab,
                    'label_encoder': self.label_encoder
                }, 'app/models/nicupiticu/nicupiticu_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
        
        return model

if __name__ == '__main__':
    trainer = NicupiticuTrainer()
    model = trainer.train() 