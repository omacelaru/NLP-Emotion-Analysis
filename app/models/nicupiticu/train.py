from model import NicupiticuTrainer
import torch

def train_model():
    print("Starting Nicupiticu model training...")
    
    try:
        # Initialize trainer
        trainer = NicupiticuTrainer()
        
        # Train model
        model = trainer.train(
            epochs=50,
            batch_size=16,
            learning_rate=0.001
        )
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        raise

if __name__ == "__main__":
    train_model() 