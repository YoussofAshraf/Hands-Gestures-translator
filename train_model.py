"""
Training Script
Train the gesture recognition model using collected data
"""

import os
from data_collector import DataCollector
from gesture_classifier import GestureClassifier


def train_model():
    """
    Train gesture recognition model
    """
    print("=" * 60)
    print("Gesture Recognition Model Training")
    print("=" * 60)
    
    # Initialize components
    collector = DataCollector()
    classifier = GestureClassifier()
    
    # Load training data
    print("\nLoading training data...")
    features, labels = collector.load_all_data()
    
    if len(features) == 0:
        print("\nNo training data found!")
        print("Please run data_collector.py first to collect gesture samples.")
        return
    
    print(f"\nLoaded {len(features)} samples")
    print(f"Feature dimension: {len(features[0])}")
    
    # Train model
    print("\nTraining model...")
    accuracy = classifier.train(features, labels)
    
    # Save model
    print("\nSaving model...")
    classifier.save_model()
    
    print("\n" + "=" * 60)
    print(f"Training completed with accuracy: {accuracy:.2%}")
    print("=" * 60)
    print("\nModel saved! You can now use it in the main application.")
    print("Run: python main.py")


if __name__ == "__main__":
    train_model()
