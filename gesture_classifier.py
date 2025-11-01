"""
Gesture Classification Module
Classifies hand gestures based on landmark features
"""

import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class GestureClassifier:
    """
    Gesture classifier using machine learning
    """
    
    def __init__(self, model_path='gesture_model.pkl'):
        """
        Initialize gesture classifier
        
        Args:
            model_path: Path to save/load the trained model
        """
        self.model_path = model_path
        self.model = None
        self.label_map = {}
        
        # Try to load existing model
        if os.path.exists(model_path):
            self.load_model()
        else:
            # Initialize with default ASL gestures
            self.initialize_default_gestures()
    
    def initialize_default_gestures(self):
        """
        Initialize with common gesture labels (ASL alphabet)
        """
        # Basic ASL alphabet letters (some are static, some are dynamic)
        self.label_map = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
            5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
            10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P',
            15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U',
            20: 'V', 21: 'W', 22: 'X', 23: 'Y',
            24: 'HELLO', 25: 'THANK YOU', 26: 'PLEASE',
            27: 'SPACE', 28: 'DELETE'
        }
        
        # Initialize Random Forest classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def train(self, features, labels):
        """
        Train the gesture classifier
        
        Args:
            features: List of feature vectors
            labels: List of corresponding labels
            
        Returns:
            Training accuracy
        """
        if len(features) == 0 or len(labels) == 0:
            print("No training data provided!")
            return 0.0
        
        features = np.array(features)
        labels = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Training completed with accuracy: {accuracy:.2%}")
        
        return accuracy
    
    def predict(self, features):
        """
        Predict gesture from features
        
        Args:
            features: Feature vector
            
        Returns:
            Predicted label and confidence
        """
        if self.model is None:
            return None, 0.0
        
        if len(features) == 0:
            return None, 0.0
        
        features = np.array(features).reshape(1, -1)
        
        try:
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = np.max(probabilities)
            
            label = self.label_map.get(prediction, f"UNKNOWN_{prediction}")
            
            return label, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0
    
    def save_model(self):
        """
        Save trained model to file
        """
        if self.model is None:
            print("No model to save!")
            return
        
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_map': self.label_map
            }, f)
        
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """
        Load trained model from file
        """
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.label_map = data['label_map']
            
            print(f"Model loaded from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.initialize_default_gestures()
    
    def add_gesture(self, label, gesture_id=None):
        """
        Add a new gesture to the label map
        
        Args:
            label: Label for the gesture
            gesture_id: ID for the gesture (auto-generated if None)
            
        Returns:
            Gesture ID
        """
        if gesture_id is None:
            gesture_id = max(self.label_map.keys()) + 1 if self.label_map else 0
        
        self.label_map[gesture_id] = label
        return gesture_id
    
    def get_gesture_labels(self):
        """
        Get all gesture labels
        
        Returns:
            List of gesture labels
        """
        return list(self.label_map.values())


class SimpleGestureRecognizer:
    """
    Simple rule-based gesture recognizer for basic hand poses
    Can work without training data
    """
    
    def __init__(self):
        """
        Initialize simple gesture recognizer
        """
        pass
    
    def recognize_finger_count(self, fingers_up):
        """
        Recognize gesture based on number of fingers up
        
        Args:
            fingers_up: List of 5 values (0 or 1) for each finger
            
        Returns:
            Gesture name
        """
        if len(fingers_up) != 5:
            return "UNKNOWN"
        
        count = sum(fingers_up)
        
        # Number gestures
        if count == 0:
            return "0 / A / S"
        elif count == 1:
            if fingers_up[1] == 1:  # Index finger
                return "1 / D"
            elif fingers_up[0] == 1:  # Thumb
                return "LIKE / 10"
        elif count == 2:
            if fingers_up[1] == 1 and fingers_up[2] == 1:  # Index and middle
                return "2 / V / PEACE"
            elif fingers_up[0] == 1 and fingers_up[1] == 1:  # Thumb and index
                return "L"
        elif count == 3:
            if fingers_up[0] == 1 and fingers_up[1] == 1 and fingers_up[4] == 1:  # Thumb, index, pinky
                return "I LOVE YOU"
            else:
                return "3 / W"
        elif count == 4:
            return "4 / B"
        elif count == 5:
            return "5 / HELLO / STOP"
        
        return f"{count} FINGERS"
    
    def recognize_from_landmarks(self, landmark_list):
        """
        Recognize gesture from landmark positions
        
        Args:
            landmark_list: List of landmark positions
            
        Returns:
            Gesture name
        """
        if len(landmark_list) < 21:
            return "NO HAND"
        
        # Calculate distances and angles for more complex gestures
        # This is a simplified version
        
        # Check if hand is open or closed
        fingertip_ids = [4, 8, 12, 16, 20]
        fingertip_positions = [landmark_list[i] for i in fingertip_ids]
        
        # More complex gesture recognition can be added here
        
        return "HAND DETECTED"
