"""
Data Collection Tool
Collect training data for gesture recognition
"""

import cv2
import os
import pickle
import numpy as np
from hand_detector import HandDetector


class DataCollector:
    """
    Collect gesture training data
    """
    
    def __init__(self, data_dir='gesture_data'):
        """
        Initialize data collector
        
        Args:
            data_dir: Directory to save collected data
        """
        self.data_dir = data_dir
        self.detector = HandDetector(max_hands=1)
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def collect_gesture_data(self, gesture_name, gesture_id, num_samples=100):
        """
        Collect samples for a specific gesture
        
        Args:
            gesture_name: Name of the gesture
            gesture_id: ID/label for the gesture
            num_samples: Number of samples to collect
        """
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)  # Width
        cap.set(4, 480)  # Height
        
        features_list = []
        labels_list = []
        
        collecting = False
        count = 0
        
        print(f"\nCollecting data for gesture: {gesture_name} (ID: {gesture_id})")
        print(f"Target samples: {num_samples}")
        print("\nInstructions:")
        print("- Press 'S' to START collecting")
        print("- Press 'Q' to QUIT")
        print("- Keep your hand in the frame and make the gesture")
        
        while True:
            success, img = cap.read()
            if not success:
                print("Failed to capture frame")
                break
            
            img = cv2.flip(img, 1)  # Mirror image
            img = self.detector.find_hands(img)
            features = self.detector.get_landmark_features(img)
            
            # Display info
            cv2.putText(img, f"Gesture: {gesture_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"Samples: {count}/{num_samples}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if collecting:
                cv2.putText(img, "COLLECTING...", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(img, "Press 'S' to start", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Collect data if started and hand is detected
            if collecting and len(features) > 0:
                features_list.append(features)
                labels_list.append(gesture_id)
                count += 1
                
                if count >= num_samples:
                    print(f"\nCompleted! Collected {count} samples")
                    break
            
            cv2.imshow("Data Collection", img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') or key == ord('S'):
                collecting = True
                print("Started collecting...")
            elif key == ord('q') or key == ord('Q'):
                print("\nCollection cancelled")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save collected data
        if len(features_list) > 0:
            self.save_data(gesture_name, gesture_id, features_list, labels_list)
            print(f"Data saved for gesture: {gesture_name}")
        
        return features_list, labels_list
    
    def save_data(self, gesture_name, gesture_id, features, labels):
        """
        Save collected data to file
        
        Args:
            gesture_name: Name of the gesture
            gesture_id: ID of the gesture
            features: List of feature vectors
            labels: List of labels
        """
        filename = os.path.join(self.data_dir, f"{gesture_name}_{gesture_id}.pkl")
        
        with open(filename, 'wb') as f:
            pickle.dump({
                'gesture_name': gesture_name,
                'gesture_id': gesture_id,
                'features': features,
                'labels': labels
            }, f)
    
    def load_all_data(self):
        """
        Load all collected data
        
        Returns:
            Tuple of (features, labels)
        """
        all_features = []
        all_labels = []
        
        if not os.path.exists(self.data_dir):
            print(f"Data directory {self.data_dir} does not exist")
            return all_features, all_labels
        
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.pkl')]
        
        if len(files) == 0:
            print("No data files found")
            return all_features, all_labels
        
        for file in files:
            filepath = os.path.join(self.data_dir, file)
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    all_features.extend(data['features'])
                    all_labels.extend(data['labels'])
                    print(f"Loaded {len(data['features'])} samples from {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        print(f"\nTotal samples loaded: {len(all_features)}")
        return all_features, all_labels
    
    def collect_multiple_gestures(self, gesture_list, samples_per_gesture=100):
        """
        Collect data for multiple gestures
        
        Args:
            gesture_list: List of tuples (gesture_name, gesture_id)
            samples_per_gesture: Number of samples per gesture
        """
        all_features = []
        all_labels = []
        
        for gesture_name, gesture_id in gesture_list:
            features, labels = self.collect_gesture_data(
                gesture_name, gesture_id, samples_per_gesture
            )
            all_features.extend(features)
            all_labels.extend(labels)
            
            print(f"\nCompleted {gesture_name}")
            print("Take a short break before the next gesture...")
            
            # Wait for user to press a key
            print("Press any key when ready for the next gesture...")
            cv2.waitKey(0)
        
        return all_features, all_labels


def main():
    """
    Main function for data collection
    """
    collector = DataCollector()
    
    print("=" * 50)
    print("Gesture Data Collection Tool")
    print("=" * 50)
    print("\nOptions:")
    print("1. Collect data for a single gesture")
    print("2. Collect data for multiple gestures (ASL letters)")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == '1':
        gesture_name = input("Enter gesture name: ")
        gesture_id = int(input("Enter gesture ID (number): "))
        num_samples = int(input("Enter number of samples (default 100): ") or "100")
        
        collector.collect_gesture_data(gesture_name, gesture_id, num_samples)
    
    elif choice == '2':
        # Common ASL letters
        gestures = [
            ('A', 0), ('B', 1), ('C', 2), ('D', 3), ('E', 4),
            ('F', 5), ('I', 8), ('K', 9), ('L', 10),
            ('O', 13), ('V', 20), ('Y', 23)
        ]
        
        print("\nWill collect data for the following gestures:")
        for name, id in gestures:
            print(f"  - {name} (ID: {id})")
        
        samples = int(input("\nSamples per gesture (default 100): ") or "100")
        
        collector.collect_multiple_gestures(gestures, samples)
    
    print("\nData collection completed!")


if __name__ == "__main__":
    main()
