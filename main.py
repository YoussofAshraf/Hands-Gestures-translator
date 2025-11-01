
"""
Main Application
Real-time hand gesture recognition with text output
"""

import cv2
import time
import pyttsx3
from hand_detector import HandDetector
from gesture_classifier import GestureClassifier, SimpleGestureRecognizer


class GestureRecognitionApp:
    """
    Main gesture recognition application
    """
    
    def __init__(self, use_ml_model=True):
        """
        Initialize the application
        
        Args:
            use_ml_model: Use ML model if True, simple recognizer if False
        """
        self.detector = HandDetector(max_hands=1)
        self.use_ml_model = use_ml_model
        
        if use_ml_model:
            self.classifier = GestureClassifier()
        else:
            self.simple_recognizer = SimpleGestureRecognizer()
        
        # Text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speed
        
        # Recognition parameters
        self.confidence_threshold = 0.7
        self.recognized_text = ""
        self.current_gesture = ""
        self.last_gesture = ""
        self.gesture_stable_count = 0
        self.stability_threshold = 5  # Frames to consider gesture stable
        
        # FPS calculation
        self.prev_time = 0
        
    def process_frame(self, img):
        """
        Process a single frame
        
        Args:
            img: Input image
            
        Returns:
            Processed image with annotations
        """
        img = self.detector.find_hands(img, draw=True)
        landmark_list = self.detector.find_position(img, draw=False)
        
        gesture = "NO HAND"
        confidence = 0.0
        
        if len(landmark_list) > 0:
            if self.use_ml_model:
                # ML-based recognition
                features = self.detector.get_landmark_features(img)
                if len(features) > 0:
                    gesture, confidence = self.classifier.predict(features)
                    if gesture is None or confidence < self.confidence_threshold:
                        gesture = "UNCERTAIN"
            else:
                # Simple rule-based recognition
                fingers = self.detector.fingers_up(landmark_list)
                gesture = self.simple_recognizer.recognize_finger_count(fingers)
                confidence = 1.0
            
            # Check gesture stability
            if gesture == self.last_gesture:
                self.gesture_stable_count += 1
            else:
                self.gesture_stable_count = 0
                self.last_gesture = gesture
            
            # Add to text if gesture is stable
            if self.gesture_stable_count == self.stability_threshold:
                if gesture not in ["NO HAND", "UNCERTAIN", "UNKNOWN"]:
                    self.current_gesture = gesture
        
        return img, gesture, confidence
    
    def draw_ui(self, img, gesture, confidence):
        """
        Draw UI elements on the image
        
        Args:
            img: Input image
            gesture: Current gesture
            confidence: Confidence score
            
        Returns:
            Image with UI elements
        """
        h, w, c = img.shape
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = curr_time
        
        # Draw top bar
        cv2.rectangle(img, (0, 0), (w, 100), (50, 50, 50), -1)
        
        # FPS
        cv2.putText(img, f"FPS: {int(fps)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Current gesture
        color = (0, 255, 0) if confidence > self.confidence_threshold else (0, 165, 255)
        cv2.putText(img, f"Gesture: {gesture}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Confidence
        if self.use_ml_model:
            cv2.putText(img, f"Confidence: {confidence:.2f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw bottom text area
        cv2.rectangle(img, (0, h - 80), (w, h), (50, 50, 50), -1)
        cv2.putText(img, f"Text: {self.recognized_text}", (10, h - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(img, "Press: SPACE=Add | BACKSPACE=Delete | ENTER=Speak | Q=Quit", 
                   (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return img
    
    def speak_text(self, text):
        """
        Speak the recognized text
        
        Args:
            text: Text to speak
        """
        if text:
            print(f"Speaking: {text}")
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")
    
    def run(self):
        """
        Run the main application loop
        """
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)  # Width
        cap.set(4, 720)   # Height
        
        print("=" * 60)
        print("Hand Gesture Recognition System")
        print("=" * 60)
        print("\nControls:")
        print("  SPACE      - Add current gesture to text")
        print("  BACKSPACE  - Delete last character")
        print("  ENTER      - Speak recognized text")
        print("  C          - Clear text")
        print("  Q          - Quit")
        print("\nStarting camera...")
        
        while True:
            success, img = cap.read()
            if not success:
                print("Failed to capture frame")
                break
            
            img = cv2.flip(img, 1)  # Mirror image
            
            # Process frame
            img, gesture, confidence = self.process_frame(img)
            
            # Draw UI
            img = self.draw_ui(img, gesture, confidence)
            
            # Display
            cv2.imshow("Hand Gesture Recognition", img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\nQuitting...")
                break
            
            elif key == ord(' '):  # Space - add gesture
                if self.current_gesture and self.current_gesture not in ["NO HAND", "UNCERTAIN"]:
                    if self.current_gesture == "SPACE":
                        self.recognized_text += " "
                    elif self.current_gesture == "DELETE" or self.current_gesture == "BACKSPACE":
                        self.recognized_text = self.recognized_text[:-1]
                    else:
                        self.recognized_text += self.current_gesture
                    print(f"Added: {self.current_gesture}")
                    self.current_gesture = ""
            
            elif key == 8:  # Backspace
                self.recognized_text = self.recognized_text[:-1]
                print("Deleted last character")
            
            elif key == 13:  # Enter - speak
                self.speak_text(self.recognized_text)
            
            elif key == ord('c') or key == ord('C'):  # Clear
                self.recognized_text = ""
                print("Text cleared")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\nFinal text:", self.recognized_text)
        print("Application closed")


def main():
    """
    Main function
    """
    print("=" * 60)
    print("Hand Gesture to Text - Deaf Communication System")
    print("=" * 60)
    print("\nSelect recognition mode:")
    print("1. Simple rule-based recognition (no training required)")
    print("2. ML-based recognition (requires trained model)")
    
    choice = input("\nEnter choice (1 or 2, default=1): ").strip() or "1"
    
    use_ml = (choice == "2")
    
    if use_ml:
        print("\nUsing ML-based recognition")
        print("Note: Make sure you have trained a model first!")
    else:
        print("\nUsing simple rule-based recognition")
        print("This recognizes basic hand poses (number of fingers)")
    
    app = GestureRecognitionApp(use_ml_model=use_ml)
    app.run()


if __name__ == "__main__":
    main()
