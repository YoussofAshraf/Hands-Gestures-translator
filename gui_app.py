"""
GUI Application
Enhanced user interface for gesture recognition
"""

import cv2
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import threading
import pyttsx3
from hand_detector import HandDetector
from gesture_classifier import GestureClassifier, SimpleGestureRecognizer


class GestureRecognitionGUI:
    """
    GUI application for gesture recognition
    """
    
    def __init__(self, window, window_title):
        """
        Initialize GUI
        
        Args:
            window: Tkinter window
            window_title: Window title
        """
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1200x700")
        
        # Initialize components
        self.detector = HandDetector(max_hands=1)
        self.classifier = GestureClassifier()
        self.simple_recognizer = SimpleGestureRecognizer()
        
        # TTS engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        
        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        
        # Variables
        self.use_ml_model = tk.BooleanVar(value=False)
        self.confidence_threshold = tk.DoubleVar(value=0.7)
        self.recognized_text = ""
        self.current_gesture = ""
        self.last_gesture = ""
        self.gesture_stable_count = 0
        self.is_running = True
        
        # Create GUI
        self.create_widgets()
        
        # Start video loop
        self.update_video()
        
        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        """
        Create GUI widgets
        """
        # Main container
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel - Video
        video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="5")
        video_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.N, tk.S, tk.W, tk.E))
        
        self.canvas = tk.Canvas(video_frame, width=640, height=480, bg='black')
        self.canvas.pack()
        
        # Right panel - Controls and output
        right_frame = ttk.Frame(main_frame, padding="5")
        right_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.N, tk.S, tk.W, tk.E))
        right_frame.columnconfigure(0, weight=1)
        
        # Settings
        settings_frame = ttk.LabelFrame(right_frame, text="Settings", padding="10")
        settings_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        settings_frame.columnconfigure(0, weight=1)
        
        ttk.Checkbutton(settings_frame, text="Use ML Model", 
                       variable=self.use_ml_model).grid(row=0, column=0, sticky=tk.W)
        
        ttk.Label(settings_frame, text="Confidence Threshold:").grid(row=1, column=0, sticky=tk.W)
        ttk.Scale(settings_frame, from_=0.0, to=1.0, variable=self.confidence_threshold,
                 orient=tk.HORIZONTAL).grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        self.confidence_label = ttk.Label(settings_frame, text="0.70")
        self.confidence_label.grid(row=2, column=1, padx=5)
        
        # Current gesture
        gesture_frame = ttk.LabelFrame(right_frame, text="Current Gesture", padding="10")
        gesture_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.gesture_label = ttk.Label(gesture_frame, text="NO HAND", 
                                       font=('Arial', 16, 'bold'), foreground='blue')
        self.gesture_label.pack()
        
        self.confidence_value_label = ttk.Label(gesture_frame, text="Confidence: 0.00")
        self.confidence_value_label.pack()
        
        # Recognized text
        text_frame = ttk.LabelFrame(right_frame, text="Recognized Text", padding="10")
        text_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        self.text_display = scrolledtext.ScrolledText(text_frame, height=10, wrap=tk.WORD)
        self.text_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Buttons
        button_frame = ttk.Frame(right_frame, padding="5")
        button_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)
        
        ttk.Button(button_frame, text="Add Gesture", 
                  command=self.add_gesture).grid(row=0, column=0, padx=2, sticky=(tk.W, tk.E))
        ttk.Button(button_frame, text="Clear Text", 
                  command=self.clear_text).grid(row=0, column=1, padx=2, sticky=(tk.W, tk.E))
        ttk.Button(button_frame, text="Speak Text", 
                  command=self.speak_text).grid(row=0, column=2, padx=2, sticky=(tk.W, tk.E))
        
        # Instructions
        instructions_frame = ttk.LabelFrame(right_frame, text="Instructions", padding="10")
        instructions_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5)
        
        instructions = """
        1. Show your hand gesture to the camera
        2. Wait for the gesture to stabilize
        3. Click 'Add Gesture' to add it to text
        4. Use 'Clear Text' to reset
        5. Use 'Speak Text' to hear the text
        
        Enable 'Use ML Model' if you have trained a model.
        """
        ttk.Label(instructions_frame, text=instructions, justify=tk.LEFT).pack()
    
    def update_video(self):
        """
        Update video frame
        """
        if not self.is_running:
            return
        
        success, frame = self.cap.read()
        
        if success:
            frame = cv2.flip(frame, 1)
            
            # Process frame
            frame = self.detector.find_hands(frame, draw=True)
            landmark_list = self.detector.find_position(frame, draw=False)
            
            gesture = "NO HAND"
            confidence = 0.0
            
            if len(landmark_list) > 0:
                if self.use_ml_model.get():
                    features = self.detector.get_landmark_features(frame)
                    if len(features) > 0:
                        gesture, confidence = self.classifier.predict(features)
                        if gesture is None or confidence < self.confidence_threshold.get():
                            gesture = "UNCERTAIN"
                else:
                    fingers = self.detector.fingers_up(landmark_list)
                    gesture = self.simple_recognizer.recognize_finger_count(fingers)
                    confidence = 1.0
                
                # Check stability
                if gesture == self.last_gesture:
                    self.gesture_stable_count += 1
                else:
                    self.gesture_stable_count = 0
                    self.last_gesture = gesture
                
                if self.gesture_stable_count >= 5:
                    self.current_gesture = gesture
            
            # Update UI
            self.gesture_label.config(text=gesture)
            self.confidence_value_label.config(text=f"Confidence: {confidence:.2f}")
            self.confidence_label.config(text=f"{self.confidence_threshold.get():.2f}")
            
            # Convert frame to ImageTk
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk
        
        # Schedule next update
        self.window.after(10, self.update_video)
    
    def add_gesture(self):
        """
        Add current gesture to text
        """
        if self.current_gesture and self.current_gesture not in ["NO HAND", "UNCERTAIN"]:
            if self.current_gesture == "SPACE":
                self.recognized_text += " "
            else:
                self.recognized_text += self.current_gesture + " "
            
            self.text_display.delete(1.0, tk.END)
            self.text_display.insert(tk.END, self.recognized_text)
            self.current_gesture = ""
    
    def clear_text(self):
        """
        Clear recognized text
        """
        self.recognized_text = ""
        self.text_display.delete(1.0, tk.END)
    
    def speak_text(self):
        """
        Speak recognized text in a separate thread
        """
        text = self.recognized_text.strip()
        if text:
            threading.Thread(target=self._speak, args=(text,), daemon=True).start()
    
    def _speak(self, text):
        """
        Internal method to speak text
        """
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
    
    def on_closing(self):
        """
        Handle window closing
        """
        self.is_running = False
        self.cap.release()
        self.window.destroy()


def main():
    """
    Main function
    """
    root = tk.Tk()
    app = GestureRecognitionGUI(root, "Hand Gesture Recognition - Deaf Communication")
    root.mainloop()


if __name__ == "__main__":
    main()
