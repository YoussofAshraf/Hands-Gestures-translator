"""
Hand Detection Module
Uses Mediapipe to detect and track hand landmarks in real-time
"""

import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    """
    Hand detector class using Mediapipe for hand tracking
    """
    
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        """
        Initialize hand detector
        
        Args:
            mode: Static image mode or video mode
            max_hands: Maximum number of hands to detect
            detection_confidence: Minimum detection confidence
            tracking_confidence: Minimum tracking confidence
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def find_hands(self, img, draw=True):
        """
        Detect hands in image
        
        Args:
            img: Input image
            draw: Whether to draw landmarks on image
            
        Returns:
            Image with landmarks drawn (if draw=True)
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return img
    
    def find_position(self, img, hand_no=0, draw=True):
        """
        Find positions of hand landmarks
        
        Args:
            img: Input image
            hand_no: Which hand to get landmarks for
            draw: Whether to draw circles on landmarks
            
        Returns:
            List of landmark positions [id, x, y]
        """
        landmark_list = []
        
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_no]
                
                for id, lm in enumerate(hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append([id, cx, cy])
                    
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        return landmark_list
    
    def get_landmark_features(self, img, hand_no=0):
        """
        Extract normalized landmark features for gesture recognition
        
        Args:
            img: Input image
            hand_no: Which hand to get features for
            
        Returns:
            Normalized feature vector
        """
        features = []
        
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_no]
                
                # Extract x, y coordinates of all 21 landmarks
                landmark_coords = []
                for lm in hand.landmark:
                    landmark_coords.append([lm.x, lm.y])
                
                landmark_coords = np.array(landmark_coords)
                
                # Normalize coordinates relative to wrist (landmark 0)
                wrist = landmark_coords[0]
                normalized_coords = landmark_coords - wrist
                
                # Flatten to 1D feature vector
                features = normalized_coords.flatten().tolist()
        
        return features
    
    def fingers_up(self, landmark_list):
        """
        Determine which fingers are up
        
        Args:
            landmark_list: List of landmark positions
            
        Returns:
            List of 5 values (0 or 1) for each finger (thumb to pinky)
        """
        fingers = []
        tip_ids = [4, 8, 12, 16, 20]  # Fingertip landmark IDs
        
        if len(landmark_list) == 0:
            return fingers
        
        # Thumb (special case - check x coordinate)
        if landmark_list[tip_ids[0]][1] < landmark_list[tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other 4 fingers
        for id in range(1, 5):
            if landmark_list[tip_ids[id]][2] < landmark_list[tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    def get_handedness(self):
        """
        Get handedness (left or right) of detected hands
        
        Returns:
            List of handedness labels
        """
        handedness_list = []
        
        if self.results.multi_handedness:
            for hand_info in self.results.multi_handedness:
                handedness_list.append(hand_info.classification[0].label)
        
        return handedness_list
