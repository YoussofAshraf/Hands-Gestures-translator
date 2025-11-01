# 🤟 Hand Gesture Recognition System

A real-time hand gesture recognition system for deaf communication, converting hand gestures into text and speech using computer vision and machine learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.12.0-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Features](#features)
- [Demo](#demo)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Training Custom Models](#training-custom-models)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Real-time Hand Detection**: Detects and tracks hand landmarks using MediaPipe
- **Two Recognition Modes**:
  - Simple rule-based recognition (no training required)
  - ML-based recognition with scikit-learn (customizable)
- **Dual Interface**:
  - Command-line application for quick testing
  - Modern GUI application with Tkinter
- **Text-to-Speech**: Converts recognized gestures to spoken words
- **Data Collection Tool**: Easily collect training data for custom gestures
- **Model Training**: Train your own gesture recognition models
- **High Performance**: Real-time processing with optimized algorithms

## 🎬 Demo

The system can recognize:

- Hand poses (number of fingers)
- Custom gestures (with trained ML model)
- Convert gestures to text
- Speak the recognized text using TTS

## 💻 System Requirements

### Hardware

- Webcam (built-in or external)
- Minimum 4GB RAM
- Processor: Intel Core i3 or equivalent

### Software

- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **Webcam drivers**: Properly configured camera

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd gemini-projects
```

### Step 2: Create Virtual Environment

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**

```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:

- `opencv-python==4.12.0.88` - Computer vision library
- `mediapipe==0.10.21` - Hand landmark detection
- `numpy==1.26.4` - Numerical computing
- `pyttsx3==2.99` - Text-to-speech
- `scikit-learn==1.7.2` - Machine learning
- `Pillow==12.0.0` - Image processing

### Step 4: Verify Installation

```bash
python test_system.py
```

This will check if all dependencies are installed correctly and test your webcam.

## 🎯 Quick Start

### Option 1: Using the Quick Start Script (Linux/macOS)

```bash
chmod +x run.sh
./run.sh
```

Then select from the menu:

1. Launch GUI Application (Recommended for beginners)
2. Launch Command-Line Application
3. Collect Training Data
4. Train ML Model
5. Run System Test

### Option 2: Direct Python Execution

**GUI Application (Recommended):**

```bash
python gui_app.py
```

**Command-Line Application:**

```bash
python main.py
```

## 📖 Usage

### GUI Application

1. **Launch the application:**

   ```bash
   python gui_app.py
   ```

2. **Select recognition mode:**

   - Toggle "Use ML Model" for machine learning recognition
   - Leave unchecked for simple rule-based recognition

3. **Controls:**
   - The camera feed shows your hand in real-time
   - Recognized gestures appear in the gesture display area
   - Click "Add to Text" to add the current gesture to the text field
   - Click "Speak Text" to hear the recognized text
   - Click "Clear Text" to reset

### Command-Line Application

1. **Launch the application:**

   ```bash
   python main.py
   ```

2. **Select mode (1 or 2):**

   - Mode 1: Simple rule-based (counts fingers)
   - Mode 2: ML-based (requires trained model)

3. **Keyboard Controls:**
   - `SPACE` - Add current gesture to text
   - `BACKSPACE` - Delete last character
   - `ENTER` - Speak the recognized text
   - `C` - Clear all text
   - `Q` - Quit application

### Simple Recognition Mode (Default)

The simple mode recognizes basic hand poses:

- 0 fingers = "ZERO"
- 1 finger = "ONE"
- 2 fingers = "TWO"
- 3 fingers = "THREE"
- 4 fingers = "FOUR"
- 5 fingers = "FIVE"

Perfect for testing without training a model!

## 📁 Project Structure

```
gemini-projects/
├── main.py                    # Command-line application
├── gui_app.py                 # GUI application
├── hand_detector.py           # Hand detection using MediaPipe
├── gesture_classifier.py      # ML-based gesture classification
├── data_collector.py          # Tool for collecting training data
├── train_model.py            # Train custom ML models
├── test_system.py            # System testing and verification
├── requirements.txt          # Python dependencies
├── run.sh                    # Quick start script
├── gesture_data/             # Training data directory
├── gesture_model.pkl/        # Trained model files
└── README.md                 # This file
```

## 🔧 How It Works

### 1. Hand Detection

- Uses **MediaPipe Hands** to detect 21 hand landmarks
- Tracks hand position and finger positions in real-time
- Works in various lighting conditions

### 2. Feature Extraction

- Extracts normalized landmark coordinates
- Calculates relative positions and angles
- Creates feature vectors for classification

### 3. Gesture Recognition

**Simple Mode:**

- Counts extended fingers
- Maps to predefined gestures
- No training required

**ML Mode:**

- Uses Random Forest classifier
- Trained on custom gesture data
- Higher accuracy for complex gestures

### 4. Text Output

- Stable gesture detection (multiple frame confirmation)
- Appends recognized gestures to text string
- Text-to-speech conversion on demand

## 🎓 Training Custom Models

### Step 1: Collect Training Data

```bash
python data_collector.py
```

1. Enter gesture name (e.g., "HELLO", "THANK_YOU")
2. Position your hand in front of the camera
3. Press `SPACE` to capture samples (30-50 samples recommended)
4. Press `Q` when done
5. Repeat for each gesture you want to recognize

### Step 2: Train the Model

```bash
python train_model.py
```

This will:

- Load collected gesture data
- Train a Random Forest classifier
- Save the model to `gesture_model.pkl/`
- Display accuracy metrics

### Step 3: Use Your Custom Model

Launch the application and select ML mode to use your trained model!

## 🐛 Troubleshooting

### Camera Not Working

**Linux:**

```bash
# Check if camera is detected
ls /dev/video*

# Test with fswebcam
sudo apt-get install fswebcam
fswebcam test.jpg
```

**Permission Issues:**

```bash
sudo usermod -a -G video $USER
# Log out and log back in
```

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or install individually
pip install opencv-python mediapipe numpy scikit-learn pyttsx3 Pillow
```

### TTS Not Working

**Linux:**

```bash
# Install espeak
sudo apt-get install espeak

# Or festival
sudo apt-get install festival
```

**macOS:**

```bash
# macOS has built-in TTS, should work out of the box
say "Test"
```

**Windows:**

```cmd
# Windows has built-in SAPI5, should work automatically
```

### Low FPS or Lag

- Close other applications using the camera
- Reduce camera resolution in the code (default: 640x480)
- Use simple recognition mode instead of ML mode
- Update your graphics drivers

### Model Not Found Error

```bash
# Train a model first
python train_model.py

# Or use simple mode (no model required)
# Select option 1 when running main.py
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report Bugs**: Open an issue describing the bug
2. **Suggest Features**: Share your ideas for improvements
3. **Submit Pull Requests**: Fix bugs or add features
4. **Improve Documentation**: Help make the docs better
5. **Share Your Models**: Contribute trained models for common gestures

### Development Setup

```bash
# Clone the repo
git clone <repository-url>
cd gemini-projects

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_system.py
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MediaPipe** by Google for hand landmark detection
- **OpenCV** for computer vision capabilities
- **scikit-learn** for machine learning tools
- **pyttsx3** for text-to-speech functionality

## 📧 Contact & Support

- **Issues**: Open an issue on GitHub
- **Questions**: Use GitHub Discussions
- **Email**: [Your email here]

## 🚀 Future Enhancements

- [ ] Support for two-hand gestures
- [ ] Sign language alphabet recognition
- [ ] Mobile app version
- [ ] Real-time translation to multiple languages
- [ ] Gesture recording and playback
- [ ] Cloud-based model training
- [ ] Integration with communication apps

---

**Made with ❤️ for the deaf community**

_Star ⭐ this repository if you find it helpful!_
