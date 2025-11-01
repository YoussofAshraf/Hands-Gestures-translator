
#!/bin/bash

# Quick Start Script for Hand Gesture Recognition System
# This script helps you quickly launch the application

echo "============================================================"
echo "Hand Motion to Text - Deaf Communication System"
echo "============================================================"
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Display menu
echo "Choose an option:"
echo ""
echo "1. Launch GUI Application (Recommended)"
echo "2. Launch Command-Line Application"
echo "3. Collect Training Data"
echo "4. Train ML Model"
echo "5. Run System Test"
echo "6. Exit"
echo ""

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        echo ""
        echo "Launching GUI Application..."
        echo "Close the window or press Ctrl+C to exit"
        python gui_app.py
        ;;
    2)
        echo ""
        echo "Launching Command-Line Application..."
        python main.py
        ;;
    3)
        echo ""
        echo "Launching Data Collection Tool..."
        python data_collector.py
        ;;
    4)
        echo ""
        echo "Training ML Model..."
        python train_model.py
        ;;
    5)
        echo ""
        echo "Running System Test..."
        python test_system.py
        ;;
    6)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "Thank you for using Hand Gesture Recognition System!"
echo "============================================================"
