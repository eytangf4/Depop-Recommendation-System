#!/bin/bash

# Setup script for Depop Deep Learning Recommendation System

echo "🧠 Setting up Deep Learning Neural Network for Depop Recommendations..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install basic Flask dependencies first
echo "Installing Flask dependencies..."
pip install Flask==2.3.3 Werkzeug==2.3.7 requests==2.31.0

# Install Deep Learning dependencies
echo "Installing PyTorch and deep learning libraries..."

# Install PyTorch (CPU version for compatibility)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other ML dependencies
echo "Installing additional ML libraries..."
pip install transformers==4.30.0
pip install numpy==1.24.3
pip install pandas==1.5.3
pip install scikit-learn==1.3.0
pip install pillow==10.0.0
pip install opencv-python==4.8.0.74
pip install timm==0.9.2
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install tqdm==4.65.0
pip install joblib==1.3.1

# Create models directory
echo "Creating models directory..."
mkdir -p models

# Create data directory for caching
echo "Creating data directory..."
mkdir -p data

echo "✅ Deep Learning setup complete!"
echo ""
echo "🚀 To start the application:"
echo "   source venv/bin/activate"
echo "   python app.py"
echo ""
echo "🧠 Features enabled:"
echo "   • Convolutional Neural Network for image analysis"
echo "   • Deep Q-Network (DQN) for reinforcement learning"
echo "   • Multi-modal feature fusion (images + text + categories)"
echo "   • Real-time learning from user feedback"
echo "   • Advanced recommendation scoring"
echo ""
echo "📊 Model capabilities:"
echo "   • Processes item images with CNN"
echo "   • Extracts text features from titles/descriptions"
echo "   • Learns user preferences from likes/dislikes"
echo "   • Provides personalized recommendations"
echo "   • Continuously improves with user interactions"
