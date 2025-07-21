import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import sqlite3
from datetime import datetime
import requests
from io import BytesIO
import cv2
import pickle

class ImageFeatureExtractor(nn.Module):
    """CNN-based image feature extractor for fashion items"""
    
    def __init__(self, feature_dim=512):
        super(ImageFeatureExtractor, self).__init__()
        
        # Convolutional layers for image processing
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, x):
        # Process through convolutional layers
        x = self.conv_layers(x)
        # Flatten the output
        x = x.view(x.size(0), -1)
        # Project to feature space
        x = self.feature_projection(x)
        return x

class DeepRecommendationNetwork(nn.Module):
    """Deep neural network for fashion recommendation with RL"""
    
    def __init__(self, image_feature_dim=512, text_feature_dim=256, categorical_dim=100, hidden_dim=1024):
        super(DeepRecommendationNetwork, self).__init__()
        
        # Image feature extractor
        self.image_encoder = ImageFeatureExtractor(image_feature_dim)
        
        # Text embedding layer for item descriptions
        self.text_encoder = nn.Sequential(
            nn.Linear(768, text_feature_dim),  # Assuming BERT-like embeddings
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Categorical feature embedding
        self.categorical_encoder = nn.Sequential(
            nn.Linear(50, categorical_dim),  # Brand, category, condition, etc.
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # User preference encoder
        self.user_encoder = nn.Sequential(
            nn.Linear(100, hidden_dim // 2),  # User preference history
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Combined feature processor
        total_feature_dim = image_feature_dim + text_feature_dim + categorical_dim + hidden_dim // 2
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Q-network for reinforcement learning (predicts recommendation scores)
        self.q_network = nn.Sequential(
            nn.Linear(hidden_dim // 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Single Q-value output
        )
        
        # Value network for advantage estimation
        self.value_network = nn.Sequential(
            nn.Linear(hidden_dim // 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, image, text_features, categorical_features, user_features):
        # Extract features from each modality
        img_features = self.image_encoder(image)
        text_features = self.text_encoder(text_features)
        cat_features = self.categorical_encoder(categorical_features)
        user_features = self.user_encoder(user_features)
        
        # Combine all features
        combined_features = torch.cat([img_features, text_features, cat_features, user_features], dim=1)
        
        # Process combined features
        fused_features = self.feature_fusion(combined_features)
        
        # Get Q-value and state value
        q_value = self.q_network(fused_features)
        state_value = self.value_network(fused_features)
        
        return q_value, state_value, fused_features

class DepopDQNAgent:
    """Deep Q-Network Agent for Depop Recommendations"""
    
    def __init__(self, model_path="models/depop_dqn.pt", device="cpu"):
        self.device = torch.device(device)
        self.model = DeepRecommendationNetwork().to(self.device)
        self.target_model = DeepRecommendationNetwork().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.model_path = model_path
        
        # RL parameters
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
        self.batch_size = 32
        self.memory = []
        self.memory_size = 10000
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load existing model if available
        self.load_model()
        
    def load_model(self):
        """Load pre-trained model if exists"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            print(f"Loaded model from {self.model_path}")
        except FileNotFoundError:
            print("No existing model found, starting fresh")
            self.update_target_network()
    
    def save_model(self):
        """Save current model state"""
        import os
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def update_target_network(self):
        """Update target network with current model weights"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def process_image(self, image_url):
        """Download and process image from URL"""
        try:
            if not image_url or image_url.strip() == '':
                # Return dummy image tensor for empty URLs
                return torch.zeros(1, 3, 224, 224).to(self.device)
                
            response = requests.get(image_url, timeout=5)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
            return image_tensor
        except Exception as e:
            if image_url and image_url.strip():
                print(f"Error processing image {image_url}: {e}")
            # Return dummy image tensor
            return torch.zeros(1, 3, 224, 224).to(self.device)
    
    def extract_text_features(self, item_title, item_description=""):
        """Extract text features from item title and description"""
        # Simple text feature extraction (can be enhanced with BERT/transformers)
        text = f"{item_title} {item_description}".lower()
        
        # Create basic text features (bag of words approach)
        fashion_keywords = [
            'vintage', 'retro', 'modern', 'classic', 'trendy', 'casual', 'formal',
            'cotton', 'denim', 'leather', 'silk', 'wool', 'polyester',
            'small', 'medium', 'large', 'oversized', 'fitted', 'loose',
            'black', 'white', 'blue', 'red', 'green', 'pink', 'brown',
            'shirt', 'dress', 'pants', 'shoes', 'jacket', 'sweater', 'skirt'
        ]
        
        # Create feature vector based on keyword presence
        features = []
        for keyword in fashion_keywords:
            features.append(1.0 if keyword in text else 0.0)
        
        # Pad or truncate to fixed size (768 to match BERT-like size)
        while len(features) < 768:
            features.append(0.0)
        features = features[:768]
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def get_item_features(self, item_data):
        """Extract categorical features from item data"""
        features = [0.0] * 50  # Fixed size feature vector
        
        # Brand encoding (simple hash-based)
        brand = item_data.get('brand', '')
        if brand:
            features[0] = hash(brand.lower()) % 100 / 100.0
        
        # Price normalization
        price = float(item_data.get('price', 0))
        features[1] = min(price / 500.0, 1.0)  # Normalize to 0-1
        
        # Category encoding
        category_map = {
            'tops': 0.1, 'bottoms': 0.2, 'dresses': 0.3, 'shoes': 0.4,
            'accessories': 0.5, 'outerwear': 0.6, 'swimwear': 0.7
        }
        category = item_data.get('category', '').lower()
        features[2] = category_map.get(category, 0.0)
        
        # Size encoding
        size_map = {
            'xs': 0.1, 's': 0.2, 'm': 0.3, 'l': 0.4, 'xl': 0.5, 'xxl': 0.6
        }
        size = item_data.get('size', '').lower()
        features[3] = size_map.get(size, 0.3)  # Default to medium
        
        # Condition encoding
        condition_map = {
            'brand_new': 1.0, 'used_like_new': 0.8, 'used_excellent': 0.6,
            'used_good': 0.4, 'used_fair': 0.2
        }
        condition = item_data.get('condition', '').lower()
        features[4] = condition_map.get(condition, 0.6)
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def get_user_features(self, user_id, db_connection):
        """Extract user preference features from database"""
        features = [0.0] * 100  # Fixed size user feature vector
        
        try:
            cursor = db_connection.cursor()
            
            # Get user preferences
            cursor.execute("""
                SELECT preference_type, preference_value, preference_score
                FROM user_preferences 
                WHERE user_id = ? AND preference_score > 0
                ORDER BY preference_score DESC LIMIT 50
            """, (user_id,))
            
            preferences = cursor.fetchall()
            
            # Encode preferences into feature vector
            for i, (pref_type, pref_value, score) in enumerate(preferences[:50]):
                if i < 50:
                    features[i] = min(score, 3.0) / 3.0  # Normalize score
                if i + 50 < 100:
                    features[i + 50] = hash(f"{pref_type}_{pref_value}") % 100 / 100.0
                    
        except Exception as e:
            print(f"Error extracting user features: {e}")
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def predict_recommendation_score(self, item_data, user_id, db_connection):
        """Predict recommendation score for an item"""
        print(f"🔍 SCORING ITEM: {item_data.get('title', 'Unknown')[:50]}... for user {user_id}")
        self.model.eval()
        
        # First check if user has already given feedback on this item
        try:
            item_url = item_data.get('item_url', '')
            print(f"🔗 Item URL: {item_url}")
            cursor = db_connection.cursor()
            cursor.execute("""
                SELECT feedback_type FROM user_feedback 
                WHERE user_id = ? AND item_url = ?
            """, (user_id, item_url))
            
            existing_feedback = cursor.fetchone()
            if existing_feedback:
                feedback_score = existing_feedback[0]
                print(f"🚫 Found existing feedback for {item_data.get('title', 'Unknown')[:50]}...: {feedback_score}")
                # If user disliked it, give very low score
                if feedback_score == -1:
                    print(f"❌ FILTERING OUT disliked item: {item_data.get('title', 'Unknown')[:50]}...")
                    return -10.0  # Very negative score for disliked items
                # If user liked it, give high score
                elif feedback_score == 1:
                    print(f"✅ Found liked item: {item_data.get('title', 'Unknown')[:50]}...")
                    return 5.0
                # If user loved it, give very high score
                elif feedback_score == 2:
                    print(f"❤️ Found loved item: {item_data.get('title', 'Unknown')[:50]}...")
                    return 10.0
            else:
                print(f"🆕 No existing feedback for {item_data.get('title', 'Unknown')[:50]}...")
        except Exception as e:
            print(f"❌ Error checking existing feedback: {e}")
        
        # If no existing feedback, use neural network to predict
        with torch.no_grad():
            # Extract all feature types
            image_tensor = self.process_image(item_data.get('image_url', ''))
            text_features = self.extract_text_features(
                item_data.get('title', ''), 
                item_data.get('description', '')
            )
            categorical_features = self.get_item_features(item_data)
            user_features = self.get_user_features(user_id, db_connection)
            
            # Get model prediction
            q_value, state_value, _ = self.model(
                image_tensor, text_features, categorical_features, user_features
            )
            
            neural_score = q_value.item()
            print(f"🧠 Neural network raw score for {item_data.get('title', 'Unknown')[:50]}...: {neural_score}")
            
            return neural_score
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """Perform one training step using experience replay"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        self.model.train()
        
        # Sample random batch from memory
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch_experiences = [self.memory[i] for i in batch]
        
        total_loss = 0.0
        
        for state, action, reward, next_state, done in batch_experiences:
            try:
                # Current Q-value
                current_q, current_value, _ = self.model(**state)
                
                # Target Q-value
                if done:
                    target_q = reward
                else:
                    with torch.no_grad():
                        next_q, next_value, _ = self.target_model(**next_state)
                        target_q = reward + self.gamma * next_q.item()
                
                # Compute loss
                target_tensor = torch.tensor([target_q], dtype=torch.float32, requires_grad=False).to(self.device)
                loss = self.criterion(current_q, target_tensor.detach())
                
                # Backpropagate
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                
            except RuntimeError as e:
                print(f"Error in training step: {e}")
                continue
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return total_loss / self.batch_size
    
    def learn_from_feedback(self, item_data, user_id, feedback_score, db_connection):
        """Learn from user feedback using RL"""
        try:
            # Extract current state
            state = {
                'image': self.process_image(item_data.get('image_url', '')),
                'text_features': self.extract_text_features(
                    item_data.get('title', ''), 
                    item_data.get('description', '')
                ),
                'categorical_features': self.get_item_features(item_data),
                'user_features': self.get_user_features(user_id, db_connection)
            }
            
            # Convert feedback to reward (-1, 0, 1, 2)
            reward_map = {-1: -1.0, 1: 0.5, 2: 1.0}
            reward = reward_map.get(feedback_score, 0.0)
            
            # For simplicity, we'll use immediate reward learning
            # Create separate state tensors to avoid gradient issues
            state_copy = {
                'image': state['image'].clone().detach(),
                'text_features': state['text_features'].clone().detach(), 
                'categorical_features': state['categorical_features'].clone().detach(),
                'user_features': state['user_features'].clone().detach()
            }
            
            # Store experience (simplified for immediate feedback)
            self.remember(state, 0, reward, state_copy, True)  # Done=True for immediate feedback
            
            # Train the model
            loss = self.train_step()
            
            # Update target network and save model more frequently
            feedback_count = len(self.memory)
            if feedback_count % 10 == 0 or feedback_count <= 50:  # Save every 10 feedbacks, or for first 50
                self.update_target_network()
                self.save_model()
                print(f"📁 Model saved after {feedback_count} feedback entries")
            
            print(f"DQN Learning: Reward={reward}, Loss={loss}")
            
        except Exception as e:
            print(f"Error in DQN learning: {e}")

# Global DQN agent instance
dqn_agent = None

def get_dqn_agent():
    """Get or create DQN agent instance"""
    global dqn_agent
    if dqn_agent is None:
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing DQN Agent on device: {device}")
        dqn_agent = DepopDQNAgent(device=device)
    return dqn_agent

from advanced_recommendation_engine import (
    get_advanced_recommendations, 
    train_advanced_model_on_feedback,
    get_advanced_recommendation_engine
)

def get_deep_rl_recommendations(items, user_id, db_connection, max_items=24):
    """Get recommendations using ADVANCED deep learning system"""
    print(f"🚀 SWITCHING TO ADVANCED RECOMMENDATION ENGINE...")
    
    # Use the new advanced recommendation engine
    return get_advanced_recommendations(items, user_id, db_connection, max_items)

def train_dqn_from_feedback(item_data, user_id, feedback_score, db_connection):
    """Train using ADVANCED deep learning system"""
    print(f"🎓 TRAINING ADVANCED MODEL on feedback...")
    
    # Use the new advanced training system
    train_advanced_model_on_feedback(item_data, user_id, feedback_score, db_connection)
    
    # Also keep the old system for compatibility (but it won't affect recommendations)
    try:
        agent = get_dqn_agent()
        agent.learn_from_feedback(item_data, user_id, feedback_score, db_connection)
    except Exception as e:
        print(f"⚠️ Old system error (ignoring): {e}")
        print(f"Error training DQN from feedback: {e}")
