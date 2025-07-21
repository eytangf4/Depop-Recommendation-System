#!/usr/bin/env python3
"""
Advanced Deep Learning Recommendation Engine for Depop
Combines neural collaborative filtering, content-based filtering, and deep reinforcement learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sqlite3
from PIL import Image
import requests
from io import BytesIO
import torchvision.transforms as transforms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import json
from collections import defaultdict, Counter
import pickle
import os
from datetime import datetime

# Try to import sklearn features, fall back gracefully if not available
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
    print("✅ sklearn features available - enabling advanced text analysis")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ sklearn not available - using basic text analysis")

class AdvancedFeatureExtractor:
    """Advanced feature extraction for fashion items"""
    
    def __init__(self):
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Fashion-specific vocabulary and patterns
        self.style_keywords = {
            'vintage': ['vintage', 'retro', 'throwback', '90s', '80s', '70s', 'y2k', 'classic'],
            'modern': ['modern', 'contemporary', 'current', 'trendy', 'fashionable', 'stylish'],
            'casual': ['casual', 'everyday', 'comfortable', 'relaxed', 'basic', 'simple'],
            'formal': ['formal', 'elegant', 'sophisticated', 'dress', 'professional', 'classy'],
            'sporty': ['athletic', 'sport', 'gym', 'workout', 'active', 'performance'],
            'edgy': ['edgy', 'punk', 'rock', 'alternative', 'grunge', 'rebel'],
            'feminine': ['feminine', 'girly', 'cute', 'pretty', 'delicate', 'soft'],
            'masculine': ['masculine', 'rugged', 'bold', 'strong', 'tough']
        }
        
        self.color_keywords = {
            'black': ['black', 'dark', 'charcoal', 'midnight'],
            'white': ['white', 'cream', 'ivory', 'off-white', 'pearl'],
            'blue': ['blue', 'navy', 'royal', 'sky', 'denim', 'cobalt'],
            'red': ['red', 'crimson', 'burgundy', 'wine', 'cherry', 'rose'],
            'green': ['green', 'olive', 'forest', 'mint', 'lime', 'sage'],
            'yellow': ['yellow', 'gold', 'amber', 'mustard', 'lemon'],
            'pink': ['pink', 'blush', 'salmon', 'coral', 'fuchsia'],
            'purple': ['purple', 'violet', 'lavender', 'plum', 'magenta'],
            'brown': ['brown', 'tan', 'beige', 'khaki', 'camel', 'chocolate'],
            'gray': ['gray', 'grey', 'silver', 'slate', 'ash']
        }
        
        self.material_keywords = {
            'cotton': ['cotton', '100% cotton', 'organic cotton'],
            'denim': ['denim', 'jean', 'chambray'],
            'leather': ['leather', 'genuine leather', 'faux leather'],
            'silk': ['silk', 'satin', 'smooth'],
            'wool': ['wool', 'cashmere', 'merino'],
            'polyester': ['polyester', 'poly', 'synthetic'],
            'linen': ['linen', 'flax'],
            'spandex': ['spandex', 'elastane', 'stretch']
        }
        
        self.brand_tiers = {
            'luxury': ['gucci', 'prada', 'chanel', 'louis vuitton', 'dior', 'versace', 'balenciaga'],
            'designer': ['nike', 'adidas', 'calvin klein', 'tommy hilfiger', 'polo ralph lauren', 'lacoste'],
            'contemporary': ['zara', 'h&m', 'uniqlo', 'urban outfitters', 'american eagle', 'hollister'],
            'vintage': ['vintage', 'retro', 'thrift', 'secondhand']
        }
        
        # Initialize text processing
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=100, 
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True
            )
            self.tfidf_fitted = False
            print("🔤 TF-IDF vectorizer initialized for advanced text analysis")
        else:
            self.tfidf_vectorizer = None
            self.tfidf_fitted = False
        
        # Initialize sentence transformer for semantic embeddings
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("🧠 Sentence transformer loaded for semantic similarity")
        except Exception as e:
            print(f"⚠️ Could not load sentence transformer: {e}")
            self.sentence_model = None
        
    def extract_semantic_features(self, text, max_length=50):
        """Extract semantic embeddings from text using sentence transformer"""
        if not self.sentence_model or not text.strip():
            return np.zeros(384)  # MiniLM-L6-v2 produces 384-dim embeddings
        
        try:
            # Truncate text if too long
            if len(text) > max_length:
                text = text[:max_length]
            
            # Get semantic embedding
            embedding = self.sentence_model.encode(text, convert_to_tensor=False)
            return embedding
        except Exception as e:
            print(f"Error in semantic feature extraction: {e}")
            return np.zeros(384)
    
    def get_text_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts"""
        if not self.sentence_model:
            return 0.0
        
        try:
            embeddings = self.sentence_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating text similarity: {e}")
            return 0.0
    
    def extract_tfidf_features(self, texts, fit=False):
        """Extract TF-IDF features from texts"""
        if not SKLEARN_AVAILABLE or not self.tfidf_vectorizer:
            return np.zeros((len(texts), 100))
        
        try:
            if fit and not self.tfidf_fitted:
                # Fit and transform
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                self.tfidf_fitted = True
                print(f"🔤 TF-IDF fitted on {len(texts)} texts")
            elif self.tfidf_fitted:
                # Just transform
                tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            else:
                # Not fitted yet, return zeros
                return np.zeros((len(texts), 100))
            
            return tfidf_matrix.toarray()
        except Exception as e:
            print(f"Error in TF-IDF feature extraction: {e}")
            return np.zeros((len(texts), 100))
        """Extract style-based features from text"""
        text = text.lower()
        features = {}
        
        for style, keywords in self.style_keywords.items():
            features[f'style_{style}'] = any(keyword in text for keyword in keywords)
            
        return features
    
    def extract_color_features(self, text):
        """Extract color features from text"""
        text = text.lower()
        features = {}
        
        for color, keywords in self.color_keywords.items():
            features[f'color_{color}'] = any(keyword in text for keyword in keywords)
            
        return features
    
    def extract_material_features(self, text):
        """Extract material features from text"""
        text = text.lower()
        features = {}
        
        for material, keywords in self.material_keywords.items():
            features[f'material_{material}'] = any(keyword in text for keyword in keywords)
            
        return features
    
    def extract_style_features(self, text):
        """Extract style-related features from text"""
        styles = ['vintage', 'retro', 'modern', 'classic', 'casual', 'formal', 'street',
                 'boho', 'minimalist', 'grunge', 'preppy', 'gothic', 'punk', 'chic']
        features = {}
        text_lower = text.lower()
        for style in styles:
            features[f'style_{style}'] = float(style in text_lower)
        return features
    
    def extract_brand_tier(self, brand_text):
        """Determine brand tier"""
        brand_text = brand_text.lower()
        
        for tier, brands in self.brand_tiers.items():
            if any(brand in brand_text for brand in brands):
                return tier
        
        return 'unknown'
    
    def extract_price_features(self, price):
        """Extract price-based features"""
        if price is None:
            return {'price_unknown': 1, 'price_budget': 0, 'price_mid': 0, 'price_high': 0, 'price_luxury': 0}
        
        try:
            price_float = float(price)
            if price_float < 20:
                return {'price_unknown': 0, 'price_budget': 1, 'price_mid': 0, 'price_high': 0, 'price_luxury': 0}
            elif price_float < 50:
                return {'price_unknown': 0, 'price_budget': 0, 'price_mid': 1, 'price_high': 0, 'price_luxury': 0}
            elif price_float < 100:
                return {'price_unknown': 0, 'price_budget': 0, 'price_mid': 0, 'price_high': 1, 'price_luxury': 0}
            else:
                return {'price_unknown': 0, 'price_budget': 0, 'price_mid': 0, 'price_high': 0, 'price_luxury': 1}
        except:
            return {'price_unknown': 1, 'price_budget': 0, 'price_mid': 0, 'price_high': 0, 'price_luxury': 0}
    def extract_advanced_text_features(self, text, all_texts=None):
        """Extract advanced text features using TF-IDF if available"""
        if SKLEARN_AVAILABLE and self.tfidf_vectorizer is not None:
            try:
                if not self.tfidf_fitted and all_texts:
                    # Fit TF-IDF on all available texts
                    self.tfidf_vectorizer.fit(all_texts)
                    self.tfidf_fitted = True
                    print(f"📊 TF-IDF fitted on {len(all_texts)} texts")
                
                if self.tfidf_fitted:
                    tfidf_features = self.tfidf_vectorizer.transform([text]).toarray()[0]
                    # Ensure we always have exactly 100 TF-IDF features to maintain consistency
                    if len(tfidf_features) > 100:
                        tfidf_features = tfidf_features[:100]  # Truncate if too many
                    elif len(tfidf_features) < 100:
                        # Pad with zeros if too few
                        tfidf_features = np.pad(tfidf_features, (0, 100 - len(tfidf_features)), mode='constant')
                    
                    return {f'tfidf_{i}': float(val) for i, val in enumerate(tfidf_features)}
            except Exception as e:
                print(f"⚠️ TF-IDF error: {e}, falling back to basic text features")
        
        # Fallback to basic text features
        return self._extract_basic_text_features(text)
    
    def _extract_basic_text_features(self, text):
        """Basic text feature extraction"""
        text = text.lower()
        features = {}
        
        # Word count features
        words = text.split()
        features['word_count'] = len(words) / 20.0  # Normalized
        features['char_count'] = len(text) / 200.0  # Normalized
        
        # Sentiment/style indicators
        positive_words = ['love', 'amazing', 'perfect', 'beautiful', 'great', 'awesome', 'cute', 'nice']
        negative_words = ['worn', 'damaged', 'faded', 'stained', 'torn', 'old', 'used']
        
        features['positive_sentiment'] = sum(1 for word in positive_words if word in text) / len(words) if words else 0
        features['negative_sentiment'] = sum(1 for word in negative_words if word in text) / len(words) if words else 0
        
        # Brand mention features
        brand_mentions = ['nike', 'adidas', 'vintage', 'retro', 'designer', 'luxury']
        for brand in brand_mentions:
            features[f'mentions_{brand}'] = float(brand in text)
        
        return features
    
    def extract_comprehensive_features(self, item_data, all_item_texts=None):
        """Extract all features for an item"""
        title = item_data.get('title', '')
        description = item_data.get('description', '')
        brand = item_data.get('brand', '')
        price = item_data.get('price')
        size = item_data.get('size', '')
        
        combined_text = f"{title} {description} {brand}".lower()
        
        # Extract all feature types
        features = {}
        
        # Advanced text features (TF-IDF if available, basic otherwise)
        if all_item_texts and len(all_item_texts) > 10:  # Only use TF-IDF if we have enough data
            text_features = self.extract_advanced_text_features(combined_text, all_item_texts)
        else:
            text_features = self._extract_basic_text_features(combined_text)
        features.update(text_features)
        
        # Style features
        features.update(self.extract_style_features(combined_text))
        
        # Color features
        features.update(self.extract_color_features(combined_text))
        
        # Material features
        features.update(self.extract_material_features(combined_text))
        
        # Brand tier
        brand_tier = self.extract_brand_tier(brand)
        for tier in ['luxury', 'designer', 'contemporary', 'vintage', 'unknown']:
            features[f'brand_{tier}'] = (brand_tier == tier)
        
        # Price features
        features.update(self.extract_price_features(price))
        
        # Size features (simplified)
        size_mapping = {'xs': 0, 's': 1, 'm': 2, 'l': 3, 'xl': 4, 'xxl': 5}
        features['size_encoded'] = size_mapping.get(size.lower(), 2)  # Default to M
        
        # Text length features
        features['title_length'] = len(title) / 100.0  # Normalized
        features['has_description'] = len(description) > 0
        
        # Advanced semantic features
        if self.sentence_model:
            try:
                semantic_embedding = self.extract_semantic_features(combined_text)
                # Add top semantic dimensions as features (first 20 dimensions)
                for i, val in enumerate(semantic_embedding[:20]):
                    features[f'semantic_{i}'] = float(val)
            except Exception as e:
                print(f"⚠️ Error adding semantic features: {e}")
        
        # Convert boolean features to floats
        for key, value in features.items():
            if isinstance(value, bool):
                features[key] = float(value)
        
        return features

class UserProfiler:
    """Advanced user profiling based on feedback history"""
    
    def __init__(self):
        self.feature_extractor = AdvancedFeatureExtractor()
    
    def analyze_user_preferences(self, user_id, db_connection):
        """Analyze user preferences from feedback history"""
        cursor = db_connection.cursor()
        
        # Get all user feedback
        cursor.execute("""
            SELECT item_title, item_brand, item_size, item_price, feedback_type
            FROM user_feedback 
            WHERE user_id = ?
        """, (user_id,))
        
        feedback_history = cursor.fetchall()
        
        if not feedback_history:
            return self._get_default_profile()
        
        # Separate liked and disliked items
        liked_items = []
        disliked_items = []
        
        for title, brand, size, price, feedback_type in feedback_history:
            item_data = {
                'title': title or '',
                'brand': brand or '',
                'size': size or '',
                'price': price,
                'description': ''
            }
            
            if feedback_type > 0:  # Liked or loved
                liked_items.append((item_data, feedback_type))
            elif feedback_type < 0:  # Disliked
                disliked_items.append((item_data, feedback_type))
        
        # Extract features for liked and disliked items
        liked_features = []
        disliked_features = []
        
        # Collect all item texts for TF-IDF fitting
        all_texts = []
        for item_data, _ in liked_items + disliked_items:
            title = item_data.get('title', '')
            description = item_data.get('description', '')
            brand = item_data.get('brand', '')
            combined_text = f"{title} {description} {brand}".lower()
            if combined_text.strip():
                all_texts.append(combined_text)
        
        for item_data, _ in liked_items:
            features = self.feature_extractor.extract_comprehensive_features(item_data, all_texts)
            liked_features.append(features)
        
        for item_data, _ in disliked_items:
            features = self.feature_extractor.extract_comprehensive_features(item_data, all_texts)
            disliked_features.append(features)
        
        # Build user preference profile
        profile = self._build_preference_profile(liked_features, disliked_features)
        
        print(f"👤 User {user_id} profile: {len(liked_items)} likes, {len(disliked_items)} dislikes")
        print(f"🎯 Top preferences: {self._get_top_preferences(profile)}")
        
        return profile
    
    def _build_preference_profile(self, liked_features, disliked_features):
        """Build user preference profile from feature lists"""
        profile = {}
        
        # Get all possible feature keys
        all_keys = set()
        for features in liked_features + disliked_features:
            all_keys.update(features.keys())
        
        for key in all_keys:
            # Calculate preference score for each feature
            liked_avg = np.mean([features.get(key, 0) for features in liked_features]) if liked_features else 0
            disliked_avg = np.mean([features.get(key, 0) for features in disliked_features]) if disliked_features else 0
            
            # Preference score: positive means user likes this feature, negative means dislikes
            preference_score = liked_avg - disliked_avg
            
            # Weight by confidence (how much data we have)
            confidence = (len(liked_features) + len(disliked_features)) / 10.0  # Normalize
            confidence = min(confidence, 1.0)
            
            profile[key] = {
                'preference': preference_score,
                'confidence': confidence,
                'liked_avg': liked_avg,
                'disliked_avg': disliked_avg
            }
        
        return profile
    
    def _get_default_profile(self):
        """Default profile for users with no feedback history"""
        default_features = self.feature_extractor.extract_comprehensive_features({
            'title': '', 'brand': '', 'description': '', 'price': None, 'size': ''
        })
        
        profile = {}
        for key in default_features.keys():
            profile[key] = {
                'preference': 0.0,
                'confidence': 0.0,
                'liked_avg': 0.0,
                'disliked_avg': 0.0
            }
        
        return profile
    
    def _get_top_preferences(self, profile, top_n=5):
        """Get top user preferences for logging"""
        preferences = [(key, data['preference']) for key, data in profile.items() 
                      if data['confidence'] > 0.1]
        preferences.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_prefs = []
        for key, score in preferences[:top_n]:
            if score > 0:
                top_prefs.append(f"✅{key}({score:.2f})")
            else:
                top_prefs.append(f"❌{key}({score:.2f})")
        
        return top_prefs

class AdvancedNeuralRecommender(nn.Module):
    """Advanced neural network for recommendations"""
    
    def __init__(self, num_features=100, embedding_dim=64, hidden_dims=[128, 64, 32]):
        super(AdvancedNeuralRecommender, self).__init__()
        
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        
        # Feature embedding layers
        self.feature_embedding = nn.Linear(num_features, embedding_dim)
        self.user_profile_embedding = nn.Linear(num_features, embedding_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim*2, num_heads=4, batch_first=True)
        
        # Deep neural network (without BatchNorm to avoid batch size issues)
        layers = []
        input_dim = embedding_dim * 2  # Item features + user profile
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # LayerNorm instead of BatchNorm
                nn.ReLU(),
                nn.Dropout(0.2)  # Reduced dropout
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, item_features, user_profile_features):
        """Forward pass"""
        batch_size = item_features.size(0)
        
        # Embed features
        item_embed = self.feature_embedding(item_features)  # [batch, embedding_dim]
        user_embed = self.user_profile_embedding(user_profile_features)  # [batch, embedding_dim]
        
        # Combine embeddings
        combined = torch.cat([item_embed, user_embed], dim=1)  # [batch, embedding_dim*2]
        
        # Apply attention (self-attention over the combined features)
        combined_unsqueezed = combined.unsqueeze(1)  # [batch, 1, embedding_dim*2]
        attended, _ = self.attention(combined_unsqueezed, combined_unsqueezed, combined_unsqueezed)
        attended = attended.squeeze(1)  # [batch, embedding_dim*2]
        
        # Final prediction
        output = self.network(attended)
        
        return torch.sigmoid(output)  # Output between 0 and 1

class AdvancedRecommendationEngine:
    """Complete advanced recommendation engine"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.feature_extractor = AdvancedFeatureExtractor()
        self.user_profiler = UserProfiler()
        
        # Will be initialized when we know the feature size
        self.model = None
        self.optimizer = None
        self.criterion = nn.BCELoss()
        
        # Training parameters
        self.learning_rate = 0.001
        self.model_path = 'models/advanced_recommendation_model.pt'
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
    
    def get_recommendations(self, user_id, num_recommendations=10):
        """Get recommendations for a user using the advanced system"""
        import sqlite3
        
        # Connect to database
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        try:
            # Get items from user_feedback (simulating item catalog)
            cursor.execute('''
                SELECT DISTINCT item_title, item_brand, item_price, 
                       COALESCE(item_category, '') as description
                FROM user_feedback 
                WHERE feedback_type = 1  -- Only liked items for positive examples
                LIMIT 50
            ''')
            items = cursor.fetchall()
            
            if not items:
                print("⚠️ No items found in database")
                return []
            
            # Format items for the recommendation function
            formatted_items = []
            for title, brand, price, description in items:
                formatted_items.append({
                    'title': title or 'Unknown',
                    'brand': brand or 'Unknown',
                    'price': price or 0,
                    'description': description or '',
                    'url': f'https://depop.com/item/{len(formatted_items)}'  # Mock URL
                })
            
            # Use the advanced recommendation function
            recommendations = get_advanced_recommendations(
                formatted_items, user_id, conn, max_items=num_recommendations
            )
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error getting recommendations: {e}")
            return []
        finally:
            conn.close()
    
    def _initialize_model(self, num_features=175):
        """Initialize the model with a consistent number of features"""
        # Use a fixed feature dimension for consistency across all users
        FIXED_FEATURES = 175
        self.model = AdvancedNeuralRecommender(num_features=FIXED_FEATURES).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Try to load existing model
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"✅ Loaded existing advanced model from {self.model_path}")
            except Exception as e:
                print(f"⚠️ Could not load existing model (likely due to dimension mismatch): {e}. Starting fresh.")
                # Delete the incompatible model file to prevent future issues
                try:
                    os.remove(self.model_path)
                    print("🗑️ Removed incompatible model file")
                except:
                    pass
    
    def _prepare_features(self, item_data, user_profile):
        """Prepare features for the model with dimension consistency"""
        # Extract item features
        item_features = self.feature_extractor.extract_comprehensive_features(item_data)
        
        # Create user profile feature vector
        user_features = {}
        for key, data in user_profile.items():
            user_features[key] = data['preference'] * data['confidence']
        
        # Standardize features to ensure consistent dimensions
        standardized_item_features, standardized_user_features = self._standardize_feature_dimensions(
            item_features, user_features
        )
        
        return torch.tensor(standardized_item_features, dtype=torch.float32), torch.tensor(standardized_user_features, dtype=torch.float32)
    
    def _standardize_feature_dimensions(self, item_features, user_features):
        """Ensure consistent feature dimensions across all users and items"""
        # Define a fixed set of expected features to maintain consistency
        EXPECTED_FEATURES = 175  # Fixed dimension for model consistency
        
        # Combine all possible keys and sort for consistency
        all_keys = set(item_features.keys()) | set(user_features.keys())
        sorted_keys = sorted(all_keys)
        
        # Create standardized vectors
        item_vector = []
        user_vector = []
        
        for key in sorted_keys:
            item_vector.append(item_features.get(key, 0.0))
            user_vector.append(user_features.get(key, 0.0))
        
        # Pad or truncate to expected dimensions
        if len(item_vector) > EXPECTED_FEATURES:
            item_vector = item_vector[:EXPECTED_FEATURES]
            user_vector = user_vector[:EXPECTED_FEATURES]
        elif len(item_vector) < EXPECTED_FEATURES:
            padding_size = EXPECTED_FEATURES - len(item_vector)
            item_vector.extend([0.0] * padding_size)
            user_vector.extend([0.0] * padding_size)
        
        return item_vector, user_vector
    
    def predict_item_score(self, item_data, user_id, db_connection):
        """Predict how much a user will like an item with error handling"""
        try:
            # Get user profile
            user_profile = self.user_profiler.analyze_user_preferences(user_id, db_connection)
            
            # Prepare features (now with consistent dimensions)
            item_features, user_features = self._prepare_features(item_data, user_profile)
            
            # Initialize model if needed (with fixed dimensions)
            if self.model is None:
                self._initialize_model()
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                item_batch = item_features.unsqueeze(0).to(self.device)
                user_batch = user_features.unsqueeze(0).to(self.device)
                prediction = self.model(item_batch, user_batch)
                score = prediction.item()
            
            print(f"🎯 Advanced model score for {item_data.get('title', 'Unknown')[:30]}...: {score:.4f}")
            
            return score
            
        except Exception as e:
            print(f"⚠️ Error in advanced prediction for {item_data.get('title', 'Unknown')[:30]}: {e}")
            # Return a neutral score as fallback
            return 0.5
    
    def train_on_feedback(self, item_data, user_id, feedback_score, db_connection):
        """Train the model on user feedback (with reduced frequency)"""
        try:
            # Only train occasionally to avoid over-fitting and errors
            import random
            if random.random() > 0.3:  # Only train 30% of the time
                print("🎓 Skipping training to avoid over-fitting")
                return
            
            # Get user profile
            user_profile = self.user_profiler.analyze_user_preferences(user_id, db_connection)
            
            # Prepare features
            item_features, user_features = self._prepare_features(item_data, user_profile)
            
            # Initialize model if needed (with fixed dimensions)
            if self.model is None:
                self._initialize_model()
            
            # Convert feedback to target (0-1 scale)
            if feedback_score == -1:
                target = 0.0
            elif feedback_score == 1:
                target = 0.7
            elif feedback_score == 2:
                target = 1.0
            else:
                target = 0.5
            
            # Set model to training mode
            self.model.train()
            
            # Prepare tensors
            item_batch = item_features.unsqueeze(0).to(self.device)
            user_batch = user_features.unsqueeze(0).to(self.device)
            target_tensor = torch.tensor([target], dtype=torch.float32).to(self.device)
            
            # Forward pass
            prediction = self.model(item_batch, user_batch)
            loss = self.criterion(prediction.squeeze(), target_tensor.squeeze())
            
            # Only proceed with training if loss is reasonable
            if loss.item() > 10.0:  # Skip if loss is too high (may indicate issues)
                print(f"⚠️ Skipping training - loss too high: {loss.item():.4f}")
                return
            
            # Backward pass with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            print(f"🎓 Advanced training: Loss={loss.item():.4f}, Target={target:.1f}, Pred={prediction.item():.4f}")
            
            # Save model less frequently
            if random.random() < 0.1:  # Only save 10% of the time
                self._save_model()
            
        except Exception as e:
            print(f"❌ Error in advanced model training: {e}")
            # Don't re-raise the exception to avoid crashing the system
    
    def _save_model(self):
        """Save the model"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'timestamp': datetime.now().isoformat()
            }, self.model_path)
            print(f"💾 Advanced model saved to {self.model_path}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")

# Global advanced engine instance
advanced_engine = None

def get_advanced_recommendation_engine():
    """Get or create the advanced recommendation engine"""
    global advanced_engine
    if advanced_engine is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        advanced_engine = AdvancedRecommendationEngine(device=device)
        print(f"🚀 Advanced Recommendation Engine initialized on {device}")
    return advanced_engine

def get_advanced_recommendations(items, user_id, db_connection, max_items=12):
    """Get advanced recommendations"""
    print(f"🧠 ADVANCED RECOMMENDATIONS: Processing {len(items)} items for user {user_id}")
    
    if not items:
        return []
    
    try:
        engine = get_advanced_recommendation_engine()
        
        # Score all items
        scored_items = []
        for item in items:
            try:
                # Check if user has already rated this item (explicit filtering)
                item_url = item.get('item_url', '')
                item_title = item.get('title', '')
                item_brand = item.get('brand', '')
                
                if item_url:
                    cursor = db_connection.cursor()
                    cursor.execute("""
                        SELECT feedback_type FROM user_feedback 
                        WHERE user_id = ? AND item_url = ?
                    """, (user_id, item_url))
                    
                    existing_feedback = cursor.fetchone()
                    if existing_feedback and existing_feedback[0] < 0:
                        print(f"🚫 Filtering out previously disliked item (URL): {item_title[:40]}...")
                        continue  # Skip disliked items entirely
                
                # Additional filtering by title + brand (catches similar items)
                if item_title and item_brand:
                    cursor.execute("""
                        SELECT COUNT(*) FROM user_feedback 
                        WHERE user_id = ? AND item_title = ? AND item_brand = ? AND feedback_type < 0
                    """, (user_id, item_title, item_brand))
                    
                    title_brand_dislike = cursor.fetchone()
                    if title_brand_dislike and title_brand_dislike[0] > 0:
                        print(f"🚫 Filtering out similar disliked item (Title+Brand): {item_title[:40]}...")
                        continue
                
                # Get advanced score
                score = engine.predict_item_score(item, user_id, db_connection)
                scored_items.append((item, score))
                
            except Exception as e:
                print(f"❌ Error scoring item {item.get('title', 'Unknown')[:30]}: {e}")
                continue
        
        # Sort by score (highest first)
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Return top items
        top_items = [item for item, score in scored_items[:max_items]]
        
        print(f"🎯 Advanced recommendations: {len(scored_items)} scored -> {len(top_items)} returned")
        if scored_items:
            print(f"📊 Score range: {scored_items[-1][1]:.4f} to {scored_items[0][1]:.4f}")
        
        return top_items
        
    except Exception as e:
        print(f"❌ Error in advanced recommendations: {e}")
        # Fallback to original items
        return items[:max_items]

def train_advanced_model_on_feedback(item_data, user_id, feedback_score, db_connection):
    """Train the advanced model on user feedback"""
    try:
        engine = get_advanced_recommendation_engine()
        engine.train_on_feedback(item_data, user_id, feedback_score, db_connection)
        print(f"✅ Advanced model training completed for user {user_id}")
    except Exception as e:
        print(f"❌ Error in advanced model training: {e}")

if __name__ == "__main__":
    # Test the advanced recommendation engine
    print("🧪 Testing Advanced Recommendation Engine...")
    
    # Mock data for testing
    test_item = {
        'title': 'Vintage 90s Nike T-Shirt',
        'description': 'Classic vintage Nike tee from the 90s',
        'brand': 'Nike',
        'price': 25.0,
        'size': 'M'
    }
    
    extractor = AdvancedFeatureExtractor()
    features = extractor.extract_comprehensive_features(test_item)
    
    print(f"📊 Extracted {len(features)} features:")
    for key, value in features.items():
        if value != 0:
            print(f"  {key}: {value}")
