#!/usr/bin/env python3
"""
Advanced Contextual Retrieval System for RL-Enhanced Recommendations
Implements dynamic query reformulation and multi-armed retrieval strategies
"""

import random
import numpy as np
from collections import defaultdict, Counter
import sqlite3
import json
from datetime import datetime, timedelta

class ContextualMultiArmedBandit:
    """Multi-armed bandit for retrieval strategy selection"""
    
    def __init__(self, epsilon=0.1, decay_rate=0.99):
        self.epsilon = epsilon  # Exploration rate
        self.decay_rate = decay_rate
        self.arm_counts = defaultdict(int)
        self.arm_rewards = defaultdict(list)
        self.contextual_performance = defaultdict(lambda: defaultdict(list))
        
        # Define retrieval arms/strategies
        self.retrieval_arms = [
            'base_query',           # Original user query
            'preference_enhanced',  # Query + user preferences
            'semantic_expansion',   # Expand with related terms
            'trending_similar',     # Popular items in category
            'cross_category',       # Explore related categories
            'brand_focused',        # Focus on preferred brands
            'style_focused',        # Focus on preferred styles
            'exploration_random'    # Pure exploration
        ]
    
    def select_retrieval_strategy(self, user_context):
        """Select retrieval strategy using contextual epsilon-greedy"""
        # Get context key for user (new vs returning, preference strength, etc.)
        context_key = self._get_context_key(user_context)
        
        # Epsilon-greedy with context awareness
        if random.random() < self.epsilon:
            # Explore: choose random strategy
            strategy = random.choice(self.retrieval_arms)
            return strategy, "exploration"
        else:
            # Exploit: choose best performing strategy for this context
            best_strategy = self._get_best_strategy_for_context(context_key)
            return best_strategy, "exploitation"
    
    def update_strategy_reward(self, strategy, context, reward):
        """Update reward for a strategy given context"""
        context_key = self._get_context_key(context)
        self.arm_counts[strategy] += 1
        self.arm_rewards[strategy].append(reward)
        self.contextual_performance[context_key][strategy].append(reward)
        
        # Decay epsilon over time (learn to exploit more)
        self.epsilon = max(0.01, self.epsilon * self.decay_rate)
    
    def _get_context_key(self, user_context):
        """Generate context key for user"""
        feedback_count = user_context.get('feedback_count', 0)
        preference_strength = user_context.get('preference_strength', 0.0)
        
        # Categorize users
        if feedback_count < 5:
            user_type = 'new'
        elif feedback_count < 20:
            user_type = 'learning'
        else:
            user_type = 'established'
        
        if preference_strength > 0.7:
            pref_type = 'strong'
        elif preference_strength > 0.3:
            pref_type = 'moderate'
        else:
            pref_type = 'weak'
        
        return f"{user_type}_{pref_type}"
    
    def _get_best_strategy_for_context(self, context_key):
        """Get best performing strategy for context"""
        context_performance = self.contextual_performance.get(context_key, {})
        
        if not context_performance:
            return 'base_query'  # Safe default
        
        # Calculate average reward for each strategy in this context
        strategy_scores = {}
        for strategy, rewards in context_performance.items():
            if rewards:
                strategy_scores[strategy] = np.mean(rewards)
        
        if strategy_scores:
            return max(strategy_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'base_query'

class AdaptiveQueryReformulator:
    """RL-based query reformulation system"""
    
    def __init__(self):
        self.bandit = ContextualMultiArmedBandit()
        self.style_keywords = {
            'vintage': ['retro', 'throwback', '90s', '80s', '70s', 'y2k', 'classic'],
            'streetwear': ['urban', 'street', 'hip-hop', 'casual', 'cool'],
            'formal': ['business', 'professional', 'office', 'smart', 'elegant'],
            'boho': ['bohemian', 'hippie', 'free-spirited', 'flowing', 'artistic'],
            'minimalist': ['simple', 'clean', 'basic', 'essential', 'minimal'],
            'grunge': ['alternative', 'edgy', 'rock', 'punk-inspired', 'distressed']
        }
        
        self.category_relations = {
            'tops': ['shirts', 'blouses', 'sweaters', 'hoodies', 't-shirts'],
            'bottoms': ['jeans', 'trousers', 'shorts', 'skirts', 'pants'],
            'dresses': ['midi', 'maxi', 'mini', 'cocktail', 'casual'],
            'outerwear': ['jackets', 'coats', 'blazers', 'cardigans'],
            'shoes': ['sneakers', 'boots', 'heels', 'flats', 'sandals'],
            'accessories': ['bags', 'jewelry', 'hats', 'scarves', 'belts']
        }
    
    def get_enhanced_retrieval_strategies(self, original_query, filters, user_preferences, user_context):
        """Get multiple retrieval strategies based on RL bandit selection"""
        strategies = []
        
        # Select primary strategy
        strategy, strategy_type = self.bandit.select_retrieval_strategy(user_context)
        
        print(f"🎯 Selected retrieval strategy: {strategy} ({strategy_type})")
        
        if strategy == 'base_query':
            strategies.append(('base', original_query, filters))
            
        elif strategy == 'preference_enhanced':
            enhanced_queries = self._generate_preference_enhanced_queries(
                original_query, filters, user_preferences
            )
            strategies.extend(enhanced_queries)
            
        elif strategy == 'semantic_expansion':
            expanded_queries = self._generate_semantic_expansions(
                original_query, filters
            )
            strategies.extend(expanded_queries)
            
        elif strategy == 'trending_similar':
            trending_queries = self._generate_trending_queries(
                original_query, filters
            )
            strategies.extend(trending_queries)
            
        elif strategy == 'cross_category':
            cross_queries = self._generate_cross_category_queries(
                original_query, filters, user_preferences
            )
            strategies.extend(cross_queries)
            
        elif strategy == 'brand_focused':
            brand_queries = self._generate_brand_focused_queries(
                original_query, filters, user_preferences
            )
            strategies.extend(brand_queries)
            
        elif strategy == 'style_focused':
            style_queries = self._generate_style_focused_queries(
                original_query, filters, user_preferences
            )
            strategies.extend(style_queries)
            
        elif strategy == 'exploration_random':
            random_queries = self._generate_exploration_queries(
                original_query, filters
            )
            strategies.extend(random_queries)
        
        # Always include base query as backup
        if not any(s[0] == 'base' for s in strategies):
            strategies.append(('base', original_query, filters))
        
        return strategies, strategy
    
    def _generate_preference_enhanced_queries(self, query, filters, preferences):
        """Generate queries enhanced with user preferences"""
        strategies = []
        
        # Add preferred styles to query
        if 'category' in preferences:
            for pref in preferences['category'][:2]:  # Top 2 categories
                if pref['score'] > 0.3:
                    enhanced_query = f"{query} {pref['value']}"
                    strategies.append(('preference_category', enhanced_query, filters))
        
        # Add preferred brands
        if 'brand' in preferences:
            for pref in preferences['brand'][:2]:  # Top 2 brands
                if pref['score'] > 0.2:
                    enhanced_filters = filters.copy()
                    enhanced_filters['brand'] = pref['value']
                    strategies.append(('preference_brand', query, enhanced_filters))
        
        # Add preferred colors
        if 'color' in preferences:
            for pref in preferences['color'][:2]:
                if pref['score'] > 0.2:
                    enhanced_query = f"{query} {pref['value']}"
                    strategies.append(('preference_color', enhanced_query, filters))
        
        return strategies
    
    def _generate_semantic_expansions(self, query, filters):
        """Generate semantically expanded queries"""
        strategies = []
        
        # Expand with style synonyms
        query_lower = query.lower()
        for style, keywords in self.style_keywords.items():
            if any(kw in query_lower for kw in [style] + keywords):
                for keyword in keywords[:2]:  # Use top 2 related terms
                    expanded_query = f"{query} {keyword}"
                    strategies.append(('semantic_expansion', expanded_query, filters))
                break
        
        return strategies
    
    def _generate_trending_queries(self, query, filters):
        """Generate trending/popular item queries"""
        trending_terms = ['popular', 'trending', 'bestseller', 'hot', 'new']
        strategies = []
        
        for term in trending_terms[:2]:
            trending_query = f"{term} {query}"
            strategies.append(('trending', trending_query, filters))
        
        return strategies
    
    def _generate_cross_category_queries(self, query, filters, preferences):
        """Generate cross-category exploration queries"""
        strategies = []
        
        # If user likes one category, try related ones
        if 'category' in preferences:
            liked_categories = [p['value'] for p in preferences['category'] if p['score'] > 0.2]
            
            for category in liked_categories[:1]:  # Top category only
                if category.lower() in self.category_relations:
                    related_items = self.category_relations[category.lower()]
                    for related in related_items[:2]:
                        cross_query = f"{query} {related}"
                        strategies.append(('cross_category', cross_query, filters))
        
        return strategies
    
    def _generate_brand_focused_queries(self, query, filters, preferences):
        """Generate brand-focused queries"""
        strategies = []
        
        if 'brand' in preferences:
            top_brands = [p['value'] for p in preferences['brand'][:3] if p['score'] > 0.1]
            
            for brand in top_brands:
                brand_filters = filters.copy()
                brand_filters['brand'] = brand
                strategies.append(('brand_focus', query, brand_filters))
        
        return strategies
    
    def _generate_style_focused_queries(self, query, filters, preferences):
        """Generate style-focused queries"""
        strategies = []
        
        # Extract style preferences from categories/subcategories
        style_terms = []
        if 'subcategory' in preferences:
            style_terms.extend([p['value'] for p in preferences['subcategory'][:2] if p['score'] > 0.2])
        
        for style in style_terms:
            style_query = f"{query} {style}"
            strategies.append(('style_focus', style_query, filters))
        
        return strategies
    
    def _generate_exploration_queries(self, query, filters):
        """Generate exploration queries for discovery"""
        strategies = []
        
        exploration_terms = ['unique', 'rare', 'special', 'limited', 'exclusive']
        random.shuffle(exploration_terms)
        
        for term in exploration_terms[:2]:
            exploration_query = f"{term} {query}"
            strategies.append(('exploration', exploration_query, filters))
        
        return strategies
    
    def update_strategy_performance(self, strategy_name, user_context, user_feedback_on_results):
        """Update strategy performance based on user feedback"""
        # Convert feedback to reward signal
        # Positive feedback on results from this strategy = positive reward
        reward = self._calculate_strategy_reward(user_feedback_on_results)
        
        self.bandit.update_strategy_reward(strategy_name, user_context, reward)
        print(f"📈 Updated strategy '{strategy_name}' with reward: {reward}")
    
    def _calculate_strategy_reward(self, feedback_results):
        """Calculate reward based on user feedback on retrieved results"""
        if not feedback_results:
            return 0.0
        
        # Count positive vs negative feedback
        positive_count = len([f for f in feedback_results if f > 0])
        negative_count = len([f for f in feedback_results if f < 0])
        total_count = len(feedback_results)
        
        if total_count == 0:
            return 0.0
        
        # Reward = (positive_rate - negative_rate) * engagement_bonus
        positive_rate = positive_count / total_count
        negative_rate = negative_count / total_count
        base_reward = positive_rate - negative_rate
        
        # Bonus for high engagement (lots of feedback)
        engagement_bonus = min(1.2, 1.0 + (total_count * 0.1))
        
        return base_reward * engagement_bonus

# Global instance
adaptive_retrieval_system = None

def get_adaptive_retrieval_system():
    """Get or create adaptive retrieval system"""
    global adaptive_retrieval_system
    if adaptive_retrieval_system is None:
        adaptive_retrieval_system = AdaptiveQueryReformulator()
        print("🎯 Adaptive Retrieval System initialized")
    return adaptive_retrieval_system

def get_contextual_user_info(user_id, db_connection):
    """Get contextual information about user for bandit"""
    cursor = db_connection.cursor()
    
    # Get feedback count
    cursor.execute("SELECT COUNT(*) FROM user_feedback WHERE user_id = ?", (user_id,))
    feedback_result = cursor.fetchone()
    feedback_count = feedback_result[0] if feedback_result else 0
    
    # Get preference strength (how consistent are they?)
    cursor.execute("""
        SELECT feedback_type, COUNT(*) 
        FROM user_feedback 
        WHERE user_id = ? 
        GROUP BY feedback_type
    """, (user_id,))
    
    feedback_distribution = cursor.fetchall()
    preference_strength = 0.0
    
    if feedback_distribution:
        total_feedback = sum(count for _, count in feedback_distribution)
        if total_feedback > 0:
            # Calculate consistency (how much do they stick to likes vs dislikes?)
            max_feedback_type_count = max(count for _, count in feedback_distribution)
            preference_strength = max_feedback_type_count / total_feedback
    
    return {
        'feedback_count': feedback_count,
        'preference_strength': preference_strength,
        'user_type': 'new' if feedback_count < 5 else ('learning' if feedback_count < 20 else 'established')
    }

if __name__ == "__main__":
    # Test the adaptive retrieval system
    print("🧪 Testing Adaptive Retrieval System...")
    
    system = get_adaptive_retrieval_system()
    
    # Mock user context and preferences
    user_context = {'feedback_count': 12, 'preference_strength': 0.6}
    user_preferences = {
        'brand': [{'value': 'Nike', 'score': 0.4}],
        'category': [{'value': 'sneakers', 'score': 0.7}]
    }
    
    strategies, selected_strategy = system.get_enhanced_retrieval_strategies(
        "vintage t-shirt", {}, user_preferences, user_context
    )
    
    print(f"🎯 Selected strategy: {selected_strategy}")
    print(f"📝 Generated {len(strategies)} retrieval strategies:")
    for i, (strategy_type, query, filters) in enumerate(strategies):
        print(f"  {i+1}. {strategy_type}: '{query}' with filters: {filters}")
