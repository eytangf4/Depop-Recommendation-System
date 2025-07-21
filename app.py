from flask import Flask, request, render_template, redirect, session, g, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
import json
from depop_api import fetch_depop_items

# Import Deep Learning RL Model
try:
    from deep_rl_model import get_deep_rl_recommendations, train_dqn_from_feedback, get_dqn_agent
    DEEP_RL_AVAILABLE = True
    print("Deep Learning RL model loaded successfully")
except ImportError as e:
    print(f"Deep Learning RL model not available: {e}")
    print("Falling back to basic RL system")
    DEEP_RL_AVAILABLE = False

# Import Advanced Neural Network Recommendation Engine
try:
    from advanced_recommendation_engine import get_advanced_recommendations, get_advanced_recommendation_engine
    ADVANCED_MODEL_AVAILABLE = True
    print("🧠 Advanced Neural Network model loaded successfully")
except ImportError as e:
    print(f"Advanced Neural Network model not available: {e}")
    ADVANCED_MODEL_AVAILABLE = False

# Import Contextual Retrieval System
try:
    from contextual_retrieval import get_adaptive_retrieval_system, get_contextual_user_info
    CONTEXTUAL_RETRIEVAL_AVAILABLE = True
    print("🎯 Contextual Retrieval System loaded successfully")
except ImportError as e:
    print(f"Contextual Retrieval System not available: {e}")
    CONTEXTUAL_RETRIEVAL_AVAILABLE = False

# Configuration: Use Advanced Neural Network as primary system
USE_ADVANCED_MODEL_PRIMARY = True  # Set to False to use DQN as primary

# --- Reinforcement Learning Functions ---

def update_user_preferences(user_id):
    """Update user preferences based on their feedback history"""
    try:
        db = get_db()
        cursor = db.cursor()
        
        # Get all user feedback
        cursor.execute("""
            SELECT item_brand, item_category, item_subcategory, item_color, 
                   item_condition, item_price, feedback_type, timestamp
            FROM user_feedback 
            WHERE user_id = ?
            ORDER BY timestamp DESC
        """, (user_id,))
        
        feedback_data = cursor.fetchall()
        
        if not feedback_data:
            return
        
        # Calculate preference scores for different attributes
        preferences = {}
        
        for row in feedback_data:
            brand, category, subcategory, color, condition, price, feedback_type, timestamp = row
            
            # Weight recent feedback more heavily (recency bias)
            import datetime
            from datetime import datetime as dt
            try:
                feedback_time = dt.fromisoformat(timestamp.replace('Z', '+00:00'))
                days_ago = (dt.now() - feedback_time).days
                time_weight = max(0.1, 1 / (1 + days_ago * 0.1))  # Decay over time
            except:
                time_weight = 1.0
            
            weighted_score = feedback_type * time_weight
            
            # Update preferences for each attribute
            if brand:
                preferences.setdefault('brand', {}).setdefault(brand, []).append(weighted_score)
            if category:
                preferences.setdefault('category', {}).setdefault(category, []).append(weighted_score)
            if subcategory:
                preferences.setdefault('subcategory', {}).setdefault(subcategory, []).append(weighted_score)
            if color:
                preferences.setdefault('color', {}).setdefault(color, []).append(weighted_score)
            if condition:
                preferences.setdefault('condition', {}).setdefault(condition, []).append(weighted_score)
            if price:
                # Price range preferences
                if price < 20:
                    price_range = 'budget'
                elif price < 50:
                    price_range = 'mid'
                elif price < 100:
                    price_range = 'premium'
                else:
                    price_range = 'luxury'
                preferences.setdefault('price_range', {}).setdefault(price_range, []).append(weighted_score)
        
        # Calculate final preference scores and store them
        for pref_type, pref_values in preferences.items():
            for pref_value, scores in pref_values.items():
                # Calculate average score for this preference
                avg_score = sum(scores) / len(scores)
                
                # Store/update preference
                cursor.execute("""
                    INSERT OR REPLACE INTO user_preferences 
                    (user_id, preference_type, preference_value, preference_score, last_updated)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (user_id, pref_type, pref_value, avg_score))
        
        db.commit()
        print(f"Updated preferences for user {user_id}")
        
    except Exception as e:
        print(f"Error updating user preferences: {e}")

def get_user_preferences(user_id):
    """Get user preferences for recommendation generation"""
    try:
        db = get_db()
        cursor = db.cursor()
        
        cursor.execute("""
            SELECT preference_type, preference_value, preference_score
            FROM user_preferences 
            WHERE user_id = ? AND preference_score > 0
            ORDER BY preference_score DESC
        """, (user_id,))
        
        rows = cursor.fetchall()
        
        preferences = {}
        for pref_type, pref_value, score in rows:
            if pref_type not in preferences:
                preferences[pref_type] = []
            preferences[pref_type].append({'value': pref_value, 'score': score})
        
        return preferences
        
    except Exception as e:
        print(f"Error getting user preferences: {e}")
        return {}

def get_rl_recommendations(query, filters, user_preferences, user_id, max_items=24):
    """Generate recommendations using advanced RL with contextual retrieval"""
    try:
        db = get_db()
        all_candidates = []
        retrieval_strategy_used = "base_query"
        
        base_cursor = None  # Track cursor for pagination
        
        # Use contextual multi-armed bandit for retrieval strategy selection
        if CONTEXTUAL_RETRIEVAL_AVAILABLE:
            try:
                retrieval_system = get_adaptive_retrieval_system()
                user_context = get_contextual_user_info(user_id, db)
                
                # Get multiple retrieval strategies from RL-enhanced system
                retrieval_strategies, primary_strategy = retrieval_system.get_enhanced_retrieval_strategies(
                    query, filters, user_preferences, user_context
                )
                retrieval_strategy_used = primary_strategy
                
                # Store strategy for performance tracking
                g.last_retrieval_strategy = primary_strategy
                
                print(f"🎯 RL Contextual Retrieval: Using {len(retrieval_strategies)} strategies, primary: {primary_strategy}")
                
                # Execute each retrieval strategy
                for strategy_type, strategy_query, strategy_filters in retrieval_strategies[:4]:  # Limit to 4 strategies
                    try:
                        strategy_items, cursor = fetch_depop_items(
                            strategy_query, 
                            **convert_filters_to_api(strategy_filters), 
                            max_items=15
                        )
                        all_candidates.extend(strategy_items)
                        
                        # Keep the cursor from the base strategy for pagination
                        if strategy_type in ['base', 'base_query'] and cursor:
                            base_cursor = cursor
                        
                        print(f"  📥 Strategy '{strategy_type}': {len(strategy_items)} items retrieved")
                    except Exception as e:
                        print(f"  ❌ Strategy '{strategy_type}' failed: {e}")
                        
            except Exception as e:
                print(f"⚠️ Contextual retrieval failed, using fallback: {e}")
                # Fallback to original approach
                base_items, base_cursor = fetch_depop_items(query, **convert_filters_to_api(filters), max_items=max_items//2)
                all_candidates = list(base_items)
        else:
            # Original approach: Base search + preference-enhanced searches
            base_items, base_cursor = fetch_depop_items(query, **convert_filters_to_api(filters), max_items=max_items//2)
            all_candidates = list(base_items)
            
            # Strategy 2: Enhanced searches based on user preferences
            if user_preferences:
                # Generate preference-based search variations
                enhanced_queries = generate_preference_queries(query, filters, user_preferences)
                
                for enhanced_query, enhanced_filters in enhanced_queries:
                    try:
                        pref_items, _ = fetch_depop_items(enhanced_query, **enhanced_filters, max_items=20)
                        all_candidates.extend(pref_items)
                    except Exception as e:
                        print(f"Error in preference-based search: {e}")
        
        # Remove duplicates based on item_url
        seen_urls = set()
        unique_candidates = []
        for item in all_candidates:
            if item.get('item_url') and item['item_url'] not in seen_urls:
                seen_urls.add(item['item_url'])
                unique_candidates.append(item)
        
        # Use model based on configuration
        if USE_ADVANCED_MODEL_PRIMARY and ADVANCED_MODEL_AVAILABLE and len(unique_candidates) > 0:
            try:
                final_items = get_advanced_recommendations(unique_candidates, user_id, db, max_items)
                print(f"🧠 Advanced Neural Network + RL Retrieval ({retrieval_strategy_used}): {len(all_candidates)} candidates -> {len(unique_candidates)} unique -> {len(final_items)} final")
                return final_items, base_cursor
            except Exception as e:
                print(f"Advanced Neural Network failed, falling back: {e}")
        elif DEEP_RL_AVAILABLE and len(unique_candidates) > 0:
            try:
                final_items = get_deep_rl_recommendations(unique_candidates, user_id, db, max_items)
                print(f"🔄 Deep Q-Network + RL Retrieval ({retrieval_strategy_used}): {len(all_candidates)} candidates -> {len(unique_candidates)} unique -> {len(final_items)} final")
                return final_items, base_cursor
            except Exception as e:
                print(f"Deep RL failed, falling back to basic scoring: {e}")
        
        # Fallback to basic preference scoring
        scored_items = score_items_by_preferences(unique_candidates, user_preferences)
        final_items = scored_items[:max_items]
        
        print(f"📊 Basic Scoring + RL Retrieval ({retrieval_strategy_used}): {len(all_candidates)} candidates -> {len(unique_candidates)} unique -> {len(final_items)} final")
        
        return final_items, base_cursor
        
    except Exception as e:
        print(f"Error in RL recommendations: {e}")
        # Fallback to basic search
        return fetch_depop_items(query, **convert_filters_to_api(filters), max_items=max_items)

def convert_filters_to_api(filters):
    """Convert internal filters to Depop API format"""
    api_filters = {
        "gender": filters["gender"] if filters["gender"] not in ["men", "women"] else ("male" if filters["gender"] == "men" else "female"),
        "category": filters["category"],
        "subcategory": filters["subcategory"][0] if filters["subcategory"] else None,
        "brand": filters["brand"] if filters["brand"] else None,
        "size": filters["size"] if filters["size"] else None,
        "color": filters["color"] if filters["color"] else None,
        "condition": filters["condition"] if filters["condition"] else None,
        "price_min": filters["price_min"],
        "price_max": filters["price_max"],
        "on_sale": filters["on_sale"],
        "sort": filters["sort"]
    }
    return api_filters

def generate_preference_queries(base_query, base_filters, user_preferences):
    """Generate additional search queries based on user preferences"""
    queries = []
    
    # Valid brands that work with Depop API (avoid problematic ones)
    valid_brands = {
        'Nike', 'Adidas', 'Zara', 'H&M', 'Urban Outfitters', 'American Eagle',
        'Forever 21', 'Brandy Melville', 'Vintage', 'Thrifted', 'Unbranded'
    }
    
    # Strategy: Add preferred brands to search
    if 'brand' in user_preferences:
        top_brands = [p['value'] for p in user_preferences['brand'][:3] if p['value'] in valid_brands]
        for brand in top_brands:
            enhanced_filters = convert_filters_to_api(base_filters.copy())
            if not enhanced_filters['brand']:  # Only if no brand filter is already set
                enhanced_filters['brand'] = [brand]
                queries.append((base_query, enhanced_filters))
    
    # Strategy: Add preferred colors
    if 'color' in user_preferences and not base_filters.get('color'):
        top_colors = [p['value'] for p in user_preferences['color'][:2]]
        for color in top_colors:
            enhanced_filters = convert_filters_to_api(base_filters.copy())
            enhanced_filters['color'] = [color]
            queries.append((base_query, enhanced_filters))
    
    # Strategy: Search in preferred categories
    if 'category' in user_preferences and not base_filters.get('category'):
        top_categories = [p['value'] for p in user_preferences['category'][:2]]
        for category in top_categories:
            enhanced_filters = convert_filters_to_api(base_filters.copy())
            enhanced_filters['category'] = category
            queries.append((base_query, enhanced_filters))
    
    return queries[:3]  # Limit to 3 additional queries to avoid API overload

def score_items_by_preferences(items, user_preferences):
    """Score items based on user preferences and return sorted list"""
    scored_items = []
    
    for item in items:
        score = 0.0
        
        # Base score (all items start equal)
        score += 1.0
        
        # Score based on brand preference
        if 'brand' in user_preferences and item.get('brand'):
            for pref in user_preferences['brand']:
                if pref['value'].lower() == item['brand'].lower():
                    score += pref['score'] * 2.0  # Brand preference is important
                    break
        
        # Score based on category preference (if we had category data)
        # This would require extracting category from item data
        
        # Score based on price preference
        if 'price_range' in user_preferences and item.get('price'):
            item_price = float(item['price']) if item['price'] else 0
            item_price_range = 'budget' if item_price < 20 else 'mid' if item_price < 50 else 'premium' if item_price < 100 else 'luxury'
            
            for pref in user_preferences['price_range']:
                if pref['value'] == item_price_range:
                    score += pref['score'] * 1.5
                    break
        
        # Add item with score
        scored_items.append((item, score))
    
    # Sort by score (descending) and return items
    scored_items.sort(key=lambda x: x[1], reverse=True)
    return [item for item, score in scored_items]

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fallback_dev_key")
DATABASE = "users.db"

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            );
        """)
        
        # User feedback table for reinforcement learning
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                item_url TEXT NOT NULL,
                item_title TEXT,
                item_brand TEXT,
                item_size TEXT,
                item_price REAL,
                item_category TEXT,
                item_subcategory TEXT,
                item_color TEXT,
                item_condition TEXT,
                item_image TEXT,
                feedback_type INTEGER NOT NULL,  -- 2: love, 1: like, -1: dislike
                search_query TEXT,
                search_filters TEXT,  -- JSON string of filters used
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                UNIQUE(user_id, item_url)  -- Prevent duplicate feedback for same item
            );
        """)
        
        # User preferences learned from feedback (for faster recommendations)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                preference_type TEXT NOT NULL,  -- 'brand', 'category', 'color', 'price_range', etc.
                preference_value TEXT NOT NULL,
                preference_score REAL NOT NULL,  -- Calculated preference strength
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                UNIQUE(user_id, preference_type, preference_value)
            );
        """)
        
        # Search history for context-aware recommendations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                search_query TEXT NOT NULL,
                search_filters TEXT,  -- JSON string
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            );
        """)
        
        db.commit()

@app.route("/")
def home():
    return render_template("login.html")


@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400
    password_hash = generate_password_hash(password)
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
        db.commit()
        user_id = cursor.lastrowid
        session["user_id"] = user_id
        session["username"] = username
        return jsonify({"message": "User created and logged in", "redirect": "/menu"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username already exists"}), 409


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    db = get_db()
    user = db.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,)).fetchone()
    if user and check_password_hash(user[1], password):
        session["user_id"] = user[0]
        session["username"] = username
        return jsonify({"message": "Logged in", "redirect": "/menu"})
    return jsonify({"error": "Invalid credentials"}), 401

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

@app.route("/reset-password", methods=["POST"])
def reset_password():
    """Simple password reset for development purposes"""
    data = request.get_json()
    username = data.get("username")
    new_password = data.get("new_password")
    
    if not username or not new_password:
        return jsonify({"error": "Username and new password required"}), 400
    
    db = get_db()
    user = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    # Generate new password hash
    password_hash = generate_password_hash(new_password)
    
    # Update password
    db.execute("UPDATE users SET password_hash = ? WHERE username = ?", (password_hash, username))
    db.commit()
    
    return jsonify({"message": f"Password reset successfully for {username}"})

@app.route("/menu")
def menu():
    if "user_id" not in session:
        return redirect("/")
    return render_template("menu.html", username=session.get("username"))


# --- Depop Item Search Route ---

import threading
import time
import queue
import json
from flask import Response, jsonify
from depop_api import fetch_depop_items


# --- Streaming progress for search ---
search_progress = {}

@app.route("/search", methods=["GET"])
def search():
    if "user_id" not in session:
        return redirect("/")
    query = request.args.get("query", "")
    return render_template("search.html", items=None, query=query)

@app.route("/search_results", methods=["POST"])
def search_results():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json()
    query = data.get("query", "")
    user_id = session["user_id"]
    
    # Store search history for learning
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO search_history (user_id, search_query, search_filters)
            VALUES (?, ?, ?)
        """, (user_id, query, json.dumps(data)))
        db.commit()
    except Exception as e:
        print(f"Error storing search history: {e}")
    
    # Handle the new hierarchy: category -> midlevel -> subcategory
    main_category = data.get("category")  # men, women, kids, everything_else
    midlevel = data.get("midlevel")  # tops, bottoms, etc.
    subcategories = data.get("subcategory", []) if isinstance(data.get("subcategory"), list) else ([data.get("subcategory")] if data.get("subcategory") else [])
    
    # Map main category to gender
    gender_mapping = {
        'men': 'male',
        'women': 'female', 
        'kids': 'kids',
        'everything_else': None
    }
    gender = gender_mapping.get(main_category) if main_category else None
    
    # Handle array-based multi-select filters
    filters = {
        "gender": gender,
        "category": midlevel,  # Use midlevel as category for Depop API
        "subcategory": subcategories,
        "brand": data.get("brand", []) if isinstance(data.get("brand"), list) else ([data.get("brand")] if data.get("brand") else []),
        "size": data.get("size", []) if isinstance(data.get("size"), list) else ([data.get("size")] if data.get("size") else []),
        "color": data.get("color", []) if isinstance(data.get("color"), list) else ([data.get("color")] if data.get("color") else []),
        "condition": data.get("condition", []) if isinstance(data.get("condition"), list) else ([data.get("condition")] if data.get("condition") else []),
        "price_min": data.get("price_min"),
        "price_max": data.get("price_max"),
        "on_sale": data.get("on_sale"),
        "sort": data.get("sort", "relevance")
    }
    
    # Get user preferences for RL-enhanced search
    user_preferences = get_user_preferences(user_id)
    
    # Generate RL-enhanced search results with contextual retrieval
    items, next_cursor = get_rl_recommendations(query, filters, user_preferences, user_id)
    
    # Store the retrieval strategy used for performance tracking
    if hasattr(g, 'last_retrieval_strategy'):
        session['last_retrieval_strategy'] = g.last_retrieval_strategy
    
    return jsonify({"items": items, "next_cursor": next_cursor})

@app.route("/search_results_page", methods=["POST"])
def search_results_page():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json()
    query = data.get("query", "")
    cursor = data.get("cursor")
    user_id = session["user_id"]
    
    # Handle the new hierarchy: category -> midlevel -> subcategory
    main_category = data.get("category")  # men, women, kids, everything_else
    midlevel = data.get("midlevel")  # tops, bottoms, etc.
    subcategories = data.get("subcategory", []) if isinstance(data.get("subcategory"), list) else ([data.get("subcategory")] if data.get("subcategory") else [])
    
    # Map main category to gender
    gender_mapping = {
        'men': 'male',
        'women': 'female', 
        'kids': 'kids',
        'everything_else': None
    }
    gender = gender_mapping.get(main_category) if main_category else None
    
    # Handle array-based multi-select filters
    filters = {
        "gender": gender,
        "category": midlevel,  # Use midlevel as category for Depop API
        "subcategory": subcategories,
        "brand": data.get("brand", []) if isinstance(data.get("brand"), list) else ([data.get("brand")] if data.get("brand") else []),
        "size": data.get("size", []) if isinstance(data.get("size"), list) else ([data.get("size")] if data.get("size") else []),
        "color": data.get("color", []) if isinstance(data.get("color"), list) else ([data.get("color")] if data.get("color") else []),
        "condition": data.get("condition", []) if isinstance(data.get("condition"), list) else ([data.get("condition")] if data.get("condition") else []),
        "price_min": data.get("price_min"),
        "price_max": data.get("price_max"),
        "on_sale": data.get("on_sale"),
        "sort": data.get("sort", "relevance")
    }
    
    # Get user preferences for RL-enhanced pagination
    user_preferences = get_user_preferences(user_id)
    
    # Debug logging for pagination
    print(f"DEBUG - Pagination API Filters: {convert_filters_to_api(filters)}")
    
    # Use RL recommendations for pagination too!
    if user_preferences:
        # Generate RL-enhanced pagination results
        items, next_cursor = get_rl_recommendations(query, filters, user_preferences, user_id, max_items=24)
        # TODO: Handle cursor-based pagination with RL (complex feature for future)
        # For now, use basic pagination but with RL scoring
        pass
    
    # Fallback to basic pagination
    api_filters = convert_filters_to_api(filters)
    items, next_cursor = fetch_depop_items(query, **api_filters, max_items=24, offset=cursor)
    
    # Apply RL scoring to paginated results
    if user_preferences:
        items = score_items_by_preferences(items, user_preferences)
    
    return jsonify({"items": items, "next_cursor": next_cursor})

# --- Feedback Route with Reinforcement Learning ---
@app.route("/feedback", methods=["POST"])
def feedback():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json() if request.is_json else request.form.to_dict()
    user_id = session["user_id"]
    
    # Extract feedback data
    item_url = data.get("item_url")
    feedback_value = data.get("feedback")  # 'love', 'like', 'dislike'
    
    # Additional item data for learning (with fallbacks for My Ratings page)
    item_title = data.get("item_title", "")
    item_brand = data.get("item_brand", "")
    item_size = data.get("item_size") or data.get("item_sizes", "")
    item_price = data.get("item_price") or 0
    item_image = data.get("item_image", "")
    item_category = data.get("item_category", "")
    item_subcategory = data.get("item_subcategory", "")
    item_color = data.get("item_color", "")
    item_condition = data.get("item_condition", "")
    search_query = data.get("search_query", "")
    search_filters = data.get("search_filters", "{}")
    
    print(f"📝 Feedback received: URL={item_url}, Type={feedback_value}, Title={item_title}, Brand={item_brand}, Price={item_price}")
    
    # Convert lists/dicts to JSON strings for SQLite storage
    if isinstance(item_size, list):
        item_size = json.dumps(item_size)
    elif item_size and not isinstance(item_size, str):
        item_size = str(item_size)
        
    if isinstance(item_category, list):
        item_category = json.dumps(item_category)
    if isinstance(item_subcategory, list):
        item_subcategory = json.dumps(item_subcategory)
    if isinstance(item_color, list):
        item_color = json.dumps(item_color)
    if isinstance(item_condition, list):
        item_condition = json.dumps(item_condition)
    if isinstance(search_filters, (dict, list)):
        search_filters = json.dumps(search_filters)
    
    if not item_url or not feedback_value:
        return jsonify({"error": "Missing required feedback data"}), 400

    # Handle removal of feedback
    if feedback_value == 'remove':
        try:
            db = get_db()
            cursor = db.cursor()
            
            # Delete the feedback
            cursor.execute("DELETE FROM user_feedback WHERE user_id = ? AND item_url = ?", 
                          (user_id, item_url))
            db.commit()
            
            # Update user preferences
            def update_preferences_with_context():
                with app.app_context():
                    update_user_preferences(user_id)
            
            from threading import Thread
            Thread(target=update_preferences_with_context).start()
            
            return jsonify({
                "message": "Rating removed successfully",
                "training_info": "Neural network weights adjusted for rating removal"
            })
            
        except Exception as e:
            print(f"Error removing feedback: {e}")
            return jsonify({"error": "Failed to remove rating"}), 500

    # Map feedback to numerical values
    feedback_mapping = {
        'love': 2,    # Double thumbs up
        'like': 1,    # Single thumbs up
        'dislike': -1 # Thumbs down
    }
    
    feedback_score = feedback_mapping.get(feedback_value)
    if feedback_score is None:
        return jsonify({"error": "Invalid feedback type"}), 400
    
    try:
        db = get_db()
        cursor = db.cursor()
        
        # Check for recent identical feedback submissions (within 5 seconds) to prevent spam
        cursor.execute("""
            SELECT id, timestamp FROM user_feedback 
            WHERE user_id = ? AND item_url = ? AND feedback_type = ?
            AND timestamp > datetime('now', '-5 seconds')
        """, (user_id, item_url, feedback_score))
        
        recent_duplicate = cursor.fetchone()
        if recent_duplicate:
            print(f"🚫 Preventing duplicate feedback submission within 5 seconds for user {user_id}")
            if request.is_json:
                return jsonify({"message": "Feedback already recorded", "training_info": "No duplicate processing needed"}), 200
            else:
                return redirect(request.referrer or "/search")
        
        # Check if feedback already exists for this item
        cursor.execute("""
            SELECT id FROM user_feedback 
            WHERE user_id = ? AND item_url = ?
        """, (user_id, item_url))
        
        existing_feedback = cursor.fetchone()
        
        if existing_feedback:
            # Update existing feedback
            cursor.execute("""
                UPDATE user_feedback 
                SET feedback_type = ?, timestamp = CURRENT_TIMESTAMP,
                    item_title = ?, item_brand = ?, item_size = ?, item_price = ?,
                    item_category = ?, item_subcategory = ?, item_color = ?, item_condition = ?, item_image = ?
                WHERE user_id = ? AND item_url = ?
            """, (feedback_score, item_title, item_brand, item_size, item_price,
                  item_category, item_subcategory, item_color, item_condition, item_image,
                  user_id, item_url))
        else:
            # Insert new feedback
            cursor.execute("""
                INSERT INTO user_feedback 
                (user_id, item_url, item_title, item_brand, item_size, item_price, 
                 item_category, item_subcategory, item_color, item_condition, item_image,
                 feedback_type, search_query, search_filters, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (user_id, item_url, item_title, item_brand, item_size, item_price,
                  item_category, item_subcategory, item_color, item_condition, item_image,
                  feedback_score, search_query, search_filters))
        
        db.commit()
        
        # Train models based on configuration
        if USE_ADVANCED_MODEL_PRIMARY:
            # Train Advanced Neural Network (primary)
            def update_preferences_with_context():
                with app.app_context():
                    update_user_preferences(user_id)
                    
                    # Update contextual retrieval system performance if available
                    if CONTEXTUAL_RETRIEVAL_AVAILABLE:
                        try:
                            retrieval_system = get_adaptive_retrieval_system()
                            user_context = get_contextual_user_info(user_id, get_db())
                            
                            # Get recent feedback to evaluate retrieval strategy performance
                            db = get_db()
                            cursor = db.cursor()
                            cursor.execute("""
                                SELECT feedback_type FROM user_feedback 
                                WHERE user_id = ? AND timestamp > datetime('now', '-1 hour')
                                ORDER BY timestamp DESC LIMIT 10
                            """, (user_id,))
                            
                            recent_feedback = [row[0] for row in cursor.fetchall()]
                            
                            # Update strategy performance (using Flask g instead of session for thread safety)
                            from flask import g
                            if hasattr(g, 'last_retrieval_strategy'):
                                strategy_name = g.last_retrieval_strategy
                                retrieval_system.update_strategy_performance(
                                    strategy_name, user_context, recent_feedback
                                )
                                print(f"📈 Updated retrieval strategy '{strategy_name}' performance")
                                
                        except Exception as e:
                            print(f"⚠️ Error updating retrieval strategy performance: {e}")
            
            from threading import Thread
            Thread(target=update_preferences_with_context).start()
            print(f"🧠 Advanced Neural Network training initiated for user {user_id}, feedback: {feedback_value}")
            
        elif DEEP_RL_AVAILABLE:
            # Train DQN model (legacy fallback)
            try:
                # Prepare item data for deep learning
                item_data = {
                    'title': item_title,
                    'brand': item_brand,
                    'price': float(item_price) if item_price else 0.0,
                    'category': item_category,
                    'subcategory': item_subcategory,
                    'color': item_color,
                    'condition': item_condition,
                    'size': item_size,
                    'item_url': item_url,
                    'image_url': data.get('item_image_url', ''),  # Add image URL if available
                    'description': data.get('item_description', '')
                }
                
                # Train the DQN model with this feedback
                train_dqn_from_feedback(item_data, user_id, feedback_score, db)
                print(f"🔄 DQN training completed for user {user_id}, feedback: {feedback_value}")
                
            except Exception as e:
                print(f"Error in DQN training: {e}")
        else:
            # Fallback: Update basic user preferences
            def update_preferences_with_context():
                with app.app_context():
                    update_user_preferences(user_id)
            
            from threading import Thread
            Thread(target=update_preferences_with_context).start()
            print(f"📊 Basic preference model updated for user {user_id}, feedback: {feedback_value}")
        
        if request.is_json:
            # Prepare training information based on which model was used
            if USE_ADVANCED_MODEL_PRIMARY:
                training_info = f"🧠 Advanced Neural Network trained with {feedback_value} feedback • Deep learning from your preferences"
            elif DEEP_RL_AVAILABLE:
                training_info = f"🔄 Deep Q-Network trained with {feedback_value} feedback • Reinforcement learning active"
            else:
                training_info = f"📊 Basic preference model updated with {feedback_value} rating"
            
            return jsonify({
                "message": "Feedback recorded successfully",
                "training_info": training_info,
                "model_status": "Learning from your feedback...",
                "primary_model": "Advanced Neural Network" if USE_ADVANCED_MODEL_PRIMARY else ("DQN" if DEEP_RL_AVAILABLE else "Basic")
            })
        else:
            return redirect(request.referrer or "/search")
            
    except Exception as e:
        print(f"Error storing feedback: {e}")
        if request.is_json:
            return jsonify({"error": "Failed to store feedback"}), 500
        else:
            return redirect(request.referrer or "/search")

# --- Get User Feedback Route ---
@app.route("/get_user_feedback", methods=["POST"])
def get_user_feedback():
    """Get user feedback for specific items to restore button states"""
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json()
    user_id = session["user_id"]
    item_urls = data.get("item_urls", [])
    
    if not item_urls:
        return jsonify({"feedback": {}})
    
    try:
        cursor =  get_db().cursor()
        
        # Create placeholders for the IN clause
        placeholders = ','.join('?' for _ in item_urls)
        query = f"""
            SELECT item_url, feedback_type 
            FROM user_feedback 
            WHERE user_id = ? AND item_url IN ({placeholders})
        """
        
        cursor.execute(query, [user_id] + item_urls)
        results = cursor.fetchall()
        
        # Convert feedback_type to readable format
        feedback_map = {2: 'love', 1: 'like', -1: 'dislike'}
        feedback_data = {
            row[0]: feedback_map.get(row[1], 'none') 
            for row in results
        }
        
        return jsonify({"feedback": feedback_data})
        
    except Exception as e:
        print(f"Error retrieving user feedback: {e}")
        return jsonify({"error": "Failed to retrieve feedback"}), 500

# --- My Ratings Routes ---
@app.route("/my-ratings")
def my_ratings():
    """Display user's ratings page"""
    if "user_id" not in session:
        return redirect("/")
    return render_template("my_ratings.html", username=session.get("username", "User"))

@app.route("/get_user_ratings", methods=["GET"])
def get_user_ratings():
    """Get all user ratings with stored item details from database"""
    print(f"🔍 get_user_ratings called, session: {session}")
    
    if "user_id" not in session:
        print("❌ No user_id in session")
        return jsonify({"error": "Unauthorized"}), 401
    
    user_id = session["user_id"]
    print(f"👤 Getting ratings for user_id: {user_id}")
    
    try:
        cursor = get_db().cursor()
        
        # Get all user feedback with stored item details, ordered by most recent first
        cursor.execute("""
            SELECT 
                uf.id as rating_id,
                uf.item_url,
                uf.item_title,
                uf.item_brand,
                uf.item_price,
                uf.item_size,
                uf.item_image,
                uf.feedback_type,
                uf.timestamp
            FROM user_feedback uf
            WHERE uf.user_id = ?
            ORDER BY uf.timestamp DESC
        """, (user_id,))
        
        results = cursor.fetchall()
        print(f"📊 Found {len(results)} ratings in database")
        
        if not results:
            print("📭 No ratings found")
            return jsonify({
                "ratings": [],
                "stats": {"liked": 0, "disliked": 0}
            })
        
        # Convert database results to rating objects
        ratings = []
        feedback_map = {2: 'love', 1: 'like', -1: 'dislike'}
        
        for row in results:
            rating_id, item_url, item_title, item_brand, item_price, item_size, item_image, feedback_type, timestamp = row
            
            # Parse sizes if they're stored as string
            sizes = []
            if item_size:
                try:
                    if isinstance(item_size, str):
                        # Try to parse as JSON first
                        import json
                        sizes = json.loads(item_size)
                    else:
                        sizes = [item_size]
                except:
                    sizes = [str(item_size)]
            
            rating_item = {
                'item_url': item_url,
                'item_title': item_title or 'Unknown Item',
                'item_brand': item_brand or 'Unknown Brand',
                'item_price': item_price or 0,
                'item_image': item_image,
                'sizes': sizes,
                'rating_info': {
                    'rating_id': rating_id,
                    'feedback_type': feedback_map.get(feedback_type, 'none'),
                    'timestamp': timestamp
                }
            }
            ratings.append(rating_item)
            print(f"✅ Added rating: {rating_item['item_title']}")
        
        # Get stats
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN feedback_type = 2 THEN 1 ELSE 0 END) as loved,
                SUM(CASE WHEN feedback_type = 1 THEN 1 ELSE 0 END) as liked,
                SUM(CASE WHEN feedback_type = -1 THEN 1 ELSE 0 END) as disliked
            FROM user_feedback 
            WHERE user_id = ?
        """, (user_id,))
        
        stats_row = cursor.fetchone()
        stats = {
            'loved': stats_row[0] or 0,
            'liked': stats_row[1] or 0,
            'disliked': stats_row[2] or 0
        }
        
        print(f"📊 Final stats: {stats}")
        print(f"🎯 Returning {len(ratings)} ratings")
        
        result = {
            "ratings": ratings,
            "stats": stats
        }
        print(f"📦 Full response: {result}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error retrieving user ratings: {e}")
        return jsonify({"error": "Failed to retrieve ratings"}), 500

@app.route("/update_rating", methods=["POST"])
def update_rating():
    """Update an existing rating"""
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json()
    rating_id = data.get("rating_id")
    feedback_type = data.get("feedback_type")
    user_id = session["user_id"]
    
    if not rating_id or not feedback_type:
        return jsonify({"error": "Missing required fields"}), 400
    
    # Convert feedback type to numeric
    feedback_map = {'love': 2, 'like': 1, 'dislike': -1}
    feedback_value = feedback_map.get(feedback_type)
    
    if feedback_value is None:
        return jsonify({"error": "Invalid feedback type"}), 400
    
    try:
        cursor = get_db().cursor()
        
        # Verify this rating belongs to the user
        cursor.execute("""
            SELECT id FROM user_feedback 
            WHERE id = ? AND user_id = ?
        """, (rating_id, user_id))
        
        if not cursor.fetchone():
            return jsonify({"error": "Rating not found"}), 404
        
        # Update the rating
        from datetime import datetime
        cursor.execute("""
            UPDATE user_feedback 
            SET feedback_type = ?, timestamp = ? 
            WHERE id = ? AND user_id = ?
        """, (feedback_value, datetime.now().isoformat(), rating_id, user_id))
        
        get_db().commit()
        
        # Update contextual retrieval strategy performance
        if CONTEXTUAL_RETRIEVAL_AVAILABLE and hasattr(g, 'last_retrieval_strategy'):
            try:
                retrieval_system = get_adaptive_retrieval_system()
                strategy_name = g.last_retrieval_strategy
                # Convert feedback to reward (like/love = positive, dislike = negative)
                reward = 1.0 if feedback_value > 0 else -0.5
                user_context = get_contextual_user_info(user_id, get_db())
                retrieval_system.update_strategy_performance(strategy_name, reward, user_context)
                print(f"📈 Strategy performance updated: {strategy_name} -> reward: {reward}")
            except Exception as e:
                print(f"Error updating strategy performance: {e}")

        # Trigger model training
        if USE_ADVANCED_MODEL_PRIMARY and ADVANCED_MODEL_AVAILABLE:
            try:
                def train_advanced_model():
                    with app.app_context():
                        from advanced_recommendation_engine import get_advanced_recommendation_engine
                        engine = get_advanced_recommendation_engine()
                        engine.train_from_feedback(user_id, get_db())
                
                from threading import Thread
                Thread(target=train_advanced_model).start()
                print(f"🧠 Advanced Neural Network training initiated for user {user_id}, feedback: {feedback_type}")
            except Exception as e:
                print(f"Error in advanced model training: {e}")
        
        # Update user preferences
        update_user_preferences(user_id)
        
        # Get updated stats
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN feedback_type = 1 OR feedback_type = 2 THEN 1 ELSE 0 END) as liked,
                SUM(CASE WHEN feedback_type = -1 THEN 1 ELSE 0 END) as disliked
            FROM user_feedback 
            WHERE user_id = ?
        """, (user_id,))
        
        stats_row = cursor.fetchone()
        stats = {
            'liked': stats_row[0] or 0,
            'disliked': stats_row[1] or 0
        }
        
        return jsonify({
            "success": True,
            "stats": stats
        })
        
    except Exception as e:
        print(f"Error updating rating: {e}")
        return jsonify({"error": "Failed to update rating"}), 500

@app.route("/delete_rating", methods=["POST"])
def delete_rating():
    """Delete a rating"""
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json()
    rating_id = data.get("rating_id")
    user_id = session["user_id"]
    
    if not rating_id:
        return jsonify({"error": "Missing rating ID"}), 400
    
    try:
        cursor = get_db().cursor()
        
        # Verify this rating belongs to the user
        cursor.execute("""
            SELECT id FROM user_feedback 
            WHERE id = ? AND user_id = ?
        """, (rating_id, user_id))
        
        if not cursor.fetchone():
            return jsonify({"error": "Rating not found"}), 404
        
        # Delete the rating
        cursor.execute("""
            DELETE FROM user_feedback 
            WHERE id = ? AND user_id = ?
        """, (rating_id, user_id))
        
        get_db().commit()
        
        # Update user preferences after deletion
        update_user_preferences(user_id)
        
        # Get updated stats
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN feedback_type = 2 THEN 1 ELSE 0 END) as loved,
                SUM(CASE WHEN feedback_type = 1 THEN 1 ELSE 0 END) as liked,
                SUM(CASE WHEN feedback_type = -1 THEN 1 ELSE 0 END) as disliked
            FROM user_feedback 
            WHERE user_id = ?
        """, (user_id,))
        
        stats_row = cursor.fetchone()
        stats = {
            'loved': stats_row[0] or 0,
            'liked': stats_row[1] or 0,
            'disliked': stats_row[2] or 0
        }
        
        return jsonify({
            "success": True,
            "stats": stats
        })
        
    except Exception as e:
        print(f"Error deleting rating: {e}")
        return jsonify({"error": "Failed to delete rating"}), 500

if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=8080)
