from flask import Flask, request, render_template, redirect, session, g, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
import json
from depop_api import fetch_depop_items

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
    
    # Pass arrays for multiple selections
    api_filters = {
        "gender": filters["gender"] if filters["gender"] != "men" and filters["gender"] != "women" else ("male" if filters["gender"] == "men" else "female"),
        "category": filters["category"],
        "subcategory": filters["subcategory"][0] if filters["subcategory"] else None,
        "brand": filters["brand"] if filters["brand"] else None,
        "size": filters["size"] if filters["size"] else None,  # Pass full size array
        "color": filters["color"] if filters["color"] else None,
        "condition": filters["condition"] if filters["condition"] else None,
        "price_min": filters["price_min"],
        "price_max": filters["price_max"],
        "on_sale": filters["on_sale"],
        "sort": filters["sort"]
    }
    
    # Debug logging
    print(f"DEBUG - Raw request data: {data}")
    print(f"DEBUG - Main category: {main_category}")
    print(f"DEBUG - Midlevel: {midlevel}")
    print(f"DEBUG - API Filters: {api_filters}")
    print(f"DEBUG - Color filters: {filters['color']}")
    print(f"DEBUG - Condition filters: {filters['condition']}")
    
    items, next_cursor = fetch_depop_items(query, **api_filters, max_items=24)
    return jsonify({"items": items, "next_cursor": next_cursor})

@app.route("/search_results_page", methods=["POST"])
def search_results_page():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json()
    query = data.get("query", "")
    cursor = data.get("cursor")
    
    # Handle array-based multi-select filters
    filters = {
        "gender": data.get("category"),  # category is now the main gender filter
        "category": None,  # Will be determined by subcategories
        "subcategory": data.get("subcategory", []) if isinstance(data.get("subcategory"), list) else ([data.get("subcategory")] if data.get("subcategory") else []),
        "brand": data.get("brand", []) if isinstance(data.get("brand"), list) else ([data.get("brand")] if data.get("brand") else []),
        "size": data.get("size", []) if isinstance(data.get("size"), list) else ([data.get("size")] if data.get("size") else []),
        "color": data.get("color", []) if isinstance(data.get("color"), list) else ([data.get("color")] if data.get("color") else []),
        "condition": data.get("condition", []) if isinstance(data.get("condition"), list) else ([data.get("condition")] if data.get("condition") else []),
        "price_min": data.get("price_min"),
        "price_max": data.get("price_max"),
        "on_sale": data.get("on_sale"),
        "sort": data.get("sort", "relevance")
    }
    
    # Pass arrays for multiple selections
    api_filters = {
        "gender": filters["gender"] if filters["gender"] != "men" and filters["gender"] != "women" else ("male" if filters["gender"] == "men" else "female"),
        "category": filters["category"],
        "subcategory": filters["subcategory"][0] if filters["subcategory"] else None,
        "brand": filters["brand"] if filters["brand"] else None,
        "size": filters["size"] if filters["size"] else None,  # Pass full size array
        "color": filters["color"] if filters["color"] else None,
        "condition": filters["condition"] if filters["condition"] else None,
        "price_min": filters["price_min"],
        "price_max": filters["price_max"],
        "on_sale": filters["on_sale"],
        "sort": filters["sort"]
    }
    
    # Debug logging for pagination
    print(f"DEBUG - Pagination API Filters: {api_filters}")
    
    items, next_cursor = fetch_depop_items(query, **api_filters, max_items=24, offset=cursor)
    return jsonify({"items": items, "next_cursor": next_cursor})

# --- Feedback Route (placeholder, not storing yet) ---
@app.route("/feedback", methods=["POST"])
def feedback():
    if "user_id" not in session:
        return redirect("/")
    # Here you would store feedback in the database
    # For now, just redirect back to search
    return redirect(request.referrer or "/search")

if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=8080)
