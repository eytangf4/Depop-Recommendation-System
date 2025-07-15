
from flask import Flask, request, render_template, redirect, session, g, Response, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
import threading
import time
import queue
import json
from depop_scraper import scrape_depop_items

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fallback_dev_key")
DATABASE = "users.db"
search_progress = {}

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
from depop_scraper import scrape_depop_items


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
    progress_id = str(time.time())
    q = queue.Queue()
    search_progress[progress_id] = q

    def run_search():
        # Simulate progress
        q.put({"progress": 10, "status": "Starting search..."})
        time.sleep(0.3)
        q.put({"progress": 30, "status": "Fetching Depop items..."})
        items = scrape_depop_items(query, max_items=10)
        q.put({"progress": 90, "status": "Processing results..."})
        time.sleep(0.2)
        q.put({"progress": 100, "status": "Done", "items": items})
        q.put(None)  # End signal

    threading.Thread(target=run_search).start()
    return jsonify({"progress_id": progress_id})

@app.route("/search_progress/<progress_id>")
def search_progress_stream(progress_id):
    def event_stream():
        q = search_progress.get(progress_id)
        if not q:
            yield f"data: {json.dumps({'progress': 100, 'status': 'Error', 'items': []})}\n\n"
            return
        while True:
            update = q.get()
            if update is None:
                break
            yield f"data: {json.dumps(update)}\n\n"
        del search_progress[progress_id]
    return Response(event_stream(), mimetype="text/event-stream")

@app.route("/search_results_page", methods=["POST"])
def search_results_page():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json()
    query = data.get("query", "")
    offset = int(data.get("offset", 0))
    max_items = int(data.get("max_items", 10))
    items = scrape_depop_items(query, max_items=max_items, offset=offset)
    return jsonify({"items": items})

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
