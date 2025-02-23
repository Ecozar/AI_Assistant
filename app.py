import sys
import os
# Ensure the project root (one level up) is in sys.path so that shared modules can be imported.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import sqlite3
import time
import json
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import fitz  # PyMuPDF for PDF extraction
import numpy as np
from flask import Flask, request, redirect, url_for, render_template_string, flash, get_flashed_messages, jsonify
from markupsafe import escape
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Import shared configuration and utilities.
from config import (
    DB_FILE, 
    UPLOAD_FOLDER, 
    TEXT_SETTINGS,  # Instead of ALLOWED_EXTENSIONS
    APPROVED_TAGS,
    get_ui_settings
)
from AI_Project_Brain.text_utils import generate_embedding, read_text_file, read_pdf_file, chunk_text
from AI_Project_Brain.auto_tagger import get_suggested_tags
from AI_Project_Brain.db_manager import db_manager
from AI_Project_Brain.logging_config import configure_logging
from AI_Project_Brain.retrieval_pipeline import get_conversation_context

# Replace existing logging setup
configure_logging(log_file="app.log")

app = Flask(__name__)
app.secret_key = get_ui_settings().get('flask_secret_key', 'default_secret_key')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
executor = ThreadPoolExecutor(max_workers=4)

# Optionally, initialize the embedding model here (if needed separately).
# (Note: generate_embedding() in text_utils.py already loads its own model.)
model = SentenceTransformer('all-MiniLM-L6-v2')

"""
Core application logic handling file processing, database operations, and web endpoints.

SSOT Requirements:
- Database Operations: All database interactions use context managers for automatic cleanup
- File Processing: Chunking and embedding generation are computationally expensive
- Settings Management: All values must come from get_ui_settings()
- Error Handling: Must maintain database consistency on failure

AI Implementation Notes:
- Must handle multipart/form-data uploads
- Must validate file types before processing
- Must use async processing for large files
- Must provide meaningful progress feedback
- Must maintain SSOT with UI settings for processing

Key Components:
1. File Upload Processing
2. Text Chunking
3. Auto-tagging
4. Database Management
5. Settings Management
"""

def init_db():
    """Initialize database with required tables."""
    try:
        with db_manager.get_cursor() as cursor:
            cursor.executescript('''
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL UNIQUE,
                    file_size INTEGER,
                    file_type TEXT,
                    upload_time TEXT
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS text_chunks USING FTS5(
                    file_id UNINDEXED,
                    chunk_index UNINDEXED,
                    content,
                    embedding
                );
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT
                );
                CREATE TABLE IF NOT EXISTS file_tags (
                    file_id INTEGER,
                    tag_id INTEGER,
                    tag_name TEXT,
                    PRIMARY KEY (file_id, tag_id),
                    FOREIGN KEY (file_id) REFERENCES files(id),
                    FOREIGN KEY (tag_id) REFERENCES tags(id)
                );
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT,
                    user_message TEXT,
                    assistant_message TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    tags TEXT
                );
                CREATE TABLE IF NOT EXISTS memory_summaries (
                    id INTEGER PRIMARY KEY,
                    time_period TEXT,
                    summary TEXT,
                    key_topics TEXT,
                    importance FLOAT,
                    tags TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_summaries_period 
                ON memory_summaries(time_period);
            ''')
            
            # Initialize default tags if tags table is empty
            cursor.execute("SELECT COUNT(*) FROM tags")
            if cursor.fetchone()[0] == 0:
                for tag, description in APPROVED_TAGS.items():
                    cursor.execute("INSERT INTO tags (name, description) VALUES (?, ?)", 
                                 (tag, description))
            
            # Add index for conversation_history timestamps for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversation_timestamp 
                ON conversation_history(timestamp)
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summarized_conversations (
                    conversation_id INTEGER,
                    summary_id TEXT,
                    summarized_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversation_history(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS topic_memory (
                    topic TEXT PRIMARY KEY,  -- Single primary key
                    memory_layer TEXT,  -- 'short', 'medium', or 'long'
                    first_seen DATETIME,
                    last_seen DATETIME,
                    mention_count INTEGER,
                    importance FLOAT
                )
            """)
            
        logging.info("Database initialized successfully")
    except Exception as e:
        logging.error(f"Database initialization error: {e}")
        raise

def upgrade_tags_table():
    """Upgrade the tags table schema to include a 'description' column if missing."""
    try:
        logging.info("Running upgrade_tags_table()")
        with db_manager.get_cursor() as cursor:
            # Get the current table info for 'tags'
            cursor.execute("PRAGMA table_info(tags)")
            columns_info = cursor.fetchall()
            columns = [row[1] for row in columns_info]
            
            if "description" not in columns:
                logging.info("Upgrading tags table: Adding 'description' column")
                cursor.execute("ALTER TABLE tags ADD COLUMN description TEXT")
                logging.info("Upgrade complete: 'description' column added")
            else:
                logging.info("No upgrade necessary; 'description' column already exists")
                
    except Exception as e:
        logging.error(f"Error upgrading tags table: {e}")
        raise

@app.route('/')
def upload_form():
    """Render the main upload form with flash messages."""
    messages = get_flashed_messages(with_categories=True)
    return render_template_string('''
        <h2>Upload a File</h2>
        {% for category, message in messages %}
            <div class="{{ category }}">{{ message }}</div>
        {% endfor %}
        <form action="/upload" method="POST" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <input type="submit" value="Upload">
        </form>
        <br>
        <nav>
            <a href="/files">🔍 View/Search Files</a> |
            <a href="/chunks">📜 View Chunked Data</a> |
            <a href="/logs">📋 View Logs</a> |
            <a href="/semantic_search">🔎 Semantic Search</a> |
            <a href="/purge_memory">🗑️ Purge Memory</a>
        </nav>
    ''', messages=messages)

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    AI Note: File upload endpoint requirements:
    - Must handle multipart/form-data
    - Must validate file types before processing
    - Must use async processing for large files
    - Must provide meaningful progress feedback
    - Must maintain SSOT with UI settings for processing
    """
    if 'file' not in request.files:
        flash("No file part in request", "error")
        return redirect(url_for('upload_form'))

    file = request.files['file']
    if file.filename == '':
        flash("No selected file", "error")
        return redirect(url_for('upload_form'))

    filename = secure_filename(file.filename)
    temp_path = None
    try:
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        file_size = os.path.getsize(temp_path)
        file_type = os.path.splitext(filename)[1].lower()
        upload_time = time.strftime('%Y-%m-%d %H:%M:%S')

        if file_type not in TEXT_SETTINGS['allowed_extensions']:
            os.remove(temp_path)
            flash("Unsupported file format", "error")
            return redirect(url_for('upload_form'))

        # Read content based on file type.
        content = (read_text_file(temp_path) if file_type == '.txt'
                   else read_pdf_file(temp_path) if file_type == '.pdf'
                   else None)

        if not content:
            flash("No content could be extracted from file", "error")
            return redirect(url_for('upload_form'))

        def process_upload():
            try:
                with db_manager.get_cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO files (filename, file_size, file_type, upload_time) VALUES (?,?,?,?)",
                        (filename, file_size, file_type, upload_time)
                    )
                    file_id = cursor.lastrowid

                    # Process and store text chunks with embeddings
                    chunks_with_embeddings = chunk_text(content)
                    
                    # Collect all text for tag analysis
                    full_text = " ".join(chunk for chunk, _ in chunks_with_embeddings)
                    
                    # Get suggested tags for the entire document
                    settings = get_ui_settings()
                    suggested_tags = get_suggested_tags(full_text, threshold=settings["auto_tag_threshold"])
                    
                    # Store chunks
                    for idx, (chunk, embedding) in enumerate(chunks_with_embeddings):
                        cursor.execute(
                            "INSERT INTO text_chunks (file_id, chunk_index, content, embedding) VALUES (?,?,?,?)",
                            (file_id, idx, chunk, json.dumps(embedding.tolist()))
                        )
                    
                    # Store tags if any were suggested
                    if suggested_tags:
                        for tag in suggested_tags:
                            cursor.execute(
                                """
                                INSERT INTO file_tags (file_id, tag_id, tag_name)
                                SELECT ?, id, ? FROM tags WHERE name = ?
                                """,
                                (file_id, tag, tag)
                            )
                    
                    logging.info("File %s processed successfully with %d chunks and tags: %s", 
                                filename, len(chunks_with_embeddings), suggested_tags if suggested_tags else "none")
            except Exception as e:
                logging.error("Error processing file %s: %s", filename, e)

        executor.submit(process_upload)
        flash(f"File '{filename}' uploaded and being processed", "success")
        return redirect(url_for('upload_form'))

    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        logging.error("Error handling upload for %s: %s", filename, e)
        flash(f"Error processing upload: {str(e)}", "error")
        return redirect(url_for('upload_form'))

@app.route('/semantic_search', methods=['GET', 'POST'])
def semantic_search():
    """Handle semantic search functionality."""
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if not query:
            flash("No query provided", "error")
            return redirect(url_for('semantic_search'))

        try:
            # Generate embedding for the query using our shared function.
            query_embedding = generate_embedding(query)
            
            # Retrieve all stored chunks.
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT file_id, chunk_index, content, embedding FROM text_chunks')
                chunks = cursor.fetchall()

            # Compute cosine similarities.
            results = []
            for file_id, chunk_index, content, embedding_json in chunks:
                chunk_embedding = np.array(json.loads(embedding_json))
                similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
                results.append((file_id, chunk_index, content, similarity))

            results.sort(key=lambda x: x[3], reverse=True)
            top_results = results[:10]

            return render_template_string('''
                <h2>Search Results:</h2>
                <ul>
                {% for file_id, chunk_index, content, similarity in results %}
                    <li>
                        <b>File ID: {{ file_id }}</b> [Chunk {{ chunk_index }}] 
                        (Similarity: {{ "%.2f"|format(similarity) }})<br>
                        {{ content[:200] }}...
                    </li><br>
                {% endfor %}
                </ul>
                <a href="/">Back to Home</a>
            ''', results=top_results)

        except Exception as e:
            logging.error("Search error: %s", e)
            flash("Error during search", "error")
            return redirect(url_for('semantic_search'))

    return render_template_string('''
        <h2>Semantic Search</h2>
        <form method="POST">
            <input type="text" name="query" placeholder="Enter your search query" required>
            <input type="submit" value="Search">
        </form>
        <br>
        <a href="/">Back to Home</a>
    ''')

@app.route('/logs', methods=['GET'])
def view_logs():
    """Display application logs with optional date/time filtering."""
    date_start = request.args.get('date_start', '').strip()
    time_start = request.args.get('time_start', '').strip()
    date_end = request.args.get('date_end', '').strip()
    time_end = request.args.get('time_end', '').strip()

    start_dt = end_dt = None
    try:
        if date_start:
            start_dt = datetime.strptime(
                f"{date_start} {time_start}" if time_start else date_start,
                "%Y-%m-%d %H:%M" if time_start else "%Y-%m-%d"
            )
        if date_end:
            end_dt = datetime.strptime(
                f"{date_end} {time_end}" if time_end else date_end,
                "%Y-%m-%d %H:%M" if time_end else "%Y-%m-%d"
            )
    except ValueError as e:
        logging.error("Date parsing error: %s", e)
        flash("Invalid date format", "error")
        return redirect(url_for('view_logs'))

    try:
        with open("app.log", "r") as log_file:
            log_lines = log_file.readlines()

        if start_dt or end_dt:
            filtered_lines = []
            for line in log_lines:
                try:
                    log_dt = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
                    if (not start_dt or log_dt >= start_dt) and (not end_dt or log_dt <= end_dt):
                        filtered_lines.append(line)
                except ValueError:
                    continue
            log_content = ''.join(filtered_lines)
        else:
            log_content = ''.join(log_lines[-100:])
    except Exception as e:
        logging.error("Error reading log file: %s", e)
        log_content = "Error reading log file."

    return render_template_string('''
        <h2>Application Logs</h2>
        <form method="GET">
            Start Date: <input type="date" name="date_start" value="{{ date_start }}">
            Start Time: <input type="time" name="time_start" value="{{ time_start }}">
            <br>
            End Date: <input type="date" name="date_end" value="{{ date_end }}">
            End Time: <input type="time" name="time_end" value="{{ time_end }}">
            <br><br>
            <input type="submit" value="Filter Logs">
        </form>
        <textarea id="logBox" style="width:100%; height:400px;" readonly>{{ content }}</textarea>
        <br>
        <button onclick="copyLogs()">Copy Logs</button>
        <br>
        <a href="/">Back to Home</a>
        <script>
            function copyLogs() {
                var copyText = document.getElementById("logBox");
                copyText.select();
                copyText.setSelectionRange(0, 99999);
                document.execCommand("copy");
                alert("Logs copied to clipboard!");
            }
        </script>
    ''', date_start=date_start, time_start=time_start,
       date_end=date_end, time_end=time_end, content=log_content)

@app.route('/chunks', methods=['GET'])
def view_chunks():
    """Display stored text chunks with associated file information."""
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT files.filename, text_chunks.chunk_index, text_chunks.content 
                FROM text_chunks 
                JOIN files ON files.id = text_chunks.file_id
                ORDER BY files.filename, text_chunks.chunk_index
            ''')
            chunks = cursor.fetchall()
    except Exception as e:
        logging.error("Error retrieving chunks: %s", e)
        flash("Error retrieving chunks", "error")
        return redirect(url_for('upload_form'))

    return render_template_string('''
        <h2>Stored Chunks:</h2>
        <ul>
        {% for filename, chunk_index, content in chunks %}
            <li>
                <b>{{ filename }}</b> [Chunk {{ chunk_index }}]<br>
                {{ content[:300] }}...
            </li><br>
        {% endfor %}
        </ul>
        <a href="/">Back to Home</a>
    ''', chunks=chunks)

@app.route('/files', methods=['GET'])
def search_files():
    """Search files and their chunks with optional filters."""
    query = request.args.get('query', '').strip()
    file_type = request.args.get('file_type', '').strip()
    start_date = request.args.get('start_date', '').strip()
    end_date = request.args.get('end_date', '').strip()
    tag = request.args.get('tag', '').strip()

    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            sql_query = """
                SELECT files.filename, files.file_size, files.file_type, files.upload_time,
                       text_chunks.chunk_index, text_chunks.content
                FROM files
                JOIN text_chunks ON files.id = text_chunks.file_id
                LEFT JOIN file_tags ON files.id = file_tags.file_id
                LEFT JOIN tags ON file_tags.tag_id = tags.id
                WHERE 1=1
            """
            params = []
            if query:
                sql_query += " AND text_chunks MATCH ?"
                params.append(f"content:{query}")
            if file_type:
                sql_query += " AND files.file_type = ?"
                params.append(file_type)
            if start_date:
                sql_query += " AND files.upload_time >= ?"
                params.append(start_date)
            if end_date:
                sql_query += " AND files.upload_time <= ?"
                params.append(end_date)
            if tag:
                sql_query += " AND tags.name = ?"
                params.append(tag)
            cursor.execute(sql_query, params)
            results = cursor.fetchall()

        return render_template_string('''
            <h2>Search Files</h2>
            <form method="GET">
                Search Content: <input type="text" name="query" value="{{ query }}">
                <br>File Type: <input type="text" name="file_type" placeholder=".txt or .pdf" value="{{ file_type }}">
                <br>From Date: <input type="date" name="start_date" value="{{ start_date }}">
                <br>To Date: <input type="date" name="end_date" value="{{ end_date }}">
                <br>Tag: <input type="text" name="tag" placeholder="Tag" value="{{ tag }}">
                <br><br>
                <input type="submit" value="Search">
            </form>
            <h2>Results:</h2>
            <ul>
            {% for filename, size, type, upload_time, chunk_index, content in results %}
                <li>
                    <b>{{ filename }}</b> [Chunk {{ chunk_index }}]
                    ({{ "%.2f"|format(size/1024) }} KB, {{ type }}) | Uploaded: {{ upload_time }}<br>
                    Preview: {{ content[:200] }}...
                </li><br>
            {% endfor %}
            </ul>
            <a href="/">Back to Home</a>
        ''', query=query, file_type=file_type, start_date=start_date, 
             end_date=end_date, tag=tag, results=results)

    except Exception as e:
        logging.error("Error searching files: %s", e)
        flash("Error searching files", "error")
        return redirect(url_for('upload_form'))

@app.route('/purge_memory', methods=['GET', 'POST'])
def purge_memory():
    """Purge all data from the database and remove uploaded files."""
    if request.method == 'POST':
        confirmation = request.form.get('confirmation', '').strip().lower()
        if confirmation == 'yes':
            try:
                logging.info("Starting memory purge operation")
                
                # Drop all tables
                with db_manager.get_cursor() as cursor:
                    logging.debug("Dropping database tables")
                    cursor.execute('DROP TABLE IF EXISTS text_chunks')
                    cursor.execute('DROP TABLE IF EXISTS file_tags')
                    cursor.execute('DROP TABLE IF EXISTS tags')
                    cursor.execute('DROP TABLE IF EXISTS files')
                    cursor.execute('DROP TABLE IF EXISTS conversation_history')
                    cursor.execute('DROP TABLE IF EXISTS auto_tag_log')
                
                logging.debug("Reinitializing database")
                init_db()

                # Remove uploaded files
                logging.debug(f"Cleaning upload folder: {UPLOAD_FOLDER}")
                files_removed = 0
                for filename in os.listdir(UPLOAD_FOLDER):
                    file_path = os.path.join(UPLOAD_FOLDER, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        files_removed += 1
                        logging.debug(f"Removed file: {filename}")

                logging.info(f"Memory purge completed. Removed {files_removed} files")
                flash("Memory purged successfully", "success")
                
            except Exception as e:
                logging.error(f"Error during memory purge: {e}", exc_info=True)
                flash("Error purging memory", "error")
                
        else:
            logging.info("Memory purge cancelled by user")
            flash("Memory purge cancelled", "info")
            
        return redirect(url_for('upload_form'))

    return render_template_string('''
        <h2>Purge Memory</h2>
        <p>Are you sure you want to purge all memory? This action cannot be undone.</p>
        <form method="POST">
            <label for="confirmation">Type "yes" to confirm:</label>
            <input type="text" name="confirmation" id="confirmation" required>
            <input type="submit" value="Purge">
        </form>
        <br>
        <a href="/">Cancel</a>
    ''')

def process_upload_file(file_path):
    """
    AI Note: This function handles file processing with progress tracking and SSOT compliance.
    - Must maintain atomicity: All database operations in single transaction
    - Must handle large files with progress tracking
    - Must preserve file content: UTF-8 encoding with fallback options
    - Must integrate with auto-tagging using UI settings threshold
    """
    try:
        total_steps = 4  # Total processing steps
        current_step = 0
        
        def update_progress(message):
            nonlocal current_step
            current_step += 1
            progress = (current_step / total_steps) * 100
            logging.info(f"Progress {progress:.0f}%: {message}")
        
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        file_type = os.path.splitext(filename)[1].lower()
        upload_time = time.strftime('%Y-%m-%d %H:%M:%S')
        
        update_progress("Reading file")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        update_progress("Processing chunks")
        chunks_with_embeddings = chunk_text(content)
        
        update_progress("Generating tags")
        settings = get_ui_settings()
        full_text = " ".join(chunk for chunk, _ in chunks_with_embeddings)
        suggested_tags = get_suggested_tags(full_text, threshold=settings["auto_tag_threshold"])
        
        update_progress("Saving to database")
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO files (filename, file_size, file_type, upload_time) VALUES (?,?,?,?)",
                (filename, file_size, file_type, upload_time)
            )
            file_id = cursor.lastrowid
            
            for idx, (chunk, embedding) in enumerate(chunks_with_embeddings):
                cursor.execute(
                    "INSERT INTO text_chunks (file_id, chunk_index, content, embedding) VALUES (?,?,?,?)",
                    (file_id, idx, chunk, json.dumps(embedding.tolist()))
                )
            
            if suggested_tags:
                for tag in suggested_tags:
                    cursor.execute(
                        """
                        INSERT INTO file_tags (file_id, tag_id, tag_name)
                        SELECT ?, id, ? FROM tags WHERE name = ?
                        """,
                        (file_id, tag, tag)
                    )
            
            logging.info("File %s processed successfully with %d chunks and tags: %s", 
                        filename, len(chunks_with_embeddings), suggested_tags if suggested_tags else "none")
            
    except Exception as e:
        logging.error("Error processing file %s: %s", filename, str(e))
        raise

@app.route('/debug/db_status')
def debug_db_status():
    """Debug endpoint to check database status"""
    if not app.debug:
        return "Debug mode not enabled", 403
        
    try:
        status = {
            'tables': {},
            'file_system': {
                'upload_folder': UPLOAD_FOLDER,
                'files': os.listdir(UPLOAD_FOLDER)
            }
        }
        
        # Get table statistics
        tables = ['files', 'text_chunks', 'tags', 'file_tags', 
                 'conversation_history', 'auto_tag_log']
                 
        for table in tables:
            status['tables'][table] = {
                'row_count': db_manager.get_table_count(table),
                'structure': db_manager.get_table_info(table)
            }
            
        return jsonify(status)
        
    except Exception as e:
        logging.error(f"Error in debug endpoint: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    upgrade_tags_table()
    app.run(debug=True)
