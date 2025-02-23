"""
DESKTOP USER INTERFACE
---------------------
Main UI component for the AI Assistant Brain Project.

System Architecture:
1. UI Layer (this module)
   - Provides user interface using Tkinter
   - Manages all user interactions
   - Delegates to other components

2. Memory Layer
   - db_manager.py: Central database management
   - conversation_logger.py: Conversation persistence
   - auto_tagging.py: Automatic content categorization

3. Processing Layer
   - text_utils.py: Text processing and embeddings
   - model_manager.py: ML model management
   - dummy_retrieval.py: Simulated LLM (to be replaced)

Key Workflows:
1. Chat Flow:
   User Input -> Text Processing -> Context Retrieval -> 
   Response Generation -> Persistence -> UI Update

2. Memory Management:
   Record Selection -> Tag Management -> Bulk Operations ->
   Database Updates -> UI Refresh

3. Document Processing:
   File Upload -> Text Extraction -> Chunking ->
   Auto-tagging -> Storage -> Preview

Technical Dependencies:
- SQLite for persistent storage
- sentence-transformers for embeddings
- Tkinter for UI components
- Flask for file upload handling

Future Integration Points:
- LLM integration will replace dummy_retrieval.py
- Vector database may replace current embedding storage
- Additional file type support in document processing

State Management:
1. Database State
   - SQLite maintains persistent state
   - db_manager.py provides thread-safe access
   - All components use same connection pool

2. UI State
   - Tkinter widgets maintain temporary state
   - Config file stores persistent UI preferences
   - Open windows tracked in self.open_windows

3. Memory State
   - Embeddings cached in model_manager
   - Conversation history in SQLite
   - Tags system maintains approved tags list

4. Thread Safety:
   - UI runs in main thread
   - Flask server runs in daemon thread
   - Database uses connection per thread
   - Model manager ensures thread-safe model access

Error Handling Strategy:
1. UI Layer:
   - Shows user-friendly error messages
   - Logs detailed errors for debugging
   - Maintains UI responsiveness during errors

2. Database Layer:
   - Uses transactions for atomicity
   - Rolls back on failure
   - Maintains connection pool health

3. Processing Layer:
   - Graceful fallback for model errors
   - Caches for offline operation
   - Validates inputs before processing

Integration Notes:
1. Current Simulation:
   - dummy_retrieval.py simulates LLM
   - Maintains same interface for future LLM
   - Uses real embeddings/similarity

2. Future LLM Integration:
   - Will replace dummy_retrieval.py
   - Must maintain same response format
   - Must handle same context structure

3. Vector DB Integration:
   - May replace current embedding storage
   - Must maintain similarity search API
   - Must support same chunking strategy
"""

import sys
import os
import logging
logger = logging.getLogger(__name__)
# Ensure the project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog, ttk
import sqlite3
import json
from pathlib import Path
from datetime import datetime

from AI_Project_Brain.conversation_logger import update_interaction, update_record_tags, log_interaction
from AI_Project_Brain.auto_tagger import get_suggested_tags, load_approved_tags, auto_tagger, add_approved_tag, remove_approved_tag
from AI_Project_Brain.advanced_prompt_builder import build_prompt
from AI_Project_inference.inference import generate_response
from AI_Project_Brain.db_manager import db_manager
from config import (
    PROJECT_ROOT,
    UI_SETTINGS,
    APPROVED_TAGS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TOKEN_BUDGETS,
    DB_SETTINGS,
    BACKUP_SETTINGS,
    MEMORY_SETTINGS
)
from AI_Project_Brain.backup_manager import backup_manager
from AI_Project_Brain.memory_manager import memory_manager
from AI_Project_Brain.retrieval_pipeline import get_conversation_context

# Define the path for the config file relative to the project root
CONFIG_FILE = os.path.join(project_root, "config", "ui_settings.json")

class StandardWindow(tk.Toplevel):
    """
    Base window class with consistent styling.
    
    Provides:
    - Standard dark theme
    - Consistent button styling
    - Common window behaviors
    """
    
    def __init__(self, master, title="Option Window", **kwargs):
        super().__init__(master, **kwargs)
        self.title(title)
        self.master = master

        # Get colors from master window (or fallback to defaults)
        self.colors = getattr(master, 'colors', {
            'bg': '#2b2b2b',
            'fg': '#ffffff',
            'button': '#404040',
            'entry': '#333333',
            'text': '#333333'
        })
        self.configure(bg=self.colors['bg'])

        # Append this window to the master's open_windows list.
        self.master.open_windows.append(self)

        # Make this window transient and give it focus
        self.transient(master)
        self.focus_set()

        # Set a standard minimum size
        self.minsize(300, 400)

    def on_close(self):
        if self in self.master.open_windows:
            self.master.open_windows.remove(self)
        self.destroy()
        if self.master.open_windows:
            last_window = self.master.open_windows[-1]
            last_window.lift()
            last_window.focus_set()

    def create_button_container(self):
        """Create standardized button container"""
        container = tk.Frame(self, bg=self.colors['bg'])
        container.pack(fill="x", pady=10, padx=10)
        return container

class BackupWindow(StandardWindow):
    """Window for managing database backups"""
    
    def __init__(self, master):
        super().__init__(master, title="Database Backup Management")
        
        # Create main container
        main_frame = tk.Frame(self, bg=self.colors['bg'])
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Status section
        status_frame = tk.LabelFrame(main_frame, text="Backup Status", 
                                   bg=self.colors['bg'], fg=self.colors['fg'])
        status_frame.pack(fill="x", pady=5)
        
        self.status_label = tk.Label(status_frame, 
                                   bg=self.colors['bg'], 
                                   fg=self.colors['fg'])
        self.status_label.pack(pady=5)
        self.update_status()
        
        # Action buttons
        btn_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        btn_frame.pack(fill="x", pady=10)
        
        tk.Button(
            btn_frame,
            text="Create Backup",
            command=self.create_backup,
            bg=self.colors['button'],
            fg=self.colors['fg']
        ).pack(side="left", padx=5)
        
        tk.Button(
            btn_frame,
            text="Restore Backup",
            command=self.restore_backup,
            bg=self.colors['button'],
            fg=self.colors['fg']
        ).pack(side="left", padx=5)
        
        # Backup list
        list_frame = tk.LabelFrame(main_frame, text="Available Backups",
                                 bg=self.colors['bg'], fg=self.colors['fg'])
        list_frame.pack(fill="both", expand=True, pady=5)
        
        self.backup_list = tk.Listbox(
            list_frame,
            bg=self.colors['entry'],
            fg=self.colors['fg'],
            selectmode="single"
        )
        self.backup_list.pack(fill="both", expand=True, padx=5, pady=5)
        self.refresh_backup_list()
        
    def update_status(self):
        """Update backup status information"""
        try:
            status = backup_manager.get_backup_status()
            
            status_text = (
                f"Last Backup: {status['last_backup'].strftime('%Y-%m-%d %H:%M:%S') if status['last_backup'] else 'Never'}\n"
                f"Next Backup: {status['next_backup'].strftime('%Y-%m-%d %H:%M:%S') if status['next_backup'] else 'Not Scheduled'}\n"
                f"Total Backups: {status['backup_count']}\n"
                f"Auto-backup: {'Enabled' if BACKUP_SETTINGS['storage']['backup_on_shutdown'] else 'Disabled'}"
            )
            self.status_label.config(text=status_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update status: {str(e)}")
    
    def refresh_backup_list(self):
        """Refresh the list of available backups"""
        self.backup_list.delete(0, tk.END)
        try:
            backup_dir = Path(db_manager.db_path).parent / BACKUP_SETTINGS['storage']['backup_dir']
            if backup_dir.exists():
                backups = sorted(
                    backup_dir.glob("*.sqlite*"),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )
                for backup in backups:
                    size = backup.stat().st_size / 1024  # KB
                    mtime = datetime.fromtimestamp(backup.stat().st_mtime)
                    self.backup_list.insert(
                        tk.END,
                        f"{backup.name} ({size:.1f} KB) - {mtime.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh backup list: {str(e)}")
    
    def create_backup(self):
        """Create a new database backup"""
        try:
            backup_manager._perform_backup()  # Direct backup without waiting for schedule
            self.update_status()
            self.refresh_backup_list()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create backup: {str(e)}")
    
    def restore_backup(self):
        """Restore database from selected backup"""
        selection = self.backup_list.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a backup to restore")
            return
            
        backup_name = self.backup_list.get(selection[0]).split(" (")[0]
        backup_path = Path(db_manager.db_path).parent / BACKUP_SETTINGS['storage']['backup_dir'] / backup_name
        
        if messagebox.askyesno("Confirm Restore", 
                             "This will replace the current database. Continue?"):
            try:
                # Verify backup before restore
                verification = backup_manager.verify_backup(backup_path)
                if not verification['is_valid']:
                    messagebox.showerror("Error", f"Backup verification failed: {verification.get('error', 'Unknown error')}")
                    return
                    
                if db_manager.restore_from_backup(backup_path):
                    messagebox.showinfo("Success", "Database restored successfully")
                else:
                    messagebox.showerror("Error", "Restore failed, safety backup used")
                self.update_status()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to restore backup: {str(e)}")

class AIAssistantUI:
    """
    Main UI class for the AI Assistant.
    
    Core Responsibilities:
    1. Chat interface management
    2. Memory viewing and editing
    3. Tag management
    4. Settings configuration
    
    Design Notes:
    - Uses dark theme for reduced eye strain
    - Implements responsive layout
    - Provides intuitive navigation
    - Handles errors gracefully
    """
    
    def __init__(self):
        """
        Initialize the UI.
        
        Note:
            Sets up all UI components
            Initializes database connection
            Loads configuration
        """
        self.root = tk.Tk()
        self.root.title("AI Assistant UI")
        self.root.geometry(UI_SETTINGS['window_size'])
        
        self.open_windows = []
        self.root.open_windows = self.open_windows
        
        self.config = self.load_config()
        
        # Load approved tags from database
        self.APPROVED_TAGS = load_approved_tags()
        
        # Set colors from UI_SETTINGS
        self.colors = UI_SETTINGS['colors']
        
        self.root.configure(bg=self.colors['bg'])
        self.style = ttk.Style()
        self.style.configure('Dark.TButton', background=self.colors['button'])
        self.style.configure('Dark.TEntry', fieldbackground=self.colors['entry'], 
                            foreground=self.colors['fg'])
        
        # Initialize auto tagger and approved tags
        self.auto_tagger = auto_tagger
        self.approved_tags = set()
        self.refresh_approved_tags()
        
        self.memory_manager = memory_manager
        self.message_widgets = {}  # Track message display widgets
        
        # Create main window components
        self.create_widgets()
        
        # Configure logging
        self.configure_logging()

    def load_config(self):
        """Load or create default configuration"""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logging.error(f"Error loading config: {e}")
        
        # All defaults in one place
        return {
            "top_n": 2,                    # Number of knowledge chunks to retrieve
            "conversation_limit": 5,        # Max conversation turns to include
            "min_similarity": 0.3,         # Minimum similarity threshold for context
            "auto_tag_threshold": 0.2,     # Threshold for auto-tagging
            "system_prompt": "...",        # Default system prompt
            "flask_port": 5000,            # Flask server port
            "flask_debug": False,          # Flask debug mode
            "flask_secret_key": "default_secret_key"  # Flask secret key
        }

    def save_config(self):
        try:
            os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=4)
            logging.info("Configuration saved successfully")
        except Exception as e:
            logging.error(f"Error saving config: {e}")

    def create_widgets(self):
        main_container = tk.Frame(self.root, bg=self.colors['bg'])
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Output area with correct colors
        self.output_area = scrolledtext.ScrolledText(
            main_container,
            width=60, 
            height=20,
            bg=self.colors['bg'],  # Changed from 'text' to 'bg'
            fg=self.colors['fg'],
            wrap=tk.WORD,
            state='disabled'
        )
        self.output_area.pack(fill="both", expand=True, pady=(0, 10))
        
        # Message input with correct colors and size
        self.message_frame = tk.Frame(main_container, bg=self.colors['bg'])
        self.message_frame.pack(fill="x", padx=10, pady=5)
        
        self.message_input = tk.Text(
            self.message_frame, 
            height=2,  # Reduced from 3 to 2
            bg=self.colors['entry'],
            fg=self.colors['fg'],
            insertbackground=self.colors['fg']  # Added cursor color
        )
        self.message_input.pack(side="left", fill="x", expand=True)
        
        self.send_button = ttk.Button(
            self.message_frame,
            text="Send",
            command=self.handle_send
        )
        self.send_button.pack(side="right", padx=5)
        
        button_container = tk.Frame(main_container, bg=self.colors['bg'])
        button_container.pack(fill="x", pady=10)
        
        buttons = [
            ("Manage Memory", self.open_memory_management),
            ("Upload Document", self.open_document_upload),
            ("Settings", self.open_settings),
            ("View Conversation History", self.open_conversation_viewer),
            ("Backup Manager", self.show_backup_manager)
        ]
        
        for text, command in buttons:
            btn = tk.Button(
                button_container,
                text=text,
                command=command,
                bg=self.colors['button'],
                fg=self.colors['fg']
            )
            btn.pack(side="left", padx=5)

    def handle_send(self):
        """Handle send button click"""
        message = self.message_input.get("1.0", "end-1c").strip()
        if not message:
            return
            
        try:
            # Process message and get memory ID
            memory_id = self.process_message(message)
            if memory_id:
                # Generate response using inference module
                response = generate_response(message)
                
                # Log the interaction
                log_interaction(self.config.get("conversation_id", "default"), 
                              message, response)
                
                # Display both message and response
                self.display_message(message, memory_id)
                self.display_response(response)
                
                # Clear input
                self.message_input.delete("1.0", "end")
            else:
                messagebox.showerror("Error", "Failed to process message")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def submit_query(self):
        query = self.query_entry.get().strip()
        if not query:
            return
        
        # Generate the response using the inference module.
        response = generate_response(query)
        
        # Log the interaction into the conversation_history table.
        log_interaction(self.config["conversation_id"], query, response)
        
        # Enable temporarily to update text
        self.output_area.config(state='normal')
        self.output_area.delete(1.0, tk.END)
        self.output_area.insert(tk.END, response)
        self.output_area.config(state='disabled')  # Make read-only again
        
        self.query_entry.delete(0, tk.END)

    def open_edit_window(self, record, refresh_callback):
        """
        Open window for editing conversation records.
        
        Args:
            record: Record to edit
            refresh_callback: Function to call after edit
            
        Note:
            Handles both messages and tags
            Provides auto-tagging support
        """
        edit_win = StandardWindow(self.root, title="Edit Record")
        edit_win.geometry("450x500")
        edit_win.minsize(450, 500)
        
        # Extract record data
        if len(record) == 4:
            record_id, timestamp, user_msg, assistant_msg = record
            current_tags = ""
        else:
            record_id, timestamp, user_msg, assistant_msg, current_tags = record
            if current_tags is None or current_tags.lower() == "none":
                current_tags = ""
        
        # Create edit fields
        tk.Label(edit_win, text=f"Editing Record ID: {record_id}", 
                bg=self.colors['bg'], fg=self.colors['fg']).pack(pady=5)
                
        # Message editing
        text_widgets = {}  # Store text widgets in a dictionary
        for label, text in [("User Message:", user_msg), 
                          ("Assistant Message:", assistant_msg)]:
            tk.Label(edit_win, text=label, bg=self.colors['bg'], 
                    fg=self.colors['fg']).pack(anchor="w", padx=10)
            text_widget = tk.Text(edit_win, height=4, width=40, wrap=tk.WORD)
            text_widget.pack(padx=10, fill='x', expand=True)
            text_widget.insert(tk.END, text)
            key = label.split(':')[0].lower().replace(' ', '')
            text_widgets[key] = text_widget
            print(f"Created text widget with key: {key}")  # Debug print
        
        # Tag management
        tk.Label(edit_win, text="Select a Tag:", bg=self.colors['bg'], 
                fg=self.colors['fg']).pack(anchor="w", padx=10)
        self.refresh_tags()  # Refresh tags from database
        approved_tags = list(self.APPROVED_TAGS.keys())  # Get current approved tags
        tag_combo = ttk.Combobox(edit_win, values=approved_tags, state="readonly")
        tag_combo.pack(padx=10, pady=5)
        if approved_tags:
            tag_combo.set(approved_tags[0])
        
        # Current tags display
        tk.Label(edit_win, text="Current Tags:", bg=self.colors['bg'], 
                fg=self.colors['fg']).pack(anchor="w", padx=10)
        current_tags_var = tk.StringVar(value=current_tags)
        tags_label = tk.Label(edit_win, textvariable=current_tags_var, 
                            bg=self.colors['bg'], fg=self.colors['fg'])
        tags_label.pack(padx=10, pady=5)
        
        # Tag management functions
        def add_tag():
            tag = tag_combo.get()
            tags = [t.strip() for t in current_tags_var.get().split(",") 
                   if t.strip()] if current_tags_var.get() else []
            if tag not in tags:
                tags.append(tag)
                current_tags_var.set(", ".join(tags))
                self.refresh_approved_tags()
            else:
                messagebox.showinfo("Info", f"Tag '{tag}' is already added.")

        def remove_tag():
            tag = tag_combo.get()
            tags = [t.strip() for t in current_tags_var.get().split(",") 
                   if t.strip()] if current_tags_var.get() else []
            if tag in tags:
                tags.remove(tag)
                current_tags_var.set(", ".join(tags))
                self.refresh_approved_tags()
            else:
                messagebox.showinfo("Info", f"Tag '{tag}' is not present.")

        def auto_tag():
            """Auto-tag function for edit window"""
            try:
                text = ""
                for widget_key in ['usermessage', 'assistantmessage']:
                    if widget_key in text_widgets:
                        text += text_widgets[widget_key].get("1.0", tk.END).strip() + " "
                
                if not text.strip():
                    messagebox.showinfo("Auto-Tagging", "No text found for tagging.")
                    return

                # Use same threshold as main UI
                suggestions = auto_tagger.get_suggested_tags(
                    text,
                    threshold=float(self.config.get('auto_tag_threshold', 0.157))  # Use UI settings
                )
                
                valid_suggestions = [
                    (tag, conf) for tag, conf in suggestions 
                    if tag in self.approved_tags
                ]
                
                if not valid_suggestions:
                    messagebox.showinfo("Auto-Tagging", "No relevant tags found.")
                    return
                    
                msg = "Suggested tags:\n" + "\n".join(
                    f"- {tag} ({conf:.2f})" for tag, conf in valid_suggestions
                )
                
                if messagebox.askyesno("Auto-Tagging", f"{msg}\n\nAdd these tags?"):
                    current = [t.strip() for t in current_tags_var.get().split(",") 
                              if t.strip()] if current_tags_var.get() else []
                    new_tags = list(set(current + [tag for tag, _ in valid_suggestions]))
                    current_tags_var.set(", ".join(new_tags))
                    
            except Exception as e:
                logger.error(f"Error in auto_tag: {e}", exc_info=True)
                messagebox.showerror("Auto-Tagging Error", str(e))

        def save_changes():
            print(f"Available text widget keys: {list(text_widgets.keys())}")  # Debug print
            new_user_msg = text_widgets['usermessage'].get("1.0", tk.END).strip()
            new_assistant_msg = text_widgets['assistantmessage'].get("1.0", tk.END).strip()
            new_tags = current_tags_var.get().strip()
            
            try:
                update_interaction(record_id, new_user_msg, new_assistant_msg, new_tags)
                messagebox.showinfo("Success", "Record updated successfully.")
                edit_win.destroy()
                refresh_callback()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to update record: {str(e)}")
        
        # Button container
        button_frame = tk.Frame(edit_win, bg=self.colors['bg'])
        button_frame.pack(pady=10)
        
        # Add buttons
        for text, command in [("Add Tag", add_tag),
                             ("Remove Tag", remove_tag),
                             ("Auto-Tag", auto_tag),
                             ("Save", save_changes)]:
            tk.Button(button_frame, text=text, command=command,
                     bg=self.colors['button'], fg=self.colors['fg']).pack(side="left", padx=5)

    def open_memory_management(self):
        """
        Open the memory management window.
        
        This window provides:
        1. View of all conversation history
        2. Record editing capabilities
        3. Bulk tag management
        4. Real-time updates
        
        Note:
            Uses TreeView for efficient display
            Supports both single and bulk operations
            Maintains consistent styling
        """
        mem_window = StandardWindow(self.root, title="Memory Management")
        mem_window.geometry("1400x500")
        
        self.refresh_tags()
        
        main_container = tk.Frame(mem_window, bg=self.colors['bg'])
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Set up TreeView with columns
        columns = ("id", "timestamp", "user_message", "assistant_message", "tags")
        tree, scrollbar = self.create_scrolled_treeview(
            main_container,
            columns,
            column_widths=[50, 150, 400, 400, 200]  # Adjusted widths
        )
        
        # Configure message columns for better display
        for col in ["user_message", "assistant_message"]:
            tree.column(col, width=400, stretch=True)
            tree.heading(col, text=col.replace("_", " ").title())
        
        # Pack the TreeView and scrollbar
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        def refresh_memory():
            """Refresh the conversation display"""
            tree.delete(*tree.get_children())
            try:
                with db_manager.get_cursor() as cursor:
                    cursor.execute("""
                        SELECT id, 
                               datetime(timestamp, 'localtime') as timestamp,
                               substr(user_message, 1, 100) as user_message,
                               substr(assistant_message, 1, 100) as assistant_message,
                               tags
                        FROM conversation_history
                        ORDER BY timestamp DESC
                    """)
                    for record in cursor.fetchall():
                        # Format the record for display
                        formatted_record = list(record)
                        # Add ellipsis for truncated messages
                        formatted_record[2] = formatted_record[2] + "..." if len(record[2]) > 100 else formatted_record[2]
                        formatted_record[3] = formatted_record[3] + "..." if len(record[3]) > 100 else formatted_record[3]
                        tree.insert("", tk.END, values=formatted_record)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load conversation history: {str(e)}")
        
        def delete_selected():
            """Delete selected conversation records"""
            selection = tree.selection()
            if not selection:
                messagebox.showinfo("Info", "Please select records to delete")
                return
            
            # Confirm deletion
            count = len(selection)
            confirm = messagebox.askyesno(
                "Confirm Delete",
                f"Are you sure you want to delete {count} selected record{'s' if count > 1 else ''}?"
            )
            if not confirm:
                return
            
            try:
                with db_manager.get_cursor() as cursor:
                    for item in selection:
                        record_id = tree.item(item)['values'][0]  # Get ID from first column
                        cursor.execute("DELETE FROM conversation_history WHERE id = ?", (record_id,))
            
                messagebox.showinfo("Success", f"Deleted {count} record{'s' if count > 1 else ''}")
                refresh_memory()  # Refresh the display
            
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete records: {str(e)}")

        # Button container
        button_container = mem_window.create_button_container()
        
        # Add operation buttons
        operations = [
            ("Edit Selected", lambda: self.edit_selected(tree, refresh_memory)),
            ("Bulk Edit Tags", lambda: self.bulk_edit_selected(tree, refresh_memory)),
            ("Delete Selected", delete_selected),  # Add delete button
            ("Refresh", refresh_memory)
        ]
        
        for text, command in operations:
            tk.Button(
                button_container,
                text=text,
                command=command,
                bg=self.colors['button'],
                fg=self.colors['fg']
            ).pack(side="left", padx=5)
        
        # Initial data load
        refresh_memory()

    def open_document_upload(self):
        file_path = filedialog.askopenfilename(
            title="Select a Document",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
        )
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                extracted_text = f.read()
            
            # Create a temporary file in the uploads directory
            upload_dir = os.path.join(project_root, "AI_Project_Brain", "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            temp_filename = os.path.basename(file_path)
            temp_path = os.path.join(upload_dir, temp_filename)
            
            # Copy file to uploads directory
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(extracted_text)
            
            # Process the file using app.py's functionality
            from AI_Project_Brain.app import process_upload_file
            process_upload_file(temp_path)
            
            # Show preview
            doc_window = StandardWindow(self.root, title="Document Content")
            doc_window.geometry("600x400")
            text_area = scrolledtext.ScrolledText(doc_window, width=70, height=20, wrap=tk.WORD)
            text_area.pack(pady=10, fill='both', expand=True)
            text_area.insert(tk.END, extracted_text)
            messagebox.showinfo("Upload Complete", "Document processed and stored in database.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process the file: {e}")

    def open_settings(self):
        settings_win = StandardWindow(self.root, title="Settings")
        settings_win.geometry("402x694")  # Made taller to accommodate new field
        settings_win.minsize(402, 694)
        
        # Create scrollable canvas for content
        canvas = tk.Canvas(settings_win, bg=self.colors['bg'])
        scrollbar = ttk.Scrollbar(settings_win, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg'])
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=380)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        tk.Label(scrollable_frame, text="Settings", font=("Arial", 16), bg=self.colors['bg'], fg=self.colors['fg']).pack(pady=10)
        
        # Existing settings fields
        tk.Label(scrollable_frame, text="Number of Knowledge Chunks (top_n):", bg=self.colors['bg'], fg=self.colors['fg']).pack(anchor="w", padx=10)
        top_n_var = tk.IntVar(value=self.config["top_n"])
        tk.Entry(scrollable_frame, textvariable=top_n_var).pack(padx=10, pady=5)
        
        tk.Label(scrollable_frame, text="Conversation History Limit:", bg=self.colors['bg'], fg=self.colors['fg']).pack(anchor="w", padx=10)
        conv_limit_var = tk.IntVar(value=self.config["conversation_limit"])
        tk.Entry(scrollable_frame, textvariable=conv_limit_var).pack(padx=10, pady=5)
        
        tk.Label(scrollable_frame, text="Auto-Tag Threshold (0-1):", bg=self.colors['bg'], fg=self.colors['fg']).pack(anchor="w", padx=10)
        auto_tag_thresh_var = tk.DoubleVar(value=self.config["auto_tag_threshold"])
        tk.Entry(scrollable_frame, textvariable=auto_tag_thresh_var).pack(padx=10, pady=5)
        
        # Add System Prompt field
        tk.Label(scrollable_frame, text="System Prompt:", bg=self.colors['bg'], fg=self.colors['fg']).pack(anchor="w", padx=10)
        system_prompt_text = tk.Text(scrollable_frame, height=5, width=40, wrap=tk.WORD,
                                    bg=self.colors['entry'], fg=self.colors['fg'])
        system_prompt_text.pack(padx=10, pady=5, fill='x', expand=True)
        system_prompt_text.insert(tk.END, self.config.get("system_prompt", ""))
        
        # Manual Tag Management Section:
        tk.Label(scrollable_frame, text="Approved Tags:", bg=self.colors['bg'], fg=self.colors['fg']).pack(anchor="w", padx=10)
        self.refresh_tags()  # Add this line before populating the listbox
        tags_listbox = tk.Listbox(scrollable_frame, selectmode=tk.SINGLE, height=6)
        for tag in self.APPROVED_TAGS.keys():
            tags_listbox.insert(tk.END, tag)
        tags_listbox.pack(padx=10, pady=5, fill="x")
        
        tk.Label(scrollable_frame, text="New Tag:", bg=self.colors['bg'], fg=self.colors['fg']).pack(anchor="w", padx=10)
        new_tag_var = tk.StringVar()
        tk.Entry(scrollable_frame, textvariable=new_tag_var).pack(padx=10, pady=5)
        
        tk.Label(scrollable_frame, text="Tag Description:", bg=self.colors['bg'], fg=self.colors['fg']).pack(anchor="w", padx=10)
        new_tag_desc_var = tk.StringVar()
        tk.Entry(scrollable_frame, textvariable=new_tag_desc_var).pack(padx=10, pady=5)
        
        def add_new_tag():
            new_tag = new_tag_var.get().strip()
            new_tag_desc = new_tag_desc_var.get().strip()
            if new_tag and new_tag not in self.APPROVED_TAGS:
                add_approved_tag(new_tag, new_tag_desc)  # Save to database
                self.refresh_tags()  # Refresh from database
                tags_listbox.insert(tk.END, new_tag)
                messagebox.showinfo("Success", f"Tag '{new_tag}' added with description '{new_tag_desc}'.")
                new_tag_var.set("")
                new_tag_desc_var.set("")
            else:
                messagebox.showwarning("Warning", "Invalid or duplicate tag.")
        
        def remove_selected_tag():
            selection = tags_listbox.curselection()
            if selection:
                tag_to_remove = tags_listbox.get(selection[0])
                remove_approved_tag(tag_to_remove)  # Remove from database
                self.refresh_tags()  # Refresh from database
                tags_listbox.delete(selection[0])
                messagebox.showinfo("Success", f"Tag '{tag_to_remove}' removed.")
            else:
                messagebox.showwarning("Warning", "Please select a tag to remove.")
        
        tk.Button(scrollable_frame, text="Add Tag", command=add_new_tag, bg=self.colors['button'], fg=self.colors['fg']).pack(pady=5)
        tk.Button(scrollable_frame, text="Remove Selected Tag", command=remove_selected_tag, bg=self.colors['button'], fg=self.colors['fg']).pack(pady=5)
        
        tk.Button(scrollable_frame, text="Review Untagged Entries", command=self.review_untagged, bg=self.colors['button'], fg=self.colors['fg']).pack(pady=10)
        
        def save_settings():
            try:
                self.config["top_n"] = int(top_n_var.get())
                self.config["conversation_limit"] = int(conv_limit_var.get())
                self.config["auto_tag_threshold"] = float(auto_tag_thresh_var.get())
                self.config["system_prompt"] = system_prompt_text.get("1.0", tk.END).strip()
                self.save_config()
                messagebox.showinfo("Success", "Settings updated successfully.")
                settings_win.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numeric values.")
        
        tk.Button(scrollable_frame, text="Save and Exit", command=save_settings, bg=self.colors['button'], fg=self.colors['fg']).pack(pady=10)

    def get_current_tags(self, record_id):
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT tags FROM conversation_history WHERE id = ?", (record_id,))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result and result[0] is not None else ""
        except Exception as e:
            logging.error(f"Error getting tags: {e}")
            raise

    def get_db_connection(self):
        db_path = os.path.join(project_root, "AI_Project_Brain", "files.db")
        return sqlite3.connect(db_path)

    def execute_db_query(self, query, params=None):
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            result = cursor.fetchall()
            conn.commit()
            return result
        except Exception as e:
            logging.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def run(self):
        """Start the UI main loop"""
        try:
            # Bind Return key to send
            self.message_input.bind('<Return>', lambda e: self.handle_send())
            
            # Start main loop
            self.root.mainloop()
            
        except Exception as e:
            logger.error(f"Error in UI main loop: {e}", exc_info=True)
            raise

    def create_scrolled_treeview(self, parent, columns, column_widths=None, **kwargs):
        """
        Create a styled TreeView with scrollbar.
        
        Args:
            parent: Parent widget
            columns: Column definitions
            column_widths: Optional width for each column
            **kwargs: Additional TreeView configuration
            
        Returns:
            tuple: (TreeView, Scrollbar)
            
        Note:
            Applies consistent dark theme
            Configures columns with provided widths
            Sets up proper scrollbar integration
        """
        style = ttk.Style()
        style.configure("Treeview", 
                       background=self.colors['text'],
                       foreground=self.colors['fg'],
                       fieldbackground=self.colors['text'])
        
        tree = ttk.Treeview(parent, columns=columns, show="headings", 
                            selectmode="extended", **kwargs)
        
        # Configure columns
        for i, col in enumerate(columns):
            tree.heading(col, text=col.capitalize())
            if column_widths and i < len(column_widths):
                tree.column(col, width=column_widths[i], minwidth=50, stretch=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        return tree, scrollbar

    def bulk_update_tags(self, record_ids, tag, operation="add"):
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            for rec_id in record_ids:
                current_tags = self.get_current_tags(rec_id)
                tags_list = [t.strip() for t in current_tags.split(",") if t.strip()] if current_tags else []
                if operation == "add" and tag not in tags_list:
                    tags_list.append(tag)
                elif operation == "remove" and tag in tags_list:
                    tags_list.remove(tag)
                new_tags = ", ".join(tags_list) if tags_list else None
                cursor.execute("UPDATE conversation_history SET tags = ? WHERE id = ?", (new_tags, rec_id))
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Error in bulk tag update: {e}")
            raise

    def update_record_tags(self, record_id, new_tags):
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE conversation_history SET tags = ? WHERE id = ?", (new_tags, record_id))
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Error updating tags: {e}")
            raise

    def edit_selected(self, tree, refresh_callback):
        selection = tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Please select a record to edit")
            return
        if len(selection) > 1:
            messagebox.showinfo("Info", "Please select only one record to edit")
            return
        item = tree.item(selection[0])
        record = item['values']
        self.open_edit_window(record, refresh_callback)

    def bulk_edit_selected(self, tree, refresh_callback):
        selection = tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Please select records to edit")
            return
        self.refresh_tags()  # Add this line before using tags
        bulk_win = StandardWindow(self.root, title="Bulk Edit Tags")
        bulk_win.geometry("400x300")
        selected_records = [tree.item(item)['values'][0] for item in selection]
        tk.Label(bulk_win, text=f"Editing {len(selected_records)} records", bg=self.colors['bg'], fg=self.colors['fg']).pack(pady=10)
        tk.Label(bulk_win, text="Select Tag:", bg=self.colors['bg'], fg=self.colors['fg']).pack()
        tag_combo = ttk.Combobox(bulk_win, values=list(self.APPROVED_TAGS.keys()), state="readonly")
        tag_combo.pack(pady=5)
        if self.APPROVED_TAGS:  # Add this check
            tag_combo.set(list(self.APPROVED_TAGS.keys())[0])
        
        def add_tag_to_all():
            tag = tag_combo.get()
            try:
                self.bulk_update_tags(selected_records, tag, operation="add")
                messagebox.showinfo("Success", f"Added tag '{tag}' to selected records")
                refresh_callback()
                bulk_win.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to update tags: {str(e)}")
        
        def remove_tag_from_all():
            tag = tag_combo.get()
            try:
                self.bulk_update_tags(selected_records, tag, operation="remove")
                messagebox.showinfo("Success", f"Removed tag '{tag}' from selected records")
                refresh_callback()
                bulk_win.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to update tags: {str(e)}")
        
        tk.Button(bulk_win, text="Add Tag to All", command=add_tag_to_all, bg=self.colors['button'], fg=self.colors['fg']).pack(pady=10)
        tk.Button(bulk_win, text="Remove Tag from All", command=remove_tag_from_all, bg=self.colors['button'], fg=self.colors['fg']).pack(pady=5)

    def review_untagged(self):
        self.refresh_tags()  # Add this line at start
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, user_message, assistant_message, tags
                FROM conversation_history
                WHERE tags IS NULL OR tags = ''
                ORDER BY timestamp DESC
            """)
            untagged = cursor.fetchall()
            conn.close()
            
            if not untagged:
                messagebox.showinfo("Info", "No untagged entries found!")
                return
            
            review_win = StandardWindow(self.root, title="Review Untagged Entries")
            review_win.geometry("800x600")
            columns = ("id", "timestamp", "user_message", "assistant_message", "tags")
            tree, scrollbar = self.create_scrolled_treeview(review_win, columns, column_widths=[50, 150, 200, 200, 100])
            
            for col in ["user_message", "assistant_message"]:
                tree.column(col, width=200, stretch=True)
                tree.heading(col, text=col.replace("_", " ").title())
            
            for record in untagged:
                tree.insert("", tk.END, values=record)
            
            tree.pack(side="left", fill="both", expand=True, pady=10)
            scrollbar.pack(side="right", fill="y")
            
            def refresh_untagged():
                tree.delete(*tree.get_children())
                conn = self.get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, timestamp, user_message, assistant_message, tags
                    FROM conversation_history
                    WHERE tags IS NULL OR tags = ''
                    ORDER BY timestamp DESC
                """)
                for record in cursor.fetchall():
                    tree.insert("", tk.END, values=record)
                conn.close()
            
            button_container = review_win.create_button_container()
            tk.Button(button_container, text="Edit Selected", command=lambda: self.edit_selected(tree, refresh_untagged),
                      bg=self.colors['button'], fg=self.colors['fg']).pack(side="left", padx=5)
            tk.Button(button_container, text="Bulk Edit", command=lambda: self.bulk_edit_selected(tree, refresh_untagged),
                      bg=self.colors['button'], fg=self.colors['fg']).pack(side="left", padx=5)
            tk.Button(button_container, text="Refresh", command=refresh_untagged,
                      bg=self.colors['button'], fg=self.colors['fg']).pack(side="left", padx=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load untagged entries: {str(e)}")

    def open_conversation_viewer(self):
        self.refresh_tags()  # Add this line at start
        conv_win = StandardWindow(self.root, title="Conversation History")
        conv_win.geometry("1000x600")
        
        main_container = tk.Frame(conv_win, bg=self.colors['bg'])
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        columns = ("id", "timestamp", "user_message", "assistant_message", "tags")
        tree, scrollbar = self.create_scrolled_treeview(main_container, columns, column_widths=[50, 150, 300, 300, 150])
        
        # Set column headings
        for col in columns:
            tree.heading(col, text=col.capitalize())
        
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Helper function: truncate text and append ellipsis if it exceeds max_length.
        def truncate_text(text, max_length=100):
            if text is None:
                return ""
            return text if len(text) <= max_length else text[:max_length] + "..."
        
        def refresh_conv():
            tree.delete(*tree.get_children())
            try:
                conn = self.get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, timestamp, user_message, assistant_message, tags
                    FROM conversation_history
                    ORDER BY timestamp DESC
                """)
                for record in cursor.fetchall():
                    rec_id, timestamp, user_message, assistant_message, tags = record
                    truncated_user_message = truncate_text(user_message, max_length=100)
                    truncated_assistant_message = truncate_text(assistant_message, max_length=100)
                    tree.insert("", tk.END, values=(rec_id, timestamp, truncated_user_message,
                                                    truncated_assistant_message, tags))
                conn.close()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load conversation history: {str(e)}")
        
        refresh_conv()
        
        btn_container = conv_win.create_button_container()
        tk.Button(
            btn_container,
            text="Refresh",
            command=refresh_conv,
            bg=self.colors['button'],
            fg=self.colors['fg']
        ).pack(side="left", padx=5)

    def refresh_tags(self):
        """Refresh approved tags from database"""
        self.APPROVED_TAGS = load_approved_tags()
        logging.debug(f"Refreshed tags: {self.APPROVED_TAGS}")

    def show_backup_manager(self):
        """Show backup management window"""
        BackupWindow(self.root)

    def refresh_approved_tags(self):
        """Refresh cached approved tags"""
        try:
            with db_manager.get_cursor() as cursor:
                cursor.execute("SELECT name FROM tags")
                self.approved_tags = {row['name'] for row in cursor.fetchall()}
                # Force auto_tagger to refresh its tags too
                self.auto_tagger.refresh_tags()
                logger.debug(f"Refreshed approved tags: {self.approved_tags}")
        except Exception as e:
            logger.error(f"Error refreshing approved tags: {e}")
            self.approved_tags = set()

    def process_message(self, message: str) -> str:
        """Process message with proper context and error handling"""
        try:
            suggestions = auto_tagger.get_suggested_tags(
                message,
                threshold=float(self.config.get('auto_tag_threshold', 0.157))
            )
            # Just extract the tags that passed the threshold
            tags = [tag for tag, _ in suggestions]
            logger.debug(f"Processing message with tags: {tags}")

            # Store in memory
            memory_id = self.memory_manager.store_memory(
                content=message,
                topics=set(tags)
            )
            logger.debug(f"Stored memory {memory_id} with topics: {tags}")
            
            # Now update the interaction and tags
            if memory_id:
                with db_manager.get_cursor() as cursor:
                    log_interaction(memory_id, message, "")
                    update_record_tags(memory_id, ", ".join(tags))
                    cursor.connection.commit()
        
            return memory_id

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return None

    def display_message(self, message: str, memory_id: str):
        """Display message in output area with memory ID reference"""
        try:
            self.output_area.config(state='normal')
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Format and insert the message
            display_text = f"[{timestamp}] User: {message}\n"
            self.output_area.insert(tk.END, display_text)
            
            # Store reference to memory ID
            self.message_widgets[memory_id] = self.output_area.index(tk.END)
            
            # Auto-scroll to bottom
            self.output_area.see(tk.END)
            self.output_area.config(state='disabled')
            
            logger.debug(f"Displayed message with ID {memory_id}")
            
        except Exception as e:
            logger.error(f"Error displaying message: {e}", exc_info=True)

    def display_response(self, response: str):
        """Display response in output area"""
        try:
            self.output_area.config(state='normal')
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Format and insert the response
            response_text = f"[{timestamp}] Assistant: {response}\n"
            self.output_area.insert(tk.END, response_text)
            
            # Auto-scroll to bottom
            self.output_area.see(tk.END)
            self.output_area.config(state='disabled')
            
            logger.debug("Displayed response")
            
        except Exception as e:
            logger.error(f"Error displaying response: {e}", exc_info=True)

    def configure_logging(self):
        # This method is now empty as the logging configuration is handled in the main script
        pass

def start_ui():
    app = AIAssistantUI()
    app.run()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    start_ui()
