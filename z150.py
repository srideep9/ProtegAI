import tkinter as tk
from tkinter import scrolledtext, ttk, filedialog
import requests
import json
import threading
import speech_recognition as sr
import pyttsx3
import time
from PIL import Image, ImageTk
import io
import base64

class protégAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("protégAI")
        self.root.geometry("900x700")
        self.root.configure(bg="#f7f7f8")  # Updated background color to match modern chat interfaces
        
        self.BASE_URL = "DEPLOYED_URL"
        self.headers = {'Content-Type': 'application/json'}
        
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 175)
        self.engine.setProperty('voice', self.engine.getProperty('voices')[0].id)
        
        self.is_listening = False
        self.speech_thread = None
        
        # Create main frame
        self.main_frame = tk.Frame(root, bg="#f7f7f8")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Create UI elements
        self.create_widgets()
        
        # Initialize state
        self.conversation_active = False
        self.messages = []
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Improved speech recognition settings
        self.recognizer.energy_threshold = 300  # Lower threshold for better detection
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8  # Reduced pause threshold for quicker responses
        
        # Loading animation variables
        self.loading_animation_active = False
        self.loading_animation_thread = None
        self.loading_dots = ""
        
    def create_widgets(self):
        # Header with logo and title
        header_frame = tk.Frame(self.main_frame, bg="#f7f7f8")
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Logo placeholder
        logo_text = tk.Label(header_frame, text="🧠", font=("Arial", 32), bg="#f7f7f8")
        logo_text.pack(side=tk.LEFT, padx=(0, 10))
        
        title_frame = tk.Frame(header_frame, bg="#f7f7f8")
        title_frame.pack(side=tk.LEFT)
        
        title_label = tk.Label(title_frame, text="protégAI", font=("Arial", 24, "bold"), bg="#f7f7f8", fg="#2d3748")
        title_label.pack(anchor="w")
        
        subtitle_label = tk.Label(title_frame, text="Voice-Powered Learning Assistant", font=("Arial", 12), bg="#f7f7f8", fg="#4a5568")
        subtitle_label.pack(anchor="w")
        
        # Create a notebook for different sections
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 5))
        
        # Create chat tab
        self.chat_tab = tk.Frame(self.notebook, bg="#f7f7f8")
        self.notebook.add(self.chat_tab, text="Chat")
        
        # Create document tab
        self.document_tab = tk.Frame(self.notebook, bg="#f7f7f8")
        self.notebook.add(self.document_tab, text="Document")
        
        # Document upload frame
        upload_frame = tk.LabelFrame(self.document_tab, text="Document Upload", font=("Arial", 12, "bold"), bg="#f7f7f8", fg="#2d3748", padx=15, pady=15)
        upload_frame.pack(fill=tk.X, pady=(10, 10), padx=10)
        
        upload_methods_frame = tk.Frame(upload_frame, bg="#f7f7f8")
        upload_methods_frame.pack(fill=tk.X)
        
        # File upload button
        self.file_button = ttk.Button(upload_methods_frame, text="Select File", command=self.upload_file)
        self.file_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Or label
        or_label = tk.Label(upload_methods_frame, text="or", bg="#f7f7f8", fg="#4a5568")
        or_label.pack(side=tk.LEFT, padx=10)
        
        # Paste text button
        self.paste_button = ttk.Button(upload_methods_frame, text="Paste Text", command=self.show_paste_dialog)
        self.paste_button.pack(side=tk.LEFT, padx=(10, 0))
        
        # Document status
        self.document_status = tk.Label(upload_frame, text="No document uploaded", bg="#f7f7f8", fg="#4a5568", font=("Arial", 10, "italic"))
        self.document_status.pack(anchor="w", pady=(10, 0))
        
        # Document preview area
        preview_frame = tk.LabelFrame(self.document_tab, text="Document Preview", font=("Arial", 12, "bold"), bg="#f7f7f8", fg="#2d3748", padx=15, pady=15)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 10), padx=10)
        
        self.document_preview = scrolledtext.ScrolledText(
            preview_frame, 
            font=("Arial", 11),
            bg="white",
            fg="#2d3748",
            wrap=tk.WORD,
            padx=10,
            pady=10,
            state=tk.DISABLED
        )
        self.document_preview.pack(fill=tk.BOTH, expand=True)
        
        # Chat area in chat tab
        self.conversation_area = tk.Canvas(self.chat_tab, bg="white", bd=0, highlightthickness=0)
        self.conversation_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))
        
        # Create a frame inside the canvas to hold the messages
        self.message_frame = tk.Frame(self.conversation_area, bg="white")
        self.message_frame_id = self.conversation_area.create_window((0, 0), window=self.message_frame, anchor="nw", width=self.conversation_area.winfo_width())
        
        # Add scrollbar to conversation area
        self.conversation_scrollbar = ttk.Scrollbar(self.chat_tab, orient="vertical", command=self.conversation_area.yview)
        self.conversation_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=(10, 0))
        self.conversation_area.configure(yscrollcommand=self.conversation_scrollbar.set)
        
        # Bind the canvas resizing event
        self.conversation_area.bind("<Configure>", self.on_canvas_configure)
        self.message_frame.bind("<Configure>", self.on_frame_configure)
        
        # Input area
        input_frame = tk.Frame(self.chat_tab, bg="#f7f7f8", pady=10)
        input_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0, 10))
        
        # Rounded input box
        self.input_container = tk.Frame(input_frame, bg="#e2e8f0", bd=0, highlightthickness=1, highlightbackground="#cbd5e0", relief=tk.SOLID)
        self.input_container.pack(fill=tk.X, side=tk.LEFT, expand=True, ipady=5)
        
        self.user_input = tk.Entry(
            self.input_container,
            font=("Arial", 12),
            bg="#e2e8f0",
            fg="#2d3748",
            bd=0,
            relief=tk.FLAT,
            insertbackground="#2d3748"
        )
        self.user_input.pack(fill=tk.X, expand=True, padx=10, ipady=5)
        self.user_input.bind("<Return>", self.send_message)
        self.user_input.focus_set()  # Set focus to input box
        
        # Voice button
        self.voice_button = ttk.Button(input_frame, text="🎤", width=3, command=self.toggle_voice_input)
        self.voice_button.pack(side=tk.LEFT, padx=(10, 0))
        
        # Send button
        self.send_button = ttk.Button(input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.LEFT, padx=10)
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Configure style
        style = ttk.Style()
        style.configure('TButton', font=('Arial', 11), padding=5)
        style.configure('TNotebook', background="#f7f7f8")
        style.configure('TNotebook.Tab', padding=(10, 5), font=('Arial', 10))
    
    def on_canvas_configure(self, event):
        self.conversation_area.itemconfig(self.message_frame_id, width=event.width)
    
    def on_frame_configure(self, event):
        # Update the scrollregion to encompass the entire frame
        self.conversation_area.configure(scrollregion=self.conversation_area.bbox("all"))
    
    def upload_file(self):
        file_path = filedialog.askopenfilename(
            title="Select a document",
            filetypes=[("Text files", "*.txt"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r') as file:
                document_text = file.read()
            
            # Update preview
            self.update_document_preview(document_text)
            
            # Upload document
            self.upload_document(document_text)
            
            # Switch to chat tab
            self.notebook.select(0)
            
        except Exception as e:
            self.update_status(f"Error reading file: {str(e)}")
    
    def update_document_preview(self, document_text):
        self.document_preview.config(state=tk.NORMAL)
        self.document_preview.delete(1.0, tk.END)
        self.document_preview.insert(tk.END, document_text)
        self.document_preview.config(state=tk.DISABLED)
    
    def show_paste_dialog(self):
        paste_dialog = tk.Toplevel(self.root)
        paste_dialog.title("Paste Document Text")
        paste_dialog.geometry("600x400")
        paste_dialog.configure(bg="#f7f7f8")
        
        instruction = tk.Label(paste_dialog, text="Paste or type your document text below:", pady=10, bg="#f7f7f8")
        instruction.pack(fill=tk.X)
        
        text_area = scrolledtext.ScrolledText(paste_dialog, wrap=tk.WORD, font=("Arial", 11))
        text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        button_frame = tk.Frame(paste_dialog, bg="#f7f7f8")
        button_frame.pack(fill=tk.X, pady=10)
        
        cancel_btn = ttk.Button(button_frame, text="Cancel", command=paste_dialog.destroy)
        cancel_btn.pack(side=tk.RIGHT, padx=10)
        
        def submit_text():
            document_text = text_area.get("1.0", tk.END).strip()
            if document_text:
                self.update_document_preview(document_text)
                self.upload_document(document_text)
                self.notebook.select(0)  # Switch to chat tab
            paste_dialog.destroy()
        
        submit_btn = ttk.Button(button_frame, text="Upload", command=submit_text)
        submit_btn.pack(side=tk.RIGHT)
    
    def upload_document(self, document_text):
        self.update_status("Uploading document...")
        self.start_loading_animation()
        
        try:
            response = requests.post(
                f"{self.BASE_URL}/api/upload_document",
                headers=self.headers,
                json={"document": document_text},
                timeout=10
            )
            
            self.stop_loading_animation()
            
            if response.status_code == 200:
                self.document_status.config(text="Document uploaded successfully!", fg="#38a169")
                self.update_status("Document uploaded. Ready for conversation.")
                self.conversation_active = True
                
                # Add system message to conversation
                self.add_message("system", "Document uploaded successfully! What topic would you like to discuss?")
            else:
                self.document_status.config(text=f"Error: {response.text}", fg="#e53e3e")
                self.update_status("Error uploading document.")
                
        except requests.exceptions.RequestException as e:
            self.stop_loading_animation()
            self.document_status.config(text=f"Connection error. Is the server running?", fg="#e53e3e")
            self.update_status(f"Connection error: {str(e)}")
    
    def send_message(self, event=None):
        if not self.conversation_active:
            self.update_status("Please upload a document first.")
            return
        
        message = self.user_input.get().strip()
        if not message:
            return
        
        self.user_input.delete(0, tk.END)
        self.add_message("user", message)
        
        threading.Thread(target=self.process_message, args=(message,), daemon=True).start()
    
    def process_message(self, message):
        self.start_loading_animation("Processing")
        
        try:
            response = requests.post(
                f"{self.BASE_URL}/api/chat",
                headers=self.headers,
                json={"message": message},
                timeout=20
            )
            
            self.stop_loading_animation()
            
            if response.status_code == 200:
                response_data = response.json()
                self.add_message("assistant", response_data["response"])
                self.update_status(f"Flow status: {response_data['flow_status']}")
            else:
                self.add_message("system", f"Error: {response.text}")
                self.update_status("Error in processing your message.")
                
        except requests.exceptions.RequestException as e:
            self.stop_loading_animation()
            self.add_message("system", "Connection error. Is the server running?")
            self.update_status(f"Connection error: {str(e)}")
    
    def toggle_voice_input(self):
        if self.is_listening:
            self.stop_listening()
        else:
            self.start_listening()
    
    def start_listening(self):
        self.is_listening = True
        self.voice_button.config(text="⏹️")
        self.update_status("Listening...")
        
        # Visual feedback
        self.input_container.config(highlightbackground="#38a169")  # Green highlight when listening
        
        if self.speech_thread is None or not self.speech_thread.is_alive():
            self.speech_thread = threading.Thread(target=self.listen_for_speech, daemon=True)
            self.speech_thread.start()
    
    def stop_listening(self):
        self.is_listening = False
        self.voice_button.config(text="🎤")
        self.update_status("Ready")
        self.input_container.config(highlightbackground="#cbd5e0")  # Reset highlight
    
    def listen_for_speech(self):
        # Faster speech recognition using non-blocking approach
        with sr.Microphone() as source:
            # Faster ambient noise adjustment
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            while self.is_listening:
                try:
                    # Set shorter timeout for faster response
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=7)
                    
                    self.root.after(0, lambda: self.update_status("Processing speech..."))
                    
                    # Use threading to avoid freezing the UI
                    def process_audio(audio_data):
                        try:
                            # Use faster API options
                            text = self.recognizer.recognize_google(audio_data, language="en-US")
                            
                            if text:
                                # Update the input field with the recognized text but don't send
                                self.root.after(0, lambda t=text: self.user_input.insert(0, t))
                                self.root.after(0, self.stop_listening)
                                # Focus on the input field for editing
                                self.root.after(10, self.user_input.focus_set)
                        except sr.UnknownValueError:
                            self.root.after(0, lambda: self.update_status("Could not understand audio. Try again."))
                            self.root.after(1000, lambda: self.update_status("Listening..."))
                        except Exception as e:
                            self.root.after(0, lambda e=e: self.update_status(f"Error: {str(e)}"))
                            self.root.after(0, self.stop_listening)
                    
                    # Start a new thread for processing the audio
                    threading.Thread(target=process_audio, args=(audio,), daemon=True).start()
                    
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    self.root.after(0, lambda e=e: self.update_status(f"Error: {str(e)}"))
                    self.root.after(0, self.stop_listening)
                    break
    
    def start_loading_animation(self, prefix="Loading"):
        """Start the loading animation in the status bar"""
        if self.loading_animation_active:
            return
            
        self.loading_animation_active = True
        self.loading_dots = ""
        
        def animate_loading():
            while self.loading_animation_active:
                self.loading_dots = (self.loading_dots + ".") if len(self.loading_dots) < 3 else ""
                self.root.after(0, lambda: self.status_bar.config(text=f"{prefix}{self.loading_dots}"))
                time.sleep(0.4)
                
        self.loading_animation_thread = threading.Thread(target=animate_loading, daemon=True)
        self.loading_animation_thread.start()
    
    def stop_loading_animation(self):
        """Stop the loading animation"""
        self.loading_animation_active = False
        if self.loading_animation_thread and self.loading_animation_thread.is_alive():
            self.loading_animation_thread.join(timeout=0.5)
        self.root.after(0, lambda: self.status_bar.config(text="Ready"))
    
    def add_message(self, sender, message):
        # Create a new message frame
        message_container = tk.Frame(self.message_frame, bg="white", padx=10, pady=10)
        message_container.pack(fill=tk.X, padx=10, pady=5)
        
        # Create message bubble based on sender
        if sender == "user":
            bubble = tk.Frame(message_container, bg="#e9f5fe", padx=12, pady=12, bd=0)
            bubble.pack(side=tk.RIGHT, anchor="e")
            
            text_label = tk.Label(bubble, text=message, wraplength=500, justify=tk.LEFT,
                                   bg="#e9f5fe", fg="#2d3748", font=("Arial", 11))
            text_label.pack()
            
        elif sender == "assistant":
            bubble = tk.Frame(message_container, bg="#f0f0f0", padx=12, pady=12, bd=0)
            bubble.pack(side=tk.LEFT, anchor="w")
            
            text_label = tk.Label(bubble, text=message, wraplength=500, justify=tk.LEFT,
                                   bg="#f0f0f0", fg="#2d3748", font=("Arial", 11))
            text_label.pack()
            
        else:  # system
            bubble = tk.Frame(message_container, bg="#f8f0ff", padx=12, pady=12, bd=0)
            bubble.pack(side=tk.LEFT, anchor="w")
            
            text_label = tk.Label(bubble, text=message, wraplength=500, justify=tk.LEFT,
                                   bg="#f8f0ff", fg="#2d3748", font=("Arial", 11))
            text_label.pack()
        
        self.messages.append({"sender": sender, "message": message})
        
        # Scroll to the bottom
        self.message_frame.update_idletasks()
        self.conversation_area.yview_moveto(1.0)
    
    def update_status(self, message):
        if not self.loading_animation_active:  # Only update if no animation is running
            self.status_bar.config(text=message)
            self.root.update_idletasks()

    def on_close(self):
        self.loading_animation_active = False  # Stop any loading animations
        try:
            response = requests.post(
                f"{self.BASE_URL}/api/clear",
                headers=self.headers,
                json={"message": "exit"},
                timeout=5
            )
        except:
            pass
        self.root.destroy()

def main():
    root = tk.Tk()
    app = protégAIApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()