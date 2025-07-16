import socket
import time
import subprocess
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from predict_cnn import parse_shr_file, preprocess_for_prediction, calibrate_confidence
import traceback
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import shutil
import requests  # Add requests library for HTTP POST requests

# Get local ipv4 address
def get_local_ipv4_address():
    """
    Retrieves the local IPv4 address of the machine.
    Needed for connecting to the Signal Hound device.
    """
    try:
        # Get the hostname of the local machine
        hostname = socket.gethostname()
        # Resolve the hostname to its corresponding IPv4 address
        ipv4_address = socket.gethostbyname(hostname)
        return ipv4_address
    except socket.gaierror:
        # Handle cases where hostname resolution fails (e.g., no network connection)
        return "Could not determine IPv4 address."

HOST = get_local_ipv4_address()  # Signal Hound IP
PORT = 5025  # SCPI port

# Default recording duration - now will be adjustable
DEFAULT_RECORDING_DURATION = 10  # seconds

#function to load the CNN model
def load_cnn_model(model_path):
    try:
        # Try standard loading first
        model = keras.models.load_model(model_path)
        print(f"CNN model loaded successfully from {model_path}")
        model.summary()
        return model
    except Exception as e:
        print(f"Error loading CNN model: {e}")
        # Try custom model loading approach as fallback
        try:
            print("Attempting alternative model loading approach...")
            # Load with compile=False to avoid optimization issues
            model = keras.models.load_model(model_path, compile=False)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            print("Alternative loading successful")
            return model
        except Exception as e2:
            print(f"Alternative loading failed: {e2}")
            traceback.print_exc()
            return None

#function to predict using the CNN model
def predict_file_cnn(model, file_path, threshold=0.5):
    try:
        # Parse the file to get sweeps
        sweeps = parse_shr_file(file_path)
        
        #check if spectral_data is valid
        if sweeps is None or len(sweeps) == 0:
            print(f"Error: No spectral data found in {file_path}")
            return None, None, None
    
        print(f"Found {len(sweeps)} sweeps in file")
        
        # Process each sweep and make predictions
        all_predictions = []
        
        # Process first few sweeps for efficiency
        num_sweeps_to_process = min(10, len(sweeps))
        for i in range(num_sweeps_to_process):
            # Preprocess data with proper normalization for CNN
            batch = preprocess_for_prediction(sweeps[i])
            
            # Make prediction
            raw_prediction = model.predict(batch, verbose=0)
            
            # Extract the prediction value
            if isinstance(raw_prediction, list):
                pred_val = raw_prediction[0][0]
            else:
                pred_val = raw_prediction[0][0]
            
            # Apply calibration for more realistic probabilities
            calibrated_val = calibrate_confidence(pred_val)
            all_predictions.append(calibrated_val)
        
        # Calculate average prediction and determine if a plane is present
        avg_prediction = np.mean(all_predictions)
        consistency = np.std(all_predictions)
        
        # Adjust confidence based on consistency
        confidence_factor = 1.0 - min(1.0, consistency * 2)
        adjusted_avg = avg_prediction * confidence_factor + 0.5 * (1 - confidence_factor)
        
        has_plane = adjusted_avg > threshold
        
        return int(has_plane), adjusted_avg, confidence_factor
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        traceback.print_exc()
        return None, None, None

class SignalHoundGUI:
    def __init__(self, root, default_model=None):
        self.root = root
        self.model = default_model
        self.model_path = './model/car_3sec_model.keras' if default_model else None
        self.is_recording = False
        self.socket = None
        self.connected = False
        self.recording_thread = None
        self.output_directory = 'C:\\Users\\Kai\\repos\\passive_detection_ai\\recordings\\'
        self.recording_duration = DEFAULT_RECORDING_DURATION
        self.prediction_threshold = 0.5  # Default threshold for prediction
        
        # Configure the root window
        root.title("PADS Autorun CNN")
        root.geometry("800x950")  # Slightly reduce the height since we're combining sections
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Create the main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configuration frame for model and directory selection
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.pack(fill=tk.X, pady=5)
        
        # Model selection
        model_frame = ttk.Frame(config_frame)
        model_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT, padx=5)
        self.model_label = ttk.Label(model_frame, text=self.model_path if self.model_path else "No model selected")
        self.model_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.select_model_btn = ttk.Button(model_frame, text="Select Model", command=self.select_model)
        self.select_model_btn.pack(side=tk.RIGHT, padx=5)
        
        # Output directory selection
        dir_frame = ttk.Frame(config_frame)
        dir_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(dir_frame, text="Output Dir:").pack(side=tk.LEFT, padx=5)
        self.dir_label = ttk.Label(dir_frame, text=self.output_directory)
        self.dir_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.select_dir_btn = ttk.Button(dir_frame, text="Select Directory", command=self.select_directory)
        self.select_dir_btn.pack(side=tk.RIGHT, padx=5)
        
        # Add threshold slider
        threshold_frame = ttk.Frame(config_frame)
        threshold_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(threshold_frame, text="Prediction Threshold:").pack(side=tk.LEFT, padx=5)
        self.threshold_var = tk.DoubleVar(value=self.prediction_threshold)
        self.threshold_slider = ttk.Scale(
            threshold_frame, 
            from_=0.1, 
            to=0.9,
            orient=tk.HORIZONTAL, 
            variable=self.threshold_var,
            command=self.update_threshold
        )
        self.threshold_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Label to display current threshold value
        self.threshold_label = ttk.Label(threshold_frame, text=f"{self.prediction_threshold:.2f}")
        self.threshold_label.pack(side=tk.RIGHT, padx=5)
        
        # Add a new section for coordinates input
        coord_frame = ttk.LabelFrame(main_frame, text="Location Coordinates", padding="10")
        coord_frame.pack(fill=tk.X, pady=5)
        
        # Create a frame for latitude and longitude inputs
        input_frame = ttk.Frame(coord_frame)
        input_frame.pack(fill=tk.X, expand=True, pady=5)
        
        # Latitude input
        ttk.Label(input_frame, text="Latitude:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.lat_var = tk.StringVar(value="")
        self.lat_entry = ttk.Entry(input_frame, textvariable=self.lat_var, width=15)
        self.lat_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Longitude input
        ttk.Label(input_frame, text="Longitude:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.lon_var = tk.StringVar(value="")
        self.lon_entry = ttk.Entry(input_frame, textvariable=self.lon_var, width=15)
        self.lon_entry.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        # Send coordinates button
        self.send_coord_btn = ttk.Button(input_frame, text="Send Coordinates", command=self.send_coordinates)
        self.send_coord_btn.grid(row=0, column=4, padx=5, pady=5, sticky="e")
        
        # Status label for coordinate submission
        self.coord_status = ttk.Label(coord_frame, text="No coordinates sent yet")
        self.coord_status.pack(fill=tk.X, pady=5)
        
        # Connection status
        status_frame = ttk.LabelFrame(main_frame, text="Connection Status", padding="10")
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Not Connected", foreground="red")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.connect_button = ttk.Button(status_frame, text="Connect", command=self.connect_to_device)
        self.connect_button.pack(side=tk.RIGHT, padx=5)
        
        # Recording controls - now including auto-mode options
        control_frame = ttk.LabelFrame(main_frame, text="Recording Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=5)
        
        # Main control buttons row
        buttons_row = ttk.Frame(control_frame)
        buttons_row.pack(fill=tk.X, pady=5)
        
        self.record_button = ttk.Button(buttons_row, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=5)
        self.record_button.state(['disabled'])
        
        # Duration control with just a spinbox
        ttk.Label(buttons_row, text="Duration (sec):").pack(side=tk.LEFT, padx=5)
        
        self.duration_var = tk.IntVar(value=self.recording_duration)
        self.duration_var.trace_add("write", self.update_duration)
        
        self.duration_spin = ttk.Spinbox(
            buttons_row, 
            from_=1, 
            to=60, 
            textvariable=self.duration_var, 
            width=5,
            command=self.update_duration
        )
        self.duration_spin.pack(side=tk.LEFT, padx=5)
        
        # Auto-mode options row
        auto_row = ttk.Frame(control_frame)
        auto_row.pack(fill=tk.X, pady=5)
        
        # Auto-mode checkbox
        self.auto_mode_var = tk.BooleanVar(value=False)
        self.auto_mode_check = ttk.Checkbutton(
            auto_row,
            text="Auto-Record and Analyze",
            variable=self.auto_mode_var,
            command=self.toggle_auto_mode
        )
        self.auto_mode_check.pack(side=tk.LEFT, padx=5)
        
        # Auto-record only checkbox (for manual classification)
        self.auto_record_only_var = tk.BooleanVar(value=False)
        self.auto_record_only_check = ttk.Checkbutton(
            auto_row,
            text="Auto-Record Only (Manual Classify)",
            variable=self.auto_record_only_var,
            command=self.toggle_auto_mode
        )
        self.auto_record_only_check.pack(side=tk.LEFT, padx=5)
        
        # Option to discard files after analysis
        self.discard_files_var = tk.BooleanVar(value=False)
        self.discard_files_check = ttk.Checkbutton(
            auto_row,
            text="Discard Files After Analysis",
            variable=self.discard_files_var
        )
        self.discard_files_check.pack(side=tk.LEFT, padx=20)
        
        # Auto-mode status
        self.auto_mode_status = ttk.Label(auto_row, text="Auto-mode disabled", foreground="grey")
        self.auto_mode_status.pack(side=tk.RIGHT, padx=5)
        
        # Classification frame - now with buttons instead of radio buttons
        class_frame = ttk.LabelFrame(main_frame, text="Classification", padding="10")
        class_frame.pack(fill=tk.X, pady=5)
        
        # Store classification choice but don't use radio buttons
        self.class_var = tk.StringVar(value="No Recording Yet")
        
        # Create a frame for the classification buttons for better layout
        buttons_frame = ttk.Frame(class_frame)
        buttons_frame.pack(fill=tk.X, expand=True)
        
        # Create styled buttons with clear visual difference
        self.plane_button = ttk.Button(buttons_frame, text="CLASSIFY AS PLANE PRESENT", 
                                     command=lambda: self.set_classification("y"),
                                     width=25)
        self.plane_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        self.no_plane_button = ttk.Button(buttons_frame, text="CLASSIFY AS NO PLANE", 
                                        command=lambda: self.set_classification("n"),
                                        width=25)
        self.no_plane_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        # Disable classification buttons initially
        self.plane_button.state(['disabled'])
        self.no_plane_button.state(['disabled'])
        
        # Classification status indicator
        self.class_status = ttk.Label(class_frame, text="No Classification Selected")
        self.class_status.pack(pady=5)
        
        # Results display
        result_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.result_text = tk.Text(result_frame, height=15, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.config(state=tk.DISABLED)
        
        # Progress bar for recording
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Status bar at bottom
        self.status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
        # If model was provided during init, update the status
        if self.model:
            self.log_message(f"Model loaded: {self.model_path}")
            
        # Auto-mode flags for controlling the loop
        self.auto_mode_running = False
        self.auto_record_only = False

    def update_duration(self, *args):
        """Update recording duration when the spinbox changes"""
        try:
            duration = self.duration_var.get()
            if duration > 0:
                if self.recording_duration != duration:
                    self.recording_duration = duration
                    self.log_message(f"Recording duration set to {duration} seconds")
            else:
                self.duration_var.set(1)  # Minimum value
        except:
            self.duration_var.set(self.recording_duration)  # Reset to current value
    
    def update_threshold(self, *args):
        """Update the prediction threshold when the slider changes"""
        try:
            threshold = self.threshold_var.get()
            self.prediction_threshold = threshold
            self.threshold_label.config(text=f"{threshold:.2f}")
            self.log_message(f"Prediction threshold set to {self.prediction_threshold:.2f}")
        except Exception as e:
            self.update_status(f"Error updating threshold: {str(e)}", True)
            self.threshold_var.set(self.prediction_threshold)  # Reset to current value
    
    def toggle_auto_mode(self):
        """Toggle auto recording and analysis mode"""
        is_auto = self.auto_mode_var.get()
        is_auto_record_only = self.auto_record_only_var.get()
        
        # Ensure only one auto mode is active at a time
        if is_auto and is_auto_record_only:
            if self.auto_mode_var.get() != self.auto_mode_running:  # Check which one was just toggled
                self.auto_record_only_var.set(False)
                is_auto_record_only = False
            else:
                self.auto_mode_var.set(False)
                is_auto = False
        
        # First, stop any current auto mode
        was_running = self.auto_mode_running
        self.auto_mode_running = False
        self.auto_record_only = False
        
        if is_auto:
            self.auto_mode_status.config(text="Auto-mode enabled", foreground="green")
            self.log_message("Auto-mode enabled: Will record and analyze continuously")
            
            # Disable classification buttons in auto-mode since it's automated
            self.plane_button.state(['disabled'])
            self.no_plane_button.state(['disabled'])
            
            self.auto_mode_running = True
            self.auto_record_only = False
            
            # If not already recording, start the auto-mode
            if not self.is_recording and not was_running:
                self.start_auto_mode()
        
        elif is_auto_record_only:
            self.auto_mode_status.config(text="Auto-Record Mode enabled", foreground="blue")
            self.log_message("Auto-Record Mode enabled: Will record continuously but wait for manual classification")
            
            self.auto_mode_running = True
            self.auto_record_only = True
            
            # If not already recording, start the auto-mode
            if not self.is_recording and not was_running:
                self.start_auto_mode()
        
        else:
            self.auto_mode_status.config(text="Auto-mode disabled", foreground="grey")
            self.log_message("Auto-mode disabled: Reverting to manual operation")
            # auto_mode_running and auto_record_only are already set to False above

    def start_auto_mode(self):
        """Start the automated recording process"""
        if not self.connected:
            messagebox.showerror("Error", "Not connected to device")
            self.auto_mode_var.set(False)
            self.auto_record_only_var.set(False)
            self.auto_mode_status.config(text="Auto-mode disabled", foreground="grey")
            self.auto_mode_running = False
            self.auto_record_only = False
            return
        
        # Make sure we're not already recording
        if self.is_recording:
            self.update_status("Recording already in progress, waiting to complete...")
            return
            
        if self.auto_mode_running:
            self.update_status("Starting automated recording...")
            self.start_recording()
    
    def set_classification(self, value):
        """Handle classification button press and automatically process recording"""
        self.class_var.set(value)
        
        # Show which classification was selected in the UI
        if value == "y":
            self.class_status.config(text="Processing as: PLANE PRESENT", foreground="blue")
        else:
            self.class_status.config(text="Processing as: NO PLANE", foreground="green")
        
        # Disable both buttons during processing to prevent double-clicks
        self.plane_button.state(['disabled'])
        self.no_plane_button.state(['disabled'])
        
        # Show processing status
        self.update_status("Processing recording with selected classification...")
        
        # Schedule the processing to happen shortly (allows UI to update first)
        self.root.after(100, self.process_recording)

    def select_model(self):
        """Open file dialog to select a model file"""
        model_file = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[
                ("Keras Models", "*.keras *.h5"),
                ("All Files", "*.*")
            ],
            initialdir=os.path.dirname(self.model_path) if self.model_path else "./model"
        )
        
        if model_file:  # If a file was selected (not canceled)
            try:
                # Try to load the selected model
                new_model = load_cnn_model(model_file)
                
                if new_model:
                    # If successful, update the model and path
                    self.model = new_model
                    self.model_path = model_file
                    self.model_label.config(text=os.path.basename(model_file))
                    self.update_status(f"Model loaded: {os.path.basename(model_file)}")
                else:
                    messagebox.showerror("Model Error", "Failed to load the selected model.")
            except Exception as e:
                messagebox.showerror("Model Error", f"Error loading model: {str(e)}")
                traceback.print_exc()
    
    def select_directory(self):
        """Open dialog to select output directory"""
        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_directory
        )
        
        if directory:  # If a directory was selected (not canceled)
            self.output_directory = directory
            if not self.output_directory.endswith(os.sep):
                self.output_directory += os.sep
            self.dir_label.config(text=self.output_directory)
            self.update_status(f"Output directory set to: {self.output_directory}")
            
            # If connected, update the device with the new directory
            if self.connected and self.socket:
                try:
                    save_command = f'REC:SWEEP:FILE:DIR {self.output_directory}\n'.encode('utf-8')
                    self.socket.sendall(save_command)
                    self.update_status("Updated recording directory on device")
                except Exception as e:
                    self.update_status(f"Failed to update directory on device: {str(e)}", True)

    def update_status(self, message, is_error=False):
        self.status_bar.config(text=message)
        if is_error:
            self.log_message(f"ERROR: {message}")
        else:
            self.log_message(message)
    
    def log_message(self, message):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.insert(tk.END, f"{message}\n")
        self.result_text.see(tk.END)
        self.result_text.config(state=tk.DISABLED)
    
    def connect_to_device(self):
        if self.connected:
            self.disconnect_from_device()
            return
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((HOST, PORT))
            self.socket.sendall(b'*IDN?\n')
            response = self.socket.recv(1024).decode('utf-8')
            
            self.status_label.config(text="Connected", foreground="green")
            self.connect_button.config(text="Disconnect")
            self.record_button.state(['!disabled'])
            self.connected = True
            
            self.update_status(f"Connected to: {response.strip()}")
            
            # Set up initial parameters
            self.socket.sendall(b'SENS:FREQ:CENT 1.5e9\n')
            self.socket.sendall(b'SENS:FREQ:SPAN 3e9\n')
            self.socket.sendall(b'SENS:FREQ:CENT:STEP 1e4\n')
            self.socket.sendall(b'INIT:CONT ON\n')
            
            save_command = f'REC:SWEEP:FILE:DIR {self.output_directory}\n'.encode('utf-8')
            self.socket.sendall(save_command)
            
        except Exception as e:
            self.update_status(f"Connection failed: {str(e)}", True)
            if self.socket:
                self.socket.close()
                self.socket = None
    
    def disconnect_from_device(self):
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        self.status_label.config(text="Disconnected", foreground="red")
        self.connect_button.config(text="Connect")
        self.record_button.state(['disabled'])
        self.connected = False
        self.update_status("Disconnected from device")
    
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        if not self.connected:
            messagebox.showerror("Error", "Not connected to device")
            return
        
        self.is_recording = True
        self.record_button.config(text="Stop Recording")
        self.update_status(f"Starting recording (duration: {self.recording_duration} seconds)...")
        
        # Start the recording in a separate thread
        self.recording_thread = threading.Thread(target=self.recording_process)
        self.recording_thread.daemon = True
        self.recording_thread.start()
    
    def recording_process(self):
        try:
            self.socket.sendall(b'SYSTEM:CLEAR\n')
            self.socket.sendall(b'REC:SWEEP:START\n')
            self.update_status("Recording in progress...")
            
            # Update progress bar based on current recording duration
            total_steps = 100
            for i in range(total_steps + 1):
                if not self.is_recording:
                    break
                self.progress_var.set(i)
                time.sleep(self.recording_duration / total_steps)
            
            if self.is_recording:  # If not manually stopped
                self.stop_recording()
                
                # If in full auto-mode, automatically process without user classification
                if self.auto_mode_running and not self.auto_record_only:
                    # Simulate automatic processing after a short delay
                    self.root.after(500, lambda: self.auto_process_recording())
        
        except Exception as e:
            self.update_status(f"Recording error: {str(e)}", True)
            self.stop_recording()
            
            # If in auto-mode, try to continue after error
            if self.auto_mode_running:
                self.root.after(2000, self.start_auto_mode)  # Retry after 2 seconds
    
    def stop_recording(self):
        if not self.is_recording:
            return
            
        try:
            self.socket.sendall(b'REC:SWEEP:STOP\n')
            self.update_status("Recording stopped")
        except Exception as e:
            self.update_status(f"Error stopping recording: {str(e)}", True)
        
        self.is_recording = False
        self.record_button.config(text="Start Recording")
        self.progress_var.set(0)
        
        # Enable classification buttons for manual mode or auto-record-only mode
        if not self.auto_mode_running or (self.auto_mode_running and self.auto_record_only):
            self.plane_button.state(['!disabled'])
            self.no_plane_button.state(['!disabled'])
            self.class_status.config(text="Please classify the recording", foreground="black")
            
            # If in auto-record-only mode, wait for user classification before continuing
            # The auto-mode will resume after process_recording() completes
        elif self.auto_mode_running:
            # Continue the loop for fully automatic mode
            self.root.after(1000, self.start_auto_mode)
    
    def auto_process_recording(self):
        """Automatically process the recording in auto-mode"""
        if not self.auto_mode_running:
            return
            
        self.update_status("Auto-analyzing recording...")
        
        try:
            # Get the newest file
            files = [os.path.join(self.output_directory, f) for f in os.listdir(self.output_directory) 
                    if os.path.isfile(os.path.join(self.output_directory, f)) and f.endswith('.shr')]
            
            if not files:
                self.update_status("No .shr files found for auto-analysis", True)
                # Continue with next recording even if no file was found
                self.root.after(1000, self.start_auto_mode)
                return
                
            newest_file = max(files, key=os.path.getmtime)
            self.update_status(f"Auto-analyzing file: {newest_file}")
            
            # Run CNN prediction with current threshold
            if self.model:
                self.update_status("Running CNN prediction...")
                prediction, confidence, confidence_factor = predict_file_cnn(
                    self.model, 
                    newest_file, 
                    threshold=self.prediction_threshold
                )
                
                if prediction is not None:
                    if prediction == 1:
                        result = f"AUTO-DETECTED: PLANE PRESENT with {confidence:.4f} confidence (factor: {confidence_factor:.2f})"
                        self.update_status(result)
                        
                        # Send POST request for aircraft detection
                        try:
                            payload = {
                                # Round confidence to 2 decimal places for better readability
                                'confidence': round(float(confidence * 100), 2)
                            }
                            response = requests.post('https://pads-website.onrender.com/signals', json=payload, timeout=5)
                            if response.status_code == 200:
                                self.update_status(f"Notification sent successfully: {response.status_code}")
                            else:
                                self.update_status(f"Notification failed with status code: {response.status_code}", True)
                        except Exception as e:
                            self.update_status(f"Failed to send detection notification: {str(e)}", True)
                    else:
                        result = f"AUTO-DETECTED: NO PLANE with {1-confidence:.4f} confidence (factor: {confidence_factor:.2f})"
                        self.update_status(result)
                        
                    # Handle the file based on user preference
                    if self.discard_files_var.get():
                        # Delete the file
                        os.remove(newest_file)
                        self.update_status(f"Discarded file: {os.path.basename(newest_file)}")
                    else:
                        # Move to appropriate folder based on prediction
                        if prediction == 1:
                            dest_folder = os.path.join(self.output_directory, 'data', 'auto_with_plane')
                        else:
                            dest_folder = os.path.join(self.output_directory, 'data', 'auto_without_plane')
                        
                        # Ensure destination folder exists
                        os.makedirs(dest_folder, exist_ok=True)
                        
                        # Move file
                        file_name = os.path.basename(newest_file)
                        destination = os.path.join(dest_folder, file_name)
                        shutil.move(newest_file, destination)
                        self.update_status(f"Auto-sorted file to {destination}")
                else:
                    self.update_status("Auto-analysis prediction failed", True)
            else:
                self.update_status("CNN model not loaded, skipping auto-analysis")
                
            # Continue with next recording if auto-mode is still active
            if self.auto_mode_running:
                self.root.after(1000, self.start_auto_mode)
                
        except Exception as e:
            self.update_status(f"Error in auto-analysis: {str(e)}", True)
            traceback.print_exc()
            
            # Try to continue after error
            if self.auto_mode_running:
                self.root.after(2000, self.start_auto_mode)  # Retry after 2 seconds
    
    def process_recording(self):
        """Process the recording with the current classification"""
        plane_present = self.class_var.get()
        if plane_present not in ['y', 'n']:
            messagebox.showerror("Error", "Please select if a plane was present")
            # Re-enable buttons in case of error
            self.plane_button.state(['!disabled'])
            self.no_plane_button.state(['!disabled'])
            return
        
        try:
            # Get the newest file
            files = [os.path.join(self.output_directory, f) for f in os.listdir(self.output_directory) 
                    if os.path.isfile(os.path.join(self.output_directory, f)) and f.endswith('.shr')]
            
            if not files:
                self.update_status("No .shr files found in the recordings directory", True)
                # Re-enable buttons in case of error
                self.plane_button.state(['!disabled'])
                self.no_plane_button.state(['!disabled'])
                return
                
            newest_file = max(files, key=os.path.getmtime)
            self.update_status(f"Found newest file: {newest_file}")
            
            # Determine destination folder
            if plane_present == 'y':
                dest_folder = os.path.join(self.output_directory, 'data', 'with_plane')
            else:
                dest_folder = os.path.join(self.output_directory, 'data', 'without_plane')
            
            # Ensure destination folder exists
            os.makedirs(dest_folder, exist_ok=True)
            
            # Move file to appropriate folder
            file_name = os.path.basename(newest_file)
            destination = os.path.join(dest_folder, file_name)
            shutil.move(newest_file, destination)
            self.update_status(f"Moved file to {destination}")
            
            # Run CNN prediction with current threshold
            if self.model:
                self.update_status("Running CNN prediction...")
                prediction, confidence, confidence_factor = predict_file_cnn(
                    self.model, 
                    destination, 
                    threshold=self.prediction_threshold
                )
                
                if prediction is not None:
                    if prediction == 1:
                        result = f"PLANE DETECTED with {confidence:.4f} confidence (factor: {confidence_factor:.2f})"
                        self.update_status(result)
                        
                        # Send POST request for aircraft detection
                        try:
                            payload = {
                                # Round confidence to 2 decimal places for better readability
                                'confidence': round(float(confidence * 100), 2)
                            }
                            response = requests.post('https://pads-website.onrender.com/signals', json=payload, timeout=5)
                            if response.status_code == 200:
                                self.update_status(f"Notification sent successfully: {response.status_code}")
                            else:
                                self.update_status(f"Notification failed with status code: {response.status_code}", True)
                        except Exception as e:
                            self.update_status(f"Failed to send detection notification: {str(e)}", True)
                    else:
                        result = f"NO PLANE DETECTED with {1-confidence:.4f} confidence (factor: {confidence_factor:.2f})"
                        self.update_status(result)
                else:
                    self.update_status("Prediction failed.", True)
            else:
                self.update_status("CNN model not loaded, skipping prediction.")
            
            # Reset UI for next recording - keep buttons disabled until next recording
            self.class_var.set("No Recording Yet")
            self.class_status.config(text="Classification complete. Ready for next recording.", foreground="black")
            
            # If in auto-record-only mode, continue with next recording after manual classification
            if self.auto_mode_running and self.auto_record_only:
                self.root.after(1000, self.start_auto_mode)
            
        except Exception as e:
            self.update_status(f"Error processing file: {str(e)}", True)
            traceback.print_exc()
            # Re-enable buttons in case of error
            self.plane_button.state(['!disabled'])
            self.no_plane_button.state(['!disabled'])
            
            # If in auto-record-only mode, still try to continue after error recovery
            if self.auto_mode_running and self.auto_record_only:
                self.root.after(2000, self.start_auto_mode)
    
    def on_close(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
            self.root.destroy()

    def send_coordinates(self):
        """Send the latitude and longitude coordinates as a POST request"""
        try:
            # Get the latitude and longitude values
            lat = self.lat_var.get().strip()
            lon = self.lon_var.get().strip()
            
            # Validate the inputs
            if not lat or not lon:
                messagebox.showerror("Error", "Please enter both latitude and longitude")
                return
            
            try:
                lat_float = float(lat)
                lon_float = float(lon)
                
                # Basic validation of coordinates
                if lat_float < -90 or lat_float > 90:
                    messagebox.showerror("Error", "Latitude must be between -90 and 90 degrees")
                    return
                
                if lon_float < -180 or lon_float > 180:
                    messagebox.showerror("Error", "Longitude must be between -180 and 180 degrees")
                    return
                
            except ValueError:
                messagebox.showerror("Error", "Coordinates must be valid numbers")
                return
            
            # Prepare the payload
            payload = {
                'latitude': lat_float,
                'longitude': lon_float
            }
            
            # Send the POST request
            self.update_status(f"Sending coordinates: {lat}, {lon}")
            response = requests.post('https://pads-website.onrender.com/sensors', json=payload, timeout=5)

            if response.status_code == 200 or response.status_code == 201:
                self.coord_status.config(text=f"Coordinates sent successfully: {lat}, {lon}")
                self.update_status(f"Coordinates sent successfully: Status code {response.status_code}")
            else:
                self.coord_status.config(text=f"Failed to send coordinates: Status {response.status_code}")
                self.update_status(f"Failed to send coordinates. Status code: {response.status_code}", True)
        
        except Exception as e:
            self.coord_status.config(text=f"Error sending coordinates")
            self.update_status(f"Error sending coordinates: {str(e)}", True)
            traceback.print_exc()

# Load the CNN model
default_model_path = './model/car_3sec_model.keras'
try:
    # Try to load default model, but don't fail if it's not there
    # The user can select a model later through the UI
    cnn_model = load_cnn_model(default_model_path)
    if not cnn_model:
        print("Default model not loaded, you can select a model in the UI")
except Exception as e:
    print(f"Error loading default CNN model: {str(e)}")
    cnn_model = None

# Start the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = SignalHoundGUI(root, cnn_model)
    root.mainloop()
