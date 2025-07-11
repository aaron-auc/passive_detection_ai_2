import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import csv
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.fft import rfft

# Function to parse the .shr file and extract trace data
def parse_shr_file(file_path):
    with open(file_path, 'rb') as f:
        trace_data = np.fromfile(f, dtype=np.float32)
        # Skip the first 130 data points
        trace_data = trace_data[130:]
        
        # Process to skip 12 values after every 38,402 values
        result = []
        i = 0
        while i < len(trace_data):
            # Add next 38,402 values
            chunk_end = min(i + 38402, len(trace_data))
            result.extend(trace_data[i:chunk_end])
            
            # Skip next 12 values if we have enough data left
            i = chunk_end + 12
        
        return np.array(result, dtype=np.float32)

# Organize trace data into individual sweeps
def organize_sweeps(trace_data, sweep_size=38402):
    # Calculate how many complete sweeps we have
    num_sweeps = len(trace_data) // sweep_size
    
    # Reshape the data to have each sweep as a row
    sweeps = np.array([trace_data[i * sweep_size:(i + 1) * sweep_size] for i in range(num_sweeps)])
    
    return sweeps

# Calculate the average values for each frequency across all sweeps
def calculate_average_frequencies(sweeps):
    return np.mean(sweeps, axis=0)

# Plot the average frequency values
def plot_average_frequencies(avg_freqs, avg_freqs2=None, labels=None, alpha=0.7, linewidth=1.0):
    plt.figure(figsize=(12, 6))
    
    # Calculate the frequency range from 9 kHz to 3 GHz
    start_freq = 9e3  # 9 kHz
    end_freq = 3e9    # 3 GHz
    
    # Generate frequency values for the x-axis
    freq_range = np.linspace(start_freq, end_freq, len(avg_freqs))
    
    # Plot first dataset with opacity set by alpha parameter and specified line width
    plt.plot(freq_range, avg_freqs, label=labels[0] if labels else 'File 1', alpha=alpha, linewidth=linewidth)
    
    # Plot second dataset if provided
    if avg_freqs2 is not None:
        # Make sure the second dataset uses the same frequency range
        if len(avg_freqs2) != len(avg_freqs):
            # Interpolate if lengths are different
            x2 = np.linspace(0, 1, len(avg_freqs2))
            x1 = np.linspace(0, 1, len(avg_freqs))
            avg_freqs2_interp = np.interp(x1, x2, avg_freqs2)
            plt.plot(freq_range, avg_freqs2_interp, label=labels[1] if labels else 'File 2', alpha=alpha, linewidth=linewidth)
        else:
            plt.plot(freq_range, avg_freqs2, label=labels[1] if labels else 'File 2', alpha=alpha, linewidth=linewidth)
        plt.legend()
    
    plt.title('Average Frequency Values Across All Sweeps')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Average Value')
    plt.grid(True)
    
    # Format the x-axis with appropriate frequency units
    plt.ticklabel_format(axis='x', style='scientific')

    plt.show()

# Extract features from spectral data (imported from learn.py)
def extract_features(spectral_data):
    spectral_data = np.nan_to_num(spectral_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    features = []
    
    # Calculate basic statistics
    features.append(np.mean(spectral_data))
    features.append(np.std(spectral_data))
    features.append(np.max(spectral_data))
    features.append(np.argmax(spectral_data))
    
    # Calculate spectral flatness
    try:
        safe_data = np.abs(spectral_data) + 1e-10
        log_values = np.log(safe_data)
        log_values = np.nan_to_num(log_values, nan=-30, posinf=-30, neginf=-30)
        log_geometric_mean = np.mean(log_values)
        geometric_mean = np.exp(log_geometric_mean)
        arithmetic_mean = np.mean(safe_data)
        
        if arithmetic_mean > 1e-10:
            spectral_flatness = geometric_mean / arithmetic_mean
        else:
            spectral_flatness = 0.0
            
        features.append(spectral_flatness)
    except Exception as e:
        print(f"Warning: Error calculating spectral flatness: {str(e)}")
        features.append(0.0)
    
    # Frequency domain features
    try:
        fft_result = np.abs(rfft(spectral_data))
        
        max_fft_points = 10000
        if len(fft_result) > max_fft_points:
            indices = np.linspace(0, len(fft_result)-1, max_fft_points, dtype=int)
            fft_result = fft_result[indices]
            
        psd = fft_result ** 2
        
        dominant_freq = np.argmax(psd)
        features.append(dominant_freq)
        
        bands = min(5, len(psd) // 100)
        if bands < 1:
            bands = 1
            
        band_size = len(psd) // bands
        for i in range(bands):
            start = i * band_size
            end = (i + 1) * band_size if i < bands - 1 else len(psd)
            band_energy = np.sum(psd[start:end])
            total_energy = np.sum(psd) + 1e-10
            features.append(band_energy / total_energy)
    except Exception as e:
        print(f"Warning: Error calculating frequency domain features: {str(e)}")
        features.extend([0.0] * (bands + 1))

    # Signal energy and entropy
    try:
        mean_energy = np.mean(spectral_data**2)
        features.append(mean_energy)
        
        if len(psd) > 10000:
            step = len(psd) // 10000
            psd_sample = psd[::step]
        else:
            psd_sample = psd
            
        normalized_psd = psd_sample / (np.sum(psd_sample) + 1e-10)
        non_zero_mask = normalized_psd > 1e-10
        if np.any(non_zero_mask):
            spectral_entropy = -np.sum(normalized_psd[non_zero_mask] * 
                                     np.log2(normalized_psd[non_zero_mask]))
        else:
            spectral_entropy = 0.0
        features.append(spectral_entropy)
    except Exception as e:
        print(f"Warning: Error calculating signal energy features: {str(e)}")
        features.extend([0.0, 0.0])
        
    # Signal variance over time windows
    try:
        max_windows = 20
        window_size = max(len(spectral_data) // max_windows, 1)
        
        window_vars = []
        for i in range(0, len(spectral_data) - window_size, window_size):
            window_vars.append(np.var(spectral_data[i:i+window_size]))
            
            if len(window_vars) >= max_windows:
                break
                
        if window_vars:
            features.append(np.mean(window_vars))
            features.append(np.std(window_vars))
            min_var = min(window_vars) + 1e-10
            max_var = max(window_vars)
            features.append(max_var / min_var)
        else:
            features.extend([0.0, 0.0, 0.0])
    except Exception as e:
        print(f"Warning: Error calculating window variance: {str(e)}")
        features.extend([0.0, 0.0, 0.0])

    # Correlation and autocorrelation
    try:
        max_lag = min(1000, len(spectral_data)//10)
        
        lag_points = [max_lag//20, max_lag//10, max_lag//5]
        
        ac_values = []
        data_mean = np.mean(spectral_data)
        data_var = np.var(spectral_data)
        
        if data_var > 0:
            for lag in lag_points:
                if lag >= len(spectral_data) or lag == 0:
                    ac_values.append(0.0)
                    continue
                
                sum_product = 0
                for i in range(len(spectral_data) - lag):
                    sum_product += (spectral_data[i] - data_mean) * (spectral_data[i + lag] - data_mean)
                
                ac_values.append(sum_product / (len(spectral_data) - lag) / data_var)
        else:
            ac_values = [0.0] * len(lag_points)
            
        features.extend(ac_values)
    except Exception as e:
        print(f"Warning: Error calculating autocorrelation: {str(e)}")
        features.extend([0.0, 0.0, 0.0])
        
    # Signal-to-noise ratio
    try:
        noise_level = np.percentile(np.abs(spectral_data), 10)
        signal_level = np.percentile(np.abs(spectral_data), 90)
        snr = 20 * np.log10((signal_level + 1e-10) / (noise_level + 1e-10))
        features.append(snr)
    except Exception as e:
        print(f"Warning: Error calculating SNR: {str(e)}")
        features.append(0.0)
    
    # Ensure features are finite
    features = np.array(features, dtype=np.float64)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features

# Get feature names for display
def get_feature_names():
    names = [
        "Mean", "Standard Deviation", "Max Value", "Max Value Index",
        "Spectral Flatness", "Dominant Frequency"
    ]
    
    # Add band energy features
    for i in range(5):
        names.append(f"Band Energy {i+1}")
    
    # Add other features
    names.extend([
        "Mean Energy", "Spectral Entropy",
        "Window Variance Mean", "Window Variance Std", "Window Variance Ratio",
        "Autocorrelation 1", "Autocorrelation 2", "Autocorrelation 3",
        "SNR"
    ])
    
    return names

class FileChooserGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SHR File Analyzer")
        self.root.geometry("700x600")
        self.root.resizable(True, True)
        
        # Create main frame
        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding=10)
        file_frame.pack(fill=tk.X, expand=False, pady=5)
        
        # First file selection
        ttk.Label(file_frame, text="With Plane File:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.file1_var = tk.StringVar()
        self.file1_entry = ttk.Entry(file_frame, textvariable=self.file1_var, width=40)
        self.file1_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_file1).grid(row=0, column=2, padx=5, pady=5)
        
        # Second file selection
        ttk.Label(file_frame, text="Without Plane File:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.file2_var = tk.StringVar()
        self.file2_entry = ttk.Entry(file_frame, textvariable=self.file2_var, width=40)
        self.file2_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_file2).grid(row=1, column=2, padx=5, pady=5)
        
        # Configure grid to expand
        file_frame.columnconfigure(1, weight=1)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create tabs
        self.plot_tab = ttk.Frame(self.notebook)
        self.features_tab = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.plot_tab, text="Plots")
        self.notebook.add(self.features_tab, text="Features")
        
        # Setup plot tab
        self.setup_plot_tab(self.plot_tab)
        
        # Setup features tab
        self.setup_features_tab(self.features_tab)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Animation state variables
        self.canvas = None
        self.fig = None
        self.ax = None
        self.sweeps1 = None
        self.sweeps2 = None
        self.current_sweep = 0
        self.max_sweeps = 0
        self.play_animation = False
        self.animation_job = None
        self.freq_range = None
        
        # Feature storage
        self.features1 = None
        self.features2 = None
        self.features_name = get_feature_names()
    
    def setup_plot_tab(self, parent):
        # Options frame
        options_frame = ttk.LabelFrame(parent, text="Plot Options", padding=10)
        options_frame.pack(fill=tk.X, expand=False, pady=5)
        
        # Opacity option
        ttk.Label(options_frame, text="Line Opacity:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.opacity_var = tk.DoubleVar(value=0.7)
        self.opacity_scale = ttk.Scale(options_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, 
                                      variable=self.opacity_var, length=200)
        self.opacity_scale.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Label(options_frame, textvariable=self.opacity_var).grid(row=0, column=2, padx=5, pady=5)
        
        # Line width option
        ttk.Label(options_frame, text="Line Width:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.linewidth_var = tk.DoubleVar(value=1.0)
        self.linewidth_scale = ttk.Scale(options_frame, from_=0.5, to=3.0, orient=tk.HORIZONTAL, 
                                        variable=self.linewidth_var, length=200)
        self.linewidth_scale.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Label(options_frame, textvariable=self.linewidth_var).grid(row=1, column=2, padx=5, pady=5)
        
        # Configure grid to expand
        options_frame.columnconfigure(1, weight=1)
        options_frame.columnconfigure(2, weight=1)
        
        # Animation controls frame
        self.animation_frame = ttk.LabelFrame(parent, text="Sweep Animation Controls", padding=10)
        self.animation_frame.pack(fill=tk.X, expand=False, pady=5)
        
        # Animation control buttons
        self.play_btn = ttk.Button(self.animation_frame, text="▶ Play", command=self.toggle_play)
        self.play_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.prev_btn = ttk.Button(self.animation_frame, text="⏮ Previous", command=self.previous_sweep)
        self.prev_btn.grid(row=0, column=1, padx=5, pady=5)
        
        self.next_btn = ttk.Button(self.animation_frame, text="⏭ Next", command=self.next_sweep)
        self.next_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Sweep slider
        self.sweep_label = ttk.Label(self.animation_frame, text="Sweep: 0/0")
        self.sweep_label.grid(row=0, column=3, padx=5, pady=5)
        
        # Animation speed control
        ttk.Label(self.animation_frame, text="Speed:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_scale = ttk.Scale(self.animation_frame, from_=0.1, to=5.0, orient=tk.HORIZONTAL, 
                                    variable=self.speed_var, length=200)
        self.speed_scale.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        ttk.Label(self.animation_frame, textvariable=self.speed_var).grid(row=1, column=3, padx=5, pady=5)
        
        # Configure animation frame grid
        self.animation_frame.columnconfigure(1, weight=1)
        self.animation_frame.columnconfigure(2, weight=1)
        self.animation_frame.columnconfigure(3, weight=1)
        
        # Action buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, expand=False, pady=10)
        ttk.Button(button_frame, text="Extract Features", command=self.extract_and_display_features).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Process & Plot", command=self.process_and_plot).pack(side=tk.RIGHT, padx=5)
        
        # Plot frame
        plot_frame = ttk.LabelFrame(parent, text="Plot", padding=10)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.plot_widget = ttk.Frame(plot_frame)
        self.plot_widget.pack(fill=tk.BOTH, expand=True)
    
    def setup_features_tab(self, parent):
        # Feature controls frame
        features_control_frame = ttk.Frame(parent, padding=10)
        features_control_frame.pack(fill=tk.X, expand=False, pady=5)
        
        # Add button to extract features
        ttk.Button(features_control_frame, text="Extract Features", 
                  command=self.extract_and_display_features).pack(side=tk.LEFT, padx=5)
        
        # Add button to visualize features
        ttk.Button(features_control_frame, text="Visualize Features", 
                  command=self.visualize_features).pack(side=tk.LEFT, padx=5)
        
        # Add button to export features to CSV
        ttk.Button(features_control_frame, text="Export to CSV", 
                  command=self.export_features_to_csv).pack(side=tk.LEFT, padx=5)
        
        # Create a label to show status of feature extraction
        self.feature_status_var = tk.StringVar(value="No features extracted yet")
        ttk.Label(features_control_frame, textvariable=self.feature_status_var, relief=tk.SUNKEN).pack(fill=tk.X, pady=5)
        
        # Create a frame for feature tables
        features_display_frame = ttk.Frame(parent, padding=10)
        features_display_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Split into two columns for the two files
        left_frame = ttk.LabelFrame(features_display_frame, text="With Plane Features", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        right_frame = ttk.LabelFrame(features_display_frame, text="Without Plane Features", padding=10)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Create treeview for file 1 features
        self.features1_tree = ttk.Treeview(left_frame, columns=('Feature', 'Value'), height=20)
        self.features1_tree.heading('Feature', text='Feature')
        self.features1_tree.heading('Value', text='Value')
        self.features1_tree.column('Feature', width=200, anchor=tk.W)
        self.features1_tree.column('Value', width=150, anchor=tk.CENTER)
        self.features1_tree.column('#0', width=0, stretch=tk.NO)  # Hide the first column
        self.features1_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # Add scrollbar for file 1 features
        features1_scroll = ttk.Scrollbar(left_frame, orient="vertical", command=self.features1_tree.yview)
        features1_scroll.pack(fill=tk.Y, side=tk.RIGHT)
        self.features1_tree.configure(yscrollcommand=features1_scroll.set)
        
        # Create treeview for file 2 features
        self.features2_tree = ttk.Treeview(right_frame, columns=('Feature', 'Value'), height=20)
        self.features2_tree.heading('Feature', text='Feature')
        self.features2_tree.heading('Value', text='Value')
        self.features2_tree.column('Feature', width=200, anchor=tk.W)
        self.features2_tree.column('Value', width=150, anchor=tk.CENTER)
        self.features2_tree.column('#0', width=0, stretch=tk.NO)  # Hide the first column
        self.features2_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # Add scrollbar for file 2 features
        features2_scroll = ttk.Scrollbar(right_frame, orient="vertical", command=self.features2_tree.yview)
        features2_scroll.pack(fill=tk.Y, side=tk.RIGHT)
        self.features2_tree.configure(yscrollcommand=features2_scroll.set)
        
    def extract_and_display_features(self):
        # Check if files are processed
        if self.sweeps1 is None:
            messagebox.showwarning("No Data", "Please process the files first.")
            return
            
        try:
            self.feature_status_var.set("Extracting features...")
            self.root.update()
            
            # Extract features from the average of sweeps in file 1
            avg_freqs1 = calculate_average_frequencies(self.sweeps1)
            self.features1 = extract_features(avg_freqs1)
            
            # Extract features from file 2 if available
            if self.sweeps2 is not None:
                avg_freqs2 = calculate_average_frequencies(self.sweeps2)
                self.features2 = extract_features(avg_freqs2)
            else:
                self.features2 = None
                
            # Clear existing data in treeviews
            for item in self.features1_tree.get_children():
                self.features1_tree.delete(item)
                
            for item in self.features2_tree.get_children():
                self.features2_tree.delete(item)
                
            # Insert features data into treeviews
            for i, feature_name in enumerate(self.features_name):
                if i < len(self.features1):
                    self.features1_tree.insert('', tk.END, 
                                            values=(feature_name, f"{self.features1[i]:.6f}"), iid=f"feature1_{i}")
                    
            if self.features2 is not None:
                for i, feature_name in enumerate(self.features_name):
                    if i < len(self.features2):
                        self.features2_tree.insert('', tk.END, 
                                                values=(feature_name, f"{self.features2[i]:.6f}"), iid=f"feature2_{i}")
            
            # Update status
            self.feature_status_var.set("Features extracted successfully")
            
            # Switch to features tab
            self.notebook.select(self.features_tab)
            
        except Exception as e:
            self.feature_status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to extract features: {str(e)}")
            
    def browse_file1(self):
        filename = filedialog.askopenfilename(
            title="Select With Plane SHR file",
            filetypes=[("SHR files", "*.shr"), ("All files", "*.*")]
        )
        if filename:
            self.file1_var.set(filename)
    
    def browse_file2(self):
        filename = filedialog.askopenfilename(
            title="Select Without Plane SHR file",
            filetypes=[("SHR files", "*.shr"), ("All files", "*.*")]
        )
        if filename:
            self.file2_var.set(filename)
    
    def process_and_plot(self):
        try:
            # Check if at least one file is selected
            if not self.file1_var.get():
                self.status_var.set("Error: Please select at least one file")
                return
            
            self.status_var.set("Processing files...")
            self.root.update()
            
            # Process first file
            trace_data1 = parse_shr_file(self.file1_var.get())
            self.sweeps1 = organize_sweeps(trace_data1)
            avg_freqs1 = calculate_average_frequencies(self.sweeps1)
            
            # Process second file if provided
            self.sweeps2 = None
            avg_freqs2 = None
            if self.file2_var.get():
                trace_data2 = parse_shr_file(self.file2_var.get())
                self.sweeps2 = organize_sweeps(trace_data2)
                avg_freqs2 = calculate_average_frequencies(self.sweeps2)
            
            # Reset features when new files are loaded
            self.features1 = None
            self.features2 = None
            
            # Clear features treeviews
            for tree in [self.features1_tree, self.features2_tree]:
                for item in tree.get_children():
                    tree.delete(item)
            
            # Update feature status
            self.feature_status_var.set("No features extracted yet")
            
            # Use fixed labels
            labels = ["With Plane", "Without Plane"]
            
            # Clear previous plot if exists
            for widget in self.plot_widget.winfo_children():
                widget.destroy()
            
            # Cancel any existing animation
            if self.animation_job:
                self.root.after_cancel(self.animation_job)
                self.animation_job = None
                self.play_animation = False
                self.play_btn.configure(text="▶ Play")
            
            # Create plot in the GUI
            self.fig = plt.Figure(figsize=(10, 5))
            self.ax = self.fig.add_subplot(111)
            
            # Calculate the frequency range from 9 kHz to 3 GHz
            start_freq = 9e3  # 9 kHz
            end_freq = 3e9    # 3 GHz
            
            # Generate frequency values for the x-axis
            self.freq_range = np.linspace(start_freq, end_freq, len(avg_freqs1))
            
            # Plot first dataset with opacity set by alpha parameter and line width
            self.ax.plot(self.freq_range, avg_freqs1, label=labels[0], alpha=self.opacity_var.get(), 
                   linewidth=self.linewidth_var.get())
            
            # Plot second dataset if provided
            if avg_freqs2 is not None:
                # Make sure the second dataset uses the same frequency range
                if len(avg_freqs2) != len(avg_freqs1):
                    # Interpolate if lengths are different
                    x2 = np.linspace(0, 1, len(avg_freqs2))
                    x1 = np.linspace(0, 1, len(avg_freqs1))
                    avg_freqs2_interp = np.interp(x1, x2, avg_freqs2)
                    self.ax.plot(self.freq_range, avg_freqs2_interp, label=labels[1], alpha=self.opacity_var.get(),
                           linewidth=self.linewidth_var.get())
                else:
                    self.ax.plot(self.freq_range, avg_freqs2, label=labels[1], alpha=self.opacity_var.get(),
                           linewidth=self.linewidth_var.get())
                self.ax.legend()
            
            self.ax.set_title('Average Frequency Values Across All Sweeps')
            self.ax.set_xlabel('Frequency (Hz)')
            self.ax.set_ylabel('Average Value')
            self.ax.grid(True)
            
            # Format the x-axis with appropriate frequency units
            self.ax.ticklabel_format(axis='x', style='scientific')
            
            # Embed plot in GUI
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_widget)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Set up animation parameters
            self.current_sweep = 0
            self.max_sweeps = len(self.sweeps1)
            if self.sweeps2 is not None:
                self.max_sweeps = min(self.max_sweeps, len(self.sweeps2))
            
            # Update sweep label
            self.sweep_label.configure(text=f"Sweep: {self.current_sweep+1}/{self.max_sweeps}")
            
            self.status_var.set("Plot created successfully")
                
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")

    def toggle_play(self):
        if self.sweeps1 is None:
            self.status_var.set("Error: No data loaded")
            return
            
        self.play_animation = not self.play_animation
        
        if self.play_animation:
            self.play_btn.configure(text="⏸ Pause")
            self.animate_sweeps()
        else:
            self.play_btn.configure(text="▶ Play")
            if self.animation_job:
                self.root.after_cancel(self.animation_job)
                self.animation_job = None
    
    def animate_sweeps(self):
        if not self.play_animation or self.sweeps1 is None:
            return
            
        self.update_sweep_plot()
        self.next_sweep()
        
        # Calculate delay based on speed (lower value = faster animation)
        delay = int(1000 / self.speed_var.get())
        self.animation_job = self.root.after(delay, self.animate_sweeps)
    
    def next_sweep(self):
        if self.sweeps1 is None:
            return
            
        self.current_sweep = (self.current_sweep + 1) % self.max_sweeps
        self.update_sweep_plot()
    
    def previous_sweep(self):
        if self.sweeps1 is None:
            return
            
        self.current_sweep = (self.current_sweep - 1) % self.max_sweeps
        self.update_sweep_plot()
    
    def update_sweep_plot(self):
        if self.sweeps1 is None or self.fig is None or self.ax is None:
            return
            
        # Clear the plot
        self.ax.clear()
        
        # Plot the current sweep from file 1
        self.ax.plot(self.freq_range, self.sweeps1[self.current_sweep], 
                    label="With Plane", alpha=self.opacity_var.get(), 
                    linewidth=self.linewidth_var.get())
        
        # Plot the current sweep from file 2 if available
        if self.sweeps2 is not None and self.current_sweep < len(self.sweeps2):
            if len(self.sweeps2[self.current_sweep]) != len(self.sweeps1[self.current_sweep]):
                # Interpolate if lengths are different
                x2 = np.linspace(0, 1, len(self.sweeps2[self.current_sweep]))
                x1 = np.linspace(0, 1, len(self.sweeps1[self.current_sweep]))
                sweep2_interp = np.interp(x1, x2, self.sweeps2[self.current_sweep])
                self.ax.plot(self.freq_range, sweep2_interp, 
                            label="Without Plane", alpha=self.opacity_var.get(),
                            linewidth=self.linewidth_var.get())
            else:
                self.ax.plot(self.freq_range, self.sweeps2[self.current_sweep], 
                            label="Without Plane", alpha=self.opacity_var.get(),
                            linewidth=self.linewidth_var.get())
        
        # Update plot labels and formatting
        self.ax.set_title(f'Sweep {self.current_sweep+1}/{self.max_sweeps}')
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Value')
        self.ax.grid(True)
        self.ax.legend()
        
        # Format the x-axis with appropriate frequency units
        self.ax.ticklabel_format(axis='x', style='scientific')
        
        # Update sweep label in the UI
        self.sweep_label.configure(text=f"Sweep: {self.current_sweep+1}/{self.max_sweeps}")
        
        # Redraw the canvas
        self.canvas.draw()
    
    def visualize_features(self):
        # Check if features are extracted
        if self.features1 is None:
            messagebox.showwarning("No Features", "Please extract features first.")
            return
            
        # Create a new window for visualization
        vis_window = tk.Toplevel(self.root)
        vis_window.title("Feature Visualization")
        vis_window.geometry("800x600")
        
        # Create figure for matplotlib
        fig = plt.Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Prepare data for plotting
        feature_names = self.features_name[:len(self.features1)]
        x = np.arange(len(feature_names))
        width = 0.35
        
        # Plot features from first file
        bars1 = ax.bar(x - width/2, self.features1, width, label="With Plane")
        
        # Plot features from second file if available
        if self.features2 is not None:
            bars2 = ax.bar(x + width/2, self.features2[:len(self.features1)], width, label="Without Plane")
            
        # Add labels and formatting
        ax.set_title("Feature Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha="right")
        ax.legend()
        
        # Set y-axis to log scale for better visualization of large ranges
        ax.set_yscale('symlog')  # Symmetric log scale handles negative values
        
        # Add grid lines
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        # Adjust layout
        fig.tight_layout()
        
        # Embed in tkinter window
        canvas = FigureCanvasTkAgg(fig, master=vis_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add explanation text
        explanation = ttk.Label(vis_window, text="Features are displayed on a symmetric log scale. Some values might be too small or large to be visible.")
        explanation.pack(pady=10)
    
    def export_features_to_csv(self):
        # Check if features are extracted
        if self.features1 is None:
            messagebox.showwarning("No Features", "Please extract features first.")
            return
            
        try:
            # Ask user for file location
            file_path = filedialog.asksaveasfilename(
                title="Save Features as CSV",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                defaultextension=".csv"
            )
            
            if not file_path:
                return  # User canceled
                
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header row
                header = ['Feature Name', 'With Plane']
                if self.features2 is not None:
                    header.append('Without Plane')
                    
                writer.writerow(header)
                
                # Write data rows
                for i, feature_name in enumerate(self.features_name):
                    if i < len(self.features1):
                        if self.features2 is not None and i < len(self.features2):
                            writer.writerow([feature_name, self.features1[i], self.features2[i]])
                        else:
                            writer.writerow([feature_name, self.features1[i]])
                            
            self.status_var.set(f"Features exported to {os.path.basename(file_path)}")
            messagebox.showinfo("Export Successful", f"Features exported to {file_path}")
            
        except Exception as e:
            self.status_var.set(f"Error exporting: {str(e)}")
            messagebox.showerror("Export Error", f"Failed to export features: {str(e)}")

def main():
    # Launch the GUI directly
    root = tk.Tk()
    app = FileChooserGUI(root)
    root.mainloop()
    
if __name__ == "__main__":
    main()