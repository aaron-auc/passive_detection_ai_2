import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import tensorflow as tf
from tensorflow import keras
import time

def parse_shr_file(file_path):
    """
    Parse a .shr file to extract spectral data as individual sweeps.
    Each sweep is expected to be 38400 data points long.
    """
    with open(file_path, 'rb') as f:
        trace_data = np.fromfile(f, dtype=np.float32)
        #skip the first 130 data points, header
        trace_data = trace_data[130:]
        
        #process to skip 12 values after every 38,402 values (one sweep)
        sweeps = []
        i = 0
        full_sweep_size = 38402  # Use full sweep size
        
        while i + full_sweep_size <= len(trace_data):
            # Extract full sweep
            sweep_data = trace_data[i:i+full_sweep_size]
            
            # Skip any sweeps that seem incomplete
            if len(sweep_data) == full_sweep_size:
                sweeps.append(sweep_data[:38400])  # Store as separate sweep, trim to target length
            
            # Move to next sweep
            i = i + full_sweep_size + 12  # Skip full sweep + header for next sweep
        
        return sweeps  # Return a list of individual sweeps

def preprocess_for_prediction(sweep_data, target_length=38400):
    """
    Preprocess a single sweep for prediction with multiple normalization options
    """
    # Ensure consistent length
    if len(sweep_data) > target_length:
        data = sweep_data[:target_length]
    else:
        data = np.pad(sweep_data, (0, max(0, target_length - len(sweep_data))))
    
    # Apply the same normalization used during training
    normalized_data = (data - np.mean(data)) / (np.std(data) + 1e-10)
    
    # Return normalized data reshaped for model input
    return np.stack([normalized_data.reshape(target_length, 1)])

def load_keras_model(model_path):
    """
    Load a trained Keras model with fallback options
    """
    try:
        # Try standard loading first
        model = keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        model.summary()
        
        # Check if model output layer has sigmoid activation (binary classification)
        output_layer = model.layers[-1]
        has_sigmoid = hasattr(output_layer, 'activation') and output_layer.activation.__name__ == 'sigmoid'
        print(f"Output layer has sigmoid activation: {has_sigmoid}")
        
        if not has_sigmoid:
            print("Warning: Model may not have sigmoid activation in output layer.")
            print("Adding sigmoid activation layer to ensure outputs between 0 and 1.")
            from tensorflow.keras.layers import Activation
            model_with_sigmoid = tf.keras.Sequential([
                model,
                Activation('sigmoid')
            ])
            model_with_sigmoid.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model_with_sigmoid
            
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
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
            return None

def calibrate_confidence(raw_prediction, calibration_factor=0.3):
    """
    Apply a calibration to overconfident predictions to make them more realistic
    A simple approach to soften extreme probabilities
    """
    # Apply softening to extreme probabilities
    # This pushes very confident predictions away from the extremes (0 or 1)
    calibrated = raw_prediction * (1 - calibration_factor) + 0.5 * calibration_factor
    return calibrated

def predict_file(model, file_path, threshold=0.5, calibrate=False):
    """
    Predict if a file contains a plane with multiple approaches
    """
    # Parse the file to get sweeps
    try:
        sweeps = parse_shr_file(file_path)
        print(f"Successfully parsed {len(sweeps)} sweeps from {file_path}")
        if sweeps:
            print(f"First sweep stats: min={np.min(sweeps[0]):.4f}, max={np.max(sweeps[0]):.4f}, length={len(sweeps[0])}")
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return None
    
    if not sweeps:
        print(f"No valid sweeps found in {file_path}")
        return None
    
    # Process each sweep and make predictions
    all_predictions = []
    all_raw_predictions = []  # Store raw predictions for analysis
    processed_sweeps = []
    
    print(f"Processing {len(sweeps)} sweeps...")
    
    # Process all sweeps
    for i, sweep_data in enumerate(sweeps):
        should_show_progress = 1
        
        if should_show_progress:
            print(f"Processing sweep {i+1}/{len(sweeps)}...")
        
        # Preprocess data with proper normalization matching training
        batch = preprocess_for_prediction(sweep_data)
        
        # Store sweeps for visualization
        processed_sweeps.append(batch[0].reshape(1, 38400, 1))
        
        # Make prediction with normalized data
        raw_prediction = model.predict(batch)
        
        # Extract the prediction value
        if isinstance(raw_prediction, list):
            pred_val = raw_prediction[0][0]
        else:
            pred_val = raw_prediction[0][0]
        
        # Store the raw prediction for debugging
        method_raw_predictions = [pred_val]
        
        # Apply calibration if enabled
        if calibrate and (pred_val > 0.95 or pred_val < 0.05):
            calibrated_val = calibrate_confidence(pred_val)
            if should_show_progress:
                print(f"  Raw prediction: {pred_val:.6f} (calibrated to {calibrated_val:.6f})")
            pred_val = calibrated_val
        elif should_show_progress:
            print(f"  Raw prediction: {pred_val:.6f}")
        
        all_predictions.append(pred_val)
        all_raw_predictions.append(method_raw_predictions)
        
        # Try to extract intermediate activations to diagnose model - only for the first sweep
        if i == 0:
            try:
                print("\nDiagnosing model layer activations...")
                # Create a model to extract intermediate layer outputs
                layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name.lower() or 'dense' in layer.name.lower()]
                activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
                activations = activation_model.predict(batch)
                
                # Print activation statistics for each layer
                for j, activation in enumerate(activations):
                    layer_name = model.layers[j].name if j < len(model.layers) else f"Layer_{j}"
                    act_min = np.min(activation)
                    act_max = np.max(activation)
                    act_mean = np.mean(activation)
                    act_zeros = np.sum(activation == 0) / activation.size * 100
                    print(f"Layer {layer_name}: min={act_min:.4f}, max={act_max:.4f}, mean={act_mean:.4f}, zeros={act_zeros:.1f}%")
            except Exception as e:
                print(f"Could not extract activations: {e}")
    
    # Print summary statistics
    print(f"\nProcessed all {len(sweeps)} sweeps.")
    print(f"Prediction stats: min={np.min(all_predictions):.4f}, max={np.max(all_predictions):.4f}, mean={np.mean(all_predictions):.4f}")
    
    # Calculate average prediction and determine if a plane is present
    # Use a more conservative approach - require multiple sweeps to agree
    avg_prediction = np.mean(all_predictions)
    consistency = np.std(all_predictions)  # Check how consistent predictions are
    
    # Adjust confidence based on consistency
    confidence_factor = 1.0 - min(1.0, consistency * 2)  # Lower confidence when predictions vary a lot
    
    # Add additional safeguards against overconfidence
    # If all predictions are extremely high (> 0.9) or low (< 0.1), this might indicate a calibration issue
    if np.all(np.array(all_predictions) > 0.9) or np.all(np.array(all_predictions) < 0.1):
        print("WARNING: All predictions are extreme values. This might indicate a calibration issue.")
        # Further adjust confidence when all predictions are extreme
        confidence_factor *= 0.7
    
    adjusted_avg = avg_prediction * confidence_factor + 0.5 * (1 - confidence_factor)
    
    print(f"Raw average: {avg_prediction:.4f}, Consistency: {consistency:.4f}")
    print(f"Confidence factor: {confidence_factor:.4f}, Adjusted average: {adjusted_avg:.4f}")
    print(f"Total sweeps analyzed: {len(all_predictions)}")
    
    # Use adjusted average for final decision
    has_plane = adjusted_avg > threshold
    
    return {
        'predictions': all_predictions,
        'raw_predictions': all_raw_predictions,
        'average': avg_prediction,
        'adjusted_average': adjusted_avg,
        'consistency': consistency,
        'confidence_factor': confidence_factor,
        'has_plane': has_plane,
        'sweeps': len(all_predictions),
        'processed_data': processed_sweeps,
        'raw_sweeps': sweeps[:min(len(all_predictions), 10)]  # Only store first 10 raw sweeps for memory efficiency
    }

def visualize_prediction(result, file_path, output_dir=None):
    """
    Visualize the prediction results
    """
    if not result or 'predictions' not in result:
        print("No valid prediction results to visualize")
        return
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Plot raw data from first sweep
    plt.subplot(2, 1, 1)
    plt.plot(result['raw_sweeps'][0])
    plt.title(f"Raw Data - First Sweep from {os.path.basename(file_path)}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    
    # Plot prediction for each sweep - may need to limit display for large sweep counts
    plt.subplot(2, 1, 2)
    
    # For readability, if there are many sweeps, display a summary rather than individual bars
    if len(result['predictions']) > 50:
        # Plot a rolling average - fix dimension mismatch issue
        window_size = max(5, len(result['predictions']) // 20)  # Adaptive window size
        rolling_avg = np.convolve(result['predictions'], np.ones(window_size)/window_size, mode='valid')
        
        # Ensure x_vals and rolling_avg have the same length
        x_vals = np.arange(len(rolling_avg))
        
        # Adjust x_vals to be centered correctly (optional)
        x_offset = window_size // 2
        x_vals = x_vals + x_offset
        
        plt.plot(x_vals, rolling_avg, 'r-', linewidth=2)
        plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.7)
        
        # Add sample points - ensure we're using existing indices only
        sample_size = min(50, len(result['predictions']))
        sample_indices = np.linspace(0, len(result['predictions'])-1, sample_size, dtype=int)
        plt.scatter(sample_indices, [result['predictions'][i] for i in sample_indices], 
                   c=['red' if p > 0.5 else 'blue' for p in [result['predictions'][i] for i in sample_indices]],
                   alpha=0.5)
        
        plt.title(f"Prediction by Sweep (Total: {len(result['predictions'])} sweeps, Avg: {result['average']:.4f})")
    else:
        # If fewer sweeps, show individual bars as before
        sweep_indices = np.arange(1, len(result['predictions']) + 1)
        plt.bar(sweep_indices, result['predictions'], color=['red' if p > 0.5 else 'blue' for p in result['predictions']])
        plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.7)
        plt.title(f"Prediction by Sweep (Avg: {result['average']:.4f}, Adj: {result['adjusted_average']:.4f})")
    
    plt.xlabel("Sweep Number")
    plt.ylabel("Prediction (1 = Plane, 0 = No Plane)")
    plt.ylim(0, 1)
    
    # Add text with results
    plt.figtext(0.5, 0.01, 
                f"File: {os.path.basename(file_path)}\n" +
                f"Sweeps analyzed: {result['sweeps']}, Raw average: {result['average']:.4f}, Adjusted avg: {result['adjusted_average']:.4f}\n" +
                f"Decision: {'PLANE DETECTED' if result['has_plane'] else 'NO PLANE DETECTED'} (Confidence: {result['confidence_factor']:.2f})",
                ha='center', fontsize=12, bbox={'facecolor':'yellow', 'alpha':0.5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save or show the figure
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"prediction_{os.path.basename(file_path)}.png")
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def process_directory(model, directory, threshold=0.5, calibrate=False, visualize=False, output_dir=None):
    """
    Process all .shr files in a directory
    """
    print(f"Processing all .shr files in directory: {directory}")
    results = {}
    
    # Get all .shr files in the directory
    shr_files = [f for f in os.listdir(directory) if f.lower().endswith('.shr')]
    
    if not shr_files:
        print("No .shr files found in the directory.")
        return results
    
    print(f"Found {len(shr_files)} .shr files to process.")
    
    # Process each file
    for i, filename in enumerate(shr_files):
        file_path = os.path.join(directory, filename)
        print(f"\n[{i+1}/{len(shr_files)}] Processing {filename}...")
        
        # Process the file
        start_time = time.time()
        result = predict_file(model, file_path, threshold, calibrate=calibrate)
        prediction_time = time.time() - start_time
        
        if result:
            # Store results
            results[filename] = result
            
            # Print summary
            print(f"File: {filename}")
            print(f"Sweeps analyzed: {result['sweeps']}")
            print(f"Raw average prediction: {result['average']:.4f}")
            print(f"Adjusted prediction: {result['adjusted_average']:.4f}")
            print(f"Decision: {'PLANE DETECTED' if result['has_plane'] else 'NO PLANE DETECTED'}")
            print(f"Processing time: {prediction_time:.2f} seconds")
            
            # Visualize if requested
            if visualize:
                visualize_prediction(result, file_path, output_dir)
    
    # Print summary of all results
    print("\n" + "="*50)
    print(f"Processed {len(results)} files")
    planes_detected = sum(1 for r in results.values() if r['has_plane'])
    print(f"Files with planes detected: {planes_detected}")
    print(f"Files without planes detected: {len(results) - planes_detected}")
    print("="*50)
    
    return results

def main():
    # Argument parser for command line options
    parser = argparse.ArgumentParser(description="Test file(s) against a trained Keras model.")
    parser.add_argument('--model', type=str, default='./model/plane_detector_cnn', help='Path to the trained Keras model.')
    parser.add_argument('--file', type=str, help='Path to the .shr file to test.')
    parser.add_argument('--dir', type=str, help='Directory containing .shr files to process.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold (default: 0.5).')
    parser.add_argument('--visualize', action='store_true', help='Visualize the prediction results.')
    parser.add_argument('--output_dir', type=str, help='Directory to save visualization results.')
    parser.add_argument('--calibration', action='store_true', help='Enable prediction calibration (default: disabled).')
    args = parser.parse_args()
    
    # Ensure either --file or --dir is provided
    if not args.file and not args.dir:
        parser.error("Either --file or --dir must be specified.")
    
    # Load the model
    model = load_keras_model(args.model)
    if model is None:
        return
    
    # Test model with a simple array to check if it's working
    print("\nTesting model with sample data...")
    
    # Create test data with varying patterns to test model responsiveness
    test_batch = np.zeros((1, 38400, 1))
    # Create test data with noise and normalize it like in training
    raw_test_data = np.random.normal(0, 1, 38400)
    test_batch[0, :, 0] = (raw_test_data - np.mean(raw_test_data)) / (np.std(raw_test_data) + 1e-10)
    
    print("Test data stats:")
    print(f"  Min: {np.min(test_batch):.4f}")
    print(f"  Max: {np.max(test_batch):.4f}")
    print(f"  Mean: {np.mean(test_batch):.4f}")
    print(f"  Std: {np.std(test_batch):.4f}")
    
    # Apply calibration to test predictions
    calibrate = args.calibration
    print(f"Prediction calibration: {'Enabled' if args.calibration else 'Disabled'}")
    
    print("\nTesting with normalized random noise...")
    test_prediction = model.predict(test_batch)
    raw_pred_val = test_prediction[0][0] if isinstance(test_prediction, np.ndarray) else test_prediction[0]
    
    if calibrate:
        calibrated_pred = calibrate_confidence(raw_pred_val)
        print(f"Test prediction: {raw_pred_val} (calibrated to {calibrated_pred})")
    else:
        print(f"Test prediction: {raw_pred_val}")
    
    # Add a second test with an extreme signal to see if the model can respond to it
    extreme_test = np.zeros((1, 38400, 1))
    # Create a pattern with higher amplitude and normalize it
    pattern = np.sin(np.linspace(0, 100*np.pi, 38400)) * 10
    pattern = (pattern - np.mean(pattern)) / (np.std(pattern) + 1e-10)
    extreme_test[0, :, 0] = pattern
    
    print("\nTesting with normalized extreme pattern...")
    extreme_prediction = model.predict(extreme_test)
    extreme_pred_val = extreme_prediction[0][0] if isinstance(extreme_prediction, np.ndarray) else extreme_prediction[0]
    print(f"Extreme pattern prediction: {extreme_pred_val}")
    
    # Process file or directory
    if args.file:
        # Make prediction on a single file
        print("\nProcessing file...")
        start_time = time.time()
        result = predict_file(model, args.file, args.threshold, calibrate=calibrate)
        prediction_time = time.time() - start_time
        
        # Print results
        print("\n" + "="*50)
        print(f"File: {os.path.basename(args.file)}")
        print(f"Sweeps analyzed: {result['sweeps']}")
        print(f"Raw average prediction: {result['average']:.4f}")
        print(f"Adjusted prediction: {result['adjusted_average']:.4f}")
        print(f"Prediction consistency: {result['consistency']:.4f}")
        print(f"Decision: {'PLANE DETECTED' if result['has_plane'] else 'NO PLANE DETECTED'}")
        print(f"Processing time: {prediction_time:.2f} seconds")
        print("="*50)
        
        # Visualize if requested
        if args.visualize:
            visualize_prediction(result, args.file, args.output_dir)
    
    elif args.dir:
        # Process all files in the directory
        process_directory(model, args.dir, args.threshold, calibrate, args.visualize, args.output_dir)

if __name__ == "__main__":
    main()
