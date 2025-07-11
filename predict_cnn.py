import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# Reuse the SHR file parser from learn_cnn.py
def parse_shr_file(file_path):
    with open(file_path, 'rb') as f:
        trace_data = np.fromfile(f, dtype=np.float32)
        #skip the first 130 data points, header
        trace_data = trace_data[130:]
        
        #process to skip 12 values after every 38,402 values (one sweep)
        sweeps = []
        i = 0
        half_sweep_size = 19201  # Only use first half of the sweep (38402 / 2)
        
        while i + half_sweep_size <= len(trace_data):
            # Extract first half of current sweep
            sweep_data = trace_data[i:i+half_sweep_size]
            
            # Skip any sweeps that seem incomplete
            if len(sweep_data) == half_sweep_size:
                sweeps.append(sweep_data[:19200])  # Store as separate sweep, trim to target length
            
            # Move to next sweep
            i = i + 38402 + 12  # Skip full sweep + header for next sweep
        
        return sweeps  # Return a list of individual sweeps

def preprocess_for_prediction(sweep_data, target_length=19200):
    """
    Preprocess a single sweep for prediction with multiple normalization options
    """
    # Ensure consistent length
    if len(sweep_data) > target_length:
        data = sweep_data[:target_length]
    else:
        data = np.pad(sweep_data, (0, max(0, target_length - len(sweep_data))))
    
    # Try different normalization approaches
    # Original normalization
    mean_val = np.mean(data)
    std_val = np.std(data) + 1e-10
    data_norm1 = (data - mean_val) / std_val
    
    # Alternative normalization: Min-Max scaling
    min_val = np.min(data)
    max_val = np.max(data)
    data_norm2 = (data - min_val) / (max_val - min_val + 1e-10)
    
    # Debug information
    print(f"Mean-Std norm stats: min={np.min(data_norm1):.4f}, max={np.max(data_norm1):.4f}, mean={np.mean(data_norm1):.4f}")
    print(f"Min-Max norm stats: min={np.min(data_norm2):.4f}, max={np.max(data_norm2):.4f}, mean={np.mean(data_norm2):.4f}")
    
    # Create a batch of differently normalized data to try multiple approaches
    batch = np.stack([
        data_norm1.reshape(target_length, 1),           # Standard normalization
        data_norm2.reshape(target_length, 1),           # Min-max normalization
        data.reshape(target_length, 1),                 # Raw data (no normalization)
        np.abs(data).reshape(target_length, 1)          # Absolute values
    ])
    
    return batch

def load_keras_model(model_path):
    """
    Load a trained Keras model with fallback options
    """
    try:
        # Try standard loading first
        model = load_model(model_path)
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
            model = load_model(model_path, compile=False)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            print("Alternative loading successful")
            return model
        except Exception as e2:
            print(f"Alternative loading failed: {e2}")
            return None

def predict_file(model, file_path, threshold=0.5):
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
    processed_sweeps = []
    
    for i, sweep_data in enumerate(sweeps):
        print(f"Processing sweep {i+1}/{len(sweeps)}...")
        
        # Try with different data preprocessing approaches
        batch = preprocess_for_prediction(sweep_data)
        
        # Store first preprocessing approach for visualization
        processed_sweeps.append(batch[0].reshape(1, 19200, 1))
        
        # Try different normalization methods
        method_predictions = []
        for j in range(batch.shape[0]):
            # Make prediction with current normalization method
            input_data = batch[j:j+1]
            raw_prediction = model.predict(input_data, verbose=0)
            
            # Extract the prediction value
            if isinstance(raw_prediction, list):
                pred_val = raw_prediction[0][0]
            else:
                pred_val = raw_prediction[0][0]
                
            method_predictions.append(pred_val)
            print(f"  Method {j+1} prediction: {pred_val:.6f}")
        
        # Choose highest confidence prediction from different methods
        best_prediction = max(method_predictions)
        all_predictions.append(best_prediction)
        print(f"Sweep {i+1} best prediction: {best_prediction:.6f}")
        
        # Try to extract intermediate activations to diagnose model
        if i == 0:
            try:
                print("\nDiagnosing model layer activations...")
                # Create a model to extract intermediate layer outputs
                layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name.lower() or 'dense' in layer.name.lower()]
                activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
                activations = activation_model.predict(processed_sweeps[0])
                
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
        
        # Only process first few sweeps during debugging
        if i >= 2:  # Process just 3 sweeps for debugging
            break
    
    # Calculate average prediction and determine if a plane is present
    avg_prediction = np.mean(all_predictions)
    has_plane = avg_prediction > threshold
    
    return {
        'predictions': all_predictions,
        'average': avg_prediction,
        'has_plane': has_plane,
        'sweeps': len(all_predictions),
        'processed_data': processed_sweeps,
        'raw_sweeps': sweeps[:len(all_predictions)]
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
    
    # Plot prediction for each sweep
    plt.subplot(2, 1, 2)
    sweep_indices = np.arange(1, len(result['predictions']) + 1)
    plt.bar(sweep_indices, result['predictions'], color=['red' if p > 0.5 else 'blue' for p in result['predictions']])
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.7)
    plt.title(f"Prediction by Sweep (Avg: {result['average']:.4f}, Plane: {'Yes' if result['has_plane'] else 'No'})")
    plt.xlabel("Sweep Number")
    plt.ylabel("Prediction (1 = Plane, 0 = No Plane)")
    plt.ylim(0, 1)
    
    # Add text with results
    plt.figtext(0.5, 0.01, 
                f"File: {os.path.basename(file_path)}\n" +
                f"Average prediction: {result['average']:.4f}\n" +
                f"Decision: {'PLANE DETECTED' if result['has_plane'] else 'NO PLANE DETECTED'}",
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

def main():
    parser = argparse.ArgumentParser(description="Test a file against a trained Keras model.")
    parser.add_argument('--model', type=str, default='./model/plane_detector_cnn', help='Path to the trained Keras model.')
    parser.add_argument('--file', type=str, required=True, help='Path to the .shr file to test.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold (default: 0.5).')
    parser.add_argument('--visualize', action='store_true', help='Visualize the prediction results.')
    parser.add_argument('--output_dir', type=str, help='Directory to save visualization results.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with extra logging.')
    args = parser.parse_args()
    
    # Enable debug information in TensorFlow if requested
    if args.debug:
        tf.config.run_functions_eagerly(True)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        print("Debug mode enabled")
    
    # Load the model
    model = load_keras_model(args.model)
    if model is None:
        return
    
    # Test model with a simple array to check if it's working
    print("\nTesting model with sample data...")
    
    # Create test data with varying patterns to test model responsiveness
    test_batch = np.zeros((4, 19200, 1))
    # Random noise
    test_batch[0, :, 0] = np.random.normal(0, 1, 19200)
    # Sine wave pattern
    test_batch[1, :, 0] = np.sin(np.linspace(0, 10*np.pi, 19200)) * 0.5
    # Pulse pattern
    test_batch[2, :, 0] = np.zeros(19200)
    test_batch[2, 9000:9200, 0] = 1.0
    # Mixed pattern
    test_batch[3, :, 0] = np.random.normal(0, 0.2, 19200) + np.sin(np.linspace(0, 6*np.pi, 19200)) * 0.5
    
    for i, test_data in enumerate(test_batch):
        pattern_name = ["Random noise", "Sine wave", "Pulse", "Mixed pattern"][i]
        print(f"\nTesting with {pattern_name}...")
        test_prediction = model.predict(test_data.reshape(1, 19200, 1), verbose=0)
        print(f"Test prediction: {test_prediction}")
    
    # Make prediction on actual file
    print("\nProcessing actual file...")
    start_time = time.time()
    result = predict_file(model, args.file, args.threshold)
    prediction_time = time.time() - start_time
    
    if result is None:
        print(f"Failed to process file: {args.file}")
        return
    
    # Print results
    print("\n" + "="*50)
    print(f"File: {os.path.basename(args.file)}")
    print(f"Sweeps analyzed: {result['sweeps']}")
    print(f"Average prediction: {result['average']:.4f}")
    print(f"Decision: {'PLANE DETECTED' if result['has_plane'] else 'NO PLANE DETECTED'}")
    print(f"Processing time: {prediction_time:.2f} seconds")
    print("="*50)
    
    # Visualize if requested
    if args.visualize:
        visualize_prediction(result, args.file, args.output_dir)

if __name__ == "__main__":
    main()
