import socket
import time
import subprocess
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from predict_cnn import parse_shr_file, preprocess_for_prediction, calibrate_confidence
import traceback

HOST = '192.168.0.80'  #signal hound ip
PORT = 5025  #scpi port

RECORDING_DURATION = 10  # seconds

#function to load the CNN model
def load_cnn_model(model_path):
    try:
        # Try standard loading first
        model = load_model(model_path)
        print(f"CNN model loaded successfully from {model_path}")
        model.summary()
        return model
    except Exception as e:
        print(f"Error loading CNN model: {e}")
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

#load the CNN model
model_path = './model/car_3sec_model.keras'
try:
    cnn_model = load_cnn_model(model_path)
except Exception as e:
    print(f"Error loading CNN model: {str(e)}")
    cnn_model = None

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print("Connected to Signal Hound Spike")
        s.sendall(b'*IDN?\n')
        response = s.recv(1024).decode('utf-8')
        print(f"Device Response: {response}")
        
        s.sendall(b'SENS:FREQ:CENT 1.5e9\n')
        s.sendall(b'SENS:FREQ:SPAN 3e9\n')
        s.sendall(b'SENS:FREQ:CENT:STEP 1e4\n')
        
        s.sendall(b'INIT:CONT ON\n')
        output_directory = './recordings/' # Adjust this as needed
        save_command = f'REC:SWEEP:FILE:DIR {output_directory}\n'.encode('utf-8')
        s.sendall(save_command)
        
        while(True):
            s.sendall(b'SYSTEM:CLEAR\n')
            s.sendall(b'REC:SWEEP:START\n')
            print("Recording started with multiple sweeps...")
            time.sleep(RECORDING_DURATION)

            s.sendall(b'REC:SWEEP:STOP\n')
            print("Recording stopped.")
            print(f"Recording saved to {output_directory}")
            
            plane_present = input("Was a plane present during this recording? (y/n): ").lower().strip()
            
            try:
                #get the newest file
                files = [os.path.join(output_directory, f) for f in os.listdir(output_directory) if os.path.isfile(os.path.join(output_directory, f)) and f.endswith('.shr')]
                if files:
                    newest_file = max(files, key=os.path.getmtime)
                    print(f"Found newest file: {newest_file}")
                    
                    #determine destination folder based on user input
                    if plane_present in ('y', 'yes'):
                        dest_folder = os.path.join(output_directory, 'data', 'with_plane')
                    else:
                        dest_folder = os.path.join(output_directory, 'data', 'without_plane')
                    
                    #ensure destination folder exists
                    os.makedirs(dest_folder, exist_ok=True)
                    
                    #move file to appropriate folder
                    file_name = os.path.basename(newest_file)
                    destination = os.path.join(dest_folder, file_name)
                    import shutil
                    shutil.move(newest_file, destination)
                    print(f"Moved file to {destination}")
                    newest_file = destination
                
                    print(f"File size: {os.path.getsize(newest_file)} bytes, Last modified: {time.ctime(os.path.getmtime(newest_file))}")
                    
                    print("Plane or no plane? (CNN model prediction)")
                    
                    if cnn_model:
                        #check model type for debugging
                        print(f"Model type: {type(cnn_model).__name__}")
                        
                        prediction, confidence, confidence_factor = predict_file_cnn(cnn_model, newest_file, threshold=0.5)
                        if prediction is not None:
                            if prediction == 1:
                                print(f"PLANE DETECTED with {confidence:.4f} confidence (factor: {confidence_factor:.2f})")
                            else:
                                print(f"NO PLANE DETECTED with {1-confidence:.4f} confidence (factor: {confidence_factor:.2f})")
                        else:
                            print("Prediction failed.")
                else:
                    print("No .shr files found in the recordings directory.")
            except Exception as e:
                print(f"Error finding or processing recorded files: {str(e)}")
                traceback.print_exc()

except Exception as e:
    print(f"Failed to connect to Signal Hound Spike: {e}")
