import socket
import time
import subprocess
import numpy as np
import os
import joblib
from learn import parse_shr_file, extract_features

HOST = '192.168.1.132'  #signal hound ip
PORT = 5025  #scpi port

#function to load the model from predict.py
def load_model(model_path):
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model

#function to predict using the model from predict.py
def predict_file(model, file_path):
    try:
        spectral_data = parse_shr_file(file_path)
        
        #check if spectral_data is valid
        if spectral_data is None or len(spectral_data) == 0:
            print(f"Error: No spectral data found in {file_path}")
            return None, None
        
        #debug: Print shape of spectral data to understand structure
        print(f"Spectral data shape: {np.shape(spectral_data)}")
        
        #extract features from the spectral data
        features = extract_features(spectral_data)
        
        #debug: Print the extracted features
        print(f"Extracted features: {features}")
        
        features = features.reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        return prediction, probability
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

#load the model
model_path = './model/sanford_test2.joblib'
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

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
        output_directory = 'C:\\Users\\kaido\\repos\\passive_detection_ai\\recordings\\'
        save_command = f'REC:SWEEP:FILE:DIR {output_directory}\n'.encode('utf-8')
        s.sendall(save_command)
        
        while(True):
            s.sendall(b'SYSTEM:CLEAR\n')
            s.sendall(b'REC:SWEEP:START\n')
            print("Recording started with multiple sweeps...")
            time.sleep(10)

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
                    
                    print("Plane or no plane? (Model prediction)")
                    
                    if model:
                        #check model type for debugging
                        print(f"Model type: {type(model).__name__}")
                        
                        prediction, probability = predict_file(model, newest_file)
                        if prediction is not None:
                            if prediction == 1:
                                print(f"PLANE DETECTED with {probability[1]:.2f} confidence")
                                print(f"Full probability array: {probability}")
                            else:
                                print(f"NO PLANE DETECTED with {probability[0]:.2f} confidence")
                                print(f"Full probability array: {probability}")
                        else:
                            print("Prediction failed.")
                else:
                    print("No .shr files found in the recordings directory.")
            except Exception as e:
                print(f"Error finding or processing recorded files: {str(e)}")
                import traceback
                traceback.print_exc()  #print the full traceback for better debugging

except Exception as e:
    print(f"Failed to connect to Signal Hound Spike: {e}")



