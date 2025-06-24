import socket
import time
import subprocess
import numpy as np
import os
import joblib
from learn import parse_shr_file, extract_features

HOST = '192.168.86.26'  #signal hound ip
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
        features = extract_features(spectral_data)
        features = features.reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        return prediction, probability
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None, None

#load the model
model_path = './model/plane_detector_2.joblib'
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

while True:
    #signal hound connect
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            print("Connected to Signal Hound Spike")

            s.sendall(b'*IDN?\n')
            response = s.recv(1024).decode('utf-8')
            print(f"Device Response: {response}")

            #s.sendall(b'INSTRUMENT:SELECT SA\n')

            #s.sendall(b'INIT:CONT OFF\n')

            output_directory = 'C:\\Users\\kaido\\repos\\passive_detection_ai\\recordings\\'
            save_command = f'REC:SWEEP:FILE:DIR {output_directory}\n'.encode('utf-8')
            s.sendall(save_command)

            s.sendall(b'REC:SWEEP:START\n')
            print("Recording started...")

            time.sleep(10)

            s.sendall(b'REC:SWEEP:STOP\n')
            print("Recording stopped.")

            #save_command = f'REC:SWEEP:FILE:DIR {output_directory}{output_filename}\n'.encode('utf-8')
            #s.sendall(save_command)            print(f"Recording saved to {output_directory}")

            time.sleep(2)
            try:
                files = [os.path.join(output_directory, f) for f in os.listdir(output_directory) if os.path.isfile(os.path.join(output_directory, f))]
                if files:
                    newest_file = max(files, key=os.path.getmtime)
                    print(f"Using newest file: {newest_file}")
                    
                    #run the newest file against the model
                    if model:
                        prediction, probability = predict_file(model, newest_file)
                        if prediction is not None:
                            if prediction == 1:
                                print(f"PLANE DETECTED with {probability[1]:.2f} confidence")
                            else:
                                print(f"NO PLANE DETECTED with {probability[0]:.2f} confidence")
                        else:
                            print("Prediction failed.")
                else:
                    print("No files found in the recordings directory.")
            except Exception as e:
                print(f"Error finding or processing recorded files: {str(e)}")


    except Exception as e:
        print(f"Failed to connect to Signal Hound Spike: {e}")

    time.sleep(5)