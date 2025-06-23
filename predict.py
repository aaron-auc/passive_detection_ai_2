import numpy as np
import os
import joblib
from scipy import signal
import glob
import argparse

from learn import parse_shr_file, extract_features

def load_model(model_path):
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model

def predict_file(model, file_path):
    try:
        #parse the spectral data from the .shr file
        spectral_data = parse_shr_file(file_path)
        features = extract_features(spectral_data)
        features = features.reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        return prediction, probability
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None, None

def main():
    #argument parsing
    parser = argparse.ArgumentParser(description="Predict plane presence in spectral data files.")
    parser.add_argument('--file_path', type=str, help='Path to the .shr file or directory containing .shr files.')
    parser.add_argument('--model', type=str, default='./model/sanford_19-22_model.joblib', help='Path to the trained model file.')
    args = parser.parse_args()
    
    model_path = args.model
    test_path = args.file_path
    
    #check if the model file exists
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    #check if the test path is a directory or a file
    if os.path.isdir(test_path):
        files = glob.glob(os.path.join(test_path, '*.shr'))
        if not files:
            print(f"No .shr files found in directory: {test_path}")
            return
        
        print(f"Found {len(files)} .shr files to process")
        
        #process each file in the directory
        results = []
        for file_path in files:
            prediction, probability = predict_file(model, file_path)
            
            if prediction is not None:
                filename = os.path.basename(file_path)
                plane_prob = probability[1] if prediction == 1 else probability[0]
                results.append((filename, prediction, plane_prob))
                status = "PLANE DETECTED" if prediction == 1 else "NO PLANE"
                print(f"{filename}: {status} (confidence: {plane_prob:.2f})")
        
        #summary of results
        if results:
            plane_count = sum(1 for _, pred, _ in results if pred == 1)
            print(f"\nSummary: {plane_count} files with planes, {len(results) - plane_count} files without planes")
    
    else:
        if not os.path.exists(test_path):
            print(f"File not found: {test_path}")
            return
        
        #process a single file
        prediction, probability = predict_file(model, test_path)
        
        if prediction is not None:
            if prediction == 1:
                print(f"PLANE DETECTED with {probability[1]:.2f} confidence")
            else:
                print(f"NO PLANE DETECTED with {probability[0]:.2f} confidence")

if __name__ == "__main__":
    main()