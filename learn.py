import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
import glob
import joblib
from scipy import signal
import argparse

def parse_shr_file(file_path):
    with open(file_path, 'rb') as f:
        header_size = 16384
        header = f.read(header_size)
        trace_data = np.fromfile(f, dtype=np.float32)
        
    return trace_data

def extract_features(spectral_data):
    spectral_data = np.nan_to_num(spectral_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    features = []
    
    #calculate basic statistics
    features.append(np.mean(spectral_data))
    features.append(np.std(spectral_data))
    features.append(np.max(spectral_data))
    features.append(np.argmax(spectral_data))
    
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
    
    #ensure features are finite
    features = np.array(features, dtype=np.float64)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features

def create_label_mapping(data_dir):
    label_files = {}
    
    #files in with_plane folder are labeled as 1
    with_plane_dir = os.path.join(data_dir, 'with_plane')
    if os.path.exists(with_plane_dir):
        for filepath in glob.glob(os.path.join(with_plane_dir, '*.shr')):
            filename = os.path.basename(filepath)
            label_files[filename] = 1
    
    #files in without_plane folder are labeled as 0
    without_plane_dir = os.path.join(data_dir, 'without_plane')
    if os.path.exists(without_plane_dir):
        for filepath in glob.glob(os.path.join(without_plane_dir, '*.shr')):
            filename = os.path.basename(filepath)
            label_files[filename] = 0
            
    print(f"Found {sum(label_files.values())} files with planes and {len(label_files) - sum(label_files.values())} files without planes.")
    
    return label_files

def load_dataset(data_dir, label_files):
    X = []
    y = []
    
    #files in with_plane folder have features extracted
    with_plane_dir = os.path.join(data_dir, 'with_plane')
    if os.path.exists(with_plane_dir):
        for filepath in glob.glob(os.path.join(with_plane_dir, '*.shr')):
            filename = os.path.basename(filepath)
            if filename in label_files:
                spectral_data = parse_shr_file(filepath)
                features = extract_features(spectral_data)
                X.append(features)
                y.append(label_files[filename])
    
    #files in without_plane folder have features extracted
    without_plane_dir = os.path.join(data_dir, 'without_plane')
    if os.path.exists(without_plane_dir):
        for filepath in glob.glob(os.path.join(without_plane_dir, '*.shr')):
            filename = os.path.basename(filepath)
            if filename in label_files:
                spectral_data = parse_shr_file(filepath)
                features = extract_features(spectral_data)
                X.append(features)
                y.append(label_files[filename])
    
    return np.array(X), np.array(y)

def train_model(X, y):
    #split the dataset into training and testing sets. should probably set a better split or get rid of this entirely.
    #just useful for quick testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    #use randomforst, gradient boosting, and svm
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    svm = SVC(kernel='rbf', probability=True, random_state=42)

    #use a voting classifier with soft voting
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
            voting='soft'
        ))
    ])
    
    #fit the model to the training data
    model.fit(X_train, y_train)
    
    #print the accuracy and classification report
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, X_test, y_test

def save_model(model, output_path):
    output_dir = os.path.dirname(output_path)
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")

def main():
    #argument parser for command line options
    parser = argparse.ArgumentParser(description="Train a plane detection model.")
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing the training data.')
    parser.add_argument('--output_model', type=str, default='./model/plane_detector.joblib', help='Path to save the trained model.')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    output_model = args.output_model
    
    #label the data in with_plane and without_plane folders
    label_files = create_label_mapping(data_dir)
    
    #extract features and load the dataset
    X, y = load_dataset(data_dir, label_files)
    
    #train the model
    model, X_test, y_test = train_model(X, y)
    
    #save the trained model to the specified output path
    save_model(model, output_model)

if __name__ == "__main__":
    main()