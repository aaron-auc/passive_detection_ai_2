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
from scipy.fft import rfft

np.random.seed(42)
import random
random.seed(42)

#TODO: make sure file is being parsed correctly
def parse_shr_file(file_path):
    with open(file_path, 'rb') as f:
        header_size = 16384
        header = f.read(header_size)
        trace_data = np.fromfile(f, dtype=np.float32)
        
    return trace_data

def extract_features(spectral_data):
    #some files are too large, so limit the data points for large files for now until performance is better
    #max_data_points = 150000
    #if len(spectral_data) > max_data_points:
    #    indices = np.linspace(0, len(spectral_data)-1, max_data_points, dtype=int)
    #    spectral_data = spectral_data[indices]
    
    spectral_data = np.nan_to_num(spectral_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    features = []
    
    #calculate basic statistics
    features.append(np.mean(spectral_data))
    features.append(np.std(spectral_data))
    features.append(np.max(spectral_data))
    features.append(np.argmax(spectral_data))
    
    #calculate skewness and kurtosis
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
    
    #frequency domain features
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

    #signal energy and entropy
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
        
    #signal variance over time windows
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

    #correlation and autocorrelation
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
        
    #signal-to-noise ratio
    try:
        noise_level = np.percentile(np.abs(spectral_data), 10)
        signal_level = np.percentile(np.abs(spectral_data), 90)
        snr = 20 * np.log10((signal_level + 1e-10) / (noise_level + 1e-10))
        features.append(snr)
    except Exception as e:
        print(f"Warning: Error calculating SNR: {str(e)}")
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
    #rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    svm = SVC(kernel='rbf', probability=True, random_state=42)

    #use a voting classifier with soft voting
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', VotingClassifier(
            #estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
            estimators=[('gb', gb), ('svm', svm)],
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