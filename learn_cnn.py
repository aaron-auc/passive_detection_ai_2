import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import glob
import joblib
import argparse
from scipy.fft import rfft
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
import random
random.seed(42)
tf.random.set_seed(42)

#TODO: make sure file is being parsed correctly
def parse_shr_file(file_path):
    with open(file_path, 'rb') as f:
        trace_data = np.fromfile(f, dtype=np.float32)
        #skip the first 130 data points, header
        trace_data = trace_data[130:]
        
        #process to skip 12 values after every 38,402 values (one sweep)
        result = []
        i = 0
        half_sweep_size = 19201  # Only use first half of the sweep (38402 / 2)
        
        while i < len(trace_data):
            #add only first half of the sweep (19201 values)
            chunk_end = min(i + half_sweep_size, len(trace_data))
            result.extend(trace_data[i:chunk_end])
            
            #skip to the beginning of the next sweep
            i = i + 38402 + 12  # Skip full sweep + header for next sweep
        
        return np.array(result, dtype=np.float32)

def preprocess_for_cnn(spectral_data, target_length=19200):
    """
    Preprocess the spectral data for CNN input by reshaping and normalizing
    """
    # Ensure consistent length for CNN input
    if len(spectral_data) > target_length:
        # Take the first target_length points
        data = spectral_data[:target_length]
    else:
        # Pad with zeros if too short
        data = np.pad(spectral_data, (0, max(0, target_length - len(spectral_data))))
    
    # Normalize the data
    data = (data - np.mean(data)) / (np.std(data) + 1e-10)
    
    # Reshape for CNN (samples, timesteps, features)
    data = data.reshape(1, target_length, 1)
    
    return data

def create_cnn_model(input_shape=(19200, 1)):
    """
    Create a CNN model for spectral data classification
    """
    model = Sequential([
        # First convolutional layer
        Conv1D(filters=32, kernel_size=64, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=8),
        
        # Second convolutional layer
        Conv1D(filters=64, kernel_size=32, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=4),
        
        # Third convolutional layer
        Conv1D(filters=128, kernel_size=16, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Flatten and dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

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
        for filepath in glob.glob(os.path.join(with_plane_dir, '*.shr')):
            filename = os.path.basename(filepath)
            if filename in label_files:
                spectral_data = parse_shr_file(filepath)
                features = extract_features(spectral_data)
                X.append(features)
                y.append(label_files[filename])
    
    return np.array(X), np.array(y)

def load_dataset_cnn(data_dir, label_files):
    """
    Load dataset and prepare it for CNN training
    """
    X = []
    y = []
    target_length = 19200  # Target length for all samples
    
    # Process files in with_plane folder
    with_plane_dir = os.path.join(data_dir, 'with_plane')
    if os.path.exists(with_plane_dir):
        for filepath in glob.glob(os.path.join(with_plane_dir, '*.shr')):
            filename = os.path.basename(filepath)
            if filename in label_files:
                spectral_data = parse_shr_file(filepath)
                # Take the first target_length points or pad if necessary
                if len(spectral_data) > target_length:
                    spectral_data = spectral_data[:target_length]
                else:
                    spectral_data = np.pad(spectral_data, (0, max(0, target_length - len(spectral_data))))
                X.append(spectral_data)
                y.append(label_files[filename])
    
    # Process files in without_plane folder
    without_plane_dir = os.path.join(data_dir, 'without_plane')
    if os.path.exists(without_plane_dir):
        for filepath in glob.glob(os.path.join(with_plane_dir, '*.shr')):
            filename = os.path.basename(filepath)
            if filename in label_files:
                spectral_data = parse_shr_file(filepath)
                if len(spectral_data) > target_length:
                    spectral_data = spectral_data[:target_length]
                else:
                    spectral_data = np.pad(spectral_data, (0, max(0, target_length - len(spectral_data))))
                X.append(spectral_data)
                y.append(label_files[filename])
    
    # Convert to numpy arrays and reshape for CNN
    X = np.array(X).reshape(-1, target_length, 1)
    y = np.array(y)
    
    return X, y

def train_model(X, y):
    #split the dataset into training and testing sets. should probably set a better split or get rid of this entirely.
    #just useful for quick testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )
    
    #use randomforst, gradient boosting, and svm
    #rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100)
    svm = SVC(kernel='rbf', probability=True)

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

def train_cnn_model(X, y):
    """
    Train a CNN model with the prepared dataset
    """
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create the CNN model
    model = create_cnn_model(input_shape=(X_train.shape[1], 1))
    
    # Define callbacks for training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    ]
    
    # Train the model
    print("Training CNN model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate the model
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('cnn_training_history.png')
    plt.close()
    
    return model, X_test, y_test

def save_model(model, output_path):
    output_dir = os.path.dirname(output_path)
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # For Keras model, use the built-in save method
    model.save(output_path)
    print(f"CNN model saved to {output_path}")

def main():
    #argument parser for command line options
    parser = argparse.ArgumentParser(description="Train a plane detection CNN model.")
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing the training data.')
    parser.add_argument('--output_model', type=str, default='./model/plane_detector_cnn', help='Path to save the trained model.')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    output_model = args.output_model
    
    #label the data in with_plane and without_plane folders
    label_files = create_label_mapping(data_dir)
    
    #extract features and load the dataset for CNN
    X, y = load_dataset_cnn(data_dir, label_files)
    
    #train the CNN model
    model, X_test, y_test = train_cnn_model(X, y)
    
    #save the trained model to the specified output path
    save_model(model, output_model)

if __name__ == "__main__":
    main()