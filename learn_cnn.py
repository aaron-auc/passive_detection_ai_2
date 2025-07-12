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
        sweeps = []
        i = 0
        sweep_size = 38402  # Use the full sweep size
        
        while i + sweep_size <= len(trace_data):
            # Extract current sweep
            sweep_data = trace_data[i:i+sweep_size]
            
            # Skip any sweeps that seem incomplete
            if len(sweep_data) == sweep_size:
                sweeps.append(sweep_data[:38400])  # Store as separate sweep, trim to target length (38400 for easy pooling)
            
            # Move to next sweep
            i = i + sweep_size + 12  # Skip full sweep + header for next sweep
        
        return sweeps  # Return a list of individual sweeps

def preprocess_for_cnn(spectral_data, target_length=38400):
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

def create_cnn_model(input_shape=(38400, 1)):
    """
    Create a CNN model for spectral data classification
    """
    model = Sequential([
        # First convolutional layer
        Conv1D(filters=32, kernel_size=128, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=8),
        
        # Second convolutional layer
        Conv1D(filters=64, kernel_size=64, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=8),
        
        # Third convolutional layer
        Conv1D(filters=128, kernel_size=32, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=4),
        Dropout(0.4),
        
        # Fourth convolutional layer
        Conv1D(filters=256, kernel_size=16, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=4),
        Dropout(0.4),
        
        # Flatten and dense layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0003),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Add data augmentation function
def augment_data(X, y):
    """
    Apply simple data augmentation techniques to increase training data
    """
    X_augmented = []
    y_augmented = []
    
    for i in range(len(X)):
        # Original data
        X_augmented.append(X[i])
        y_augmented.append(y[i])
        
        # Add random noise
        noise_level = 0.05
        noise = np.random.normal(0, noise_level, X[i].shape)
        X_augmented.append(X[i] + noise)
        y_augmented.append(y[i])
        
        # Small time shift (5% of signal length)
        shift_amount = int(X[i].shape[0] * 0.05)
        if shift_amount > 0:
            # Shift right
            shifted = np.roll(X[i], shift_amount, axis=0)
            X_augmented.append(shifted)
            y_augmented.append(y[i])
            
            # Shift left
            shifted = np.roll(X[i], -shift_amount, axis=0)
            X_augmented.append(shifted)
            y_augmented.append(y[i])
    
    return np.array(X_augmented), np.array(y_augmented)

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

def load_dataset_cnn(data_dir, label_files):
    """
    Load dataset and prepare it for CNN training
    Memory-optimized version
    """
    X = []
    y = []
    target_length = 38400  # Updated target length for all samples
    
    # Process files in batch to reduce memory consumption
    def process_directory(directory, label):
        batch_size = 20  # Process files in smaller batches
        all_files = glob.glob(os.path.join(directory, '*.shr'))
        
        for i in range(0, len(all_files), batch_size):
            batch_files = all_files[i:i+batch_size]
            batch_X = []
            batch_y = []
            
            for filepath in batch_files:
                filename = os.path.basename(filepath)
                if filename in label_files:
                    sweeps = parse_shr_file(filepath)
                    for sweep_data in sweeps:
                        # Pad if necessary
                        if len(sweep_data) < target_length:
                            sweep_data = np.pad(sweep_data, (0, target_length - len(sweep_data)))
                        elif len(sweep_data) > target_length:
                            sweep_data = sweep_data[:target_length]
                        
                        # Normalize data early to save memory
                        sweep_data = (sweep_data - np.mean(sweep_data)) / (np.std(sweep_data) + 1e-10)
                        batch_X.append(sweep_data)
                        batch_y.append(label)
            
            # Convert batch to numpy array and append
            if batch_X:
                X.extend(batch_X)
                y.extend(batch_y)
                # Clear batch data to free memory
                batch_X = None
                batch_y = None
                import gc
                gc.collect()  # Force garbage collection
    
    # Process files in with_plane folder
    with_plane_dir = os.path.join(data_dir, 'with_plane')
    if os.path.exists(with_plane_dir):
        process_directory(with_plane_dir, 1)
    
    # Process files in without_plane folder
    without_plane_dir = os.path.join(data_dir, 'without_plane')
    if os.path.exists(without_plane_dir):
        process_directory(without_plane_dir, 0)
    
    # Convert to numpy arrays and reshape for CNN - do this in batches
    print(f"Total samples: {len(X)}")
    batch_size = 100
    final_X = []
    
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        batch_array = np.array(batch).reshape(-1, target_length, 1)
        final_X.append(batch_array)
        # Clear memory
        batch = None
        import gc
        gc.collect()
    
    X = np.vstack(final_X)
    y = np.array(y)
    
    return X, y

def train_cnn_model(X, y):
    """
    Train a CNN model with the prepared dataset
    Memory-optimized version
    """
    # Use lower precision to reduce memory usage
    X = X.astype(np.float32)  # Use float32 instead of float64
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Clear the original X data to free memory
    X = None
    import gc
    gc.collect()
    
    # Apply data augmentation in batches
    def batch_augment(X_batch, y_batch, batch_size=50):
        X_aug = []
        y_aug = []
        
        for i in range(0, len(X_batch), batch_size):
            end = min(i + batch_size, len(X_batch))
            X_batch_aug, y_batch_aug = augment_data(X_batch[i:end], y_batch[i:end])
            X_aug.append(X_batch_aug)
            y_aug.extend(y_batch_aug)
            # Free memory
            X_batch_aug = None
            gc.collect()
        
        return np.vstack(X_aug), np.array(y_aug)
    
    print("Augmenting training data...")
    X_train, y_train = batch_augment(X_train, y_train)
    
    # Create a more memory-efficient model
    model = create_cnn_model(input_shape=(X_train.shape[1], 1))
    
    # Use a generator to feed data in batches during training
    def data_generator(X_data, y_data, batch_size=8):
        num_samples = len(X_data)
        while True:
            indices = np.random.permutation(num_samples)
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                yield X_data[batch_indices], y_data[batch_indices]
    
    # Define callbacks for training with memory optimization
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=10,
            restore_best_weights=True,
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2,
            patience=3,
            min_lr=0.000001,
            verbose=1
        ),
        # Add ModelCheckpoint to save best model instead of keeping in memory
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./temp_best_model.keras',
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
    ]
    
    # Calculate steps per epoch based on batch size
    train_batch_size = 8  # Smaller batch size to reduce memory usage
    steps_per_epoch = len(X_train) // train_batch_size
    validation_steps = len(X_test) // train_batch_size
    
    # Train the model using a generator - removed problematic parameters
    print("Training CNN model with generators...")
    history = model.fit(
        data_generator(X_train, y_train, train_batch_size),
        steps_per_epoch=steps_per_epoch,
        validation_data=data_generator(X_test, y_test, train_batch_size),
        validation_steps=validation_steps,
        epochs=30,  # Reduced number of epochs
        callbacks=callbacks,
        verbose=1
        # Removed use_multiprocessing and workers parameters
    )
    
    # Evaluate the model in batches
    test_batch_size = 8
    y_pred_prob = []
    
    for i in range(0, len(X_test), test_batch_size):
        batch_X = X_test[i:i+test_batch_size]
        batch_pred = model.predict(batch_X)
        y_pred_prob.extend(batch_pred.flatten())
    
    y_pred = (np.array(y_pred_prob) > 0.5).astype(int)
    
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
    
    # Ensure the file has the .keras extension
    if not output_path.endswith('.keras'):
        output_path = f"{output_path}.keras"
    
    # Save using the TensorFlow 2.x method with .keras format
    model.save(output_path, save_format='keras')
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