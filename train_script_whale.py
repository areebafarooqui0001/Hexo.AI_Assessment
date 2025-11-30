import os
import pandas as pd
import numpy as np
import librosa
import lightgbm as lgb
import warnings

# Suppress warnings during feature extraction
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuration ---
SEEDS = [0, 1, 2]
N_MFCC = 13
# Define DATASET_PATH based on environment variable or fallback
DATASET_PATH = os.environ.get('DATASET_PATH', '/mnt/c/Users/areeb/Downloads/Areeba/Hexo_AI.Assessment/data/whale')

# --- Helper Functions ---

def get_full_path(base_path, relative_path):
    """
    Constructs the full path, handling potential mixed path separators 
    (like Windows backslashes in the CSVs).
    """
    # Normalize path separators
    normalized_path = relative_path.replace('\\', os.sep)
    
    # The relative path is typically 'subdir/filename.aif'
    # We need to join base_path with the relative path components.
    
    # Split the normalized path into components
    parts = normalized_path.split(os.sep)
    
    # If the path starts with 'train' or 'test', we join base_path with the components
    if len(parts) >= 2:
        # parts[0] is 'train' or 'test', parts[-1] is the filename
        return os.path.join(base_path, parts[0], parts[-1])
    else:
        # Fallback for unexpected structure
        return os.path.join(base_path, normalized_path)


def extract_features(file_path, sr=22050):
    """Extracts MFCCs and their derivatives from an audio file."""
    try:
        # Load audio file. Using duration limit for consistency and speed.
        y, sr = librosa.load(file_path, sr=sr, mono=True, duration=5.0)
        
        # 1. MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        
        # 2. Delta and Delta-Delta
        mfccs_delta = librosa.feature.delta(mfccs)
        mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Combine features (3 * N_MFCC)
        features = np.vstack([mfccs, mfccs_delta, mfccs_delta2])
        
        # Aggregate features (Mean and Std Dev across time axis)
        mean_features = np.mean(features, axis=1)
        std_features = np.std(features, axis=1)
        
        # Concatenate mean and std features (Total 6 * N_MFCC features)
        return np.hstack([mean_features, std_features])
        
    except Exception as e:
        # print(f"Error processing {file_path}: {e}")
        # Return a vector of NaNs if processing fails
        return np.full(N_MFCC * 3 * 2, np.nan)

def prepare_data(df, base_path):
    """Processes the dataframe to extract features for all files."""
    X_features = []
    
    for _, row in df.iterrows():
        relative_path = row['filename']
        full_path = get_full_path(base_path, relative_path)
        
        features = extract_features(full_path)
        X_features.append(features)
        
    X = pd.DataFrame(X_features)
    
    # Impute NaNs if any file failed to load (using mean imputation)
    if X.isnull().any().any():
        X = X.fillna(X.mean()) 
    
    return X

# --- Main Execution ---

def run_audio_classification(dataset_path):
    
    # 1. Load Data
    try:
        train_csv_path = os.path.join(dataset_path, 'train.csv')
        test_csv_path = os.path.join(dataset_path, 'test.csv')
        
        df_train = pd.read_csv(train_csv_path)
        df_test = pd.read_csv(test_csv_path)
    except FileNotFoundError as e:
        print(f"Error loading CSV files: {e}")
        return
    
    # Identify ID and Target
    ID_COL = 'filename'
    TARGET_COL = 'label'
    
    # 2. Feature Extraction
    print("Starting feature extraction...")
    X_train_features = prepare_data(df_train, dataset_path)
    y_train = df_train[TARGET_COL]
    X_test_features = prepare_data(df_test, dataset_path)
    
    if X_train_features.shape[0] == 0 or X_test_features.shape[0] == 0:
        print("Feature extraction failed or resulted in empty dataframes. Aborting.")
        return
        
    # 3. Model Training (LightGBM)
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_estimators': 100,
        'learning_rate': 0.05,
        'num_leaves': 10,
        'verbose': -1,
        'n_jobs': -1,
    }
    
    test_predictions = []
    
    for seed in SEEDS:
        print(f"Training model with seed {seed}...")
        lgb_params['random_state'] = seed
        
        model = lgb.LGBMClassifier(**lgb_params)
        
        # Train on the full, small dataset
        model.fit(X_train_features, y_train)
        
        # Predict probabilities
        y_pred_proba = model.predict_proba(X_test_features)[:, 1]
        test_predictions.append(y_pred_proba)
        
    # 4. Ensemble Predictions (Averaging probabilities across seeds)
    avg_predictions = np.mean(test_predictions, axis=0)
    
    # 5. Create Submission File
    submission_df = pd.DataFrame({
        ID_COL: df_test[ID_COL],
        TARGET_COL: avg_predictions
    })
    
    submission_df.to_csv('submission.csv', index=False)
    print("Submission file created successfully.")

if __name__ == '__main__':
    run_audio_classification(DATASET_PATH)