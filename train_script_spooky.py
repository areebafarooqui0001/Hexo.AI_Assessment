from pathlib import Path
dataset_path = r'''/mnt/c/Users/areeb/Downloads/Areeba/Hexo_AI.Assessment/data/spooky'''

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
# Determine paths
# Use environment variable or default path
DATASET_PATH = os.environ.get('DATASET_PATH', './')
SUBMISSION_PATH = os.environ.get('SUBMISSION_PATH', 'submission.csv')
SEEDS = [0, 1, 2]

# File names
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

# Column identification
ID_COL = 'id'
TEXT_COL = 'text'
TARGET_COL = 'author'

# --- Data Loading ---
try:
    train_df = pd.read_csv(os.path.join(DATASET_PATH, TRAIN_FILE))
    test_df = pd.read_csv(os.path.join(DATASET_PATH, TEST_FILE))
except FileNotFoundError as e:
    # Fallback attempt if the path structure is flat
    try:
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
    except Exception as e_inner:
        raise RuntimeError(f"Could not load train or test CSV files: {e_inner}")

# --- Preprocessing and Feature Engineering ---

# 1. Handle missing text
train_df[TEXT_COL] = train_df[TEXT_COL].fillna('')
test_df[TEXT_COL] = test_df[TEXT_COL].fillna('')

# 2. Label Encoding the Target
le = LabelEncoder()
y_train_encoded = le.fit_transform(train_df[TARGET_COL])
CLASS_NAMES = le.classes_

print(f"Detected classes: {CLASS_NAMES}")

# 3. TF-IDF Vectorization
# Using standard parameters suitable for general text classification
tfidf = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2), # Unigrams and Bigrams
    max_features=20000, # Limit features
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(train_df[TEXT_COL])
X_test_tfidf = tfidf.transform(test_df[TEXT_COL])

print(f"Shape of training features: {X_train_tfidf.shape}")

# --- Model Training (Ensemble of Seeds) ---

# Initialize storage for predictions (probabilities)
test_predictions_proba = np.zeros((X_test_tfidf.shape[0], len(CLASS_NAMES)))

for seed in SEEDS:
    print(f"Training model with seed: {seed}")
    
    # Logistic Regression is robust and effective for TF-IDF features
    model = LogisticRegression(
        solver='sag', 
        multi_class='multinomial',
        C=1.0, 
        random_state=seed,
        n_jobs=-1,
        max_iter=1000
    )
    
    model.fit(X_train_tfidf, y_train_encoded)
    
    # Predict probabilities
    proba = model.predict_proba(X_test_tfidf)
    test_predictions_proba += proba / len(SEEDS)

print("Averaging predictions complete.")

# --- Submission Generation ---

submission_df = pd.DataFrame({ID_COL: test_df[ID_COL]})

# Add probability columns using the original class names
for i, class_name in enumerate(CLASS_NAMES):
    submission_df[class_name] = test_predictions_proba[:, i]

# Ensure ID column is preserved exactly and is the first column
submission_cols = [ID_COL] + list(CLASS_NAMES)
submission_df = submission_df[submission_cols]

# Save the submission file
submission_df.to_csv(SUBMISSION_PATH, index=False)

print(f"Submission saved successfully to {SUBMISSION_PATH}")