from pathlib import Path
dataset_path = r'''/mnt/c/Users/areeb/OneDrive/Desktop/Hexo_AI.Assessment/data/melanoma'''

import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
# Determine dataset path from environment variable or default
DATASET_PATH = os.environ.get('DATASET_PATH', '.')
SUBMISSION_PATH = os.environ.get('SUBMISSION_PATH', 'submission.csv')
SEEDS = [0, 1, 2]

# Define file paths
TRAIN_FILE = os.path.join(DATASET_PATH, 'train.csv')
TEST_FILE = os.path.join(DATASET_PATH, 'test.csv')

# --- Data Loading ---
try:
    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)
except FileNotFoundError:
    # Attempt to load files assuming they might be in a different relative location
    # This handles cases where DATASET_PATH points to the parent directory of the CSVs
    TRAIN_FILE = os.path.join(DATASET_PATH, 'melanoma', 'train.csv')
    TEST_FILE = os.path.join(DATASET_PATH, 'melanoma', 'test.csv')
    try:
        df_train = pd.read_csv(TRAIN_FILE)
        df_test = pd.read_csv(TEST_FILE)
    except FileNotFoundError as e:
        print(f"Critical Error: Could not find train.csv or test.csv. Checked paths: {os.path.join(DATASET_PATH, 'train.csv')}, {TRAIN_FILE}")
        raise e

# --- ID and Target Detection ---
ID_COL = 'image_name'
TARGET_COL = 'target'

# --- Feature Engineering and Preprocessing ---

# Combine data for consistent encoding and imputation
df_test[TARGET_COL] = np.nan
# Drop columns not present in test or irrelevant for modeling
df_combined = pd.concat([
    df_train.drop(columns=['diagnosis', 'benign_malignant'], errors='ignore'), 
    df_test
], ignore_index=True)

# Identify features to use
EXCLUDE_COLS = [ID_COL, TARGET_COL]
FEATURES = [col for col in df_combined.columns if col not in EXCLUDE_COLS]

CATEGORICAL_FEATURES = ['sex', 'anatom_site_general_challenge', 'patient_id']
NUMERICAL_FEATURES = ['age_approx']

# 1. Imputation
for col in NUMERICAL_FEATURES:
    df_combined[col].fillna(df_combined[col].median(), inplace=True)

for col in CATEGORICAL_FEATURES:
    # Fill NaNs with a specific category 'missing'
    df_combined[col].fillna('missing', inplace=True)

# 2. Encoding Categorical Features
# Use Label Encoding for all categorical features, suitable for tree-based models like LGBM.
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    df_combined[col] = le.fit_transform(df_combined[col].astype(str))

# Separate back into train and test
X = df_combined.iloc[:len(df_train)].drop(columns=EXCLUDE_COLS)
y = df_combined.iloc[:len(df_train)][TARGET_COL]
X_test = df_combined.iloc[len(df_train):].drop(columns=EXCLUDE_COLS)

# Check target variability
if y.nunique() > 1:
    TASK_TYPE = 'binary'
    METRIC = 'auc'
else:
    TASK_TYPE = 'constant'
    CONSTANT_PREDICTION = y.iloc[0]
    print(f"Warning: Target column '{TARGET_COL}' is constant (value: {CONSTANT_PREDICTION}). Using constant prediction.")

# --- Model Training (if not constant) ---

if TASK_TYPE == 'binary':
    
    # LightGBM Parameters
    lgb_params = {
        'objective': 'binary',
        'metric': METRIC,
        'boosting_type': 'gbdt',
        'n_estimators': 100,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'n_jobs': -1,
    }

    test_predictions = np.zeros(len(X_test))
    
    # Identify indices of categorical features in the final feature matrix X
    cat_feature_indices = [X.columns.get_loc(c) for c in CATEGORICAL_FEATURES if c in X.columns]

    for seed in SEEDS:
        lgb_params['seed'] = seed
        
        model = lgb.LGBMClassifier(**lgb_params)
        
        try:
            # Explicitly pass categorical features
            model.fit(X, y, categorical_feature=cat_feature_indices)
        except Exception:
            # Fallback if explicit categorical handling fails
            model.fit(X, y)

        # Predict probabilities for the positive class (1)
        preds = model.predict_proba(X_test)[:, 1]
        test_predictions += preds / len(SEEDS)
        
    final_predictions = test_predictions

else: # TASK_TYPE == 'constant'
    # Predict the constant value (0.0 or 1.0)
    final_predictions = np.full(len(X_test), float(CONSTANT_PREDICTION))


# --- Submission Generation ---

submission_df = pd.DataFrame({
    ID_COL: df_test[ID_COL],
    'target': final_predictions
})

# Save the submission file
submission_df.to_csv(SUBMISSION_PATH, index=False)

print(f"Submission saved to {SUBMISSION_PATH}")