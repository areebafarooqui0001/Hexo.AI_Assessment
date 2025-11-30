from pathlib import Path
dataset_path = r'''/mnt/c/Users/areeb/OneDrive/Desktop/Hexo_AI.Assessment/data/melanoma'''

import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
# Determine dataset path
DATASET_PATH = os.environ.get('DATASET_PATH', '/mnt/c/Users/areeb/OneDrive/Desktop/Hexo_AI.Assessment/data/melanoma')
SUBMISSION_PATH = os.environ.get('SUBMISSION_PATH', 'submission.csv')
SEEDS = [0, 1, 2]

# --- Data Loading ---
try:
    train_path = os.path.join(DATASET_PATH, 'train.csv')
    test_path = os.path.join(DATASET_PATH, 'test.csv')
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
except Exception as e:
    # If data loading fails, we cannot proceed.
    raise RuntimeError(f"Error loading data: {e}")

# --- Automatic ID and Target Detection ---
ID_COL = 'image_name'
TARGET_COL = 'target'

# --- Critical Check: Constant Target Handling ---
if df_train[TARGET_COL].nunique() <= 1:
    print(f"Warning: Target column '{TARGET_COL}' is constant or nearly constant ({df_train[TARGET_COL].nunique()} unique values). Falling back to constant prediction.")
    
    # Fallback strategy: Predict the single observed value (or 0.0 if binary classification is expected)
    constant_prediction = df_train[TARGET_COL].mode().iloc[0]
    
    submission_df = pd.DataFrame({
        ID_COL: df_test[ID_COL],
        TARGET_COL: float(constant_prediction)
    })
    
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    exit()

# --- Feature Engineering and Preprocessing ---

# Combine data for consistent preprocessing
df_train['is_train'] = 1
df_test['is_train'] = 0
df_combined = pd.concat([df_train.drop(columns=[TARGET_COL]), df_test], ignore_index=True)

# Features to use (excluding IDs and redundant/highly correlated columns like diagnosis/benign_malignant)
EXCLUDE_COLS = [ID_COL, TARGET_COL, 'is_train', 'diagnosis', 'benign_malignant']
features = [col for col in df_train.columns if col not in EXCLUDE_COLS]

# Identify feature types
categorical_cols = df_combined[features].select_dtypes(include=['object']).columns.tolist()
numerical_cols = df_combined[features].select_dtypes(include=['float64', 'int64']).columns.tolist()

# 1. Handle Categorical Features
# Impute missing categorical values with 'missing'
for col in categorical_cols:
    df_combined[col] = df_combined[col].fillna('missing')

# Apply One-Hot Encoding
df_combined = pd.get_dummies(df_combined, columns=categorical_cols, dummy_na=False)

# 2. Handle Numerical Features (Imputation)
# Impute missing numerical values with the median
imputer = SimpleImputer(strategy='median')
df_combined[numerical_cols] = imputer.fit_transform(df_combined[numerical_cols])

# Update feature list after OHE
feature_cols = [col for col in df_combined.columns if col not in EXCLUDE_COLS and col != 'patient_id'] # Drop patient_id due to high cardinality relative to N=50

# Re-split data
X = df_combined[df_combined['is_train'] == 1].drop(columns=['is_train'])
X_test = df_combined[df_combined['is_train'] == 0].drop(columns=['is_train'])
y = df_train[TARGET_COL]

# Align columns
X = X[feature_cols]
X_test = X_test[feature_cols]

# --- Model Training (LightGBM) ---

def train_lgbm(X, y, X_test, seeds):
    test_preds = np.zeros(X_test.shape[0])
    
    for seed in seeds:
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'n_estimators': 100, 
            'learning_rate': 0.05,
            'num_leaves': 10,
            'max_depth': 3,
            'seed': seed,
            'n_jobs': -1,
            'verbose': -1,
            'colsample_bytree': 0.7,
            'subsample': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }
        
        model = lgb.LGBMClassifier(**params)
        
        # Train on the entire small dataset
        model.fit(X, y)
        
        # Predict probabilities
        test_preds += model.predict_proba(X_test)[:, 1] / len(seeds)
        
    return test_preds

# Train and predict
test_predictions = train_lgbm(X, y, X_test, SEEDS)

# --- Submission Generation ---
submission_df = pd.DataFrame({
    ID_COL: df_test[ID_COL],
    TARGET_COL: test_predictions
})

# Save the submission file
submission_df.to_csv(SUBMISSION_PATH, index=False)