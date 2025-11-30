import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# --- Configuration ---
SEEDS = [0, 1, 2]
N_SPLITS = 5
TARGET_COLUMN = 'target'
ID_COLUMN = 'id'

def load_data(dataset_path):
    """Loads training and testing data."""
    train_path = os.path.join(dataset_path, 'train.csv')
    test_path = os.path.join(dataset_path, 'test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Required files (train.csv, test.csv) not found in {dataset_path}")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df

def feature_engineering(df):
    """Decomposes the string feature f_27 into 10 individual character columns."""
    if 'f_27' in df.columns:
        # Ensure f_27 is treated as string before decomposition
        f27_series = df['f_27'].astype(str)
        for i in range(10):
            df[f'f_27_{i}'] = f27_series.str[i]
        df = df.drop('f_27', axis=1)
    return df

def preprocess(train_df, test_df):
    """Handles feature engineering and categorical encoding."""
    
    # 1. Feature Engineering
    X = feature_engineering(train_df.drop(columns=[ID_COLUMN, TARGET_COLUMN]))
    y = train_df[TARGET_COLUMN]
    train_ids = train_df[ID_COLUMN]
    
    X_test = feature_engineering(test_df.drop(columns=[ID_COLUMN]))
    test_ids = test_df[ID_COLUMN]
    
    # Align columns
    common_cols = list(X.columns.intersection(X_test.columns))
    X = X[common_cols]
    X_test = X_test[common_cols]
    
    # 2. Identify categorical features
    
    # Low cardinality integers (f_07 to f_18, f_29, f_30)
    int_cat_cols = [col for col in X.columns if X[col].dtype == 'int64' and X[col].nunique() < 10]
    
    # Engineered string features (f_27_0 to f_27_9)
    str_cat_cols = [col for col in X.columns if X[col].dtype == 'object']
    
    cat_features = int_cat_cols + str_cat_cols
    
    # 3. Encoding Categorical Features (Ordinal Encoding using LabelEncoder)
    # We fit the encoder on combined train/test unique values to prevent unseen labels in test set.
    
    for col in cat_features:
        # Convert to string to handle mixed types (int/object) consistently
        X[col] = X[col].astype(str)
        X_test[col] = X_test[col].astype(str)
        
        # Fit LE on combined unique values
        all_values = pd.concat([X[col], X_test[col]]).unique()
        le = LabelEncoder()
        le.fit(all_values)
        
        # Transform
        X[col] = le.transform(X[col])
        X_test[col] = le.transform(X_test[col])
        
        # Convert back to integer type for LGBM categorical handling
        X[col] = X[col].astype('int32')
        X_test[col] = X_test[col].astype('int32')
        
    return X, X_test, y, train_ids, test_ids, cat_features

def train_and_predict(X, X_test, y, cat_features, test_ids):
    """Trains LightGBM models using Stratified K-Fold and multiple seeds."""
    
    test_preds = np.zeros(X_test.shape[0])
    
    # LightGBM parameters for binary classification
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'n_jobs': -1,
        'max_depth': 7,
        'min_child_samples': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
    }
    
    for seed in SEEDS:
        lgb_params['seed'] = seed
        
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        fold_test_preds = np.zeros(X_test.shape[0])
        
        for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            
            model = lgb.LGBMClassifier(**lgb_params)
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(100, verbose=False)],
                categorical_feature=cat_features
            )
            
            # Test prediction
            fold_test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
        
        test_preds += fold_test_preds / len(SEEDS)
        
    return test_preds

def create_submission(test_ids, predictions):
    """Creates the submission file."""
    submission_df = pd.DataFrame({
        ID_COLUMN: test_ids,
        TARGET_COLUMN: predictions
    })
    submission_df.to_csv('submission.csv', index=False)
    print("Submission file created successfully.")

def run_pipeline(dataset_path):
    """Main pipeline execution function."""
    print(f"Starting pipeline for dataset at {dataset_path}")
    
    try:
        train_df_raw, test_df_raw = load_data(dataset_path)
    except FileNotFoundError as e:
        print(e)
        return

    print("Preprocessing and Feature Engineering...")
    X, X_test, y, train_ids, test_ids, cat_features = preprocess(train_df_raw.copy(), test_df_raw.copy())
    
    print(f"Training LGBM model using {len(SEEDS)} seeds and {N_SPLITS} folds...")
    print(f"Total features: {X.shape[1]}. Categorical features: {len(cat_features)}")
    
    predictions = train_and_predict(X, X_test, y, cat_features, test_ids)
    
    print("Creating submission file...")
    create_submission(test_ids, predictions)

# --- Execution Environment Setup ---

# Determine dataset path dynamically based on environment variable or command line argument
dataset_path = os.environ.get('DATASET_PATH')
if not dataset_path and len(sys.argv) > 1:
    dataset_path = sys.argv[1]
elif not dataset_path:
    # Fallback to current directory if path is not provided
    dataset_path = '.'

if __name__ == '__main__':
    run_pipeline(dataset_path)