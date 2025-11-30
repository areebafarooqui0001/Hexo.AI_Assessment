from pathlib import Path
dataset_path = r'''/mnt/c/Users/areeb/Downloads/Areeba/Hexo_AI.Assessment/data/text'''

import pandas as pd
import os
import sys

# Set up paths
# Read DATASET_PATH from environment variable or use current directory as fallback
DATASET_PATH = os.environ.get('DATASET_PATH', '.')
SUBMISSION_PATH = os.environ.get('SUBMISSION_PATH', 'submission.csv')

# --- Data Loading ---
try:
    train_path = os.path.join(DATASET_PATH, 'en_train.csv')
    test_path = os.path.join(DATASET_PATH, 'en_test_2.csv')

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
except FileNotFoundError as e:
    print(f"Error loading data: {e}", file=sys.stderr)
    sys.exit(1)

# --- ID Generation ---
# Create a composite ID based on sentence_id and token_id for submission
df_test['id'] = df_test['sentence_id'].astype(str) + '_' + df_test['token_id'].astype(str)
ID_COLUMN = 'id'

# --- Modeling Strategy: Lookup Table + Identity Fallback ---
# This is a Text Normalization task (predicting 'after' from 'before').
# Given the small size and specialized nature, a lookup table derived from training data 
# combined with identity mapping for unknown tokens is the most robust baseline.

# 1. Create the lookup map (before -> after). 
def get_most_frequent_after(group):
    """Returns the most frequent 'after' value for a given 'before' token."""
    # Handles potential conflicts by choosing the most frequent mapping
    return group.value_counts().idxmax()

# Group by 'before' and aggregate to create the deterministic lookup map
lookup_map = df_train.groupby('before')['after'].agg(get_most_frequent_after).to_dict()

# 2. Define the prediction function
def predict_after(row, lookup):
    """Applies lookup or falls back to identity mapping."""
    before = row['before']
    
    # Handle missing values
    if pd.isna(before):
        return '' 
    
    # Check lookup table
    if before in lookup:
        return lookup[before]
    else:
        # Identity mapping (fallback for unknown tokens, typically PLAIN class)
        return before

# 3. Apply prediction
df_test['after'] = df_test.apply(lambda row: predict_after(row, lookup_map), axis=1)

# --- Submission Generation ---
submission_df = df_test[[ID_COLUMN, 'after']]
submission_df.columns = ['id', 'after'] # Ensure column names match expected format

# Write the submission file
submission_df.to_csv(SUBMISSION_PATH, index=False)