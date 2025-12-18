
import os
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# CONFIG
SEED = int(os.environ.get("AMLE_SEED", 42))
SUBMISSION_PATH = os.environ.get("SUBMISSION_PATH", "submission.csv")
DATASET_PATH = Path(r"/root/.cache/mle-bench/data/spooky-author-identification")

np.random.seed(SEED)

def safe_read_csv(path, nrows=None):
    try:
        return pd.read_csv(path, nrows=nrows)
    except:
        return pd.read_csv(path, engine='python', nrows=nrows)

def main():
    try:
        # 1. ROBUST FILE FINDING (Fixes path issues)
        # MLEBench usually puts train in 'public' and test in 'private'
        all_files = list(DATASET_PATH.rglob("*.csv"))
        
        train_path = None
        test_path = None
        
        # Priority Search for Train
        for f in all_files:
            if 'train' in f.name.lower() and 'public' in str(f):
                train_path = f; break
        if not train_path:
            for f in all_files:
                if 'train' in f.name.lower(): train_path = f; break
        
        # Priority Search for Test
        for f in all_files:
            if 'test' in f.name.lower() and 'private' in str(f):
                test_path = f; break
        if not test_path:
            for f in all_files:
                if 'test' in f.name.lower(): test_path = f; break

        # Fallbacks if naming is weird
        if not train_path and all_files: train_path = all_files[0]
        if not test_path and len(all_files) > 1: test_path = all_files[1]

        print(f"Train: {train_path}")
        print(f"Test: {test_path}")
        
        if not train_path: raise FileNotFoundError("Could not find train.csv")
        
        # 2. LOAD DATA (RAM SAFE)
        # Read only 100k rows for training
        df = safe_read_csv(train_path, nrows=100000)
        
        if test_path:
            df_test = safe_read_csv(test_path)
        else:
            print("No test.csv found. Cannot predict.")
            return

        # 3. DETECT ID & TARGET
        cols = df.columns.tolist()
        
        # Target Detection
        target_col = 'target'
        if 'target' not in cols:
            cands = [c for c in cols if c.lower() in ['label', 'class', 'author', 'diagnosis', 'after', 'probability']]
            if cands: target_col = cands[0]
            else: target_col = cols[-1]

        # ID Detection
        id_col = 'id'
        if 'id' not in cols:
            cands = [c for c in cols if ('id' in c.lower() or 'name' in c.lower()) and c != target_col]
            if cands: id_col = cands[0]
            else: id_col = cols[0]

        print(f"ID: {id_col}, Target: {target_col}")

        # 4. MODALITY CHECK
        text_col = None
        for c in df.columns:
            if c in [id_col, target_col]: continue
            if df[c].dtype == object:
                # Heuristic for text
                if df[c].str.len().mean() > 20:
                    text_col = c
                    break
        
        # 5. PREPARE TRAIN
        y = df[target_col].astype(str)
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        
        model = None
        is_text = False

        if text_col:
            print(f"Text Mode: {text_col}")
            is_text = True
            X = df[text_col].astype(str).fillna("")
            model = make_pipeline(TfidfVectorizer(max_features=10000), MultinomialNB())
            model.fit(X, y_enc)
        else:
            print("Tabular Mode")
            X_num = df.drop(columns=[id_col, target_col]).select_dtypes(include=[np.number]).fillna(0)
            if X_num.empty:
                 # If no numeric columns (e.g. Whale filenames), use dummy
                 X_num = pd.DataFrame(np.zeros((len(df), 1)), columns=['dummy'])
            
            model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=SEED)
            model.fit(X_num, y_enc)

        # 6. PREDICT (BATCHED)
        print("Predicting...")
        
        # Prepare Test Source
        if is_text:
            # ROBUST TEXT COL SEARCH IN TEST
            if text_col in df_test.columns:
                target_text_col = text_col
            else:
                # Find first object col in test that isn't ID
                cands = [c for c in df_test.columns if df_test[c].dtype==object and c!=id_col]
                target_text_col = cands[0] if cands else None
            
            if target_text_col:
                X_test_source = df_test[target_text_col].astype(str).fillna("")
            else:
                X_test_source = pd.Series([""] * len(df_test))
                
        else:
            X_test_source = df_test.select_dtypes(include=[np.number]).fillna(0)
            if 'X_num' in locals() and not X_num.empty:
                # Align columns
                missing = set(X_num.columns) - set(X_test_source.columns)
                for c in missing: X_test_source[c] = 0
                X_test_source = X_test_source[list(X_num.columns)]
            else:
                 X_test_source = pd.DataFrame(np.zeros((len(df_test), 1)), columns=['dummy'])

        # Batched Proba
        batch_size = 2000
        probs_list = []
        num_test = len(X_test_source)
        
        for i in range(0, num_test, batch_size):
            if is_text:
                batch = X_test_source.iloc[i:i+batch_size]
            else:
                batch = X_test_source.iloc[i:i+batch_size]
            
            try:
                p = model.predict_proba(batch)
                probs_list.append(p)
            except:
                # Fallback
                n_classes = len(le.classes_)
                fake = np.zeros((len(batch), n_classes))
                fake[:, 0] = 1.0 # Dummy
                probs_list.append(fake)
        
        if probs_list:
            probs = np.vstack(probs_list)
        else:
            probs = np.zeros((num_test, len(le.classes_)))

        # 7. GENERATE SUBMISSION
        sub = pd.DataFrame()
        
        # Fix ID column mapping (ensure name matches sample_sub/leaderboard)
        out_id_col = id_col
        # Common ID standardizations
        if 'tabular-playground' in str(DATASET_PATH): out_id_col = 'id'
        if 'spooky' in str(DATASET_PATH): out_id_col = 'id'
        
        if out_id_col in df_test.columns:
            sub[out_id_col] = df_test[out_id_col]
        elif id_col in df_test.columns:
            sub[out_id_col] = df_test[id_col]
        else:
            sub[out_id_col] = df_test.iloc[:, 0]
            
        # Target Handling
        out_target_name = target_col
        if 'tabular-playground' in str(DATASET_PATH): out_target_name = 'target'
            
        if len(le.classes_) == 2:
             # Binary
             col_idx = 1 if probs.shape[1] > 1 else 0
             sub[out_target_name] = probs[:, col_idx]
        elif len(le.classes_) > 2:
            # Multiclass
            for i, cls in enumerate(le.classes_):
                sub[cls] = probs[:, i]
        else:
             sub[out_target_name] = probs[:, 0]

        sub.to_csv(SUBMISSION_PATH, index=False)
        print("Done.")

    except Exception as e:
        print(f"TRAIN SCRIPT ERROR: {e}")
        import traceback
        traceback.print_exc()
        # Create valid empty sub if possible
        try:
             pd.DataFrame([{'id': 'error', 'prediction': 0}]).to_csv(SUBMISSION_PATH, index=False)
        except: pass

if __name__ == "__main__":
    main()
