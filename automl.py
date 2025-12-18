#!/usr/bin/env python3

from __future__ import annotations
import os
import sys
import re
import json
import time
import argparse
import subprocess
import traceback
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

try:
    import pandas as pd
    import numpy as np
except Exception:
    print("Missing core python packages (pandas/numpy). Please install them. Exiting.")
    raise


try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

# Environment defaults
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-preview-09-2025")
DEFAULT_SUBPROCESS_TIMEOUT = int(os.getenv("AMLE_SUBPROCESS_TIMEOUT", "3600"))
DEFAULT_MAX_RETRIES = int(os.getenv("AMLE_MAX_RETRIES", "3"))
AGENT_LOG_PATH = os.getenv("AMLE_LOG_PATH", "agent_trace.log")

COMPETITION_MAP = {
    "melanoma": "siim-isic-melanoma-classification",
    "spooky": "spooky-author-identification",
    "tabular": "tabular-playground-series-may-2022",
    "text": "text-normalization-challenge-english-language",
    "whale": "the-icml-2013-whale-challenge-right-whale-redux",
}


# Optional Google GenAI client
genai_available = False
genai_client = None
try:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore

    genai_available = True
    if GEMINI_API_KEY:
        try:
            genai_client = genai.Client(api_key=GEMINI_API_KEY)
        except Exception:
            genai_client = None
except Exception:
    genai_available = False
    genai_client = None

DEFAULT_SEEDS = [0, 1, 2]


# -------------------------
# Simple logger
# -------------------------
class Logger:
    def __init__(self, path: str = AGENT_LOG_PATH):
        self.path = Path(path)
        self._lines: List[str] = []

    def log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        self._lines.append(line)

    def dump(self):
        try:
            self.path.write_text("\n".join(self._lines), encoding="utf-8")
        except Exception as e:
            print(f"Failed to write log file {self.path}: {e}")


logger = Logger()


# -------------------------
# System-level prompt for LLM
# -------------------------
SYSTEM_PROMPT = r"""
You are an autonomous machine learning agent.
Your job is to read an arbitrary dataset directory, detect its structure, determine the correct modeling strategy, train a model, and produce a valid submission.csv.
You must operate without user hints, without competition hardcoding, and without guessing.
Preserve ID exactly.
CRITICAL: You must read the Random Seed from the environment variable 'AMLE_SEED'.
CRITICAL: You must write the final submission file to the path in environment variable 'SUBMISSION_PATH'.
"""


# -------------------------
# AutoMLEAgent
# -------------------------
class AutoMLEAgent:
    def __init__(
        self,
        dataset_path: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: int = DEFAULT_SUBPROCESS_TIMEOUT,
    ):
        self.dataset_path = Path(dataset_path).resolve()
        self.max_retries = int(max_retries)
        self.timeout = int(timeout)
        self.logs = logger
        self.llm_client = genai_client if genai_available and genai_client else None
        self.system_prompt = SYSTEM_PROMPT
        self._analysis_cache: Optional[Dict[str, Any]] = None

        # --- Extract Dataset Name for dynamic file naming ---
        self.dataset_name = self.dataset_path.name

        self.competition_name = COMPETITION_MAP.get(
            self.dataset_name, self.dataset_name
        )

        self.results_root = Path("mlebench_results") / self.competition_name
        self.results_root.mkdir(parents=True, exist_ok=True)

        self.logs.log(
            f"Agent initialized. dataset_path={self.dataset_path}, max_retries={self.max_retries}, timeout={self.timeout}"
        )
        if self.llm_client:
            self.logs.log("LLM client available.")
        else:
            self.logs.log("LLM client NOT available; will use fallback templates.")

    # -------------------------
    # Analyze dataset
    # -------------------------
    def analyze_data(self) -> Dict[str, Any]:
        self.logs.log("Starting dataset analysis.")
        if not self.dataset_path.exists():
            msg = f"Dataset path not found: {self.dataset_path}"
            self.logs.log(msg)
            return {"error": msg, "path_exists": False}

        files = list(self.dataset_path.iterdir())
        summary = {
            "path": str(self.dataset_path),
            "num_files": len(files),
            "files": [p.name for p in files],
            "csvs": [],
            "images": [],
            "audio": [],
            "others": [],
            "csv_details": {},
            "path_exists": True,
        }

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        audio_exts = {".aif", ".aiff", ".wav", ".mp3", ".flac", ".ogg"}
        csvs: List[Path] = []

        for p in files:
            if p.is_file():
                ext = p.suffix.lower()
                if ext == ".csv":
                    csvs.append(p)
                    summary["csvs"].append(p.name)
                elif ext in image_exts:
                    summary["images"].append(p.name)
                elif ext in audio_exts:
                    summary["audio"].append(p.name)
                else:
                    summary["others"].append(p.name)

        for csv_path in csvs:
            try:
                df = pd.read_csv(csv_path, nrows=50)
                cols = list(df.columns)
                sample = df.head(5).to_dict(orient="records")
                id_candidates = [
                    c
                    for c in cols
                    if re.search(r"\b(id|image|filename|file)\b", c, re.I)
                ]
                target_candidates = [
                    c
                    for c in cols
                    if re.search(r"\b(target|label|class|diagnosis|y)\b", c, re.I)
                ]
                unique_counts = {c: int(df[c].nunique(dropna=False)) for c in cols}
                types = {c: str(df[c].dtype) for c in cols}
                summary["csv_details"][csv_path.name] = {
                    "columns": cols,
                    "sample_rows": sample,
                    "id_candidates": id_candidates,
                    "target_candidates": target_candidates,
                    "unique_counts": unique_counts,
                    "dtypes": types,
                }
            except Exception as e:
                summary["csv_details"][csv_path.name] = {"error": str(e)}

        self._analysis_cache = summary
        self.logs.log("Dataset analysis complete.")
        return summary

    # -------------------------
    # Extract code from LLM response text
    # -------------------------
    def _extract_code_from_text(self, text: str) -> str:
        if not text:
            return ""
        matches = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.S | re.I)
        if matches:
            matches.sort(key=len, reverse=True)
            return matches[0].strip()
        return text.strip()

    # -------------------------
    # Generate code via LLM or fallback
    # -------------------------
    def generate_code(
        self, context: Dict[str, Any], error_trace: Optional[str] = None
    ) -> str:
        if not self.llm_client:
            self.logs.log("LLM unavailable: returning fallback template.")
            return self.fallback_template(
                context=context, include_reason="llm_unavailable"
            )

        # Build prompt
        user_msg = f"""
Dataset summary (JSON):
{json.dumps(context, indent=2)}

Instructions:
- Produce a standalone Python script that reads data from `dataset_path` variable or DATASET_PATH env var.
- Detect ID and target columns automatically and preserve ID column exactly in submission.csv.
- Train a reasonable model depending on modality (detect text/image/tabular).
- IMPORTANT: Use `int(os.environ.get("AMLE_SEED", 42))` as the random seed.
- IMPORTANT: Save the submission dataframe to `os.environ.get("SUBMISSION_PATH", "submission.csv")`.
- If earlier attempts failed, error trace is provided.

ERROR_TRACE:
{error_trace if error_trace else "<none>"}

Return ONLY python code in a code block.
"""

        # --- ROBUST RETRY LOGIC FOR QUOTA/RATE LIMITS ---
        max_quota_retries = 5
        base_retry_delay = 60  # Start with 60 seconds

        for attempt in range(max_quota_retries + 1):
            try:
                self.logs.log(
                    f"Requesting code generation from LLM (Attempt {attempt+1})."
                )
                response = self.llm_client.models.generate_content(
                    model=MODEL_NAME,
                    contents=user_msg,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_prompt,
                        temperature=0.12,
                    ),
                )
                raw_text = getattr(response, "text", None) or str(response)
                code = self._extract_code_from_text(raw_text)
                if not code:
                    self.logs.log("LLM returned no code; using fallback.")
                    return self.fallback_template(
                        context=context, include_reason="llm_empty"
                    )
                # ensure dataset_path exists in script
                if "dataset_path" not in code:
                    header = f"from pathlib import Path\ndataset_path = r'''{str(self.dataset_path)}'''\n"
                    code = header + "\n" + code
                self.logs.log("LLM code generation completed.")
                return code

            except Exception as e:
                error_str = str(e)
                # Check for 429/Resource Exhausted errors
                if (
                    "429" in error_str
                    or "RESOURCE_EXHAUSTED" in error_str
                    or "Quota exceeded" in error_str
                ):
                    if attempt < max_quota_retries:
                        sleep_time = base_retry_delay * (
                            2**attempt
                        )  # Exponential backoff
                        self.logs.log(
                            f"WARNING: API Quota Exceeded (429). Sleeping for {sleep_time}s before retry..."
                        )
                        time.sleep(sleep_time)
                        continue  # Retry the loop
                    else:
                        self.logs.log("Max quota retries exhausted. Using Fallback.")

                # If not a quota error or max retries hit, proceed to fallback
                self.logs.log(f"LLM generation error: {e}")
                return self.fallback_template(
                    context=context, include_reason=f"llm_exception: {str(e)[:200]}"
                )

        return self.fallback_template(context, "unknown_failure")

    # -------------------------
    # Deterministic fallback script (SMARTER)
    # -------------------------
    def fallback_template(
        self, context: Optional[Dict[str, Any]] = None, include_reason: str = "fallback"
    ) -> str:
        """
        Returns a standalone Python script (string) that handles Tabular AND Text data.
        """
        dataset_path_str = str(self.dataset_path)
        reason_comment = f"# Fallback generated due to: {include_reason}\n"

        script = f"""{reason_comment}
# Auto-generated fallback training script (Robust Tabular + Text)
import os
import sys
from pathlib import Path
import traceback
import pandas as pd
import numpy as np
import re
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

SEED = int(os.getenv("AMLE_SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)

dataset_path = Path(r\"\"\"{dataset_path_str}\"\"\")
submission_path = os.getenv("SUBMISSION_PATH", "submission.csv")

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine='python')

def find_csv():
    prefer = ['train.csv', 'training.csv', 'train_labels.csv']
    for name in prefer:
        p = dataset_path / name
        if p.exists(): return p
    csvs = list(dataset_path.glob('*.csv'))
    if not csvs: return None
    csvs.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    return csvs[0]

def detect_columns(df):
    cols = list(df.columns)
    id_candidates = [c for c in cols if c and re.search(r"\\b(id|idx|image|file)\\b", c, re.I)]
    
    unique_ids = []
    for c in cols:
        try:
            if df[c].nunique(dropna=False) == len(df): unique_ids.append(c)
        except: continue
        
    candidates = [c for c in id_candidates if c in cols]
    for c in unique_ids:
        if c in cols and c not in candidates: candidates.append(c)
        
    id_col = candidates[0] if candidates else (cols[0] if cols else 'id')
    
    target_candidates = [c for c in cols if c and re.search(r"\\b(target|label|class|author)\\b", c, re.I)]
    target_col = target_candidates[0] if target_candidates else (cols[-1] if cols and cols[-1] != id_col else None)
    
    return id_col, target_col

def main():
    try:
        csv_path = find_csv()
        if not csv_path: 
            pd.DataFrame([{{'id': 'error', 'prediction': 0}}]).to_csv(submission_path, index=False)
            return

        df = safe_read_csv(csv_path)
        id_col, target_col = detect_columns(df)
        print(f"Detected id={{id_col}}, target={{target_col}}")

        if not target_col:
            # Create dummy zero submission
            out = pd.DataFrame({{id_col: df[id_col], 'prediction': 0}})
            out.to_csv(submission_path, index=False)
            return

        # --- MODALITY DETECTION ---
        # Check for long text columns
        text_col = None
        for col in df.columns:
            if col in [id_col, target_col]: continue
            if df[col].dtype == object:
                # Heuristic: Average length > 20 chars implies text feature
                try:
                    mean_len = df[col].astype(str).str.len().mean()
                    if mean_len > 20:
                        text_col = col
                        break
                except: continue

        X = df.drop(columns=[id_col, target_col])
        y = df[target_col]
        
        # Handle Target Encoding
        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))
        
        model = None
        
        # --- TRAIN ---
        if text_col:
            print(f"Text modality detected in column: {{text_col}}. Using TF-IDF + Naive Bayes.")
            # Use only the text column
            X_text = df[text_col].astype(str).fillna("")
            X_train, X_val, y_train, y_val = train_test_split(X_text, y_enc, test_size=0.2, random_state=SEED)
            
            model = make_pipeline(TfidfVectorizer(max_features=5000), MultinomialNB())
            model.fit(X_train, y_train)
            
            # Predict
            preds_prob = model.predict_proba(df[text_col].astype(str).fillna(""))
        
        else:
            print("Tabular modality detected. Using Random Forest.")
            # Simple tabular prep
            X_num = X.select_dtypes(include=[np.number]).fillna(0)
            # Encode cats
            X_cat = X.select_dtypes(include=[object])
            if not X_cat.empty:
                oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                X_cat_enc = oe.fit_transform(X_cat.astype(str))
                X_final = np.hstack([X_num.values, X_cat_enc])
            else:
                X_final = X_num.values
                
            X_train, X_val, y_train, y_val = train_test_split(X_final, y_enc, test_size=0.2, random_state=SEED)
            
            model = RandomForestClassifier(n_estimators=100, random_state=SEED)
            model.fit(X_train, y_train)
            preds_prob = model.predict_proba(X_final)

        # --- SUBMISSION GENERATION ---
        sample_path = dataset_path / 'sample_submission.csv'
        out_df = pd.DataFrame()
        out_df[id_col] = df[id_col]
        
        if sample_path.exists():
            sample = safe_read_csv(sample_path)
            target_cols = [c for c in sample.columns if c != id_col and c != 'id']
            if len(target_cols) == len(le.classes_):
                for i, cls in enumerate(le.classes_):
                    if cls in target_cols:
                        out_df[cls] = preds_prob[:, i]
                    else:
                        out_df[target_cols[i]] = preds_prob[:, i]
            else:
                out_df['prediction'] = preds_prob[:, 1] if preds_prob.shape[1] > 1 else preds_prob[:, 0]
                
            # CHECK FOR TEST SET
            test_path = dataset_path / 'test.csv'
            if test_path.exists():
                test_df = safe_read_csv(test_path)
                out_df = pd.DataFrame()
                out_df[id_col] = test_df[id_col]
                
                if text_col:
                    test_preds = model.predict_proba(test_df[text_col].astype(str).fillna(""))
                else:
                    t_num = test_df.select_dtypes(include=[np.number]).fillna(0)
                    t_cat = test_df.select_dtypes(include=[object])
                    if not t_cat.empty:
                        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                        t_cat_enc = oe.fit_transform(t_cat.astype(str))
                        t_final = np.hstack([t_num.values, t_cat_enc])
                    else:
                        t_final = t_num.values
                    try:
                        test_preds = model.predict_proba(t_final)
                    except:
                        test_preds = np.zeros((len(test_df), len(le.classes_)))

                if len(target_cols) == len(le.classes_):
                    for i, cls in enumerate(le.classes_):
                        if cls in target_cols:
                            out_df[cls] = test_preds[:, i]
                        else:
                            out_df[target_cols[i]] = test_preds[:, i]
        else:
            out_df['prediction'] = preds_prob[:, 1] if preds_prob.shape[1] > 1 else preds_prob[:, 0]

        out_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {{submission_path}}")

    except Exception as e:
        print(e)
        traceback.print_exc()
        pd.DataFrame([{{'id': 'error', 'prediction': 0}}]).to_csv(submission_path, index=False)

if __name__ == '__main__':
    main()
"""
        return script

    # -------------------------
    # Write script to disk
    # -------------------------

    def _write_script(self, code: str, path: str) -> Path:
        p = Path(path)
        p.write_text(code, encoding="utf-8")
        self.logs.log(f"Wrote script to {p.resolve()}")
        return p

    # -------------------------
    # Execute script in subprocess
    # -------------------------
    def execute_script(
        self, script_path: Path, env_vars: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        env = env_vars if env_vars else os.environ.copy()
        env["DATASET_PATH"] = str(self.dataset_path)

        self.logs.log(f"Executing script {script_path} with timeout {self.timeout}s")
        try:
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            output = f"RETURN CODE: {completed.returncode}\n\nSTDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
            success = completed.returncode == 0
            if success:
                self.logs.log("Script executed with returncode 0")
            else:
                self.logs.log(f"Script executed with returncode {completed.returncode}")
            return success, output
        except subprocess.TimeoutExpired as e:
            self.logs.log("Script execution timed out.")
            out = f"Timeout after {self.timeout} seconds."
            return False, out
        except Exception as e:
            tb = traceback.format_exc()
            self.logs.log(f"Script execution failed: {e}")
            return False, f"Execution failed: {e}\n{tb}"

    # -------------------------
    # Main run loop
    # -------------------------
    def run(self):
        summary = self.analyze_data()

        if not summary.get("path_exists", True):
            self._write_emergency_submission()
            self.logs.dump()
            return

        # 1. OPTIMIZATION: Generate Code ONCE, Run 3 Times
        self.logs.log("Generating Master Training Script (reused for all seeds)...")
        code = self.generate_code(summary)

        # Save explicitly as train_script.py in the competition results folder
        script_path = self.results_root / "train_script.py"
        self._write_script(code, str(script_path))

        for seed in DEFAULT_SEEDS:
            self.logs.log(f"=== Running seed {seed} ===")

            seed_dir = self.results_root / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            submission_path = seed_dir / "submission.csv"

            # Prepare Environment Variables for this specific run
            run_env = os.environ.copy()
            run_env["AMLE_SEED"] = str(seed)
            run_env["SUBMISSION_PATH"] = str(submission_path)
            run_env["DATASET_PATH"] = str(self.dataset_path)

            # Execute the Master Script
            success, out = self.execute_script(script_path, env_vars=run_env)

            if not success:
                self.logs.log(
                    f"Seed {seed} failed. Attempting emergency fallback write."
                )
                self._write_emergency_submission(str(submission_path))

        self.logs.dump()

    # -------------------------
    # Emergency submission writer
    # -------------------------
    def _write_emergency_submission(self, submission_filename: str = "submission.csv"):
        try:
            pd.DataFrame([{"id": "emergency", "prediction": 0}]).to_csv(
                submission_filename, index=False
            )
            self.logs.log(
                f"Emergency submission.csv written with single placeholder row to {submission_filename}."
            )
        except Exception as e:
            self.logs.log(f"Failed to write emergency submission.csv at all: {e}")


# -------------------------
# CLI entrypoint
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="AutoMLEAgent - analyze dataset and produce submission.csv"
    )
    parser.add_argument(
        "--dataset_path", required=True, help="Path to the dataset directory"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retries for code generation",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_SUBPROCESS_TIMEOUT,
        help="Timeout (s) for script execution",
    )
    args = parser.parse_args()

    agent = AutoMLEAgent(
        dataset_path=args.dataset_path,
        max_retries=args.max_retries,
        timeout=args.timeout,
    )
    agent.run()


if __name__ == "__main__":
    main()
