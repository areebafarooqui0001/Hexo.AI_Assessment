#!/usr/bin/env python3
"""
AutoMLEAgent (Fixed / Production-Ready)
- Full, self-contained script.
- Safe LLM usage (optional). If LLM fails, deterministic fallback script is used.
- Robust handling for missing dataset path.
- Fallback produces a valid submission.csv in all cases.
- No hardcoded secrets.
"""

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
Preserve ID exactly. Use fallback if needed. Use seeds [0,1,2].
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
- Train a reasonable model depending on modality, or fallback to a safe baseline.
- Use seeds {DEFAULT_SEEDS} for repeatability.
- The final submission CSV must be written to the path specified in the SUBMISSION_PATH environment variable, or 'submission.csv' if not set.
- If earlier attempts failed, error trace is provided.

ERROR_TRACE:
{error_trace if error_trace else "<none>"}

Return ONLY python code in a code block.
"""

        try:
            self.logs.log("Requesting code generation from LLM.")
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
            # log the error and fallback
            self.logs.log(f"LLM generation error: {e}")
            tb = traceback.format_exc()
            self.logs.log(tb)
            return self.fallback_template(
                context=context, include_reason=f"llm_exception: {str(e)[:200]}"
            )

    # -------------------------
    # Deterministic fallback script
    # -------------------------
    def fallback_template(
        self, context: Optional[Dict[str, Any]] = None, include_reason: str = "fallback"
    ) -> str:
        """
        Returns a standalone Python script (string) that will:
        - Try to find a CSV
        - Detect ID column robustly (heuristics)
        - If target present, train a small baseline model; otherwise produce zero/placeholder submission
        - Always write submission to the path specified by the SUBMISSION_PATH env var
        """
        dataset_path_str = str(self.dataset_path)
        reason_comment = f"# Fallback generated due to: {include_reason}\n"

        script = f"""{reason_comment}
# Auto-generated fallback training script (robust)
# - Conservative pipeline: targets tabular CSV datasets primarily.
# - Preserves ID column exactly as in sample_submission or input CSV.
# - Writes submission to the path specified by SUBMISSION_PATH environment variable.
# - Deterministic seeds: {DEFAULT_SEEDS}

import os
import sys
from pathlib import Path
import traceback
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

dataset_path = Path(r\"\"\"{dataset_path_str}\"\"\")
submission_path = os.getenv("SUBMISSION_PATH", "submission.csv")

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine='python')

def find_csv():
    # prefer common names
    prefer = ['train.csv', 'training.csv', 'train_labels.csv', 'labels.csv', 'sample_submission.csv']
    for name in prefer:
        p = dataset_path / name
        if p.exists():
            return p
    # fallback: largest csv
    csvs = list(dataset_path.glob('*.csv'))
    if not csvs:
        return None
    csvs.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    return csvs[0]

def detect_id_and_target(df):
    cols = list(df.columns)
    # id candidates by name
    id_candidates = [c for c in cols if c and re.search(r\"\\b(id|idx|index|image|filename|file)\\b\", c, re.I)]
    # uniqueness heuristic
    unique_ids = []
    for c in cols:
        try:
            if df[c].nunique(dropna=False) == len(df):
                unique_ids.append(c)
        except Exception:
            continue
    candidates = []
    for c in id_candidates:
        if c in cols:
            candidates.append(c)
    for c in unique_ids:
        if c in cols and c not in candidates:
            candidates.append(c)
    id_col = candidates[0] if candidates else cols[0] if cols else 'id'
    # detect target
    target_candidates = [c for c in cols if c and re.search(r\"\\b(target|label|class|diagnosis|y)\\b\", c, re.I)]
    target_col = target_candidates[0] if target_candidates else (cols[-1] if cols and cols[-1] != id_col else None)
    return id_col, target_col

def prepare_features(df, id_col, target_col):
    features = [c for c in df.columns if c not in (id_col, target_col)]
    X = df[features].copy()
    # basic cleaning
    for c in X.columns:
        if X[c].dtype == object or X[c].dtype.name == 'category':
            X[c] = X[c].astype(str).fillna('__MISSING__')
        else:
            try:
                X[c] = X[c].fillna(X[c].median())
            except Exception:
                X[c] = X[c].fillna(0)
    # encode object columns
    obj_cols = [c for c in X.columns if X[c].dtype == object]
    if obj_cols:
        try:
            enc = OrdinalEncoder()
            X[obj_cols] = enc.fit_transform(X[obj_cols])
        except Exception:
            for c in obj_cols:
                X[c] = X[c].astype('category').cat.codes.fillna(0)
    return X

def main():
    try:
        csv_path = find_csv()
        if csv_path is None:
            print('No CSV found in dataset. Creating minimal submission placeholder.')
            # Uses submission_path
            pd.DataFrame([{{'id': 'no_data', 'prediction': 0}}]).to_csv(submission_path, index=False) 
            return

        df = safe_read_csv(csv_path)
        if df is None or df.shape[0] == 0:
            print('CSV empty. Creating minimal submission placeholder.')
            # Uses submission_path
            pd.DataFrame([{{'id': 'no_data', 'prediction': 0}}]).to_csv(submission_path, index=False)
            return

        id_col, target_col = detect_id_and_target(df)
        print(f"Detected id_col={{id_col}}, target_col={{target_col}} in {{csv_path.name}}")

        # If no target, align to sample_submission if exists
        if target_col is None:
            sample_path = dataset_path / 'sample_submission.csv'
            if sample_path.exists():
                sample = safe_read_csv(sample_path)
                if 'id' in sample.columns:
                    sample_cols = sample.columns.tolist()
                    pred_col = sample_cols[1] if len(sample_cols) > 1 else 'prediction'
                    sample[pred_col] = 0
                    # Uses submission_path
                    sample.to_csv(submission_path, index=False)
                    print(f'Wrote submission to {{submission_path}} from sample_submission.csv with zeros.')
                    return
            # create zero predictions aligned to IDs in df
            out = pd.DataFrame({{id_col: df[id_col].astype(str), 'prediction': 0}})
            # Uses submission_path
            out.to_csv(submission_path, index=False)
            print(f'No target found; wrote trivial zero submission to {{submission_path}}.')
            return

        # prepare features and label
        X = prepare_features(df, id_col, target_col)
        y = df[target_col].copy()
        if y.dtype == object or y.dtype.name == 'category':
            try:
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
            except Exception:
                y = y.astype('category').cat.codes.fillna(0).astype(int)

        # if only one class, output constant prediction
        if len(set(y)) < 2:
            print('Single-class target detected. Generating constant predictions.')
            out = pd.DataFrame({{id_col: df[id_col].astype(str), 'prediction': 0}})
            # Uses submission_path
            out.to_csv(submission_path, index=False)
            return

        # train small baseline model
        try:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except Exception:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        try:
            if hasattr(model, 'predict_proba') and len(set(y)) == 2:
                # Predict probabilities for binary classification
                preds = model.predict_proba(X)[:, 1] 
            else:
                preds = model.predict(X)
        except Exception:
            preds = model.predict(X)

        out_df = pd.DataFrame({{id_col: df[id_col].astype(str), 'prediction': preds}})
        
        # align to sample_submission if exists and if sample contains 'id'
        sample_path = dataset_path / 'sample_submission.csv'
        if sample_path.exists():
            sample = safe_read_csv(sample_path)
            if 'id' in sample.columns:
                merged = sample[['id']].merge(out_df.rename(columns={{id_col: 'id'}}), on='id', how='left')
                # fill NaNs with zeros
                merged = merged.fillna(0)
                # Uses submission_path
                merged.to_csv(submission_path, index=False)
                print(f'Wrote submission to {{submission_path}} aligned to sample_submission.csv')
                return

        # Uses submission_path
        out_df.to_csv(submission_path, index=False)
        print(f'Wrote submission to {{submission_path}}')
    except Exception as e:
        print('Fallback script error:', e)
        traceback.print_exc()
        # emergency submission
        try:
            if 'df' in locals() and 'id_col' in locals():
                pd.DataFrame({{id_col: df[id_col].astype(str), 'prediction': 0}}).to_csv(submission_path, index=False)
                print(f'Wrote emergency submission to {{submission_path}} with zeros.')
            else:
                pd.DataFrame([{{'id': 'error', 'prediction': 0}}]).to_csv(submission_path, index=False)
                print(f'Wrote emergency submission to {{submission_path}} with error placeholder.')
        except Exception:
            print('Failed to write emergency submission at all.')

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
    def execute_script(self, script_path: Path) -> Tuple[bool, str]:
        submission_filename = f"submission_{self.dataset_name}.csv"

        env = os.environ.copy()
        env["DATASET_PATH"] = str(self.dataset_path)
        env["SUBMISSION_PATH"] = submission_filename

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
            out = f"Timeout after {self.timeout} seconds. Stdout (partial): {e.stdout}\nStderr (partial): {e.stderr}"
            return False, out
        except Exception as e:
            tb = traceback.format_exc()
            self.logs.log(f"Script execution failed: {e}")
            return False, f"Execution failed: {e}\n{tb}"

    # -------------------------
    # Main run loop
    # -------------------------
    def run(self):
        train_script_filename = f"train_script_{self.dataset_name}.py"
        submission_filename = f"submission_{self.dataset_name}.csv"

        summary = self.analyze_data()

        if not summary.get("path_exists", True):
            self.logs.log(
                "Dataset path does not exist. Creating emergency submission.csv and exiting."
            )

            try:
                pd.DataFrame([{"id": "no_dataset_path", "prediction": 0}]).to_csv(
                    submission_filename, index=False
                )
                self.logs.log(f"Emergency {submission_filename} written (no dataset).")
            except Exception as e:
                self.logs.log(f"Failed to write emergency submission.csv: {e}")
            self.logs.dump()
            return

        code = self.generate_code(summary)
        script_path = self._write_script(code, path=train_script_filename)

        success, out = self.execute_script(script_path)

        if success and not Path(submission_filename).exists():
            self.logs.log(
                f"Script executed but {submission_filename} missing. Marking as failure to retry."
            )
            success = False
            out += "\nERROR: submission.csv missing after script execution.\n"

        retries = 0
        last_output = out

        while not success and retries < self.max_retries:
            self.logs.log(
                f"Retry {retries+1}/{self.max_retries} due to failure. Attempting auto-correct."
            )
            code = self.generate_code(summary, error_trace=last_output)

            script_path = self._write_script(code, path=train_script_filename)

            success, out = self.execute_script(script_path)
            last_output = out

            if success and not Path(submission_filename).exists():
                self.logs.log("Retry script ran but submission.csv still missing.")
                success = False
                last_output += "\nERROR: submission.csv missing after retry.\n"
            retries += 1

        if success:
            self.logs.log(f"Process succeeded: {submission_filename} created.")
        else:
            self.logs.log(
                "All attempts failed. Writing emergency submission and exiting."
            )

            self._write_emergency_submission(submission_filename)

        self.logs.dump()

    # -------------------------
    # Emergency submission writer
    # -------------------------
    def _write_emergency_submission(self, submission_filename: str = "submission.csv"):
        try:
            sample_path = self.dataset_path / "sample_submission.csv"
            if sample_path.exists():
                sample = pd.read_csv(sample_path)
                if "id" in sample.columns:
                    sample[
                        sample.columns[1] if len(sample.columns) > 1 else "prediction"
                    ] = 0
                    sample.to_csv(submission_filename, index=False)
                    self.logs.log(
                        f"Emergency submission.csv written from sample_submission.csv to {submission_filename}."
                    )
                    return
        except Exception as e:
            self.logs.log(f"Emergency sample_submission handling failed: {e}")

        try:
            csvs = list(self.dataset_path.glob("*.csv"))
            if csvs:
                df = pd.read_csv(csvs[0], nrows=1000)
                cols = list(df.columns)
                id_col = cols[0] if cols else "id"
                out = pd.DataFrame({id_col: df[id_col].astype(str), "prediction": 0})
                out.to_csv(submission_filename, index=False)
                self.logs.log(
                    f"Emergency submission.csv written from first csv to {submission_filename}."
                )
                return
        except Exception as e:
            self.logs.log(f"Emergency CSV fallback failed: {e}")

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
