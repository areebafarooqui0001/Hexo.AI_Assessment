import pandas as pd
import json
import re
import os
from pathlib import Path

# --- CONFIGURATION ---
# Define the 5 competitions required
COMPETITIONS = [
    "tabular-playground-series-may-2022",
    "spooky-author-identification",
    "the-icml-2013-whale-challenge-right-whale-redux",
    "text-normalization-challenge-english-language",
    "siim-isic-melanoma-classification",
]

# Columns required by the assessment image
COLUMNS = [
    "competition_id",
    "score",
    "gold_threshold",
    "silver_threshold",
    "bronze_threshold",
    "median_threshold",
    "any_medal",
    "gold_medal",
    "silver_medal",
    "bronze_medal",
    "above_median",
    "submission_exists",
    "valid_submission",
    "is_lower_better",
    "created_at",
    "submission_path",
]


def parse_logs():
    rows = []
    found_comps = set()

    print("Scanning .log files in current directory...")

    # scan all files ending in .log
    log_files = list(Path(".").glob("*.log"))

    for log_file in log_files:
        try:
            content = log_file.read_text(encoding="utf-8", errors="ignore")
            # Regex to find JSON blocks that contain "competition_id"
            # Looking for structure like { ... "competition_id": "name" ... }
            matches = re.findall(r'(\{.*?"competition_id".*?\})', content, re.DOTALL)

            for match in matches:
                try:
                    data = json.loads(match)
                    comp_id = data.get("competition_id")

                    if comp_id in COMPETITIONS and comp_id not in found_comps:
                        row = {col: data.get(col, "N/A") for col in COLUMNS}
                        rows.append(row)
                        found_comps.add(comp_id)
                        print(f" -> Found report for: {comp_id}")
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            print(f"Skipping file {log_file}: {e}")

    # Add empty rows for missing competitions (e.g. Melanoma, Text)
    for comp in COMPETITIONS:
        if comp not in found_comps:
            print(f" -> No report found for: {comp} (Adding empty row)")
            empty_row = {col: "" for col in COLUMNS}
            empty_row["competition_id"] = comp
            empty_row["submission_exists"] = "false"
            empty_row["valid_submission"] = "false"
            rows.append(empty_row)

    return rows


def main():
    data = parse_logs()

    # Create DataFrame
    df = pd.DataFrame(data)

    # Sort by competition list order
    df["sort_index"] = df["competition_id"].apply(
        lambda x: COMPETITIONS.index(x) if x in COMPETITIONS else 99
    )
    df = df.sort_values("sort_index").drop(columns="sort_index")

    # Reorder columns to match requirement exactly
    df = df[COLUMNS]

    # Save to CSV
    csv_filename = "final_submission_table.csv"
    df.to_csv(csv_filename, index=False)

    print("\n" + "=" * 50)
    print(f"SUCCESS! Table saved to: {csv_filename}")
    print("=" * 50)
    print("Markdown Table:")
    print("-" * 20)
    print(df.to_markdown(index=False))
    print("-" * 20)


if __name__ == "__main__":
    main()
