import os
import pandas as pd
import glob


DATA_PATH = r"C:\Users\parde\Desktop\Project\Areeba\data\whale"


def generate_csvs(base_path):
    print(f"Scanning directory: {base_path}")

    # 1. FIND THE TRAIN FOLDER
    train_dir = os.path.join(base_path, "train")
    if not os.path.exists(train_dir):
        train_dir = base_path

    print(f"Looking for training files in: {train_dir}")
    train_files = glob.glob(os.path.join(train_dir, "*.aif"))

    if not train_files:
        print("ERROR: No .aif files found! Check your path.")
        return

    print(f"Found {len(train_files)} files. Extracting labels...")

    data = []
    for filepath in train_files:
        filename = os.path.basename(filepath)
        # "train10001_1.aif" -> Label is 1
        try:
            label_part = filename.split("_")[-1]
            label = int(label_part.split(".")[0])

            rel_path = os.path.join("train", filename)
            data.append({"filename": rel_path, "label": label})
        except:
            print(f"Skipping format mismatch: {filename}")

    df = pd.DataFrame(data)
    save_path = os.path.join(base_path, "train.csv")
    df.to_csv(save_path, index=False)
    print(f" Success! Generated {save_path} with {len(df)} rows.")

    # 2. HANDLE TEST FILES
    test_dir = os.path.join(base_path, "test")
    test_files = glob.glob(os.path.join(test_dir, "*.aif"))
    if test_files:
        test_data = [
            {"filename": os.path.join("test", os.path.basename(f))} for f in test_files
        ]
        pd.DataFrame(test_data).to_csv(os.path.join(base_path, "test.csv"), index=False)
        print(f" Generated test.csv with {len(test_files)} rows.")


if __name__ == "__main__":
    generate_csvs(DATA_PATH)
