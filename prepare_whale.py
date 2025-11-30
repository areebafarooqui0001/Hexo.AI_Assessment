import os
import pandas as pd
import glob

# --- CONFIGURATION ---
# POINT THIS TO THE FOLDER CONTAINING YOUR 'train' AND 'test' FOLDERS
DATA_PATH = r"C:\Users\parde\Desktop\Project\Areeba\data\whale" 

def generate_csvs(base_path):
    print(f"Scanning directory: {base_path}")
    
    # 1. FIND THE TRAIN FOLDER
    # We look for a folder named 'train' or just .aif files directly
    train_dir = os.path.join(base_path, "train")
    if not os.path.exists(train_dir):
        # Fallback: maybe files are just in base_path?
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
        # LOGIC: "train10001_1.aif" -> Label is 1
        try:
            # Split by underscore, take the last part ("1.aif"), split by dot
            label_part = filename.split('_')[-1] 
            label = int(label_part.split('.')[0])
            
            # We save the RELATIVE path so the agent can find it easily
            # e.g., "train/train10001_1.aif"
            rel_path = os.path.join("train", filename)
            data.append({"filename": rel_path, "label": label})
        except:
            print(f"Skipping format mismatch: {filename}")

    df = pd.DataFrame(data)
    save_path = os.path.join(base_path, "train.csv")
    df.to_csv(save_path, index=False)
    print(f"✅ Success! Generated {save_path} with {len(df)} rows.")

    # 2. HANDLE TEST FILES (Optional but good)
    test_dir = os.path.join(base_path, "test")
    test_files = glob.glob(os.path.join(test_dir, "*.aif"))
    if test_files:
        test_data = [{"filename": os.path.join("test", os.path.basename(f))} for f in test_files]
        pd.DataFrame(test_data).to_csv(os.path.join(base_path, "test.csv"), index=False)
        print(f"✅ Generated test.csv with {len(test_files)} rows.")

if __name__ == "__main__":
    generate_csvs(DATA_PATH)