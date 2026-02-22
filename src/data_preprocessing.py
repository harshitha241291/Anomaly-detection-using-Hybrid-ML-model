# import os
# import glob
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# import joblib
# from tqdm import tqdm

# RAW_PATH = "data/raw/"
# PROCESSED_TRAIN_PATH = "data/processed/train/"
# PROCESSED_TEST_PATH = "data/processed/test/"
# MODEL_PATH = "models/"
# CHUNK_SIZE = 100000

# import shutil

# def reset_processed_folders():
#     for path in [PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH]:
#         if os.path.exists(path):
#             shutil.rmtree(path)
#         os.makedirs(path)

# def create_folders():
#     os.makedirs(PROCESSED_TRAIN_PATH, exist_ok=True)
#     os.makedirs(PROCESSED_TEST_PATH, exist_ok=True)
#     os.makedirs(MODEL_PATH, exist_ok=True)

# def get_file_split():
#     files = sorted(glob.glob(RAW_PATH + "*.csv"))
#     print("Files found:")
#     for f in files:
#         print(f)

#     train_files = files[:5]
#     test_files = files[5:]

#     print("\nTrain files:", train_files)
#     print("Test files:", test_files)

#     return train_files, test_files


# def clean_chunk(chunk):
#     chunk.columns = chunk.columns.str.strip()

#     drop_cols = [
#         'Flow ID', 'Source IP', 'Destination IP',
#         'Source Port', 'Destination Port'
#     ]
#     chunk.drop(columns=[c for c in drop_cols if c in chunk.columns], inplace=True)

#     chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
#     chunk.dropna(inplace=True)

#     chunk['binary_label'] = chunk['Label'].apply(
#         lambda x: 0 if x == 'BENIGN' else 1
#     )

#     X = chunk.drop(columns=['Label', 'binary_label'])
#     y = chunk['binary_label']

#     return X, y


# def fit_scaler(train_files):
#     print("\nFitting scaler on TRAIN data only...")
#     scaler = StandardScaler()

#     for file in train_files:
#         print(f"Processing for scaler: {file}")
#         for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE):
#             X, _ = clean_chunk(chunk)
#             scaler.partial_fit(X)

#     joblib.dump(scaler, MODEL_PATH + "scaler.pkl")
#     print("Scaler saved successfully.")

#     return scaler


# def transform_and_save(files, scaler, save_path):
#     for file in files:
#         print(f"Transforming: {file}")
#         file_name = os.path.basename(file).replace(".csv", ".parquet")

#         for i, chunk in enumerate(pd.read_csv(file, chunksize=CHUNK_SIZE)):
#             X, y = clean_chunk(chunk)

#             X_scaled = scaler.transform(X)

#             processed_df = pd.DataFrame(X_scaled, columns=X.columns)
#             processed_df['binary_label'] = y.values

#             output_file = save_path + f"{file_name}_part{i}.parquet"
#             processed_df.to_parquet(output_file, index=False)


# def main():
#     create_folders()
#     train_files, test_files = get_file_split()

#     scaler = fit_scaler(train_files)

#     print("\nProcessing TRAIN data...")
#     transform_and_save(train_files, scaler, PROCESSED_TRAIN_PATH)

#     print("\nProcessing TEST data...")
#     transform_and_save(test_files, scaler, PROCESSED_TEST_PATH)

#     print("\nPreprocessing completed successfully.")


# if __name__ == "__main__":
#     main()

import os
import glob
import shutil
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

RAW_PATH = "data/raw/"
PROCESSED_TRAIN_PATH = "data/processed/train/"
PROCESSED_TEST_PATH = "data/processed/test/"
MODEL_PATH = "models/"
CHUNK_SIZE = 100000


def reset_processed_folders():
    """Clear old processed data before a fresh run."""
    for path in [PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)


def create_folders():
    os.makedirs(PROCESSED_TRAIN_PATH, exist_ok=True)
    os.makedirs(PROCESSED_TEST_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)


def get_file_split():
    files = sorted(glob.glob(RAW_PATH + "*.csv"))

    print(f"Total files found: {len(files)}")
    for f in files:
        print(f"  {f}")

    if len(files) < 2:
        raise ValueError(
            f"Need at least 2 CSV files in '{RAW_PATH}'. "
            f"Found {len(files)}. Check that all CICIDS2017 CSVs are present."
        )

    # Robust 80/20 split instead of hardcoded index
    split_idx = max(1, int(len(files) * 0.8))
    train_files = files[:split_idx]
    test_files = files[split_idx:]

    print(f"\nTrain files ({len(train_files)}):")
    for f in train_files:
        print(f"  {f}")
    print(f"\nTest files ({len(test_files)}):")
    for f in test_files:
        print(f"  {f}")

    return train_files, test_files


def clean_chunk(chunk):
    """Clean a single chunk: drop irrelevant cols, handle inf/nan, create binary label."""
    chunk.columns = chunk.columns.str.strip()

    # FIX: Added 'Timestamp' — CICIDS2017 includes this non-numeric column
    drop_cols = [
        'Flow ID', 'Source IP', 'Destination IP',
        'Source Port', 'Destination Port', 'Timestamp'
    ]
    chunk.drop(columns=[c for c in drop_cols if c in chunk.columns], inplace=True)

    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    chunk.dropna(inplace=True)

    chunk['binary_label'] = chunk['Label'].apply(
        lambda x: 0 if str(x).strip() == 'BENIGN' else 1
    )

    X = chunk.drop(columns=['Label', 'binary_label'])
    y = chunk['binary_label']

    # FIX: Reset index to prevent misalignment after dropna
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X, y


def fit_scaler(train_files):
    """Fit StandardScaler on training data only (prevents data leakage)."""
    print("\nFitting scaler on TRAIN data only...")
    scaler = StandardScaler()

    for file in train_files:
        print(f"  Fitting on: {os.path.basename(file)}")
        for chunk in tqdm(
            pd.read_csv(file, chunksize=CHUNK_SIZE, low_memory=False),
            desc="  Chunks"
        ):
            X, _ = clean_chunk(chunk)
            if len(X) > 0:
                scaler.partial_fit(X)

    joblib.dump(scaler, MODEL_PATH + "scaler.pkl")
    print("Scaler saved to models/scaler.pkl")

    return scaler


def transform_and_save(files, scaler, save_path, split_name=""):
    """Transform files with fitted scaler and save as parquet."""
    if not files:
        print(f"  WARNING: No files to process for {split_name}!")
        return

    for file in files:
        print(f"  Transforming: {os.path.basename(file)}")
        file_name = os.path.basename(file).replace(".csv", ".parquet")

        for i, chunk in enumerate(
            tqdm(
                pd.read_csv(file, chunksize=CHUNK_SIZE, low_memory=False),
                desc=f"  {os.path.basename(file)}"
            )
        ):
            X, y = clean_chunk(chunk)

            if len(X) == 0:
                print(f"    Skipping empty chunk {i}")
                continue

            X_scaled = scaler.transform(X)

            processed_df = pd.DataFrame(X_scaled, columns=X.columns)
            processed_df['binary_label'] = y.values

            output_file = os.path.join(save_path, f"{file_name}_part{i}.parquet")
            processed_df.to_parquet(output_file, index=False)

    saved = glob.glob(save_path + "*.parquet")
    print(f"  Saved {len(saved)} parquet files to {save_path}")


def main():
    # FIX: Reset folders first so stale data doesn't persist
    print("Resetting processed folders...")
    reset_processed_folders()
    create_folders()

    train_files, test_files = get_file_split()

    scaler = fit_scaler(train_files)

    print("\nProcessing TRAIN data...")
    transform_and_save(train_files, scaler, PROCESSED_TRAIN_PATH, "train")

    print("\nProcessing TEST data...")
    transform_and_save(test_files, scaler, PROCESSED_TEST_PATH, "test")

    # Summary
    train_parts = glob.glob(PROCESSED_TRAIN_PATH + "*.parquet")
    test_parts = glob.glob(PROCESSED_TEST_PATH + "*.parquet")
    print(f"\n✅ Preprocessing complete.")
    print(f"   Train parquet files: {len(train_parts)}")
    print(f"   Test parquet files:  {len(test_parts)}")

    if len(test_parts) == 0:
        print("\n⚠️  Test folder is still empty!")
        print("   Check that you have more than 5 CSV files in data/raw/")
        print(f"   Currently found {len(train_files) + len(test_files)} total files.")


if __name__ == "__main__":
    main()
