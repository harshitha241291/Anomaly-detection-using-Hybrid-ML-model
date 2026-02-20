import os
import glob
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


def create_folders():
    os.makedirs(PROCESSED_TRAIN_PATH, exist_ok=True)
    os.makedirs(PROCESSED_TEST_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)


def get_file_split():
    files = sorted(glob.glob(RAW_PATH + "*.csv"))
    print("Files found:")
    for f in files:
        print(f)

    train_files = files[:5]
    test_files = files[5:]

    print("\nTrain files:", train_files)
    print("Test files:", test_files)

    return train_files, test_files


def clean_chunk(chunk):
    chunk.columns = chunk.columns.str.strip()

    drop_cols = [
        'Flow ID', 'Source IP', 'Destination IP',
        'Source Port', 'Destination Port'
    ]
    chunk.drop(columns=[c for c in drop_cols if c in chunk.columns], inplace=True)

    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    chunk.dropna(inplace=True)

    chunk['binary_label'] = chunk['Label'].apply(
        lambda x: 0 if x == 'BENIGN' else 1
    )

    X = chunk.drop(columns=['Label', 'binary_label'])
    y = chunk['binary_label']

    return X, y


def fit_scaler(train_files):
    print("\nFitting scaler on TRAIN data only...")
    scaler = StandardScaler()

    for file in train_files:
        print(f"Processing for scaler: {file}")
        for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE):
            X, _ = clean_chunk(chunk)
            scaler.partial_fit(X)

    joblib.dump(scaler, MODEL_PATH + "scaler.pkl")
    print("Scaler saved successfully.")

    return scaler


def transform_and_save(files, scaler, save_path):
    for file in files:
        print(f"Transforming: {file}")
        file_name = os.path.basename(file).replace(".csv", ".parquet")

        for i, chunk in enumerate(pd.read_csv(file, chunksize=CHUNK_SIZE)):
            X, y = clean_chunk(chunk)

            X_scaled = scaler.transform(X)

            processed_df = pd.DataFrame(X_scaled, columns=X.columns)
            processed_df['binary_label'] = y.values

            output_file = save_path + f"{file_name}_part{i}.parquet"
            processed_df.to_parquet(output_file, index=False)


def main():
    create_folders()
    train_files, test_files = get_file_split()

    scaler = fit_scaler(train_files)

    print("\nProcessing TRAIN data...")
    transform_and_save(train_files, scaler, PROCESSED_TRAIN_PATH)

    print("\nProcessing TEST data...")
    transform_and_save(test_files, scaler, PROCESSED_TEST_PATH)

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()
