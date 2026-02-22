import os
import glob
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
PROCESSED_TRAIN_PATH = "data/processed/train/"
PROCESSED_TEST_PATH  = "data/processed/test/"
SCORES_PATH          = "data/scores/"
MODEL_PATH           = "models/"

RANDOM_STATE  = 42
N_ESTIMATORS  = 100    # reduced for memory efficiency
MAX_SAMPLES   = 50000  # fixed int cap → predictable RAM usage per tree
CONTAMINATION = 0.1

# Max benign rows to train on (lower if still OOM, raise for better accuracy)
MAX_BENIGN_ROWS = 500_000


# ─── Setup ────────────────────────────────────────────────────────────────────
def create_folders():
    os.makedirs(SCORES_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)


def get_files(folder):
    files = sorted(glob.glob(folder + "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {folder}")
    return files


# ─── Step 1: Collect benign samples (memory-capped) ───────────────────────────
def collect_benign_sample(train_files):
    """
    Read parquet files one by one, keep BENIGN rows only.
    Stops once MAX_BENIGN_ROWS is reached to avoid OOM.
    """
    print(f"\n[1/5] Collecting up to {MAX_BENIGN_ROWS:,} benign rows for IF training...")
    benign_chunks = []
    collected = 0

    for f in tqdm(train_files, desc="  Scanning train files"):
        if collected >= MAX_BENIGN_ROWS:
            break
        df = pd.read_parquet(f)
        benign = df[df['binary_label'] == 0].drop(columns=['binary_label'])
        benign_chunks.append(benign)
        collected += len(benign)
        del df  # free RAM immediately

    X_benign = pd.concat(benign_chunks, ignore_index=True)

    if len(X_benign) > MAX_BENIGN_ROWS:
        X_benign = X_benign.sample(MAX_BENIGN_ROWS, random_state=RANDOM_STATE)

    print(f"  Benign samples collected: {len(X_benign):,}")
    return X_benign


# ─── Step 2: Train Isolation Forest ───────────────────────────────────────────
def train_isolation_forest(X_benign):
    print(f"\n[2/5] Training Isolation Forest...")
    print(f"  n_estimators={N_ESTIMATORS}, max_samples={MAX_SAMPLES}, contamination={CONTAMINATION}")

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_benign)
    del X_benign  # free RAM after training

    model_out = os.path.join(MODEL_PATH, "isolation_forest.pkl")
    joblib.dump(model, model_out)
    print(f"  Model saved → {model_out}")
    return model


# ─── Steps 3 & 4: Score files one at a time ───────────────────────────────────
def score_files(model, files, split_name):
    """
    Score parquet files ONE AT A TIME to avoid OOM.
    Saves scores incrementally and evaluates at the end.
    """
    print(f"\nScoring {split_name} set ({len(files)} files)...")

    all_true   = []
    all_preds  = []
    all_scores = []
    score_parts = []
    out_path   = os.path.join(SCORES_PATH, f"if_scores_{split_name}.parquet")

    for f in tqdm(files, desc=f"  Scoring {split_name}"):
        df = pd.read_parquet(f)
        y  = df['binary_label'].values
        X  = df.drop(columns=['binary_label'])

        anomaly_score = -model.score_samples(X)            # higher = more anomalous
        binary_pred   = (model.predict(X) == -1).astype(int)

        all_true.append(y)
        all_preds.append(binary_pred)
        all_scores.append(anomaly_score)

        score_parts.append(pd.DataFrame({
            'if_anomaly_score': anomaly_score,
            'if_binary_pred':   binary_pred,
            'binary_label':     y
        }))

        del df, X

    # Concat and save
    pd.concat(score_parts, ignore_index=True).to_parquet(out_path, index=False)
    del score_parts
    print(f"  Scores saved → {out_path}")

    # Evaluate against ground truth
    y_true_all  = np.concatenate(all_true)
    y_pred_all  = np.concatenate(all_preds)
    score_all   = np.concatenate(all_scores)

    print(f"\n{'='*50}")
    print(f"  Evaluation — {split_name.upper()} set")
    print(f"{'='*50}")
    print(classification_report(y_true_all, y_pred_all, target_names=['Benign', 'Attack']))

    try:
        auc = roc_auc_score(y_true_all, score_all)
        print(f"  ROC-AUC: {auc:.4f}")
    except Exception as e:
        print(f"  AUC error: {e}")

    return len(y_true_all)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    create_folders()

    train_files = get_files(PROCESSED_TRAIN_PATH)
    test_files  = get_files(PROCESSED_TEST_PATH)
    print(f"Train parquet files : {len(train_files)}")
    print(f"Test parquet files  : {len(test_files)}")

    # Step 1: Collect benign training sample (memory-safe)
    X_benign = collect_benign_sample(train_files)

    # Step 2: Train IF on benign only
    model = train_isolation_forest(X_benign)

    # Step 3: Score full train set
    print("\n[3/5] Scoring TRAIN set...")
    n_train = score_files(model, train_files, "train")

    # Step 4: Score test set
    print("\n[4/5] Scoring TEST set...")
    n_test = score_files(model, test_files, "test")

    # Step 5: Summary
    print(f"\n[5/5] Summary")
    print("="*50)
    print(f"  Train samples scored : {n_train:,}")
    print(f"  Test samples scored  : {n_test:,}")
    print(f"  Scores saved to      : {SCORES_PATH}")
    print("\n✅ Isolation Forest complete!")
    print("   Next step → Autoencoder, then Supervised Classifier (XGBoost + RF)")


if __name__ == "__main__":
    main()