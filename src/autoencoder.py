import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
import joblib

tf.random.set_seed(42)
np.random.seed(42)

# ─── Paths ────────────────────────────────────────────────────────────────────
PROCESSED_TRAIN_PATH = "data/processed/train/"
PROCESSED_TEST_PATH  = "data/processed/test/"
SCORES_PATH          = "data/scores/"
MODEL_PATH           = "models/"

# ─── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE       = 2048
EPOCHS           = 30
LEARNING_RATE    = 0.001
VALIDATION_SPLIT = 0.1
MAX_BENIGN_ROWS  = 200_000   # ← lowered to avoid OOM
ENCODING_DIM     = 32        # bottleneck size


# ─── Setup ────────────────────────────────────────────────────────────────────
def create_folders():
    os.makedirs(SCORES_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)


def get_files(folder):
    files = sorted(glob.glob(folder + "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {folder}")
    return files


# ─── Step 1: Collect benign sample ────────────────────────────────────────────
def collect_benign_sample(train_files):
    """
    Collect benign rows one file at a time.
    Slices exactly 'remaining' rows per file so we never allocate more than MAX_BENIGN_ROWS.
    """
    print(f"\n[1/6] Collecting up to {MAX_BENIGN_ROWS:,} benign rows...")
    chunks = []
    collected = 0

    for f in tqdm(train_files, desc="  Scanning"):
        if collected >= MAX_BENIGN_ROWS:
            break

        df = pd.read_parquet(f)
        benign = df[df['binary_label'] == 0].drop(columns=['binary_label'])

        # ── KEY FIX: slice only what we still need ────────────────────────────
        remaining = MAX_BENIGN_ROWS - collected
        if len(benign) > remaining:
            benign = benign.iloc[:remaining]

        chunks.append(benign)
        collected += len(benign)
        del df, benign   # free RAM immediately after each file

    X_benign = pd.concat(chunks, ignore_index=True)
    del chunks

    print(f"  Benign samples : {len(X_benign):,}")
    print(f"  Feature dims   : {X_benign.shape[1]}")
    return X_benign.values.astype(np.float32)


# ─── Step 2: Build Autoencoder ────────────────────────────────────────────────
def build_autoencoder(input_dim):
    """
    Symmetric encoder-decoder:
    Input → 256 → 128 → 32(bottleneck) → 128 → 256 → Output

    Trained to reconstruct NORMAL traffic.
    High reconstruction error = pattern doesn't match normal = anomaly.
    """
    inputs = layers.Input(shape=(input_dim,))

    # Encoder
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    bottleneck = layers.Dense(ENCODING_DIM, activation='relu', name='bottleneck')(x)

    # Decoder
    x = layers.Dense(128, activation='relu')(bottleneck)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(input_dim, activation='linear')(x)

    model = Model(inputs, outputs, name="Autoencoder")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse'
    )
    model.summary()
    return model


# ─── Step 3: Train ────────────────────────────────────────────────────────────
def train_autoencoder(autoencoder, X_benign):
    print(f"\n[3/6] Training Autoencoder on {len(X_benign):,} benign samples...")

    cb = [
        callbacks.EarlyStopping(
            monitor='val_loss', patience=5,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, min_lr=1e-6, verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_PATH, "autoencoder_best.keras"),
            monitor='val_loss', save_best_only=True, verbose=1
        )
    ]

    history = autoencoder.fit(
        X_benign, X_benign,          # input = target (reconstruction)
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=cb,
        verbose=1
    )

    autoencoder.save(os.path.join(MODEL_PATH, "autoencoder_final.keras"))
    print(f"  Model saved → {MODEL_PATH}autoencoder_final.keras")
    return autoencoder, history


# ─── Step 4: Compute threshold ────────────────────────────────────────────────
def compute_threshold(autoencoder, X_benign):
    """
    Threshold = mean + 2*std of reconstruction errors on benign training data.
    Anything above this → flagged as anomaly.
    """
    print("\n[4/6] Computing anomaly threshold...")
    recon  = autoencoder.predict(X_benign, batch_size=BATCH_SIZE, verbose=0)
    errors = np.mean(np.square(X_benign - recon), axis=1)

    threshold = float(np.mean(errors) + 2 * np.std(errors))
    print(f"  Benign error  → mean: {np.mean(errors):.6f}, std: {np.std(errors):.6f}")
    print(f"  Threshold     → {threshold:.6f}  (mean + 2*std)")

    joblib.dump(threshold, os.path.join(MODEL_PATH, "ae_threshold.pkl"))
    print(f"  Threshold saved → {MODEL_PATH}ae_threshold.pkl")
    return threshold


# ─── Step 5: Score files ──────────────────────────────────────────────────────
def score_files(autoencoder, threshold, files, split_name):
    """
    Score one parquet file at a time → no OOM.
    Outputs:
      ae_recon_error : continuous anomaly score (higher = more anomalous)
      ae_binary_pred : 1 if error > threshold else 0
    """
    print(f"\nScoring {split_name} ({len(files)} files)...")
    all_true, all_preds, all_errors, score_parts = [], [], [], []
    out_path = os.path.join(SCORES_PATH, f"ae_scores_{split_name}.parquet")

    for f in tqdm(files, desc=f"  {split_name}"):
        df    = pd.read_parquet(f)
        y     = df['binary_label'].values
        X     = df.drop(columns=['binary_label']).values.astype(np.float32)
        del df

        recon       = autoencoder.predict(X, batch_size=BATCH_SIZE, verbose=0)
        recon_error = np.mean(np.square(X - recon), axis=1)
        binary_pred = (recon_error > threshold).astype(int)
        del X, recon

        all_true.append(y)
        all_preds.append(binary_pred)
        all_errors.append(recon_error)
        score_parts.append(pd.DataFrame({
            'ae_recon_error': recon_error,
            'ae_binary_pred': binary_pred,
            'binary_label':   y
        }))

    pd.concat(score_parts, ignore_index=True).to_parquet(out_path, index=False)
    del score_parts
    print(f"  Scores saved → {out_path}")

    y_true_all  = np.concatenate(all_true)
    y_pred_all  = np.concatenate(all_preds)
    error_all   = np.concatenate(all_errors)

    print(f"\n{'='*55}")
    print(f"  Evaluation — {split_name.upper()}")
    print(f"{'='*55}")
    print(classification_report(y_true_all, y_pred_all, target_names=['Benign', 'Attack']))
    try:
        auc = roc_auc_score(y_true_all, error_all)
        print(f"  ROC-AUC: {auc:.4f}")
    except Exception as e:
        print(f"  AUC error: {e}")

    return len(y_true_all)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    create_folders()

    train_files = get_files(PROCESSED_TRAIN_PATH)
    test_files  = get_files(PROCESSED_TEST_PATH)
    print(f"Train files : {len(train_files)}")
    print(f"Test files  : {len(test_files)}")

    # 1. Collect benign data
    X_benign  = collect_benign_sample(train_files)
    input_dim = X_benign.shape[1]

    # 2. Build model
    print(f"\n[2/6] Building Autoencoder (input={input_dim}, bottleneck={ENCODING_DIM})...")
    autoencoder = build_autoencoder(input_dim)

    # 3. Train
    autoencoder, _ = train_autoencoder(autoencoder, X_benign)

    # 4. Threshold
    threshold = compute_threshold(autoencoder, X_benign)
    del X_benign

    # 5. Score train
    print("\n[5/6] Scoring TRAIN set...")
    n_train = score_files(autoencoder, threshold, train_files, "train")

    # 6. Score test
    print("\n[6/6] Scoring TEST set...")
    n_test = score_files(autoencoder, threshold, test_files, "test")

    print(f"\n✅ Autoencoder complete!")
    print("="*55)
    print(f"  Train scored    : {n_train:,}")
    print(f"  Test scored     : {n_test:,}")
    print(f"  Scores in       : {SCORES_PATH}")
    print("\n  Ready for Supervised Classifier:")
    print(f"    if_scores_train.parquet  ← Isolation Forest")
    print(f"    ae_scores_train.parquet  ← Autoencoder")
    print(f"\n  Next → Supervised Classifier (XGBoost + Random Forest)")


if __name__ == "__main__":
    main()