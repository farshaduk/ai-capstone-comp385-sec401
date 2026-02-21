"""
FARUD — Full Training Pipeline
================================
COMP 385 AI Capstone — AI-Powered Rental Fraud & Trust Scoring System

This script executes the COMPLETE training pipeline:
  Step 1: Build BERT fraud_dataset.csv (merge legitimate + scam texts)
  Step 2: Preprocess rental_listings_dataset.csv (feature engineering)
  Step 3: Train BERT fraud classifier (DistilBERT fine-tuning)
  Step 4: Train Isolation Forest (unsupervised anomaly detection)
  Step 5: Load price benchmarks (price anomaly reference data)

Author: Group #2
"""

import os
import sys
import shutil
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Project paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(PROJECT_DIR, "backend")
DATA_DIR = os.path.join(BACKEND_DIR, "data")
SELECTED_DIR = os.path.join(PROJECT_DIR, "DATA", "selected_datasets")

# Add backend to path
sys.path.insert(0, BACKEND_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FARUD_TRAINING")


# ══════════════════════════════════════════════════════════════════════
#  STEP 1 — Build BERT Training Dataset (fraud_dataset.csv)
# ══════════════════════════════════════════════════════════════════════

def step1_build_bert_dataset():
    """
    Merge legitimate rental listings + scam messages into a single CSV
    with columns: text, label (0=legitimate, 1=fraud)

    Sources:
      - rental_listings_dataset.csv      → title + " — " + description → label=0
      - toronto_rent_scam_messages_rich.csv → message column → label=1
      - toronto_rent_scam_messages.csv      → message column → label=1
    """
    logger.info("=" * 70)
    logger.info("STEP 1 — Building BERT Training Dataset")
    logger.info("=" * 70)

    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Legitimate listings ──
    listings_path = os.path.join(SELECTED_DIR, "rental_listings_dataset.csv")
    df_listings = pd.read_csv(listings_path)
    logger.info(f"Loaded {len(df_listings)} legitimate listings")

    # Combine title + description into a single text field
    df_listings["text"] = (
        df_listings["title"].fillna("").astype(str)
        + " — "
        + df_listings["description"].fillna("").astype(str)
    )
    df_listings["label"] = 0  # legitimate
    legitimate = df_listings[["text", "label"]].copy()

    # Filter out empty/very short texts
    legitimate = legitimate[legitimate["text"].str.strip().str.len() > 10]
    logger.info(f"  Legitimate after filter: {len(legitimate)}")

    # ── Scam messages (rich — 10 scam types) ──
    rich_path = os.path.join(SELECTED_DIR, "toronto_rent_scam_messages_rich.csv")
    df_rich = pd.read_csv(rich_path)
    logger.info(f"Loaded {len(df_rich)} rich scam messages")

    df_rich["text"] = df_rich["message"].fillna("").astype(str)
    # The dataset has "scam" and "legitimate" labels
    df_rich["label"] = (df_rich["label"].str.lower() == "scam").astype(int)
    scam_rich = df_rich[["text", "label"]].copy()
    scam_rich = scam_rich[scam_rich["text"].str.strip().str.len() > 5]
    logger.info(f"  Rich scam (label=1): {(scam_rich['label']==1).sum()}, legitimate (label=0): {(scam_rich['label']==0).sum()}")

    # ── Scam messages (basic) ──
    basic_path = os.path.join(SELECTED_DIR, "toronto_rent_scam_messages.csv")
    df_basic = pd.read_csv(basic_path)
    logger.info(f"Loaded {len(df_basic)} basic scam messages")

    df_basic["text"] = df_basic["message"].fillna("").astype(str)
    df_basic["label"] = (df_basic["label"].str.lower() == "scam").astype(int)
    scam_basic = df_basic[["text", "label"]].copy()
    scam_basic = scam_basic[scam_basic["text"].str.strip().str.len() > 5]

    # Keep only SCAM rows from basic dataset (avoids duplicate legitimate texts)
    scam_basic_fraud_only = scam_basic[scam_basic["label"] == 1]
    logger.info(f"  Basic scam (label=1 only): {len(scam_basic_fraud_only)}")

    # ── Combine ──
    combined = pd.concat([legitimate, scam_rich, scam_basic_fraud_only], ignore_index=True)

    # Drop exact duplicates
    combined.drop_duplicates(subset=["text"], keep="first", inplace=True)

    # Shuffle
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"\n  === Combined Dataset ===")
    logger.info(f"  Total samples:  {len(combined)}")
    logger.info(f"  Legitimate (0): {(combined['label']==0).sum()}")
    logger.info(f"  Fraud (1):      {(combined['label']==1).sum()}")
    logger.info(f"  Avg text len:   {combined['text'].str.len().mean():.0f} chars")

    # ── Balance classes (undersample majority) ──
    n_fraud = (combined["label"] == 1).sum()
    n_legit = (combined["label"] == 0).sum()

    if n_legit > n_fraud * 1.5:
        # Undersample legitimate to 1.2x fraud count (slight imbalance is realistic)
        target_legit = int(n_fraud * 1.2)
        legit_sample = combined[combined["label"] == 0].sample(
            n=target_legit, random_state=42
        )
        fraud_sample = combined[combined["label"] == 1]
        combined = pd.concat([legit_sample, fraud_sample], ignore_index=True)
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(f"\n  === After Balancing ===")
        logger.info(f"  Total samples:  {len(combined)}")
        logger.info(f"  Legitimate (0): {(combined['label']==0).sum()}")
        logger.info(f"  Fraud (1):      {(combined['label']==1).sum()}")

    # ── Save ──
    output_path = os.path.join(DATA_DIR, "fraud_dataset.csv")
    combined.to_csv(output_path, index=False)
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"\n  ✓ Saved: {output_path} ({file_size:.2f} MB)")

    return output_path


# ══════════════════════════════════════════════════════════════════════
#  STEP 2 — Preprocess Rental Listings (Feature Engineering Pipeline)
# ══════════════════════════════════════════════════════════════════════

def step2_preprocess_rental_listings():
    """
    Run the full data preprocessing + feature engineering pipeline
    on rental_listings_dataset.csv for Isolation Forest training.

    Pipeline steps:
      1. Automated type inference
      2. Data cleaning (duplicates, type casting, missing values)
      3. KNN imputation for numeric gaps
      4. Ensemble outlier detection (IQR + Isolation Forest + LOF) → clip
      5. Text feature engineering (TF-IDF + SVD, fraud-linguistic patterns)
      6. Numerical feature engineering (scaling, polynomial, binning, log)
      7. Categorical encoding (frequency + count)
      8. Geospatial features (K-Means clustering, haversine distance)
      9. Feature selection (variance, correlation, mutual information)
    """
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2 — Preprocessing Rental Listings (Feature Engineering)")
    logger.info("=" * 70)

    from application.use_cases.data_preprocessing_pipeline import (
        create_rental_fraud_pipeline,
    )

    # Load dataset
    src = os.path.join(SELECTED_DIR, "rental_listings_dataset.csv")
    df = pd.read_csv(src, low_memory=False)
    logger.info(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

    # Smart column detection
    target_col = None
    for cand in ["is_fraud", "fraud", "label", "fraud_label", "target"]:
        if cand in df.columns:
            target_col = cand
            break

    text_cols = []
    for cand in ["description", "title", "listing_title", "listing_text",
                 "text", "content", "listing_description"]:
        if cand in df.columns:
            text_cols.append(cand)

    exclude_cols = [c for c in ["listing_id", "address", "link", "url"] if c in df.columns]

    logger.info(f"  Target col:  {target_col}")
    logger.info(f"  Text cols:   {text_cols}")
    logger.info(f"  Exclude:     {exclude_cols}")

    # Run pipeline
    pipeline = create_rental_fraud_pipeline(use_embeddings=False, include_geo=True)
    processed_df = pipeline.fit_transform(
        df,
        target_column=target_col,
        text_columns=text_cols if text_cols else None,
        exclude_columns=exclude_cols if exclude_cols else None,
    )

    logger.info(f"\n  === Preprocessing Results ===")
    logger.info(f"  Original:  {df.shape[0]} rows × {df.shape[1]} cols")
    logger.info(f"  Processed: {processed_df.shape[0]} rows × {processed_df.shape[1]} cols")
    logger.info(f"  Features created: {processed_df.shape[1]}")

    # Quality report
    try:
        quality = pipeline.get_quality_report(df)
        logger.info(f"  Quality score: {quality.data_quality_score * 100:.1f}%")
    except Exception as e:
        logger.warning(f"  Quality report skipped: {e}")

    # Save processed data
    processed_dir = os.path.join(DATA_DIR, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, "rental_listings_processed.csv")
    processed_df.to_csv(processed_path, index=False)
    file_size = os.path.getsize(processed_path) / (1024 * 1024)
    logger.info(f"\n  ✓ Saved: {processed_path} ({file_size:.2f} MB)")

    return processed_path


# ══════════════════════════════════════════════════════════════════════
#  STEP 3 — Train BERT Fraud Classifier
# ══════════════════════════════════════════════════════════════════════

def step3_train_bert():
    """
    Fine-tune DistilBERT on the fraud dataset.

    Config:
      - Model:      distilbert-base-uncased
      - Max length:  256 tokens
      - Batch size:  16
      - LR:          2e-5 with linear warmup (10%)
      - Epochs:      4
      - Splits:      70% train / 10% val / 20% test
    """
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3 — Training BERT Fraud Classifier (DistilBERT)")
    logger.info("=" * 70)

    from application.use_cases.bert_fraud_classifier import (
        train_fraud_model,
    )

    dataset_path = os.path.join(DATA_DIR, "fraud_dataset.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"fraud_dataset.csv not found at {dataset_path}. Run Step 1 first.")

    # Verify dataset
    df = pd.read_csv(dataset_path)
    logger.info(f"  Dataset:     {len(df)} samples")
    logger.info(f"  Columns:     {list(df.columns)}")
    logger.info(f"  Label dist:  {df['label'].value_counts().to_dict()}")

    # Train
    logger.info("\n  Starting BERT training... (this may take 10-30 min on CPU)")
    start = time.time()

    results = train_fraud_model(
        dataset_path=dataset_path,
        output_name=f"bert_fraud_v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    elapsed = time.time() - start

    # Report
    metrics = results["metrics"]
    logger.info(f"\n  === BERT Training Results ===")
    logger.info(f"  Time:       {elapsed / 60:.1f} min")
    logger.info(f"  Accuracy:   {metrics['accuracy']:.4f}")
    logger.info(f"  Precision:  {metrics['precision']:.4f}")
    logger.info(f"  Recall:     {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:   {metrics['f1_score']:.4f}")
    logger.info(f"  ROC AUC:    {metrics['roc_auc']:.4f}")
    logger.info(f"  Best Epoch: {metrics['best_epoch']}")
    logger.info(f"  Model path: {results['model_path']}")
    logger.info(f"  Dataset split: {results['dataset_size']}")

    cm = metrics.get("confusion_matrix", [])
    if cm and len(cm) == 2:
        logger.info(f"  Confusion Matrix:")
        logger.info(f"    TN={cm[0][0]}, FP={cm[0][1]}")
        logger.info(f"    FN={cm[1][0]}, TP={cm[1][1]}")

    logger.info(f"\n  ✓ BERT model saved to: {results['model_path']}")
    return results


# ══════════════════════════════════════════════════════════════════════
#  STEP 4 — Train Isolation Forest (Unsupervised Anomaly Detection)
# ══════════════════════════════════════════════════════════════════════

def step4_train_isolation_forest():
    """
    Train Isolation Forest on the preprocessed rental listings features.

    The model learns normal rental listing patterns and flags anomalies
    as potential fraud based on statistical deviations.
    """
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4 — Training Isolation Forest (Anomaly Detection)")
    logger.info("=" * 70)

    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    import joblib

    # Load preprocessed data
    processed_path = os.path.join(DATA_DIR, "processed", "rental_listings_processed.csv")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed data not found at {processed_path}. Run Step 2 first.")

    df = pd.read_csv(processed_path, low_memory=False)
    logger.info(f"Loaded processed data: {df.shape[0]} rows × {df.shape[1]} cols")

    # Select numeric features only
    features_df = df.select_dtypes(include=[np.number])
    features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    logger.info(f"Numeric features: {features_df.shape[1]}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)
    logger.info(f"Features scaled to zero-mean unit-var")

    # Train Isolation Forest
    logger.info(f"\n  Training Isolation Forest...")
    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=0.05,  # Expect ~5% anomalies
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    start = time.time()
    model.fit(X_scaled)
    elapsed = time.time() - start
    logger.info(f"  Training time: {elapsed:.1f}s")

    # Predict anomalies
    predictions = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)

    n_anomalies = (predictions == -1).sum()
    n_normal = (predictions == 1).sum()
    logger.info(f"\n  === Isolation Forest Results ===")
    logger.info(f"  Normal:    {n_normal} ({n_normal/len(predictions)*100:.1f}%)")
    logger.info(f"  Anomalies: {n_anomalies} ({n_anomalies/len(predictions)*100:.1f}%)")
    logger.info(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    logger.info(f"  Score mean:  {scores.mean():.4f}")

    # Save model + scaler
    model_dir = os.path.join(BACKEND_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"IsolationForest_v{version}"
    model_path = os.path.join(model_dir, model_name)
    os.makedirs(model_path, exist_ok=True)

    joblib.dump(model, os.path.join(model_path, "isolation_forest.joblib"))
    joblib.dump(scaler, os.path.join(model_path, "scaler.joblib"))

    # Save feature names for inference consistency
    feature_names = list(features_df.columns)
    import json
    with open(os.path.join(model_path, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    # Save metrics
    metrics = {
        "model_type": "IsolationForest",
        "n_estimators": 200,
        "contamination": 0.05,
        "n_samples": int(len(predictions)),
        "n_features": int(features_df.shape[1]),
        "n_anomalies": int(n_anomalies),
        "n_normal": int(n_normal),
        "anomaly_rate": float(n_anomalies / len(predictions)),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "training_time_seconds": float(elapsed),
        "trained_at": datetime.now().isoformat(),
    }
    metrics_dir = os.path.join(model_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\n  ✓ Model saved:  {model_path}")
    logger.info(f"  ✓ Metrics saved: {metrics_dir}/{model_name}_metrics.json")

    return model_path, metrics


# ══════════════════════════════════════════════════════════════════════
#  STEP 5 — Load Price Benchmarks (Reference Data)
# ══════════════════════════════════════════════════════════════════════

def step5_load_price_benchmarks():
    """
    Copy/load the Toronto price benchmarks dataset for price anomaly detection.

    This data is used by the indicator engine to flag listings with
    prices significantly above or below market rates for their area.
    """
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5 — Loading Price Benchmarks Data")
    logger.info("=" * 70)

    src = os.path.join(SELECTED_DIR, "toronto_price_benchmarks_2026.csv")
    dst = os.path.join(DATA_DIR, "toronto_price_benchmarks_2026.csv")

    if not os.path.exists(src):
        raise FileNotFoundError(f"Price benchmarks not found at {src}")

    shutil.copy2(src, dst)
    df = pd.read_csv(dst, low_memory=False)

    logger.info(f"  Rows:    {len(df)}")
    logger.info(f"  Columns: {list(df.columns)}")

    # Compute summary statistics for price reference
    if "price" in df.columns:
        logger.info(f"\n  === Price Statistics ===")
        logger.info(f"  Mean:   ${df['price'].mean():,.0f}")
        logger.info(f"  Median: ${df['price'].median():,.0f}")
        logger.info(f"  Std:    ${df['price'].std():,.0f}")
        logger.info(f"  Min:    ${df['price'].min():,.0f}")
        logger.info(f"  Max:    ${df['price'].max():,.0f}")

    file_size = os.path.getsize(dst) / (1024 * 1024)
    logger.info(f"\n  ✓ Copied: {dst} ({file_size:.2f} MB)")

    # Also copy rentfaster canadian data as supplementary
    rent_src = os.path.join(SELECTED_DIR, "rentfaster_canadian.csv")
    rent_dst = os.path.join(DATA_DIR, "rentfaster_canadian.csv")
    if os.path.exists(rent_src):
        shutil.copy2(rent_src, rent_dst)
        rf_size = os.path.getsize(rent_dst) / (1024 * 1024)
        logger.info(f"  ✓ Copied: {rent_dst} ({rf_size:.2f} MB)")

    return dst


# ══════════════════════════════════════════════════════════════════════
#  MAIN — Run All Steps
# ══════════════════════════════════════════════════════════════════════

def main():
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  FARUD — Full Training Pipeline                                  ║")
    logger.info("║  AI-Powered Rental Fraud & Trust Scoring System                   ║")
    logger.info("║  COMP 385 AI Capstone — Group #2                             ║")
    logger.info("╚" + "═" * 68 + "╝")
    logger.info(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Backend: {BACKEND_DIR}")
    logger.info(f"  Data:    {SELECTED_DIR}")
    logger.info("")

    overall_start = time.time()
    results = {}

    # ── Step 1: Build BERT dataset ──
    try:
        results["step1"] = step1_build_bert_dataset()
        logger.info("  ✓ Step 1 COMPLETE\n")
    except Exception as e:
        logger.error(f"  ✗ Step 1 FAILED: {e}")
        raise

    # ── Step 2: Preprocess rental listings ──
    try:
        results["step2"] = step2_preprocess_rental_listings()
        logger.info("  ✓ Step 2 COMPLETE\n")
    except Exception as e:
        logger.error(f"  ✗ Step 2 FAILED: {e}")
        raise

    # ── Step 3: Train BERT ──
    try:
        results["step3"] = step3_train_bert()
        logger.info("  ✓ Step 3 COMPLETE\n")
    except Exception as e:
        logger.error(f"  ✗ Step 3 FAILED: {e}")
        raise

    # ── Step 4: Train Isolation Forest ──
    try:
        model_path, metrics = step4_train_isolation_forest()
        results["step4"] = {"model_path": model_path, "metrics": metrics}
        logger.info("  ✓ Step 4 COMPLETE\n")
    except Exception as e:
        logger.error(f"  ✗ Step 4 FAILED: {e}")
        raise

    # ── Step 5: Load price benchmarks ──
    try:
        results["step5"] = step5_load_price_benchmarks()
        logger.info("  ✓ Step 5 COMPLETE\n")
    except Exception as e:
        logger.error(f"  ✗ Step 5 FAILED: {e}")
        raise

    # ── Summary ──
    total_time = time.time() - overall_start
    logger.info("=" * 70)
    logger.info("  TRAINING PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total time: {total_time / 60:.1f} minutes")
    logger.info(f"  Step 1: fraud_dataset.csv   → {results['step1']}")
    logger.info(f"  Step 2: processed features  → {results['step2']}")
    logger.info(f"  Step 3: BERT model          → {results['step3']['model_path']}")
    logger.info(f"  Step 4: Isolation Forest     → {results['step4']['model_path']}")
    logger.info(f"  Step 5: Price benchmarks    → {results['step5']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
