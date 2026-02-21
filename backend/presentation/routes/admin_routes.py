from fastapi import APIRouter, Depends, HTTPException, Form, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List
from infrastructure.database import (
    get_db, UserModel, DatasetModel, MLModelModel, 
    RiskAnalysisModel, SubscriptionPlanModel, FeedbackModel,
    AuditLogModel, ListingModel
)
from presentation.schemas import (
    UserResponse, UserUpdate,
    AuditLogResponse, DashboardStats, SubscriptionPlanResponse, SubscriptionPlanCreate,
    SubscriptionPlanUpdate
)
from presentation.dependencies import get_current_admin, get_client_ip
from application.use_cases.user_use_cases import UserUseCases
import os

router = APIRouter(prefix="/admin", tags=["Admin"])


# Dashboard & Statistics
@router.get("/dashboard", response_model=DashboardStats)
async def get_dashboard_stats(
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get dashboard statistics"""
    
    # Count total users
    total_users = await db.scalar(select(func.count(UserModel.id)))
    
    # Count total analyses
    total_analyses = await db.scalar(select(func.count(RiskAnalysisModel.id)))
    
    # Count high-risk analyses (case-insensitive match for "high" / "Very High")
    high_risk_analyses = await db.scalar(
        select(func.count(RiskAnalysisModel.id)).where(
            func.lower(RiskAnalysisModel.risk_level).in_(["high", "very high", "very_high"])
        )
    )
    
    return {
        "total_users": total_users or 0,
        "total_analyses": total_analyses or 0,
        "high_risk_analyses": high_risk_analyses or 0
    }


# ============================================================================
# SYSTEM DATASET DISCOVERY & COMPREHENSIVE ANALYSIS REPORTS
# ============================================================================

@router.get("/datasets")
async def list_system_datasets(
    current_admin = Depends(get_current_admin)
):
    """
    Discover all datasets used by the FARUD system.
    
    Scans the following directories for CSV files:
      - DATA/selected_datasets/  (source training datasets)
      - backend/data/            (generated/processed datasets)
      - backend/data/processed/  (preprocessed feature datasets)
    
    Returns dataset metadata including size, row/column counts,
    purpose in the pipeline, and quick statistics.
    """
    import pandas as pd
    import json
    from datetime import datetime
    
    BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PROJECT_DIR = os.path.dirname(BACKEND_DIR)
    SELECTED_DIR = os.path.join(PROJECT_DIR, "DATA", "selected_datasets")
    DATA_DIR = os.path.join(BACKEND_DIR, "data")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
    
    # ── Dataset registry: maps filename → metadata ──
    dataset_registry = {
        # Source training datasets (DATA/selected_datasets/)
        "rental_listings_dataset.csv": {
            "category": "source",
            "purpose": "Legitimate rental listings used for BERT training (Step 1) and IsolationForest feature engineering (Step 2)",
            "pipeline_steps": ["Step 1: BERT Dataset Builder", "Step 2: Feature Engineering"],
            "tags": ["training", "legitimate", "listings"],
        },
        "toronto_rent_scam_messages_rich.csv": {
            "category": "source",
            "purpose": "Rich scam messages with 10 scam type categories for BERT fraud classification training (Step 1)",
            "pipeline_steps": ["Step 1: BERT Dataset Builder"],
            "tags": ["training", "scam", "messages"],
        },
        "toronto_rent_scam_messages.csv": {
            "category": "source",
            "purpose": "Basic scam messages dataset for supplementing BERT training data (Step 1)",
            "pipeline_steps": ["Step 1: BERT Dataset Builder"],
            "tags": ["training", "scam", "messages"],
        },
        "toronto_price_benchmarks_2026.csv": {
            "category": "source",
            "purpose": "Toronto price benchmarks for price anomaly detection — flags listings above/below market rates (Step 5)",
            "pipeline_steps": ["Step 5: Price Benchmarks"],
            "tags": ["reference", "pricing", "benchmarks"],
        },
        "rentfaster_canadian.csv": {
            "category": "source",
            "purpose": "RentFaster Canadian rental data used as supplementary price reference data (Step 5)",
            "pipeline_steps": ["Step 5: Price Benchmarks"],
            "tags": ["reference", "pricing", "supplementary"],
        },
        "apartments_for_rent_100K.csv": {
            "category": "source",
            "purpose": "Large apartment rental listings dataset — available for future model enhancement",
            "pipeline_steps": [],
            "tags": ["available", "listings", "large-scale"],
        },
        "scam_messages_risk_labeled.csv": {
            "category": "source",
            "purpose": "Risk-labeled scam messages dataset — available for future risk scoring enhancement",
            "pipeline_steps": [],
            "tags": ["available", "scam", "risk-labeled"],
        },
        # Generated datasets (backend/data/)
        "fraud_dataset.csv": {
            "category": "generated",
            "purpose": "Combined fraud/legitimate text dataset generated by Step 1 — used directly by BERT DistilBERT fine-tuning (Step 3)",
            "pipeline_steps": ["Step 1: Generated Output", "Step 3: BERT Training Input"],
            "tags": ["generated", "bert", "training"],
        },
        "image_hashes.json": {
            "category": "reference",
            "purpose": "Known image hash database used by the image analysis engine for reverse image fraud detection",
            "pipeline_steps": ["Image Analysis Engine"],
            "tags": ["reference", "images", "hashes"],
        },
        # Processed datasets (backend/data/processed/)
        "rental_listings_processed.csv": {
            "category": "processed",
            "purpose": "Fully preprocessed and feature-engineered rental listings used by IsolationForest anomaly detection (Step 4)",
            "pipeline_steps": ["Step 2: Generated Output", "Step 4: IsolationForest Training Input"],
            "tags": ["processed", "features", "isolation-forest"],
        },
    }
    
    datasets = []
    dataset_id = 0
    
    # ── Scan directories ──
    scan_dirs = [
        ("source", SELECTED_DIR, "DATA/selected_datasets"),
        ("generated", DATA_DIR, "backend/data"),
        ("processed", PROCESSED_DIR, "backend/data/processed"),
    ]
    
    for category_default, scan_path, display_path in scan_dirs:
        if not os.path.exists(scan_path):
            continue
        
        for fname in sorted(os.listdir(scan_path)):
            fpath = os.path.join(scan_path, fname)
            if not os.path.isfile(fpath):
                continue
            if not fname.endswith(('.csv', '.json')):
                continue
            
            # Skip synthetic/upload leftovers in backend/data
            if scan_path == DATA_DIR and fname.startswith("synthetic_dataset"):
                continue
            
            dataset_id += 1
            
            # Get file stats
            file_stat = os.stat(fpath)
            file_size_mb = round(file_stat.st_size / (1024 * 1024), 2)
            modified_at = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            
            # Registry lookup
            registry = dataset_registry.get(fname, {})
            purpose = registry.get("purpose", "Dataset file in the system")
            category = registry.get("category", category_default)
            pipeline_steps = registry.get("pipeline_steps", [])
            tags = registry.get("tags", [])
            
            # Quick stats
            row_count = None
            col_count = None
            columns = None
            
            if fname.endswith('.csv'):
                try:
                    df = pd.read_csv(fpath, nrows=0)
                    columns = list(df.columns)
                    col_count = len(columns)
                    # Count rows efficiently
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        row_count = sum(1 for _ in f) - 1  # subtract header
                except Exception:
                    pass
            elif fname.endswith('.json'):
                try:
                    with open(fpath, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        row_count = len(data)
                    elif isinstance(data, dict):
                        row_count = len(data)
                except Exception:
                    pass
            
            # Determine status
            status = "active" if pipeline_steps else "available"
            
            datasets.append({
                "id": f"ds_{dataset_id}",
                "name": fname,
                "category": category,
                "location": f"{display_path}/{fname}",
                "purpose": purpose,
                "pipeline_steps": pipeline_steps,
                "tags": tags,
                "status": status,
                "file_size_mb": file_size_mb,
                "row_count": row_count,
                "col_count": col_count,
                "columns": columns,
                "modified_at": modified_at,
            })
    
    return {
        "datasets": datasets,
        "total": len(datasets),
        "summary": {
            "source": sum(1 for d in datasets if d["category"] == "source"),
            "generated": sum(1 for d in datasets if d["category"] == "generated"),
            "processed": sum(1 for d in datasets if d["category"] == "processed"),
            "reference": sum(1 for d in datasets if d["category"] == "reference"),
            "active": sum(1 for d in datasets if d["status"] == "active"),
            "available": sum(1 for d in datasets if d["status"] == "available"),
        },
    }


@router.get("/datasets/{dataset_id}/report")
async def get_dataset_report(
    dataset_id: str,
    current_admin = Depends(get_current_admin)
):
    """
    Generate a comprehensive analysis report for a system dataset.
    
    Returns:
      - Overview (name, purpose, category, pipeline role)
      - Schema Analysis (column types, dtypes breakdown)
      - Data Quality (missing values, duplicates, completeness)
      - Statistical Summary (numeric column stats, distributions)
      - Categorical Analysis (unique values, top categories)
      - Text Analysis (avg length, vocabulary size — for text columns)
      - Label Distribution (class balance for labeled datasets)
      - Data Lineage (where the dataset comes from, what uses it)
      - Recommendations
    """
    import pandas as pd
    import json
    import numpy as np
    from datetime import datetime
    
    BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PROJECT_DIR = os.path.dirname(BACKEND_DIR)
    SELECTED_DIR = os.path.join(PROJECT_DIR, "DATA", "selected_datasets")
    DATA_DIR = os.path.join(BACKEND_DIR, "data")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
    
    # ── Resolve dataset_id to file path ──
    scan_dirs = [
        ("source", SELECTED_DIR, "DATA/selected_datasets"),
        ("generated", DATA_DIR, "backend/data"),
        ("processed", PROCESSED_DIR, "backend/data/processed"),
    ]
    
    target_file = None
    target_name = None
    target_location = None
    counter = 0
    
    for category_default, scan_path, display_path in scan_dirs:
        if not os.path.exists(scan_path):
            continue
        for fname in sorted(os.listdir(scan_path)):
            fpath = os.path.join(scan_path, fname)
            if not os.path.isfile(fpath):
                continue
            if not fname.endswith(('.csv', '.json')):
                continue
            if scan_path == DATA_DIR and fname.startswith("synthetic_dataset"):
                continue
            counter += 1
            if f"ds_{counter}" == dataset_id:
                target_file = fpath
                target_name = fname
                target_location = f"{display_path}/{fname}"
                break
        if target_file:
            break
    
    if not target_file or not os.path.exists(target_file):
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    
    # ── Dataset registry for lineage ──
    lineage_map = {
        "rental_listings_dataset.csv": {
            "origin": "Scraped/curated rental listings from Canadian rental platforms",
            "used_by": ["BERT Dataset Builder (Step 1)", "Feature Engineering Pipeline (Step 2)"],
            "generates": ["fraud_dataset.csv (via Step 1)", "rental_listings_processed.csv (via Step 2)"],
            "models_trained": ["DistilBERT Fraud Classifier", "IsolationForest Anomaly Detector"],
        },
        "toronto_rent_scam_messages_rich.csv": {
            "origin": "Curated Toronto rental scam messages with 10 scam type categories",
            "used_by": ["BERT Dataset Builder (Step 1)"],
            "generates": ["fraud_dataset.csv (combined with legitimate texts)"],
            "models_trained": ["DistilBERT Fraud Classifier"],
        },
        "toronto_rent_scam_messages.csv": {
            "origin": "Basic Toronto rental scam message collection",
            "used_by": ["BERT Dataset Builder (Step 1) — fraud-only rows"],
            "generates": ["fraud_dataset.csv (supplementary fraud texts)"],
            "models_trained": ["DistilBERT Fraud Classifier"],
        },
        "toronto_price_benchmarks_2026.csv": {
            "origin": "Toronto 2026 rental price benchmark data by area/ward/property type",
            "used_by": ["Price Benchmark Loader (Step 5)", "Indicator Engine (price anomaly detection)"],
            "generates": [],
            "models_trained": [],
        },
        "rentfaster_canadian.csv": {
            "origin": "RentFaster.ca Canadian rental listings dataset",
            "used_by": ["Price Benchmark Loader (Step 5) — supplementary reference"],
            "generates": [],
            "models_trained": [],
        },
        "fraud_dataset.csv": {
            "origin": "Generated by Step 1 — merges legitimate listings + scam messages, class-balanced",
            "used_by": ["BERT DistilBERT Fine-Tuning (Step 3)"],
            "generates": ["bert_fraud_models/latest/ (model.safetensors)"],
            "models_trained": ["DistilBERT Fraud Classifier"],
        },
        "rental_listings_processed.csv": {
            "origin": "Generated by Step 2 — full preprocessing + feature engineering pipeline output",
            "used_by": ["IsolationForest Training (Step 4)"],
            "generates": ["IsolationForest model (isolation_forest.joblib)"],
            "models_trained": ["IsolationForest Anomaly Detector"],
        },
        "image_hashes.json": {
            "origin": "Curated database of known image hashes for fraud detection",
            "used_by": ["Image Analysis Engine (reverse image search)"],
            "generates": [],
            "models_trained": [],
        },
    }
    
    # ── Build report ──
    file_stat = os.stat(target_file)
    file_size_mb = round(file_stat.st_size / (1024 * 1024), 2)
    
    report = {
        "dataset_id": dataset_id,
        "dataset_name": target_name,
        "location": target_location,
        "file_size_mb": file_size_mb,
        "generated_at": datetime.now().isoformat(),
    }
    
    lineage = lineage_map.get(target_name, {})
    
    # ── Handle JSON files ──
    if target_name.endswith('.json'):
        try:
            with open(target_file, 'r') as f:
                data = json.load(f)
            
            report["overview"] = {
                "file_type": "JSON",
                "purpose": lineage.get("origin", "JSON data file"),
                "total_entries": len(data) if isinstance(data, (list, dict)) else "N/A",
                "data_structure": "Array" if isinstance(data, list) else "Object" if isinstance(data, dict) else type(data).__name__,
                "file_size": f"{file_size_mb} MB",
            }
            
            if isinstance(data, dict):
                report["schema_analysis"] = {
                    "top_level_keys": len(data),
                    "sample_keys": list(data.keys())[:20],
                    "value_types": {k: type(v).__name__ for k, v in list(data.items())[:10]},
                }
            elif isinstance(data, list) and len(data) > 0:
                sample = data[0]
                report["schema_analysis"] = {
                    "total_records": len(data),
                    "record_structure": type(sample).__name__,
                    "sample_keys": list(sample.keys()) if isinstance(sample, dict) else [],
                }
            
            report["data_lineage"] = lineage if lineage else {
                "origin": "System data file",
                "used_by": [],
                "generates": [],
                "models_trained": [],
            }
            
            report["recommendations"] = [
                {"type": "info", "message": f"JSON file with {len(data) if isinstance(data, (list, dict)) else 'unknown'} entries"},
            ]
            
            return report
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to analyze JSON file: {str(e)}")
    
    # ── CSV Analysis ──
    try:
        df = pd.read_csv(target_file, low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")
    
    # ── Overview ──
    report["overview"] = {
        "file_type": "CSV",
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "file_size": f"{file_size_mb} MB",
        "memory_usage": f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB",
        "purpose": lineage.get("origin", "Dataset in the FARUD system"),
    }
    
    # ── Schema Analysis ──
    dtype_counts = df.dtypes.value_counts()
    schema = {
        "columns": [],
        "dtype_summary": {str(k): int(v) for k, v in dtype_counts.items()},
    }
    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "non_null": int(df[col].notna().sum()),
            "null_count": int(df[col].isna().sum()),
            "null_pct": round(df[col].isna().mean() * 100, 2),
            "unique": int(df[col].nunique()),
        }
        sample_vals = df[col].dropna().head(3).tolist()
        col_info["sample_values"] = [str(v)[:100] for v in sample_vals]
        schema["columns"].append(col_info)
    report["schema_analysis"] = schema
    
    # ── Data Quality ──
    total_cells = len(df) * len(df.columns)
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    complete_rows = int(df.dropna().shape[0])
    completeness = round((1 - missing_cells / total_cells) * 100, 2) if total_cells > 0 else 100
    
    quality = {
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_pct": round(missing_cells / total_cells * 100, 2) if total_cells > 0 else 0,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round(duplicate_rows / len(df) * 100, 2) if len(df) > 0 else 0,
        "complete_rows": complete_rows,
        "complete_rows_pct": round(complete_rows / len(df) * 100, 2) if len(df) > 0 else 0,
        "completeness_score": completeness,
    }
    
    missing_by_col = df.isna().sum()
    cols_with_missing = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
    if len(cols_with_missing) > 0:
        quality["columns_with_missing"] = {
            col: {"count": int(cnt), "pct": round(cnt / len(df) * 100, 2)}
            for col, cnt in cols_with_missing.head(10).items()
        }
    
    if completeness >= 95 and duplicate_rows == 0:
        quality["grade"] = "A"
        quality["grade_label"] = "Excellent"
    elif completeness >= 85:
        quality["grade"] = "B"
        quality["grade_label"] = "Good"
    elif completeness >= 70:
        quality["grade"] = "C"
        quality["grade_label"] = "Fair"
    else:
        quality["grade"] = "D"
        quality["grade_label"] = "Needs Improvement"
    
    report["data_quality"] = quality
    
    # ── Statistical Summary (numeric columns) ──
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        stats_list = []
        for col in numeric_cols[:20]:
            desc = df[col].describe()
            stats_list.append({
                "column": col,
                "mean": round(float(desc.get("mean", 0)), 4),
                "std": round(float(desc.get("std", 0)), 4),
                "min": round(float(desc.get("min", 0)), 4),
                "q25": round(float(desc.get("25%", 0)), 4),
                "median": round(float(desc.get("50%", 0)), 4),
                "q75": round(float(desc.get("75%", 0)), 4),
                "max": round(float(desc.get("max", 0)), 4),
                "skew": round(float(df[col].skew()), 4) if df[col].notna().sum() > 2 else None,
            })
        report["statistical_summary"] = {
            "numeric_column_count": len(numeric_cols),
            "columns": stats_list,
        }
    
    # ── Categorical Analysis ──
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        cat_analysis = []
        for col in cat_cols[:15]:
            val_counts = df[col].value_counts().head(10)
            cat_analysis.append({
                "column": col,
                "unique_values": int(df[col].nunique()),
                "top_values": {str(k): int(v) for k, v in val_counts.items()},
                "avg_length": round(df[col].dropna().astype(str).str.len().mean(), 1) if df[col].notna().any() else 0,
            })
        report["categorical_analysis"] = {
            "categorical_column_count": len(cat_cols),
            "columns": cat_analysis,
        }
    
    # ── Text Analysis (for text-heavy datasets) ──
    text_candidates = ["text", "message", "description", "title", "content", "listing_text"]
    text_cols_found = [c for c in text_candidates if c in df.columns]
    if text_cols_found:
        text_analysis = {}
        for col in text_cols_found:
            texts = df[col].dropna().astype(str)
            if len(texts) == 0:
                continue
            lengths = texts.str.len()
            word_counts = texts.str.split().str.len()
            text_analysis[col] = {
                "total_texts": int(len(texts)),
                "avg_char_length": round(float(lengths.mean()), 1),
                "min_char_length": int(lengths.min()),
                "max_char_length": int(lengths.max()),
                "avg_word_count": round(float(word_counts.mean()), 1),
                "vocabulary_size": int(len(set(" ".join(texts.head(5000).tolist()).lower().split()))),
                "empty_or_short": int((lengths < 10).sum()),
            }
        if text_analysis:
            report["text_analysis"] = text_analysis
    
    # ── Label Distribution (for labeled datasets) ──
    label_candidates = ["label", "is_fraud", "fraud", "fraud_label", "target", "scam_type"]
    for lbl_col in label_candidates:
        if lbl_col in df.columns:
            val_counts = df[lbl_col].value_counts()
            total = len(df)
            distribution = {}
            for val, cnt in val_counts.items():
                distribution[str(val)] = {
                    "count": int(cnt),
                    "percentage": round(cnt / total * 100, 2),
                }
            
            if len(val_counts) == 2:
                ratio = val_counts.min() / val_counts.max()
                if ratio >= 0.8:
                    balance = "Balanced"
                elif ratio >= 0.4:
                    balance = "Moderately Imbalanced"
                else:
                    balance = "Highly Imbalanced"
            else:
                balance = f"{len(val_counts)} classes"
            
            report["label_distribution"] = {
                "label_column": lbl_col,
                "classes": len(val_counts),
                "distribution": distribution,
                "class_balance": balance,
            }
            break
    
    # ── Data Preview (first 5 rows) ──
    preview_rows = df.head(5).fillna("").to_dict(orient="records")
    report["data_preview"] = {
        "columns": list(df.columns),
        "rows": [{k: str(v)[:200] for k, v in row.items()} for row in preview_rows],
    }
    
    # ── Data Lineage ──
    report["data_lineage"] = lineage if lineage else {
        "origin": "Dataset file in the FARUD system",
        "used_by": [],
        "generates": [],
        "models_trained": [],
    }
    
    # ── Recommendations ──
    recommendations = []
    
    if completeness >= 95:
        recommendations.append({"type": "success", "message": f"Excellent data completeness at {completeness}% — ready for ML training"})
    elif completeness >= 80:
        recommendations.append({"type": "warning", "message": f"Data completeness at {completeness}% — consider imputation for missing values"})
    else:
        recommendations.append({"type": "warning", "message": f"Low data completeness at {completeness}% — significant missing data may affect model quality"})
    
    if duplicate_rows > 0:
        recommendations.append({"type": "warning", "message": f"{duplicate_rows} duplicate rows detected ({quality['duplicate_pct']}%) — may bias model training"})
    else:
        recommendations.append({"type": "success", "message": "No duplicate rows detected"})
    
    if "label_distribution" in report:
        balance = report["label_distribution"]["class_balance"]
        if "Balanced" in balance and "Imbalanced" not in balance:
            recommendations.append({"type": "success", "message": f"Class distribution is {balance.lower()} — good for classifier training"})
        elif "Imbalanced" in balance:
            recommendations.append({"type": "warning", "message": f"Class distribution is {balance.lower()} — consider oversampling/undersampling techniques"})
    
    if len(df) < 100:
        recommendations.append({"type": "warning", "message": f"Small dataset ({len(df)} rows) — may not be sufficient for robust model training"})
    elif len(df) >= 10000:
        recommendations.append({"type": "success", "message": f"Large dataset ({len(df):,} rows) — sufficient volume for reliable model training"})
    else:
        recommendations.append({"type": "info", "message": f"Dataset has {len(df):,} rows — adequate for model training"})
    
    if text_cols_found:
        recommendations.append({"type": "info", "message": f"Text columns detected ({', '.join(text_cols_found)}) — suitable for NLP/BERT-based analysis"})
    
    if numeric_cols:
        recommendations.append({"type": "info", "message": f"{len(numeric_cols)} numeric features available for statistical/ML analysis"})
    
    if lineage.get("models_trained"):
        recommendations.append({"type": "success", "message": f"Actively used to train: {', '.join(lineage['models_trained'])}"})
    
    report["recommendations"] = recommendations
    
    return report


# ============================================================================
# TRAINED MODEL DISCOVERY & COMPREHENSIVE ANALYSIS REPORTS
# ============================================================================

@router.get("/trained-models")
async def list_trained_models(
    current_admin = Depends(get_current_admin)
):
    """
    Discover all trained models on disk (BERT + IsolationForest).
    
    Scans the backend/models/ directory for:
      - bert_fraud_models/  (DistilBERT fine-tuned classifiers)
      - IsolationForest_*/  (unsupervised anomaly detectors)
    
    Returns metadata, metrics, and file info for each discovered model.
    """
    import json
    import glob

    BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(BACKEND_DIR, "models")

    trained_models = []

    # ── Discover BERT models ──
    bert_base = os.path.join(models_dir, "bert_fraud_models")
    if os.path.isdir(bert_base):
        for entry in sorted(os.listdir(bert_base)):
            model_dir = os.path.join(bert_base, entry)
            if not os.path.isdir(model_dir):
                continue

            safetensors_path = os.path.join(model_dir, "model.safetensors")
            if not os.path.exists(safetensors_path):
                continue

            # Load metrics
            metrics = {}
            metrics_path = os.path.join(model_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)

            # Load training config
            training_config = {}
            config_path = os.path.join(model_dir, "training_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    training_config = json.load(f)

            # Load model architecture config
            arch_config = {}
            arch_path = os.path.join(model_dir, "config.json")
            if os.path.exists(arch_path):
                with open(arch_path, "r") as f:
                    arch_config = json.load(f)

            # File sizes
            files_info = {}
            for fname in os.listdir(model_dir):
                fpath = os.path.join(model_dir, fname)
                if os.path.isfile(fpath):
                    files_info[fname] = {
                        "size_bytes": os.path.getsize(fpath),
                        "size_mb": round(os.path.getsize(fpath) / (1024 * 1024), 2),
                    }

            is_latest = (entry == "latest")

            trained_models.append({
                "id": f"bert_{entry}",
                "name": entry,
                "model_type": "DistilBERT",
                "algorithm": "Fine-tuned DistilBERT for Sequence Classification",
                "category": "supervised",
                "is_latest": is_latest,
                "model_dir": model_dir,
                "metrics": metrics,
                "training_config": training_config,
                "architecture": arch_config,
                "files": files_info,
                "status": "active" if is_latest else "archived",
            })

    # ── Discover Isolation Forest models ──
    for entry in sorted(os.listdir(models_dir)):
        if not entry.startswith("IsolationForest_"):
            continue
        model_dir = os.path.join(models_dir, entry)
        if not os.path.isdir(model_dir):
            continue

        joblib_path = os.path.join(model_dir, "isolation_forest.joblib")
        if not os.path.exists(joblib_path):
            continue

        # Load metrics from metrics/ directory
        metrics = {}
        metrics_path = os.path.join(models_dir, "metrics", f"{entry}_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

        # Load feature names
        feature_names = []
        features_path = os.path.join(model_dir, "feature_names.json")
        if os.path.exists(features_path):
            with open(features_path, "r") as f:
                feature_names = json.load(f)

        # File sizes
        files_info = {}
        for fname in os.listdir(model_dir):
            fpath = os.path.join(model_dir, fname)
            if os.path.isfile(fpath):
                files_info[fname] = {
                    "size_bytes": os.path.getsize(fpath),
                    "size_mb": round(os.path.getsize(fpath) / (1024 * 1024), 2),
                }

        trained_models.append({
            "id": f"iforest_{entry}",
            "name": entry,
            "model_type": "IsolationForest",
            "algorithm": "Unsupervised Anomaly Detection (scikit-learn)",
            "category": "unsupervised",
            "is_latest": True,
            "model_dir": model_dir,
            "metrics": metrics,
            "training_config": {
                "n_estimators": metrics.get("n_estimators", 200),
                "contamination": metrics.get("contamination", 0.05),
                "max_features": 1.0,
                "random_state": 42,
            },
            "architecture": {
                "n_features": metrics.get("n_features", len(feature_names)),
                "feature_names": feature_names,
            },
            "files": files_info,
            "status": "active",
        })

    return {
        "total_models": len(trained_models),
        "models": trained_models,
    }


@router.get("/trained-models/{model_id}/report")
async def get_trained_model_report(
    model_id: str,
    current_admin = Depends(get_current_admin)
):
    """
    Get comprehensive analysis report for a trained model.
    
    Returns:
      - Model overview (type, algorithm, architecture)
      - Performance metrics (accuracy, precision, recall, F1, ROC AUC)
      - Training configuration (hyperparameters)
      - Training history (loss curves per epoch)
      - Confusion matrix analysis
      - Feature importance (IsolationForest) or architecture details (BERT)
      - Model file inventory with sizes
      - Deployment readiness checklist
      - Business impact analysis
      - Recommendations
    """
    import json

    BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(BACKEND_DIR, "models")

    # ── Resolve model directory ──
    if model_id.startswith("bert_"):
        model_name = model_id[5:]  # strip "bert_" prefix
        model_dir = os.path.join(models_dir, "bert_fraud_models", model_name)
        model_type = "DistilBERT"
    elif model_id.startswith("iforest_"):
        model_name = model_id[8:]  # strip "iforest_" prefix
        model_dir = os.path.join(models_dir, model_name)
        model_type = "IsolationForest"
    else:
        raise HTTPException(status_code=400, detail="Invalid model ID format")

    if not os.path.isdir(model_dir):
        raise HTTPException(status_code=404, detail="Model directory not found")

    report = {
        "model_id": model_id,
        "model_name": model_name,
        "model_type": model_type,
        "generated_at": __import__('datetime').datetime.now().isoformat(),
    }

    # ── BERT Comprehensive Report ──
    if model_type == "DistilBERT":
        # Load metrics
        metrics = {}
        metrics_path = os.path.join(model_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

        # Load training config
        training_config = {}
        config_path = os.path.join(model_dir, "training_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                training_config = json.load(f)

        # Load architecture config
        arch_config = {}
        arch_path = os.path.join(model_dir, "config.json")
        if os.path.exists(arch_path):
            with open(arch_path, "r") as f:
                arch_config = json.load(f)

        # Overview
        report["overview"] = {
            "algorithm": "DistilBERT for Sequence Classification",
            "base_model": training_config.get("model_name", "distilbert-base-uncased"),
            "task": "Binary Classification (Fraud vs Legitimate)",
            "approach": "Transfer Learning — Fine-tuned pre-trained transformer",
            "framework": "PyTorch + HuggingFace Transformers",
            "architecture_type": arch_config.get("model_type", "distilbert"),
            "num_labels": 2,
            "is_latest": model_name == "latest",
        }

        # Architecture details
        report["architecture"] = {
            "hidden_size": arch_config.get("dim", 768),
            "num_attention_heads": arch_config.get("n_heads", 12),
            "num_hidden_layers": arch_config.get("n_layers", 6),
            "intermediate_size": arch_config.get("hidden_dim", 3072),
            "vocab_size": arch_config.get("vocab_size", 30522),
            "max_position_embeddings": arch_config.get("max_position_embeddings", 512),
            "activation_function": arch_config.get("activation", "gelu"),
            "attention_dropout": arch_config.get("attention_dropout", 0.1),
            "hidden_dropout": arch_config.get("dropout", 0.1),
            "classifier_dropout": arch_config.get("seq_classif_dropout", 0.2),
        }

        # Training configuration
        report["training_config"] = {
            "epochs": training_config.get("num_epochs", 4),
            "batch_size": training_config.get("batch_size", 16),
            "learning_rate": training_config.get("learning_rate", 2e-5),
            "warmup_ratio": training_config.get("warmup_ratio", 0.1),
            "weight_decay": training_config.get("weight_decay", 0.01),
            "max_sequence_length": training_config.get("max_length", 256),
            "optimizer": "AdamW",
            "scheduler": "Linear warmup + decay",
            "train_split": 1.0 - training_config.get("test_size", 0.2) - training_config.get("val_size", 0.1),
            "val_split": training_config.get("val_size", 0.1),
            "test_split": training_config.get("test_size", 0.2),
            "random_seed": training_config.get("random_seed", 42),
        }

        # Performance metrics
        accuracy = metrics.get("accuracy", 0)
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)
        f1 = metrics.get("f1_score", 0)
        roc_auc = metrics.get("roc_auc", 0)
        cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])

        tn = cm[0][0] if len(cm) > 0 and len(cm[0]) > 0 else 0
        fp = cm[0][1] if len(cm) > 0 and len(cm[0]) > 1 else 0
        fn = cm[1][0] if len(cm) > 1 and len(cm[1]) > 0 else 0
        tp = cm[1][1] if len(cm) > 1 and len(cm[1]) > 1 else 0
        total_samples = tn + fp + fn + tp

        report["performance_metrics"] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "false_positive_rate": fp / (tn + fp) if (tn + fp) > 0 else 0,
            "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
            "total_test_samples": total_samples,
        }

        # Confusion matrix analysis
        report["confusion_matrix"] = {
            "matrix": cm,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "labels": ["Legitimate (0)", "Fraud (1)"],
            "interpretation": {
                "correctly_identified_legit": tn,
                "false_alarms": fp,
                "missed_fraud": fn,
                "correctly_caught_fraud": tp,
            },
        }

        # Training history
        training_loss = metrics.get("training_loss_history", [])
        validation_loss = metrics.get("validation_loss_history", [])
        report["training_history"] = {
            "training_loss_per_epoch": training_loss,
            "validation_loss_per_epoch": validation_loss,
            "best_epoch": metrics.get("best_epoch", len(training_loss)),
            "total_training_time_seconds": metrics.get("total_training_time", 0),
            "total_training_time_minutes": round(metrics.get("total_training_time", 0) / 60, 1),
            "convergence_analysis": {
                "initial_train_loss": training_loss[0] if training_loss else None,
                "final_train_loss": training_loss[-1] if training_loss else None,
                "initial_val_loss": validation_loss[0] if validation_loss else None,
                "final_val_loss": validation_loss[-1] if validation_loss else None,
                "loss_reduction_pct": round((1 - training_loss[-1] / training_loss[0]) * 100, 2) if training_loss and training_loss[0] > 0 else 0,
                "overfitting_detected": (validation_loss[-1] > validation_loss[-2]) if len(validation_loss) >= 2 else False,
            },
        }

        # Business impact analysis
        report["business_impact"] = {
            "fraud_detection_rate": f"{recall * 100:.2f}%",
            "false_alarm_rate": f"{(fp / (tn + fp) * 100) if (tn + fp) > 0 else 0:.2f}%",
            "estimated_fraud_caught_per_1000": round(recall * 1000),
            "estimated_false_alarms_per_1000": round((fp / (tn + fp)) * 1000) if (tn + fp) > 0 else 0,
            "operational_efficiency": f"{accuracy * 100:.2f}%",
            "risk_assessment": "LOW" if accuracy > 0.98 and recall > 0.95 else "MEDIUM" if accuracy > 0.90 else "HIGH",
        }

        # Deployment readiness
        readiness_checks = [
            {"check": "Model file exists (model.safetensors)", "passed": os.path.exists(os.path.join(model_dir, "model.safetensors")), "severity": "critical"},
            {"check": "Tokenizer files present", "passed": os.path.exists(os.path.join(model_dir, "vocab.txt")), "severity": "critical"},
            {"check": "Configuration file present", "passed": os.path.exists(os.path.join(model_dir, "config.json")), "severity": "critical"},
            {"check": "Metrics file present", "passed": os.path.exists(os.path.join(model_dir, "metrics.json")), "severity": "high"},
            {"check": "Training config present", "passed": os.path.exists(os.path.join(model_dir, "training_config.json")), "severity": "medium"},
            {"check": "Accuracy > 95%", "passed": accuracy > 0.95, "severity": "high"},
            {"check": "F1 Score > 90%", "passed": f1 > 0.90, "severity": "high"},
            {"check": "ROC AUC > 90%", "passed": roc_auc > 0.90, "severity": "high"},
            {"check": "No missed fraud (FN = 0)", "passed": fn == 0, "severity": "medium"},
            {"check": "Low false alarms (FP ≤ 5)", "passed": fp <= 5, "severity": "medium"},
        ]
        passed_count = sum(1 for c in readiness_checks if c["passed"])
        report["deployment_readiness"] = {
            "checks": readiness_checks,
            "passed": passed_count,
            "total": len(readiness_checks),
            "score": f"{passed_count / len(readiness_checks) * 100:.0f}%",
            "verdict": "READY" if passed_count == len(readiness_checks) else "READY_WITH_WARNINGS" if passed_count >= 7 else "NOT_READY",
        }

        # Recommendations
        recommendations = []
        if accuracy > 0.99:
            recommendations.append({"type": "info", "message": "Exceptional accuracy — verify no data leakage between train/test splits"})
        if fn > 0:
            recommendations.append({"type": "warning", "message": f"{fn} fraud cases missed — consider adjusting classification threshold"})
        if fp > 5:
            recommendations.append({"type": "warning", "message": f"{fp} false alarms — may cause user fatigue, consider threshold tuning"})
        if metrics.get("total_training_time", 0) > 10000:
            recommendations.append({"type": "info", "message": "Long training time — consider GPU acceleration for future training"})
        if not training_loss:
            recommendations.append({"type": "warning", "message": "No training loss history — cannot analyze convergence"})
        if len(training_loss) >= 2 and validation_loss and validation_loss[-1] > validation_loss[-2]:
            recommendations.append({"type": "warning", "message": "Validation loss increased in last epoch — possible overfitting"})
        if accuracy > 0.95 and recall > 0.95:
            recommendations.append({"type": "success", "message": "Model meets production-grade performance thresholds"})
        report["recommendations"] = recommendations

    # ── Isolation Forest Comprehensive Report ──
    elif model_type == "IsolationForest":
        # Load metrics
        metrics = {}
        metrics_path = os.path.join(models_dir, "metrics", f"{model_name}_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

        # Load feature names
        feature_names = []
        features_path = os.path.join(model_dir, "feature_names.json")
        if os.path.exists(features_path):
            with open(features_path, "r") as f:
                feature_names = json.load(f)

        n_samples = metrics.get("n_samples", 0)
        n_anomalies = metrics.get("n_anomalies", 0)
        n_normal = metrics.get("n_normal", 0)

        # Overview
        report["overview"] = {
            "algorithm": "Isolation Forest (scikit-learn)",
            "task": "Unsupervised Anomaly Detection",
            "approach": "Ensemble of random isolation trees",
            "framework": "scikit-learn",
            "purpose": "Detect statistically anomalous rental listings based on engineered features",
        }

        # Architecture / hyperparameters
        report["architecture"] = {
            "n_estimators": metrics.get("n_estimators", 200),
            "contamination": metrics.get("contamination", 0.05),
            "max_features": 1.0,
            "max_samples": "auto",
            "n_features": metrics.get("n_features", len(feature_names)),
            "feature_names": feature_names,
            "random_state": 42,
        }

        # Training configuration
        report["training_config"] = {
            "n_estimators": metrics.get("n_estimators", 200),
            "contamination": metrics.get("contamination", 0.05),
            "max_samples": "auto",
            "max_features": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "scaler": "StandardScaler (zero-mean, unit-variance)",
            "preprocessing": "9-stage feature engineering pipeline",
        }

        # Performance metrics
        anomaly_rate = metrics.get("anomaly_rate", 0)
        report["performance_metrics"] = {
            "n_samples": n_samples,
            "n_anomalies": n_anomalies,
            "n_normal": n_normal,
            "anomaly_rate": anomaly_rate,
            "anomaly_rate_pct": f"{anomaly_rate * 100:.1f}%",
            "score_mean": metrics.get("score_mean", 0),
            "score_std": metrics.get("score_std", 0),
            "score_min": metrics.get("score_min", 0),
            "score_max": metrics.get("score_max", 0),
            "training_time_seconds": metrics.get("training_time_seconds", 0),
            "trained_at": metrics.get("trained_at", "unknown"),
        }

        # Feature analysis
        feature_categories = {
            "price_features": [f for f in feature_names if "price" in f.lower()],
            "text_features": [f for f in feature_names if "tfidf" in f.lower() or "char_count" in f.lower() or "word" in f.lower()],
            "property_features": [f for f in feature_names if any(k in f.lower() for k in ["bedroom", "bathroom", "furnished", "deposit"])],
            "geo_features": [f for f in feature_names if any(k in f.lower() for k in ["geo", "lat", "lon"])],
            "interaction_features": [f for f in feature_names if " " in f],
            "categorical_features": [f for f in feature_names if "count" in f.lower() or "bin" in f.lower()],
        }
        report["feature_analysis"] = {
            "total_features": len(feature_names),
            "feature_categories": feature_categories,
            "category_counts": {k: len(v) for k, v in feature_categories.items()},
        }

        # Anomaly score distribution
        report["score_distribution"] = {
            "mean": metrics.get("score_mean", 0),
            "std": metrics.get("score_std", 0),
            "min": metrics.get("score_min", 0),
            "max": metrics.get("score_max", 0),
            "interpretation": (
                "Scores > 0 indicate normal behavior; scores < 0 indicate anomalous behavior. "
                "The contamination parameter (0.05) means ~5% of training data is treated as anomalous."
            ),
        }

        # Business impact analysis
        report["business_impact"] = {
            "anomalies_detected": n_anomalies,
            "detection_rate": f"{anomaly_rate * 100:.1f}%",
            "normal_listings_verified": n_normal,
            "estimated_anomalies_per_1000": round(anomaly_rate * 1000),
            "use_case": "Supplementary anomaly flagging based on structural listing features",
            "risk_assessment": "LOW" if 0.02 <= anomaly_rate <= 0.10 else "MEDIUM",
        }

        # Deployment readiness
        readiness_checks = [
            {"check": "Model file exists (isolation_forest.joblib)", "passed": os.path.exists(os.path.join(model_dir, "isolation_forest.joblib")), "severity": "critical"},
            {"check": "Scaler file exists (scaler.joblib)", "passed": os.path.exists(os.path.join(model_dir, "scaler.joblib")), "severity": "critical"},
            {"check": "Feature names file exists", "passed": os.path.exists(os.path.join(model_dir, "feature_names.json")), "severity": "high"},
            {"check": "Metrics file exists", "passed": os.path.exists(os.path.join(models_dir, "metrics", f"{model_name}_metrics.json")), "severity": "high"},
            {"check": "Anomaly rate within expected range (2-10%)", "passed": 0.02 <= anomaly_rate <= 0.10, "severity": "medium"},
            {"check": "Sufficient training samples (> 1000)", "passed": n_samples > 1000, "severity": "high"},
            {"check": "Feature count > 10", "passed": len(feature_names) > 10, "severity": "medium"},
            {"check": "Training time < 60s", "passed": metrics.get("training_time_seconds", 999) < 60, "severity": "low"},
        ]
        passed_count = sum(1 for c in readiness_checks if c["passed"])
        report["deployment_readiness"] = {
            "checks": readiness_checks,
            "passed": passed_count,
            "total": len(readiness_checks),
            "score": f"{passed_count / len(readiness_checks) * 100:.0f}%",
            "verdict": "READY" if passed_count == len(readiness_checks) else "READY_WITH_WARNINGS" if passed_count >= 6 else "NOT_READY",
        }

        # Recommendations
        recommendations = []
        if anomaly_rate < 0.02:
            recommendations.append({"type": "warning", "message": "Very low anomaly rate — model may be too conservative"})
        if anomaly_rate > 0.15:
            recommendations.append({"type": "warning", "message": "High anomaly rate — model may be too aggressive, consider tuning contamination"})
        if len(feature_names) < 10:
            recommendations.append({"type": "warning", "message": "Low feature count — consider adding more engineered features"})
        if n_samples < 5000:
            recommendations.append({"type": "info", "message": "Moderate training set — larger datasets may improve generalization"})
        if 0.03 <= anomaly_rate <= 0.08:
            recommendations.append({"type": "success", "message": "Anomaly rate within healthy range for rental fraud detection"})
        recommendations.append({"type": "info", "message": "IsolationForest is used as supplementary signal alongside the primary BERT classifier"})
        report["recommendations"] = recommendations

    # ── File inventory ──
    files_info = {}
    total_size = 0
    for fname in os.listdir(model_dir):
        fpath = os.path.join(model_dir, fname)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            total_size += size
            files_info[fname] = {
                "size_bytes": size,
                "size_mb": round(size / (1024 * 1024), 2),
            }
    report["files"] = {
        "inventory": files_info,
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
    }

    return report


# User Management
@router.get("/users", response_model=List[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    role: str = None,
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """List all users"""
    
    users = await UserUseCases.list_users(db, skip, limit, role)
    return users


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get user details"""
    
    user = await UserUseCases.get_user(db, user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    updates: UserUpdate,
    request: Request,
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Update user information"""
    
    try:
        user = await UserUseCases.update_user(
            db=db,
            user_id=user_id,
            **updates.model_dump(exclude_none=True)
        )
        
        # Log action
        await UserUseCases.log_action(
            db=db,
            user_id=current_admin.id,
            action="user_updated",
            entity_type="user",
            entity_id=user_id,
            details=updates.model_dump(exclude_none=True),
            ip_address=get_client_ip(request)
        )
        
        return user
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/users/{user_id}/deactivate", response_model=UserResponse)
async def deactivate_user(
    user_id: int,
    request: Request,
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Deactivate a user"""
    
    try:
        user = await UserUseCases.deactivate_user(db, user_id)
        
        # Log action
        await UserUseCases.log_action(
            db=db,
            user_id=current_admin.id,
            action="user_deactivated",
            entity_type="user",
            entity_id=user_id,
            ip_address=get_client_ip(request)
        )
        
        return user
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Audit Logs
@router.get("/audit-logs")
async def get_audit_logs(
    skip: int = 0,
    limit: int = 25,
    user_id: int = None,
    action: str = None,
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get audit logs with user email, pagination, and optional filters"""
    
    # Build base query with optional filters
    base_query = select(AuditLogModel)
    count_query = select(func.count(AuditLogModel.id))
    
    if user_id:
        base_query = base_query.where(AuditLogModel.user_id == user_id)
        count_query = count_query.where(AuditLogModel.user_id == user_id)
    
    if action:
        base_query = base_query.where(AuditLogModel.action.ilike(f"%{action}%"))
        count_query = count_query.where(AuditLogModel.action.ilike(f"%{action}%"))
    
    # Get total count
    total = await db.scalar(count_query) or 0
    
    # Get paginated logs
    query = base_query.order_by(AuditLogModel.created_at.desc()).offset(skip).limit(limit)
    result = await db.execute(query)
    logs = result.scalars().all()
    
    # Enrich logs with user email
    items = []
    for log in logs:
        log_dict = {
            "id": log.id,
            "user_id": log.user_id,
            "user_email": None,
            "action": log.action,
            "entity_type": log.entity_type,
            "entity_id": log.entity_id,
            "details": log.details,
            "ip_address": log.ip_address,
            "created_at": log.created_at
        }
        
        if log.user_id:
            user_result = await db.execute(
                select(UserModel).where(UserModel.id == log.user_id)
            )
            user = user_result.scalar_one_or_none()
            if user:
                log_dict["user_email"] = user.email
        
        items.append(log_dict)
    
    return {
        "items": items,
        "total": total,
        "skip": skip,
        "limit": limit,
        "pages": (total + limit - 1) // limit if limit > 0 else 0
    }


# Feedback Review (FR14)
@router.get("/feedback")
async def get_all_feedback(
    skip: int = 0,
    limit: int = 100,
    feedback_type: str = None,
    status: str = None,
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get all user feedback for review"""
    
    query = select(FeedbackModel)
    
    if feedback_type:
        query = query.where(FeedbackModel.feedback_type == feedback_type)
    
    if status:
        query = query.where(FeedbackModel.status == status)
    
    query = query.offset(skip).limit(limit).order_by(FeedbackModel.created_at.desc())
    
    result = await db.execute(query)
    feedback_list = result.scalars().all()
    
    return [
        {
            "id": f.id,
            "analysis_id": f.analysis_id,
            "user_id": f.user_id,
            "feedback_type": f.feedback_type,
            "comments": f.comments,
            "status": f.status or "pending",
            "reviewed_by": f.reviewed_by,
            "reviewed_at": f.reviewed_at.isoformat() if f.reviewed_at else None,
            "created_at": f.created_at.isoformat()
        }
        for f in feedback_list
    ]


@router.get("/feedback/stats")
async def get_feedback_stats(
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get feedback statistics for model improvement insights"""
    
    # Total feedback
    total = await db.scalar(select(func.count(FeedbackModel.id)))
    
    # Feedback by type
    safe_count = await db.scalar(
        select(func.count(FeedbackModel.id)).where(FeedbackModel.feedback_type == "safe")
    )
    fraud_count = await db.scalar(
        select(func.count(FeedbackModel.id)).where(FeedbackModel.feedback_type == "fraud")
    )
    unsure_count = await db.scalar(
        select(func.count(FeedbackModel.id)).where(FeedbackModel.feedback_type == "unsure")
    )
    
    # Feedback by review status
    pending_count = await db.scalar(
        select(func.count(FeedbackModel.id)).where(
            (FeedbackModel.status == "pending") | (FeedbackModel.status.is_(None))
        )
    )
    approved_count = await db.scalar(
        select(func.count(FeedbackModel.id)).where(FeedbackModel.status == "approved")
    )
    rejected_count = await db.scalar(
        select(func.count(FeedbackModel.id)).where(FeedbackModel.status == "rejected")
    )
    
    return {
        "total_feedback": total or 0,
        "safe_reports": safe_count or 0,
        "fraud_reports": fraud_count or 0,
        "unsure_reports": unsure_count or 0,
        "fraud_confirmation_rate": float(fraud_count / total * 100) if total else 0.0,
        "pending_review": pending_count or 0,
        "approved": approved_count or 0,
        "rejected": rejected_count or 0
    }


@router.put("/feedback/{feedback_id}/review")
async def review_feedback(
    feedback_id: int,
    request: Request,
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Approve or reject user feedback before it enters auto-learning"""
    from datetime import datetime as dt
    
    body = await request.json()
    new_status = body.get("status")
    
    if new_status not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="Status must be 'approved' or 'rejected'")
    
    # Find the feedback
    result = await db.execute(select(FeedbackModel).where(FeedbackModel.id == feedback_id))
    feedback = result.scalar_one_or_none()
    
    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found")
    
    # Update review fields
    feedback.status = new_status
    feedback.reviewed_by = current_admin.id
    feedback.reviewed_at = dt.utcnow()
    
    # Audit log
    audit = AuditLogModel(
        user_id=current_admin.id,
        action=f"feedback_{new_status}",
        details=f"Feedback #{feedback_id} {new_status} (type: {feedback.feedback_type}, analysis: {feedback.analysis_id})",
        ip_address=request.client.host if request.client else "unknown"
    )
    db.add(audit)
    
    await db.commit()
    
    return {
        "id": feedback.id,
        "status": feedback.status,
        "reviewed_by": feedback.reviewed_by,
        "reviewed_at": feedback.reviewed_at.isoformat() if feedback.reviewed_at else None,
        "message": f"Feedback #{feedback_id} has been {new_status}"
    }


# ======================== LISTING APPROVAL ========================

@router.get("/listings")
async def get_all_listings_admin(
    skip: int = 0,
    limit: int = 100,
    listing_status: str = None,
    property_type: str = None,
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get all listings for admin review with optional status filter"""
    
    query = select(ListingModel)
    
    if listing_status:
        query = query.where(ListingModel.listing_status == listing_status)
    if property_type:
        query = query.where(ListingModel.property_type == property_type)
    
    query = query.offset(skip).limit(limit).order_by(ListingModel.created_at.desc())
    
    result = await db.execute(query)
    listings = result.scalars().all()
    
    listing_dicts = []
    for l in listings:
        # Look up owner name
        owner = (await db.execute(select(UserModel).where(UserModel.id == l.owner_id))).scalar_one_or_none()
        d = {
            "id": l.id,
            "owner_id": l.owner_id,
            "owner_name": owner.full_name if owner else "Unknown",
            "owner_email": owner.email if owner else "Unknown",
            "title": l.title,
            "address": l.address,
            "city": l.city,
            "province": l.province,
            "postal_code": l.postal_code,
            "price": l.price,
            "beds": l.beds,
            "baths": l.baths,
            "sqft": l.sqft,
            "property_type": l.property_type,
            "description": l.description,
            "amenities": l.amenities or [],
            "laundry": l.laundry,
            "utilities": l.utilities,
            "pet_friendly": l.pet_friendly,
            "parking_included": l.parking_included,
            "available_date": l.available_date,
            "is_active": l.is_active,
            "is_verified": l.is_verified,
            "listing_status": l.listing_status or "pending_review",
            "admin_notes": l.admin_notes,
            "reviewed_by": l.reviewed_by,
            "reviewed_at": l.reviewed_at.isoformat() if l.reviewed_at else None,
            "risk_score": l.risk_score,
            "views": l.views or 0,
            "created_at": l.created_at.isoformat() if l.created_at else None,
        }
        listing_dicts.append(d)
    
    return listing_dicts


@router.get("/listings/stats")
async def get_listing_stats(
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get listing review statistics"""
    
    total = await db.scalar(select(func.count(ListingModel.id)))
    
    pending_count = await db.scalar(
        select(func.count(ListingModel.id)).where(
            (ListingModel.listing_status == "pending_review") | (ListingModel.listing_status.is_(None))
        )
    )
    approved_count = await db.scalar(
        select(func.count(ListingModel.id)).where(ListingModel.listing_status == "approved")
    )
    rejected_count = await db.scalar(
        select(func.count(ListingModel.id)).where(ListingModel.listing_status == "rejected")
    )
    disabled_count = await db.scalar(
        select(func.count(ListingModel.id)).where(ListingModel.listing_status == "disabled")
    )
    
    return {
        "total_listings": total or 0,
        "pending_review": pending_count or 0,
        "approved": approved_count or 0,
        "rejected": rejected_count or 0,
        "disabled": disabled_count or 0,
    }


@router.put("/listings/{listing_id}/review")
async def review_listing(
    listing_id: int,
    request: Request,
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Approve, reject, or disable a landlord listing"""
    from datetime import datetime as dt
    
    body = await request.json()
    new_status = body.get("status")
    admin_notes = body.get("admin_notes", "")
    
    if new_status not in ("approved", "rejected", "disabled"):
        raise HTTPException(status_code=400, detail="Status must be 'approved', 'rejected', or 'disabled'")
    
    # Find the listing
    result = await db.execute(select(ListingModel).where(ListingModel.id == listing_id))
    listing = result.scalar_one_or_none()
    
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")
    
    # Update review fields
    listing.listing_status = new_status
    listing.admin_notes = admin_notes
    listing.reviewed_by = current_admin.id
    listing.reviewed_at = dt.utcnow()
    
    # Auto-activate approved listings, deactivate rejected/disabled ones
    if new_status == "approved":
        listing.is_active = True
        listing.is_verified = True
    elif new_status in ("rejected", "disabled"):
        listing.is_active = False
        listing.is_verified = False
    
    # Audit log
    audit = AuditLogModel(
        user_id=current_admin.id,
        action=f"listing_{new_status}",
        details=f"Listing #{listing_id} '{listing.title}' {new_status} (owner: {listing.owner_id}){' — ' + admin_notes if admin_notes else ''}",
        ip_address=request.client.host if request.client else "unknown"
    )
    db.add(audit)
    
    await db.commit()
    
    return {
        "id": listing.id,
        "listing_status": listing.listing_status,
        "is_active": listing.is_active,
        "is_verified": listing.is_verified,
        "reviewed_by": listing.reviewed_by,
        "reviewed_at": listing.reviewed_at.isoformat() if listing.reviewed_at else None,
        "admin_notes": listing.admin_notes,
        "message": f"Listing #{listing_id} has been {new_status}"
    }


@router.post("/learning/run")
async def run_auto_learning(
    days_back: int = 30,
    request: Request = None,
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Run auto-learning from user feedback.
    
    Analyzes confirmed fraud/safe feedback to:
    - Discover new patterns
    - Calibrate indicator weights
    - Generate actionable insights
    """
    from application.use_cases.auto_learning_engine import auto_learning_engine
    
    try:
        result = await auto_learning_engine.learn_from_feedback(db, days_back)
        
        # Log action
        await UserUseCases.log_action(
            db=db,
            user_id=current_admin.id,
            action="auto_learning_run",
            entity_type="learning",
            entity_id=None,
            details={
                "days_back": days_back,
                "patterns_found": result.get("patterns_found", 0),
                "status": result.get("status")
            },
            ip_address=get_client_ip(request) if request else None
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-learning failed: {str(e)}")


@router.get("/learning/stats")
async def get_learning_stats(
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get current auto-learning statistics"""
    from application.use_cases.auto_learning_engine import auto_learning_engine
    
    return auto_learning_engine.get_learning_stats()


@router.get("/learning/retraining-dataset")
async def get_retraining_dataset(
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Get labeled dataset from confirmed feedback for model retraining.
    
    Returns samples with confirmed fraud/safe labels for supervised learning.
    """
    from application.use_cases.auto_learning_engine import auto_learning_engine
    
    samples = await auto_learning_engine.get_retraining_dataset(db)
    
    return {
        "total_samples": len(samples),
        "fraud_samples": len([s for s in samples if s["label"] == 1]),
        "safe_samples": len([s for s in samples if s["label"] == 0]),
        "samples": samples
    }


# =====================================================================
# SUBSCRIPTION PLAN MANAGEMENT
# =====================================================================

@router.get("/subscription-plans", response_model=List[SubscriptionPlanResponse])
async def get_subscription_plans(
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get all subscription plans (including inactive)"""
    
    result = await db.execute(select(SubscriptionPlanModel))
    plans = result.scalars().all()
    return plans


@router.get("/subscription-plans/{plan_id}", response_model=SubscriptionPlanResponse)
async def get_subscription_plan(
    plan_id: int,
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get subscription plan by ID"""
    
    result = await db.execute(
        select(SubscriptionPlanModel).where(SubscriptionPlanModel.id == plan_id)
    )
    plan = result.scalar_one_or_none()
    
    if not plan:
        raise HTTPException(status_code=404, detail="Subscription plan not found")
    
    return plan


@router.post("/subscription-plans", response_model=SubscriptionPlanResponse)
async def create_subscription_plan(
    data: SubscriptionPlanCreate,
    request: Request,
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Create a new subscription plan"""
    
    # Check if plan name already exists
    result = await db.execute(
        select(SubscriptionPlanModel).where(SubscriptionPlanModel.name == data.name)
    )
    existing = result.scalar_one_or_none()
    
    if existing:
        raise HTTPException(status_code=400, detail="Plan name already exists")
    
    plan = SubscriptionPlanModel(
        name=data.name,
        display_name=data.display_name,
        price=data.price,
        scans_per_month=data.scans_per_month,
        features=data.features,
        is_active=True
    )
    
    db.add(plan)
    await db.commit()
    await db.refresh(plan)
    
    # Log action
    await UserUseCases.log_action(
        db=db,
        user_id=current_admin.id,
        action="subscription_plan_created",
        entity_type="subscription_plan",
        entity_id=plan.id,
        details={"name": data.name, "price": data.price},
        ip_address=get_client_ip(request)
    )
    
    return plan


@router.patch("/subscription-plans/{plan_id}", response_model=SubscriptionPlanResponse)
async def update_subscription_plan(
    plan_id: int,
    data: SubscriptionPlanUpdate,
    request: Request,
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Update a subscription plan"""
    
    result = await db.execute(
        select(SubscriptionPlanModel).where(SubscriptionPlanModel.id == plan_id)
    )
    plan = result.scalar_one_or_none()
    
    if not plan:
        raise HTTPException(status_code=404, detail="Subscription plan not found")
    
    # Update fields
    if data.display_name is not None:
        plan.display_name = data.display_name
    if data.price is not None:
        plan.price = data.price
    if data.scans_per_month is not None:
        plan.scans_per_month = data.scans_per_month
    if data.features is not None:
        plan.features = data.features
    if data.is_active is not None:
        plan.is_active = data.is_active
    
    await db.commit()
    await db.refresh(plan)
    
    # Log action
    await UserUseCases.log_action(
        db=db,
        user_id=current_admin.id,
        action="subscription_plan_updated",
        entity_type="subscription_plan",
        entity_id=plan.id,
        details={"name": plan.name},
        ip_address=get_client_ip(request)
    )
    
    return plan


@router.delete("/subscription-plans/{plan_id}")
async def delete_subscription_plan(
    plan_id: int,
    request: Request,
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Delete a subscription plan (soft delete by deactivating)"""
    
    result = await db.execute(
        select(SubscriptionPlanModel).where(SubscriptionPlanModel.id == plan_id)
    )
    plan = result.scalar_one_or_none()
    
    if not plan:
        raise HTTPException(status_code=404, detail="Subscription plan not found")
    
    # Check if users are on this plan
    user_count = await db.scalar(
        select(func.count(UserModel.id)).where(UserModel.subscription_plan == plan.name)
    )
    
    if user_count > 0:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot delete plan. {user_count} users are currently on this plan."
        )
    
    # Soft delete (deactivate)
    plan.is_active = False
    await db.commit()
    
    # Log action
    await UserUseCases.log_action(
        db=db,
        user_id=current_admin.id,
        action="subscription_plan_deleted",
        entity_type="subscription_plan",
        entity_id=plan.id,
        details={"name": plan.name},
        ip_address=get_client_ip(request)
    )
    
    return {"message": "Subscription plan deactivated"}


# =====================================================================
# AI ENGINES DASHBOARD & MANAGEMENT (Complete Integration)
# =====================================================================

@router.get("/ai-engines/status")
async def get_all_ai_engines_status(
    current_admin = Depends(get_current_admin)
):
    """
    Get comprehensive status of all AI engines in the system.
    
    This is the admin dashboard for monitoring AI component health.
    """
    engines = {}
    
    # 1. BERT Fraud Classifier
    try:
        from application.use_cases.bert_fraud_classifier import get_fraud_classifier
        classifier = get_fraud_classifier()
        engines["bert_classifier"] = {
            "name": "BERT Fraud Classifier",
            "status": "ready" if classifier.is_trained else "not_trained",
            "is_real_ai": True,
            "model": classifier.config.model_name,
            "description": "Fine-tuned DistilBERT for fraud text classification",
            "requires_training": not classifier.is_trained
        }
    except ImportError:
        engines["bert_classifier"] = {
            "name": "BERT Fraud Classifier",
            "status": "unavailable",
            "is_real_ai": True,
            "error": "Dependencies not installed (transformers, torch)"
        }
    except Exception as e:
        engines["bert_classifier"] = {
            "name": "BERT Fraud Classifier",
            "status": "error",
            "is_real_ai": True,
            "error": str(e)
        }
    
    # 2. Real XAI Engine
    try:
        from application.use_cases.real_xai_engine import real_xai_engine, TORCH_AVAILABLE, SHAP_AVAILABLE
        engines["xai_engine"] = {
            "name": "Real XAI Engine",
            "status": "available",
            "is_real_ai": True,
            "capabilities": {
                "integrated_gradients": TORCH_AVAILABLE,
                "attention_weights": TORCH_AVAILABLE,
                "shap_values": SHAP_AVAILABLE
            },
            "description": "Integrated Gradients + Attention Weights + SHAP for model explanations"
        }
    except Exception as e:
        engines["xai_engine"] = {
            "name": "Real XAI Engine",
            "status": "error",
            "is_real_ai": True,
            "error": str(e)
        }
    
    # 3. Message Analysis Engine
    try:
        from application.use_cases.message_analysis_engine import message_analysis_engine, TRANSFORMERS_AVAILABLE
        engines["message_analysis"] = {
            "name": "Message/Conversation Analysis",
            "status": "available",
            "is_real_ai": True,
            "nlp_available": TRANSFORMERS_AVAILABLE,
            "description": "BERT-based message risk analysis with social engineering detection"
        }
    except Exception as e:
        engines["message_analysis"] = {
            "name": "Message/Conversation Analysis",
            "status": "error",
            "is_real_ai": True,
            "error": str(e)
        }
    
    # 4. Cross-Document Consistency Engine
    try:
        from application.use_cases.cross_document_engine import cross_document_engine, SPACY_AVAILABLE
        engines["cross_document"] = {
            "name": "Cross-Document Consistency",
            "status": "available",
            "is_real_ai": True,
            "nlp_available": SPACY_AVAILABLE,
            "description": "NLP-based entity extraction and cross-document verification"
        }
    except Exception as e:
        engines["cross_document"] = {
            "name": "Cross-Document Consistency",
            "status": "error",
            "is_real_ai": True,
            "error": str(e)
        }
    
    # 5. Price Anomaly Engine
    try:
        from application.use_cases.price_anomaly_engine import price_anomaly_engine
        engines["price_anomaly"] = {
            "name": "Price Anomaly Detection",
            "status": "available",
            "is_real_ai": True,
            "description": "Statistical analysis with Canadian rental market data (7 major cities)"
        }
    except Exception as e:
        engines["price_anomaly"] = {
            "name": "Price Anomaly Detection",
            "status": "error",
            "is_real_ai": True,
            "error": str(e)
        }
    
    # 6. Address Validation Engine
    try:
        from application.use_cases.address_validation_engine import address_validation_engine
        engines["address_validation"] = {
            "name": "Address Validation (Geocoding)",
            "status": "available",
            "is_real_ai": True,
            "description": "Nominatim/OpenStreetMap geocoding with suspicious pattern detection"
        }
    except Exception as e:
        engines["address_validation"] = {
            "name": "Address Validation",
            "status": "error",
            "is_real_ai": True,
            "error": str(e)
        }
    
    # 7. Real Image Classification Engine
    try:
        from application.use_cases.real_image_engine import real_image_engine
        # Check torch availability inline since real_image_engine uses lazy loading
        try:
            import torch
            IMG_TORCH = True
        except ImportError:
            IMG_TORCH = False
        engines["image_classification"] = {
            "name": "CNN Image Classification",
            "status": "available" if IMG_TORCH else "limited",
            "is_real_ai": True,
            "cnn_available": IMG_TORCH,
            "description": "ResNet50 pretrained CNN for property image classification"
        }
    except Exception as e:
        engines["image_classification"] = {
            "name": "CNN Image Classification",
            "status": "error",
            "is_real_ai": True,
            "error": str(e)
        }
    
    # 8. OCR Document Engine
    try:
        from application.use_cases.ocr_engine import ocr_engine
        engines["ocr_engine"] = {
            "name": "OCR Document Analysis",
            "status": "available",
            "is_real_ai": True,
            "description": "Tesseract OCR with document type classification and fraud pattern detection"
        }
    except Exception as e:
        engines["ocr_engine"] = {
            "name": "OCR Document Analysis",
            "status": "error",
            "is_real_ai": True,
            "error": str(e)
        }
    
    # 9. Auto-Learning Engine
    try:
        from application.use_cases.auto_learning_engine import auto_learning_engine
        engines["auto_learning"] = {
            "name": "Auto-Learning (Feedback Loop)",
            "status": "available",
            "is_real_ai": True,
            "description": "Continuous learning from user feedback for model improvement"
        }
    except Exception as e:
        engines["auto_learning"] = {
            "name": "Auto-Learning",
            "status": "error",
            "is_real_ai": True,
            "error": str(e)
        }
    
    # 10. Indicator Engine (Rule-based but integrated with AI)
    try:
        from application.use_cases.indicator_engine import indicator_engine
        engines["indicator_engine"] = {
            "name": "Indicator Engine",
            "status": "available",
            "is_real_ai": False,
            "note": "Rule-based - provides explainability layer for BERT predictions",
            "description": "Multi-layer indicator detection integrated with BERT scores"
        }
    except Exception as e:
        engines["indicator_engine"] = {
            "name": "Indicator Engine",
            "status": "error",
            "is_real_ai": False,
            "error": str(e)
        }
    
    # Calculate summary
    total = len(engines)
    available = sum(1 for e in engines.values() if e.get("status") in ["available", "ready"])
    real_ai_count = sum(1 for e in engines.values() if e.get("is_real_ai"))
    
    return {
        "summary": {
            "total_engines": total,
            "available": available,
            "real_ai_components": real_ai_count,
            "health_percentage": f"{(available/total)*100:.0f}%"
        },
        "engines": engines
    }


@router.post("/ai-engines/test-bert")
async def test_bert_prediction(
    text: str = Form(...),
    current_admin = Depends(get_current_admin)
):
    """
    Test BERT fraud classifier with sample text.
    
    Returns prediction with confidence and XAI explanation.
    """
    from application.use_cases.bert_fraud_classifier import get_fraud_classifier
    from application.use_cases.real_xai_engine import get_xai_explanation
    
    classifier = get_fraud_classifier()
    
    if not classifier.is_trained:
        raise HTTPException(status_code=400, detail="BERT model not trained. Please train first.")
    
    # Get prediction
    prediction = classifier.predict(text)
    
    # Get XAI explanation
    pred_label = "fraud" if prediction['fraud_probability'] > 0.5 else "safe"
    confidence = prediction['fraud_probability'] if pred_label == "fraud" else (1 - prediction['fraud_probability'])
    explanation = get_xai_explanation(text, pred_label, confidence, "combined")
    
    return {
        "prediction": prediction,
        "explanation": explanation
    }


@router.post("/ai-engines/test-message-analysis")
async def test_message_analysis(
    message: str = Form(...),
    sender: str = Form(default="unknown"),
    current_admin = Depends(get_current_admin)
):
    """Test message analysis engine with sample message."""
    from application.use_cases.message_analysis_engine import message_analysis_engine
    
    result = message_analysis_engine.analyze_message(message, sender)
    return result.to_dict()


@router.post("/ai-engines/test-cross-document")
async def test_cross_document_verification(
    request: Request,
    current_admin = Depends(get_current_admin)
):
    """
    Test cross-document consistency verification.
    
    Send JSON body:
    {
        "documents": [
            {"name": "id_card", "type": "id", "text": "OCR text from ID..."},
            {"name": "pay_stub", "type": "paystub", "text": "OCR text from pay stub..."}
        ],
        "expected_name": "John Doe",
        "expected_address": "123 Main St"
    }
    """
    from application.use_cases.cross_document_engine import cross_document_engine
    
    body = await request.json()
    documents = body.get("documents", [])
    expected_name = body.get("expected_name")
    expected_address = body.get("expected_address")
    
    if not documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    
    result = cross_document_engine.analyze_documents(
        documents, expected_name, expected_address
    )
    return result.to_dict()


@router.get("/ai-engines/training-data-stats")
async def get_training_data_statistics(
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get statistics about available training data."""
    import os
    import csv
    
    BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_path = os.path.join(BACKEND_DIR, "data", "fraud_dataset.csv")
    
    stats = {
        "fraud_dataset": {
            "exists": os.path.exists(dataset_path),
            "path": dataset_path
        }
    }
    
    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                stats["fraud_dataset"]["total_rows"] = len(rows)
                
                # Count by label
                fraud_count = sum(1 for r in rows if r.get('label', '').lower() in ['1', 'fraud', 'true'])
                safe_count = len(rows) - fraud_count
                stats["fraud_dataset"]["fraud_examples"] = fraud_count
                stats["fraud_dataset"]["safe_examples"] = safe_count
                
                # Analyze fraud types if available
                if 'fraud_type' in rows[0]:
                    fraud_types = {}
                    for r in rows:
                        ft = r.get('fraud_type', 'unknown')
                        fraud_types[ft] = fraud_types.get(ft, 0) + 1
                    stats["fraud_dataset"]["fraud_types"] = fraud_types
        except Exception as e:
            stats["fraud_dataset"]["error"] = str(e)
    
    # Get feedback-based training data
    from application.use_cases.auto_learning_engine import auto_learning_engine
    feedback_samples = await auto_learning_engine.get_retraining_dataset(db)
    
    stats["feedback_data"] = {
        "total_samples": len(feedback_samples),
        "fraud_samples": len([s for s in feedback_samples if s["label"] == 1]),
        "safe_samples": len([s for s in feedback_samples if s["label"] == 0])
    }
    
    return stats


# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================

@router.get("/analytics/overview")
async def get_analytics_overview(
    days: int = 30,
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Get comprehensive analytics overview.
    
    Returns aggregated metrics for the specified time window:
    - Analysis volume trends (daily)
    - Fraud detection rates over time
    - Risk score distributions
    - User registration trends
    - Feedback trends
    """
    from datetime import datetime, timedelta
    from sqlalchemy import cast, Date, case
    
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    # ── Analysis Volume Trend (daily counts) ──
    analysis_rows = await db.execute(
        select(
            cast(RiskAnalysisModel.created_at, Date).label("date"),
            func.count(RiskAnalysisModel.id).label("count")
        )
        .where(RiskAnalysisModel.created_at >= cutoff)
        .group_by(cast(RiskAnalysisModel.created_at, Date))
        .order_by(cast(RiskAnalysisModel.created_at, Date))
    )
    analysis_trend = [
        {"date": str(row.date), "count": row.count}
        for row in analysis_rows.all()
    ]
    
    # ── Fraud Detection Rate Trend (daily) ──
    fraud_rate_rows = await db.execute(
        select(
            cast(RiskAnalysisModel.created_at, Date).label("date"),
            func.count(RiskAnalysisModel.id).label("total"),
            func.sum(
                case(
                    (RiskAnalysisModel.risk_level.in_(["high", "very_high"]), 1),
                    else_=0
                )
            ).label("flagged")
        )
        .where(RiskAnalysisModel.created_at >= cutoff)
        .group_by(cast(RiskAnalysisModel.created_at, Date))
        .order_by(cast(RiskAnalysisModel.created_at, Date))
    )
    fraud_rate_trend = []
    for row in fraud_rate_rows.all():
        total = row.total or 0
        flagged = row.flagged or 0
        fraud_rate_trend.append({
            "date": str(row.date),
            "total": total,
            "flagged": flagged,
            "rate": round((flagged / total * 100), 1) if total > 0 else 0.0
        })
    
    # ── Risk Score Distribution (buckets) ──
    score_rows = await db.execute(
        select(RiskAnalysisModel.risk_score)
        .where(RiskAnalysisModel.created_at >= cutoff)
    )
    scores = [row[0] for row in score_rows.all() if row[0] is not None]
    
    buckets = {"0-20": 0, "20-40": 0, "40-60": 0, "60-80": 0, "80-100": 0}
    for s in scores:
        pct = s * 100 if s <= 1.0 else s
        if pct < 20:
            buckets["0-20"] += 1
        elif pct < 40:
            buckets["20-40"] += 1
        elif pct < 60:
            buckets["40-60"] += 1
        elif pct < 80:
            buckets["60-80"] += 1
        else:
            buckets["80-100"] += 1
    
    score_distribution = [
        {"range": k, "count": v} for k, v in buckets.items()
    ]
    
    # ── Risk Level Breakdown ──
    level_rows = await db.execute(
        select(
            RiskAnalysisModel.risk_level,
            func.count(RiskAnalysisModel.id).label("count")
        )
        .where(RiskAnalysisModel.created_at >= cutoff)
        .group_by(RiskAnalysisModel.risk_level)
    )
    risk_level_breakdown = [
        {"level": row.risk_level or "unknown", "count": row.count}
        for row in level_rows.all()
    ]
    
    # ── User Registration Trend (daily) ──
    user_rows = await db.execute(
        select(
            cast(UserModel.created_at, Date).label("date"),
            func.count(UserModel.id).label("count")
        )
        .where(UserModel.created_at >= cutoff)
        .group_by(cast(UserModel.created_at, Date))
        .order_by(cast(UserModel.created_at, Date))
    )
    user_trend = [
        {"date": str(row.date), "count": row.count}
        for row in user_rows.all()
    ]
    
    # ── User Role Distribution ──
    role_rows = await db.execute(
        select(
            UserModel.role,
            func.count(UserModel.id).label("count")
        )
        .group_by(UserModel.role)
    )
    role_distribution = [
        {"role": row.role or "unknown", "count": row.count}
        for row in role_rows.all()
    ]
    
    # ── Feedback Trend (daily) ──
    feedback_rows = await db.execute(
        select(
            cast(FeedbackModel.created_at, Date).label("date"),
            FeedbackModel.feedback_type,
            func.count(FeedbackModel.id).label("count")
        )
        .where(FeedbackModel.created_at >= cutoff)
        .group_by(cast(FeedbackModel.created_at, Date), FeedbackModel.feedback_type)
        .order_by(cast(FeedbackModel.created_at, Date))
    )
    feedback_trend = []
    fb_by_date = {}
    for row in feedback_rows.all():
        d = str(row.date)
        if d not in fb_by_date:
            fb_by_date[d] = {"date": d, "safe": 0, "fraud": 0, "unsure": 0}
        fb_by_date[d][row.feedback_type] = row.count
    feedback_trend = list(fb_by_date.values())
    
    # ── Top-Level Aggregates ──
    total_analyses = await db.scalar(
        select(func.count(RiskAnalysisModel.id)).where(RiskAnalysisModel.created_at >= cutoff)
    ) or 0
    total_flagged = await db.scalar(
        select(func.count(RiskAnalysisModel.id)).where(
            RiskAnalysisModel.created_at >= cutoff,
            RiskAnalysisModel.risk_level.in_(["high", "very_high"])
        )
    ) or 0
    total_users_period = await db.scalar(
        select(func.count(UserModel.id)).where(UserModel.created_at >= cutoff)
    ) or 0
    total_feedback_period = await db.scalar(
        select(func.count(FeedbackModel.id)).where(FeedbackModel.created_at >= cutoff)
    ) or 0
    avg_risk = await db.scalar(
        select(func.avg(RiskAnalysisModel.risk_score)).where(RiskAnalysisModel.created_at >= cutoff)
    )
    
    return {
        "period_days": days,
        "summary": {
            "total_analyses": total_analyses,
            "total_flagged": total_flagged,
            "fraud_rate": round((total_flagged / total_analyses * 100), 1) if total_analyses > 0 else 0.0,
            "new_users": total_users_period,
            "total_feedback": total_feedback_period,
            "avg_risk_score": round(float(avg_risk), 3) if avg_risk else 0.0
        },
        "analysis_trend": analysis_trend,
        "fraud_rate_trend": fraud_rate_trend,
        "score_distribution": score_distribution,
        "risk_level_breakdown": risk_level_breakdown,
        "user_trend": user_trend,
        "role_distribution": role_distribution,
        "feedback_trend": feedback_trend
    }


@router.get("/analytics/model-accuracy")
async def get_model_accuracy_stats(
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Get model accuracy metrics based on user feedback.
    
    Compares model predictions against confirmed user feedback
    to calculate precision, recall, and F1 scores.
    """
    from sqlalchemy.orm import aliased
    
    # Join feedback with their original analyses
    result = await db.execute(
        select(FeedbackModel, RiskAnalysisModel)
        .join(RiskAnalysisModel, FeedbackModel.analysis_id == RiskAnalysisModel.id)
        .where(FeedbackModel.feedback_type.in_(["safe", "fraud"]))
    )
    
    pairs = result.all()
    
    if len(pairs) == 0:
        return {
            "status": "insufficient_data",
            "message": "No confirmed feedback data to calculate accuracy",
            "sample_count": 0,
            "metrics": None
        }
    
    # Calculate confusion matrix
    tp = fp = tn = fn = 0
    for feedback, analysis in pairs:
        model_flagged = analysis.risk_level in ["high", "very_high"]
        user_confirmed_fraud = feedback.feedback_type == "fraud"
        
        if model_flagged and user_confirmed_fraud:
            tp += 1
        elif model_flagged and not user_confirmed_fraud:
            fp += 1
        elif not model_flagged and user_confirmed_fraud:
            fn += 1
        else:
            tn += 1
    
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    return {
        "status": "success",
        "sample_count": total,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        },
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn
        },
        "interpretation": {
            "accuracy_pct": f"{accuracy * 100:.1f}%",
            "precision_pct": f"{precision * 100:.1f}%",
            "recall_pct": f"{recall * 100:.1f}%",
            "f1_pct": f"{f1 * 100:.1f}%"
        }
    }


@router.get("/analytics/top-indicators")
async def get_top_fraud_indicators(
    limit: int = 15,
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Get the most common fraud indicators across all analyses.
    
    Aggregates risk_indicators JSON from flagged analyses to show
    which signals fire most frequently.
    """
    result = await db.execute(
        select(RiskAnalysisModel.risk_indicators)
        .where(RiskAnalysisModel.risk_level.in_(["high", "very_high"]))
        .where(RiskAnalysisModel.risk_indicators.isnot(None))
    )
    
    indicator_counts = {}
    for row in result.all():
        indicators = row[0]
        if isinstance(indicators, list):
            for ind in indicators:
                name = ind.get("name") or ind.get("indicator") or ind.get("code") or str(ind)
                indicator_counts[name] = indicator_counts.get(name, 0) + 1
        elif isinstance(indicators, dict):
            for key, val in indicators.items():
                indicator_counts[key] = indicator_counts.get(key, 0) + 1
    
    sorted_indicators = sorted(indicator_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    return {
        "total_flagged_analyses": await db.scalar(
            select(func.count(RiskAnalysisModel.id)).where(
                RiskAnalysisModel.risk_level.in_(["high", "very_high"])
            )
        ) or 0,
        "top_indicators": [
            {"indicator": name, "count": count}
            for name, count in sorted_indicators
        ]
    }


# ============================================================================
# MONITORING ENDPOINTS
# ============================================================================

@router.get("/monitoring/system-health")
async def get_system_health(
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Get comprehensive system health metrics.
    
    Returns:
    - Server uptime and resource usage
    - Database statistics
    - API performance metrics
    - Storage usage
    """
    import time
    import psutil
    import platform
    from datetime import datetime
    
    # ── Server Info ──
    process = psutil.Process()
    boot_time = psutil.boot_time()
    uptime_seconds = time.time() - boot_time
    
    server_info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_total_mb": round(psutil.virtual_memory().total / (1024 * 1024), 1),
        "memory_used_mb": round(psutil.virtual_memory().used / (1024 * 1024), 1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_total_gb": round(psutil.disk_usage('/').total / (1024**3), 1) if platform.system() != 'Windows' else round(psutil.disk_usage('C:\\').total / (1024**3), 1),
        "disk_used_gb": round(psutil.disk_usage('/').used / (1024**3), 1) if platform.system() != 'Windows' else round(psutil.disk_usage('C:\\').used / (1024**3), 1),
        "disk_percent": psutil.disk_usage('/').percent if platform.system() != 'Windows' else psutil.disk_usage('C:\\').percent,
        "uptime_hours": round(uptime_seconds / 3600, 1),
        "process_memory_mb": round(process.memory_info().rss / (1024 * 1024), 1)
    }
    
    # ── Database Stats ──
    total_users = await db.scalar(select(func.count(UserModel.id))) or 0
    total_analyses = await db.scalar(select(func.count(RiskAnalysisModel.id))) or 0
    total_feedback = await db.scalar(select(func.count(FeedbackModel.id))) or 0
    total_audit_logs = await db.scalar(select(func.count(AuditLogModel.id))) or 0
    total_datasets = await db.scalar(select(func.count(DatasetModel.id))) or 0
    total_models = await db.scalar(select(func.count(MLModelModel.id))) or 0
    
    # Database file size
    from config import get_settings
    settings = get_settings()
    db_path = settings.DATABASE_URL.replace("sqlite+aiosqlite:///", "")
    db_size_mb = 0
    if os.path.exists(db_path):
        db_size_mb = round(os.path.getsize(db_path) / (1024 * 1024), 2)
    
    database_stats = {
        "engine": "SQLite (async)",
        "file_size_mb": db_size_mb,
        "tables": {
            "users": total_users,
            "risk_analyses": total_analyses,
            "feedback": total_feedback,
            "audit_logs": total_audit_logs,
            "datasets": total_datasets,
            "ml_models": total_models
        },
        "total_records": total_users + total_analyses + total_feedback + total_audit_logs + total_datasets + total_models
    }
    
    # ── Storage Usage ──
    BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(BACKEND_DIR, "models")
    data_dir = os.path.join(BACKEND_DIR, "data")
    uploads_dir = os.path.join(data_dir, "uploads")
    
    def dir_size_mb(path):
        total = 0
        if os.path.exists(path):
            for dirpath, dirnames, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    try:
                        total += os.path.getsize(fp)
                    except OSError:
                        pass
        return round(total / (1024 * 1024), 2)
    
    storage = {
        "models_mb": dir_size_mb(models_dir),
        "data_mb": dir_size_mb(data_dir),
        "uploads_mb": dir_size_mb(uploads_dir),
        "database_mb": db_size_mb
    }
    storage["total_mb"] = round(
        storage["models_mb"] + storage["data_mb"] + storage["uploads_mb"] + storage["database_mb"], 2
    )
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "server": server_info,
        "database": database_stats,
        "storage": storage
    }


@router.get("/monitoring/ai-engines-health")
async def get_ai_engines_health(
    current_admin = Depends(get_current_admin)
):
    """
    Get real-time health status of all AI engines.
    
    Tests each engine's actual availability by attempting imports
    and basic operations. Measures response latency for each.
    """
    import time
    from datetime import datetime
    
    engines = []
    
    engine_checks = [
        ("BERT Fraud Classifier", "application.use_cases.bert_fraud_classifier", "get_fraud_classifier", True),
        ("Real XAI Engine", "application.use_cases.real_xai_engine", "real_xai_engine", True),
        ("Message Analysis", "application.use_cases.message_analysis_engine", "message_analysis_engine", True),
        ("Cross-Document Engine", "application.use_cases.cross_document_engine", "cross_document_engine", True),
        ("Price Anomaly Detection", "application.use_cases.price_anomaly_engine", "price_anomaly_engine", True),
        ("Address Validation", "application.use_cases.address_validation_engine", "address_validation_engine", True),
        ("CNN Image Classification", "application.use_cases.real_image_engine", "real_image_engine", True),
        ("OCR Document Analysis", "application.use_cases.ocr_engine", "ocr_engine", True),
        ("Auto-Learning Engine", "application.use_cases.auto_learning_engine", "auto_learning_engine", True),
        ("Indicator Engine", "application.use_cases.indicator_engine", "indicator_engine", False),
    ]
    
    for name, module_path, attr_name, is_real_ai in engine_checks:
        start = time.perf_counter()
        try:
            import importlib
            mod = importlib.import_module(module_path)
            obj = getattr(mod, attr_name)
            
            # Special check for BERT
            if attr_name == "get_fraud_classifier":
                classifier = obj()
                status = "ready" if classifier.is_trained else "loaded_not_trained"
            else:
                status = "available"
            
            latency_ms = round((time.perf_counter() - start) * 1000, 1)
            engines.append({
                "name": name,
                "status": status,
                "is_real_ai": is_real_ai,
                "latency_ms": latency_ms,
                "error": None
            })
        except Exception as e:
            latency_ms = round((time.perf_counter() - start) * 1000, 1)
            engines.append({
                "name": name,
                "status": "error",
                "is_real_ai": is_real_ai,
                "latency_ms": latency_ms,
                "error": str(e)
            })
    
    available = sum(1 for e in engines if e["status"] in ["available", "ready", "loaded_not_trained"])
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "total_engines": len(engines),
        "available": available,
        "health_percentage": round((available / len(engines)) * 100, 1) if engines else 0,
        "engines": engines
    }


@router.get("/monitoring/recent-errors")
async def get_recent_errors(
    limit: int = 50,
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Get recent error-level audit log entries.
    
    Filters audit logs for error actions and failed operations
    to surface system issues.
    """
    result = await db.execute(
        select(AuditLogModel)
        .where(
            AuditLogModel.action.in_([
                "login_failed", "auth_error", "analysis_failed",
                "auto_learning_failed", "system_error", "api_error",
                "model_training_failed", "preprocessing_failed"
            ])
        )
        .order_by(AuditLogModel.created_at.desc())
        .limit(limit)
    )
    error_logs = result.scalars().all()
    
    # Also get recent 500-level actions (any action containing "error" or "failed")
    result2 = await db.execute(
        select(AuditLogModel)
        .where(AuditLogModel.action.like("%error%"))
        .order_by(AuditLogModel.created_at.desc())
        .limit(limit)
    )
    error_logs_2 = result2.scalars().all()
    
    # Merge and deduplicate
    seen_ids = set()
    all_errors = []
    for log in list(error_logs) + list(error_logs_2):
        if log.id not in seen_ids:
            seen_ids.add(log.id)
            all_errors.append({
                "id": log.id,
                "action": log.action,
                "entity_type": log.entity_type,
                "details": log.details,
                "ip_address": log.ip_address,
                "created_at": log.created_at.isoformat() if log.created_at else None,
                "user_id": log.user_id
            })
    
    # Sort by date descending
    all_errors.sort(key=lambda x: x["created_at"] or "", reverse=True)
    
    return {
        "total_errors": len(all_errors),
        "errors": all_errors[:limit]
    }


@router.get("/monitoring/activity-feed")
async def get_activity_feed(
    limit: int = 30,
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Get real-time activity feed showing recent system events.
    
    Returns the most recent audit log entries across all action types,
    enriched with user email where available.
    """
    result = await db.execute(
        select(AuditLogModel, UserModel.email)
        .outerjoin(UserModel, AuditLogModel.user_id == UserModel.id)
        .order_by(AuditLogModel.created_at.desc())
        .limit(limit)
    )
    
    activities = []
    for log, user_email in result.all():
        activities.append({
            "id": log.id,
            "action": log.action,
            "entity_type": log.entity_type,
            "user_email": user_email,
            "details": log.details,
            "ip_address": log.ip_address,
            "created_at": log.created_at.isoformat() if log.created_at else None
        })
    
    return {
        "total": len(activities),
        "activities": activities
    }


@router.get("/monitoring/dependency-versions")
async def get_dependency_versions(
    current_admin = Depends(get_current_admin)
):
    """
    Get installed versions of key Python dependencies.
    
    Lists framework, ML, and NLP library versions for
    environment auditing and compatibility checks.
    """
    import platform
    
    deps = {}
    packages = [
        "fastapi", "uvicorn", "sqlalchemy", "pydantic",
        "transformers", "torch", "sklearn", "scipy",
        "nltk", "spacy", "pandas", "numpy",
        "Pillow", "pytesseract", "aiofiles", "python-jose",
        "passlib", "psutil"
    ]
    
    for pkg in packages:
        try:
            if pkg == "sklearn":
                import sklearn
                deps[pkg] = sklearn.__version__
            elif pkg == "Pillow":
                from PIL import Image
                import PIL
                deps[pkg] = PIL.__version__
            else:
                mod = __import__(pkg.replace("-", "_"))
                deps[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            deps[pkg] = None
    
    return {
        "python_version": platform.python_version(),
        "system": platform.system(),
        "architecture": platform.machine(),
        "dependencies": deps
    }


# ============================================================================
# DATA PREPROCESSING PIPELINE ENDPOINTS
# ============================================================================

@router.get("/preprocessing/status")
async def get_preprocessing_pipeline_status(
    current_admin = Depends(get_current_admin)
):
    """
    Get status of the data preprocessing pipeline.
    
    Returns information about:
    - Pipeline components and their availability
    - Supported feature engineering methods
    - Configuration options
    """
    from application.use_cases.data_preprocessing_pipeline import (
        DataPreprocessingPipeline,
        ColumnType,
        ImputationStrategy,
        OutlierMethod,
        ScalingMethod,
        TextVectorizationMethod,
        SKLEARN_AVAILABLE,
        TORCH_AVAILABLE,
        NLTK_AVAILABLE
    )
    
    return {
        "pipeline_status": "available",
        "components": {
            "type_inference_engine": {
                "status": "available",
                "description": "ML-powered automatic column type detection",
                "supported_types": [t.value for t in ColumnType]
            },
            "missing_value_handler": {
                "status": "available" if SKLEARN_AVAILABLE else "limited",
                "description": "Advanced missing value imputation (KNN, MICE, iterative)",
                "strategies": [s.value for s in ImputationStrategy],
                "knn_available": SKLEARN_AVAILABLE
            },
            "outlier_detection_engine": {
                "status": "available" if SKLEARN_AVAILABLE else "limited",
                "description": "Multi-method outlier detection (Isolation Forest, LOF, statistical)",
                "methods": [m.value for m in OutlierMethod],
                "ml_methods_available": SKLEARN_AVAILABLE
            },
            "text_feature_engine": {
                "status": "available",
                "description": "NLP text feature extraction (TF-IDF, embeddings, fraud patterns)",
                "vectorization_methods": [v.value for v in TextVectorizationMethod],
                "embeddings_available": TORCH_AVAILABLE,
                "fraud_specific_features": True
            },
            "numerical_feature_engine": {
                "status": "available" if SKLEARN_AVAILABLE else "limited",
                "description": "Numerical feature engineering (scaling, polynomial, binning)",
                "scaling_methods": [s.value for s in ScalingMethod],
                "polynomial_features": SKLEARN_AVAILABLE,
                "interaction_features": SKLEARN_AVAILABLE
            },
            "categorical_feature_engine": {
                "status": "available",
                "description": "Categorical encoding (frequency, target, one-hot)",
                "encoding_methods": ["frequency", "target", "onehot", "ordinal"]
            },
            "geospatial_feature_engine": {
                "status": "available" if SKLEARN_AVAILABLE else "limited",
                "description": "Geospatial features (clustering, distance, density)",
                "clustering_available": SKLEARN_AVAILABLE
            },
            "feature_selection_engine": {
                "status": "available" if SKLEARN_AVAILABLE else "limited",
                "description": "Intelligent feature selection (mutual info, correlation, variance)",
                "methods": ["variance", "correlation", "mutual_info", "combined"]
            }
        },
        "dependencies": {
            "sklearn": SKLEARN_AVAILABLE,
            "torch": TORCH_AVAILABLE,
            "nltk": NLTK_AVAILABLE
        }
    }


@router.post("/preprocessing/analyze-dataset/{dataset_id}")
async def analyze_dataset_with_pipeline(
    dataset_id: int,
    request: Request,
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze a dataset using the preprocessing pipeline.
    
    This performs:
    - Automatic type inference for all columns
    - Data quality assessment
    - Missing value analysis
    - Outlier detection
    - Feature engineering recommendations
    
    Request body (optional):
    {
        "target_column": "fraud_label",
        "text_columns": ["description", "title"],
        "exclude_columns": ["id", "timestamp"]
    }
    """
    import pandas as pd
    from application.use_cases.data_preprocessing_pipeline import (
        DataPreprocessingPipeline,
        create_rental_fraud_pipeline
    )
    
    # Get dataset
    result = await db.execute(select(DatasetModel).where(DatasetModel.id == dataset_id))
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Load dataset
    BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(BACKEND_DIR, dataset.file_path)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset file not found")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read dataset: {e}")
    
    # Parse request body
    try:
        body = await request.json()
    except:
        body = {}
    
    target_column = body.get("target_column")
    text_columns = body.get("text_columns")
    exclude_columns = body.get("exclude_columns", [])
    
    # Create and fit pipeline
    pipeline = create_rental_fraud_pipeline(use_embeddings=False)
    
    # Generate quality report
    quality_report = pipeline.get_quality_report(df)
    
    # Perform type inference
    type_map = pipeline.type_engine.infer_types(df)
    
    # Generate column analysis
    column_analysis = {}
    for col, col_type in type_map.items():
        series = df[col]
        analysis = {
            "detected_type": col_type.value,
            "original_dtype": str(series.dtype),
            "null_count": int(series.isnull().sum()),
            "null_percentage": round(series.isnull().mean() * 100, 2),
            "unique_count": int(series.nunique()),
            "unique_percentage": round(series.nunique() / len(series) * 100, 2),
            "sample_values": series.dropna().head(5).tolist()
        }
        
        # Add statistics for numeric columns
        if pd.api.types.is_numeric_dtype(series):
            analysis["statistics"] = {
                "mean": round(float(series.mean()), 4) if pd.notna(series.mean()) else None,
                "std": round(float(series.std()), 4) if pd.notna(series.std()) else None,
                "min": float(series.min()) if pd.notna(series.min()) else None,
                "max": float(series.max()) if pd.notna(series.max()) else None,
                "median": float(series.median()) if pd.notna(series.median()) else None,
                "skewness": round(float(series.skew()), 4) if len(series.dropna()) > 2 else None
            }
        
        column_analysis[col] = analysis
    
    # Log action
    await UserUseCases.log_action(
        db=db,
        user_id=current_admin.id,
        action="dataset_analyzed_with_pipeline",
        entity_type="dataset",
        entity_id=str(dataset_id),
        details={
            "dataset_name": dataset.name,
            "rows": len(df),
            "columns": len(df.columns),
            "quality_score": quality_report.data_quality_score
        },
        ip_address=get_client_ip(request)
    )
    
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset.name,
        "shape": {
            "rows": len(df),
            "columns": len(df.columns)
        },
        "quality_score": round(quality_report.data_quality_score * 100, 2),
        "data_quality": {
            "duplicate_rows": quality_report.duplicate_rows,
            "total_missing_values": sum(quality_report.missing_value_summary.values()),
            "missing_by_column": quality_report.missing_value_summary
        },
        "column_analysis": column_analysis,
        "type_summary": {
            col_type.value: len([c for c, t in type_map.items() if t == col_type])
            for col_type in set(type_map.values())
        },
        "recommendations": quality_report.recommendations,
        "warnings": quality_report.warnings
    }


@router.post("/preprocessing/process-dataset/{dataset_id}")
async def process_dataset_with_pipeline(
    dataset_id: int,
    request: Request,
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Process a dataset through the full preprocessing and feature engineering pipeline.
    
    This performs:
    - Missing value imputation (KNN, MICE)
    - Outlier detection and handling
    - Text feature engineering (TF-IDF, fraud patterns)
    - Numerical feature engineering (scaling, polynomial)
    - Categorical encoding (frequency, target)
    - Geospatial features (clustering, distance)
    - Feature selection (mutual info, correlation)
    
    Request body:
    {
        "target_column": "fraud_label",
        "text_columns": ["description", "title"],
        "exclude_columns": ["id"],
        "save_processed": true,
        "options": {
            "use_embeddings": false,
            "include_geo": true,
            "scaling_method": "robust",
            "outlier_handling": "clip"
        }
    }
    
    Returns processed dataset statistics and optionally saves the processed data.
    """
    import pandas as pd
    from application.use_cases.data_preprocessing_pipeline import (
        DataPreprocessingPipeline,
        create_rental_fraud_pipeline,
        ScalingMethod,
        TextVectorizationMethod
    )
    
    # Get dataset
    result = await db.execute(select(DatasetModel).where(DatasetModel.id == dataset_id))
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Load dataset
    BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(BACKEND_DIR, dataset.file_path)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset file not found")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read dataset: {e}")
    
    # Parse request body
    try:
        body = await request.json()
    except:
        body = {}
    
    target_column = body.get("target_column")
    text_columns = body.get("text_columns")
    exclude_columns = body.get("exclude_columns", [])
    save_processed = body.get("save_processed", False)
    options = body.get("options", {})
    
    # Create pipeline with options
    use_embeddings = options.get("use_embeddings", False)
    include_geo = options.get("include_geo", True)
    
    pipeline = create_rental_fraud_pipeline(
        use_embeddings=use_embeddings,
        include_geo=include_geo
    )
    
    # Apply custom options
    scaling_method = options.get("scaling_method")
    if scaling_method:
        try:
            pipeline.numerical_engine.scaling_method = ScalingMethod(scaling_method)
        except:
            pass
    
    outlier_handling = options.get("outlier_handling", "clip")
    pipeline.outlier_handling = outlier_handling
    
    # Process dataset
    try:
        df_processed = pipeline.fit_transform(
            df,
            target_column=target_column,
            text_columns=text_columns,
            exclude_columns=exclude_columns
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    
    # Feature selection report
    feature_report = None
    if pipeline.selection_engine:
        feature_report = pipeline.selection_engine.get_feature_report()
    
    # Save processed dataset if requested
    processed_path = None
    if save_processed:
        processed_filename = f"processed_{dataset.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        processed_path = os.path.join(BACKEND_DIR, "data", "processed", processed_filename)
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df_processed.to_csv(processed_path, index=False)
    
    # Log action
    await UserUseCases.log_action(
        db=db,
        user_id=current_admin.id,
        action="dataset_processed_with_pipeline",
        entity_type="dataset",
        entity_id=str(dataset_id),
        details={
            "dataset_name": dataset.name,
            "original_shape": list(df.shape),
            "processed_shape": list(df_processed.shape),
            "features_created": len(df_processed.columns) - len(df.columns),
            "saved_to": processed_path
        },
        ip_address=get_client_ip(request)
    )
    
    return {
        "success": True,
        "dataset_id": dataset_id,
        "dataset_name": dataset.name,
        "original_shape": {
            "rows": len(df),
            "columns": len(df.columns)
        },
        "processed_shape": {
            "rows": len(df_processed),
            "columns": len(df_processed.columns)
        },
        "features_created": len(df_processed.columns) - len(df.columns),
        "column_assignments": pipeline.column_assignments,
        "type_inference": {k: v.value for k, v in pipeline.type_map.items()},
        "feature_selection": feature_report,
        "transformation_log": pipeline.transformation_log[-20:],  # Last 20 entries
        "processed_file": processed_path,
        "sample_data": df_processed.head(5).to_dict(orient="records")
    }


@router.post("/preprocessing/extract-features")
async def extract_features_from_text(
    request: Request,
    current_admin = Depends(get_current_admin)
):
    """
    Extract features from raw text using the text feature engine.
    
    This is useful for testing the text feature extraction on individual listings.
    
    Request body:
    {
        "text": "Beautiful 2BR apartment available immediately...",
        "include_embeddings": false,
        "include_fraud_features": true
    }
    """
    import pandas as pd
    from application.use_cases.data_preprocessing_pipeline import TextFeatureEngine
    
    body = await request.json()
    text = body.get("text", "")
    include_embeddings = body.get("include_embeddings", False)
    include_fraud_features = body.get("include_fraud_features", True)
    
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    # Create text engine
    from application.use_cases.data_preprocessing_pipeline import TextVectorizationMethod
    engine = TextFeatureEngine(
        vectorization_method=TextVectorizationMethod.TFIDF,
        use_fraud_features=include_fraud_features
    )
    
    # Create single-row DataFrame
    text_series = pd.Series([text])
    
    # Fit and transform
    features_df = engine.fit_transform(text_series, prefix="text")
    
    # Convert to dict
    features = features_df.iloc[0].to_dict()
    
    # Clean NaN values
    features = {k: (float(v) if pd.notna(v) else None) for k, v in features.items()}
    
    # Categorize features
    categorized = {
        "statistical_features": {k: v for k, v in features.items() if any(x in k for x in ['char_count', 'word_count', 'ratio', 'length', 'sentence'])},
        "fraud_features": {k: v for k, v in features.items() if any(x in k for x in ['urgency', 'payment', 'scam', 'has_', 'fraud'])},
        "tfidf_features": {k: v for k, v in features.items() if 'tfidf' in k},
        "embedding_features": {k: v for k, v in features.items() if 'embed' in k}
    }
    
    return {
        "text_length": len(text),
        "word_count": len(text.split()),
        "total_features_extracted": len(features),
        "features_by_category": {
            "statistical": len(categorized["statistical_features"]),
            "fraud_specific": len(categorized["fraud_features"]),
            "tfidf": len(categorized["tfidf_features"]),
            "embeddings": len(categorized["embedding_features"])
        },
        "all_features": features,
        "categorized_features": categorized
    }


@router.post("/preprocessing/detect-outliers")
async def detect_outliers_in_data(
    request: Request,
    current_admin = Depends(get_current_admin)
):
    """
    Detect outliers in numerical data using multiple methods.
    
    Request body:
    {
        "data": [1200, 1500, 1300, 150, 1400, 1600, 50000],
        "column_name": "price",
        "methods": ["isolation_forest", "iqr", "zscore"],
        "contamination": 0.05
    }
    """
    import numpy as np
    from application.use_cases.data_preprocessing_pipeline import (
        OutlierDetectionEngine,
        OutlierMethod
    )
    import pandas as pd
    
    body = await request.json()
    data = body.get("data", [])
    column_name = body.get("column_name", "value")
    method_names = body.get("methods", ["isolation_forest", "iqr"])
    contamination = body.get("contamination", 0.05)
    
    if not data:
        raise HTTPException(status_code=400, detail="No data provided")
    
    # Parse methods
    methods = []
    for m in method_names:
        try:
            methods.append(OutlierMethod(m))
        except:
            pass
    
    if not methods:
        methods = [OutlierMethod.ISOLATION_FOREST, OutlierMethod.IQR]
    
    # Create DataFrame
    df = pd.DataFrame({column_name: data})
    
    # Create outlier engine
    engine = OutlierDetectionEngine(
        methods=methods,
        contamination=contamination
    )
    
    # Fit and detect
    engine.fit(df, [column_name])
    outlier_masks = engine.detect(df)
    
    # Get outlier indices and values
    mask = outlier_masks.get(column_name, np.zeros(len(data), dtype=bool))
    outlier_indices = np.where(mask)[0].tolist()
    outlier_values = [data[i] for i in outlier_indices]
    
    # Calculate statistics
    data_array = np.array(data)
    
    return {
        "total_values": len(data),
        "outliers_detected": int(mask.sum()),
        "outlier_percentage": round(mask.mean() * 100, 2),
        "outlier_indices": outlier_indices,
        "outlier_values": outlier_values,
        "methods_used": [m.value for m in methods],
        "statistics": {
            "mean": float(np.mean(data_array)),
            "std": float(np.std(data_array)),
            "median": float(np.median(data_array)),
            "min": float(np.min(data_array)),
            "max": float(np.max(data_array)),
            "q1": float(np.percentile(data_array, 25)),
            "q3": float(np.percentile(data_array, 75)),
            "iqr": float(np.percentile(data_array, 75) - np.percentile(data_array, 25))
        },
        "thresholds": engine.thresholds.get(column_name, {})
    }


@router.post("/preprocessing/infer-types")
async def infer_column_types(
    request: Request,
    current_admin = Depends(get_current_admin)
):
    """
    Infer column types from sample data.
    
    Request body:
    {
        "columns": {
            "price": [1200, 1500, 1300, 1400],
            "title": ["Apt 1", "Apt 2", "Apt 3", "Apt 4"],
            "lat": [51.05, 51.06, 51.04, 51.07],
            "is_furnished": [true, false, true, false]
        }
    }
    """
    import pandas as pd
    from application.use_cases.data_preprocessing_pipeline import (
        TypeInferenceEngine
    )
    
    body = await request.json()
    columns = body.get("columns", {})
    
    if not columns:
        raise HTTPException(status_code=400, detail="No columns provided")
    
    # Create DataFrame
    df = pd.DataFrame(columns)
    
    # Infer types
    engine = TypeInferenceEngine()
    type_map = engine.infer_types(df)
    
    # Prepare detailed results
    results = {}
    for col, col_type in type_map.items():
        series = df[col]
        results[col] = {
            "inferred_type": col_type.value,
            "original_dtype": str(series.dtype),
            "sample_size": len(series),
            "unique_count": int(series.nunique()),
            "null_count": int(series.isnull().sum()),
            "sample_values": series.head(3).tolist(),
            "recommended_processing": _get_processing_recommendation(col_type)
        }
    
    return {
        "columns_analyzed": len(columns),
        "type_inference_results": results,
        "type_summary": {
            t.value: len([c for c, ct in type_map.items() if ct == t])
            for t in set(type_map.values())
        }
    }


def _get_processing_recommendation(col_type) -> str:
    """Get processing recommendation for a column type."""
    from application.use_cases.data_preprocessing_pipeline import ColumnType
    
    recommendations = {
        ColumnType.NUMERIC_CONTINUOUS: "Apply robust scaling, consider polynomial features",
        ColumnType.NUMERIC_DISCRETE: "Consider binning or one-hot encoding if cardinality is low",
        ColumnType.CATEGORICAL: "Apply frequency or target encoding",
        ColumnType.BINARY: "Keep as 0/1, no transformation needed",
        ColumnType.TEXT_SHORT: "Apply TF-IDF or embeddings",
        ColumnType.TEXT_LONG: "Apply TF-IDF with dimensionality reduction, extract fraud features",
        ColumnType.DATETIME: "Extract temporal features (day, month, hour, is_weekend)",
        ColumnType.GEOSPATIAL_LAT: "Combine with longitude for clustering and distance features",
        ColumnType.GEOSPATIAL_LON: "Combine with latitude for clustering and distance features",
        ColumnType.CURRENCY: "Apply log transformation, detect price anomalies",
        ColumnType.ID: "Exclude from modeling, use only for joins",
        ColumnType.URL: "Extract domain, count parameters, detect suspicious patterns",
        ColumnType.EMAIL: "Extract domain, detect free email providers",
        ColumnType.PHONE: "Validate format, extract area code",
        ColumnType.UNKNOWN: "Requires manual inspection"
    }
    
    return recommendations.get(col_type, "No recommendation available")


# =========================================================================
# ADMIN SETTINGS
# =========================================================================

@router.get("/settings")
async def get_admin_settings(
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all admin-configurable settings.
    
    Returns three sections:
      1. Risk Tuning — the 6 coefficients that control fraud risk level determination
      2. Session & Auth — token expiry, environment
      3. System Info — database, platform, Python version, uptime
    """
    import platform
    import sys
    import psutil
    from config import get_settings
    
    settings = get_settings()
    
    # ── 1. Risk Tuning ──
    risk_tuning = {
        "risk_base_thresholds": settings.RISK_BASE_THRESHOLDS,
        "risk_severity_shift_coefficient": settings.RISK_SEVERITY_SHIFT_COEFFICIENT,
        "risk_confidence_shift_coefficient": settings.RISK_CONFIDENCE_SHIFT_COEFFICIENT,
        "risk_max_threshold_shift": settings.RISK_MAX_THRESHOLD_SHIFT,
        "risk_severity_baseline": settings.RISK_SEVERITY_BASELINE,
        "risk_confidence_baseline": settings.RISK_CONFIDENCE_BASELINE,
        "descriptions": {
            "risk_base_thresholds": "Boundaries between Very Low / Low / Medium / High / Very High risk levels. Lower values = stricter detection.",
            "risk_severity_shift_coefficient": "How much indicator severity shifts thresholds. Higher = severity has more impact (0.10–0.25 recommended).",
            "risk_confidence_shift_coefficient": "How much model confidence shifts thresholds. Higher = confidence has more impact (0.05–0.20 recommended).",
            "risk_max_threshold_shift": "Maximum ± shift cap for thresholds. Prevents extreme shifts (0.15–0.25 recommended).",
            "risk_severity_baseline": "Neutral point for severity. Below = more lenient, Above = stricter (0.3–0.5 recommended).",
            "risk_confidence_baseline": "Neutral point for confidence. Standard is 0.70 (70% confidence as neutral)."
        }
    }
    
    # ── 2. Session & Auth ──
    session_auth = {
        "access_token_expire_minutes": settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        "environment": settings.ENVIRONMENT,
        "algorithm": settings.ALGORITHM,
    }
    
    # ── 3. System Info ──
    BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = settings.DATABASE_URL.replace("sqlite+aiosqlite:///", "")
    db_size_mb = 0
    if os.path.exists(db_path):
        db_size_mb = round(os.path.getsize(db_path) / (1024 * 1024), 2)
    
    uptime_seconds = psutil.time.time() - psutil.boot_time()
    
    system_info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "os": platform.system(),
        "architecture": platform.machine(),
        "database_engine": "SQLite (async via aiosqlite)",
        "database_path": db_path,
        "database_size_mb": db_size_mb,
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "uptime_hours": round(uptime_seconds / 3600, 1),
        "backend_dir": BACKEND_DIR,
    }
    
    return {
        "risk_tuning": risk_tuning,
        "session_auth": session_auth,
        "system_info": system_info
    }


@router.put("/settings/risk-tuning")
async def update_risk_tuning(
    request: Request,
    current_admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Update risk tuning coefficients.
    
    These are written to the .env file so they persist across restarts.
    The in-memory settings cache is also invalidated.
    """
    from config import get_settings, Settings
    
    body = await request.json()
    
    # Validate fields
    allowed_fields = {
        "risk_base_thresholds",
        "risk_severity_shift_coefficient",
        "risk_confidence_shift_coefficient",
        "risk_max_threshold_shift",
        "risk_severity_baseline",
        "risk_confidence_baseline"
    }
    
    env_key_map = {
        "risk_base_thresholds": "RISK_BASE_THRESHOLDS",
        "risk_severity_shift_coefficient": "RISK_SEVERITY_SHIFT_COEFFICIENT",
        "risk_confidence_shift_coefficient": "RISK_CONFIDENCE_SHIFT_COEFFICIENT",
        "risk_max_threshold_shift": "RISK_MAX_THRESHOLD_SHIFT",
        "risk_severity_baseline": "RISK_SEVERITY_BASELINE",
        "risk_confidence_baseline": "RISK_CONFIDENCE_BASELINE"
    }
    
    updates = {}
    for key, value in body.items():
        if key not in allowed_fields:
            raise HTTPException(status_code=400, detail=f"Unknown setting: {key}")
        
        # Validate types
        if key == "risk_base_thresholds":
            if not isinstance(value, list) or len(value) != 4:
                raise HTTPException(status_code=400, detail="risk_base_thresholds must be a list of 4 floats")
            for i, v in enumerate(value):
                if not isinstance(v, (int, float)) or v < 0 or v > 1:
                    raise HTTPException(status_code=400, detail=f"Threshold {i} must be between 0 and 1")
            # Ensure ascending order
            if not all(value[i] < value[i+1] for i in range(len(value)-1)):
                raise HTTPException(status_code=400, detail="Thresholds must be in ascending order")
        else:
            if not isinstance(value, (int, float)):
                raise HTTPException(status_code=400, detail=f"{key} must be a number")
            if value < 0 or value > 1:
                raise HTTPException(status_code=400, detail=f"{key} must be between 0 and 1")
        
        updates[key] = value
    
    if not updates:
        raise HTTPException(status_code=400, detail="No settings provided")
    
    # Write to .env file
    BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(BACKEND_DIR, ".env")
    
    # Read existing .env content
    env_lines = {}
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    env_lines[k.strip()] = v.strip()
    
    # Update values
    for key, value in updates.items():
        env_key = env_key_map[key]
        if key == "risk_base_thresholds":
            env_lines[env_key] = str(value)
        else:
            env_lines[env_key] = str(value)
        
        # Also update in-memory via environment variable
        os.environ[env_key] = str(value)
    
    # Write back .env
    with open(env_path, 'w') as f:
        for k, v in env_lines.items():
            f.write(f"{k}={v}\n")
    
    # Clear settings cache so next get_settings() picks up new values
    from config import get_settings
    get_settings.cache_clear()
    
    # Audit log
    ip_address = request.client.host if request.client else None
    audit_log = AuditLogModel(
        user_id=current_admin.id,
        action="update_risk_settings",
        entity_type="settings",
        entity_id=None,
        details={"updated_fields": list(updates.keys()), "new_values": updates},
        ip_address=ip_address
    )
    db.add(audit_log)
    await db.commit()
    
    # Return updated settings
    new_settings = get_settings()
    return {
        "status": "success",
        "message": f"Updated {len(updates)} risk tuning setting(s)",
        "risk_tuning": {
            "risk_base_thresholds": new_settings.RISK_BASE_THRESHOLDS,
            "risk_severity_shift_coefficient": new_settings.RISK_SEVERITY_SHIFT_COEFFICIENT,
            "risk_confidence_shift_coefficient": new_settings.RISK_CONFIDENCE_SHIFT_COEFFICIENT,
            "risk_max_threshold_shift": new_settings.RISK_MAX_THRESHOLD_SHIFT,
            "risk_severity_baseline": new_settings.RISK_SEVERITY_BASELINE,
            "risk_confidence_baseline": new_settings.RISK_CONFIDENCE_BASELINE,
        }
    }
