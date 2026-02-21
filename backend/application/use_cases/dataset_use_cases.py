import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from infrastructure.database import DatasetModel
import json
import os
import logging

logger = logging.getLogger(__name__)

# Get the backend directory (grandparent of use_cases folder)
# Path: application/use_cases/dataset_use_cases.py -> application/use_cases -> application -> backend
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DatasetUseCases:
    
    @staticmethod
    def _safe_float(value) -> float:
        """Convert value to JSON-compliant float (handles NaN and Infinity)"""
        if value is None:
            return None
        try:
            f = float(value)
            if np.isnan(f) or np.isinf(f):
                return None
            return f
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _resolve_file_path(file_path: str) -> str:
        """Resolve relative file path to absolute path from backend directory"""
        if os.path.isabs(file_path):
            return file_path
        return os.path.join(BACKEND_DIR, file_path)
    
    @staticmethod
    async def upload_dataset(
        db: AsyncSession,
        name: str,
        description: str,
        file_path: str,
        uploaded_by: int
    ) -> DatasetModel:
        """Upload and analyze a dataset"""
        
        # Read the dataset
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        # Calculate statistics
        statistics = DatasetUseCases._calculate_statistics(df)
        
        # Create dataset record
        dataset = DatasetModel(
            name=name,
            description=description,
            file_path=file_path,
            record_count=len(df),
            column_count=len(df.columns),
            statistics=statistics,
            uploaded_by=uploaded_by
        )
        
        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)
        
        return dataset

    # =================================================================
    # SMART AUTO-PREPROCESSING & FEATURE ENGINEERING
    # =================================================================

    @staticmethod
    async def preprocess_dataset(
        db: AsyncSession,
        dataset_id: int,
        *,
        use_embeddings: bool = False,
    ) -> DatasetModel:
        """
        Smart auto-preprocessing pipeline.

        Automatically detects the dataset schema and applies:
          - Type inference (numeric, text, geo, categorical, binary, etc.)
          - Data cleaning (duplicates, type casting, missing imputation)
          - Outlier detection (IQR + Isolation Forest ensemble) → clipping
          - Text feature engineering (TF-IDF + SVD, fraud-linguistic patterns)
          - Numerical feature engineering (scaling, polynomial, binning, log)
          - Categorical encoding (frequency + count)
          - Geospatial features (K-Means clustering, Haversine distance)
          - Feature selection (variance, correlation, mutual information)

        The processed CSV is saved alongside the original and the dataset
        record is updated with the path + report.  The processed version
        is then used automatically by model training.
        """
        from application.use_cases.data_preprocessing_pipeline import (
            create_rental_fraud_pipeline,
        )

        result = await db.execute(
            select(DatasetModel).where(DatasetModel.id == dataset_id)
        )
        dataset = result.scalar_one_or_none()
        if not dataset:
            raise ValueError("Dataset not found")

        # Mark as processing
        dataset.preprocessing_status = "processing"
        await db.commit()

        try:
            # ── Load raw data ──
            raw_path = DatasetUseCases._resolve_file_path(dataset.file_path)
            if dataset.file_path.endswith('.csv'):
                df = pd.read_csv(raw_path, low_memory=False)
            else:
                df = pd.read_json(raw_path)

            original_shape = df.shape

            # ── Smart column detection ──
            # Detect target column
            target_col = None
            for cand in ['is_fraud', 'fraud', 'label', 'fraud_label', 'target']:
                if cand in df.columns:
                    target_col = cand
                    break

            # Detect text columns
            text_cols = []
            for cand in ['description', 'listing_title', 'listing_text', 'title',
                         'text', 'content', 'listing_description']:
                if cand in df.columns:
                    text_cols.append(cand)

            # Auto-detect remaining text-like columns (avg string length > 50)
            for col in df.select_dtypes(include='object').columns:
                if col in text_cols:
                    continue
                avg_len = df[col].dropna().astype(str).str.len().mean()
                if avg_len > 50:
                    text_cols.append(col)

            # Exclude non-useful columns
            exclude_cols = []
            for cand in ['link', 'url', 'address', 'rentfaster_id']:
                if cand in df.columns:
                    exclude_cols.append(cand)

            # ── Run pipeline ──
            pipeline = create_rental_fraud_pipeline(
                use_embeddings=use_embeddings,
                include_geo=True,
            )

            processed_df = pipeline.fit_transform(
                df,
                target_column=target_col,
                text_columns=text_cols if text_cols else None,
                exclude_columns=exclude_cols if exclude_cols else None,
            )

            # ── Re-attach the target column so it's available for training ──
            if target_col and target_col in df.columns:
                # Align indices after dedup
                target_series = df.loc[processed_df.index, target_col]
                processed_df[target_col] = target_series.values

            # ── Save processed CSV ──
            processed_dir = os.path.join(BACKEND_DIR, "data", "processed")
            os.makedirs(processed_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(dataset.file_path))[0]
            processed_filename = f"{base_name}_processed.csv"
            processed_abs = os.path.join(processed_dir, processed_filename)
            processed_df.to_csv(processed_abs, index=False)
            processed_rel = f"data/processed/{processed_filename}"

            # ── Build preprocessing report ──
            quality = pipeline.get_quality_report(df)
            report = {
                "original_shape": list(original_shape),
                "processed_shape": list(processed_df.shape),
                "features_created": int(processed_df.shape[1]),
                "quality_score": round(quality.data_quality_score * 100, 2),
                "target_column": target_col,
                "text_columns_detected": text_cols,
                "excluded_columns": exclude_cols,
                "type_map": pipeline.column_assignments,
                "transformation_log": pipeline.transformation_log,
            }

            # ── Update dataset record ──
            dataset.processed_file_path = processed_rel
            dataset.preprocessing_status = "completed"
            dataset.preprocessing_report = report
            dataset.feature_count = processed_df.shape[1]

            await db.commit()
            await db.refresh(dataset)

            logger.info(
                "Preprocessed dataset %d: %s → %s (%d features)",
                dataset_id, original_shape, processed_df.shape,
                processed_df.shape[1],
            )
            return dataset

        except Exception as exc:
            logger.error("Preprocessing failed for dataset %d: %s", dataset_id, exc)
            dataset.preprocessing_status = "failed"
            dataset.preprocessing_report = {"error": str(exc)}
            await db.commit()
            raise
    
    @staticmethod
    def _calculate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive statistics for the dataset"""
        
        stats = {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "missing_values": {},
            "duplicate_records": int(df.duplicated().sum()),
            "duplicate_percentage": float(df.duplicated().sum() / len(df) * 100),
            "column_types": {},
            "numeric_summary": {},
            "categorical_summary": {},
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024)
        }
        
        # Missing values analysis
        for col in df.columns:
            missing_count = int(df[col].isna().sum())
            stats["missing_values"][col] = {
                "count": missing_count,
                "percentage": float(missing_count / len(df) * 100)
            }
            stats["column_types"][col] = str(df[col].dtype)
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            stats["numeric_summary"][col] = {
                "mean": DatasetUseCases._safe_float(df[col].mean()) if not df[col].isna().all() else None,
                "median": DatasetUseCases._safe_float(df[col].median()) if not df[col].isna().all() else None,
                "std": DatasetUseCases._safe_float(df[col].std()) if not df[col].isna().all() else None,
                "min": DatasetUseCases._safe_float(df[col].min()) if not df[col].isna().all() else None,
                "max": DatasetUseCases._safe_float(df[col].max()) if not df[col].isna().all() else None,
                "q25": DatasetUseCases._safe_float(df[col].quantile(0.25)) if not df[col].isna().all() else None,
                "q75": DatasetUseCases._safe_float(df[col].quantile(0.75)) if not df[col].isna().all() else None
            }
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            unique_count = int(df[col].nunique())
            value_counts = df[col].value_counts().head(10).to_dict()
            stats["categorical_summary"][col] = {
                "unique_count": unique_count,
                "top_values": {str(k): int(v) for k, v in value_counts.items()}
            }
        
        return stats
    
    @staticmethod
    async def get_dataset(db: AsyncSession, dataset_id: int) -> DatasetModel:
        """Get dataset by ID"""
        result = await db.execute(select(DatasetModel).where(DatasetModel.id == dataset_id))
        return result.scalar_one_or_none()
    
    @staticmethod
    async def list_datasets(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[DatasetModel]:
        """List all datasets"""
        result = await db.execute(
            select(DatasetModel).offset(skip).limit(limit).order_by(DatasetModel.created_at.desc())
        )
        return result.scalars().all()
    
    @staticmethod
    async def get_dataset_preview(file_path: str, limit: int = 10) -> List[Dict]:
        """Get preview of dataset records"""
        resolved_path = DatasetUseCases._resolve_file_path(file_path)
        if file_path.endswith('.csv'):
            df = pd.read_csv(resolved_path, nrows=limit)
        elif file_path.endswith('.json'):
            df = pd.read_json(resolved_path)
            df = df.head(limit)
        else:
            raise ValueError("Unsupported file format")
        
        # Convert to list of dictionaries
        records = df.to_dict('records')
        # Convert numpy types to native Python types
        records = [{k: (v.item() if hasattr(v, 'item') else v) for k, v in record.items()} for record in records]
        return records
    
    @staticmethod
    async def delete_dataset(db: AsyncSession, dataset_id: int) -> bool:
        """Delete a dataset"""
        result = await db.execute(select(DatasetModel).where(DatasetModel.id == dataset_id))
        dataset = result.scalar_one_or_none()
        
        if not dataset:
            return False
        
        # Delete the file
        resolved_path = DatasetUseCases._resolve_file_path(dataset.file_path)
        if os.path.exists(resolved_path):
            os.remove(resolved_path)
        
        await db.delete(dataset)
        await db.commit()
        return True
    
    @staticmethod
    async def get_dataset_analysis(db: AsyncSession, dataset_id: int) -> Dict[str, Any]:
        """Get comprehensive dataset analysis report"""
        
        result = await db.execute(select(DatasetModel).where(DatasetModel.id == dataset_id))
        dataset = result.scalar_one_or_none()
        
        if not dataset:
            raise ValueError("Dataset not found")
        
        # Resolve the file path to absolute path
        resolved_path = DatasetUseCases._resolve_file_path(dataset.file_path)
        
        try:
            # Read dataset for analysis
            if dataset.file_path.endswith('.csv'):
                df = pd.read_csv(resolved_path)
            elif dataset.file_path.endswith('.json'):
                df = pd.read_json(resolved_path)
            else:
                raise ValueError("Unsupported file format")
        except Exception as e:
            raise ValueError(f"Failed to read dataset file: {str(e)}")
        
        statistics = dataset.statistics or {}
        
        # Overview
        overview = {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
            "file_size_mb": float(os.path.getsize(resolved_path) / 1024 / 1024) if os.path.exists(resolved_path) else 0,
            "file_format": dataset.file_path.split('.')[-1].upper(),
            "column_names": list(df.columns)
        }
        
        # Data Quality Analysis
        total_missing = sum(df[col].isna().sum() for col in df.columns)
        total_cells = len(df) * len(df.columns)
        missing_percentage = (total_missing / total_cells * 100) if total_cells > 0 else 0
        
        duplicate_count = int(df.duplicated().sum())
        duplicate_percentage = (duplicate_count / len(df) * 100) if len(df) > 0 else 0
        
        data_quality = {
            "completeness_score": float(100 - missing_percentage),
            "total_missing_values": int(total_missing),
            "missing_percentage": float(missing_percentage),
            "duplicate_records": duplicate_count,
            "duplicate_percentage": float(duplicate_percentage),
            "unique_records": int(len(df) - duplicate_count),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        # Column Analysis
        column_analysis = {}
        for col in df.columns:
            try:
                missing_count = int(df[col].isna().sum())
                unique_count = int(df[col].nunique())
                
                col_info = {
                    "data_type": str(df[col].dtype),
                    "missing_count": missing_count,
                    "missing_percentage": float(missing_count / len(df) * 100) if len(df) > 0 else 0.0,
                    "unique_count": unique_count,
                    "uniqueness_ratio": float(unique_count / len(df)) if len(df) > 0 else 0.0
                }
                
                # Add type-specific info
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_info["is_numeric"] = True
                    try:
                        col_info["has_outliers"] = bool(DatasetUseCases._detect_outliers(df[col]))
                    except Exception:
                        col_info["has_outliers"] = False
                else:
                    col_info["is_numeric"] = False
                    col_info["is_categorical"] = True
                    col_info["cardinality"] = "high" if unique_count > len(df) * 0.5 else "low"
                
                column_analysis[col] = col_info
            except Exception:
                # Skip columns that cause errors
                continue
        
        # Numeric Statistics (Enhanced)
        numeric_statistics = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            try:
                if not df[col].isna().all():
                    values = df[col].dropna()
                    
                    if len(values) == 0:
                        continue
                    
                    q1 = DatasetUseCases._safe_float(values.quantile(0.25)) or 0.0
                    q3 = DatasetUseCases._safe_float(values.quantile(0.75)) or 0.0
                    iqr = q3 - q1
                    
                    # Calculate skewness and kurtosis with error handling
                    try:
                        skewness = float(values.skew())
                        if np.isnan(skewness) or np.isinf(skewness):
                            skewness = 0.0
                    except Exception:
                        skewness = 0.0
                    
                    try:
                        kurtosis = float(values.kurtosis())
                        if np.isnan(kurtosis) or np.isinf(kurtosis):
                            kurtosis = 0.0
                    except Exception:
                        kurtosis = 0.0
                    
                    numeric_statistics[col] = {
                        "count": int(values.count()),
                        "mean": DatasetUseCases._safe_float(values.mean()),
                        "median": DatasetUseCases._safe_float(values.median()),
                        "std": DatasetUseCases._safe_float(values.std()),
                        "min": DatasetUseCases._safe_float(values.min()),
                        "max": DatasetUseCases._safe_float(values.max()),
                        "q25": q1,
                        "q50": DatasetUseCases._safe_float(values.median()),
                        "q75": q3,
                        "iqr": DatasetUseCases._safe_float(iqr),
                        "range": DatasetUseCases._safe_float(values.max() - values.min()),
                        "coefficient_of_variation": DatasetUseCases._safe_float(values.std() / values.mean()) if values.mean() != 0 else 0,
                        "skewness": skewness,
                        "kurtosis": kurtosis
                    }
            except Exception:
                # Skip columns that cause errors
                continue
        
        # Categorical Statistics (Enhanced)
        categorical_statistics = {}
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            try:
                if not df[col].isna().all():
                    values = df[col].dropna()
                    
                    if len(values) == 0:
                        continue
                    
                    value_counts = values.value_counts()
                    
                    # Calculate entropy with error handling
                    try:
                        probs = value_counts / len(values)
                        entropy = float(-sum(probs * np.log2(probs)))
                        if np.isnan(entropy) or np.isinf(entropy):
                            entropy = 0.0
                    except Exception:
                        entropy = 0.0
                    
                    categorical_statistics[col] = {
                        "unique_count": int(values.nunique()),
                        "most_frequent": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                        "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                        "most_frequent_percentage": float(value_counts.iloc[0] / len(values) * 100) if len(value_counts) > 0 and len(values) > 0 else 0,
                        "least_frequent": str(value_counts.index[-1]) if len(value_counts) > 0 else None,
                        "least_frequent_count": int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
                        "top_5_values": {str(k): int(v) for k, v in value_counts.head(5).items()},
                        "entropy": entropy
                    }
            except Exception:
                # Skip columns that cause errors
                continue
        
        # Recommendations
        recommendations = []
        
        try:
            if missing_percentage > 5:
                recommendations.append(f"Consider handling missing values ({missing_percentage:.1f}% of data is missing)")
            
            if duplicate_percentage > 1:
                recommendations.append(f"Remove duplicate records ({duplicate_count} duplicates found)")
            
            for col, stats in column_analysis.items():
                if stats.get("missing_percentage", 0) > 20:
                    recommendations.append(f"Column '{col}' has {stats['missing_percentage']:.1f}% missing values - consider imputation or removal")
                
                if stats.get("is_numeric") and stats.get("has_outliers"):
                    recommendations.append(f"Column '{col}' contains outliers - consider outlier treatment")
                
                if stats.get("is_categorical") and stats.get("cardinality") == "high" and stats.get("uniqueness_ratio", 0) > 0.9:
                    recommendations.append(f"Column '{col}' has high cardinality - may need encoding strategy for ML models")
            
            if len(numeric_cols) == 0:
                recommendations.append("No numeric columns found - consider feature engineering for ML models")
            
            if overview.get("memory_usage_mb", 0) > 100:
                recommendations.append(f"Dataset is large ({overview['memory_usage_mb']:.1f} MB) - consider data sampling or chunking for processing")
            
            if not recommendations:
                recommendations.append("Dataset quality is good - ready for modeling")
        except Exception:
            recommendations.append("Dataset loaded successfully")
        
        return {
            "id": dataset.id,
            "name": dataset.name,
            "description": dataset.description,
            "file_path": dataset.file_path,
            "created_at": dataset.created_at,
            "overview": overview,
            "data_quality": data_quality,
            "column_analysis": column_analysis,
            "numeric_statistics": numeric_statistics,
            "categorical_statistics": categorical_statistics,
            "recommendations": recommendations
        }
    
    @staticmethod
    def _detect_outliers(series: pd.Series) -> bool:
        """Detect if a numeric series has outliers using IQR method"""
        if series.isna().all():
            return False
        
        values = series.dropna()
        
        if len(values) < 4:
            return False
        
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = ((values < lower_bound) | (values > upper_bound)).sum()
        return bool(outliers > 0)
    
    @staticmethod
    async def generate_synthetic_fraud_data(
        db: AsyncSession,
        base_dataset_id: int,
        fraud_percentage: float,
        uploaded_by: int
    ) -> DatasetModel:
        """Generate synthetic fraud data based on existing dataset"""
        
        # Get base dataset
        base_dataset = await DatasetUseCases.get_dataset(db, base_dataset_id)
        if not base_dataset:
            raise ValueError("Base dataset not found")
        
        # Read base dataset
        resolved_path = DatasetUseCases._resolve_file_path(base_dataset.file_path)
        if base_dataset.file_path.endswith('.csv'):
            df = pd.read_csv(resolved_path)
        else:
            df = pd.read_json(resolved_path)
        
        # Generate synthetic fraud records
        fraud_count = int(len(df) * fraud_percentage)
        synthetic_records = []
        
        for _ in range(fraud_count):
            # Create suspicious patterns
            record = {
                "listing_title": np.random.choice([
                    "URGENT!!! AMAZING DEAL!!!",
                    "Too good to be true apartment",
                    "Luxury apt - wire deposit now",
                    "Beautiful home - pay now or lose it"
                ]),
                "description": np.random.choice([
                    "Must wire money immediately. Owner overseas. No viewing possible.",
                    "Send deposit via Western Union. Keys will be mailed.",
                    "Pay first month + deposit today. Property available tomorrow.",
                    "Cash only. No questions asked. Move in today."
                ]),
                "price": np.random.uniform(300, 800),  # Suspiciously low
                "contact_method": np.random.choice([
                    "email_only",
                    "whatsapp_only",
                    "telegram",
                    "burner_phone"
                ]),
                "urgency_level": np.random.uniform(0.8, 1.0),
                "payment_method": np.random.choice([
                    "wire_transfer_only",
                    "western_union",
                    "gift_cards",
                    "cryptocurrency"
                ]),
                "location": "Unspecified",
                "is_fraud": 1
            }
            synthetic_records.append(record)
        
        # Combine with original data (mark as non-fraud)
        if 'is_fraud' not in df.columns:
            df['is_fraud'] = 0
        
        synthetic_df = pd.DataFrame(synthetic_records)
        combined_df = pd.concat([df, synthetic_df], ignore_index=True)
        
        # Save synthetic dataset
        synthetic_file_path = f"data/synthetic_dataset_{base_dataset_id}.csv"
        os.makedirs("data", exist_ok=True)
        combined_df.to_csv(synthetic_file_path, index=False)
        
        # Create dataset record
        return await DatasetUseCases.upload_dataset(
            db=db,
            name=f"Synthetic Dataset (based on {base_dataset.name})",
            description=f"Synthetic fraud data ({fraud_percentage*100}% fraud records)",
            file_path=synthetic_file_path,
            uploaded_by=uploaded_by
        )

