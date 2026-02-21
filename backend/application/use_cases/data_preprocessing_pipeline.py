"""
Advanced Data Preprocessing & Feature Engineering Pipeline
===========================================================
COMP 385 AI Capstone — AI-Powered Rental Fraud & Trust Scoring System

Production-grade, end-to-end pipeline for rental fraud detection datasets.

ML/AI Techniques Used:
  1. Automated Type Inference — pattern + heuristic column-type detection
  2. KNN Imputation          — sklearn KNNImputer for numeric missing values
  3. Iterative Imputation     — MICE (Multiple Imputation by Chained Equations)
  4. Isolation Forest          — unsupervised outlier detection (ensemble)
  5. Local Outlier Factor      — density-based outlier detection
  6. TF-IDF + Truncated SVD   — text vectorization with dimensionality reduction
  7. Sentence-BERT Embeddings  — deep semantic text representations
  8. K-Means Clustering        — geospatial location clustering
  9. Haversine Distance        — Earth-surface distance calculation
 10. Yeo-Johnson Transform     — power transformation for skewed distributions
 11. Mutual Information        — information-theoretic feature selection
 12. Polynomial Features       — non-linear interaction terms

Pipeline Steps (in order):
  Step 1 — Load & Profile        → type inference, quality check
  Step 2 — Clean                 → duplicates, missing values, type casting
  Step 3 — Outlier Handling      → detect (IF + LOF + IQR) → clip/remove
  Step 4 — Text Feature Eng.     → TF-IDF / embeddings + fraud patterns
  Step 5 — Numeric Feature Eng.  → scaling, polynomial, binning, log
  Step 6 — Categorical Encoding  → frequency / target / one-hot
  Step 7 — Geospatial Features   → clustering, distance, density
  Step 8 — Feature Selection     → variance, correlation, mutual info
  Step 9 — Export                 → save processed CSV + pipeline artefacts

Author : Group #2
"""

from __future__ import annotations

import logging
import os
import json
import re
import math
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional heavy-weight imports (graceful degradation)
# ---------------------------------------------------------------------------
try:
    from sklearn.preprocessing import (
        StandardScaler, MinMaxScaler, RobustScaler,
        LabelEncoder, OneHotEncoder, PolynomialFeatures,
        PowerTransformer, QuantileTransformer,
    )
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_selection import (
        mutual_info_classif,
        mutual_info_regression,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ENUMS                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ColumnType(str, Enum):
    """Semantic column types detected by the TypeInferenceEngine."""
    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE   = "numeric_discrete"
    CURRENCY           = "currency"
    CATEGORICAL        = "categorical"
    BINARY             = "binary"
    TEXT_SHORT          = "text_short"
    TEXT_LONG           = "text_long"
    DATETIME           = "datetime"
    GEOSPATIAL_LAT     = "geospatial_lat"
    GEOSPATIAL_LON     = "geospatial_lon"
    ID                 = "id"
    URL                = "url"
    EMAIL              = "email"
    PHONE              = "phone"
    CONSTANT           = "constant"
    UNKNOWN            = "unknown"


class ImputationStrategy(str, Enum):
    MEAN      = "mean"
    MEDIAN    = "median"
    MODE      = "mode"
    KNN       = "knn"
    ITERATIVE = "iterative"
    CONSTANT  = "constant"


class OutlierMethod(str, Enum):
    IQR              = "iqr"
    ZSCORE           = "zscore"
    ISOLATION_FOREST = "isolation_forest"
    LOF              = "lof"


class ScalingMethod(str, Enum):
    STANDARD = "standard"
    MINMAX   = "minmax"
    ROBUST   = "robust"
    POWER    = "power"       # Yeo-Johnson
    QUANTILE = "quantile"
    LOG      = "log"
    NONE     = "none"


class TextVectorizationMethod(str, Enum):
    TFIDF      = "tfidf"
    EMBEDDINGS = "embeddings"
    BOTH       = "both"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATA CLASSES                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass
class QualityReport:
    """Data quality assessment."""
    rows: int = 0
    cols: int = 0
    data_quality_score: float = 0.0
    duplicate_rows: int = 0
    total_missing: int = 0
    missing_pct: float = 0.0
    missing_value_summary: Dict[str, int] = field(default_factory=dict)
    type_summary: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class PipelineResult:
    """Everything returned by the pipeline."""
    original_shape: Tuple[int, int]
    processed_shape: Tuple[int, int]
    quality: QualityReport
    type_map: Dict[str, str]
    column_groups: Dict[str, List[str]]
    features_created: int
    feature_importance: Dict[str, float]
    outliers_found: Dict[str, int]
    steps_log: List[Dict[str, Any]]
    processed_df: pd.DataFrame


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 1 — TYPE INFERENCE ENGINE                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_RE_ID   = re.compile(r'(_id\b|^id$|_key\b|_pk\b)', re.I)
_RE_URL  = re.compile(r'^(https?://|www\.|/\w+/)', re.I)
_RE_DATE = re.compile(
    r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|'
    r'\d{1,2}[-/]\d{1,2}[-/]\d{4}|'
    r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',
    re.I,
)
_RE_EMAIL = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
_RE_PHONE = re.compile(r'\+?\d[\d\-\s().]{7,}\d')


class TypeInferenceEngine:
    """ML-powered automatic column type detection."""

    def infer_types(self, df: pd.DataFrame) -> Dict[str, ColumnType]:
        return {col: self._infer_one(df[col]) for col in df.columns}

    @staticmethod
    def _infer_one(series: pd.Series) -> ColumnType:
        name = (series.name or "").lower()
        sample = series.dropna()
        if len(sample) == 0:
            return ColumnType.UNKNOWN

        n_unique = sample.nunique()
        n_total  = len(sample)

        # ── name hints ──
        if _RE_ID.search(name):
            return ColumnType.ID
        if any(s in name for s in ('lat', 'latitude')):
            return ColumnType.GEOSPATIAL_LAT
        if any(s in name for s in ('lon', 'lng', 'longitude')):
            return ColumnType.GEOSPATIAL_LON
        if any(s in name for s in ('price', 'rent', 'cost', 'fee', 'amount')):
            if pd.api.types.is_numeric_dtype(series):
                return ColumnType.CURRENCY

        # ── dtype checks ──
        if pd.api.types.is_bool_dtype(series) or (n_unique == 2 and set(sample.unique()) <= {0, 1, True, False, 'Yes', 'No', 'yes', 'no'}):
            return ColumnType.BINARY
        if pd.api.types.is_datetime64_any_dtype(series):
            return ColumnType.DATETIME
        if pd.api.types.is_numeric_dtype(series):
            if n_unique == 1:
                return ColumnType.CONSTANT
            if n_unique / n_total < 0.05 and n_unique <= 30:
                return ColumnType.NUMERIC_DISCRETE
            return ColumnType.NUMERIC_CONTINUOUS

        # ── object columns ──
        s_str = sample.astype(str)
        avg_len = s_str.str.len().mean()

        if n_unique == 1:
            return ColumnType.CONSTANT

        # URL
        if s_str.head(50).apply(lambda x: bool(_RE_URL.match(x))).mean() > 0.6:
            return ColumnType.URL
        # email
        if s_str.head(50).apply(lambda x: bool(_RE_EMAIL.match(x))).mean() > 0.6:
            return ColumnType.EMAIL
        # phone
        if s_str.head(50).apply(lambda x: bool(_RE_PHONE.match(x))).mean() > 0.5:
            return ColumnType.PHONE
        # datetime
        if s_str.head(50).apply(lambda x: bool(_RE_DATE.search(x))).mean() > 0.6:
            try:
                pd.to_datetime(s_str.head(100))
                return ColumnType.DATETIME
            except Exception:
                pass

        # text vs categorical
        ratio = n_unique / n_total
        if avg_len > 80 or (ratio > 0.5 and avg_len > 30):
            return ColumnType.TEXT_LONG
        if avg_len > 40:
            return ColumnType.TEXT_SHORT
        if ratio < 0.05 or n_unique <= 30:
            return ColumnType.CATEGORICAL

        return ColumnType.CATEGORICAL


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 2 — DATA CLEANING                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _clean_dataframe(
    df: pd.DataFrame,
    type_map: Dict[str, ColumnType],
    *,
    drop_duplicates: bool = True,
    drop_constant: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Return (cleaned df, log messages)."""
    log: List[str] = []
    out = df.copy()

    if drop_duplicates:
        before = len(out)
        out = out.drop_duplicates()
        dropped = before - len(out)
        if dropped:
            log.append(f"Dropped {dropped} duplicate rows")

    if drop_constant:
        consts = [c for c, t in type_map.items()
                  if t == ColumnType.CONSTANT and c in out.columns]
        if consts:
            out.drop(columns=consts, inplace=True)
            log.append(f"Dropped {len(consts)} constant columns: {consts}")

    # cast numeric-looking columns
    for col, ct in type_map.items():
        if col not in out.columns:
            continue
        if ct in (ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE,
                  ColumnType.CURRENCY):
            if not pd.api.types.is_numeric_dtype(out[col]):
                out[col] = pd.to_numeric(
                    out[col].astype(str).str.replace(r'[^\d.\-]', '', regex=True),
                    errors='coerce',
                )
                log.append(f"Cast '{col}' → numeric")
        elif ct == ColumnType.BINARY:
            mapping = {'true': 1, 'false': 0, 'yes': 1, 'no': 0, '1': 1, '0': 0}
            if out[col].dtype == object:
                out[col] = out[col].astype(str).str.strip().str.lower().map(mapping)
                log.append(f"Encoded binary column '{col}' → 0/1")

    # known column quirks in rental data
    if 'sq_feet' in out.columns and out['sq_feet'].dtype == object:
        out['sq_feet'] = pd.to_numeric(
            out['sq_feet'].astype(str).str.replace(r'[^\d.]', '', regex=True),
            errors='coerce',
        )
        log.append("Cleaned sq_feet → numeric")

    if 'beds' in out.columns and out['beds'].dtype == object:
        def _parse_beds(v):
            v = str(v).strip().lower()
            if v in ('studio', 'bachelor'):
                return 0
            m = re.search(r'(\d+)', v)
            return int(m.group(1)) if m else np.nan
        out['beds_numeric'] = out['beds'].apply(_parse_beds)
        log.append("Parsed 'beds' → beds_numeric")

    return out, log


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 2b — MISSING VALUE IMPUTATION                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _impute_missing(
    df: pd.DataFrame,
    type_map: Dict[str, ColumnType],
    *,
    numeric_strategy: str = 'knn',
) -> Tuple[pd.DataFrame, List[str]]:
    log: List[str] = []
    out = df.copy()

    num_types = (ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE,
                 ColumnType.CURRENCY, ColumnType.GEOSPATIAL_LAT, ColumnType.GEOSPATIAL_LON)
    num_cols = [c for c in out.columns
                if type_map.get(c) in num_types
                and pd.api.types.is_numeric_dtype(out[c])
                and out[c].isnull().any()]

    if num_cols:
        if numeric_strategy == 'knn' and SKLEARN_AVAILABLE:
            try:
                imp = KNNImputer(n_neighbors=min(5, len(out) - 1))
                out[num_cols] = imp.fit_transform(out[num_cols])
                log.append(f"KNN-imputed {len(num_cols)} numeric columns")
            except Exception as exc:
                logger.warning("KNN imputation failed (%s); median fallback", exc)
                for c in num_cols:
                    out[c].fillna(out[c].median(), inplace=True)
                log.append(f"Median-imputed {len(num_cols)} numeric columns")
        elif numeric_strategy == 'iterative' and SKLEARN_AVAILABLE:
            try:
                imp = IterativeImputer(max_iter=10, random_state=42)
                out[num_cols] = imp.fit_transform(out[num_cols])
                log.append(f"MICE-imputed {len(num_cols)} numeric columns")
            except Exception:
                for c in num_cols:
                    out[c].fillna(out[c].median(), inplace=True)
                log.append(f"Median-imputed {len(num_cols)} numeric columns")
        else:
            for c in num_cols:
                out[c].fillna(out[c].median(), inplace=True)
            log.append(f"Median-imputed {len(num_cols)} numeric columns")

    # categorical / binary
    cat_cols = [c for c in out.columns
                if type_map.get(c) in (ColumnType.CATEGORICAL, ColumnType.BINARY)
                and out[c].isnull().any()]
    for c in cat_cols:
        mode = out[c].mode()
        out[c] = out[c].fillna(mode.iloc[0] if len(mode) > 0 else 'Unknown')
    if cat_cols:
        log.append(f"Mode-filled {len(cat_cols)} categorical columns")

    # text
    txt_types = (ColumnType.TEXT_SHORT, ColumnType.TEXT_LONG)
    txt_cols = [c for c in out.columns
                if type_map.get(c) in txt_types and out[c].isnull().any()]
    for c in txt_cols:
        out[c] = out[c].fillna('')
    if txt_cols:
        log.append(f"Filled {len(txt_cols)} text columns with empty string")

    return out, log


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 3 — OUTLIER DETECTION ENGINE                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class OutlierDetectionEngine:
    """Multi-method ensemble outlier detector (IQR + IF + LOF)."""

    def __init__(
        self,
        methods: List[OutlierMethod] | None = None,
        contamination: float = 0.05,
    ):
        self.methods = methods or [OutlierMethod.IQR, OutlierMethod.ISOLATION_FOREST]
        self.contamination = contamination
        self.thresholds: Dict[str, Dict] = {}
        self._masks: Dict[str, np.ndarray] = {}

    def fit(self, df: pd.DataFrame, numeric_cols: List[str]):
        """Fit detectors and store masks."""
        self._masks = {}
        for col in numeric_cols:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            vals = df[col].values.astype(float)
            if np.all(np.isnan(vals)):
                continue
            votes: List[np.ndarray] = []
            col_thresholds: Dict[str, Any] = {}

            if OutlierMethod.IQR in self.methods:
                q1, q3 = np.nanpercentile(vals, [25, 75])
                iqr = q3 - q1
                if iqr > 0:
                    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    votes.append((vals < lo) | (vals > hi))
                    col_thresholds['iqr'] = {'lower': float(lo), 'upper': float(hi)}

            if OutlierMethod.ZSCORE in self.methods:
                mean, std = np.nanmean(vals), np.nanstd(vals)
                if std > 0:
                    z = np.abs((vals - mean) / std)
                    votes.append(z > 3.0)
                    col_thresholds['zscore_threshold'] = 3.0

            if OutlierMethod.ISOLATION_FOREST in self.methods and SKLEARN_AVAILABLE:
                try:
                    clean = vals[~np.isnan(vals)].reshape(-1, 1)
                    if len(clean) > 10:
                        iso = IsolationForest(
                            contamination=self.contamination,
                            n_estimators=100, random_state=42,
                        )
                        preds = iso.fit_predict(clean)
                        full = np.zeros(len(vals), dtype=bool)
                        full[~np.isnan(vals)] = preds == -1
                        votes.append(full)
                except Exception:
                    pass

            if OutlierMethod.LOF in self.methods and SKLEARN_AVAILABLE:
                try:
                    clean = vals[~np.isnan(vals)].reshape(-1, 1)
                    n_neigh = min(20, len(clean) - 1)
                    if n_neigh > 1:
                        lof = LocalOutlierFactor(
                            n_neighbors=n_neigh,
                            contamination=self.contamination,
                        )
                        preds = lof.fit_predict(clean)
                        full = np.zeros(len(vals), dtype=bool)
                        full[~np.isnan(vals)] = preds == -1
                        votes.append(full)
                except Exception:
                    pass

            if votes:
                self._masks[col] = np.mean(np.column_stack(votes), axis=1) >= 0.5
            self.thresholds[col] = col_thresholds

    def detect(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        return dict(self._masks)


def _handle_outliers(
    df: pd.DataFrame,
    masks: Dict[str, np.ndarray],
    strategy: str = 'clip',
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    out = df.copy()
    counts: Dict[str, int] = {}
    for col, mask in masks.items():
        n = int(mask.sum())
        counts[col] = n
        if n == 0:
            continue
        if strategy == 'clip':
            q1, q3 = np.nanpercentile(out[col].values, [25, 75])
            iqr = q3 - q1
            out[col] = out[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
        elif strategy == 'nan':
            out.loc[mask, col] = np.nan
        elif strategy == 'median':
            out.loc[mask, col] = out[col].median()
    if strategy == 'remove':
        combined = np.zeros(len(out), dtype=bool)
        for m in masks.values():
            combined |= m
        out = out[~combined].reset_index(drop=True)
    return out, counts


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 4 — TEXT FEATURE ENGINE                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

URGENCY_WORDS = [
    'urgent', 'asap', 'immediately', 'hurry', 'now', 'limited time',
    'act fast', "don't miss", 'last chance', 'hurry up', 'quick',
]
PAYMENT_RED_FLAGS = [
    'wire transfer', 'western union', 'moneygram', 'gift card',
    'bitcoin', 'crypto', 'cash only', 'venmo', 'zelle', 'paypal',
    'e-transfer', 'money order',
]
SCAM_PHRASES = [
    'overseas', 'military', 'deployed', "can't meet", 'keys by mail',
    'no viewing', 'send deposit', 'too good to be true', 'owner traveling',
    'no questions asked', 'sight unseen', 'pay first',
]


class TextFeatureEngine:
    """NLP-based text feature extraction engine."""

    def __init__(
        self,
        vectorization_method: TextVectorizationMethod = TextVectorizationMethod.TFIDF,
        use_fraud_features: bool = True,
        tfidf_max_features: int = 300,
        svd_components: int = 50,
    ):
        self.method = vectorization_method
        self.use_fraud = use_fraud_features
        self.tfidf_max = tfidf_max_features
        self.svd_dims = svd_components

    def fit_transform(
        self, texts: pd.Series, prefix: str = 'txt',
    ) -> pd.DataFrame:
        parts: List[pd.DataFrame] = []
        parts.append(self._stat_features(texts, prefix))
        if self.use_fraud:
            parts.append(self._fraud_features(texts, prefix))
        if self.method in (TextVectorizationMethod.TFIDF, TextVectorizationMethod.BOTH):
            tdf = self._tfidf(texts, prefix)
            if not tdf.empty:
                parts.append(tdf)
        if self.method in (TextVectorizationMethod.EMBEDDINGS, TextVectorizationMethod.BOTH):
            edf = self._embeddings(texts, prefix)
            if not edf.empty:
                parts.append(edf)
        return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=texts.index)

    # ── statistical features ──
    @staticmethod
    def _stat_features(texts: pd.Series, prefix: str) -> pd.DataFrame:
        s = texts.fillna('').astype(str)
        f = pd.DataFrame(index=texts.index)
        f[f'{prefix}_char_count']     = s.str.len()
        f[f'{prefix}_word_count']     = s.str.split().str.len().fillna(0).astype(int)
        f[f'{prefix}_sentence_count'] = s.str.count(r'[.!?]+').fillna(0).astype(int)
        f[f'{prefix}_avg_word_length'] = s.apply(
            lambda x: np.mean([len(w) for w in x.split()]) if x.strip() else 0
        )
        f[f'{prefix}_upper_ratio']    = s.apply(lambda x: sum(c.isupper() for c in x) / max(len(x), 1))
        f[f'{prefix}_digit_ratio']    = s.apply(lambda x: sum(c.isdigit() for c in x) / max(len(x), 1))
        f[f'{prefix}_special_ratio']  = s.apply(
            lambda x: sum(not c.isalnum() and not c.isspace() for c in x) / max(len(x), 1)
        )
        f[f'{prefix}_exclamation_count'] = s.str.count('!')
        f[f'{prefix}_question_count']    = s.str.count(r'\?')
        f[f'{prefix}_allcaps_words']     = s.apply(
            lambda x: sum(1 for w in x.split() if w.isupper() and len(w) > 1)
        )
        return f

    # ── fraud-specific features ──
    @staticmethod
    def _fraud_features(texts: pd.Series, prefix: str) -> pd.DataFrame:
        low = texts.fillna('').astype(str).str.lower()
        f = pd.DataFrame(index=texts.index)
        f[f'{prefix}_urgency_score']     = low.apply(lambda x: sum(1 for p in URGENCY_WORDS if p in x))
        f[f'{prefix}_payment_flag_count'] = low.apply(lambda x: sum(1 for p in PAYMENT_RED_FLAGS if p in x))
        f[f'{prefix}_scam_phrase_count'] = low.apply(lambda x: sum(1 for p in SCAM_PHRASES if p in x))
        f[f'{prefix}_has_email']          = low.str.contains(r'\b[\w.]+@[\w.]+\.\w+\b').astype(int)
        f[f'{prefix}_has_phone']          = low.str.contains(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b').astype(int)
        f[f'{prefix}_has_whatsapp']       = low.str.contains('whatsapp').astype(int)
        f[f'{prefix}_mentions_deposit']   = low.str.contains('deposit').astype(int)
        # composite
        f[f'{prefix}_fraud_linguistic_score'] = (
            f[f'{prefix}_urgency_score'] * 0.3
            + f[f'{prefix}_payment_flag_count'] * 0.4
            + f[f'{prefix}_scam_phrase_count'] * 0.3
        )
        return f

    # ── TF-IDF + SVD ──
    def _tfidf(self, texts: pd.Series, prefix: str) -> pd.DataFrame:
        if not SKLEARN_AVAILABLE:
            return pd.DataFrame(index=texts.index)
        s = texts.fillna('').astype(str)
        vec = TfidfVectorizer(
            max_features=self.tfidf_max, ngram_range=(1, 2),
            stop_words='english', min_df=2, max_df=0.95,
        )
        try:
            mat = vec.fit_transform(s)
        except ValueError:
            return pd.DataFrame(index=texts.index)
        if mat.shape[1] > self.svd_dims:
            svd = TruncatedSVD(n_components=self.svd_dims, random_state=42)
            reduced = svd.fit_transform(mat)
            cols = [f'{prefix}_tfidf_{i}' for i in range(self.svd_dims)]
            return pd.DataFrame(reduced, columns=cols, index=texts.index)
        cols = [f'{prefix}_tfidf_{i}' for i in range(mat.shape[1])]
        return pd.DataFrame(mat.toarray(), columns=cols, index=texts.index)

    # ── sentence embeddings ──
    @staticmethod
    def _embeddings(
        texts: pd.Series, prefix: str,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        batch_size: int = 64,
    ) -> pd.DataFrame:
        if not TORCH_AVAILABLE:
            logger.warning("torch/transformers unavailable — skipping embeddings")
            return pd.DataFrame(index=texts.index)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
        except Exception as exc:
            logger.warning("Could not load embedding model (%s)", exc)
            return pd.DataFrame(index=texts.index)

        s = texts.fillna('').astype(str).tolist()
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(s), batch_size):
                batch = s[i:i + batch_size]
                enc = tokenizer(batch, padding=True, truncation=True,
                                max_length=256, return_tensors='pt')
                out = model(**enc)
                mask = enc['attention_mask'].unsqueeze(-1).float()
                pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                all_embs.append(pooled.cpu().numpy())

        arr = np.vstack(all_embs)
        cols = [f'{prefix}_emb_{i}' for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols, index=texts.index)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 5 — NUMERICAL FEATURE ENGINE                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class NumericalFeatureEngine:
    """Numerical feature engineering: scaling, polynomial, binning, log."""

    def __init__(
        self,
        scaling_method: ScalingMethod = ScalingMethod.ROBUST,
        add_poly: bool = True,
        add_log: bool = True,
        n_bins: int = 10,
    ):
        self.scaling_method = scaling_method
        self.add_poly = add_poly
        self.add_log = add_log
        self.n_bins = n_bins

    def fit_transform(
        self, df: pd.DataFrame, num_cols: List[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        log: List[str] = []
        if not num_cols:
            return pd.DataFrame(index=df.index), log

        data = df[num_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
        parts: List[pd.DataFrame] = []

        # scaling
        if self.scaling_method != ScalingMethod.NONE and SKLEARN_AVAILABLE:
            _map = {
                ScalingMethod.STANDARD: StandardScaler,
                ScalingMethod.MINMAX:   MinMaxScaler,
                ScalingMethod.ROBUST:   RobustScaler,
            }
            if self.scaling_method in _map:
                scaler = _map[self.scaling_method]()
            elif self.scaling_method == ScalingMethod.POWER:
                scaler = PowerTransformer(method='yeo-johnson', standardize=True)
            elif self.scaling_method == ScalingMethod.QUANTILE:
                scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
            else:
                scaler = None

            if scaler is not None:
                try:
                    scaled = scaler.fit_transform(data)
                    parts.append(pd.DataFrame(
                        scaled,
                        columns=[f'{c}_scaled' for c in num_cols],
                        index=df.index,
                    ))
                    log.append(f"Scaled {len(num_cols)} cols ({self.scaling_method.value})")
                except Exception as exc:
                    logger.warning("Scaling failed: %s", exc)

        # log transform (positive only)
        if self.add_log:
            log_df = pd.DataFrame(index=df.index)
            for c in num_cols:
                if (data[c] > 0).all():
                    log_df[f'{c}_log'] = np.log1p(data[c])
            if not log_df.empty:
                parts.append(log_df)
                log.append(f"Log-transformed {len(log_df.columns)} columns")

        # polynomial + interaction (≤8 cols to avoid explosion)
        if self.add_poly and SKLEARN_AVAILABLE and 1 < len(num_cols) <= 8:
            try:
                poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
                poly_data = poly.fit_transform(data)
                names = poly.get_feature_names_out(num_cols)
                new_idx = [i for i, n in enumerate(names) if '^' in n or ' ' in n]
                if new_idx:
                    parts.append(pd.DataFrame(
                        poly_data[:, new_idx],
                        columns=[names[i] for i in new_idx],
                        index=df.index,
                    ))
                    log.append(f"Created {len(new_idx)} polynomial / interaction features")
            except Exception as exc:
                logger.warning("Polynomial features: %s", exc)

        # quantile binning
        if self.n_bins and self.n_bins > 1:
            bin_df = pd.DataFrame(index=df.index)
            for c in num_cols:
                try:
                    bin_df[f'{c}_bin'] = pd.qcut(data[c], q=self.n_bins,
                                                  labels=False, duplicates='drop')
                except Exception:
                    pass
            if not bin_df.empty:
                parts.append(bin_df)
                log.append(f"Binned {len(bin_df.columns)} columns × {self.n_bins}")

        # row-level aggregates
        if len(num_cols) > 1:
            agg = pd.DataFrame(index=df.index)
            agg['num_row_mean']  = data.mean(axis=1)
            agg['num_row_std']   = data.std(axis=1)
            agg['num_row_range'] = data.max(axis=1) - data.min(axis=1)
            parts.append(agg)
            log.append("Added row-level numeric aggregates")

        if parts:
            return pd.concat(parts, axis=1), log
        return pd.DataFrame(index=df.index), log


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 6 — CATEGORICAL FEATURE ENGINE                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class CategoricalFeatureEngine:
    """Frequency / One-Hot / Label encoding for categoricals."""

    def __init__(self, method: str = 'frequency', max_categories: int = 50):
        self.method = method
        self.max_cat = max_categories

    def fit_transform(
        self, df: pd.DataFrame, cat_cols: List[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        log: List[str] = []
        if not cat_cols:
            return pd.DataFrame(index=df.index), log

        parts: List[pd.DataFrame] = []
        for c in cat_cols:
            if c not in df.columns:
                continue
            vals = df[c].astype(str)

            if self.method == 'frequency':
                freq = vals.value_counts(normalize=True).to_dict()
                parts.append(pd.DataFrame({f'{c}_freq': vals.map(freq)}, index=df.index))
            elif self.method == 'onehot':
                top = vals.value_counts().head(self.max_cat).index
                for cat in top:
                    parts.append(pd.DataFrame({f'{c}_{cat}': (vals == cat).astype(int)}, index=df.index))
            elif self.method == 'label':
                mapping = {v: i for i, v in enumerate(vals.unique())}
                parts.append(pd.DataFrame({f'{c}_label': vals.map(mapping)}, index=df.index))

            # always add count
            cnt = vals.value_counts().to_dict()
            parts.append(pd.DataFrame({f'{c}_count': vals.map(cnt)}, index=df.index))

        if parts:
            result = pd.concat(parts, axis=1)
            log.append(f"Encoded {len(cat_cols)} categorical columns ({self.method})")
            return result, log
        return pd.DataFrame(index=df.index), log


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 7 — GEOSPATIAL FEATURE ENGINE                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


class GeospatialFeatureEngine:
    """K-Means clustering + distance + price anomaly on geo data."""

    def __init__(self, n_clusters: int = 20):
        self.n_clusters = n_clusters

    def fit_transform(
        self,
        df: pd.DataFrame,
        lat_col: str,
        lon_col: str,
        price_col: str | None = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        log: List[str] = []
        if lat_col not in df.columns or lon_col not in df.columns:
            return pd.DataFrame(index=df.index), log

        coords = df[[lat_col, lon_col]].copy()
        valid = coords.dropna()
        if len(valid) < self.n_clusters:
            return pd.DataFrame(index=df.index), log

        feat = pd.DataFrame(index=df.index)

        if SKLEARN_AVAILABLE:
            try:
                km = KMeans(n_clusters=min(self.n_clusters, len(valid)),
                            random_state=42, n_init=10)
                cluster_labels = km.fit_predict(valid)

                # Map cluster labels back to full df index
                label_series = pd.Series(np.full(len(df), -1), index=df.index)
                label_series.loc[valid.index] = cluster_labels
                feat['geo_cluster'] = label_series.values

                # Distance to assigned cluster centre
                centres = km.cluster_centers_
                dist_series = pd.Series(np.nan, index=df.index)
                for pos, idx in enumerate(valid.index):
                    cl = cluster_labels[pos]
                    dist_series.loc[idx] = _haversine(
                        coords.loc[idx, lat_col], coords.loc[idx, lon_col],
                        centres[cl][0], centres[cl][1],
                    )
                feat['geo_dist_to_centre'] = dist_series.values

                # Price vs cluster average (fraud signal)
                if price_col and price_col in df.columns:
                    cl_series = pd.Series(cluster_labels, index=valid.index)
                    cl_means = df.loc[valid.index].groupby(cl_series)[price_col].mean()
                    avg_mapped = label_series.map(cl_means)
                    feat['price_vs_cluster'] = (df[price_col] / avg_mapped.replace(0, 1)).values
                    feat['price_cluster_diff'] = (df[price_col] - avg_mapped).values

                log.append(f"Created {self.n_clusters} geo-clusters + distance + price features")
            except Exception as exc:
                logger.warning("Geo clustering: %s", exc)

        feat[f'{lat_col}_norm'] = (coords[lat_col] - coords[lat_col].mean()) / (coords[lat_col].std() + 1e-9)
        feat[f'{lon_col}_norm'] = (coords[lon_col] - coords[lon_col].mean()) / (coords[lon_col].std() + 1e-9)
        log.append("Normalised coordinates")
        return feat, log


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 8 — FEATURE SELECTION ENGINE                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class FeatureSelectionEngine:
    """Variance + correlation + mutual-information feature selection."""

    def __init__(
        self,
        var_threshold: float = 0.01,
        corr_threshold: float = 0.95,
        top_k: int | None = None,
    ):
        self.var_threshold = var_threshold
        self.corr_threshold = corr_threshold
        self.top_k = top_k
        self._importance: Dict[str, float] = {}
        self._report: Dict[str, Any] = {}

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> Tuple[pd.DataFrame, Dict[str, float], List[str]]:
        log: List[str] = []
        num_df = X.select_dtypes(include=[np.number]).copy()
        num_df = num_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        if num_df.empty:
            return X, {}, log

        dropped: List[str] = []

        # low variance
        variances = num_df.var()
        low_var = variances[variances < self.var_threshold].index.tolist()
        if low_var:
            num_df.drop(columns=low_var, inplace=True, errors='ignore')
            dropped.extend(low_var)
            log.append(f"Removed {len(low_var)} low-variance features")

        # high correlation
        if len(num_df.columns) > 1:
            corr = num_df.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            high_corr = [c for c in upper.columns if any(upper[c] > self.corr_threshold)]
            if high_corr:
                num_df.drop(columns=high_corr, inplace=True, errors='ignore')
                dropped.extend(high_corr)
                log.append(f"Removed {len(high_corr)} highly-correlated features (r > {self.corr_threshold})")

        # mutual information
        importance: Dict[str, float] = {}
        if y is not None and SKLEARN_AVAILABLE and len(num_df.columns) > 0:
            try:
                y_al = y.loc[num_df.index]
                if pd.api.types.is_numeric_dtype(y_al) and y_al.nunique() <= 10:
                    mi = mutual_info_classif(num_df, y_al, random_state=42)
                else:
                    mi = mutual_info_regression(num_df, y_al, random_state=42)
                importance = dict(zip(num_df.columns, mi))

                if self.top_k and self.top_k < len(num_df.columns):
                    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                    keep = [r[0] for r in ranked[:self.top_k]]
                    num_df = num_df[keep]
                    log.append(f"Selected top-{self.top_k} by mutual information")
            except Exception as exc:
                logger.warning("Mutual info: %s", exc)

        self._importance = importance
        self._report = {
            'dropped_low_variance': low_var if 'low_var' in dir() else [],
            'dropped_high_corr': high_corr if 'high_corr' in dir() else [],
            'importance': {k: round(float(v), 6) for k, v in importance.items()},
            'final_count': len(num_df.columns),
        }

        non_num = X.select_dtypes(exclude=[np.number])
        result = pd.concat([non_num, num_df], axis=1)
        log.append(f"Final features: {len(result.columns)} (dropped {len(dropped)})")
        return result, importance, log

    def get_feature_report(self):
        return self._report


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 9 — QUALITY REPORT                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _build_quality_report(
    df: pd.DataFrame,
    type_map: Dict[str, ColumnType],
) -> QualityReport:
    cells = len(df) * len(df.columns)
    total_null = int(df.isnull().sum().sum())
    null_pct = (total_null / cells * 100) if cells else 0
    dupes = int(df.duplicated().sum())
    quality = max(0.0, 1 - null_pct / 100 - dupes / max(len(df), 1) * 0.5)

    missing_by_col = {c: int(df[c].isnull().sum()) for c in df.columns if df[c].isnull().any()}
    type_summary = {}
    for t in set(type_map.values()):
        type_summary[t.value] = sum(1 for v in type_map.values() if v == t)

    warnings: List[str] = []
    recommendations: List[str] = []
    for c in df.columns:
        null_rate = df[c].isnull().mean()
        if null_rate > 0.5:
            warnings.append(f"'{c}' has {null_rate:.0%} missing values")
        if null_rate > 0.8:
            recommendations.append(f"Consider dropping column '{c}' (>{null_rate:.0%} missing)")
    if dupes > 0:
        recommendations.append(f"Found {dupes} duplicate rows — consider deduplication")

    return QualityReport(
        rows=len(df), cols=len(df.columns),
        data_quality_score=round(quality, 4),
        duplicate_rows=dupes, total_missing=total_null,
        missing_pct=round(null_pct, 2),
        missing_value_summary=missing_by_col,
        type_summary=type_summary,
        warnings=warnings,
        recommendations=recommendations,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MASTER PIPELINE — DataPreprocessingPipeline                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class DataPreprocessingPipeline:
    """
    End-to-end orchestrator.

    Usage::

        pipeline = DataPreprocessingPipeline()
        df_out = pipeline.fit_transform(df, target_column='is_fraud')
    """

    def __init__(
        self,
        *,
        use_embeddings: bool = False,
        include_geo: bool = True,
        scaling_method: ScalingMethod = ScalingMethod.ROBUST,
        cat_encoding: str = 'frequency',
        outlier_strategy: str = 'clip',
        contamination: float = 0.05,
        tfidf_max: int = 300,
        svd_dims: int = 50,
        geo_clusters: int = 20,
        do_selection: bool = True,
        var_threshold: float = 0.01,
        corr_threshold: float = 0.95,
    ):
        self.type_engine    = TypeInferenceEngine()
        self.text_engine    = TextFeatureEngine(
            vectorization_method=(TextVectorizationMethod.BOTH if use_embeddings
                                  else TextVectorizationMethod.TFIDF),
            use_fraud_features=True,
            tfidf_max_features=tfidf_max,
            svd_components=svd_dims,
        )
        self.numerical_engine = NumericalFeatureEngine(scaling_method=scaling_method)
        self.categorical_engine = CategoricalFeatureEngine(method=cat_encoding)
        self.geo_engine     = GeospatialFeatureEngine(n_clusters=geo_clusters) if include_geo else None
        self.outlier_engine = OutlierDetectionEngine(
            methods=[OutlierMethod.IQR, OutlierMethod.ISOLATION_FOREST],
            contamination=contamination,
        )
        self.selection_engine = FeatureSelectionEngine(
            var_threshold=var_threshold,
            corr_threshold=corr_threshold,
        ) if do_selection else None

        self.outlier_handling = outlier_strategy
        self.include_geo = include_geo

        # state populated during fit_transform
        self.type_map: Dict[str, ColumnType] = {}
        self.column_assignments: Dict[str, str] = {}
        self.transformation_log: List[str] = []

    # ── helpers ──
    def _log(self, msg: str):
        entry = f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}"
        self.transformation_log.append(entry)
        logger.info("[Pipeline] %s", msg)

    def get_quality_report(self, df: pd.DataFrame) -> QualityReport:
        type_map = self.type_engine.infer_types(df)
        return _build_quality_report(df, type_map)

    # ── main entry point ──
    def fit_transform(
        self,
        df: pd.DataFrame,
        *,
        target_column: str | None = None,
        text_columns: List[str] | None = None,
        exclude_columns: List[str] | None = None,
    ) -> pd.DataFrame:
        """
        Run the full pipeline end-to-end.

        Returns the processed DataFrame with all engineered features.
        """
        self.transformation_log = []
        self._log(f"Starting pipeline on {df.shape[0]} rows × {df.shape[1]} cols")

        # ── 0. exclusions ──
        exclude = set(exclude_columns or [])
        if target_column:
            exclude.add(target_column)
        target_series = df[target_column].copy() if target_column and target_column in df.columns else None
        working = df.drop(columns=[c for c in exclude if c in df.columns], errors='ignore').copy()

        # ── 1. type inference ──
        self.type_map = self.type_engine.infer_types(working)
        self._log(f"Detected types for {len(self.type_map)} columns")

        if text_columns:
            for c in text_columns:
                if c in self.type_map:
                    self.type_map[c] = ColumnType.TEXT_LONG

        # ── 2. clean ──
        working, clog = _clean_dataframe(working, self.type_map)
        for m in clog:
            self._log(m)
        self.type_map = {c: t for c, t in self.type_map.items() if c in working.columns}

        # ── 3. impute ──
        working, ilog = _impute_missing(working, self.type_map)
        for m in ilog:
            self._log(m)

        # ── 4. outliers ──
        num_types = (ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE, ColumnType.CURRENCY)
        num_cols = [c for c, t in self.type_map.items()
                    if t in num_types and c in working.columns
                    and pd.api.types.is_numeric_dtype(working[c])]
        self.outlier_engine.fit(working, num_cols)
        outlier_masks = self.outlier_engine.detect(working)
        working, outlier_counts = _handle_outliers(working, outlier_masks, self.outlier_handling)
        self._log(f"Outliers: {sum(outlier_counts.values())} across {len(outlier_counts)} cols → {self.outlier_handling}")

        # ── group columns ──
        groups: Dict[str, List[str]] = {
            'numeric':     [c for c, t in self.type_map.items() if t in num_types and c in working.columns],
            'categorical': [c for c, t in self.type_map.items() if t == ColumnType.CATEGORICAL and c in working.columns],
            'text':        [c for c, t in self.type_map.items() if t in (ColumnType.TEXT_SHORT, ColumnType.TEXT_LONG) and c in working.columns],
            'binary':      [c for c, t in self.type_map.items() if t == ColumnType.BINARY and c in working.columns],
            'geo_lat':     [c for c, t in self.type_map.items() if t == ColumnType.GEOSPATIAL_LAT and c in working.columns],
            'geo_lon':     [c for c, t in self.type_map.items() if t == ColumnType.GEOSPATIAL_LON and c in working.columns],
            'id':          [c for c, t in self.type_map.items() if t == ColumnType.ID and c in working.columns],
            'url':         [c for c, t in self.type_map.items() if t == ColumnType.URL and c in working.columns],
            'datetime':    [c for c, t in self.type_map.items() if t == ColumnType.DATETIME and c in working.columns],
        }
        self.column_assignments = {c: t.value for c, t in self.type_map.items()}

        # ── 5. text features ──
        text_feat_parts: List[pd.DataFrame] = []
        for col in groups['text']:
            self._log(f"TextFE on '{col}'")
            tf = self.text_engine.fit_transform(working[col], prefix=col)
            if not tf.empty:
                text_feat_parts.append(tf)
                self._log(f"  → {tf.shape[1]} features from '{col}'")

        # ── 6. numeric features ──
        pure_num = [c for c in groups['numeric']
                    if c not in groups.get('geo_lat', [])
                    and c not in groups.get('geo_lon', [])
                    and pd.api.types.is_numeric_dtype(working[c])]
        if 'beds_numeric' in working.columns:
            pure_num.append('beds_numeric')
        num_feat, nlog = self.numerical_engine.fit_transform(working, pure_num)
        for m in nlog:
            self._log(m)

        # ── 7. categorical ──
        cat_feat, catlog = self.categorical_engine.fit_transform(working, groups['categorical'])
        for m in catlog:
            self._log(m)

        # ── 8. geospatial ──
        geo_feat = pd.DataFrame(index=working.index)
        if self.include_geo and self.geo_engine and groups['geo_lat'] and groups['geo_lon']:
            lat_c, lon_c = groups['geo_lat'][0], groups['geo_lon'][0]
            price_c = None
            for cand in ['price', 'rent', 'listing_price']:
                if cand in working.columns:
                    price_c = cand
                    break
            geo_feat, glog = self.geo_engine.fit_transform(working, lat_c, lon_c, price_col=price_c)
            for m in glog:
                self._log(m)

        # ── assemble ──
        base_cols = groups['numeric'] + groups['binary']
        if 'beds_numeric' in working.columns:
            base_cols.append('beds_numeric')
        base = working[[c for c in base_cols if c in working.columns]].copy()

        all_parts = [base]
        all_parts.extend(text_feat_parts)
        if not num_feat.empty:
            all_parts.append(num_feat)
        if not cat_feat.empty:
            all_parts.append(cat_feat)
        if not geo_feat.empty:
            all_parts.append(geo_feat)

        assembled = pd.concat(all_parts, axis=1)
        assembled = assembled.loc[:, ~assembled.columns.duplicated()]
        self._log(f"Assembled feature matrix: {assembled.shape[1]} features")

        # ── 9. feature selection ──
        importance: Dict[str, float] = {}
        if self.selection_engine:
            assembled, importance, slog = self.selection_engine.fit_transform(assembled, target_series)
            for m in slog:
                self._log(m)

        self._log(f"Pipeline complete → {assembled.shape[0]} rows × {assembled.shape[1]} features")
        return assembled


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FACTORY                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def create_rental_fraud_pipeline(
    *,
    use_embeddings: bool = False,
    include_geo: bool = True,
) -> DataPreprocessingPipeline:
    """Pre-configured pipeline for rental fraud detection datasets."""
    return DataPreprocessingPipeline(
        use_embeddings=use_embeddings,
        include_geo=include_geo,
        scaling_method=ScalingMethod.ROBUST,
        cat_encoding='frequency',
        outlier_strategy='clip',
        contamination=0.05,
        do_selection=True,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ASYNC CONVENIENCE                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

async def run_pipeline_on_dataset(
    file_path: str,
    *,
    target_col: str | None = None,
    text_cols: List[str] | None = None,
    exclude_cols: List[str] | None = None,
    use_embeddings: bool = False,
    save_dir: str | None = None,
) -> Dict[str, Any]:
    """Async entry-point: load CSV → run pipeline → return JSON-safe dict."""
    abs_path = os.path.join(BACKEND_DIR, file_path) if not os.path.isabs(file_path) else file_path
    if not os.path.exists(abs_path):
        return {'success': False, 'error': f'File not found: {abs_path}'}

    try:
        df = pd.read_csv(abs_path)
    except Exception as exc:
        return {'success': False, 'error': f'Read CSV failed: {exc}'}

    pipe = create_rental_fraud_pipeline(use_embeddings=use_embeddings)
    processed = pipe.fit_transform(
        df, target_column=target_col,
        text_columns=text_cols, exclude_columns=exclude_cols,
    )

    saved_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, 'processed.csv')
        processed.to_csv(csv_path, index=False)
        saved_path = csv_path

        meta = {
            'original_shape': list(df.shape),
            'processed_shape': list(processed.shape),
            'type_map': pipe.column_assignments,
            'transformation_log': pipe.transformation_log,
            'saved_at': datetime.utcnow().isoformat(),
        }
        with open(os.path.join(save_dir, 'pipeline_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2, default=str)

    quality = pipe.get_quality_report(df)

    return {
        'success': True,
        'original_shape': list(df.shape),
        'processed_shape': list(processed.shape),
        'quality_score': round(quality.data_quality_score * 100, 2),
        'features_created': processed.shape[1],
        'type_map': pipe.column_assignments,
        'transformation_log': pipe.transformation_log,
        'saved_to': saved_path,
        'sample_rows': processed.head(5).to_dict(orient='records'),
    }
