from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List
import os


class Settings(BaseSettings):
    # Use absolute path to ensure consistent database location
    DATABASE_URL: str = f"sqlite+aiosqlite:///{os.path.dirname(os.path.abspath(__file__))}/rental_fraud.db"
    SECRET_KEY: str = "your-secret-key-change-in-production-09876543210987654321"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ENVIRONMENT: str = "development"
    
    # =========================================================================
    # RISK LEVEL DETERMINATION SETTINGS
    # =========================================================================
    # These settings control how the system determines fraud risk levels.
    # The risk level is NOT based on static thresholds alone — it combines:
    #   1. Base risk score (from ML model + rule-based indicators)
    #   2. Model confidence (how certain the system is)
    #   3. Cumulative severity of detected fraud indicators
    #
    # The final composite score is mapped to risk levels (VERY_LOW, LOW,
    # MEDIUM, HIGH, VERY_HIGH) using adaptive thresholds that shift based
    # on indicator pressure and confidence.
    # =========================================================================
    
    # -------------------------------------------------------------------------
    # BASE THRESHOLDS
    # -------------------------------------------------------------------------
    # These are the starting boundaries for risk level classification.
    # Format: [VERY_LOW/LOW boundary, LOW/MEDIUM boundary, 
    #          MEDIUM/HIGH boundary, HIGH/VERY_HIGH boundary]
    #
    # Example with default [0.15, 0.30, 0.50, 0.70]:
    #   - composite_score < 0.15 → VERY_LOW
    #   - composite_score < 0.30 → LOW
    #   - composite_score < 0.50 → MEDIUM
    #   - composite_score < 0.70 → HIGH
    #   - composite_score >= 0.70 → VERY_HIGH
    #
    # ADMIN TIP: Lower values = stricter (more listings flagged as risky)
    #            Higher values = more lenient (fewer false positives)
    # -------------------------------------------------------------------------
    RISK_BASE_THRESHOLDS: List[float] = [0.15, 0.30, 0.50, 0.70]
    
    # -------------------------------------------------------------------------
    # SEVERITY SHIFT COEFFICIENT
    # -------------------------------------------------------------------------
    # Controls how much the cumulative severity of detected indicators
    # affects the risk level thresholds.
    #
    # When indicators with high severity are detected, thresholds shift DOWN
    # (stricter detection). This coefficient controls the magnitude.
    #
    # Formula: shift += (severity_normalized - 0.4) * SEVERITY_SHIFT_COEFFICIENT
    #
    # Higher value (e.g., 0.25) = Severity has MORE impact on risk level
    # Lower value (e.g., 0.10) = Severity has LESS impact on risk level
    #
    # ADMIN TIP: Increase if you want multiple fraud indicators to more
    #            aggressively push listings toward higher risk levels.
    # -------------------------------------------------------------------------
    RISK_SEVERITY_SHIFT_COEFFICIENT: float = 0.18
    
    # -------------------------------------------------------------------------
    # CONFIDENCE SHIFT COEFFICIENT
    # -------------------------------------------------------------------------
    # Controls how much the model's confidence affects risk level thresholds.
    #
    # When confidence is HIGH (>0.7), thresholds shift DOWN (stricter).
    # When confidence is LOW (<0.7), thresholds shift UP (more lenient,
    # because we're less certain about the prediction).
    #
    # Formula: shift += (confidence - 0.7) * CONFIDENCE_SHIFT_COEFFICIENT
    #
    # Higher value (e.g., 0.20) = Confidence has MORE impact
    # Lower value (e.g., 0.05) = Confidence has LESS impact
    #
    # ADMIN TIP: Increase if you trust high-confidence predictions more
    #            and want them to be more decisive.
    # -------------------------------------------------------------------------
    RISK_CONFIDENCE_SHIFT_COEFFICIENT: float = 0.12
    
    # -------------------------------------------------------------------------
    # MAXIMUM THRESHOLD SHIFT
    # -------------------------------------------------------------------------
    # Caps how much the thresholds can shift up or down from their base values.
    # This prevents extreme shifts that could cause erratic behavior.
    #
    # Value of 0.20 means thresholds can shift at most ±0.20 from base.
    # Example: base threshold 0.50 can become 0.30 to 0.70 at most.
    #
    # ADMIN TIP: Keep this moderate (0.15-0.25) for stable behavior.
    # -------------------------------------------------------------------------
    RISK_MAX_THRESHOLD_SHIFT: float = 0.20
    
    # -------------------------------------------------------------------------
    # SEVERITY BASELINE
    # -------------------------------------------------------------------------
    # The "neutral point" for severity. When severity_normalized equals this
    # value, severity contributes zero shift to thresholds.
    #
    # Below this value → thresholds shift UP (more lenient)
    # Above this value → thresholds shift DOWN (stricter)
    #
    # ADMIN TIP: Lower value (e.g., 0.3) = stricter overall
    #            Higher value (e.g., 0.5) = more lenient overall
    # -------------------------------------------------------------------------
    RISK_SEVERITY_BASELINE: float = 0.4
    
    # -------------------------------------------------------------------------
    # CONFIDENCE BASELINE
    # -------------------------------------------------------------------------
    # The "neutral point" for confidence. When confidence equals this value,
    # confidence contributes zero shift to thresholds.
    #
    # Below this value → thresholds shift UP (more lenient, less certain)
    # Above this value → thresholds shift DOWN (stricter, more certain)
    #
    # ADMIN TIP: Standard value is 0.70 (70% confidence as neutral).
    # -------------------------------------------------------------------------
    RISK_CONFIDENCE_BASELINE: float = 0.7

    # -------------------------------------------------------------------------
    # GOOGLE CLOUD VISION API KEY
    # -------------------------------------------------------------------------
    # Used for reverse image search (Web Detection) via the Google Cloud
    # Vision API.  The first 1 000 requests/month are FREE, then $1.50/1 000.
    #
    # Set this in the .env file:
    #   GOOGLE_CLOUD_VISION_API_KEY=AIza...
    #
    # When left empty, web detection is silently skipped — all other image
    # analysis features continue to work normally.
    # -------------------------------------------------------------------------
    GOOGLE_CLOUD_VISION_API_KEY: str = ""
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()

