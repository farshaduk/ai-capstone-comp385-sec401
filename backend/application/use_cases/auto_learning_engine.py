"""
Auto-Learning Engine - Continuous improvement from user feedback

This module implements an auto-learning system that improves fraud detection
based on confirmed user feedback. Key capabilities:

1. Feedback Aggregation - Collect confirmed fraud/safe labels
2. Pattern Learning - Extract patterns from confirmed cases
3. Weight Calibration - Adjust indicator weights based on real data
4. Model Retraining Triggers - Signal when retraining is beneficial

The engine does NOT replace the existing detection system, but enhances it
by learning from real-world feedback.
"""

import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import logging
import os


@dataclass
class FeedbackPattern:
    """A pattern learned from confirmed feedback"""
    pattern_type: str  # 'keyword', 'price_range', 'indicator_combination'
    pattern_value: Any
    fraud_count: int
    safe_count: int
    confidence: float
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "pattern_value": self.pattern_value,
            "fraud_count": self.fraud_count,
            "safe_count": self.safe_count,
            "confidence": self.confidence,
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class LearningInsight:
    """Insight derived from feedback analysis"""
    insight_type: str
    description: str
    evidence_count: int
    recommended_action: str
    priority: str  # 'high', 'medium', 'low'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_type": self.insight_type,
            "description": self.description,
            "evidence_count": self.evidence_count,
            "recommended_action": self.recommended_action,
            "priority": self.priority
        }


@dataclass
class CalibrationResult:
    """Result of weight calibration"""
    indicator_code: str
    original_weight: float
    calibrated_weight: float
    sample_size: int
    confidence: float


class AutoLearningEngine:
    """
    Auto-Learning Engine for continuous fraud detection improvement.
    
    Responsibilities:
    - Analyze user feedback to find patterns
    - Calibrate indicator weights based on confirmed data
    - Generate insights for model improvement
    - Track learning metrics over time
    """
    
    # Minimum samples needed for reliable pattern learning
    MIN_SAMPLES_FOR_LEARNING = 10
    
    # Confidence threshold for applying learned weights
    CONFIDENCE_THRESHOLD = 0.7
    
    # Path to store learned patterns
    LEARNED_PATTERNS_PATH = "data/learned_patterns.json"
    
    def __init__(self):
        """Initialize the auto-learning engine"""
        self._logger = logging.getLogger(__name__)
        self._learned_patterns: Dict[str, FeedbackPattern] = {}
        self._calibrated_weights: Dict[str, float] = {}
        self._load_learned_patterns()
    
    def _load_learned_patterns(self):
        """Load previously learned patterns from disk"""
        try:
            if os.path.exists(self.LEARNED_PATTERNS_PATH):
                with open(self.LEARNED_PATTERNS_PATH, 'r') as f:
                    data = json.load(f)
                    self._learned_patterns = {
                        k: FeedbackPattern(
                            pattern_type=v["pattern_type"],
                            pattern_value=v["pattern_value"],
                            fraud_count=v["fraud_count"],
                            safe_count=v["safe_count"],
                            confidence=v["confidence"],
                            last_updated=datetime.fromisoformat(v["last_updated"])
                        )
                        for k, v in data.get("patterns", {}).items()
                    }
                    self._calibrated_weights = data.get("calibrated_weights", {})
                    self._logger.info(f"Loaded {len(self._learned_patterns)} learned patterns")
        except Exception as e:
            self._logger.warning(f"Failed to load learned patterns: {e}")
    
    def _save_learned_patterns(self):
        """Save learned patterns to disk"""
        try:
            os.makedirs(os.path.dirname(self.LEARNED_PATTERNS_PATH), exist_ok=True)
            data = {
                "patterns": {k: v.to_dict() for k, v in self._learned_patterns.items()},
                "calibrated_weights": self._calibrated_weights,
                "last_saved": datetime.utcnow().isoformat()
            }
            with open(self.LEARNED_PATTERNS_PATH, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self._logger.error(f"Failed to save learned patterns: {e}")
    
    async def learn_from_feedback(
        self,
        db: AsyncSession,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze recent feedback to learn new patterns.
        
        Args:
            db: Database session
            days_back: How many days of feedback to analyze
        
        Returns:
            Learning report with patterns found and insights
        """
        from infrastructure.database import FeedbackModel, RiskAnalysisModel
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        # Get feedback with associated analyses (only admin-approved feedback)
        result = await db.execute(
            select(FeedbackModel, RiskAnalysisModel)
            .join(RiskAnalysisModel, FeedbackModel.analysis_id == RiskAnalysisModel.id)
            .where(FeedbackModel.created_at >= cutoff_date)
            .where(FeedbackModel.feedback_type.in_(['safe', 'fraud']))
            .where(FeedbackModel.status == 'approved')
        )
        
        feedback_data = result.all()
        
        if len(feedback_data) < self.MIN_SAMPLES_FOR_LEARNING:
            return {
                "status": "insufficient_data",
                "message": f"Need at least {self.MIN_SAMPLES_FOR_LEARNING} confirmed feedback items",
                "current_count": len(feedback_data),
                "patterns_found": 0,
                "insights": []
            }
        
        # Separate fraud and safe cases
        fraud_cases = [(f, a) for f, a in feedback_data if f.feedback_type == 'fraud']
        safe_cases = [(f, a) for f, a in feedback_data if f.feedback_type == 'safe']
        
        # Learn patterns
        keyword_patterns = self._learn_keyword_patterns(fraud_cases, safe_cases)
        indicator_patterns = self._learn_indicator_patterns(fraud_cases, safe_cases)
        price_patterns = self._learn_price_patterns(fraud_cases, safe_cases)
        
        # Update learned patterns
        all_patterns = {**keyword_patterns, **indicator_patterns, **price_patterns}
        self._learned_patterns.update(all_patterns)
        
        # Calibrate weights
        calibration_results = self._calibrate_indicator_weights(fraud_cases, safe_cases)
        
        # Generate insights
        insights = self._generate_insights(
            fraud_cases, safe_cases, all_patterns, calibration_results
        )
        
        # Save learned patterns
        self._save_learned_patterns()
        
        return {
            "status": "success",
            "total_feedback": len(feedback_data),
            "fraud_count": len(fraud_cases),
            "safe_count": len(safe_cases),
            "patterns_found": len(all_patterns),
            "patterns": [p.to_dict() for p in all_patterns.values()],
            "calibration_results": [
                {
                    "indicator": c.indicator_code,
                    "original_weight": c.original_weight,
                    "calibrated_weight": c.calibrated_weight,
                    "sample_size": c.sample_size,
                    "confidence": c.confidence
                }
                for c in calibration_results
            ],
            "insights": [i.to_dict() for i in insights],
            "learning_timestamp": datetime.utcnow().isoformat()
        }
    
    def _learn_keyword_patterns(
        self,
        fraud_cases: List[Tuple],
        safe_cases: List[Tuple]
    ) -> Dict[str, FeedbackPattern]:
        """Learn keyword patterns from confirmed cases"""
        patterns = {}
        
        # Extract words from fraud cases
        fraud_words = {}
        for _, analysis in fraud_cases:
            if analysis.listing_text:
                words = analysis.listing_text.lower().split()
                for word in words:
                    if len(word) > 4:  # Skip short words
                        fraud_words[word] = fraud_words.get(word, 0) + 1
        
        # Extract words from safe cases
        safe_words = {}
        for _, analysis in safe_cases:
            if analysis.listing_text:
                words = analysis.listing_text.lower().split()
                for word in words:
                    if len(word) > 4:
                        safe_words[word] = safe_words.get(word, 0) + 1
        
        # Find discriminative keywords
        for word, fraud_count in fraud_words.items():
            safe_count = safe_words.get(word, 0)
            total = fraud_count + safe_count
            
            if total >= 5:  # Minimum occurrences
                fraud_ratio = fraud_count / total
                
                if fraud_ratio > 0.7:  # Strongly associated with fraud
                    confidence = min(0.5 + (total / 50), 0.95)
                    patterns[f"keyword_{word}"] = FeedbackPattern(
                        pattern_type="keyword_fraud",
                        pattern_value=word,
                        fraud_count=fraud_count,
                        safe_count=safe_count,
                        confidence=confidence,
                        last_updated=datetime.utcnow()
                    )
                elif fraud_ratio < 0.3:  # Strongly associated with safe
                    confidence = min(0.5 + (total / 50), 0.95)
                    patterns[f"keyword_safe_{word}"] = FeedbackPattern(
                        pattern_type="keyword_safe",
                        pattern_value=word,
                        fraud_count=fraud_count,
                        safe_count=safe_count,
                        confidence=confidence,
                        last_updated=datetime.utcnow()
                    )
        
        return patterns
    
    def _learn_indicator_patterns(
        self,
        fraud_cases: List[Tuple],
        safe_cases: List[Tuple]
    ) -> Dict[str, FeedbackPattern]:
        """Learn which indicator combinations are most predictive"""
        patterns = {}
        
        # Count indicator occurrences in fraud cases
        fraud_indicators = {}
        for _, analysis in fraud_cases:
            if analysis.risk_indicators:
                indicators = analysis.risk_indicators
                if isinstance(indicators, list):
                    for ind in indicators:
                        code = ind.get('code', str(ind))
                        fraud_indicators[code] = fraud_indicators.get(code, 0) + 1
        
        # Count indicator occurrences in safe cases
        safe_indicators = {}
        for _, analysis in safe_cases:
            if analysis.risk_indicators:
                indicators = analysis.risk_indicators
                if isinstance(indicators, list):
                    for ind in indicators:
                        code = ind.get('code', str(ind))
                        safe_indicators[code] = safe_indicators.get(code, 0) + 1
        
        # Find discriminative indicators
        all_indicators = set(fraud_indicators.keys()) | set(safe_indicators.keys())
        
        for indicator in all_indicators:
            fraud_count = fraud_indicators.get(indicator, 0)
            safe_count = safe_indicators.get(indicator, 0)
            total = fraud_count + safe_count
            
            if total >= 3:  # Minimum occurrences
                fraud_ratio = fraud_count / total if total > 0 else 0
                confidence = min(0.5 + (total / 30), 0.95)
                
                patterns[f"indicator_{indicator}"] = FeedbackPattern(
                    pattern_type="indicator",
                    pattern_value=indicator,
                    fraud_count=fraud_count,
                    safe_count=safe_count,
                    confidence=confidence,
                    last_updated=datetime.utcnow()
                )
        
        return patterns
    
    def _learn_price_patterns(
        self,
        fraud_cases: List[Tuple],
        safe_cases: List[Tuple]
    ) -> Dict[str, FeedbackPattern]:
        """Learn price range patterns"""
        patterns = {}
        
        fraud_prices = [a.listing_price for _, a in fraud_cases if a.listing_price]
        safe_prices = [a.listing_price for _, a in safe_cases if a.listing_price]
        
        if len(fraud_prices) >= 5 and len(safe_prices) >= 5:
            fraud_median = np.median(fraud_prices)
            safe_median = np.median(safe_prices)
            
            # If fraud listings tend to be significantly cheaper
            if fraud_median < safe_median * 0.7:
                patterns["price_too_low"] = FeedbackPattern(
                    pattern_type="price_range",
                    pattern_value={
                        "fraud_median": fraud_median,
                        "safe_median": safe_median,
                        "threshold": safe_median * 0.6
                    },
                    fraud_count=len(fraud_prices),
                    safe_count=len(safe_prices),
                    confidence=0.75,
                    last_updated=datetime.utcnow()
                )
        
        return patterns
    
    def _calibrate_indicator_weights(
        self,
        fraud_cases: List[Tuple],
        safe_cases: List[Tuple]
    ) -> List[CalibrationResult]:
        """Calibrate indicator weights based on feedback"""
        from application.use_cases.indicator_engine import IndicatorEngine
        
        results = []
        
        # Default weights from indicator engine
        default_weights = {
            "OFF_PLATFORM_PAYMENT": 0.20,
            "CRYPTO_PAYMENT": 0.25,
            "GIFT_CARD_PAYMENT": 0.25,
            "UPFRONT_PAYMENT": 0.18,
            "OWNER_UNAVAILABLE": 0.18,
            "NO_VIEWING": 0.22,
            "HIGH_URGENCY": 0.10,
            "SUSPICIOUS_LOW_PRICE": 0.20,
        }
        
        # Count true positives and false positives for each indicator
        indicator_stats = {}
        
        for _, analysis in fraud_cases:
            if analysis.risk_indicators:
                for ind in analysis.risk_indicators:
                    code = ind.get('code', str(ind)) if isinstance(ind, dict) else str(ind)
                    if code not in indicator_stats:
                        indicator_stats[code] = {"tp": 0, "fp": 0}
                    indicator_stats[code]["tp"] += 1
        
        for _, analysis in safe_cases:
            if analysis.risk_indicators:
                for ind in analysis.risk_indicators:
                    code = ind.get('code', str(ind)) if isinstance(ind, dict) else str(ind)
                    if code not in indicator_stats:
                        indicator_stats[code] = {"tp": 0, "fp": 0}
                    indicator_stats[code]["fp"] += 1
        
        # Calculate calibrated weights
        for code, stats in indicator_stats.items():
            total = stats["tp"] + stats["fp"]
            if total >= 5:
                precision = stats["tp"] / total
                original_weight = default_weights.get(code, 0.10)
                
                # Adjust weight based on precision
                # High precision = increase weight
                # Low precision = decrease weight
                calibrated = original_weight * (0.5 + precision)
                calibrated = min(max(calibrated, 0.05), 0.35)  # Clamp
                
                confidence = min(0.5 + (total / 50), 0.95)
                
                results.append(CalibrationResult(
                    indicator_code=code,
                    original_weight=original_weight,
                    calibrated_weight=round(calibrated, 3),
                    sample_size=total,
                    confidence=round(confidence, 2)
                ))
                
                # Store calibrated weight if confidence is high enough
                if confidence >= self.CONFIDENCE_THRESHOLD:
                    self._calibrated_weights[code] = calibrated
        
        return results
    
    def _generate_insights(
        self,
        fraud_cases: List[Tuple],
        safe_cases: List[Tuple],
        patterns: Dict[str, FeedbackPattern],
        calibrations: List[CalibrationResult]
    ) -> List[LearningInsight]:
        """Generate actionable insights from learning"""
        insights = []
        
        # Insight: New fraud keywords discovered
        new_fraud_keywords = [
            p.pattern_value for p in patterns.values()
            if p.pattern_type == "keyword_fraud" and p.confidence > 0.8
        ]
        if new_fraud_keywords:
            insights.append(LearningInsight(
                insight_type="new_fraud_keywords",
                description=f"Discovered {len(new_fraud_keywords)} new keywords strongly associated with fraud: {', '.join(new_fraud_keywords[:5])}",
                evidence_count=len(new_fraud_keywords),
                recommended_action="Consider adding these keywords to the Indicator Engine patterns",
                priority="high" if len(new_fraud_keywords) >= 3 else "medium"
            ))
        
        # Insight: Indicators needing weight adjustment
        significant_changes = [
            c for c in calibrations
            if abs(c.calibrated_weight - c.original_weight) > 0.05 and c.confidence > 0.7
        ]
        if significant_changes:
            insights.append(LearningInsight(
                insight_type="weight_calibration",
                description=f"{len(significant_changes)} indicators have significantly different effectiveness than expected",
                evidence_count=sum(c.sample_size for c in significant_changes),
                recommended_action="Review and update indicator weights in the Indicator Engine",
                priority="high"
            ))
        
        # Insight: False positive rate
        total_feedback = len(fraud_cases) + len(safe_cases)
        if total_feedback >= 20:
            # Cases where system said high risk but user said safe
            false_positives = [
                (f, a) for f, a in safe_cases
                if a.risk_score and a.risk_score > 0.6
            ]
            fp_rate = len(false_positives) / len(safe_cases) if safe_cases else 0
            
            if fp_rate > 0.3:
                insights.append(LearningInsight(
                    insight_type="high_false_positive_rate",
                    description=f"False positive rate is {fp_rate:.1%} - system is flagging too many legitimate listings",
                    evidence_count=len(false_positives),
                    recommended_action="Review indicator thresholds and consider reducing sensitivity",
                    priority="high"
                ))
        
        # Insight: Model retraining suggested
        if len(fraud_cases) >= 50 and len(safe_cases) >= 50:
            insights.append(LearningInsight(
                insight_type="retraining_recommended",
                description="Sufficient labeled data available for model retraining",
                evidence_count=len(fraud_cases) + len(safe_cases),
                recommended_action="Consider retraining the ML model with the new labeled feedback data",
                priority="medium"
            ))
        
        return insights
    
    def get_calibrated_weight(self, indicator_code: str, default: float) -> float:
        """
        Get the calibrated weight for an indicator.
        
        Falls back to default if no calibration is available.
        """
        return self._calibrated_weights.get(indicator_code, default)
    
    def get_learned_fraud_keywords(self) -> List[str]:
        """Get list of keywords learned to be associated with fraud"""
        return [
            p.pattern_value for p in self._learned_patterns.values()
            if p.pattern_type == "keyword_fraud" and p.confidence > 0.7
        ]
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get current learning statistics"""
        return {
            "total_patterns": len(self._learned_patterns),
            "calibrated_weights": len(self._calibrated_weights),
            "fraud_keywords": len(self.get_learned_fraud_keywords()),
            "last_updated": max(
                (p.last_updated for p in self._learned_patterns.values()),
                default=None
            )
        }
    
    async def get_retraining_dataset(
        self,
        db: AsyncSession,
        min_confidence: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Generate a labeled dataset for model retraining from confirmed feedback.
        
        Returns:
            List of labeled samples ready for training
        """
        from infrastructure.database import FeedbackModel, RiskAnalysisModel
        
        result = await db.execute(
            select(FeedbackModel, RiskAnalysisModel)
            .join(RiskAnalysisModel, FeedbackModel.analysis_id == RiskAnalysisModel.id)
            .where(FeedbackModel.feedback_type.in_(['safe', 'fraud']))
            .where(FeedbackModel.status == 'approved')
        )
        
        feedback_data = result.all()
        
        training_samples = []
        for feedback, analysis in feedback_data:
            label = 1 if feedback.feedback_type == 'fraud' else 0
            
            training_samples.append({
                "listing_text": analysis.listing_text,
                "listing_price": analysis.listing_price,
                "label": label,
                "feedback_type": feedback.feedback_type,
                "original_score": analysis.risk_score,
                "feedback_date": feedback.created_at.isoformat(),
                "reviewed_at": feedback.reviewed_at.isoformat() if feedback.reviewed_at else None
            })
        
        return training_samples


# Singleton instance for easy import
auto_learning_engine = AutoLearningEngine()
