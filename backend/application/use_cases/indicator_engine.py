"""
Indicator Engine - Rule-based fraud indicator detection for explainability

This module provides a RULE-BASED system for detecting and categorizing
fraud indicators in rental listings. It works alongside the BERT classifier
(bert_fraud_classifier.py) which provides the REAL AI analysis.

This engine provides:
- EXPLAINABILITY: Specific indicators users can understand
- Pattern detection using keyword and regex matching
- Risk level calculation based on indicator severity
- Human-readable risk stories

Each indicator has:
- A unique CODE for programmatic identification
- A SEVERITY level (1-5, where 5 is most severe)
- EVIDENCE extracted from the listing
- CATEGORY for grouping related indicators

NOTE: This is intentionally rule-based, not ML. The real AI is in BERT.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

# Import settings for configurable risk level parameters
from config import get_settings

# NOTE:
# This indicator engine is rule-based, providing explainability
# alongside the BERT classifier.

_nlp_engine = None  # Deprecated - kept for compatibility

def get_nlp_engine():
    """DEPRECATED: NLP Engine removed. Returns None."""
    return None


class IndicatorCategory(str, Enum):
    """Categories of fraud indicators"""
    PAYMENT = "payment"
    URGENCY = "urgency"
    CONTACT = "contact"
    IDENTITY = "identity"
    PRICING = "pricing"
    TEXT_STYLE = "text_style"
    CONTENT = "content"
    ML_ANOMALY = "ml_anomaly"
    NLP_SEMANTIC = "nlp_semantic"  


class RiskLevel(str, Enum):
    """Standardized risk levels"""
    VERY_LOW = "Very Low"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"


@dataclass
class Indicator:
    """
    Enterprise-grade indicator structure.
    
    Attributes:
        code: Unique identifier (e.g., OFF_PLATFORM_PAYMENT)
        category: Category grouping
        severity: 1-5 scale (5 = most severe)
        evidence: Extracted text/data that triggered the indicator
        description: Human-readable explanation
        impact_score: Contribution to overall risk (0.0-1.0)
    """
    code: str
    category: IndicatorCategory
    severity: int  # 1-5
    evidence: List[str]
    description: str
    impact_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to enterprise output format with user-friendly description"""
        # Get user-friendly category name
        category_friendly = {
            IndicatorCategory.PAYMENT: "Payment Concerns",
            IndicatorCategory.URGENCY: "Pressure Tactics",
            IndicatorCategory.CONTACT: "Contact Issues",
            IndicatorCategory.IDENTITY: "Identity Concerns",
            IndicatorCategory.PRICING: "Pricing Issues",
            IndicatorCategory.TEXT_STYLE: "Writing Concerns",
            IndicatorCategory.CONTENT: "Content Issues",
            IndicatorCategory.ML_ANOMALY: "Unusual Patterns",
            IndicatorCategory.NLP_SEMANTIC: "Language Concerns"
        }.get(self.category, self.category.value)
        
        return {
            "code": self.code,
            "category": self.category.value,
            "category_display": category_friendly,
            "severity": self.severity,
            "evidence": self.evidence,
            "description": self.description,
            "description_friendly": IndicatorEngine._get_user_friendly_description(self) if hasattr(IndicatorEngine, '_get_user_friendly_description') else self.description,
            "impact_score": round(self.impact_score, 3)
        }


@dataclass
class IndicatorPattern:
    """Pattern definition for rule-based detection"""
    code: str
    category: IndicatorCategory
    keywords: List[str]
    base_severity: int
    description_template: str
    impact_per_match: float
    max_impact: float
    severity_escalation_threshold: int = 2  # Increase severity after N matches


class IndicatorEngine:
    """
    Enterprise-grade Indicator Engine for fraud detection.
    
    Responsibilities:
    - Pattern-based indicator detection
    - Severity calculation
    - Evidence extraction
    - Confidence scoring
    - Enterprise output formatting
    """
    
    # =========================================================================
    # INDICATOR PATTERN DEFINITIONS
    # Each pattern defines how to detect a specific fraud indicator
    # =========================================================================
    
    PATTERNS: List[IndicatorPattern] = [
        # Payment-related indicators
        IndicatorPattern(
            code="OFF_PLATFORM_PAYMENT",
            category=IndicatorCategory.PAYMENT,
            keywords=['wire transfer', 'western union', 'moneygram'],
            base_severity=5,
            description_template="Off-platform payment method requested: {evidence}",
            impact_per_match=0.20,
            max_impact=0.35
        ),
        IndicatorPattern(
            code="CRYPTO_PAYMENT",
            category=IndicatorCategory.PAYMENT,
            keywords=['bitcoin', 'cryptocurrency', 'crypto', 'btc', 'ethereum'],
            base_severity=5,
            description_template="Cryptocurrency payment requested: {evidence}",
            impact_per_match=0.25,
            max_impact=0.35
        ),
        IndicatorPattern(
            code="GIFT_CARD_PAYMENT",
            category=IndicatorCategory.PAYMENT,
            keywords=['gift card', 'giftcard', 'itunes card', 'google play card'],
            base_severity=5,
            description_template="Gift card payment requested: {evidence}",
            impact_per_match=0.25,
            max_impact=0.35
        ),
        IndicatorPattern(
            code="UPFRONT_PAYMENT",
            category=IndicatorCategory.PAYMENT,
            keywords=['upfront payment', 'deposit now', 'pay before viewing', 
                     'pay first', 'advance payment', 'pay in advance'],
            base_severity=4,
            description_template="Upfront payment demanded before viewing: {evidence}",
            impact_per_match=0.18,
            max_impact=0.30
        ),
        IndicatorPattern(
            code="CASH_ONLY",
            category=IndicatorCategory.PAYMENT,
            keywords=['cash only', 'cash payment only'],
            base_severity=3,
            description_template="Cash-only payment required: {evidence}",
            impact_per_match=0.12,
            max_impact=0.20
        ),
        IndicatorPattern(
            code="DIRECT_MONEY_REQUEST",
            category=IndicatorCategory.PAYMENT,
            keywords=['send money', 'send me money', 'send usd', 'send cad',
                     'send cash', 'send funds', 'send payment', 'send the money',
                     'transfer money', 'transfer funds', 'transfer usd', 'transfer cad',
                     'send me usd', 'send me cad', 'send me cash', 'send me the money',
                     'pay me', 'pay upfront', 'money order', 'e-transfer',
                     'etransfer', 'interac', 'zelle', 'venmo', 'cashapp',
                     'cash app', 'paypal friends', 'sent abroad', 'send abroad'],
            base_severity=5,
            description_template="Direct money transfer requested: {evidence}",
            impact_per_match=0.25,
            max_impact=0.40
        ),
        
        # Urgency-related indicators
        IndicatorPattern(
            code="HIGH_URGENCY",
            category=IndicatorCategory.URGENCY,
            keywords=['urgent', 'hurry', 'immediately', 'asap', 'right now',
                     'act fast', 'don\'t miss', 'last chance', 'limited time',
                     'act now', 'today only', 'won\'t last'],
            base_severity=3,
            description_template="High-pressure urgency tactics detected: {evidence}",
            impact_per_match=0.10,
            max_impact=0.25,
            severity_escalation_threshold=3
        ),
        
        # Contact-related indicators
        IndicatorPattern(
            code="LIMITED_CONTACT",
            category=IndicatorCategory.CONTACT,
            keywords=['email only', 'no phone', 'no calls', 'text only',
                     'whatsapp only', 'telegram only'],
            base_severity=3,
            description_template="Limited/unusual contact methods: {evidence}",
            impact_per_match=0.10,
            max_impact=0.20
        ),
        IndicatorPattern(
            code="ANONYMOUS_CONTACT",
            category=IndicatorCategory.CONTACT,
            keywords=['burner phone', 'temporary number', 'anonymous'],
            base_severity=4,
            description_template="Anonymous/untraceable contact method: {evidence}",
            impact_per_match=0.15,
            max_impact=0.25
        ),
        
        # Identity-related indicators
        IndicatorPattern(
            code="OWNER_UNAVAILABLE",
            category=IndicatorCategory.IDENTITY,
            keywords=['owner overseas', 'out of country', 'abroad', 
                     'military deployment', 'missionary', 'working overseas',
                     'relocated', 'moved away'],
            base_severity=4,
            description_template="Owner claims to be unavailable/overseas: {evidence}",
            impact_per_match=0.18,
            max_impact=0.30
        ),
        IndicatorPattern(
            code="NO_VIEWING",
            category=IndicatorCategory.IDENTITY,
            keywords=['no viewing', 'can\'t meet', 'cannot meet', 'no visits',
                     'keys will be mailed', 'send keys', 'mail keys'],
            base_severity=5,
            description_template="No in-person viewing offered: {evidence}",
            impact_per_match=0.22,
            max_impact=0.35
        ),
        
        # Content-related indicators
        IndicatorPattern(
            code="TOO_GOOD_TRUE",
            category=IndicatorCategory.CONTENT,
            keywords=['too good to be true', 'amazing deal', 'unbelievable price',
                     'best deal', 'steal', 'once in lifetime'],
            base_severity=3,
            description_template="Suspicious 'too good to be true' language: {evidence}",
            impact_per_match=0.12,
            max_impact=0.20
        ),
        IndicatorPattern(
            code="PERSONAL_STORY",
            category=IndicatorCategory.CONTENT,
            keywords=['divorce', 'death in family', 'sick relative', 'emergency',
                     'desperate', 'must sell', 'need money'],
            base_severity=3,
            description_template="Emotional manipulation through personal story: {evidence}",
            impact_per_match=0.10,
            max_impact=0.20
        ),
    ]
    
    def __init__(self):
        """Initialize the indicator engine"""
        self._pattern_map = {p.code: p for p in self.PATTERNS}
    
    def analyze(
        self,
        text: str,
        price: Optional[float] = None,
        location: Optional[str] = None,
        ml_score: Optional[float] = None,
        use_nlp: bool = True
    ) -> Tuple[List[Indicator], float, float]:
        """
        Enterprise fraud analysis pipeline.
        
        Professional pipeline flow:
        signals → calibrated_score → confidence → indicator_pressure → risk_level → story
        
        This method returns the components needed for risk_level and story generation.
        
        Args:
            text: Listing text to analyze
            price: Optional listing price
            location: Optional location string
            ml_score: Optional ML anomaly score (0-1, higher = more anomalous)
            use_nlp: Whether to use NLP semantic analysis (default True)
        
        Returns:
            Tuple of (indicators, calibrated_risk_score, confidence)
        """
        # =====================================================================
        # STAGE 1: SIGNAL COLLECTION
        # Gather independent signals from multiple detection sources
        # =====================================================================
        signals: List[Indicator] = []
        text_lower = text.lower()
        
        # 1a. Rule-based pattern detection
        for pattern in self.PATTERNS:
            indicator = self._check_pattern(text_lower, pattern)
            if indicator:
                signals.append(indicator)
        
        # 1a2. Regex-based monetary request detection
        # Catches "send me 1000 usd", "$5000 deposit", "10000 dollars sent" etc.
        monetary_signals = self._detect_monetary_requests(text_lower)
        signals.extend(monetary_signals)
        
        # 1b. Text style analysis signals
        style_signals = self._analyze_text_style(text)
        signals.extend(style_signals)
        
        # 1c. Price anomaly signals
        if price is not None:
            price_signal = self._analyze_price(price, location)
            if price_signal:
                signals.append(price_signal)
        
        # 1d. ML anomaly signals
        if ml_score is not None and ml_score > 0.5:
            ml_signal = self._create_ml_indicator(ml_score)
            signals.append(ml_signal)
        
        # 1e. NLP semantic signals
        nlp_signals = []
        if use_nlp:
            nlp_signals = self._analyze_with_nlp(text)
            signals.extend(nlp_signals)
        
        # =====================================================================
        # STAGE 2: SCORE CALIBRATION
        # Calibrate raw scores based on signal weights and interactions
        # =====================================================================
        calibrated_score = self._calibrate_score(signals, ml_score)
        
        # =====================================================================
        # STAGE 3: CONFIDENCE CALCULATION
        # Measure agreement between independent detection sources
        # =====================================================================
        confidence = self._calculate_confidence(signals, ml_score, use_nlp)
        
        # =====================================================================
        # STAGE 4: INDICATOR PRESSURE (computed in get_risk_level)
        # Cumulative severity pressure from all signals
        # This is factored into risk_level determination
        # =====================================================================
        # Note: indicator_pressure is calculated within get_risk_level()
        # as part of the severity_contribution calculation
        
        return signals, calibrated_score, confidence
    
    def _calibrate_score(
        self,
        indicators: List[Indicator],
        ml_score: Optional[float] = None
    ) -> float:
        """
        Calibrate raw risk score based on signal interactions.
        
        Calibration considers:
        - Impact scores from each indicator (adjusted by auto-learning weights)
        - Cross-category signal reinforcement
        - ML vs rule agreement/disagreement
        - Diminishing returns for redundant signals
        """
        if not indicators:
            return 0.0
        
        # Apply auto-learning calibrated weights if available
        try:
            from application.use_cases.auto_learning_engine import AutoLearningEngine
            learning_engine = AutoLearningEngine()
            for ind in indicators:
                calibrated = learning_engine.get_calibrated_weight(ind.code, ind.impact_score)
                ind.impact_score = calibrated
        except Exception:
            pass  # Auto-learning not available, use default weights
        
        # Base score from (potentially calibrated) impact scores
        base_score = sum(ind.impact_score for ind in indicators)
        
        # =====================================================================
        # CROSS-CATEGORY REINFORCEMENT
        # Multiple categories detecting risk = higher calibrated score
        # =====================================================================
        categories = set(ind.category for ind in indicators)
        category_count = len(categories)
        
        if category_count >= 4:
            base_score *= 1.15  # 15% boost for 4+ categories
        elif category_count >= 3:
            base_score *= 1.10  # 10% boost for 3 categories
        elif category_count >= 2:
            base_score *= 1.05  # 5% boost for 2 categories
        
        # =====================================================================
        # HIGH SEVERITY COMPOUNDING
        # Multiple critical indicators compound the risk
        # =====================================================================
        critical_count = sum(1 for ind in indicators if ind.severity >= 4)
        high_severity_count = sum(1 for ind in indicators if ind.severity >= 3)
        
        if critical_count >= 3:
            base_score *= 1.20  # 20% boost for 3+ critical
        elif critical_count >= 2:
            base_score *= 1.12  # 12% boost for 2 critical
        elif critical_count >= 1 and high_severity_count >= 3:
            base_score *= 1.08  # 8% boost for 1 critical + multiple high
        
        # =====================================================================
        # ML ALIGNMENT CALIBRATION
        # Adjust score based on ML agreement/disagreement
        # =====================================================================
        if ml_score is not None:
            rule_score = sum(
                ind.impact_score for ind in indicators 
                if ind.category not in [IndicatorCategory.ML_ANOMALY]
            )
            
            ml_indicates_risk = ml_score > 0.6
            rules_indicate_risk = rule_score > 0.25
            
            if ml_indicates_risk == rules_indicate_risk:
                # Agreement - boost confidence in the score
                base_score *= 1.05
            elif ml_indicates_risk and not rules_indicate_risk:
                # ML sees something rules don't - slight boost
                base_score *= 1.02
            elif not ml_indicates_risk and rules_indicate_risk:
                # Rules see something ML doesn't - slight dampening
                base_score *= 0.98
        
        # =====================================================================
        # DIMINISHING RETURNS FOR SAME-CATEGORY SIGNALS
        # Many indicators in same category = reduced marginal impact
        # =====================================================================
        for category in categories:
            category_indicators = [ind for ind in indicators if ind.category == category]
            if len(category_indicators) > 3:
                # Apply diminishing returns for 4+ same-category signals
                excess = len(category_indicators) - 3
                diminishing_factor = 1.0 - (excess * 0.02)  # 2% reduction per excess
                base_score *= max(diminishing_factor, 0.90)  # Cap at 10% reduction
        
        return min(base_score, 1.0)
    
    def _analyze_with_nlp(self, text: str) -> List[Indicator]:
        """
        Analyze text using NLP engine for semantic similarity detection.
        
        Uses concept-based matching and zero-shot classification, which is
        more flexible than exact template matching.
        """
        indicators = []
        nlp_engine = get_nlp_engine()
        
        if nlp_engine is None:
            return indicators  # NLP not available, skip gracefully
        
        try:
            result = nlp_engine.analyze(text)
            
            if result is None:
                return indicators
            
            # 1. Concept-based matches (semantic similarity to abstract fraud concepts)
            if result.matched_concepts:
                for concept in result.matched_concepts[:3]:  # Top 3 matches
                    if concept["score"] > 0.4:
                        severity = concept.get("severity", 3)
                        if concept["score"] > 0.6:
                            severity = min(5, severity + 1)
                        
                        impact = min(concept["score"] * 0.18, 0.20)
                        
                        indicators.append(Indicator(
                            code=f"NLP_{concept['key'].upper()}",
                            category=IndicatorCategory.NLP_SEMANTIC,
                            severity=severity,
                            evidence=[
                                f"Concept: {concept['concept'][:80]}...",
                                f"Semantic match: {concept['score']:.1%}"
                            ],
                            description=f"Text semantically matches '{concept['key'].replace('_', ' ')}' fraud pattern ({concept['score']:.1%})",
                            impact_score=impact
                        ))
            
            # 2. Zero-shot classification results
            if result.classification_result:
                scam_score = result.classification_result.get("potential scam or fraud", 0)
                if scam_score > 0.6:
                    severity = 3 if scam_score < 0.8 else 4
                    
                    indicators.append(Indicator(
                        code="NLP_CLASSIFICATION_SCAM",
                        category=IndicatorCategory.NLP_SEMANTIC,
                        severity=severity,
                        evidence=[f"AI classification: {scam_score:.1%} scam probability"],
                        description=f"AI classifier indicates {scam_score:.1%} probability of scam",
                        impact_score=min(scam_score * 0.15, 0.15)
                    ))
            
            # 3. Fuzzy keyword matches (catches typos like "w1re", "paypai")
            if result.fuzzy_matches:
                severity = 3 if len(result.fuzzy_matches) == 1 else 4
                impact = min(len(result.fuzzy_matches) * 0.08, 0.18)
                
                evidence = [f"'{match}'" for match in result.fuzzy_matches[:5]]
                
                indicators.append(Indicator(
                    code="FUZZY_KEYWORD_DETECTED",
                    category=IndicatorCategory.NLP_SEMANTIC,
                    severity=severity,
                    evidence=evidence,
                    description=f"Possible obfuscated scam keywords: {', '.join(result.fuzzy_matches[:3])}",
                    impact_score=impact
                ))
            
            # 4. Manipulation score (emotional manipulation patterns)
            if result.manipulation_score > 0.4:
                severity = 2 if result.manipulation_score < 0.6 else 3
                
                indicators.append(Indicator(
                    code="MANIPULATIVE_LANGUAGE",
                    category=IndicatorCategory.NLP_SEMANTIC,
                    severity=severity,
                    evidence=[f"Manipulation score: {result.manipulation_score:.1%}"],
                    description=f"Text shows emotional manipulation patterns ({result.manipulation_score:.1%})",
                    impact_score=min(result.manipulation_score * 0.12, 0.12)
                ))
        
        except Exception as e:
            # Log but don't fail - NLP is supplementary
            import logging
            logging.warning(f"NLP analysis failed: {e}")
        
        return indicators
    
    def _check_pattern(self, text_lower: str, pattern: IndicatorPattern) -> Optional[Indicator]:
        """Check if pattern matches and create indicator"""
        matches = []
        for keyword in pattern.keywords:
            if keyword in text_lower:
                # Extract evidence (keyword + surrounding context)
                evidence = self._extract_evidence(text_lower, keyword)
                matches.append(evidence)
        
        if not matches:
            return None
        
        # Calculate severity (may escalate based on match count)
        severity = pattern.base_severity
        if len(matches) >= pattern.severity_escalation_threshold:
            severity = min(5, severity + 1)
        
        # Calculate impact
        impact = min(len(matches) * pattern.impact_per_match, pattern.max_impact)
        
        # Format description
        evidence_str = ", ".join(matches[:3])  # Limit to 3 examples
        description = pattern.description_template.format(evidence=evidence_str)
        
        return Indicator(
            code=pattern.code,
            category=pattern.category,
            severity=severity,
            evidence=matches[:5],  # Limit stored evidence
            description=description,
            impact_score=impact
        )
    
    def _detect_monetary_requests(self, text_lower: str) -> List[Indicator]:
        """
        Detect suspicious monetary requests using regex patterns.
        
        Catches phrases like:
        - "send me 1000 usd"
        - "$5000 sent abroad" 
        - "transfer 10000 dollars"
        - "pay 500 before"
        """
        indicators = []
        
        # Patterns for suspicious monetary requests in listing text
        monetary_patterns = [
            # "send/transfer [me] <amount> [usd/cad/dollars/cash]"
            (r'(?:send|transfer|wire|forward|deposit)\s+(?:me\s+)?(?:\$?\s*\d{2,}[\d,]*)\s*(?:usd|cad|dollars?|bucks|cash)?', 
             "Direct money send/transfer request with specific amount"),
            # "$<amount> sent/transfer/send"
            (r'\$\s*\d{2,}[\d,]*\s*(?:usd|cad|dollars?)?\s*(?:sent|transfer|send|wire|deposit)',
             "Money amount with transfer instruction"),
            # "<amount> usd/cad/dollars sent/send/transfer"
            (r'\d{2,}[\d,]*\s*(?:usd|cad|dollars?|bucks)\s*(?:sent|send|transfer|wire|deposit|forward)',
             "Currency amount with transfer instruction"),
            # "pay me <amount>"
            (r'pay\s+me\s+(?:\$?\s*\d{2,}[\d,]*)',
             "Direct payment demand with specific amount"),
            # "send <amount> to/via/through"
            (r'send\s+(?:\$?\s*\d{2,}[\d,]*)\s*(?:usd|cad|dollars?)?\s*(?:to|via|through|using)',
             "Money transfer to specific destination"),
        ]
        
        for pattern_regex, description in monetary_patterns:
            matches_found = re.findall(pattern_regex, text_lower)
            if matches_found:
                # Already caught by DIRECT_MONEY_REQUEST keyword pattern? 
                # This catches the specific amount variants
                evidence_list = []
                for match in matches_found[:3]:
                    evidence = self._extract_evidence(text_lower, match)
                    evidence_list.append(evidence)
                
                indicators.append(Indicator(
                    code="MONETARY_TRANSFER_REQUEST",
                    category=IndicatorCategory.PAYMENT,
                    severity=5,
                    evidence=evidence_list,
                    description=f"Suspicious monetary request detected: {description}",
                    impact_score=0.30
                ))
                break  # One match is enough
        
        return indicators
    
    def _extract_evidence(self, text: str, keyword: str, context_chars: int = 30) -> str:
        """Extract keyword with surrounding context as evidence"""
        idx = text.find(keyword)
        if idx == -1:
            return keyword
        
        start = max(0, idx - context_chars)
        end = min(len(text), idx + len(keyword) + context_chars)
        
        evidence = text[start:end].strip()
        if start > 0:
            evidence = "..." + evidence
        if end < len(text):
            evidence = evidence + "..."
        
        return evidence
    
    def _analyze_text_style(self, text: str) -> List[Indicator]:
        """Analyze text style for suspicious patterns"""
        indicators = []
        
        # Excessive uppercase
        if len(text) > 50:
            uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if uppercase_ratio > 0.3:
                indicators.append(Indicator(
                    code="EXCESSIVE_CAPS",
                    category=IndicatorCategory.TEXT_STYLE,
                    severity=2,
                    evidence=[f"{uppercase_ratio*100:.1f}% uppercase characters"],
                    description=f"Excessive use of capital letters ({uppercase_ratio*100:.1f}%)",
                    impact_score=0.08
                ))
        
        # Excessive exclamation marks
        exclamation_count = text.count('!')
        if exclamation_count > 5:
            indicators.append(Indicator(
                code="EXCESSIVE_PUNCTUATION",
                category=IndicatorCategory.TEXT_STYLE,
                severity=2,
                evidence=[f"{exclamation_count} exclamation marks"],
                description=f"Excessive exclamation marks ({exclamation_count}) suggesting urgency",
                impact_score=min(exclamation_count * 0.02, 0.10)
            ))
        
        # Very short listing (potential low-effort scam)
        word_count = len(text.split())
        if word_count < 20:
            indicators.append(Indicator(
                code="MINIMAL_DESCRIPTION",
                category=IndicatorCategory.TEXT_STYLE,
                severity=2,
                evidence=[f"Only {word_count} words"],
                description=f"Very short listing description ({word_count} words)",
                impact_score=0.05
            ))
        
        return indicators
    
    def _analyze_price(self, price: float, location: Optional[str] = None) -> Optional[Indicator]:
        """Analyze price for suspicious patterns with location awareness.
        
        Thresholds derived from Rentals.ca National Rent Report (Jan 2026 data):
          - Suspicious: ~40% of avg market rent (almost certainly a scam)
          - Below-market: ~60% of avg market rent (unusually cheap, warrants review)
        Source: https://rentals.ca/national-rent-report (Feb 2026)
        """
        # Default thresholds for rural/unknown areas (est. avg ~$1,200)
        suspicious_threshold = 500
        below_market_threshold = 750
        
        if location:
            loc_lower = location.lower()
            # Tier 1: Most expensive markets — avg rent $2,500-$2,650
            # Vancouver $2,650 | Toronto $2,504 | North Vancouver $2,958
            if any(city in loc_lower for city in ['toronto', 'vancouver', 'north vancouver']):
                suspicious_threshold = 1000
                below_market_threshold = 1500
            # Tier 2: Expensive markets — avg rent $2,100-$2,450
            # Mississauga $2,446 | Halifax $2,270 | Victoria $2,224 | Ottawa $2,107
            # Oakville $2,502 | Kingston $2,315 | Burlington $2,376 | Burnaby $2,505
            elif any(city in loc_lower for city in [
                'victoria', 'ottawa', 'halifax', 'mississauga',
                'oakville', 'kingston', 'burlington', 'burnaby',
                'brampton', 'scarborough', 'north york'
            ]):
                suspicious_threshold = 900
                below_market_threshold = 1400
            # Tier 3: Mid-range markets — avg rent $1,650-$1,900
            # Montreal $1,913 | Calgary $1,815 | Winnipeg $1,648
            # Hamilton $1,800~ | Kitchener $1,700~ | London $1,650~
            elif any(city in loc_lower for city in [
                'montreal', 'calgary', 'winnipeg', 'hamilton',
                'kitchener', 'london', 'waterloo', 'guelph'
            ]):
                suspicious_threshold = 750
                below_market_threshold = 1100
            # Tier 4: Affordable markets — avg rent $1,370-$1,490
            # Edmonton $1,488 | Quebec City $1,489 | Regina $1,374 | Saskatoon $1,371
            elif any(city in loc_lower for city in [
                'edmonton', 'saskatoon', 'regina', 'quebec'
            ]):
                suspicious_threshold = 600
                below_market_threshold = 900
        
        if price < suspicious_threshold:
            return Indicator(
                code="SUSPICIOUS_LOW_PRICE",
                category=IndicatorCategory.PRICING,
                severity=4,
                evidence=[f"${price}/month (threshold: ${suspicious_threshold} for {location or 'default'})"],
                description=f"Suspiciously low price (${price}/month) for rental in {location or 'this area'}",
                impact_score=0.20
            )
        elif price < below_market_threshold:
            return Indicator(
                code="BELOW_MARKET_PRICE",
                category=IndicatorCategory.PRICING,
                severity=3,
                evidence=[f"${price}/month (threshold: ${below_market_threshold} for {location or 'default'})"],
                description=f"Below-market price (${price}/month) for {location or 'this area'} may indicate scam",
                impact_score=0.12
            )
        
        return None
    
    def _create_ml_indicator(self, ml_score: float) -> Indicator:
        """Create indicator from ML anomaly score"""
        severity = 2 if ml_score < 0.6 else 3 if ml_score < 0.75 else 4
        
        return Indicator(
            code="ML_ANOMALY_DETECTED",
            category=IndicatorCategory.ML_ANOMALY,
            severity=severity,
            evidence=[f"Anomaly score: {ml_score:.2f}"],
            description=f"Machine learning model detected unusual patterns (score: {ml_score:.2f})",
            impact_score=ml_score * 0.20  # ML contributes up to 20%
        )
    
    def _calculate_risk_score(self, indicators: List[Indicator]) -> float:
        """
        Calculate overall risk score from indicators.
        
        The score is the sum of impact scores, capped at 1.0,
        with a bonus for multiple high-severity indicators.
        """
        if not indicators:
            return 0.0
        
        # Sum of impact scores
        base_score = sum(ind.impact_score for ind in indicators)
        
        # Bonus for multiple high-severity indicators (compounding risk)
        high_severity_count = sum(1 for ind in indicators if ind.severity >= 4)
        if high_severity_count >= 2:
            base_score *= 1.1  # 10% bonus
        if high_severity_count >= 3:
            base_score *= 1.1  # Additional 10% bonus
        
        return min(base_score, 1.0)
    
    def _calculate_confidence(
        self, 
        indicators: List[Indicator],
        ml_score: Optional[float] = None,
        nlp_used: bool = False
    ) -> float:
        """
        Calculate confidence based on indicator agreement.
        
        Higher confidence when:
        - Multiple indicators from different categories agree
        - ML score aligns with rule-based indicators
        - NLP agrees with rule-based indicators
        - Clear evidence is present
        """
        if not indicators:
            return 0.70  # Base confidence for clean listings
        
        # Base confidence
        confidence = 0.65
        
        # Boost for multiple categories agreeing
        categories = set(ind.category for ind in indicators)
        if len(categories) >= 2:
            confidence += 0.05
        if len(categories) >= 3:
            confidence += 0.05
        if len(categories) >= 4:
            confidence += 0.03
        
        # Boost for high-severity indicators with clear evidence
        high_severity_with_evidence = sum(
            1 for ind in indicators 
            if ind.severity >= 4 and len(ind.evidence) > 0
        )
        confidence += min(high_severity_with_evidence * 0.03, 0.10)
        
        # Boost if ML agrees with rules (both indicate risk or both don't)
        if ml_score is not None:
            rule_based_risk = sum(ind.impact_score for ind in indicators if ind.category != IndicatorCategory.ML_ANOMALY)
            ml_indicates_risk = ml_score > 0.5
            rules_indicate_risk = rule_based_risk > 0.3
            
            if ml_indicates_risk == rules_indicate_risk:
                confidence += 0.08  # Agreement bonus
            else:
                confidence -= 0.05  # Disagreement penalty
        
        # Boost if NLP agrees with rule-based detection
        if nlp_used:
            nlp_indicators = [i for i in indicators if i.category == IndicatorCategory.NLP_SEMANTIC]
            rule_indicators = [i for i in indicators if i.category not in [IndicatorCategory.ML_ANOMALY, IndicatorCategory.NLP_SEMANTIC]]
            
            nlp_indicates_risk = any(i.severity >= 3 for i in nlp_indicators)
            rules_indicate_risk = any(i.severity >= 3 for i in rule_indicators)
            
            if nlp_indicators and rule_indicators:
                if nlp_indicates_risk == rules_indicate_risk:
                    confidence += 0.06  # NLP agreement bonus
                elif nlp_indicates_risk and not rules_indicate_risk:
                    # NLP found something rules missed - moderate boost
                    confidence += 0.03
        
        return min(max(confidence, 0.50), 0.95)  # Clamp between 0.50 and 0.95
    
    @staticmethod
    def _get_user_friendly_description(indicator: 'Indicator') -> str:
        """
        Convert technical indicator descriptions to user-friendly language.
        
        Removes jargon like 'ML', 'NLP', 'anomaly', 'semantic', etc.
        and provides plain English explanations that any user can understand.
        """
        # Map technical codes to user-friendly descriptions
        user_friendly_codes = {
            # ML/AI related - explain what it means, not how it works
            "ML_ANOMALY_DETECTED": "Our system detected unusual patterns in this listing that don't match typical legitimate rentals",
            "ML_HIGH_FRAUD_PROBABILITY": "This listing has characteristics commonly found in fraudulent posts",
            "ML_PATTERN_MISMATCH": "The listing details don't fit the normal pattern for genuine rentals in this area",
            
            # NLP/Semantic related - focus on what was found
            "NLP_FRAUD_LANGUAGE": "The listing uses language commonly found in scam posts",
            "NLP_MANIPULATION_DETECTED": "The wording appears designed to pressure or manipulate readers",
            "NLP_SCAM_SIMILARITY": "The text closely matches known scam listing templates",
            "NLP_URGENCY_LANGUAGE": "The listing uses pushy language to rush your decision",
            "NLP_SUSPICIOUS_TONE": "The tone of the listing raises concerns",
            
            # Payment related
            "WIRE_TRANSFER_REQUEST": "Requests wire transfer payment - a common scam tactic",
            "CRYPTO_PAYMENT": "Asks for cryptocurrency payment - legitimate landlords rarely do this",
            "GIFT_CARD_PAYMENT": "Mentions gift cards as payment - this is almost always a scam",
            "UPFRONT_PAYMENT": "Demands large upfront payment before viewing the property",
            "PAYMENT_BEFORE_VIEWING": "Wants payment before you can see the property",
            "UNUSUAL_PAYMENT": "Requests unusual or untraceable payment methods",
            
            # Urgency related  
            "URGENCY_PRESSURE": "Uses high-pressure tactics to rush your decision",
            "LIMITED_TIME_OFFER": "Creates artificial urgency with 'limited time' claims",
            "ACT_NOW": "Pressures you to 'act now' or lose the opportunity",
            "MULTIPLE_APPLICANTS": "Claims many other people are interested to pressure you",
            
            # Pricing related
            "PRICE_TOO_LOW": "The price is suspiciously low for this type of property and location",
            "UNREALISTIC_DEAL": "The deal seems too good to be true",
            "PRICE_ANOMALY": "The pricing doesn't match similar properties in the area",
            
            # Contact related
            "NO_PHONE": "No phone number provided - makes it hard to verify the landlord",
            "FOREIGN_CONTACT": "Contact information suggests the person may not be local",
            "GENERIC_EMAIL": "Uses a generic email that's hard to trace",
            "CONTACT_MISMATCH": "Contact details don't match the listed property",
            
            # Identity related
            "ABSENT_LANDLORD": "Landlord claims to be away and can't show the property in person",
            "OVERSEAS_OWNER": "Owner claims to be overseas - a very common scam excuse",
            "PROXY_RENTAL": "Someone else is handling the rental on behalf of the owner",
            "UNVERIFIABLE_OWNER": "Unable to verify who actually owns the property",
            
            # Content related
            "STOCK_PHOTOS": "Images may be stock photos rather than actual property photos",
            "COPIED_TEXT": "The listing text appears to be copied from elsewhere",
            "INCONSISTENT_DETAILS": "Details in the listing don't add up",
            "MISSING_ADDRESS": "No specific address provided",
            "VAGUE_DESCRIPTION": "The description is vague and lacks specific details"
        }
        
        # Check if we have a user-friendly version
        code = indicator.code
        if code in user_friendly_codes:
            return user_friendly_codes[code]
        
        # For unknown codes, clean up the description
        desc = indicator.description
        
        # Remove common technical prefixes/terms
        technical_terms = [
            ("Machine learning model detected ", "Our analysis found "),
            ("ML model ", "Our analysis "),
            ("NLP analysis ", "Language analysis "),
            ("Semantic analysis ", "Text analysis "),
            ("AI detected ", "We detected "),
            ("Algorithm detected ", "We found "),
            ("anomaly", "unusual pattern"),
            ("anomalies", "unusual patterns"),
            (" ML ", " "),
            (" NLP ", " "),
            (" AI ", " "),
            ("semantic", "language"),
            ("neural network", "analysis system"),
            ("classifier", "analysis"),
            ("model score", "risk score"),
            ("confidence score", "certainty level"),
        ]
        
        for old, new in technical_terms:
            desc = desc.replace(old, new)
        
        return desc
    
    @staticmethod
    def get_risk_level(
        risk_score: float,
        confidence: float = None,
        indicators: List[Indicator] = None
    ) -> RiskLevel:
        """
        Determine risk level dynamically based on multiple factors.
        
        We do not rely on static thresholds. Risk levels are determined using:
        1. risk_score: Base numeric score (0.0-1.0)
        2. confidence: Model/indicator agreement confidence (0.0-1.0)
        3. cumulative severity: Sum of independent fraud indicator severities
        
        The final risk level is computed by weighting these factors:
        - Base score contribution (40%)
        - Confidence-adjusted score (30%)
        - Cumulative severity contribution (30%)
        
        Args:
            risk_score: Base risk score (0.0-1.0)
            confidence: Model confidence (0.0-1.0), defaults to 0.7
            indicators: List of detected indicators for severity analysis
        
        Returns:
            RiskLevel enum value
        """
        # Handle defaults
        if confidence is None:
            confidence = 0.70
        if indicators is None:
            indicators = []
        
        # =====================================================================
        # 1. BASE SCORE CONTRIBUTION (40%)
        # =====================================================================
        base_contribution = risk_score * 0.40
        
        # =====================================================================
        # 2. CONFIDENCE-ADJUSTED CONTRIBUTION (30%)
        # High confidence amplifies the score direction
        # Low confidence pulls toward medium
        # =====================================================================
        # Confidence factor: high confidence (>0.8) amplifies, low (<0.6) dampens
        confidence_factor = (confidence - 0.5) * 2  # Maps 0.5-1.0 to 0.0-1.0
        confidence_factor = max(0, min(confidence_factor, 1.0))
        
        # If high risk + high confidence = amplify risk
        # If low risk + high confidence = amplify safety
        # If low confidence = pull toward middle (0.5)
        if confidence >= 0.7:
            # High confidence: amplify the direction
            confidence_adjusted = risk_score * (1 + (confidence - 0.7) * 0.5)
        else:
            # Low confidence: dampen toward middle
            dampening = (0.7 - confidence) * 0.8
            confidence_adjusted = risk_score * (1 - dampening) + 0.4 * dampening
        
        confidence_contribution = min(confidence_adjusted, 1.0) * 0.30
        
        # =====================================================================
        # 3. CUMULATIVE SEVERITY CONTRIBUTION (30%)
        # Based on sum of independent indicator severities
        # =====================================================================
        if indicators:
            # Calculate cumulative severity
            total_severity = sum(ind.severity for ind in indicators)
            
            # Count critical indicators (severity >= 4)
            critical_count = sum(1 for ind in indicators if ind.severity >= 4)
            
            # Count high severity indicators (severity >= 3)
            high_severity_count = sum(1 for ind in indicators if ind.severity >= 3)
            
            # Normalize cumulative severity (cap at 15 for full contribution)
            # Single critical indicator (severity 5) = 0.33 contribution
            # Two critical indicators = 0.67 contribution
            # Three+ critical indicators or many moderate = full contribution
            severity_normalized = min(total_severity / 15.0, 1.0)
            
            # Boost for multiple critical indicators
            if critical_count >= 2:
                severity_normalized = min(severity_normalized * 1.3, 1.0)
            elif critical_count >= 1 and high_severity_count >= 3:
                severity_normalized = min(severity_normalized * 1.2, 1.0)
            severity_contribution = severity_normalized * 0.30
        else:
            # No indicators = low severity contribution
            severity_contribution = 0.05  # Small base for no indicators
            severity_normalized = 0.0
        
        # =====================================================================
        # COMPOSITE SCORE CALCULATION
        # =====================================================================
        composite_score = base_contribution + confidence_contribution + severity_contribution
        
        # Ensure within bounds
        composite_score = max(0.0, min(composite_score, 1.0))
        
        # =====================================================================
        # RISK LEVEL DETERMINATION
        # Dynamic boundaries based on composite score
        # =====================================================================
        
        # Critical override: Any single severity-5 indicator with high confidence
        if indicators:
            has_critical = any(ind.severity == 5 for ind in indicators)
            if has_critical and confidence >= 0.75 and risk_score >= 0.5:
                return RiskLevel.VERY_HIGH

        # Load configurable parameters from settings
        settings = get_settings()
        
        # Adaptive thresholds (avoid static hard-coded boundaries)
        # Start from conservative base thresholds and shift them based on
        # indicator pressure (severity_normalized) and confidence.
        base_thresholds = settings.RISK_BASE_THRESHOLDS

        # severity_normalized in [0,1], confidence in [0,1]
        # Positive shift (reduce threshold) when severity and confidence are high
        # so the system moves to stricter detection.
        shift = (
            (severity_normalized - settings.RISK_SEVERITY_BASELINE) * settings.RISK_SEVERITY_SHIFT_COEFFICIENT +
            (confidence - settings.RISK_CONFIDENCE_BASELINE) * settings.RISK_CONFIDENCE_SHIFT_COEFFICIENT
        )
        # Clamp shift to reasonable bounds
        max_shift = settings.RISK_MAX_THRESHOLD_SHIFT
        shift = max(-max_shift, min(max_shift, shift))

        thresholds = [
            max(0.02, min(0.98, t - shift)) for t in base_thresholds
        ]

        # Map composite_score into RiskLevel using adaptive thresholds
        if composite_score < thresholds[0]:
            return RiskLevel.VERY_LOW
        elif composite_score < thresholds[1]:
            return RiskLevel.LOW
        elif composite_score < thresholds[2]:
            return RiskLevel.MEDIUM
        elif composite_score < thresholds[3]:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH
    
    @staticmethod
    def generate_risk_story(
        risk_score: float,
        indicators: List[Indicator],
        confidence: float = None
    ) -> str:
        """
        Generate dynamic human-readable risk narrative.
        
        The story is generated based on:
        - Indicator pressure (cumulative severity)
        - Categories of detected issues
        - Specific evidence found
        - Confidence level
        
        This is NOT static text - it's dynamically composed from signals.
        """
        # Calculate indicator pressure for narrative calibration
        if indicators:
            total_severity = sum(ind.severity for ind in indicators)
            critical_count = sum(1 for ind in indicators if ind.severity >= 4)
            high_count = sum(1 for ind in indicators if ind.severity >= 3)
            categories_detected = set(ind.category for ind in indicators)
        else:
            total_severity = 0
            critical_count = 0
            high_count = 0
            categories_detected = set()
        
        # Dynamic opening based on indicator pressure, not just risk level
        if critical_count >= 2 or total_severity >= 15:
            opening = "🚨 ALERT: Multiple critical fraud indicators detected in this listing."
        elif critical_count >= 1 or total_severity >= 10:
            opening = "🚨 This listing exhibits significant fraud indicators requiring immediate caution."
        elif high_count >= 2 or total_severity >= 6:
            opening = "⚠️ This listing shows concerning patterns that warrant careful verification."
        elif total_severity >= 3:
            opening = "⚠️ Minor risk indicators detected. Exercise standard caution."
        elif total_severity > 0:
            opening = "✅ This listing shows minimal concerns with minor indicators noted."
        else:
            opening = "✅ This listing appears legitimate with no significant red flags detected."
        
        story = opening
        
        # Add confidence context if low
        if confidence is not None and confidence < 0.65:
            story += f" (Analysis confidence: {confidence:.0%} - results may require additional verification)"
        
        if not indicators:
            story += " Our analysis found no major concerns across payment methods, pricing, urgency tactics, or language patterns. However, always exercise caution and verify listing details independently."
            return story

        # Compute a local risk level to tailor final recommendations and tone
        local_risk_level = IndicatorEngine.get_risk_level(risk_score, confidence=confidence, indicators=indicators)
        
        # Group by severity
        critical = [i for i in indicators if i.severity >= 4]
        moderate = [i for i in indicators if 2 <= i.severity < 4]
        low_severity = [i for i in indicators if i.severity < 2]
        
        # User-friendly category descriptions (no technical jargon)
        category_descriptions = {
            IndicatorCategory.PAYMENT: "suspicious payment requests",
            IndicatorCategory.URGENCY: "pressure tactics to rush your decision",
            IndicatorCategory.CONTACT: "unusual contact information",
            IndicatorCategory.IDENTITY: "landlord identity concerns",
            IndicatorCategory.PRICING: "unusual pricing",
            IndicatorCategory.TEXT_STYLE: "writing style concerns",
            IndicatorCategory.CONTENT: "suspicious listing content",
            IndicatorCategory.ML_ANOMALY: "unusual patterns in the listing",
            IndicatorCategory.NLP_SEMANTIC: "suspicious language patterns"
        }
        
        detected_issues = [
            category_descriptions.get(cat, cat.value) 
            for cat in categories_detected
        ]
        
        if detected_issues:
            issues_text = ", ".join(detected_issues[:4])
            story += f"\n\nDetected issue categories: {issues_text}."
        
        # Critical issues with evidence - use user-friendly descriptions
        if critical:
            story += f"\n\n🔴 **Critical Issues ({len(critical)}):**\n"
            for ind in critical[:4]:
                friendly_desc = IndicatorEngine._get_user_friendly_description(ind)
                story += f"  • {friendly_desc}\n"
                if ind.evidence:
                    story += f"    Found: \"{ind.evidence[0][:80]}{'...' if len(ind.evidence[0]) > 80 else ''}\"\n"
        
        # Moderate concerns - use user-friendly descriptions
        if moderate:
            story += f"\n\n🟡 **Moderate Concerns ({len(moderate)}):**\n"
            for ind in moderate[:3]:
                friendly_desc = IndicatorEngine._get_user_friendly_description(ind)
                story += f"  • {friendly_desc}\n"
        
        # Low severity notes (only if no critical/moderate)
        if low_severity and not critical and not moderate:
            story += f"\n\n🔵 **Minor Notes ({len(low_severity)}):**\n"
            for ind in low_severity[:2]:
                friendly_desc = IndicatorEngine._get_user_friendly_description(ind)
                story += f"  • {friendly_desc}\n"
        
        # Language analysis findings (user-friendly, no "NLP" or "AI" jargon)
        nlp_findings = [i for i in indicators if i.category == IndicatorCategory.NLP_SEMANTIC]
        if nlp_findings and not any(i in critical for i in nlp_findings):
            story += f"\n\n🔍 **Language Analysis:**\n"
            for ind in nlp_findings[:2]:
                friendly_desc = IndicatorEngine._get_user_friendly_description(ind)
                story += f"  • {friendly_desc}\n"
        
        # Dynamic recommendations based on specific indicators found
        story += "\n\n💡 **Recommendations:**\n"
        
        # Payment-related recommendations
        payment_indicators = [i for i in indicators if i.category == IndicatorCategory.PAYMENT]
        if payment_indicators:
            story += "  • NEVER use wire transfers, cryptocurrency, or gift cards for rental payments\n"
            story += "  • Only pay through verified, traceable platforms with buyer protection\n"
        
        # Urgency-related recommendations
        urgency_indicators = [i for i in indicators if i.category == IndicatorCategory.URGENCY]
        if urgency_indicators:
            story += "  • Don't rush - legitimate landlords won't pressure you with artificial deadlines\n"
        
        # Price-related recommendations
        price_indicators = [i for i in indicators if i.category == IndicatorCategory.PRICING]
        if price_indicators:
            story += "  • Compare prices with similar listings in the area - if it seems too good to be true, it probably is\n"
        
        # General recommendations based on computed local risk level
        if local_risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
            story += "  • DO NOT send money without viewing the property in person\n"
            story += "  • Verify the landlord's identity through official channels\n"
            story += "  • Consider reporting this listing to the platform\n"
        elif local_risk_level == RiskLevel.MEDIUM:
            story += "  • Request to view the property before making any payments\n"
            story += "  • Verify contact information and landlord identity\n"
        else:
            story += "  • Standard precautions: verify details and view property before committing\n"
        
        return story


# Singleton instance for easy import
indicator_engine = IndicatorEngine()
