"""
Message/Conversation Analysis Engine - Real AI Communication Risk Assessment

This module implements NLP-based communication risk analysis as specified
in the capstone proposal (FR1, Appendix A):
"NLP-based communication risk analysis to messages exchanged between renters and landlords"

Key capabilities:
1. BERT-based message classification (fraud patterns vs legitimate)
2. Conversation flow analysis (detecting escalation tactics)
3. Sentiment trajectory tracking (manipulation detection)
4. Urgency pattern recognition
5. Social engineering tactic identification
6. Multi-turn dialogue risk assessment

For capstone: COMP385 AI Project
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import AI libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class MessageRiskLevel(str, Enum):
    """Risk levels for individual messages"""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TacticType(str, Enum):
    """Social engineering tactic types"""
    URGENCY = "urgency"
    AUTHORITY = "authority"
    SCARCITY = "scarcity"
    SOCIAL_PROOF = "social_proof"
    RECIPROCITY = "reciprocity"
    EMOTIONAL_APPEAL = "emotional_appeal"
    ISOLATION = "isolation"
    TRUST_BUILDING = "trust_building"
    FEAR = "fear"
    GREED = "greed"


@dataclass
class MessageAnalysisResult:
    """Analysis result for a single message"""
    message_id: str
    content: str
    sender: str  # 'renter' or 'landlord'
    timestamp: Optional[datetime]
    
    # Risk assessment
    risk_level: MessageRiskLevel
    risk_score: float  # 0-1
    confidence: float
    
    # Detected tactics
    tactics_detected: List[TacticType]
    tactic_evidence: Dict[str, List[str]]
    
    # NLP analysis
    sentiment: float  # -1 to 1
    urgency_score: float  # 0-1
    manipulation_score: float  # 0-1
    
    # Specific patterns
    payment_mentions: List[str]
    contact_redirect_attempts: List[str]
    suspicious_phrases: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "content_preview": self.content[:100] + "..." if len(self.content) > 100 else self.content,
            "sender": self.sender,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "risk": {
                "level": self.risk_level.value,
                "score": round(self.risk_score, 3),
                "confidence": round(self.confidence, 3)
            },
            "tactics": {
                "detected": [t.value for t in self.tactics_detected],
                "evidence": self.tactic_evidence
            },
            "nlp_scores": {
                "sentiment": round(self.sentiment, 3),
                "urgency": round(self.urgency_score, 3),
                "manipulation": round(self.manipulation_score, 3)
            },
            "patterns": {
                "payment_mentions": self.payment_mentions,
                "contact_redirects": self.contact_redirect_attempts,
                "suspicious_phrases": self.suspicious_phrases
            }
        }


@dataclass
class ConversationAnalysisResult:
    """Analysis result for entire conversation"""
    conversation_id: str
    total_messages: int
    analyzed_messages: int
    
    # Overall assessment
    overall_risk_level: MessageRiskLevel
    overall_risk_score: float
    confidence: float
    
    # Per-message analysis
    message_analyses: List[MessageAnalysisResult]
    
    # Conversation patterns
    escalation_detected: bool
    escalation_points: List[int]  # Message indices where risk escalated
    
    # Red flags
    red_flags: List[Dict[str, Any]]
    
    # Recommendations
    recommendation: str
    action_items: List[str]
    
    # Progression analysis
    risk_trajectory: List[float]  # Risk score over time
    sentiment_trajectory: List[float]  # Sentiment over time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "summary": {
                "total_messages": self.total_messages,
                "analyzed": self.analyzed_messages,
                "overall_risk": self.overall_risk_level.value,
                "risk_score": round(self.overall_risk_score, 3),
                "confidence": round(self.confidence, 3)
            },
            "escalation": {
                "detected": self.escalation_detected,
                "escalation_points": self.escalation_points
            },
            "red_flags": self.red_flags,
            "recommendation": self.recommendation,
            "action_items": self.action_items,
            "trajectories": {
                "risk": [round(r, 3) for r in self.risk_trajectory],
                "sentiment": [round(s, 3) for s in self.sentiment_trajectory]
            },
            "messages": [m.to_dict() for m in self.message_analyses]
        }


class MessageAnalysisEngine:
    """
    Real AI engine for analyzing messages/conversations for fraud patterns.
    
    Uses:
    1. Fine-tuned BERT for message classification
    2. Sentiment analysis for manipulation detection
    3. Pattern recognition for social engineering tactics
    4. Sequence analysis for conversation flow
    """
    
    # Social engineering tactic patterns
    TACTIC_PATTERNS = {
        TacticType.URGENCY: [
            r'\b(urgent|urgently|immediately|right\s*now|asap|hurry|quick|fast|limited\s*time)\b',
            r'\b(expires?|deadline|last\s*chance|act\s*now|don\'t\s*wait)\b',
            r'\b(today\s*only|hours?\s*left|running\s*out)\b',
            r'(!{2,}|\?{2,})',  # Multiple punctuation
        ],
        TacticType.AUTHORITY: [
            r'\b(official|authorized|verified|certified|licensed)\b',
            r'\b(government|legal|attorney|lawyer|court)\b',
            r'\b(manager|director|supervisor|representative)\b',
        ],
        TacticType.SCARCITY: [
            r'\b(only\s*one|last\s*(one|unit)|few\s*left|rare|exclusive)\b',
            r'\b(high\s*demand|many\s*interested|several\s*applicants)\b',
            r'\b(won\'t\s*last|going\s*fast|selling\s*quick)\b',
        ],
        TacticType.EMOTIONAL_APPEAL: [
            r'\b(please|beg|desperate|help|need|trust\s*me)\b',
            r'\b(god|blessing|prayer|faith|honest)\b',
            r'\b(family|children|kids|elderly|sick)\b',
            r'\b(dream|perfect|amazing|wonderful)\b',
        ],
        TacticType.TRUST_BUILDING: [
            r'\b(honest|trustworthy|reliable|genuine|sincere)\b',
            r'\b(promise|guarantee|assured|certain)\b',
            r'\b(personal|confidential|between\s*us)\b',
        ],
        TacticType.FEAR: [
            r'\b(lose|miss\s*out|regret|risk|danger)\b',
            r'\b(scam|fraud|careful|warning)\b',  # Ironic use
            r'\b(problem|issue|trouble|difficult)\b',
        ],
        TacticType.GREED: [
            r'\b(deal|discount|save|cheap|bargain|free)\b',
            r'\b(bonus|extra|included|throw\s*in)\b',
            r'\b(half\s*price|reduced|special\s*offer)\b',
        ],
        TacticType.ISOLATION: [
            r'\b(private|secret|confidential|don\'t\s*tell)\b',
            r'\b(just\s*(you|us)|between\s*us|special\s*for\s*you)\b',
            r'\b(offline|outside|direct|personal\s*email)\b',
        ],
    }
    
    # Payment red flag patterns
    PAYMENT_PATTERNS = [
        r'\b(wire\s*transfer|western\s*union|moneygram)\b',
        r'\b(bitcoin|btc|crypto|ethereum|usdt)\b',
        r'\b(gift\s*card|itunes|amazon\s*card|google\s*play)\b',
        r'\b(zelle|venmo|cashapp|paypal\s*friends)\b',
        r'\b(cash|money\s*order|bank\s*draft)\b',
        r'\b(deposit|upfront|advance|before\s*viewing)\b',
    ]
    
    # Contact redirect patterns
    CONTACT_REDIRECT_PATTERNS = [
        r'\b(text\s*me|whatsapp|telegram|signal)\b',
        r'\b(personal\s*email|private\s*number|direct\s*contact)\b',
        r'\b(call\s*me\s*at|reach\s*me\s*at|contact\s*me\s*on)\b',
        r'\b(off\s*platform|outside\s*the\s*site|direct\s*message)\b',
        r'[\w\.-]+@[\w\.-]+\.\w+',  # Email addresses
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # Phone numbers
    ]
    
    def __init__(self):
        self.sentiment_analyzer = None
        self.classifier = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization of ML models"""
        if self._initialized:
            return
        
        if TRANSFORMERS_AVAILABLE:
            try:
                # Load sentiment analyzer
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1  # CPU
                )
                
                # Try to load fraud classifier
                try:
                    from application.use_cases.bert_fraud_classifier import get_fraud_classifier
                    self.classifier = get_fraud_classifier()
                except Exception:
                    logger.warning("Fraud classifier not available, using sentiment-based analysis")
                
                self._initialized = True
                logger.info("Message Analysis Engine initialized with BERT models")
            except Exception as e:
                logger.warning(f"Failed to initialize ML models: {e}")
        else:
            logger.warning("Transformers not available, using pattern-based analysis")
    
    def analyze_message(
        self,
        message: str,
        sender: str = "unknown",
        message_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> MessageAnalysisResult:
        """
        Analyze a single message for fraud indicators.
        
        Args:
            message: The message content
            sender: Who sent it ('renter' or 'landlord')
            message_id: Optional message identifier
            timestamp: Optional timestamp
            
        Returns:
            MessageAnalysisResult with full analysis
        """
        self._ensure_initialized()
        
        message_id = message_id or f"msg_{hash(message) % 10000}"
        message_lower = message.lower()
        
        # 1. Detect tactics
        tactics_detected = []
        tactic_evidence = {}
        
        for tactic, patterns in self.TACTIC_PATTERNS.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, message_lower, re.IGNORECASE)
                matches.extend(found)
            if matches:
                tactics_detected.append(tactic)
                tactic_evidence[tactic.value] = list(set(matches))[:5]
        
        # 2. Detect payment mentions
        payment_mentions = []
        for pattern in self.PAYMENT_PATTERNS:
            found = re.findall(pattern, message_lower, re.IGNORECASE)
            payment_mentions.extend(found)
        payment_mentions = list(set(payment_mentions))
        
        # 3. Detect contact redirects
        contact_redirects = []
        for pattern in self.CONTACT_REDIRECT_PATTERNS:
            found = re.findall(pattern, message_lower, re.IGNORECASE)
            contact_redirects.extend([str(m) for m in found])
        contact_redirects = list(set(contact_redirects))[:5]
        
        # 4. Compute sentiment
        sentiment = self._compute_sentiment(message)
        
        # 5. Compute urgency score
        urgency_score = self._compute_urgency(message_lower)
        
        # 6. Compute manipulation score
        manipulation_score = self._compute_manipulation(
            tactics_detected, sentiment, urgency_score
        )
        
        # 7. Get BERT classification if available
        bert_score = 0.0
        if self.classifier and self.classifier.is_trained:
            try:
                prediction = self.classifier.predict(message)
                bert_score = prediction.get('fraud_probability', 0)
            except Exception:
                pass
        
        # 8. Compute overall risk score
        risk_score = self._compute_risk_score(
            bert_score=bert_score,
            tactics_count=len(tactics_detected),
            payment_mentions=len(payment_mentions),
            contact_redirects=len(contact_redirects),
            manipulation_score=manipulation_score,
            urgency_score=urgency_score
        )
        
        # 9. Determine risk level
        risk_level = self._score_to_level(risk_score)
        
        # 10. Extract suspicious phrases
        suspicious_phrases = self._extract_suspicious_phrases(message)
        
        return MessageAnalysisResult(
            message_id=message_id,
            content=message,
            sender=sender,
            timestamp=timestamp,
            risk_level=risk_level,
            risk_score=risk_score,
            confidence=0.85 if self.classifier else 0.70,
            tactics_detected=tactics_detected,
            tactic_evidence=tactic_evidence,
            sentiment=sentiment,
            urgency_score=urgency_score,
            manipulation_score=manipulation_score,
            payment_mentions=payment_mentions,
            contact_redirect_attempts=contact_redirects,
            suspicious_phrases=suspicious_phrases
        )
    
    def analyze_conversation(
        self,
        messages: List[Dict[str, Any]],
        conversation_id: Optional[str] = None
    ) -> ConversationAnalysisResult:
        """
        Analyze an entire conversation for fraud patterns.
        
        This performs:
        1. Individual message analysis
        2. Escalation pattern detection
        3. Risk trajectory analysis
        4. Cross-message pattern correlation
        
        Args:
            messages: List of message dicts with 'content', 'sender', optional 'timestamp'
            conversation_id: Optional conversation identifier
            
        Returns:
            ConversationAnalysisResult with full analysis
        """
        conversation_id = conversation_id or f"conv_{datetime.now().timestamp()}"
        
        # Analyze each message
        message_analyses = []
        risk_trajectory = []
        sentiment_trajectory = []
        
        for i, msg in enumerate(messages):
            content = msg.get('content', '')
            sender = msg.get('sender', 'unknown')
            timestamp = msg.get('timestamp')
            
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except:
                    timestamp = None
            
            analysis = self.analyze_message(
                message=content,
                sender=sender,
                message_id=f"{conversation_id}_msg_{i}",
                timestamp=timestamp
            )
            
            message_analyses.append(analysis)
            risk_trajectory.append(analysis.risk_score)
            sentiment_trajectory.append(analysis.sentiment)
        
        # Detect escalation
        escalation_detected, escalation_points = self._detect_escalation(risk_trajectory)
        
        # Compile red flags
        red_flags = self._compile_red_flags(message_analyses)
        
        # Calculate overall risk
        overall_risk_score = self._calculate_overall_risk(
            message_analyses,
            escalation_detected,
            red_flags
        )
        overall_risk_level = self._score_to_level(overall_risk_score)
        
        # Generate recommendations
        recommendation, action_items = self._generate_recommendations(
            overall_risk_level,
            red_flags,
            escalation_detected
        )
        
        return ConversationAnalysisResult(
            conversation_id=conversation_id,
            total_messages=len(messages),
            analyzed_messages=len(message_analyses),
            overall_risk_level=overall_risk_level,
            overall_risk_score=overall_risk_score,
            confidence=0.85 if self.classifier else 0.70,
            message_analyses=message_analyses,
            escalation_detected=escalation_detected,
            escalation_points=escalation_points,
            red_flags=red_flags,
            recommendation=recommendation,
            action_items=action_items,
            risk_trajectory=risk_trajectory,
            sentiment_trajectory=sentiment_trajectory
        )
    
    def _compute_sentiment(self, text: str) -> float:
        """Compute sentiment score from -1 (negative) to 1 (positive)"""
        if self.sentiment_analyzer:
            try:
                result = self.sentiment_analyzer(text[:512])
                label = result[0]['label']
                score = result[0]['score']
                return score if label == 'POSITIVE' else -score
            except Exception:
                pass
        
        # Fallback: Simple pattern-based
        positive_words = len(re.findall(
            r'\b(good|great|excellent|happy|love|wonderful|amazing)\b',
            text.lower()
        ))
        negative_words = len(re.findall(
            r'\b(bad|terrible|awful|angry|hate|horrible|scam)\b',
            text.lower()
        ))
        
        total = positive_words + negative_words
        if total == 0:
            return 0.0
        return (positive_words - negative_words) / total
    
    def _compute_urgency(self, text: str) -> float:
        """Compute urgency score from 0 to 1"""
        urgency_indicators = 0
        
        # Check patterns
        for pattern in self.TACTIC_PATTERNS[TacticType.URGENCY]:
            if re.search(pattern, text, re.IGNORECASE):
                urgency_indicators += 1
        
        # Additional urgency signals
        if re.search(r'(!{2,})', text):
            urgency_indicators += 1
        if re.search(r'\b(now|today|tonight)\b', text, re.IGNORECASE):
            urgency_indicators += 0.5
        if re.search(r'\b\d+\s*(hours?|minutes?|days?)\s*(left|remaining)\b', text, re.IGNORECASE):
            urgency_indicators += 1
        
        return min(1.0, urgency_indicators / 5)
    
    def _compute_manipulation(
        self,
        tactics: List[TacticType],
        sentiment: float,
        urgency: float
    ) -> float:
        """Compute manipulation score from 0 to 1"""
        # Score based on number of tactics
        tactic_score = min(len(tactics) / 4, 1.0)
        
        # High urgency with emotional appeal = manipulation
        if TacticType.EMOTIONAL_APPEAL in tactics and urgency > 0.5:
            tactic_score += 0.2
        
        # Trust building with urgency = classic manipulation
        if TacticType.TRUST_BUILDING in tactics and TacticType.URGENCY in tactics:
            tactic_score += 0.2
        
        # Isolation tactics
        if TacticType.ISOLATION in tactics:
            tactic_score += 0.3
        
        return min(1.0, tactic_score)
    
    def _compute_risk_score(
        self,
        bert_score: float,
        tactics_count: int,
        payment_mentions: int,
        contact_redirects: int,
        manipulation_score: float,
        urgency_score: float
    ) -> float:
        """Compute overall risk score from 0 to 1"""
        # Weighted combination
        scores = [
            (bert_score, 0.30),  # BERT classification
            (min(tactics_count / 3, 1.0), 0.20),  # Tactics
            (min(payment_mentions / 2, 1.0) * 0.9, 0.20),  # Payment (critical)
            (min(contact_redirects / 2, 1.0) * 0.7, 0.10),  # Contact redirects
            (manipulation_score, 0.10),
            (urgency_score, 0.10)
        ]
        
        weighted_sum = sum(score * weight for score, weight in scores)
        
        # Critical escalators
        if payment_mentions >= 2:
            weighted_sum = min(1.0, weighted_sum + 0.2)
        if tactics_count >= 4:
            weighted_sum = min(1.0, weighted_sum + 0.1)
        
        return weighted_sum
    
    def _score_to_level(self, score: float) -> MessageRiskLevel:
        """Convert numeric score to risk level"""
        if score >= 0.8:
            return MessageRiskLevel.CRITICAL
        elif score >= 0.6:
            return MessageRiskLevel.HIGH
        elif score >= 0.4:
            return MessageRiskLevel.MEDIUM
        elif score >= 0.2:
            return MessageRiskLevel.LOW
        else:
            return MessageRiskLevel.SAFE
    
    def _extract_suspicious_phrases(self, message: str) -> List[str]:
        """Extract suspicious phrases from message"""
        phrases = []
        
        suspicious_patterns = [
            r'(send\s+(money|payment|deposit)\s+\w+\s+\w+)',
            r'(before\s+(you\s+)?see(ing)?\s+the\s+(property|place|apartment))',
            r'(can\'t\s+meet|not\s+available|out\s+of\s+(town|country))',
            r'(wire|transfer)\s+\w+\s+\w+',
            r'(text\s+me\s+(at|on)\s+\d+)',
            r'(email\s+me\s+(at|on)\s+[\w@.]+)',
            r'(don\'t\s+use\s+this\s+(site|platform))',
            r'(deal\s+with\s+me\s+directly)',
        ]
        
        for pattern in suspicious_patterns:
            matches = re.findall(pattern, message.lower())
            for match in matches:
                if isinstance(match, tuple):
                    phrases.append(match[0])
                else:
                    phrases.append(match)
        
        return phrases[:5]
    
    def _detect_escalation(
        self,
        risk_trajectory: List[float]
    ) -> Tuple[bool, List[int]]:
        """Detect if risk is escalating over conversation"""
        if len(risk_trajectory) < 2:
            return False, []
        
        escalation_points = []
        escalation_detected = False
        
        for i in range(1, len(risk_trajectory)):
            # Significant increase (>0.2)
            if risk_trajectory[i] - risk_trajectory[i-1] > 0.2:
                escalation_points.append(i)
                escalation_detected = True
        
        # Also detect sustained high risk
        high_risk_count = sum(1 for r in risk_trajectory if r > 0.6)
        if high_risk_count >= len(risk_trajectory) // 2:
            escalation_detected = True
        
        return escalation_detected, escalation_points
    
    def _compile_red_flags(
        self,
        analyses: List[MessageAnalysisResult]
    ) -> List[Dict[str, Any]]:
        """Compile all red flags from message analyses"""
        red_flags = []
        
        # Aggregate payment mentions
        all_payments = []
        for a in analyses:
            all_payments.extend(a.payment_mentions)
        if all_payments:
            red_flags.append({
                "type": "payment_fraud_risk",
                "severity": "high",
                "description": f"Suspicious payment methods mentioned: {', '.join(set(all_payments))}",
                "count": len(all_payments)
            })
        
        # Aggregate contact redirects
        all_redirects = []
        for a in analyses:
            all_redirects.extend(a.contact_redirect_attempts)
        if all_redirects:
            red_flags.append({
                "type": "contact_redirect",
                "severity": "medium",
                "description": "Attempts to move communication off-platform detected",
                "count": len(all_redirects)
            })
        
        # Aggregate tactics
        all_tactics = set()
        for a in analyses:
            all_tactics.update(a.tactics_detected)
        if len(all_tactics) >= 3:
            red_flags.append({
                "type": "social_engineering",
                "severity": "high",
                "description": f"Multiple manipulation tactics detected: {', '.join(t.value for t in all_tactics)}",
                "count": len(all_tactics)
            })
        
        # High urgency across messages
        avg_urgency = sum(a.urgency_score for a in analyses) / len(analyses) if analyses else 0
        if avg_urgency > 0.5:
            red_flags.append({
                "type": "high_pressure",
                "severity": "medium",
                "description": f"High-pressure tactics throughout conversation (avg urgency: {avg_urgency:.0%})",
                "count": 1
            })
        
        return red_flags
    
    def _calculate_overall_risk(
        self,
        analyses: List[MessageAnalysisResult],
        escalation: bool,
        red_flags: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall conversation risk"""
        if not analyses:
            return 0.0
        
        # Average risk
        avg_risk = sum(a.risk_score for a in analyses) / len(analyses)
        
        # Max risk
        max_risk = max(a.risk_score for a in analyses)
        
        # Weighted combination
        overall = avg_risk * 0.4 + max_risk * 0.4
        
        # Escalation penalty
        if escalation:
            overall += 0.1
        
        # Red flag penalty
        high_severity_flags = sum(1 for f in red_flags if f['severity'] == 'high')
        overall += high_severity_flags * 0.05
        
        return min(1.0, overall)
    
    def _generate_recommendations(
        self,
        risk_level: MessageRiskLevel,
        red_flags: List[Dict[str, Any]],
        escalation: bool
    ) -> Tuple[str, List[str]]:
        """Generate recommendations based on analysis"""
        action_items = []
        
        if risk_level == MessageRiskLevel.CRITICAL:
            recommendation = "⛔ STOP COMMUNICATION IMMEDIATELY. This conversation shows clear signs of a rental scam."
            action_items = [
                "Do NOT send any money or personal information",
                "Report this listing to the platform immediately",
                "Block all contact with this person",
                "File a report with local authorities if you've shared sensitive info"
            ]
        elif risk_level == MessageRiskLevel.HIGH:
            recommendation = "⚠️ HIGH RISK. Multiple fraud indicators detected. Proceed with extreme caution."
            action_items = [
                "Request an in-person viewing before any commitment",
                "Verify the landlord's identity through official channels",
                "Never send money before seeing the property",
                "Use only the platform's official payment methods"
            ]
        elif risk_level == MessageRiskLevel.MEDIUM:
            recommendation = "⚠️ Some concerning patterns detected. Verify legitimacy before proceeding."
            action_items = [
                "Ask for verification of property ownership",
                "Request to meet in person",
                "Research the landlord online",
                "Trust your instincts if something feels off"
            ]
        elif risk_level == MessageRiskLevel.LOW:
            recommendation = "Minor concerns noted. Standard caution advised."
            action_items = [
                "Verify property details before signing",
                "Use secure payment methods",
                "Get all agreements in writing"
            ]
        else:
            recommendation = "✅ Conversation appears legitimate. Standard rental precautions apply."
            action_items = [
                "Proceed with normal due diligence",
                "Review lease terms carefully",
                "Document all communications"
            ]
        
        # Add specific action items based on red flags
        for flag in red_flags:
            if flag['type'] == 'payment_fraud_risk':
                if "Never send wire transfers or gift cards" not in action_items:
                    action_items.insert(0, "Never send wire transfers or gift cards")
            elif flag['type'] == 'contact_redirect':
                if "Keep all communication on the platform" not in action_items:
                    action_items.insert(0, "Keep all communication on the platform")
        
        return recommendation, action_items[:5]


# Singleton instance
message_analysis_engine = MessageAnalysisEngine()


def analyze_message(
    message: str,
    sender: str = "unknown"
) -> Dict[str, Any]:
    """Convenience function to analyze a single message"""
    result = message_analysis_engine.analyze_message(message, sender)
    return result.to_dict()


def analyze_conversation(
    messages: List[Dict[str, Any]],
    conversation_id: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function to analyze a conversation"""
    result = message_analysis_engine.analyze_conversation(messages, conversation_id)
    return result.to_dict()
