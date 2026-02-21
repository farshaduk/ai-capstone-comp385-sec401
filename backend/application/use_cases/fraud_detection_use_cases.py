import re
import numpy as np
import base64
from typing import Dict, Any, List, Tuple, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from infrastructure.database import RiskAnalysisModel, MLModelModel, UserModel, FeedbackModel
from application.use_cases.indicator_engine import indicator_engine, Indicator, RiskLevel, IndicatorEngine
import pandas as pd
from datetime import datetime
import aiohttp
from bs4 import BeautifulSoup
import asyncio
import os
import logging

# WeasyPrint is optional - requires GTK libraries on Windows
try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except (OSError, ImportError):
    WEASYPRINT_AVAILABLE = False
    HTML = None

# BERT Fraud Classifier - the REAL AI component
try:
    from application.use_cases.bert_fraud_classifier import get_fraud_classifier, BertFraudClassifier
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    get_fraud_classifier = None

# Price Anomaly Detection Engine
try:
    from application.use_cases.price_anomaly_engine import price_anomaly_engine
    PRICE_ANOMALY_AVAILABLE = True
except ImportError:
    PRICE_ANOMALY_AVAILABLE = False
    price_anomaly_engine = None

# Address Validation Engine
try:
    from application.use_cases.address_validation_engine import address_validation_engine
    ADDRESS_VALIDATION_AVAILABLE = True
except ImportError:
    ADDRESS_VALIDATION_AVAILABLE = False
    address_validation_engine = None

logger = logging.getLogger(__name__)

# Get the backend directory (grandparent of use_cases folder)
# Path: application/use_cases/fraud_detection_use_cases.py -> application/use_cases -> application -> backend
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FraudDetectionUseCases:
    """
    Fraud Detection Use Cases - Powered by REAL AI.
    
    Architecture:
    1. BERT-based text classification (fine-tuned DistilBERT) - Primary AI signal
    2. Rule-based Indicator Engine - Explainability and specific pattern detection
    3. Signal fusion for robust detection
    
    This replaces the old MVP code that used:
    - IsolationForest on character counts (removed - not real fraud detection)
    - Keyword matching (indicator engine still provides this for explainability)
    """
    
    @staticmethod
    async def analyze_listing(
        db: AsyncSession,
        user_id: int,
        listing_text: str,
        listing_price: float = None,
        location: str = None
    ) -> Dict[str, Any]:
        """
        Analyze a rental listing for fraud indicators.
        
        This method uses:
        1. BERT classifier (primary AI signal) - Understands semantic fraud patterns
        2. Indicator Engine (explainability) - Provides specific fraud indicators
        
        Returns:
        - risk_score: 0-100 integer scale
        - risk_level: Standardized level (Very Low to Very High)
        - confidence: 0.0-1.0 based on signal agreement
        - indicators: List of coded indicators with severity and evidence
        - risk_story: Human-readable narrative
        - bert_prediction: BERT model output (if available)
        """
        
        # Check if user has scans remaining
        result = await db.execute(select(UserModel).where(UserModel.id == user_id))
        user = result.scalar_one_or_none()
        
        if not user:
            raise ValueError("User not found")
        
        if user.scans_remaining <= 0:
            raise ValueError("No scans remaining. Please upgrade your subscription.")
        
        # =====================================================================
        # BERT CLASSIFIER (PRIMARY AI SIGNAL)
        # Fine-tuned DistilBERT for fraud text classification
        # =====================================================================
        bert_prediction = None
        bert_fraud_probability = None
        
        if BERT_AVAILABLE:
            try:
                classifier = get_fraud_classifier()
                if classifier.is_trained:
                    bert_prediction = classifier.predict(listing_text)
                    bert_fraud_probability = bert_prediction.get('fraud_probability', None)
                    logger.info(f"BERT prediction: {bert_prediction}")
            except Exception as e:
                logger.warning(f"BERT prediction failed: {e}")
                bert_prediction = None
        
        # =====================================================================
        # PRICE ANOMALY DETECTION (Statistical AI)
        # Compares price to Canadian market data
        # =====================================================================
        price_analysis = None
        price_risk_contribution = 0.0
        
        if PRICE_ANOMALY_AVAILABLE and listing_price is not None:
            try:
                # Extract bedrooms from text if possible
                bedrooms = 1  # Default
                bedroom_match = re.search(r'(\d+)\s*(?:bed(?:room)?s?|br)', listing_text.lower())
                if bedroom_match:
                    bedrooms = int(bedroom_match.group(1))
                
                price_analysis = price_anomaly_engine.analyze(
                    price=listing_price,
                    location=location,
                    bedrooms=bedrooms,
                    listing_text=listing_text
                )
                
                # Price contributes to overall risk — higher weight for clear anomalies
                price_risk_raw = getattr(price_analysis, 'risk_score', 0.0)
                if price_risk_raw >= 0.80:  # Suspiciously/extremely low
                    price_risk_contribution = price_risk_raw * 0.30
                elif price_risk_raw >= 0.50:  # Slightly low
                    price_risk_contribution = price_risk_raw * 0.20
                else:
                    price_risk_contribution = price_risk_raw * 0.10
                
                logger.info(f"Price analysis: {getattr(price_analysis, 'risk_level', 'unknown')}, deviation: {getattr(price_analysis, 'price_deviation_percent', 0)}%")
            except Exception as e:
                logger.warning(f"Price anomaly detection failed: {e}")
        
        # =====================================================================
        # ADDRESS VALIDATION (Geocoding AI)
        # Validates address using OpenStreetMap/Nominatim
        # =====================================================================
        address_validation = None
        address_risk_contribution = 0.0
        
        if ADDRESS_VALIDATION_AVAILABLE:
            # If no location provided, try to extract one from the listing text
            effective_location = location
            if not effective_location:
                try:
                    effective_location = address_validation_engine.extract_address_from_text(listing_text)
                    if effective_location:
                        logger.info(f"Auto-extracted address from listing text: {effective_location}")
                except Exception as e:
                    logger.debug(f"Address extraction from text failed: {e}")
            
            if effective_location:
                try:
                    address_validation = await address_validation_engine.validate(effective_location)
                    
                    # Address contributes to overall risk
                    if address_validation:
                        address_risk_contribution = getattr(address_validation, 'risk_score', 0.0) * 0.1
                        
                        # If we extracted the address, also populate location for price analysis
                        if not location and effective_location:
                            location = effective_location
                        
                        logger.info(f"Address validation: {getattr(address_validation, 'status', 'unknown')}")
                except Exception as e:
                    logger.warning(f"Address validation failed: {e}")
        
        # =====================================================================
        # INDICATOR ENGINE ANALYSIS (Rule-based explainability)
        # =====================================================================
        # Pass BERT probability as ML signal for indicator weighting
        indicators, risk_score_raw, confidence = indicator_engine.analyze(
            text=listing_text,
            price=listing_price,
            location=location,
            ml_score=bert_fraud_probability  # BERT is our ML signal now
        )
        
        # =====================================================================
        # MERGE PRICE ANOMALY INDICATORS INTO MAIN LIST
        # The price_anomaly_engine is bedroom/property-type aware and more
        # precise than the indicator_engine's flat city thresholds.
        # =====================================================================
        if price_analysis is not None:
            price_risk_level = getattr(price_analysis, 'risk_level', None)
            price_deviation = getattr(price_analysis, 'price_deviation_percent', 0)
            price_mkt_avg = getattr(price_analysis, 'market_average', 0)
            
            # Only add if indicator_engine didn't already flag a price issue
            has_price_indicator = any(
                getattr(ind, 'code', '') in ('SUSPICIOUS_LOW_PRICE', 'BELOW_MARKET_PRICE')
                for ind in indicators
            )
            
            if not has_price_indicator and price_risk_level and price_risk_level.value != 'normal':
                from application.use_cases.indicator_engine import Indicator, IndicatorCategory
                
                if price_risk_level.value == 'extremely_low':
                    indicators.append(Indicator(
                        code="PRICE_EXTREMELY_LOW",
                        category=IndicatorCategory.PRICING,
                        severity=5,
                        evidence=[f"Listed: ${listing_price:.0f}/mo", f"Market avg: ${price_mkt_avg:.0f}/mo", f"{abs(price_deviation):.0f}% below market"],
                        description=f"Price is {abs(price_deviation):.0f}% below market average — major scam indicator",
                        impact_score=0.25
                    ))
                elif price_risk_level.value == 'suspiciously_low':
                    indicators.append(Indicator(
                        code="PRICE_SUSPICIOUSLY_LOW",
                        category=IndicatorCategory.PRICING,
                        severity=4,
                        evidence=[f"Listed: ${listing_price:.0f}/mo", f"Market avg: ${price_mkt_avg:.0f}/mo", f"{abs(price_deviation):.0f}% below market"],
                        description=f"Price is {abs(price_deviation):.0f}% below market average for this property type — potential scam",
                        impact_score=0.20
                    ))
                elif price_risk_level.value == 'slightly_low':
                    indicators.append(Indicator(
                        code="PRICE_BELOW_MARKET",
                        category=IndicatorCategory.PRICING,
                        severity=3,
                        evidence=[f"Listed: ${listing_price:.0f}/mo", f"Market avg: ${price_mkt_avg:.0f}/mo", f"{abs(price_deviation):.0f}% below market"],
                        description=f"Price is {abs(price_deviation):.0f}% below market average — warrants verification",
                        impact_score=0.12
                    ))
                elif price_risk_level.value == 'unusually_high':
                    indicators.append(Indicator(
                        code="PRICE_UNUSUALLY_HIGH",
                        category=IndicatorCategory.PRICING,
                        severity=2,
                        evidence=[f"Listed: ${listing_price:.0f}/mo", f"Market avg: ${price_mkt_avg:.0f}/mo", f"{price_deviation:.0f}% above market"],
                        description=f"Price is {price_deviation:.0f}% above market average",
                        impact_score=0.05
                    ))
        
        # =====================================================================
        # FUSE ALL AI SIGNALS
        # BERT provides semantic understanding (trained AI)
        # Price Anomaly provides market intelligence
        # Address Validation provides location verification
        # Indicators provide explainability (rule-based)
        # =====================================================================
        # Count critical / high-severity indicators for adaptive weighting
        critical_count = sum(1 for ind in indicators if ind.severity >= 4)
        high_severity_count = sum(1 for ind in indicators if ind.severity >= 3)
        
        if bert_fraud_probability is not None:
            # Adaptive fusion weights based on indicator severity.
            # When clear fraud indicators are detected (gift cards, overseas
            # wire transfers, etc.), BERT should NOT suppress the score.
            if critical_count >= 3:
                bert_weight, indicator_weight = 0.25, 0.50
            elif critical_count >= 2:
                bert_weight, indicator_weight = 0.30, 0.45
            elif critical_count >= 1:
                bert_weight, indicator_weight = 0.35, 0.40
            elif high_severity_count >= 2:
                bert_weight, indicator_weight = 0.40, 0.35
            else:
                bert_weight, indicator_weight = 0.50, 0.25
            
            base_score = (bert_fraud_probability * bert_weight) + (risk_score_raw * indicator_weight)
            fused_score = base_score + price_risk_contribution + address_risk_contribution
            
            # Critical indicator floor: prevent BERT from suppressing
            # unambiguous fraud signals (gift cards, overseas money, etc.)
            if critical_count >= 3:
                min_floor = 0.75
            elif critical_count >= 2:
                min_floor = 0.60
            elif critical_count >= 1:
                min_floor = 0.45
            else:
                min_floor = 0.0
            
            fused_score = max(fused_score, min_floor)
            
            # Boost confidence if multiple signals agree
            agreement_count = 0
            if bert_fraud_probability > 0.5:
                agreement_count += 1
            if risk_score_raw > 0.4:
                agreement_count += 1
            if price_risk_contribution > 0.05:
                agreement_count += 1
            if address_risk_contribution > 0.03:
                agreement_count += 1
            
            if agreement_count >= 3:
                confidence = min(confidence + 0.15, 0.95)
            elif agreement_count >= 2:
                confidence = min(confidence + 0.1, 0.95)
            
            risk_score_raw = min(fused_score, 1.0)
        else:
            # No BERT available - use indicator engine + other signals
            risk_score_raw = risk_score_raw + price_risk_contribution + address_risk_contribution
            
            # Apply critical indicator floor even without BERT
            if critical_count >= 3:
                risk_score_raw = max(risk_score_raw, 0.75)
            elif critical_count >= 2:
                risk_score_raw = max(risk_score_raw, 0.60)
            elif critical_count >= 1:
                risk_score_raw = max(risk_score_raw, 0.45)
            
            risk_score_raw = min(risk_score_raw, 1.0)
        
        # Convert indicators to dict format for storage/API
        indicators_dict = [ind.to_dict() for ind in indicators]
        
        # Get risk level - dynamically determined using risk score, confidence,
        # and cumulative severity of independent fraud indicators
        risk_level_enum = IndicatorEngine.get_risk_level(
            risk_score=risk_score_raw,
            confidence=confidence,
            indicators=indicators
        )
        risk_level = risk_level_enum.value  # String value
        
        # Generate risk story (dynamic narrative from signals)
        # Note: generate_risk_story now computes the risk level internally
        risk_story = IndicatorEngine.generate_risk_story(
            risk_score_raw, indicators, confidence=confidence
        )
        
        # Convert to 0-100 scale for enterprise output
        risk_score_100 = int(round(risk_score_raw * 100))
        
        # Save analysis to database
        model_version = "bert_v1" if (bert_prediction is not None) else "indicator_engine_v1"
        
        analysis = RiskAnalysisModel(
            user_id=user_id,
            listing_text=listing_text[:1000],  # Truncate for storage
            risk_score=risk_score_raw,  # Store as 0-1 internally
            risk_level=risk_level,
            risk_indicators=indicators_dict,
            risk_story=risk_story,
            confidence=confidence,
            model_version=model_version
        )
        
        db.add(analysis)
        
        # Decrease user's scan count
        user.scans_remaining -= 1
        
        await db.commit()
        await db.refresh(analysis)
        
        # =====================================================================
        # OUTPUT FORMAT
        # =====================================================================
        return {
            # Core identification
            "id": analysis.id,
            
            # Risk assessment
            "risk_score": risk_score_100,  # 0-100 scale
            "risk_score_normalized": round(risk_score_raw, 3),  # 0-1 scale
            "risk_level": risk_level,
            "confidence": round(confidence, 2),
            
            # Detailed indicators with codes
            "indicators": indicators_dict,
            "indicator_count": len(indicators),
            "indicator_summary": {
                "critical": sum(1 for i in indicators if i.severity >= 4),
                "moderate": sum(1 for i in indicators if 2 <= i.severity < 4),
                "low": sum(1 for i in indicators if i.severity < 2),
            },
            
            # Human-readable explanation
            "risk_story": risk_story,
            
            # Metadata
            "scans_remaining": user.scans_remaining,
            "model_version": model_version,
            "analysis_timestamp": analysis.created_at.isoformat(),
            
            # BERT AI Model (the REAL AI for text analysis)
            "bert_enabled": BERT_AVAILABLE and bert_prediction is not None,
            "bert_prediction": bert_prediction if bert_prediction else None,
            
            # Price Anomaly Analysis (AI for market comparison)
            "price_analysis_enabled": PRICE_ANOMALY_AVAILABLE and price_analysis is not None,
            "price_analysis": price_analysis.to_dict() if price_analysis else None,
            
            # Address Validation (AI for location verification)
            "address_validation_enabled": ADDRESS_VALIDATION_AVAILABLE and address_validation is not None,
            "address_validation": address_validation.to_dict() if address_validation else None,
            
            # AI Components Summary
            "ai_components": {
                "bert_nlp": BERT_AVAILABLE and bert_prediction is not None,
                "price_anomaly": PRICE_ANOMALY_AVAILABLE and price_analysis is not None,
                "address_validation": ADDRESS_VALIDATION_AVAILABLE and address_validation is not None,
                "indicator_engine": True  # Always enabled
            }
        }
    
    # =========================================================================
    # LEGACY CODE REMOVED:
    # - _ml_model_predict: Used IsolationForest on character counts (not real AI)
    # - _calculate_risk_level: Replaced by IndicatorEngine.get_risk_level()
    # - _generate_risk_story: Replaced by IndicatorEngine.generate_risk_story()
    # 
    # The real AI is now in bert_fraud_classifier.py
    # =========================================================================
    
    @staticmethod
    async def get_user_analysis_history(
        db: AsyncSession,
        user_id: int,
        skip: int = 0,
        limit: int = 50
    ) -> List[RiskAnalysisModel]:
        """Get user's analysis history"""
        
        result = await db.execute(
            select(RiskAnalysisModel)
            .where(RiskAnalysisModel.user_id == user_id)
            .offset(skip)
            .limit(limit)
            .order_by(RiskAnalysisModel.created_at.desc())
        )
        
        return result.scalars().all()
    
    @staticmethod
    async def get_analysis(db: AsyncSession, analysis_id: int) -> RiskAnalysisModel:
        """Get specific analysis"""
        
        result = await db.execute(
            select(RiskAnalysisModel).where(RiskAnalysisModel.id == analysis_id)
        )
        
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_explanation(
        db: AsyncSession,
        analysis_id: int,
        user_id: int
    ) -> Dict[str, Any]:
        """
        Get AI explainability report for an analysis.
        
        Provides REAL XAI explanations using:
        - Integrated Gradients for token attribution
        - Attention weights from BERT
        - SHAP values (if available)
        
        Plus rule-based explanations for indicator breakdown.
        """
        from application.use_cases.indicator_engine import Indicator, IndicatorCategory
        
        # Get the analysis
        result = await db.execute(
            select(RiskAnalysisModel).where(RiskAnalysisModel.id == analysis_id)
        )
        analysis = result.scalar_one_or_none()
        
        if not analysis:
            raise ValueError("Analysis not found")
        
        if analysis.user_id != user_id:
            raise ValueError("Access denied")
        
        # Reconstruct indicators from stored data
        indicators = []
        if analysis.risk_indicators:
            for ind_data in analysis.risk_indicators:
                indicators.append(Indicator(
                    code=ind_data.get("code", "UNKNOWN"),
                    category=IndicatorCategory(ind_data.get("category", "content")),
                    severity=ind_data.get("severity", 3) if isinstance(ind_data.get("severity"), int) else 3,
                    evidence=ind_data.get("evidence", []),
                    description=ind_data.get("description", ""),
                    impact_score=ind_data.get("impact_score", 0.1)
                ))
        
        # Try to get REAL XAI explanation first
        real_xai_explanation = None
        try:
            from application.use_cases.real_xai_engine import get_xai_explanation
            
            pred_label = "fraud" if analysis.risk_score > 0.5 else "safe"
            confidence = analysis.risk_score if pred_label == "fraud" else (1.0 - analysis.risk_score)
            
            real_xai_explanation = get_xai_explanation(
                text=analysis.listing_text,
                prediction=pred_label,
                confidence=confidence,
                method="combined"
            )
        except Exception as e:
            logger.warning(f"Real XAI failed, falling back to rule-based: {e}")
        
        # Get rule-based explanation as backup/complement
        from application.use_cases.explainability_engine import explainability_engine
        rule_explanation = explainability_engine.explain(
            indicators=indicators,
            risk_score=analysis.risk_score,  # Already 0-1
            confidence=analysis.confidence,
            text=analysis.listing_text,
            price=analysis.listing_price
        )
        
        # Combine both explanations
        combined_explanation = rule_explanation.to_dict()
        
        if real_xai_explanation:
            combined_explanation["real_xai"] = {
                "is_real_ai": True,
                "methods_used": real_xai_explanation.get("methods_used", []),
                "token_level_explanation": real_xai_explanation.get("token_level_explanation", {}),
                "reasoning_chain": real_xai_explanation.get("reasoning_chain", [])
            }
        else:
            combined_explanation["real_xai"] = {
                "is_real_ai": False,
                "fallback_reason": "Real XAI engine not available, using rule-based explanations"
            }
        
        return combined_explanation
    
    @staticmethod
    async def crawl_url(url: str, timeout: int = 10) -> Dict[str, Any]:
        """Crawl a URL and extract listing text content"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    if response.status != 200:
                        raise ValueError(f"Failed to fetch URL: HTTP {response.status}")
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove script and style elements
                    for element in soup(["script", "style", "nav", "header", "footer"]):
                        element.decompose()
                    
                    # Extract title
                    title = soup.find('title')
                    title_text = title.get_text().strip() if title else ""
                    
                    # Extract meta description
                    meta_desc = soup.find('meta', attrs={'name': 'description'})
                    description = meta_desc.get('content', '') if meta_desc else ""
                    
                    # Extract main content - prioritize common listing selectors
                    main_content = ""
                    
                    # Common selectors for rental listing content
                    selectors = [
                        '.listing-description', '.property-description',
                        '.description', '#description', '.details',
                        '.listing-details', '.property-details',
                        'article', 'main', '.content', '#content'
                    ]
                    
                    for selector in selectors:
                        element = soup.select_one(selector)
                        if element:
                            main_content = element.get_text(separator=' ', strip=True)
                            break
                    
                    # Fallback to body text if no specific content found
                    if not main_content:
                        body = soup.find('body')
                        if body:
                            main_content = body.get_text(separator=' ', strip=True)
                    
                    # Clean up whitespace
                    main_content = ' '.join(main_content.split())
                    
                    # Extract price if present
                    price = None
                    price_patterns = [
                        r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                        r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:per\s+month|/month|monthly)',
                        r'rent[:\s]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
                    ]
                    
                    for pattern in price_patterns:
                        match = re.search(pattern, main_content, re.IGNORECASE)
                        if match:
                            price_str = match.group(1).replace(',', '')
                            try:
                                price = float(price_str)
                                break
                            except ValueError:
                                continue
                    
                    # Combine title, description, and content
                    full_text = f"{title_text}\n{description}\n{main_content}"
                    
                    # Truncate if too long
                    if len(full_text) > 10000:
                        full_text = full_text[:10000]
                    
                    return {
                        "title": title_text,
                        "description": description,
                        "content": main_content,
                        "full_text": full_text,
                        "extracted_price": price,
                        "url": url
                    }
                    
        except asyncio.TimeoutError:
            raise ValueError(f"URL crawl timed out after {timeout} seconds")
        except aiohttp.ClientError as e:
            raise ValueError(f"Failed to crawl URL: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error processing URL: {str(e)}")
    
    @staticmethod
    async def analyze_url(
        db: AsyncSession,
        user_id: int,
        url: str,
        listing_price: float = None,
        location: str = None
    ) -> Dict[str, Any]:
        """Analyze a rental listing URL for fraud indicators"""
        
        # Crawl the URL
        crawl_result = await FraudDetectionUseCases.crawl_url(url)
        
        # Use extracted price if not provided
        if listing_price is None and crawl_result.get("extracted_price"):
            listing_price = crawl_result["extracted_price"]
        
        # Analyze the extracted text
        result = await FraudDetectionUseCases.analyze_listing(
            db=db,
            user_id=user_id,
            listing_text=crawl_result["full_text"],
            listing_price=listing_price,
            location=location
        )
        
        # Add URL-specific information
        result["source_url"] = url
        result["extracted_title"] = crawl_result["title"]
        
        return result
    
    @staticmethod
    async def submit_feedback(
        db: AsyncSession,
        user_id: int,
        analysis_id: int,
        feedback_type: str,
        comments: str = None
    ) -> FeedbackModel:
        """Submit feedback for an analysis"""
        
        # Verify analysis exists and belongs to user
        result = await db.execute(
            select(RiskAnalysisModel).where(RiskAnalysisModel.id == analysis_id)
        )
        analysis = result.scalar_one_or_none()
        
        if not analysis:
            raise ValueError("Analysis not found")
        
        if analysis.user_id != user_id:
            raise ValueError("Access denied")
        
        # Check for existing feedback
        existing = await db.execute(
            select(FeedbackModel).where(
                FeedbackModel.analysis_id == analysis_id,
                FeedbackModel.user_id == user_id
            )
        )
        if existing.scalar_one_or_none():
            raise ValueError("Feedback already submitted for this analysis")
        
        # Create feedback record
        feedback = FeedbackModel(
            analysis_id=analysis_id,
            user_id=user_id,
            feedback_type=feedback_type,
            comments=comments
        )
        
        db.add(feedback)
        await db.commit()
        await db.refresh(feedback)
        
        return feedback
    
    @staticmethod
    async def get_user_feedback(
        db: AsyncSession,
        user_id: int,
        skip: int = 0,
        limit: int = 50
    ) -> List[FeedbackModel]:
        """Get user's feedback history"""
        
        result = await db.execute(
            select(FeedbackModel)
            .where(FeedbackModel.user_id == user_id)
            .offset(skip)
            .limit(limit)
            .order_by(FeedbackModel.created_at.desc())
        )
        
        return result.scalars().all()
    
    @staticmethod
    async def generate_report(
        db: AsyncSession,
        analysis_id: int,
        user_id: int,
        format: str = "html"
    ) -> Dict[str, Any]:
        """Generate a comprehensive risk analysis report"""
        
        # Get analysis
        result = await db.execute(
            select(RiskAnalysisModel).where(RiskAnalysisModel.id == analysis_id)
        )
        analysis = result.scalar_one_or_none()
        
        if not analysis:
            raise ValueError("Analysis not found")
        
        if analysis.user_id != user_id:
            raise ValueError("Access denied")
        
        # Get feedback if exists
        feedback_result = await db.execute(
            select(FeedbackModel).where(
                FeedbackModel.analysis_id == analysis_id,
                FeedbackModel.user_id == user_id
            )
        )
        feedback = feedback_result.scalar_one_or_none()
        
        # Generate report content
        report_data = {
            "report_title": "Rental Fraud Risk Analysis Report",
            "generated_at": datetime.utcnow().isoformat(),
            "analysis_id": analysis.id,
            "analysis_date": analysis.created_at.isoformat(),
            "risk_score": analysis.risk_score,
            "risk_level": analysis.risk_level,
            "confidence": analysis.confidence,
            "model_version": analysis.model_version,
            "listing_text": analysis.listing_text,
            "risk_indicators": analysis.risk_indicators or [],
            "risk_story": analysis.risk_story,
            "feedback": {
                "type": feedback.feedback_type,
                "comments": feedback.comments,
                "submitted_at": feedback.created_at.isoformat()
            } if feedback else None,
            "recommendations": FraudDetectionUseCases._generate_recommendations(
                analysis.risk_level,
                analysis.risk_indicators or []
            )
        }
        
        if format == "html":
            html_content = FraudDetectionUseCases._generate_html_report(report_data)
            return {
                "format": "html",
                "content": html_content,
                "filename": f"fraud_analysis_report_{analysis_id}.html"
            }
        else:
            # For PDF, generate actual PDF bytes using WeasyPrint
            html_content = FraudDetectionUseCases._generate_html_report(report_data)
            pdf_bytes = HTML(string=html_content).write_pdf()
            pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
            return {
                "format": "pdf",
                "content": pdf_base64,
                "filename": f"fraud_analysis_report_{analysis_id}.pdf"
            }
    
    @staticmethod
    def _generate_recommendations(risk_level: str, indicators: List[Dict]) -> List[str]:
        """Generate recommendations based on risk level and indicators"""
        
        recommendations = []
        
        if risk_level in ["very_high", "high"]:
            recommendations.extend([
                "DO NOT send any money or deposits without viewing the property in person",
                "Verify the landlord's identity through official channels (property records, ID verification)",
                "Request to see the property before signing any agreement",
                "Use secure payment methods with buyer protection (avoid wire transfers, gift cards)",
                "Report this listing to the platform if you believe it is fraudulent",
                "Contact local authorities if you have been a victim of fraud"
            ])
        elif risk_level == "medium":
            recommendations.extend([
                "Exercise caution and verify all claims before proceeding",
                "Request to view the property in person before making any commitment",
                "Verify the landlord's identity and ownership of the property",
                "Use secure payment methods and keep records of all transactions",
                "Read reviews and check references before signing a lease"
            ])
        else:
            recommendations.extend([
                "Verify the property details and landlord information",
                "View the property in person before signing any agreement",
                "Use standard rental agreements and read all terms carefully",
                "Document the condition of the property before moving in"
            ])
        
        # Add indicator-specific recommendations (handle both old and new field names)
        indicator_categories = [i.get("category") or i.get("type") for i in indicators]
        
        if "payment" in indicator_categories or "payment_method" in indicator_categories:
            if "Never send money via wire transfer, gift cards, or cryptocurrency" not in recommendations:
                recommendations.append("Never send money via wire transfer, gift cards, or cryptocurrency")
        
        if "urgency" in indicator_categories:
            if "Be wary of high-pressure tactics - legitimate landlords allow time for decision-making" not in recommendations:
                recommendations.append("Be wary of high-pressure tactics - legitimate landlords allow time for decision-making")
        
        if "contact" in indicator_categories or "contact_method" in indicator_categories:
            if "Request multiple contact methods and verify the landlord's phone number" not in recommendations:
                recommendations.append("Request multiple contact methods and verify the landlord's phone number")
        
        return recommendations
    
    @staticmethod
    def _generate_html_report(data: Dict[str, Any]) -> str:
        """Generate HTML report content"""
        
        risk_color = {
            "very_low": "#10b981",
            "low": "#3b82f6",
            "medium": "#f59e0b",
            "high": "#f97316",
            "very_high": "#ef4444"
        }.get(data["risk_level"], "#6b7280")
        
        indicators_html = ""
        for indicator in data["risk_indicators"]:
            # Handle both old (string) and new (1-5 number) severity formats
            severity_raw = indicator.get("severity", 1)
            if isinstance(severity_raw, int):
                # New format: 1-5 number
                severity_label = "High" if severity_raw >= 4 else "Medium" if severity_raw >= 2 else "Low"
                severity_color = "#ef4444" if severity_raw >= 4 else "#f59e0b" if severity_raw >= 2 else "#3b82f6"
            else:
                # Old format: string
                severity_label = str(severity_raw).capitalize()
                severity_color = {
                    "high": "#ef4444",
                    "medium": "#f59e0b",
                    "low": "#3b82f6"
                }.get(str(severity_raw).lower(), "#6b7280")
            
            # Handle both old and new field names for evidence/examples
            evidence_list = indicator.get("evidence") or indicator.get("examples") or []
            evidence_html = ""
            if evidence_list:
                evidence_html = "<ul style='margin: 8px 0; padding-left: 20px;'>" + \
                    "".join(f"<li style='color: #666;'>{ex}</li>" for ex in evidence_list) + \
                    "</ul>"
            
            # Handle both old and new field names for category/type
            category = indicator.get('category_display') or indicator.get('category') or indicator.get('type') or 'unknown'
            
            # Handle both old and new field names for impact
            impact = indicator.get('impact_score') or indicator.get('impact') or 0
            
            # Use user-friendly description if available
            description = indicator.get('description_friendly') or indicator.get('description') or ''
            
            indicators_html += f"""
            <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div>
                        <span style="background-color: {severity_color}20; color: {severity_color}; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600;">
                            {severity_label.upper()}
                        </span>
                        <span style="background-color: #f3f4f6; color: #374151; padding: 4px 12px; border-radius: 12px; font-size: 12px; margin-left: 8px;">
                            {str(category).replace('_', ' ').title()}
                        </span>
                    </div>
                    <span style="color: {severity_color}; font-weight: 700; font-size: 18px;">
                        +{int(impact * 100)}%
                    </span>
                </div>
                <p style="margin: 12px 0 0 0; color: #1f2937; font-weight: 500;">{description}</p>
                {evidence_html}
            </div>
            """
        
        recommendations_html = "<ul style='margin: 0; padding-left: 20px;'>" + \
            "".join(f"<li style='margin-bottom: 8px; color: #374151;'>{rec}</li>" for rec in data["recommendations"]) + \
            "</ul>"
        
        feedback_html = ""
        if data.get("feedback"):
            feedback_html = f"""
            <div style="background-color: #f0fdf4; border: 1px solid #86efac; border-radius: 8px; padding: 16px; margin-top: 24px;">
                <h3 style="margin: 0 0 12px 0; color: #166534;">User Feedback</h3>
                <p><strong>Outcome:</strong> {data['feedback']['type'].upper()}</p>
                {f"<p><strong>Comments:</strong> {data['feedback']['comments']}</p>" if data['feedback'].get('comments') else ""}
                <p style="color: #666; font-size: 12px;">Submitted: {data['feedback']['submitted_at']}</p>
            </div>
            """
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{data['report_title']}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif; line-height: 1.6; color: #1f2937; background-color: #f9fafb; }}
        .container {{ max-width: 800px; margin: 0 auto; padding: 24px; }}
        .header {{ background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%); color: white; padding: 32px; border-radius: 12px; margin-bottom: 24px; }}
        .card {{ background: white; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .risk-score {{ display: flex; align-items: center; justify-content: space-between; }}
        .score-circle {{ width: 120px; height: 120px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 36px; font-weight: 700; color: white; }}
        h1 {{ margin-bottom: 8px; }}
        h2 {{ color: #374151; margin-bottom: 16px; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; }}
        .meta {{ font-size: 14px; color: rgba(255,255,255,0.8); }}
        .listing-text {{ background-color: #f9fafb; padding: 16px; border-radius: 8px; white-space: pre-wrap; word-wrap: break-word; font-size: 14px; color: #4b5563; max-height: 300px; overflow-y: auto; }}
        .risk-story {{ background-color: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 16px; white-space: pre-wrap; }}
        .footer {{ text-align: center; color: #9ca3af; font-size: 12px; margin-top: 32px; padding-top: 16px; border-top: 1px solid #e5e7eb; }}
        @media print {{ body {{ background: white; }} .container {{ padding: 0; }} }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ {data['report_title']}</h1>
            <p class="meta">Generated: {data['generated_at']} | Analysis ID: {data['analysis_id']}</p>
        </div>
        
        <div class="card">
            <div class="risk-score">
                <div>
                    <h2 style="border: none; padding: 0;">Risk Assessment</h2>
                    <p style="font-size: 24px; font-weight: 700; color: {risk_color}; text-transform: uppercase;">
                        {data['risk_level'].replace('_', ' ')}
                    </p>
                    <p style="color: #6b7280;">Confidence: {int(data['confidence'] * 100)}%</p>
                    <p style="color: #9ca3af; font-size: 12px; margin-top: 8px;">Model: {data['model_version']}</p>
                </div>
                <div class="score-circle" style="background-color: {risk_color};">
                    {int(data['risk_score'] * 100)}%
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>📝 Analyzed Listing</h2>
            <div class="listing-text">{data['listing_text']}</div>
        </div>
        
        <div class="card">
            <h2>📊 Analysis Summary</h2>
            <div class="risk-story">{data['risk_story']}</div>
        </div>
        
        <div class="card">
            <h2>⚠️ Risk Indicators ({len(data['risk_indicators'])})</h2>
            {indicators_html if indicators_html else '<p style="color: #6b7280;">No specific risk indicators detected.</p>'}
        </div>
        
        <div class="card">
            <h2>💡 Recommendations</h2>
            {recommendations_html}
        </div>
        
        {feedback_html}
        
        <div class="footer">
            <p>AI-Powered Rental Fraud & Trust Scoring System</p>
            <p>This report is generated for informational purposes only and should not be considered legal advice.</p>
            <p>Analysis Date: {data['analysis_date']}</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html

