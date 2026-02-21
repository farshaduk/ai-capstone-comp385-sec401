from fastapi import APIRouter, Depends, HTTPException, Request, File, Form, UploadFile, Query
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from infrastructure.database import get_db, SubscriptionPlanModel, PaymentModel, UserModel
from sqlalchemy import select
from presentation.schemas import (
    RiskAnalysisRequest, RiskAnalysisResponse, URLAnalysisRequest,
    AnalysisHistoryResponse, SubscriptionPlanResponse, UserResponse,
    FeedbackRequest, FeedbackResponse, ReportExportRequest,
    PaymentRequest, PaymentResponse
)
from presentation.dependencies import get_current_user, get_client_ip, require_feature, check_feature_access
from application.use_cases.fraud_detection_use_cases import FraudDetectionUseCases
from application.use_cases.user_use_cases import UserUseCases
import uuid
from datetime import datetime

router = APIRouter(prefix="/renter", tags=["Renter"])


@router.post("/analyze", response_model=RiskAnalysisResponse)
async def analyze_listing(
    data: RiskAnalysisRequest,
    request: Request,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Analyze a rental listing for fraud risk"""
    
    try:
        result = await FraudDetectionUseCases.analyze_listing(
            db=db,
            user_id=current_user.id,
            listing_text=data.listing_text,
            listing_price=data.listing_price,
            location=data.location
        )
        
        # Log action
        await UserUseCases.log_action(
            db=db,
            user_id=current_user.id,
            action="listing_analyzed",
            entity_type="risk_analysis",
            entity_id=result["id"],
            details={
                "risk_score": result["risk_score"],
                "risk_level": result["risk_level"]
            },
            ip_address=get_client_ip(request)
        )
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/history", response_model=List[AnalysisHistoryResponse])
async def get_analysis_history(
    skip: int = 0,
    limit: int = 50,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's analysis history"""
    
    analyses = await FraudDetectionUseCases.get_user_analysis_history(
        db=db,
        user_id=current_user.id,
        skip=skip,
        limit=limit
    )
    
    return analyses


@router.get("/history/{analysis_id}")
async def get_analysis_detail(
    analysis_id: int,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get specific analysis details"""
    
    analysis = await FraudDetectionUseCases.get_analysis(db, analysis_id)
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Ensure user owns this analysis
    if analysis.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "id": analysis.id,
        "listing_text": analysis.listing_text,
        "risk_score": analysis.risk_score,
        "risk_level": analysis.risk_level,
        "risk_indicators": analysis.risk_indicators,
        "risk_story": analysis.risk_story,
        "confidence": analysis.confidence,
        "model_version": analysis.model_version,
        "created_at": analysis.created_at.isoformat()
    }


@router.get("/subscription/plans", response_model=List[SubscriptionPlanResponse])
async def get_subscription_plans(
    db: AsyncSession = Depends(get_db)
):
    """Get available subscription plans"""
    
    result = await db.execute(
        select(SubscriptionPlanModel).where(SubscriptionPlanModel.is_active == True)
    )
    
    plans = result.scalars().all()
    return plans


@router.post("/subscription/upgrade", response_model=UserResponse)
async def upgrade_subscription(
    plan_name: str,
    request: Request,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Upgrade user subscription"""
    
    # Validate plan exists
    result = await db.execute(
        select(SubscriptionPlanModel).where(
            SubscriptionPlanModel.name == plan_name,
            SubscriptionPlanModel.is_active == True
        )
    )
    
    plan = result.scalar_one_or_none()
    
    if not plan:
        raise HTTPException(status_code=404, detail="Subscription plan not found")
    
    try:
        user = await UserUseCases.update_subscription(
            db=db,
            user_id=current_user.id,
            new_plan=plan_name
        )
        
        # Log action
        await UserUseCases.log_action(
            db=db,
            user_id=current_user.id,
            action="subscription_upgraded",
            entity_type="user",
            entity_id=current_user.id,
            details={"new_plan": plan_name},
            ip_address=get_client_ip(request)
        )
        
        return user
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/subscription/payment", response_model=PaymentResponse)
async def process_payment(
    data: PaymentRequest,
    request: Request,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Process payment for subscription upgrade (simulation).
    
    This is a simulated payment processor for testing purposes.
    In production, this would integrate with a real payment gateway.
    """
    
    # Validate plan exists
    result = await db.execute(
        select(SubscriptionPlanModel).where(
            SubscriptionPlanModel.name == data.plan_name,
            SubscriptionPlanModel.is_active == True
        )
    )
    
    plan = result.scalar_one_or_none()
    
    if not plan:
        raise HTTPException(status_code=404, detail="Subscription plan not found")
    
    # Validate card number (simulation - accept any 16 digit number)
    card_number = data.card_number.replace(" ", "").replace("-", "")
    if len(card_number) != 16 or not card_number.isdigit():
        raise HTTPException(status_code=400, detail="Invalid card number")
    
    # Validate CVV
    if len(data.cvv) not in [3, 4] or not data.cvv.isdigit():
        raise HTTPException(status_code=400, detail="Invalid CVV")
    
    # Validate expiry
    current_year = datetime.now().year % 100
    current_month = datetime.now().month
    if data.expiry_year < current_year or (data.expiry_year == current_year and data.expiry_month < current_month):
        raise HTTPException(status_code=400, detail="Card has expired")
    
    # Simulate payment processing (always succeeds in simulation)
    # In production: integrate with Stripe, PayPal, etc.
    transaction_id = f"TXN_{uuid.uuid4().hex[:12].upper()}"
    
    # Create payment record
    payment = PaymentModel(
        user_id=current_user.id,
        transaction_id=transaction_id,
        plan_name=data.plan_name,
        amount=plan.price,
        status="completed",
        card_last_four=card_number[-4:],
        cardholder_name=data.cardholder_name
    )
    
    db.add(payment)
    
    # Update user subscription
    user_result = await db.execute(
        select(UserModel).where(UserModel.id == current_user.id)
    )
    user = user_result.scalar_one()
    
    user.subscription_plan = data.plan_name
    user.scans_remaining = plan.scans_per_month
    
    await db.commit()
    
    # Log action
    await UserUseCases.log_action(
        db=db,
        user_id=current_user.id,
        action="payment_processed",
        entity_type="payment",
        entity_id=payment.id,
        details={
            "plan_name": data.plan_name,
            "amount": plan.price,
            "transaction_id": transaction_id
        },
        ip_address=get_client_ip(request)
    )
    
    return PaymentResponse(
        success=True,
        transaction_id=transaction_id,
        message=f"Payment successful! You are now on the {plan.display_name}.",
        plan_name=data.plan_name,
        amount=plan.price
    )


@router.get("/subscription/payments")
async def get_payment_history(
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's payment history"""
    
    result = await db.execute(
        select(PaymentModel)
        .where(PaymentModel.user_id == current_user.id)
        .order_by(PaymentModel.created_at.desc())
    )
    
    payments = result.scalars().all()
    
    return [
        {
            "id": p.id,
            "transaction_id": p.transaction_id,
            "plan_name": p.plan_name,
            "amount": p.amount,
            "status": p.status,
            "card_last_four": p.card_last_four,
            "created_at": p.created_at.isoformat()
        }
        for p in payments
    ]


@router.get("/subscription/current")
async def get_current_subscription(
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current subscription details with features"""
    
    result = await db.execute(
        select(SubscriptionPlanModel).where(
            SubscriptionPlanModel.name == current_user.subscription_plan
        )
    )
    
    plan = result.scalar_one_or_none()
    
    if not plan:
        # Return default free plan features
        return {
            "plan_name": current_user.subscription_plan,
            "display_name": "Free Plan",
            "features": {
                "basic_analysis": True,
                "risk_score": True,
                "history": 1000,
                "support": "community"
            },
            "scans_remaining": current_user.scans_remaining,
            "scans_per_month": 1000
        }
    
    return {
        "plan_name": plan.name,
        "display_name": plan.display_name,
        "features": plan.features,
        "scans_remaining": current_user.scans_remaining,
        "scans_per_month": plan.scans_per_month,
        "price": plan.price
    }


@router.get("/stats")
async def get_user_stats(
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user statistics"""
    
    from sqlalchemy import func
    from infrastructure.database import RiskAnalysisModel
    
    # Total analyses
    total = await db.scalar(
        select(func.count(RiskAnalysisModel.id)).where(
            RiskAnalysisModel.user_id == current_user.id
        )
    )
    
    # High risk analyses (fraud detected)
    high_risk = await db.scalar(
        select(func.count(RiskAnalysisModel.id)).where(
            RiskAnalysisModel.user_id == current_user.id,
            RiskAnalysisModel.risk_level.in_(["high", "very_high"])
        )
    )
    
    # Safe analyses (low/very_low risk)
    safe = await db.scalar(
        select(func.count(RiskAnalysisModel.id)).where(
            RiskAnalysisModel.user_id == current_user.id,
            RiskAnalysisModel.risk_level.in_(["low", "very_low"])
        )
    )
    
    # Medium risk analyses
    medium_risk = await db.scalar(
        select(func.count(RiskAnalysisModel.id)).where(
            RiskAnalysisModel.user_id == current_user.id,
            RiskAnalysisModel.risk_level == "medium"
        )
    )
    
    # Average risk score
    avg_score = await db.scalar(
        select(func.avg(RiskAnalysisModel.risk_score)).where(
            RiskAnalysisModel.user_id == current_user.id
        )
    )
    
    return {
        "total_analyses": total or 0,
        "safe_count": safe or 0,
        "fraud_count": high_risk or 0,
        "medium_count": medium_risk or 0,
        "high_risk_count": high_risk or 0,
        "average_risk_score": float(avg_score) if avg_score else 0.0,
        "scans_remaining": current_user.scans_remaining,
        "subscription_plan": current_user.subscription_plan
    }


@router.post("/analyze-url", response_model=RiskAnalysisResponse)
async def analyze_url(
    data: URLAnalysisRequest,
    request: Request,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Analyze a rental listing URL for fraud risk (FR10, FR11)"""
    
    try:
        result = await FraudDetectionUseCases.analyze_url(
            db=db,
            user_id=current_user.id,
            url=data.url,
            listing_price=data.listing_price,
            location=data.location
        )
        
        # Log action
        await UserUseCases.log_action(
            db=db,
            user_id=current_user.id,
            action="url_analyzed",
            entity_type="risk_analysis",
            entity_id=result["id"],
            details={
                "url": data.url,
                "risk_score": result["risk_score"],
                "risk_level": result["risk_level"]
            },
            ip_address=get_client_ip(request)
        )
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL analysis failed: {str(e)}")


@router.post("/analyze-images")
async def analyze_images(
    request: Request,
    image_urls: List[str] = Query(..., alias="image_urls[]"),
    listing_id: str = None,
    current_user = Depends(require_feature("image_analysis")),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze listing images for authenticity.
    
    Requires: Premium or Enterprise subscription (image_analysis feature)
    
    Detects:
    - AI-generated images (DALL-E, Midjourney, etc.)
    - Stolen/duplicate images from other listings
    - Stock photos
    - Manipulated images
    """
    from application.use_cases.real_image_engine import real_image_engine as image_analysis_engine
    import httpx
    
    if not image_urls:
        raise HTTPException(status_code=400, detail="No image URLs provided")
    
    if len(image_urls) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per analysis")
    
    try:
        # Download images from URLs
        images = []
        async with httpx.AsyncClient(timeout=15.0) as client:
            for url in image_urls:
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    filename = url.split('/')[-1].split('?')[0] or 'image'
                    images.append((resp.content, filename))
                except Exception as dl_err:
                    logger.warning(f"Failed to download image {url}: {dl_err}")
        
        if not images:
            raise HTTPException(status_code=400, detail="Could not download any images")
        
        result = await image_analysis_engine.analyze_multiple_images(images)
        
        # Log action
        await UserUseCases.log_action(
            db=db,
            user_id=current_user.id,
            action="images_analyzed",
            entity_type="image_analysis",
            entity_id=None,
            details={
                "image_count": len(image_urls),
                "risk_level": result.get("overall_risk_level"),
                "risk_score": result.get("overall_risk_score")
            },
            ip_address=get_client_ip(request)
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")


@router.post("/analyze-uploaded-images")
async def analyze_uploaded_images(
    request: Request,
    files: List[UploadFile] = File(...),
    listing_id: str = Form(None),
    current_user = Depends(require_feature("image_analysis")),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze uploaded listing images for authenticity.
    
    Requires: Premium or Enterprise subscription (image_analysis feature)
    
    Accepts uploaded image files (JPEG, PNG, GIF, WebP).
    Maximum 10 images, 10MB each.
    """
    from application.use_cases.real_image_engine import real_image_engine as image_analysis_engine
    import base64
    
    if not files:
        raise HTTPException(status_code=400, detail="No images uploaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per analysis")
    
    # Validate and read files
    allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    image_data_list = []
    
    for file in files:
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type: {file.filename}. Allowed: JPEG, PNG, GIF, WebP"
            )
        
        # Read file content
        content = await file.read()
        
        if len(content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(
                status_code=400,
                detail=f"File too large: {file.filename}. Maximum 10MB per image"
            )
        
        # Create data URI for the image
        base64_content = base64.b64encode(content).decode('utf-8')
        data_uri = f"data:{file.content_type};base64,{base64_content}"
        
        image_data_list.append({
            "data_uri": data_uri,
            "filename": file.filename,
            "content_type": file.content_type,
            "raw_bytes": content
        })
    
    try:
        # Build (bytes, filename) tuples for the engine
        images = [(item["raw_bytes"], item["filename"]) for item in image_data_list]
        
        result = await image_analysis_engine.analyze_multiple_images(images)
        
        # Log action
        await UserUseCases.log_action(
            db=db,
            user_id=current_user.id,
            action="uploaded_images_analyzed",
            entity_type="image_analysis",
            entity_id=None,
            details={
                "image_count": len(files),
                "filenames": [f.filename for f in files],
                "risk_level": result.get("overall_risk_level"),
                "risk_score": result.get("overall_risk_score")
            },
            ip_address=get_client_ip(request)
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    data: FeedbackRequest,
    request: Request,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Submit feedback for an analysis (FR4, FR14)"""
    
    try:
        feedback = await FraudDetectionUseCases.submit_feedback(
            db=db,
            user_id=current_user.id,
            analysis_id=data.analysis_id,
            feedback_type=data.feedback_type,
            comments=data.comments
        )
        
        # Log action
        await UserUseCases.log_action(
            db=db,
            user_id=current_user.id,
            action="feedback_submitted",
            entity_type="feedback",
            entity_id=feedback.id,
            details={
                "analysis_id": data.analysis_id,
                "feedback_type": data.feedback_type
            },
            ip_address=get_client_ip(request)
        )
        
        return feedback
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")


@router.get("/feedback")
async def get_feedback_history(
    skip: int = 0,
    limit: int = 50,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's feedback history"""
    
    feedback_list = await FraudDetectionUseCases.get_user_feedback(
        db=db,
        user_id=current_user.id,
        skip=skip,
        limit=limit
    )
    
    return [
        {
            "id": f.id,
            "analysis_id": f.analysis_id,
            "feedback_type": f.feedback_type,
            "comments": f.comments,
            "created_at": f.created_at.isoformat()
        }
        for f in feedback_list
    ]


@router.post("/report/export")
async def export_report(
    data: ReportExportRequest,
    request: Request,
    current_user = Depends(require_feature("export_reports")),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate and export analysis report (FR12)
    
    Requires: Premium or Enterprise subscription (export_reports feature)
    """
    
    try:
        report = await FraudDetectionUseCases.generate_report(
            db=db,
            analysis_id=data.analysis_id,
            user_id=current_user.id,
            format=data.format
        )
        
        # Log action
        await UserUseCases.log_action(
            db=db,
            user_id=current_user.id,
            action="report_exported",
            entity_type="risk_analysis",
            entity_id=data.analysis_id,
            details={"format": data.format},
            ip_address=get_client_ip(request)
        )
        
        return report
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.get("/history/{analysis_id}/explain")
async def get_analysis_explanation(
    analysis_id: int,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get AI explainability report for an analysis (XAI - Explainable AI)"""
    
    try:
        explanation = await FraudDetectionUseCases.get_explanation(
            db=db,
            analysis_id=analysis_id,
            user_id=current_user.id
        )
        
        return explanation
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")


# =====================================================================
# MESSAGE / CONVERSATION ANALYSIS (Required by Proposal FR1)
# =====================================================================

@router.post("/analyze-message")
async def analyze_single_message(
    message: str = Form(...),
    sender: str = Form(default="unknown"),
    request: Request = None,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze a single message for fraud indicators.
    
    Uses BERT-based NLP to detect:
    - Social engineering tactics
    - Payment fraud patterns
    - Urgency manipulation
    - Contact redirect attempts
    
    This is real AI analysis, not keyword matching.
    """
    from application.use_cases.message_analysis_engine import message_analysis_engine
    
    try:
        result = message_analysis_engine.analyze_message(message, sender)
        
        # Log action
        await UserUseCases.log_action(
            db=db,
            user_id=current_user.id,
            action="message_analyzed",
            entity_type="message",
            entity_id=0,
            details={
                "risk_level": result.risk_level.value,
                "risk_score": result.risk_score,
                "tactics_count": len(result.tactics_detected)
            },
            ip_address=get_client_ip(request)
        )
        
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Message analysis failed: {str(e)}")


@router.post("/analyze-conversation")
async def analyze_full_conversation(
    request: Request,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze a full conversation thread for fraud patterns.
    
    Accepts JSON body with format:
    {
        "messages": [
            {"content": "message text", "sender": "landlord/renter", "timestamp": "optional ISO date"},
            ...
        ],
        "conversation_id": "optional ID"
    }
    
    Analyzes:
    - Individual message risk
    - Escalation patterns over conversation
    - Cross-message correlation
    - Social engineering progression
    """
    from application.use_cases.message_analysis_engine import message_analysis_engine
    
    try:
        body = await request.json()
        messages = body.get("messages", [])
        conversation_id = body.get("conversation_id")
        
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        result = message_analysis_engine.analyze_conversation(messages, conversation_id)
        
        # Log action
        await UserUseCases.log_action(
            db=db,
            user_id=current_user.id,
            action="conversation_analyzed",
            entity_type="conversation",
            entity_id=0,
            details={
                "message_count": len(messages),
                "overall_risk": result.overall_risk_level.value,
                "escalation_detected": result.escalation_detected
            },
            ip_address=get_client_ip(request)
        )
        
        return result.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversation analysis failed: {str(e)}")


# =====================================================================
# REAL XAI (Explainable AI) ENDPOINTS
# =====================================================================

@router.post("/explain-text")
async def get_real_xai_explanation(
    text: str = Form(...),
    method: str = Form(default="combined"),
    request: Request = None,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get real AI explainability for text analysis.
    
    Uses actual model introspection:
    - Integrated Gradients for gradient-based attribution
    - Attention weights from transformer layers
    - SHAP values (if available)
    
    Returns token-level attributions showing exactly WHY
    the model made its prediction.
    
    Methods: "integrated_gradients", "attention_weights", "shap", "combined"
    """
    from application.use_cases.real_xai_engine import get_xai_explanation
    from application.use_cases.bert_fraud_classifier import get_fraud_classifier
    
    try:
        # First get prediction from BERT
        classifier = get_fraud_classifier()
        if classifier.is_trained:
            prediction = classifier.predict(text)
            pred_label = "fraud" if prediction['fraud_probability'] > 0.5 else "safe"
            confidence = prediction['fraud_probability'] if pred_label == "fraud" else (1 - prediction['fraud_probability'])
        else:
            pred_label = "unknown"
            confidence = 0.5
        
        # Get XAI explanation
        explanation = get_xai_explanation(text, pred_label, confidence, method)
        
        # Add what-if analysis from indicator + explainability engine
        try:
            from application.use_cases.explainability_engine import explainability_engine
            from application.use_cases.indicator_engine import indicator_engine
            indicators, indicator_risk, conf = indicator_engine.analyze(text=text, price=None)
            report = explainability_engine.explain(
                indicators=indicators, risk_score=indicator_risk, confidence=conf,
                text=text
            )
            report_dict = report.to_dict()
            explanation["what_if_analysis"] = report_dict.get("what_if_analysis", [])
            explanation["top_contributors"] = report_dict.get("top_contributors", [])
            explanation["summary"] = report_dict.get("summary", "")
            explanation["indicator_risk_score"] = indicator_risk
        except Exception:
            explanation["what_if_analysis"] = []
            explanation["top_contributors"] = []
            explanation["summary"] = ""
            indicator_risk = 0.0
        
        # Compute combined verdict — use BERT probability + indicator risk
        bert_fraud_prob = prediction.get('fraud_probability', 0.5) if classifier.is_trained else 0.5
        combined_risk = max(bert_fraud_prob, indicator_risk)
        explanation["is_fraud"] = combined_risk > 0.5
        explanation["risk_score"] = round(combined_risk, 3)
        explanation["bert_verdict"] = pred_label
        explanation["bert_confidence"] = round(confidence, 3)
        
        # Log action
        await UserUseCases.log_action(
            db=db,
            user_id=current_user.id,
            action="xai_explanation_generated",
            entity_type="xai",
            entity_id=0,
            details={"method": method, "prediction": pred_label},
            ip_address=get_client_ip(request)
        )
        
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"XAI explanation failed: {str(e)}")

