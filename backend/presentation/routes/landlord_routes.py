"""
Landlord Routes - API endpoints for landlord features

Provides endpoints for:
1. Document verification (paystubs, IDs, employment letters)
2. Tenant background analysis
3. Application fraud detection
4. Multi-document verification sets
"""

from fastapi import APIRouter, Depends, HTTPException, Request, File, Form, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum
import base64

from infrastructure.database import get_db
from presentation.dependencies import get_current_user, get_client_ip
from application.use_cases.ocr_engine import ocr_engine, DocumentType, DocumentRiskLevel
from application.use_cases.real_image_engine import real_image_engine
from application.use_cases.user_use_cases import UserUseCases


router = APIRouter(prefix="/landlord", tags=["Landlord"])


# Request/Response Schemas
class DocumentTypeEnum(str, Enum):
    """Document types for landlord verification"""
    PAYSTUB = "paystub"
    ID_CARD = "id_card"
    BANK_STATEMENT = "bank_statement"
    RENTAL_APPLICATION = "rental_application"
    EMPLOYMENT_LETTER = "employment_letter"
    TAX_DOCUMENT = "tax_document"
    UTILITY_BILL = "utility_bill"


class DocumentVerificationRequest(BaseModel):
    """Request for document verification"""
    document_base64: str = Field(..., description="Base64 encoded document image")
    document_type: DocumentTypeEnum = Field(..., description="Expected document type")
    applicant_name: Optional[str] = Field(None, description="Applicant name to verify")
    filename: Optional[str] = Field(None, description="Original filename")


class TenantVerificationRequest(BaseModel):
    """Request for full tenant verification"""
    applicant_name: str = Field(..., description="Applicant's full name")
    documents: List[DocumentVerificationRequest] = Field(..., description="Documents to verify")


class PropertyImageRequest(BaseModel):
    """Request for property image verification"""
    image_base64: str = Field(..., description="Base64 encoded property image")
    filename: Optional[str] = Field(None, description="Original filename")


class MultiImageRequest(BaseModel):
    """Request for multiple property images"""
    images: List[PropertyImageRequest] = Field(..., description="List of images")


class DocumentVerificationResponse(BaseModel):
    """Response for document verification"""
    success: bool
    document_type: str
    risk_level: str
    risk_score: float
    extracted_name: Optional[str]
    extracted_employer: Optional[str]
    extracted_amounts: List[float]
    quality_score: float
    consistency_score: float
    indicators: List[dict]
    explanation: str


class TenantVerificationResponse(BaseModel):
    """Response for full tenant verification"""
    success: bool
    applicant_name: str
    document_count: int
    overall_risk_score: float
    overall_risk_level: str
    verified_count: int
    suspicious_count: int
    name_consistent: Optional[bool]
    documents: List[dict]
    summary: str
    recommendation: str
    cross_document_analysis: Optional[dict] = None


# Helper functions
def decode_base64_image(base64_str: str) -> bytes:
    """Decode base64 image data"""
    # Handle data URL format
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    return base64.b64decode(base64_str)


def get_recommendation(risk_score: float, suspicious_count: int, name_consistent: Optional[bool]) -> str:
    """Generate recommendation based on verification results"""
    if risk_score < 0.2 and suspicious_count == 0 and name_consistent != False:
        return "✅ APPROVE - All documents verified successfully. Low fraud risk."
    elif risk_score < 0.4 and suspicious_count == 0:
        return "✅ LIKELY APPROVE - Documents appear genuine. Consider proceeding."
    elif risk_score < 0.6 or suspicious_count == 1:
        return "⚠️ REVIEW - Some concerns detected. Request additional verification."
    elif risk_score < 0.8 or name_consistent == False:
        return "🚨 CAUTION - Multiple issues detected. Conduct thorough manual review."
    else:
        return "❌ REJECT - High fraud risk. Documents appear suspicious or fraudulent."


# API Endpoints
@router.post("/verify-document", response_model=DocumentVerificationResponse)
async def verify_document(
    request_data: DocumentVerificationRequest,
    request: Request,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Verify a single document (paystub, ID, employment letter, etc.)
    
    Uses OCR to extract text and AI to detect fraud patterns.
    """
    try:
        # Decode image
        image_data = decode_base64_image(request_data.document_base64)
        
        # Map enum to internal type
        doc_type = DocumentType(request_data.document_type.value)
        
        # Analyze document
        result = await ocr_engine.analyze_document(
            image_data=image_data,
            expected_type=doc_type,
            applicant_name=request_data.applicant_name
        )
        
        # Log action
        await UserUseCases.log_action(
            db=db,
            user_id=current_user.id,
            action="document_verified",
            entity_type="document_verification",
            entity_id=None,
            details={
                "document_type": request_data.document_type.value,
                "risk_level": result.risk_level.value,
                "risk_score": result.risk_score
            },
            ip_address=get_client_ip(request)
        )
        
        return DocumentVerificationResponse(
            success=True,
            document_type=result.document_type.value,
            risk_level=result.risk_level.value,
            risk_score=result.risk_score,
            extracted_name=result.extracted_data.name,
            extracted_employer=result.extracted_data.employer,
            extracted_amounts=result.extracted_data.amounts,
            quality_score=result.quality_score,
            consistency_score=result.consistency_score,
            indicators=result.indicators,
            explanation=result.explanation
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Document verification failed: {str(e)}"
        )


@router.post("/verify-document-upload")
async def verify_document_upload(
    request: Request,
    document_type: DocumentTypeEnum = Form(...),
    applicant_name: Optional[str] = Form(None),
    file: UploadFile = File(...),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Verify a document via file upload.
    
    Alternative to base64 endpoint for direct file uploads.
    """
    try:
        # Read file
        image_data = await file.read()
        
        # Map enum to internal type
        doc_type = DocumentType(document_type.value)
        
        # Analyze document
        result = await ocr_engine.analyze_document(
            image_data=image_data,
            expected_type=doc_type,
            applicant_name=applicant_name
        )
        
        # Log action
        await UserUseCases.log_action(
            db=db,
            user_id=current_user.id,
            action="document_verified",
            entity_type="document_verification",
            entity_id=None,
            details={
                "document_type": document_type.value,
                "filename": file.filename,
                "risk_level": result.risk_level.value
            },
            ip_address=get_client_ip(request)
        )
        
        return {
            "success": True,
            "filename": file.filename,
            "result": result.to_dict()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Document verification failed: {str(e)}"
        )


@router.post("/verify-tenant", response_model=TenantVerificationResponse)
async def verify_tenant(
    request_data: TenantVerificationRequest,
    request: Request,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Verify a complete tenant application with multiple documents.
    
    Analyzes all documents together for:
    - Name consistency across documents
    - Overall fraud risk
    - Document authenticity
    """
    try:
        # Prepare documents for analysis
        documents = []
        for doc in request_data.documents:
            image_data = decode_base64_image(doc.document_base64)
            doc_type = DocumentType(doc.document_type.value)
            filename = doc.filename or "document"
            documents.append((image_data, doc_type, filename))
        
        # Analyze all documents
        result = await ocr_engine.verify_documents_set(
            documents=documents,
            applicant_name=request_data.applicant_name
        )
        
        # Generate recommendation
        recommendation = get_recommendation(
            result["overall_risk_score"],
            result["suspicious_count"],
            result.get("name_consistent")
        )
        
        # Log action
        await UserUseCases.log_action(
            db=db,
            user_id=current_user.id,
            action="tenant_verified",
            entity_type="tenant_verification",
            entity_id=None,
            details={
                "applicant_name": request_data.applicant_name,
                "document_count": result["document_count"],
                "overall_risk_level": result["overall_risk_level"],
                "recommendation": recommendation[:50]
            },
            ip_address=get_client_ip(request)
        )
        
        return TenantVerificationResponse(
            success=True,
            applicant_name=request_data.applicant_name,
            document_count=result["document_count"],
            overall_risk_score=result["overall_risk_score"],
            overall_risk_level=result["overall_risk_level"],
            verified_count=result["verified_count"],
            suspicious_count=result["suspicious_count"],
            name_consistent=result.get("name_consistent"),
            documents=result["documents"],
            summary=result["summary"],
            recommendation=recommendation,
            cross_document_analysis=result.get("cross_document_analysis")
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tenant verification failed: {str(e)}"
        )


@router.post("/verify-property-image")
async def verify_property_image(
    request_data: PropertyImageRequest,
    request: Request,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Verify if a property image is authentic.
    
    Uses CNN to detect:
    - AI-generated images
    - Stock photos
    - Non-property images
    - Stolen/reused images
    """
    try:
        # Decode image
        image_data = decode_base64_image(request_data.image_base64)
        
        # Analyze image
        result = await real_image_engine.analyze_image(
            image_data=image_data,
            filename=request_data.filename
        )
        
        # Log action
        await UserUseCases.log_action(
            db=db,
            user_id=current_user.id,
            action="property_image_verified",
            entity_type="image_verification",
            entity_id=None,
            details={
                "risk_level": result.risk_level.value,
                "is_property_image": result.is_property_image
            },
            ip_address=get_client_ip(request)
        )
        
        return {
            "success": True,
            "result": result.to_dict()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image verification failed: {str(e)}"
        )


@router.post("/verify-property-images")
async def verify_property_images(
    request_data: MultiImageRequest,
    request: Request,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Verify multiple property images from a listing.
    
    Provides aggregate analysis to detect fake listings.
    """
    try:
        # Prepare images
        images = []
        for img in request_data.images:
            image_data = decode_base64_image(img.image_base64)
            images.append((image_data, img.filename))
        
        # Analyze all images
        result = await real_image_engine.analyze_multiple_images(images)
        
        # Log action
        await UserUseCases.log_action(
            db=db,
            user_id=current_user.id,
            action="property_images_verified",
            entity_type="multi_image_verification",
            entity_id=None,
            details={
                "image_count": result["image_count"],
                "overall_risk_level": result["overall_risk_level"]
            },
            ip_address=get_client_ip(request)
        )
        
        return {
            "success": True,
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image verification failed: {str(e)}"
        )


@router.post("/verify-listing-images-upload")
async def verify_listing_images_upload(
    request: Request,
    files: List[UploadFile] = File(...),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Verify listing images via file upload.
    
    Accepts multiple image files for analysis.
    """
    try:
        # Prepare images
        images = []
        for file in files:
            image_data = await file.read()
            images.append((image_data, file.filename))
        
        # Analyze all images
        result = await real_image_engine.analyze_multiple_images(images)
        
        # Log action
        await UserUseCases.log_action(
            db=db,
            user_id=current_user.id,
            action="listing_images_verified",
            entity_type="multi_image_verification",
            entity_id=None,
            details={
                "image_count": result["image_count"],
                "filenames": [f.filename for f in files]
            },
            ip_address=get_client_ip(request)
        )
        
        return {
            "success": True,
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image verification failed: {str(e)}"
        )


@router.get("/verification-history")
async def get_verification_history(
    skip: int = 0,
    limit: int = 50,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get landlord's verification history.
    
    Returns past document and image verifications.
    """
    try:
        from infrastructure.database import AuditLogModel
        from sqlalchemy import select
        
        # Get audit logs for verification actions
        result = await db.execute(
            select(AuditLogModel)
            .where(AuditLogModel.user_id == current_user.id)
            .where(AuditLogModel.action.in_([
                "document_verified",
                "tenant_verified",
                "property_image_verified",
                "property_images_verified"
            ]))
            .order_by(AuditLogModel.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        logs = result.scalars().all()
        
        return {
            "success": True,
            "count": len(logs),
            "history": [
                {
                    "id": log.id,
                    "action": log.action,
                    "entity_type": log.entity_type,
                    "details": log.details,
                    "created_at": log.created_at.isoformat()
                }
                for log in logs
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve history: {str(e)}"
        )


@router.get("/dashboard-stats")
async def get_landlord_dashboard_stats(
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get dashboard statistics for landlord.
    
    Includes verification counts and fraud detection rates.
    """
    try:
        from infrastructure.database import AuditLogModel
        from sqlalchemy import select, func
        
        # Count verifications by type
        verification_actions = [
            "document_verified",
            "tenant_verified",
            "property_image_verified"
        ]
        
        stats = {}
        for action in verification_actions:
            result = await db.execute(
                select(func.count(AuditLogModel.id))
                .where(AuditLogModel.user_id == current_user.id)
                .where(AuditLogModel.action == action)
            )
            stats[action] = result.scalar() or 0
        
        return {
            "success": True,
            "stats": {
                "total_document_verifications": stats.get("document_verified", 0),
                "total_tenant_verifications": stats.get("tenant_verified", 0),
                "total_image_verifications": stats.get("property_image_verified", 0),
                "total_verifications": sum(stats.values())
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve stats: {str(e)}"
        )

# =====================================================================
# CROSS-DOCUMENT CONSISTENCY VERIFICATION (Real AI)
# =====================================================================

@router.post("/verify-cross-document")
async def verify_cross_document_consistency(
    request: Request,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Verify consistency across multiple documents using AI.
    
    This uses real NLP to:
    - Extract entities (names, addresses, dates) from each document
    - Compare entities across documents
    - Detect inconsistencies that indicate fraud
    
    Request body:
    {
        "documents": [
            {
                "name": "id_card",
                "type": "id" | "paystub" | "lease" | "other",
                "text": "OCR extracted text from document..."
            },
            ...
        ],
        "expected_name": "John Doe",  // optional
        "expected_address": "123 Main St"  // optional
    }
    
    Returns consistency report with:
    - Overall consistency level
    - Entity-level comparison
    - Critical issues (name mismatches, SSN differences)
    - Recommendation (approve/review/reject)
    """
    from application.use_cases.cross_document_engine import cross_document_engine
    
    try:
        body = await request.json()
        documents = body.get("documents", [])
        expected_name = body.get("expected_name")
        expected_address = body.get("expected_address")
        
        if not documents:
            raise HTTPException(status_code=400, detail="No documents provided")
        
        if len(documents) < 2 and not expected_name and not expected_address:
            raise HTTPException(
                status_code=400, 
                detail="Need at least 2 documents, or provide expected_name/expected_address for comparison"
            )
        
        # Analyze documents
        result = cross_document_engine.analyze_documents(
            documents=documents,
            expected_name=expected_name,
            expected_address=expected_address
        )
        
        # Log action
        await UserUseCases.log_action(
            db=db,
            user_id=current_user.id,
            action="cross_document_verified",
            entity_type="cross_document",
            entity_id=None,
            details={
                "document_count": len(documents),
                "consistency_level": result.overall_consistency.value,
                "consistency_score": result.consistency_score,
                "critical_issues": len(result.critical_issues)
            },
            ip_address=get_client_ip(request)
        )
        
        return result.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cross-document verification failed: {str(e)}"
        )


@router.post("/verify-full-application")
async def verify_full_rental_application(
    request: Request,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Complete rental application verification combining all AI engines.
    
    This is the comprehensive verification endpoint that:
    1. OCR extracts text from all documents
    2. Cross-document engine verifies consistency
    3. Image engine verifies property photos (if provided)
    4. Generates unified verification report
    
    Request body:
    {
        "applicant_name": "John Doe",
        "expected_address": "123 Main St, Toronto, ON",
        "documents": [
            {
                "name": "id_card",
                "type": "id",
                "image_base64": "..."
            },
            {
                "name": "pay_stub",
                "type": "paystub",
                "image_base64": "..."
            }
        ],
        "property_images": [  // optional
            {"image_base64": "...", "filename": "living_room.jpg"}
        ]
    }
    """
    from application.use_cases.cross_document_engine import cross_document_engine
    from application.use_cases.ocr_engine import ocr_engine, DocumentType
    
    try:
        body = await request.json()
        applicant_name = body.get("applicant_name")
        expected_address = body.get("expected_address")
        documents = body.get("documents", [])
        property_images = body.get("property_images", [])
        
        if not documents:
            raise HTTPException(status_code=400, detail="No documents provided")
        
        # Step 1: OCR extract text from all documents
        extracted_documents = []
        ocr_results = []
        
        for doc in documents:
            try:
                # Decode and OCR
                image_data = decode_base64_image(doc.get("image_base64", ""))
                doc_type_str = doc.get("type", "other")
                
                # Map to DocumentType
                doc_type_map = {
                    "id": DocumentType.ID_CARD,
                    "paystub": DocumentType.PAYSTUB,
                    "lease": DocumentType.RENTAL_APPLICATION,
                    "bank": DocumentType.BANK_STATEMENT,
                    "employment": DocumentType.EMPLOYMENT_LETTER,
                }
                doc_type = doc_type_map.get(doc_type_str, DocumentType.ID_CARD)
                
                # Run OCR
                ocr_result = await ocr_engine.analyze_document(
                    image_data=image_data,
                    expected_type=doc_type,
                    applicant_name=applicant_name
                )
                
                ocr_results.append({
                    "name": doc.get("name", "document"),
                    "type": doc_type_str,
                    "ocr_result": ocr_result.to_dict()
                })
                
                # Add to cross-document analysis
                extracted_documents.append({
                    "name": doc.get("name", "document"),
                    "type": doc_type_str,
                    "text": ocr_result.extracted_data.raw_text or ""
                })
                
            except Exception as e:
                ocr_results.append({
                    "name": doc.get("name", "document"),
                    "type": doc.get("type", "unknown"),
                    "error": str(e)
                })
        
        # Step 2: Cross-document consistency check
        cross_doc_result = None
        if len(extracted_documents) >= 2 or applicant_name or expected_address:
            cross_doc_result = cross_document_engine.analyze_documents(
                documents=extracted_documents,
                expected_name=applicant_name,
                expected_address=expected_address
            )
        
        # Step 3: Property image verification (if provided)
        image_results = []
        if property_images:
            for img in property_images:
                try:
                    image_data = decode_base64_image(img.get("image_base64", ""))
                    img_result = await real_image_engine.analyze_image(
                        image_data=image_data,
                        filename=img.get("filename")
                    )
                    image_results.append(img_result.to_dict())
                except Exception as e:
                    image_results.append({"error": str(e)})
        
        # Step 4: Generate unified report
        overall_risk = 0.0
        risk_factors = []
        
        # Factor in OCR results
        for ocr in ocr_results:
            if "ocr_result" in ocr:
                ocr_risk = ocr["ocr_result"].get("risk_score", 0)
                overall_risk += ocr_risk * 0.3
                if ocr_risk > 0.5:
                    risk_factors.append(f"Suspicious document: {ocr['name']}")
        
        # Factor in cross-document consistency
        if cross_doc_result:
            consistency_risk = 1 - cross_doc_result.consistency_score
            overall_risk += consistency_risk * 0.5
            
            if cross_doc_result.critical_issues:
                risk_factors.extend([
                    f"Critical: {issue['description']}" 
                    for issue in cross_doc_result.critical_issues
                ])
        
        # Factor in image verification
        if image_results:
            img_risk = sum(
                r.get("risk_score", 0) for r in image_results if "risk_score" in r
            ) / len(image_results)
            overall_risk += img_risk * 0.2
            
            for r in image_results:
                if r.get("risk_score", 0) > 0.5:
                    risk_factors.append("Suspicious property image detected")
        
        # Normalize
        overall_risk = min(1.0, overall_risk)
        
        # Generate recommendation
        if overall_risk < 0.2 and not risk_factors:
            recommendation = "✅ APPROVE - Application verified successfully"
            risk_level = "low"
        elif overall_risk < 0.4:
            recommendation = "✅ LIKELY APPROVE - Minor concerns, proceed with caution"
            risk_level = "low"
        elif overall_risk < 0.6:
            recommendation = "⚠️ REVIEW REQUIRED - Significant concerns detected"
            risk_level = "medium"
        elif overall_risk < 0.8:
            recommendation = "🚨 HIGH RISK - Multiple issues require manual verification"
            risk_level = "high"
        else:
            recommendation = "❌ REJECT - High fraud probability"
            risk_level = "critical"
        
        # Log action
        await UserUseCases.log_action(
            db=db,
            user_id=current_user.id,
            action="full_application_verified",
            entity_type="full_application",
            entity_id=None,
            details={
                "applicant_name": applicant_name,
                "document_count": len(documents),
                "image_count": len(property_images),
                "overall_risk": overall_risk,
                "risk_level": risk_level
            },
            ip_address=get_client_ip(request)
        )
        
        return {
            "success": True,
            "applicant_name": applicant_name,
            "overall_risk_score": round(overall_risk, 3),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "risk_factors": risk_factors,
            "document_analysis": ocr_results,
            "cross_document_verification": cross_doc_result.to_dict() if cross_doc_result else None,
            "property_image_verification": image_results if image_results else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Full application verification failed: {str(e)}"
        )