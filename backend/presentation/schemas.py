from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# Auth Schemas
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserRegister(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    role: Optional[str] = "renter"


# User Schemas
class UserBase(BaseModel):
    email: EmailStr
    full_name: str
    role: str
    subscription_plan: str


class UserResponse(UserBase):
    id: int
    is_active: bool
    scans_remaining: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    subscription_plan: Optional[str] = None
    is_active: Optional[bool] = None


# Dataset Schemas
class DatasetUpload(BaseModel):
    name: str
    description: Optional[str] = ""


class DatasetResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    file_path: str
    record_count: int
    column_count: int
    statistics: Dict[str, Any]
    uploaded_by: Optional[int]
    created_at: datetime
    
    class Config:
        from_attributes = True


class SyntheticDataRequest(BaseModel):
    base_dataset_id: int
    fraud_percentage: float = Field(default=0.1, ge=0.0, le=0.5)


class DatasetAnalysisResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    file_path: str
    created_at: datetime
    overview: Dict[str, Any]
    data_quality: Dict[str, Any]
    column_analysis: Dict[str, Any]
    numeric_statistics: Dict[str, Any]
    categorical_statistics: Dict[str, Any]
    recommendations: List[str]


# Model Schemas
class ModelTrainRequest(BaseModel):
    name: str
    dataset_id: int


class ModelResponse(BaseModel):
    id: int
    name: str
    version: str
    dataset_id: Optional[int]
    model_path: str
    status: str
    metrics: Dict[str, Any]
    is_active: bool
    trained_by: Optional[int]
    created_at: datetime
    
    class Config:
        from_attributes = True
        protected_namespaces = ()


class ModelAnalysisResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    id: int
    name: str
    version: str
    status: str
    is_active: bool
    created_at: datetime
    model_type: str
    algorithm_details: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    training_details: Dict[str, Any]
    visualizations: Dict[str, str]
    dataset_info: Dict[str, Any]
    threshold_analysis: List[Dict[str, Any]]
    performance_benchmarks: Dict[str, Any]
    business_impact: Dict[str, Any]
    model_comparison: List[Dict[str, Any]]
    deployment_readiness: List[Dict[str, Any]]
    error_analysis: Dict[str, Any]
    monitoring_recommendations: Dict[str, Any]
    ab_test_recommendations: Dict[str, Any]
    feature_importance: List[Dict[str, Any]]


# Risk Analysis Schemas
class RiskAnalysisRequest(BaseModel):
    listing_text: str = Field(..., min_length=10)
    listing_price: Optional[float] = None
    location: Optional[str] = None


class URLAnalysisRequest(BaseModel):
    url: str = Field(..., min_length=10)
    listing_price: Optional[float] = None
    location: Optional[str] = None


class FeedbackRequest(BaseModel):
    analysis_id: int
    feedback_type: str = Field(..., pattern="^(safe|fraud|unsure)$")
    comments: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: int
    analysis_id: int
    user_id: int
    feedback_type: str
    comments: Optional[str]
    status: Optional[str] = "pending"
    reviewed_by: Optional[int] = None
    reviewed_at: Optional[datetime] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class FeedbackReviewRequest(BaseModel):
    status: str = Field(..., pattern="^(approved|rejected)$")


class ReportExportRequest(BaseModel):
    analysis_id: int
    format: str = Field(default="html", pattern="^(html|pdf)$")


# =========================================================================
# ENTERPRISE-GRADE INDICATOR SCHEMAS
# =========================================================================

class IndicatorSchema(BaseModel):
    """Enterprise indicator format with standardized codes and severity"""
    code: str  # e.g., "OFF_PLATFORM_PAYMENT", "HIGH_URGENCY"
    category: str  # e.g., "payment", "urgency", "contact"
    severity: int  # 1-5 scale (5 = most severe)
    evidence: List[str]  # Extracted text that triggered the indicator
    description: str  # Human-readable explanation
    impact_score: float  # Contribution to overall risk (0.0-1.0)


class IndicatorSummarySchema(BaseModel):
    """Summary counts by severity level"""
    critical: int  # severity >= 4
    moderate: int  # severity 2-3
    low: int  # severity < 2


class RiskIndicator(BaseModel):
    """Legacy indicator format (deprecated, for backward compatibility)"""
    type: str
    severity: str
    description: str
    impact: float
    examples: List[str]


class RiskAnalysisResponse(BaseModel):
    """
    Enterprise-grade risk analysis response.
    
    Provides standardized, explainable fraud detection output with:
    - risk_score: 0-100 integer scale (enterprise standard)
    - risk_level: Categorized level (Very Low to Very High)
    - confidence: Agreement-based confidence score
    - indicators: List of coded indicators with severity and evidence
    """
    # Core identification
    id: int
    
    # Risk assessment (enterprise format)
    risk_score: int  # 0-100 scale
    risk_score_normalized: Optional[float] = None  # 0-1 scale
    risk_level: str  # "Very Low", "Low", "Medium", "High", "Very High"
    confidence: float  # 0.0-1.0 based on indicator agreement
    
    # Detailed indicators
    indicators: List[Dict[str, Any]]  # Enterprise indicator format
    indicator_count: Optional[int] = None
    indicator_summary: Optional[Dict[str, int]] = None
    
    # Human-readable explanation
    risk_story: str
    
    # Metadata
    scans_remaining: int
    model_version: str
    ml_model_used: Optional[bool] = None
    analysis_timestamp: Optional[str] = None
    
    # AI Components
    bert_enabled: Optional[bool] = None
    bert_prediction: Optional[Dict[str, Any]] = None
    price_analysis_enabled: Optional[bool] = None
    price_analysis: Optional[Dict[str, Any]] = None
    address_validation_enabled: Optional[bool] = None
    address_validation: Optional[Dict[str, Any]] = None
    ai_components: Optional[Dict[str, bool]] = None
    
    # Legacy fields (deprecated)
    risk_indicators: Optional[List[Dict[str, Any]]] = None  # Use 'indicators' instead
    created_at: Optional[str] = None  # Use 'analysis_timestamp' instead
    
    class Config:
        protected_namespaces = ()


class AnalysisHistoryResponse(BaseModel):
    id: int
    listing_text: str
    risk_score: float
    risk_level: str
    created_at: datetime
    
    class Config:
        from_attributes = True


# Audit Log Schemas
class AuditLogResponse(BaseModel):
    id: int
    user_id: Optional[int]
    user_email: Optional[str] = None
    action: str
    entity_type: Optional[str]
    entity_id: Optional[int]
    details: Dict[str, Any]
    ip_address: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


# Subscription Schemas
class SubscriptionPlanResponse(BaseModel):
    id: int
    name: str
    display_name: str
    price: float
    scans_per_month: int
    features: Dict[str, Any]
    is_active: bool
    
    class Config:
        from_attributes = True


class SubscriptionPlanCreate(BaseModel):
    name: str
    display_name: str
    price: float
    scans_per_month: int
    features: Dict[str, Any]


class SubscriptionPlanUpdate(BaseModel):
    display_name: Optional[str] = None
    price: Optional[float] = None
    scans_per_month: Optional[int] = None
    features: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class PaymentRequest(BaseModel):
    plan_name: str
    card_number: str
    expiry_month: int
    expiry_year: int
    cvv: str
    cardholder_name: str


class PaymentResponse(BaseModel):
    success: bool
    transaction_id: str
    message: str
    plan_name: str
    amount: float


# Statistics Schemas
class DashboardStats(BaseModel):
    total_users: int
    total_analyses: int
    high_risk_analyses: int

