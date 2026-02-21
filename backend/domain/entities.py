from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class UserRole(str, Enum):
    ADMIN = "admin"
    RENTER = "renter"
    LANDLORD = "landlord"


class SubscriptionPlan(str, Enum):
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class RiskLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ModelStatus(str, Enum):
    TRAINING = "training"
    COMPLETED = "completed"
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"


class User:
    def __init__(
        self,
        id: Optional[int] = None,
        email: str = "",
        hashed_password: str = "",
        full_name: str = "",
        role: UserRole = UserRole.RENTER,
        subscription_plan: SubscriptionPlan = SubscriptionPlan.FREE,
        is_active: bool = True,
        scans_remaining: int = 1000,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None
    ):
        self.id = id
        self.email = email
        self.hashed_password = hashed_password
        self.full_name = full_name
        self.role = role
        self.subscription_plan = subscription_plan
        self.is_active = is_active
        self.scans_remaining = scans_remaining
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()


class Dataset:
    def __init__(
        self,
        id: Optional[int] = None,
        name: str = "",
        description: str = "",
        file_path: str = "",
        record_count: int = 0,
        column_count: int = 0,
        statistics: Optional[Dict[str, Any]] = None,
        uploaded_by: Optional[int] = None,
        created_at: Optional[datetime] = None
    ):
        self.id = id
        self.name = name
        self.description = description
        self.file_path = file_path
        self.record_count = record_count
        self.column_count = column_count
        self.statistics = statistics or {}
        self.uploaded_by = uploaded_by
        self.created_at = created_at or datetime.utcnow()


class MLModel:
    def __init__(
        self,
        id: Optional[int] = None,
        name: str = "",
        version: str = "",
        dataset_id: Optional[int] = None,
        model_path: str = "",
        status: ModelStatus = ModelStatus.TRAINING,
        metrics: Optional[Dict[str, Any]] = None,
        is_active: bool = False,
        trained_by: Optional[int] = None,
        created_at: Optional[datetime] = None
    ):
        self.id = id
        self.name = name
        self.version = version
        self.dataset_id = dataset_id
        self.model_path = model_path
        self.status = status
        self.metrics = metrics or {}
        self.is_active = is_active
        self.trained_by = trained_by
        self.created_at = created_at or datetime.utcnow()


class RiskAnalysis:
    def __init__(
        self,
        id: Optional[int] = None,
        user_id: Optional[int] = None,
        listing_text: str = "",
        risk_score: float = 0.0,
        risk_level: RiskLevel = RiskLevel.LOW,
        risk_indicators: Optional[List[Dict[str, Any]]] = None,
        risk_story: str = "",
        confidence: float = 0.0,
        model_version: str = "",
        created_at: Optional[datetime] = None
    ):
        self.id = id
        self.user_id = user_id
        self.listing_text = listing_text
        self.risk_score = risk_score
        self.risk_level = risk_level
        self.risk_indicators = risk_indicators or []
        self.risk_story = risk_story
        self.confidence = confidence
        self.model_version = model_version
        self.created_at = created_at or datetime.utcnow()


class AuditLog:
    def __init__(
        self,
        id: Optional[int] = None,
        user_id: Optional[int] = None,
        action: str = "",
        entity_type: str = "",
        entity_id: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        created_at: Optional[datetime] = None
    ):
        self.id = id
        self.user_id = user_id
        self.action = action
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.details = details or {}
        self.ip_address = ip_address
        self.created_at = created_at or datetime.utcnow()

