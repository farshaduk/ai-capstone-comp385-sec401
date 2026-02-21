from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, JSON, Text, Enum as SQLEnum
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from config import get_settings

settings = get_settings()

Base = declarative_base()

# Async engine for SQLite
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False}
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


class UserModel(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    role = Column(String, default="renter")
    subscription_plan = Column(String, default="free")
    is_active = Column(Boolean, default=True)
    scans_remaining = Column(Integer, default=1000)
    phone = Column(String, nullable=True, default="")
    address = Column(String, nullable=True, default="")
    bio = Column(Text, nullable=True, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DatasetModel(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    file_path = Column(String, nullable=False)
    processed_file_path = Column(String, nullable=True)
    record_count = Column(Integer, default=0)
    column_count = Column(Integer, default=0)
    feature_count = Column(Integer, default=0)
    statistics = Column(JSON)
    preprocessing_status = Column(String, default="pending")   # pending | processing | completed | failed
    preprocessing_report = Column(JSON, nullable=True)
    uploaded_by = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class MLModelModel(Base):
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    dataset_id = Column(Integer)
    model_path = Column(String, nullable=False)
    status = Column(String, default="training")
    metrics = Column(JSON)
    is_active = Column(Boolean, default=False)
    trained_by = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class RiskAnalysisModel(Base):
    __tablename__ = "risk_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    listing_text = Column(Text, nullable=False)
    listing_price = Column(Float, nullable=True)
    risk_score = Column(Float, default=0.0)
    risk_level = Column(String)
    risk_indicators = Column(JSON)
    risk_story = Column(Text)
    confidence = Column(Float, default=0.0)
    model_version = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditLogModel(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    action = Column(String, nullable=False)
    entity_type = Column(String)
    entity_id = Column(Integer)
    details = Column(JSON)
    ip_address = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class SubscriptionPlanModel(Base):
    __tablename__ = "subscription_plans"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    display_name = Column(String, nullable=False)
    price = Column(Float, default=0.0)
    scans_per_month = Column(Integer, default=1000)
    features = Column(JSON)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class FeedbackModel(Base):
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, nullable=False)
    user_id = Column(Integer, nullable=False)
    feedback_type = Column(String, nullable=False)  # safe, fraud, unsure
    comments = Column(Text)
    status = Column(String, default="pending")  # pending, approved, rejected
    reviewed_by = Column(Integer, nullable=True)  # admin user_id who reviewed
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PaymentModel(Base):
    __tablename__ = "payments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    transaction_id = Column(String, unique=True, nullable=False)
    plan_name = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    status = Column(String, default="completed")  # completed, failed, pending
    card_last_four = Column(String)
    cardholder_name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


# ============ PROPERTY MANAGEMENT MODELS ============

class ListingModel(Base):
    __tablename__ = "listings"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, nullable=False)
    title = Column(String, nullable=False)
    address = Column(String, nullable=False)
    city = Column(String, default="Toronto")
    province = Column(String, default="ON")
    postal_code = Column(String)
    price = Column(Float, nullable=False)
    beds = Column(Integer, default=1)
    baths = Column(Float, default=1)
    sqft = Column(Integer)
    property_type = Column(String, default="apartment")
    description = Column(Text)
    amenities = Column(JSON, default=list)
    laundry = Column(String, default="in_unit")
    utilities = Column(String, default="not_included")
    pet_friendly = Column(Boolean, default=False)
    parking_included = Column(Boolean, default=False)
    available_date = Column(String)
    is_active = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)
    listing_status = Column(String, default="pending_review")  # pending_review, approved, rejected
    admin_notes = Column(Text)
    reviewed_by = Column(Integer)
    reviewed_at = Column(DateTime)
    risk_score = Column(Float, default=0.0)
    views = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SavedListingModel(Base):
    __tablename__ = "saved_listings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    listing_id = Column(Integer, nullable=False)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class ApplicationModel(Base):
    __tablename__ = "applications"

    id = Column(Integer, primary_key=True, index=True)
    listing_id = Column(Integer, nullable=False)
    applicant_id = Column(Integer, nullable=False)
    landlord_id = Column(Integer, nullable=False)
    message = Column(Text)
    status = Column(String, default="pending")  # pending, approved, rejected, viewing_scheduled
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ApplicationMessageModel(Base):
    __tablename__ = "application_messages"

    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, nullable=False)
    sender_id = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class LeaseModel(Base):
    __tablename__ = "leases"

    id = Column(Integer, primary_key=True, index=True)
    listing_id = Column(Integer, nullable=False)
    landlord_id = Column(Integer, nullable=False)
    tenant_id = Column(Integer, nullable=False)
    start_date = Column(String, nullable=False)
    end_date = Column(String, nullable=False)
    rent = Column(Float, nullable=False)
    deposit = Column(Float, default=0.0)
    status = Column(String, default="active")  # active, expired, pending_signature, expiring
    created_at = Column(DateTime, default=datetime.utcnow)


class MessageModel(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    sender_id = Column(Integer, nullable=False)
    receiver_id = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


