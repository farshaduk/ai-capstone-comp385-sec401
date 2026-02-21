from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from infrastructure.database import UserModel, AuditLogModel
from application.use_cases.auth_use_cases import AuthUseCases
from datetime import datetime


class UserUseCases:
    
    @staticmethod
    async def create_user(
        db: AsyncSession,
        email: str,
        password: str,
        full_name: str,
        role: str = "renter",
        subscription_plan: str = "free"
    ) -> UserModel:
        """Create a new user"""
        
        # Check if user already exists
        result = await db.execute(select(UserModel).where(UserModel.email == email))
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            raise ValueError("Email already registered")
        
        # Determine initial scans based on subscription
        scans_map = {
            "free": 1000,
            "basic": 50,
            "premium": 200,
            "enterprise": 1000
        }
        
        user = UserModel(
            email=email,
            hashed_password=AuthUseCases.get_password_hash(password),
            full_name=full_name,
            role=role,
            subscription_plan=subscription_plan,
            scans_remaining=scans_map.get(subscription_plan, 10),
            is_active=True
        )
        
        db.add(user)
        await db.commit()
        await db.refresh(user)
        
        return user
    
    @staticmethod
    async def get_user(db: AsyncSession, user_id: int) -> Optional[UserModel]:
        """Get user by ID"""
        
        result = await db.execute(select(UserModel).where(UserModel.id == user_id))
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_user_by_email(db: AsyncSession, email: str) -> Optional[UserModel]:
        """Get user by email"""
        
        result = await db.execute(select(UserModel).where(UserModel.email == email))
        return result.scalar_one_or_none()
    
    @staticmethod
    async def list_users(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        role: Optional[str] = None
    ) -> List[UserModel]:
        """List users with optional role filter"""
        
        query = select(UserModel)
        
        if role:
            query = query.where(UserModel.role == role)
        
        query = query.offset(skip).limit(limit).order_by(UserModel.created_at.desc())
        
        result = await db.execute(query)
        return result.scalars().all()
    
    @staticmethod
    async def update_user(
        db: AsyncSession,
        user_id: int,
        **updates
    ) -> UserModel:
        """Update user information"""
        
        user = await UserUseCases.get_user(db, user_id)
        
        if not user:
            raise ValueError("User not found")
        
        for key, value in updates.items():
            if hasattr(user, key) and value is not None:
                setattr(user, key, value)
        
        user.updated_at = datetime.utcnow()
        
        await db.commit()
        await db.refresh(user)
        
        return user
    
    @staticmethod
    async def update_subscription(
        db: AsyncSession,
        user_id: int,
        new_plan: str
    ) -> UserModel:
        """Update user subscription plan"""
        
        scans_map = {
            "free": 1000,
            "basic": 50,
            "premium": 200,
            "enterprise": 1000
        }
        
        user = await UserUseCases.get_user(db, user_id)
        
        if not user:
            raise ValueError("User not found")
        
        user.subscription_plan = new_plan
        user.scans_remaining = scans_map.get(new_plan, 10)
        user.updated_at = datetime.utcnow()
        
        await db.commit()
        await db.refresh(user)
        
        return user
    
    @staticmethod
    async def deactivate_user(db: AsyncSession, user_id: int) -> UserModel:
        """Deactivate a user"""
        
        user = await UserUseCases.get_user(db, user_id)
        
        if not user:
            raise ValueError("User not found")
        
        user.is_active = False
        user.updated_at = datetime.utcnow()
        
        await db.commit()
        await db.refresh(user)
        
        return user
    
    @staticmethod
    async def log_action(
        db: AsyncSession,
        user_id: int,
        action: str,
        entity_type: str = None,
        entity_id: int = None,
        details: dict = None,
        ip_address: str = None
    ):
        """Log user action for audit trail"""
        
        audit_log = AuditLogModel(
            user_id=user_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            details=details or {},
            ip_address=ip_address
        )
        
        db.add(audit_log)
        await db.commit()
    
    @staticmethod
    async def get_audit_logs(
        db: AsyncSession,
        user_id: Optional[int] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[AuditLogModel]:
        """Get audit logs"""
        
        query = select(AuditLogModel)
        
        if user_id:
            query = query.where(AuditLogModel.user_id == user_id)
        
        query = query.offset(skip).limit(limit).order_by(AuditLogModel.created_at.desc())
        
        result = await db.execute(query)
        return result.scalars().all()

