from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from infrastructure.database import get_db, UserModel, SubscriptionPlanModel
from application.use_cases.auth_use_cases import AuthUseCases
from typing import Optional, Callable

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> UserModel:
    """Get current authenticated user"""
    
    token = credentials.credentials
    user = await AuthUseCases.get_current_user(db, token)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    return user


async def get_current_admin(
    current_user: UserModel = Depends(get_current_user)
) -> UserModel:
    """Ensure current user is an admin"""
    
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return current_user


async def get_current_landlord(
    current_user: UserModel = Depends(get_current_user)
) -> UserModel:
    """Ensure current user is a landlord"""
    
    if current_user.role != "landlord":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Landlord access required"
        )
    
    return current_user


async def get_current_renter(
    current_user: UserModel = Depends(get_current_user)
) -> UserModel:
    """Ensure current user is a renter"""
    
    if current_user.role != "renter":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Renter access required"
        )
    
    return current_user


async def get_optional_user(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> Optional[UserModel]:
    """Get current user if authenticated, otherwise None"""
    
    auth_header = request.headers.get("Authorization")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    
    token = auth_header.split(" ")[1]
    user = await AuthUseCases.get_current_user(db, token)
    
    return user if user and user.is_active else None


def get_client_ip(request: Request) -> str:
    """Get client IP address from request"""
    
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0]
    
    return request.client.host if request.client else "unknown"


async def check_feature_access(
    feature_name: str,
    user: UserModel,
    db: AsyncSession
) -> bool:
    """Check if user has access to a specific feature based on their subscription plan"""
    
    result = await db.execute(
        select(SubscriptionPlanModel).where(
            SubscriptionPlanModel.name == user.subscription_plan
        )
    )
    
    plan = result.scalar_one_or_none()
    
    if not plan:
        # Default free plan features
        default_features = {
            "basic_analysis": True,
            "risk_score": True,
            "history": 1000,
            "support": "community"
        }
        return default_features.get(feature_name, False)
    
    return plan.features.get(feature_name, False)


def require_feature(feature_name: str) -> Callable:
    """Dependency factory that requires a specific feature"""
    
    async def _check_feature(
        current_user: UserModel = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
    ):
        has_access = await check_feature_access(feature_name, current_user, db)
        
        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This feature requires a subscription plan that includes '{feature_name.replace('_', ' ')}'. Please upgrade your plan."
            )
        
        return current_user
    
    return _check_feature

