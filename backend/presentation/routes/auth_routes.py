from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from infrastructure.database import get_db
from presentation.schemas import Token, UserLogin, UserRegister, UserResponse
from presentation.dependencies import get_current_user, get_client_ip
from application.use_cases.auth_use_cases import AuthUseCases
from application.use_cases.user_use_cases import UserUseCases
from datetime import timedelta
from config import get_settings

router = APIRouter(prefix="/auth", tags=["Authentication"])
settings = get_settings()


@router.post("/register", response_model=UserResponse)
async def register(
    user_data: UserRegister,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user"""
    
    try:
        # Security: Only allow 'renter' and 'landlord' roles via self-registration
        # Admin accounts must be created through seed_data or by existing admins
        allowed_roles = ["renter", "landlord"]
        role = user_data.role if user_data.role in allowed_roles else "renter"
        
        user = await UserUseCases.create_user(
            db=db,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name,
            role=role
        )
        
        # Log action
        await UserUseCases.log_action(
            db=db,
            user_id=user.id,
            action="user_registered",
            entity_type="user",
            entity_id=user.id,
            ip_address=get_client_ip(request)
        )
        
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/login", response_model=Token)
async def login(
    user_credentials: UserLogin,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Login and get access token"""
    
    user = await AuthUseCases.authenticate_user(
        db=db,
        email=user_credentials.email,
        password=user_credentials.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = AuthUseCases.create_access_token(
        data={"sub": user.email},
        expires_delta=access_token_expires
    )
    
    # Log action
    await UserUseCases.log_action(
        db=db,
        user_id=user.id,
        action="user_login",
        ip_address=get_client_ip(request)
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user = Depends(get_current_user)
):
    """Get current user information"""
    return current_user

