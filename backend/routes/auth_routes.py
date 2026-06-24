# backend/routes/auth_routes.py
"""
Authentication endpoints: Signup, Login, and Identity fetching.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.database import get_db, User, UserProfile
from backend.auth import (
    get_password_hash, 
    verify_password, 
    create_access_token, 
    get_current_user
)

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Pydantic schemas
class AuthRequest(BaseModel):
    username: str
    password: str
    role: str = "User"  # Default role is User

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str

class UserMeResponse(BaseModel):
    id: int
    username: str
    role: str

# ═══════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════

@router.post("/signup", response_model=AuthResponse)
def signup(request: AuthRequest, db: Session = Depends(get_db)):
    # Check if username exists
    existing = db.query(User).filter(User.username == request.username).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
        
    hashed_pwd = get_password_hash(request.password)
    user = User(
        username=request.username, 
        password_hash=hashed_pwd, 
        role="User"  # Force public signups to 'User' role
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Create profile
    profile = UserProfile(user_id=user.id, age=None, gender=None, location="United States", chronic_conditions="")
    db.add(profile)
    db.commit()
    
    token = create_access_token(data={"sub": user.username})
    return {
        "access_token": token, 
        "token_type": "bearer", 
        "username": user.username,
        "role": user.role
    }

@router.post("/login", response_model=AuthResponse)
def login(request: AuthRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == request.username).first()
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
        
    token = create_access_token(data={"sub": user.username})
    return {
        "access_token": token, 
        "token_type": "bearer", 
        "username": user.username,
        "role": user.role
    }

@router.get("/me", response_model=UserMeResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "role": current_user.role
    }
