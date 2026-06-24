# backend/routes/profile_routes.py
"""
Profile management and user activity logging routes.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
from backend.database import get_db, User, UserProfile, UserActivity
from backend.auth import get_current_user

router = APIRouter(prefix="/profile", tags=["Profiles & Activity"])

# Pydantic schemas
class ProfileSchema(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    location: Optional[str] = "United States"
    chronic_conditions: Optional[str] = ""

class ActivityLogSchema(BaseModel):
    activity_type: str  # "search", "predict", "click_drug", "view_graph"
    details: Optional[str] = ""

# ═══════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════

@router.get("", response_model=ProfileSchema)
def get_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    if not profile:
        # Create a blank profile if missing
        profile = UserProfile(user_id=current_user.id, age=None, gender=None, location="United States", chronic_conditions="")
        db.add(profile)
        db.commit()
        db.refresh(profile)
    return profile

@router.put("", response_model=ProfileSchema)
def update_profile(
    profile_data: ProfileSchema,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    if not profile:
        profile = UserProfile(user_id=current_user.id)
        db.add(profile)
        
    profile.age = profile_data.age
    profile.gender = profile_data.gender
    profile.location = profile_data.location
    profile.chronic_conditions = profile_data.chronic_conditions
    
    db.commit()
    db.refresh(profile)
    
    # Log the update action
    log = UserActivity(
        user_id=current_user.id,
        activity_type="profile_update",
        details=f"Updated profile: Age={profile.age}, Gender={profile.gender}, Conditions={profile.chronic_conditions}"
    )
    db.add(log)
    db.commit()
    
    return profile

@router.post("/activity")
def log_activity(
    activity: ActivityLogSchema,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    log = UserActivity(
        user_id=current_user.id,
        activity_type=activity.activity_type,
        details=activity.details
    )
    db.add(log)
    db.commit()
    return {"status": "success", "message": "Activity logged successfully"}
