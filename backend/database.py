# backend/database.py
"""
SQLite database configuration and schema models using SQLAlchemy.
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = "sqlite:///./healthcare_platform.db"

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False}  # Needed for SQLite in FastAPI
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ═══════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="User")  # "User", "Admin", "Analyst"
    
    profile = relationship("UserProfile", back_populates="user", uselist=False)
    activities = relationship("UserActivity", back_populates="user")

class UserProfile(Base):
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    location = Column(String, default="United States")
    chronic_conditions = Column(String, default="")  # Comma-separated list (e.g. "Diabetes,Gerd")
    
    user = relationship("User", back_populates="profile")

class UserActivity(Base):
    __tablename__ = "user_activities"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    activity_type = Column(String, nullable=False)  # "search", "predict", "click_drug", "view_graph"
    details = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="activities")

# Helper to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize Database & Seed default users
def init_db():
    Base.metadata.create_all(bind=engine)
    
    # Check if we need to seed
    db = SessionLocal()
    try:
        # Check if default admin exists
        admin_exists = db.query(User).filter(User.username == "admin").first()
        if not admin_exists:
            import bcrypt
            
            def hash_seed_pw(pw: str) -> str:
                return bcrypt.hashpw(pw.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            # Create Default Admin
            admin = User(
                username="admin", 
                password_hash=hash_seed_pw("admin123"), 
                role="Admin"
            )
            # Create Default Analyst
            analyst = User(
                username="analyst", 
                password_hash=hash_seed_pw("analyst123"), 
                role="Analyst"
            )
            # Create Default User
            normal_user = User(
                username="user", 
                password_hash=hash_seed_pw("user123"), 
                role="User"
            )
            
            db.add_all([admin, analyst, normal_user])
            db.commit()
            
            # Seed profile for user
            db.refresh(normal_user)
            db.refresh(admin)
            db.refresh(analyst)
            
            p1 = UserProfile(user_id=normal_user.id, age=32, gender="Male", location="New York", chronic_conditions="Diabetes , Gerd")
            p2 = UserProfile(user_id=admin.id, age=45, gender="Female", location="San Francisco", chronic_conditions="")
            p3 = UserProfile(user_id=analyst.id, age=28, gender="Male", location="Chicago", chronic_conditions="Allergy")
            
            db.add_all([p1, p2, p3])
            db.commit()
            
            print("Database seeded with default accounts (admin123, analyst123, user123)")
    except Exception as e:
        print("Error seeding database:", e)
    finally:
        db.close()
