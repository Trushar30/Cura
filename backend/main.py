# backend/main.py
"""
Main FastAPI server initialization, routing setup, and middleware settings.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.database import init_db
from backend.routes import (
    auth_routes,
    profile_routes,
    recommender_routes,
    graph_routes,
    analytics_routes
)

app = FastAPI(
    title="Personalized Healthcare Recommendation System API",
    description="Backend ML & AI model serving API for symptom prediction, drug recommendations, and usage tracking.",
    version="1.0.0"
)

# ═══════════════════════════════════════════════════════════════════════
# MIDDLEWARE (CORS)
# ═══════════════════════════════════════════════════════════════════════

origins = [
    "http://localhost:5173",  # Default Vite React port
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "*"                       # Allow all for local debugging simplicity
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════
# STARTUP HOOKS
# ═══════════════════════════════════════════════════════════════════════

@app.on_event("startup")
def startup_event():
    # Initialize SQLite database and seed default accounts
    init_db()

# ═══════════════════════════════════════════════════════════════════════
# ROUTERS
# ═══════════════════════════════════════════════════════════════════════

app.include_router(auth_routes.router, prefix="/api")
app.include_router(profile_routes.router, prefix="/api")
app.include_router(recommender_routes.router, prefix="/api")
app.include_router(graph_routes.router, prefix="/api")
app.include_router(analytics_routes.router, prefix="/api")

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "system": "Personalized Healthcare Recommendation System API",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
