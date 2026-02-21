from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from infrastructure.database import Base, engine
from infrastructure.seed_data import seed_database, update_subscription_plans
from infrastructure.database import AsyncSessionLocal
from presentation.routes import auth_routes, admin_routes, renter_routes, landlord_routes, property_routes
from config import get_settings
import time
import os

settings = get_settings()

app = FastAPI(
    title="AI-Powered Rental Fraud & Trust Scoring System",
    description="Enterprise-level platform for detecting rental fraud and providing trust scores",
    version="1.0.0"
)

# Configure CORS. In development allow local frontends; when running in
# development mode enable broader access to simplify local testing while
# keeping production behavior stricter.
if settings.ENVIRONMENT == "development":
    allow_origins = ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173", "http://localhost"]
else:
    allow_origins = ["http://localhost:3000", "http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Determine the origin header from the request to include in CORS response
    origin = request.headers.get("origin", "")
    headers = dict(exc.headers) if exc.headers else {}
    if origin in allow_origins:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=headers
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Determine the origin header from the request to include in CORS response
    origin = request.headers.get("origin", "")
    headers = {}
    if origin in allow_origins:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc) if settings.ENVIRONMENT == "development" else "An error occurred"
        },
        headers=headers
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database and seed data"""
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Migrate: add new preprocessing columns if they don't exist (SQLite)
    async with engine.begin() as conn:
        for col_def in [
            ("processed_file_path", "TEXT"),
            ("feature_count", "INTEGER DEFAULT 0"),
            ("preprocessing_status", "TEXT DEFAULT 'pending'"),
            ("preprocessing_report", "TEXT"),
        ]:
            try:
                await conn.execute(
                    __import__('sqlalchemy').text(
                        f"ALTER TABLE datasets ADD COLUMN {col_def[0]} {col_def[1]}"
                    )
                )
            except Exception:
                pass  # Column already exists

    # Migrate: add profile columns to users table if they don't exist
    async with engine.begin() as conn:
        for col_def in [
            ("phone", "TEXT DEFAULT ''"),
            ("address", "TEXT DEFAULT ''"),
            ("bio", "TEXT DEFAULT ''"),
        ]:
            try:
                await conn.execute(
                    __import__('sqlalchemy').text(
                        f"ALTER TABLE users ADD COLUMN {col_def[0]} {col_def[1]}"
                    )
                )
            except Exception:
                pass  # Column already exists
    
    # Migrate: add feedback review columns if they don't exist
    async with engine.begin() as conn:
        for col_def in [
            ("status", "TEXT DEFAULT 'pending'"),
            ("reviewed_by", "INTEGER"),
            ("reviewed_at", "TEXT"),
        ]:
            try:
                await conn.execute(
                    __import__('sqlalchemy').text(
                        f"ALTER TABLE feedback ADD COLUMN {col_def[0]} {col_def[1]}"
                    )
                )
            except Exception:
                pass  # Column already exists
    
    # Migrate: add listing approval columns if they don't exist
    async with engine.begin() as conn:
        for col_def in [
            ("listing_status", "TEXT DEFAULT 'pending_review'"),
            ("admin_notes", "TEXT"),
            ("reviewed_by", "INTEGER"),
            ("reviewed_at", "TEXT"),
        ]:
            try:
                await conn.execute(
                    __import__('sqlalchemy').text(
                        f"ALTER TABLE listings ADD COLUMN {col_def[0]} {col_def[1]}"
                    )
                )
            except Exception:
                pass  # Column already exists
    
    # Migrate: set existing listings with is_active=1 to approved status
    async with engine.begin() as conn:
        try:
            await conn.execute(
                __import__('sqlalchemy').text(
                    "UPDATE listings SET listing_status = 'approved' WHERE is_active = 1 AND (listing_status IS NULL OR listing_status = 'pending_review')"
                )
            )
        except Exception:
            pass
    
    # Seed database
    async with AsyncSessionLocal() as session:
        await seed_database(session)
    
    # Ensure all plans have all features enabled
    async with AsyncSessionLocal() as session:
        await update_subscription_plans(session)
    
    print("✅ Database initialized and seeded")
    print(f"🚀 Application running in {settings.ENVIRONMENT} mode")


# Mount static files for model visualizations
os.makedirs("models", exist_ok=True)
app.mount("/models", StaticFiles(directory="models"), name="models")

# Include routers
app.include_router(auth_routes.router, prefix="/api")
app.include_router(admin_routes.router, prefix="/api")
app.include_router(renter_routes.router, prefix="/api")
app.include_router(landlord_routes.router, prefix="/api")
app.include_router(property_routes.router, prefix="/api")


# Health check
@app.get("/")
async def root():
    return {
        "message": "AI-Powered Rental Fraud & Trust Scoring System",
        "version": "1.0.0",
        "status": "active"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

