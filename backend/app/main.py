"""
FastAPI application entry point for fashion recommendation system.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

from .api import router


# Create FastAPI application
app = FastAPI(
    title="Fashion Recommendation API",
    description="AI-powered fashion recommendation system using skin tone analysis and CLIP embeddings",
    version="1.0.0"
)

# Configure CORS for frontend integration
origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in origins_env.split(",") if o.strip()] if origins_env != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Configure via ALLOWED_ORIGINS env (comma-separated)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(router, prefix="/api", tags=["recommendations"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Fashion Recommendation API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "analyze_skin_tone": "POST /api/analyze-skin-tone",
            "recommend_outfits": "POST /api/recommend-outfits",
            "health": "GET /api/health",
            "styles": "GET /api/styles"
        }
    }


@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    print("\n" + "="*60)
    print("🎨 Fashion Recommendation API Starting...")
    print("="*60)
    
    # Pre-load models and index images
    try:
        print("\n📦 Loading CLIP model...")
        from app.clip_utils import get_clip_model
        get_clip_model()
        print("✅ CLIP model loaded")
        
        print("\n📸 Scanning and indexing clothing images...")
        from app.image_indexer import get_indexer
        from app.recommender import get_recommender
        
        indexer = get_indexer()
        catalog = indexer.scan_and_index()
        
        if len(catalog) == 0:
            print("⚠️  No images found in assets/clothes/")
            print("   Add images to assets/clothes/ and restart the server")
        else:
            print(f"✅ Indexed {len(catalog)} clothing items")
        
        print("\n📦 Initializing recommender with catalog...")
        recommender = get_recommender(catalog)
        print(f"✅ Recommender ready with {len(recommender.catalog)} items")
        
    except Exception as e:
        print(f"⚠️  Warning: Initialization error: {e}")
        print("   System will work with empty catalog.")
    
    print("\n" + "="*60)
    print("✅ API Ready!")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🖼️  Image endpoint: GET /api/images/{filename}")
    print("="*60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    print("\n👋 Fashion Recommendation API shutting down...")


if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
