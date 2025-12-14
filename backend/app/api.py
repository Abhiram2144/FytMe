"""
FastAPI route definitions for fashion recommendation system.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from pathlib import Path

from app.skin_tone import predict_skin_tone
from app.recommender import get_recommender
from app.image_indexer import get_indexer
from app.config import VALID_STYLES


# Request/Response models
class SkinToneResponse(BaseModel):
    """Response model for skin tone analysis."""
    fitzpatrick_type: str = Field(..., description="Fitzpatrick skin tone type (I-VI)")
    skin_tone_label: str = Field(..., description="Human-readable skin tone label")


class RecommendationRequest(BaseModel):
    """Request model for outfit recommendations."""
    fitzpatrick_type: str = Field(..., description="Fitzpatrick skin tone type (I-VI)")
    preferred_style: str = Field(..., description="Preferred fashion style/vibe")


class ClothingItem(BaseModel):
    """Model for a single clothing item."""
    image: str = Field(..., description="Image filename or URL")
    category: str = Field(..., description="Clothing category")
    color: str = Field(..., description="Predicted color")
    style: str = Field(..., description="Predicted style")


class Outfit(BaseModel):
    """Model for a complete outfit recommendation."""
    items: List[ClothingItem] = Field(..., description="List of clothing items in outfit")
    score: float = Field(..., description="Outfit recommendation score (0-1)")
    explanation: str = Field(..., description="Human-readable explanation for recommendation")
    shirt_pant_match_score: float = Field(..., description="Pairwise shirt–pant match score (0-1)")


class RecommendationResponse(BaseModel):
    """Response model for outfit recommendations."""
    outfits: List[Outfit] = Field(..., description="List of recommended outfits")


# Create API router
router = APIRouter()


@router.post("/analyze-skin-tone", response_model=SkinToneResponse)
async def analyze_skin_tone(file: UploadFile = File(...)):
    """
    Analyze skin tone from uploaded selfie image.
    
    Args:
        file: Uploaded image file (JPEG, PNG)
        
    Returns:
        Skin tone analysis result with Fitzpatrick type
        
    Raises:
        HTTPException: If image processing fails
    """
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Predict skin tone
        result = predict_skin_tone(image_bytes)
        
        return SkinToneResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Skin tone analysis failed: {str(e)}"
        )


@router.post("/recommend-outfits", response_model=RecommendationResponse)
async def recommend_outfits(request: RecommendationRequest):
    """
    Recommend outfits based on skin tone and style preferences.
    
    Args:
        request: Recommendation request with fitzpatrick_type and preferred_style
        
    Returns:
        List of top 3 recommended outfits with scores and explanations
        
    Raises:
        HTTPException: If recommendation generation fails
    """
    try:
        # Validate Fitzpatrick type
        if request.fitzpatrick_type not in ["I", "II", "III", "IV", "V", "VI"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid fitzpatrick_type. Must be one of: I, II, III, IV, V, VI"
            )
        
        # Validate style
        if request.preferred_style.lower() not in [s.lower() for s in VALID_STYLES]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid preferred_style. Must be one of: {', '.join(VALID_STYLES)}"
            )
        
        # Get recommender instance
        recommender = get_recommender()
        
        # Generate recommendations
        outfits = recommender.recommend_outfits(
            fitzpatrick_type=request.fitzpatrick_type,
            preferred_style=request.preferred_style,
            num_outfits=3
        )

        if len(outfits) == 0:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No outfits available. Please add at least one top and one bottom to assets/clothes/. Shoes are optional."
                )
            )
        
        # Convert to response format
        response_outfits = []
        for outfit in outfits:
            items = [ClothingItem(**item) for item in outfit['items']]
            response_outfits.append(Outfit(
                items=items,
                score=outfit['score'],
                explanation=outfit['explanation'],
                shirt_pant_match_score=outfit.get('shirt_pant_match_score', 0.0)
            ))
        
        return RecommendationResponse(outfits=response_outfits)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation generation failed: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "fashion-recommendation-api"
    }


@router.get("/debug/catalog")
async def debug_catalog():
    """Debug endpoint to inspect indexed catalog (counts and samples)."""
    recommender = get_recommender()
    if len(recommender.catalog) == 0:
        return {"message": "Catalog is empty"}

    category_counts = {}
    style_counts = {}
    for item in recommender.catalog:
        category_counts[item.get("category", "unknown")] = category_counts.get(item.get("category", "unknown"), 0) + 1
        style_counts[item.get("style", "unknown")] = style_counts.get(item.get("style", "unknown"), 0) + 1

    return {
        "total_items": len(recommender.catalog),
        "category_counts": category_counts,
        "style_counts": style_counts,
        "sample_items": [
            {
                "image": item.get("image"),
                "category": item.get("category"),
                "color": item.get("color"),
                "style": item.get("style")
            }
            for item in recommender.catalog[:5]
        ]
    }


@router.get("/styles")
async def get_available_styles():
    """
    Get list of available style categories.
    
    Returns:
        List of valid style/vibe options
    """
    return {
        "styles": VALID_STYLES
    }


@router.post("/debug/reindex")
async def force_reindex():
    """Force reindexing: clears catalog cache, re-indexes, and reloads recommender."""
    try:
        indexer = get_indexer()
        # Delete cache file if exists
        cache_file = indexer.cache_file
        if cache_file.exists():
            cache_file.unlink()
            print(f"🧹 Deleted cache: {cache_file}")

        catalog = indexer.scan_and_index()
        recommender = get_recommender()
        recommender.update_catalog(catalog)
        return {"status": "ok", "items": len(catalog)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reindex failed: {e}")


@router.get("/images/{image_name}")
async def serve_image(image_name: str):
    """
    Serve clothing images directly from assets/clothes/ directory.
    
    Args:
        image_name: Name of the image file
        
    Returns:
        Image file
        
    Raises:
        HTTPException: If image not found
    """
    # Get image path
    assets_dir = Path("assets/clothes")
    image_path = assets_dir / image_name
    
    # Check if file exists
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Image not found: {image_name}"
        )
    
    # Serve image
    return FileResponse(image_path)
