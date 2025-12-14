"""
Test script for fashion recommendation backend.
Run this to verify all components are working correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import SKIN_TONE_COLOR_MAP, VALID_STYLES, FITZPATRICK_DESCRIPTIONS
from app.clip_utils import get_clip_model
from app.recommender import get_recommender


def test_config():
    """Test configuration loading."""
    print("\n" + "="*60)
    print("Testing Configuration...")
    print("="*60)
    
    print(f"\n✓ Loaded {len(SKIN_TONE_COLOR_MAP)} Fitzpatrick types")
    print(f"✓ Loaded {len(VALID_STYLES)} style categories")
    print(f"✓ Styles: {', '.join(VALID_STYLES[:5])}...")
    
    print("\n✅ Configuration test passed!")


def test_clip():
    """Test CLIP model loading and similarity."""
    print("\n" + "="*60)
    print("Testing CLIP Model...")
    print("="*60)
    
    try:
        model = get_clip_model()
        print("\n✓ CLIP model loaded successfully")
        
        # Test similarity
        text1 = "old money beige shirt"
        text2 = "preppy beige button-up"
        similarity = model.compute_text_similarity(text1, text2)
        
        print(f"✓ Similarity between '{text1}' and '{text2}': {similarity:.3f}")
        
        print("\n✅ CLIP test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ CLIP test failed: {e}")
        return False


def test_recommender():
    """Test recommendation engine."""
    print("\n" + "="*60)
    print("Testing Recommender...")
    print("="*60)
    
    try:
        recommender = get_recommender()
        print(f"\n✓ Loaded {len(recommender.metadata)} clothing items")
        
        # Test recommendation
        outfits = recommender.recommend_outfits(
            fitzpatrick_type="III",
            preferred_style="old money",
            num_outfits=3
        )
        
        print(f"✓ Generated {len(outfits)} outfit recommendations")
        
        if len(outfits) > 0:
            print(f"\nTop outfit (score: {outfits[0]['score']}):")
            for item in outfits[0]['items']:
                print(f"  - {item['color']} {item['category']}")
            print(f"\nExplanation: {outfits[0]['explanation']}")
        
        print("\n✅ Recommender test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Recommender test failed: {e}")
        return False


def test_skin_tone():
    """Test skin tone detection (without actual image)."""
    print("\n" + "="*60)
    print("Testing Skin Tone Module...")
    print("="*60)
    
    try:
        from app.skin_tone import FITZPATRICK_DESCRIPTIONS
        print(f"\n✓ Loaded {len(FITZPATRICK_DESCRIPTIONS)} Fitzpatrick descriptions")
        
        for fitz_type, desc in FITZPATRICK_DESCRIPTIONS.items():
            print(f"  Type {fitz_type}: {desc}")
        
        print("\n✅ Skin tone module test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Skin tone test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("🎨 Fashion Recommendation Backend - Test Suite")
    print("="*60)
    
    results = []
    
    # Run tests
    test_config()
    results.append(("Skin Tone", test_skin_tone()))
    results.append(("CLIP", test_clip()))
    results.append(("Recommender", test_recommender()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Backend is ready to run.")
        print("\nTo start the server, run:")
        print("  uvicorn app.main:app --reload")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
