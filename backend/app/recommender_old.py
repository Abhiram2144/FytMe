"""
Fashion recommendation engine.
Filters, ranks, and assembles outfit combinations based on user preferences.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

from app.config import (
    SKIN_TONE_COLOR_MAP,
    OUTFIT_CATEGORIES,
    COLOR_HARMONY,
    SCORING_WEIGHTS,
    STYLE_COMPATIBILITY
)
from app.clip_utils import (
    get_clip_model,
    generate_clothing_description,
    generate_user_intent
)


class FashionRecommender:
    """Main recommendation engine for outfit suggestions."""
    
    def __init__(self, metadata_path: str = None):
        """
        Initialize recommender with clothing metadata.
        
        Args:
            metadata_path: Path to clothes_metadata.csv file (or 1000img.csv)
        """
        if metadata_path is None:
            # Default path - try 1000_class first, fallback to clothes_metadata.csv
            backend_dir = Path(__file__).parent.parent
            metadata_path_1000 = backend_dir / "data" / "1000_class" / "1000img.csv"
            metadata_path_legacy = backend_dir / "data" / "clothes_metadata.csv"
            
            if metadata_path_1000.exists():
                metadata_path = metadata_path_1000
                print(f"Using 1000_class dataset with {metadata_path_1000}")
            else:
                metadata_path = metadata_path_legacy
                print(f"Fallback to legacy dataset: {metadata_path_legacy}")
        
        print(f"Loading clothing metadata from {metadata_path}...")
        self.metadata = pd.read_csv(metadata_path)
        
        # Normalize column names if using 1000_class format
        if 'label' in self.metadata.columns:
            # Map 1000_class format to expected format
            self.metadata = self.metadata.rename(columns={'label': 'category'})
            # Add default values for missing columns
            if 'predicted_color' not in self.metadata.columns:
                self.metadata['predicted_color'] = ''
            if 'predicted_pattern' not in self.metadata.columns:
                self.metadata['predicted_pattern'] = 'solid'
            if 'predicted_style' not in self.metadata.columns:
                self.metadata['predicted_style'] = ''
        
        # Initialize CLIP model
        self.clip_model = get_clip_model()
        
        print(f"Loaded {len(self.metadata)} clothing items.")
        
    def recommend_outfits(
        self,
        fitzpatrick_type: str,
        preferred_style: str,
        num_outfits: int = 3
    ) -> List[Dict]:
        """
        Recommend complete outfits based on user preferences.
        
        Args:
            fitzpatrick_type: Fitzpatrick skin tone type (I-VI)
            preferred_style: User's preferred style/vibe
            num_outfits: Number of outfits to return
            
        Returns:
            List of outfit dictionaries with items, scores, and explanations
        """
        print(f"\nGenerating recommendations for Fitzpatrick {fitzpatrick_type}, style: {preferred_style}")
        
        # Get compatible colors for this skin tone
        compatible_colors = SKIN_TONE_COLOR_MAP.get(fitzpatrick_type, {}).get("colors", [])
        
        # Filter and rank items by category
        tops = self._filter_and_rank_items("top", preferred_style, compatible_colors)
        bottoms = self._filter_and_rank_items("bottom", preferred_style, compatible_colors)
        shoes = self._filter_and_rank_items("shoes", preferred_style, compatible_colors)

        # Graceful degradation: if any category is empty, return no outfits
        if min(len(tops), len(bottoms), len(shoes)) == 0:
            return []
        
        # Generate user intent for CLIP similarity
        user_intent = generate_user_intent(fitzpatrick_type, preferred_style)
        
        # Assemble outfit combinations
        outfits = []
        
        # Try to create multiple diverse outfits
        max_combinations = min(len(tops), len(bottoms), len(shoes), num_outfits * 3)
        
        for i in range(max_combinations):
            top_idx = i % len(tops)
            bottom_idx = i % len(bottoms)
            shoe_idx = i % len(shoes)
            
            top = tops.iloc[top_idx]
            bottom = bottoms.iloc[bottom_idx]
            shoe = shoes.iloc[shoe_idx]
            
            # Calculate outfit score
            outfit_score, explanation = self._score_outfit(
                top, bottom, shoe,
                preferred_style,
                compatible_colors,
                user_intent
            )
            
            outfit = {
                "items": [
                    self._item_to_dict(top),
                    self._item_to_dict(bottom),
                    self._item_to_dict(shoe)
                ],
                "score": round(outfit_score, 2),
                "explanation": explanation
            }
            
            outfits.append(outfit)
            
            if len(outfits) >= num_outfits * 2:  # Generate extra to pick best
                break
        
        # Sort by score and return top N
        outfits.sort(key=lambda x: x['score'], reverse=True)
        
        # Ensure diversity in top results
        final_outfits = self._diversify_outfits(outfits[:num_outfits * 2], num_outfits)
        
        return final_outfits
    
    def _filter_and_rank_items(
        self,
        category_type: str,
        preferred_style: str,
        compatible_colors: List[str]
    ) -> pd.DataFrame:
        """
        Filter and rank clothing items by category.
        
        Args:
            category_type: "top", "bottom", or "shoes"
            preferred_style: User's preferred style
            compatible_colors: List of skin-tone-compatible colors
            
        Returns:
            Filtered and ranked DataFrame of items
        """
        # Get valid categories for this type
        valid_categories = OUTFIT_CATEGORIES.get(category_type, [])
        
        # Filter by category
        items = self.metadata[
            self.metadata['category'].str.lower().isin(valid_categories)
        ].copy()
        
        if len(items) == 0:
            return pd.DataFrame()
        
        # Calculate base scores for each item
        scores = []
        for _, item in items.iterrows():
            score = self._calculate_item_score(
                item,
                preferred_style,
                compatible_colors
            )
            scores.append(score)
        
        items['score'] = scores
        
        # Sort by score descending
        items = items.sort_values('score', ascending=False)
        
        # Return top items (limit to avoid too many combinations)
        return items.head(10)
    
    def _calculate_item_score(
        self,
        item: pd.Series,
        preferred_style: str,
        compatible_colors: List[str]
    ) -> float:
        """
        Calculate a score for a single clothing item.
        
        Args:
            item: Item row from metadata DataFrame
            preferred_style: User's preferred style
            compatible_colors: List of compatible colors
            
        Returns:
            Item score (0 to 1)
        """
        score = 0.0
        
        # Style match (40% weight)
        item_style = str(item.get('predicted_style', '')).lower()
        if item_style == preferred_style.lower():
            score += 0.40
        elif preferred_style.lower() in STYLE_COMPATIBILITY:
            compatible_styles = STYLE_COMPATIBILITY[preferred_style.lower()]
            if item_style in compatible_styles:
                score += 0.25  # Partial credit for compatible styles
        
        # Color compatibility (30% weight)
        item_color = str(item.get('predicted_color', '')).lower()
        compatible_colors_lower = [c.lower() for c in compatible_colors]
        
        if item_color in compatible_colors_lower:
            score += 0.30
        elif any(color in item_color for color in compatible_colors_lower):
            score += 0.20  # Partial credit for color contains compatible color
        
        # Base score for having valid data (10% weight)
        if pd.notna(item.get('predicted_color')) and item.get('predicted_color') != '':
            score += 0.05
        if pd.notna(item.get('predicted_style')) and item.get('predicted_style') != '':
            score += 0.05
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _score_outfit(
        self,
        top: pd.Series,
        bottom: pd.Series,
        shoe: pd.Series,
        preferred_style: str,
        compatible_colors: List[str],
        user_intent: str
    ) -> Tuple[float, str]:
        """
        Score a complete outfit combination.
        
        Args:
            top, bottom, shoe: Item Series from metadata
            preferred_style: User's preferred style
            compatible_colors: List of compatible colors
            user_intent: User intent string for CLIP similarity
            
        Returns:
            Tuple of (score, explanation)
        """
        # Individual item scores
        top_score = self._calculate_item_score(top, preferred_style, compatible_colors)
        bottom_score = self._calculate_item_score(bottom, preferred_style, compatible_colors)
        shoe_score = self._calculate_item_score(shoe, preferred_style, compatible_colors)
        
        base_score = (top_score + bottom_score + shoe_score) / 3
        
        # Color harmony bonus
        harmony_bonus = self._check_color_harmony(top, bottom, shoe)
        
        # CLIP semantic similarity
        outfit_description = self._generate_outfit_description(top, bottom, shoe)
        clip_similarity = self.clip_model.compute_text_similarity(user_intent, outfit_description)
        
        # Combine scores using weights
        # Config-driven scoring weights (fallback to sane defaults)
        base_w = SCORING_WEIGHTS.get("base", 0.5)
        harmony_w = SCORING_WEIGHTS.get("harmony", 0.3)
        clip_w = SCORING_WEIGHTS.get("clip", 0.2)

        final_score = (
            base_score * base_w +
            harmony_bonus * harmony_w +
            clip_similarity * clip_w
        )
        
        # Generate explanation
        explanation = self._generate_explanation(
            top, bottom, shoe,
            preferred_style,
            compatible_colors,
            harmony_bonus > 0.5,
            clip_similarity
        )
        
        return final_score, explanation
    
    def _check_color_harmony(self, top: pd.Series, bottom: pd.Series, shoe: pd.Series) -> float:
        """
        Check if colors in outfit harmonize well together.
        
        Returns:
            Harmony score (0 to 1)
        """
        top_color = str(top.get('predicted_color', '')).lower()
        bottom_color = str(bottom.get('predicted_color', '')).lower()
        shoe_color = str(shoe.get('predicted_color', '')).lower()
        
        score = 0.0
        checks = 0
        
        # Check top-bottom harmony
        if top_color in COLOR_HARMONY:
            if bottom_color in COLOR_HARMONY[top_color]:
                score += 0.4
            checks += 1
        
        # Check bottom-shoe harmony
        if bottom_color in COLOR_HARMONY:
            if shoe_color in COLOR_HARMONY[bottom_color]:
                score += 0.3
            checks += 1
        
        # Check top-shoe harmony
        if top_color in COLOR_HARMONY:
            if shoe_color in COLOR_HARMONY[top_color]:
                score += 0.3
            checks += 1
        
        # Neutral colors (black, white, gray, beige) always harmonize
        neutral_colors = ['black', 'white', 'gray', 'grey', 'beige', 'cream']
        neutral_count = sum([
            any(n in top_color for n in neutral_colors),
            any(n in bottom_color for n in neutral_colors),
            any(n in shoe_color for n in neutral_colors)
        ])
        
        if neutral_count >= 2:
            score += 0.3
        
        return min(score, 1.0)
    
    def _generate_outfit_description(
        self,
        top: pd.Series,
        bottom: pd.Series,
        shoe: pd.Series
    ) -> str:
        """Generate natural language description of complete outfit."""
        top_desc = generate_clothing_description(top.to_dict())
        bottom_desc = generate_clothing_description(bottom.to_dict())
        shoe_desc = generate_clothing_description(shoe.to_dict())
        
        return f"{top_desc} with {bottom_desc} and {shoe_desc}"
    
    def _generate_explanation(
        self,
        top: pd.Series,
        bottom: pd.Series,
        shoe: pd.Series,
        preferred_style: str,
        compatible_colors: List[str],
        has_harmony: bool,
        clip_similarity: float
    ) -> str:
        """Generate human-readable explanation for outfit recommendation."""
        explanations = []
        
        # Style alignment
        top_style = str(top.get('predicted_style', '')).lower()
        if top_style == preferred_style.lower():
            explanations.append(f"This outfit perfectly matches your {preferred_style} aesthetic")
        
        # Color compatibility
        top_color = str(top.get('predicted_color', '')).lower()
        bottom_color = str(bottom.get('predicted_color', '')).lower()
        
        compatible_items = []
        if top_color in [c.lower() for c in compatible_colors]:
            compatible_items.append(f"{top_color} top")
        if bottom_color in [c.lower() for c in compatible_colors]:
            compatible_items.append(f"{bottom_color} bottom")
        
        if compatible_items:
            explanations.append(
                f"The {' and '.join(compatible_items)} complement your skin tone beautifully"
            )
        
        # Color harmony
        if has_harmony:
            explanations.append("The colors harmonize well together for a cohesive look")
        
        # CLIP semantic match
        if clip_similarity > 0.7:
            explanations.append("AI analysis confirms this combination aligns with your style preferences")
        
        # Fallback explanation
        if not explanations:
            explanations.append(f"A versatile {preferred_style} outfit that works well with your features")
        
        return ". ".join(explanations) + "."
    
    def _item_to_dict(self, item: pd.Series) -> Dict:
        """Convert item Series to dictionary for API response."""
        return {
            "image": str(item.get('image', '')),
            "category": str(item.get('category', '')),
            "color": str(item.get('predicted_color', '')),
            "pattern": str(item.get('predicted_pattern', '')),
            "style": str(item.get('predicted_style', ''))
        }
    
    def _diversify_outfits(self, outfits: List[Dict], num_outfits: int) -> List[Dict]:
        """
        Ensure diversity in outfit recommendations.
        Avoid recommending too similar outfits.
        
        Args:
            outfits: List of outfit candidates
            num_outfits: Number of final outfits to return
            
        Returns:
            Diversified list of outfits
        """
        if len(outfits) <= num_outfits:
            return outfits
        
        diverse_outfits = [outfits[0]]  # Always include top-scored outfit
        
        for outfit in outfits[1:]:
            # Check if this outfit is too similar to already selected ones
            is_diverse = True
            
            for selected in diverse_outfits:
                # Simple diversity check: compare colors
                outfit_colors = set([
                    item['color'].lower() for item in outfit['items']
                ])
                selected_colors = set([
                    item['color'].lower() for item in selected['items']
                ])
                
                # If more than 2 colors match, consider too similar
                if len(outfit_colors & selected_colors) > 2:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_outfits.append(outfit)
            
            if len(diverse_outfits) >= num_outfits:
                break
        
        # If we couldn't get enough diverse outfits, fill with highest scoring ones
        while len(diverse_outfits) < num_outfits and len(diverse_outfits) < len(outfits):
            for outfit in outfits:
                if outfit not in diverse_outfits:
                    diverse_outfits.append(outfit)
                    break
        
        return diverse_outfits[:num_outfits]


# Global recommender instance
_recommender = None


def get_recommender() -> FashionRecommender:
    """
    Get or initialize the global recommender instance.
    Uses singleton pattern.
    
    Returns:
        FashionRecommender instance
    """
    global _recommender
    if _recommender is None:
        _recommender = FashionRecommender()
    return _recommender
