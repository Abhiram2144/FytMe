"""
Fashion recommendation engine - In-Memory Catalog Version.
Uses automatically indexed clothing images with CLIP inference.
"""

import numpy as np
from typing import List, Dict, Tuple
import random

from app.config import (
    SKIN_TONE_COLOR_MAP,
    FITZPATRICK_DESCRIPTIONS,
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
    """Main recommendation engine using in-memory catalog."""
    
    def __init__(self, catalog: List[Dict] = None):
        """
        Initialize recommender with indexed catalog.
        
        Args:
            catalog: List of dicts from ClothingIndexer
        """
        self.catalog = catalog if catalog is not None else []
        print(f"Recommender initialized with {len(self.catalog)} items")
        
    def update_catalog(self, catalog: List[Dict]):
        """Update catalog (called when indexer finishes)."""
        self.catalog = catalog
        print(f"Catalog updated: {len(self.catalog)} items")
        
    def recommend_outfits(
        self,
        fitzpatrick_type: str,
        preferred_style: str,
        num_outfits: int = 3
    ) -> List[Dict]:
        """
        Generate outfit recommendations.
        
        Args:
            fitzpatrick_type: User's Fitzpatrick skin tone (I-VI)
            preferred_style: Desired fashion style
            num_outfits: Number of outfits to return
            
        Returns:
            List of outfit dicts with items, score, explanation
        """
        if len(self.catalog) == 0:
            print("Warning: Catalog is empty")
            return []
        
        # Get compatible colors for skin tone
        compatible_colors = SKIN_TONE_COLOR_MAP.get(fitzpatrick_type, {}).get("colors", [])
        
        # Filter and score all items
        tops = self._filter_and_score(
            category_group="top",
            style=preferred_style,
            compatible_colors=compatible_colors,
            fitzpatrick_type=fitzpatrick_type
        )
        
        bottoms = self._filter_and_score(
            category_group="bottom",
            style=preferred_style,
            compatible_colors=compatible_colors,
            fitzpatrick_type=fitzpatrick_type
        )
        
        shoes = self._filter_and_score(
            category_group="shoes",
            style=preferred_style,
            compatible_colors=compatible_colors,
            fitzpatrick_type=fitzpatrick_type
        )

        # Require at least one top and one bottom; shoes optional
        if len(tops) == 0 or len(bottoms) == 0:
            print(f"Warning: Missing required category (tops={len(tops)}, bottoms={len(bottoms)})")
            return []

        # Assemble outfits (with or without shoes)
        outfits = self._assemble_outfits(tops, bottoms, shoes, fitzpatrick_type, preferred_style)
        
        # Diversify results
        outfits = self._diversify_outfits(outfits, num_outfits * 2)
        
        # Return top N
        return outfits[:num_outfits]
    
    def _filter_and_score(
        self,
        category_group: str,
        style: str,
        compatible_colors: List[str],
        fitzpatrick_type: str
    ) -> List[Dict]:
        """
        Filter items by category and score them.
        
        Args:
            category_group: "top", "bottom", or "shoes"
            style: Preferred style
            compatible_colors: Colors that work with skin tone
            fitzpatrick_type: User's skin tone
            
        Returns:
            List of scored items sorted by score
        """
        # Category mapping
        category_map = {
            "top": ["shirt", "t-shirt", "jacket", "hoodie", "sweater", "blouse"],
            "bottom": ["pants", "jeans", "skirt", "shorts", "dress"],
            "shoes": ["shoes", "sneakers", "boots", "loafers", "sandals"]
        }
        
        valid_categories = category_map.get(category_group, [])
        
        # Filter items
        filtered = [
            item for item in self.catalog
            if item["category"] in valid_categories
        ]
        
        # Score each item
        scored_items = []
        for item in filtered:
            score = self._calculate_item_score(
                item,
                style,
                compatible_colors,
                fitzpatrick_type
            )
            scored_items.append({
                **item,
                "base_score": score
            })
        
        # Sort by score
        scored_items.sort(key=lambda x: x["base_score"], reverse=True)
        
        return scored_items
    
    def _calculate_item_score(
        self,
        item: Dict,
        preferred_style: str,
        compatible_colors: List[str],
        fitzpatrick_type: str
    ) -> float:
        """
        Calculate item score based on multiple factors.
        
        Returns:
            Float score (0-1)
        """
        # Style match (40%) using distributions
        style_score = self._style_alignment(item, preferred_style)
        
        # Color compatibility (30%)
        color_score = 1.0 if item["color"] in compatible_colors else 0.3
        
        # Weighted average
        final_score = (
            style_score * SCORING_WEIGHTS.get("style_match", 0.4) +
            color_score * SCORING_WEIGHTS.get("color_compatibility", 0.3) +
            0.5 * SCORING_WEIGHTS.get("clip_similarity", 0.2) +  # Placeholder for CLIP
            0.7 * SCORING_WEIGHTS.get("color_harmony", 0.1)  # Base harmony score
        )
        
        return final_score

    def _style_dict(self, item: Dict) -> Dict[str, float]:
        styles = item.get("styles", []) or []
        return {s.get("label"): s.get("score", 0.0) for s in styles if s.get("label")}

    def _style_alignment(self, item: Dict, target_style: str) -> float:
        styles = self._style_dict(item)
        if not styles:
            # Fallback to single style label if present
            if item.get("style") == target_style:
                return 1.0
            if item.get("style") in STYLE_COMPATIBILITY.get(target_style, []):
                return 0.7
            return 0.3

        direct = styles.get(target_style, 0.0)
        compat = max([styles.get(s, 0.0) for s in STYLE_COMPATIBILITY.get(target_style, [])] or [0.0])
        # Combine: primary weight + partial credit for compatible styles
        score = direct + 0.6 * compat
        return float(max(0.0, min(1.0, score)))

    def _primary_style(self, item: Dict) -> str:
        styles = item.get("styles", []) or []
        if styles:
            return styles[0].get("label", item.get("style", "unknown"))
        return item.get("style", "unknown")
    
    def _assemble_outfits(
        self,
        tops: List[Dict],
        bottoms: List[Dict],
        shoes: List[Dict],
        fitzpatrick_type: str,
        preferred_style: str,
        max_combinations: int = 50
    ) -> List[Dict]:
        """
        Assemble and score outfit combinations.
        
        Returns:
            List of outfit dicts sorted by score
        """
        outfits = []
        
        # Try top N items from each category
        top_items = min(10, len(tops))
        bottom_items = min(10, len(bottoms))
        shoe_items = min(10, len(shoes)) if len(shoes) > 0 else 0

        combinations = 0
        if shoe_items > 0:
            for top in tops[:top_items]:
                for bottom in bottoms[:bottom_items]:
                    for shoe in shoes[:shoe_items]:
                        if combinations >= max_combinations:
                            break

                        outfit = {
                            "items": [
                                {
                                    "image": top["image"],
                                    "category": top["category"],
                                    "color": top["color"],
                                    "style": top["style"]
                                },
                                {
                                    "image": bottom["image"],
                                    "category": bottom["category"],
                                    "color": bottom["color"],
                                    "style": bottom["style"]
                                },
                                {
                                    "image": shoe["image"],
                                    "category": shoe["category"],
                                    "color": shoe["color"],
                                    "style": shoe["style"]
                                }
                            ]
                        }

                        score, explanation, pair_score = self._score_outfit(
                            outfit,
                            fitzpatrick_type,
                            preferred_style,
                            top, bottom, shoe
                        )

                        outfit["score"] = score
                        outfit["explanation"] = explanation
                        outfit["shirt_pant_match_score"] = pair_score
                        outfits.append(outfit)
                        combinations += 1
        else:
            for top in tops[:top_items]:
                for bottom in bottoms[:bottom_items]:
                    if combinations >= max_combinations:
                        break

                    outfit = {
                        "items": [
                            {
                                "image": top["image"],
                                "category": top["category"],
                                "color": top["color"],
                                "style": top["style"]
                            },
                            {
                                "image": bottom["image"],
                                "category": bottom["category"],
                                "color": bottom["color"],
                                "style": bottom["style"]
                            }
                        ]
                    }

                    score, explanation, pair_score = self._score_outfit(
                        outfit,
                        fitzpatrick_type,
                        preferred_style,
                        top, bottom, None
                    )

                    outfit["score"] = score
                    outfit["explanation"] = explanation
                    outfit["shirt_pant_match_score"] = pair_score
                    outfits.append(outfit)
                    combinations += 1
        
        # Sort by score
        outfits.sort(key=lambda x: x["score"], reverse=True)
        
        return outfits
    
    def _score_outfit(
        self,
        outfit: Dict,
        fitzpatrick_type: str,
        preferred_style: str,
        top: Dict,
        bottom: Dict,
        shoe: Dict = None
    ) -> Tuple[float, str, float]:
        """
        Score complete outfit and generate explanation.
        
        Returns:
            (score, explanation)
        """
        # Base score (average of individual item scores)
        scores = [top["base_score"], bottom["base_score"]]
        if shoe is not None:
            scores.append(shoe["base_score"])
        base_score = np.mean(scores)
        
        # Color harmony bonus
        if shoe is not None:
            harmony_score = self._calculate_color_harmony(
                top["color"],
                bottom["color"],
                shoe["color"]
            )
        else:
            harmony_score = self._calculate_color_harmony_two_piece(
                top["color"],
                bottom["color"]
            )
        
        # Pairwise shirt–pant match score
        pair_score = self.compute_pair_match_score(top, bottom, preferred_style)

        # CLIP semantic similarity (using user intent) for full outfit
        clip_score = self._calculate_clip_similarity(outfit, fitzpatrick_type, preferred_style)
        
        # Final weighted score with pairwise emphasis
        final_score = (
            0.4 * base_score +
            0.35 * pair_score +
            0.25 * clip_score
        )
        
        # Generate explanation
        explanation = self._generate_explanation(
            outfit,
            preferred_style,
            fitzpatrick_type,
            final_score
        )
        
        return final_score, explanation, pair_score

    def _is_neutral(self, color: str) -> bool:
        return color in {"white", "beige", "gray", "black"}

    def compute_pair_match_score(self, shirt: Dict, pant: Dict, preferred_style: str) -> float:
        """Compute stylistic match score for shirt–pant pair in [0,1]."""
        shirt_color = shirt.get("color", "unknown")
        pant_color = pant.get("color", "unknown")
        shirt_style_primary = self._primary_style(shirt)
        pant_style_primary = self._primary_style(pant)
        shirt_styles = self._style_dict(shirt)
        pant_styles = self._style_dict(pant)

        # 1) Color harmony (40%) with neutral handling and identical penalty
        color_score = 0.0
        if pant_color in COLOR_HARMONY.get(shirt_color, []):
            color_score = 1.0
        else:
            color_score = 0.4  # base partial credit if not strongly harmonious

        # Neutral adjustment: two neutrals together get moderate score unless paired with a non-neutral
        shirt_neutral = self._is_neutral(shirt_color)
        pant_neutral = self._is_neutral(pant_color)
        if shirt_neutral and pant_neutral:
            color_score = min(color_score, 0.7)
        # Identical colors penalty
        if shirt_color == pant_color:
            color_score = max(0.0, color_score - 0.2)

        # 2) Style coherence (30%) using distributions
        shared_overlap = 0.0
        for lbl, s_score in shirt_styles.items():
            shared_overlap = max(shared_overlap, min(s_score, pant_styles.get(lbl, 0.0)))

        target_align = self._style_alignment(shirt, preferred_style) * self._style_alignment(pant, preferred_style)
        compat_align = 0.0
        for lbl, s_score in shirt_styles.items():
            for comp in STYLE_COMPATIBILITY.get(lbl, []):
                compat_align = max(compat_align, min(s_score, pant_styles.get(comp, 0.0)) * 0.7)

        style_score = max(shared_overlap, target_align, compat_align)
        if style_score == 0:
            style_score = 0.3

        # 3) CLIP semantic compatibility (30%)
        try:
            clip_wrapper = get_clip_model()
            pair_desc = f"{shirt_color} {shirt_style_primary} shirt with {pant_color} {pant_style_primary} pants"
            target = f"{preferred_style} outfit"
            clip_sim = float(clip_wrapper.compute_text_similarity(target, pair_desc))
        except Exception as e:
            print(f"Pair CLIP error: {e}")
            clip_sim = 0.5

        pair_score = 0.4 * color_score + 0.3 * style_score + 0.3 * clip_sim
        return float(max(0.0, min(1.0, pair_score)))
    
    def _calculate_color_harmony_two_piece(self, color1: str, color2: str) -> float:
        """Color harmony for two-piece outfit (top/bottom)."""
        harmony_bonus = 0.0
        if color2 in COLOR_HARMONY.get(color1, []):
            harmony_bonus += 0.6
        return min(harmony_bonus, 1.0)

    def _calculate_color_harmony(self, color1: str, color2: str, color3: str) -> float:
        """Color harmony for three-piece outfit (top/bottom/shoes)."""
        harmony_bonus = 0.0
        if color2 in COLOR_HARMONY.get(color1, []):
            harmony_bonus += 0.4
        if color3 in COLOR_HARMONY.get(color1, []):
            harmony_bonus += 0.3
        if color3 in COLOR_HARMONY.get(color2, []):
            harmony_bonus += 0.3
        return min(harmony_bonus, 1.0)
    
    def _calculate_clip_similarity(
        self,
        outfit: Dict,
        fitzpatrick_type: str,
        preferred_style: str
    ) -> float:
        """
        Calculate CLIP semantic similarity between outfit and user intent.
        
        Returns:
            Float score (0-1)
        """
        try:
            # Generate descriptions
            outfit_desc = generate_clothing_description(outfit["items"])
            user_intent = generate_user_intent(preferred_style, fitzpatrick_type)
            
            # Get CLIP model
            clip_wrapper = get_clip_model()
            
            # Compute similarity
            similarity = clip_wrapper.compute_text_similarity(user_intent, outfit_desc)
            
            return float(similarity)
            
        except Exception as e:
            print(f"CLIP similarity error: {e}")
            return 0.5  # Neutral score on error
    
    def _generate_explanation(
        self,
        outfit: Dict,
        preferred_style: str,
        fitzpatrick_type: str,
        score: float
    ) -> str:
        """Generate natural language explanation for outfit."""
        
        items = outfit["items"]
        top_color = items[0]["color"]
        bottom_color = items[1]["color"]
        shoe_color = items[2]["color"] if len(items) > 2 else None
        
        skin_desc = FITZPATRICK_DESCRIPTIONS.get(fitzpatrick_type, "medium")
        
        parts = [
            f"This {preferred_style} outfit perfectly suits your style.",
            f"The {top_color} top and {bottom_color} bottom create a harmonious look."
        ]

        if shoe_color:
            parts.append(f"The {shoe_color} footwear ties the palette together.")

        parts.append(f"These colors complement your {skin_desc} skin tone beautifully.")
        parts.append(f"Confidence score: {int(score * 100)}%")
        
        return " ".join(parts)
    
    def _diversify_outfits(self, outfits: List[Dict], max_results: int) -> List[Dict]:
        """
        Ensure diversity in outfit recommendations by color combination and avoiding repeated images.
        
        Returns:
            Diversified list of outfits
        """
        if len(outfits) == 0:
            return []
        
        diversified = []
        used_images = set()
        used_color_pairs = set()
        
        for outfit in outfits:
            items = outfit["items"]
            # color combination key (top, bottom)
            if len(items) < 2:
                continue
            color_key = (items[0]["color"], items[1]["color"])  # (shirt_color, pant_color)
            images = tuple(item["image"] for item in items)

            if color_key in used_color_pairs:
                continue
            if images in used_images:
                continue

            diversified.append(outfit)
            used_color_pairs.add(color_key)
            used_images.add(images)

            if len(diversified) >= max_results:
                break
        
        return diversified


# Global singleton
_recommender = None

def get_recommender(catalog: List[Dict] = None) -> FashionRecommender:
    """Get or create global recommender instance."""
    global _recommender
    if _recommender is None:
        _recommender = FashionRecommender(catalog)
    elif catalog is not None:
        _recommender.update_catalog(catalog)
    return _recommender
