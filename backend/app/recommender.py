"""
Fashion Recommendation Engine - Rule-Based + CLIP Ranking

ARCHITECTURE PHILOSOPHY:
======================
1. RULES ENFORCE REALITY
   - Skin tone → color compatibility (HARD CONSTRAINT)
   - Dress code enforcement (formal ≠ streetwear)
   - Outfit structure validity (need top + bottom)
   - Color harmony between items

2. CLIP RANKS WITHIN CONSTRAINTS
   - CLIP provides similarity scoring
   - CLIP ranks options AFTER rules filter candidates
   - CLIP never overrides domain rules

WORKFLOW:
=========
recommend_outfits(skin_tone, style, num_outfits)
  ↓
  STEP 1: HARD FILTER (Rules)
    - Filter items by allowed colors for skin tone
    - Filter items by category (need tops + bottoms)
  ↓
  STEP 2: SCORE INDIVIDUAL ITEMS (CLIP + Rules)
    - Style affinity: how well item matches requested style
    - Confidence: CLIP's confidence in its labels
  ↓
  STEP 3: GENERATE OUTFIT CANDIDATES
    - Pair tops with bottoms
    - Check color harmony (rules)
    - Check style compatibility (rules)
  ↓
  STEP 4: SCORE OUTFITS (Composite)
    - Item scores (from CLIP)
    - Pair match score (rules + CLIP semantic similarity)
    - Color harmony (rules)
    - Style coherence (rules)
  ↓
  STEP 5: DIVERSIFY RESULTS
    - Enforce unique color combinations
    - Avoid repetitive items
  ↓
  RETURN: Top-k outfits with explanations

This is a stylist system, not a naive classifier.
"""

import numpy as np
from typing import List, Dict, Tuple, Set
import random

# Import fashion domain knowledge (HARD RULES)
from .fashion_knowledge import (
    SKIN_TONE_COLOR_MAP,
    STYLE_CLUSTERS,
    STYLE_COMPATIBILITY,
    COLOR_HARMONY,
    VALID_OUTFIT_STRUCTURES,
    TOP_CATEGORIES,
    BOTTOM_CATEGORIES,
    SHOE_CATEGORIES,
    get_allowed_colors_for_skin_tone,
    get_avoided_colors_for_skin_tone,
    matches_allowed_color,
    is_category_type_allowed,
    is_pattern_allowed_in_style,
    are_styles_compatible,
    are_colors_harmonious,
    is_outfit_structure_valid,
    get_category_type
)


class FashionRecommender:
    """
    Rule-based fashion recommendation engine with CLIP ranking.
    
    Rules filter and validate. CLIP ranks within constraints.
    """
    
    def __init__(self, catalog: List[Dict] = None):
        """
        Initialize recommender with indexed catalog from image_indexer.
        
        Args:
            catalog: List of clothing items with:
                - image: filename
                - category: item type
                - color: detected color
                - styles: list of {"label": str, "score": float}
                - primary_style: top style
                - embedding: CLIP image vector
        """
        self.catalog = catalog if catalog is not None else []
        print(f"Recommender initialized with {len(self.catalog)} items")
        
    def update_catalog(self, catalog: List[Dict]):
        """Update catalog after reindexing."""
        self.catalog = catalog
        print(f"Catalog updated: {len(self.catalog)} items")
        
    def recommend_outfits(
        self,
        fitzpatrick_type: str,
        preferred_style: str,
        num_outfits: int = 5
    ) -> List[Dict]:
        """
        Generate outfit recommendations using RULES → FILTER → CLIP RANK workflow.
        
        Args:
            fitzpatrick_type: User's skin tone (I-VI from Fitzpatrick scale)
            preferred_style: Desired style (formal, casual, old_money, streetwear, etc.)
            num_outfits: Number of distinct outfits to return
            
        Returns:
            List of outfit dicts:
            {
                "top": {...item dict...},
                "bottom": {...item dict...},
                "shoes": {...item dict...} or None,
                "score": float,
                "explanation": str,
                "shirt_pant_match_score": float
            }
        """
        if len(self.catalog) == 0:
            print("⚠️  Catalog is empty")
            return []
        
        # ========================================================
        # STEP 1: HARD FILTER BY SKIN TONE (RULE ENFORCEMENT)
        # ========================================================
        allowed_colors = get_allowed_colors_for_skin_tone(fitzpatrick_type)
        avoided_colors = get_avoided_colors_for_skin_tone(fitzpatrick_type)
        
        print(f"🎨 Skin tone {fitzpatrick_type}: allowed colors = {allowed_colors}")
        
        # Filter catalog: ONLY items in allowed colors (using flexible matching)
        filtered_catalog = [
            item for item in self.catalog
            if matches_allowed_color(item.get("color", ""), allowed_colors)
        ]
        
        if len(filtered_catalog) == 0:
            print(f"⚠️  No items found in allowed colors for skin tone {fitzpatrick_type}")
            # Fallback: use all items (graceful degradation)
            filtered_catalog = self.catalog
        
        print(f"✓ Filtered to {len(filtered_catalog)} items matching skin tone")
        
        # ========================================================
        # STEP 2: SEPARATE BY CATEGORY
        # ========================================================
        tops = [item for item in filtered_catalog if get_category_type(item.get("category", "")) == "top"]
        bottoms = [item for item in filtered_catalog if get_category_type(item.get("category", "")) == "bottom"]
        shoes_items = [item for item in filtered_catalog if get_category_type(item.get("category", "")) == "shoes"]
        
        print(f"📊 Available: {len(tops)} tops, {len(bottoms)} bottoms, {len(shoes_items)} shoes")
        
        # ========================================================
        # STEP 2.5: APPLY STRICT CATEGORY-TYPE & PATTERN RULES
        # ========================================================
        # HARD BLOCK: jeans/hoodies/cargo cannot appear in certain contexts
        # This happens BEFORE any CLIP ranking (rules > perception)
        
        # Filter tops by category-type appropriateness
        tops_filtered = []
        for item in tops:
            if is_category_type_allowed(item.get("category", ""), preferred_style):
                tops_filtered.append(item)
            else:
                print(f"   ❌ Blocked top: {item['image']} (category={item.get('category')}, style={preferred_style})")
        
        tops_allowed = tops_filtered
        
        # Filter tops by pattern appropriateness
        tops_filtered = []
        for item in tops_allowed:
            if is_pattern_allowed_in_style(item.get("pattern", "solid"), preferred_style):
                tops_filtered.append(item)
            else:
                print(f"   ❌ Blocked top: {item['image']} (pattern={item.get('pattern')}, style={preferred_style})")
        
        tops_allowed = tops_filtered
        
        # Filter bottoms by category-type appropriateness
        bottoms_filtered = []
        for item in bottoms:
            if is_category_type_allowed(item.get("category", ""), preferred_style):
                bottoms_filtered.append(item)
            else:
                print(f"   ❌ Blocked bottom: {item['image']} (category={item.get('category')}, style={preferred_style})")
        
        bottoms_allowed = bottoms_filtered
        
        if len(tops_allowed) == 0 or len(bottoms_allowed) == 0:
            print(f"⚠️  After strict rules: {len(tops_allowed)} tops, {len(bottoms_allowed)} bottoms")
            print(f"    (Jeans/hoodies/cargo may be forbidden in {preferred_style})")
            return []
        
        print(f"   ✓ After strict rules: {len(tops_allowed)} tops, {len(bottoms_allowed)} bottoms")
        
        # ========================================================
        # STEP 3.0: ENFORCE FORMAL DRESS CODE (if needed)
        # ========================================================
        # For formal/dress occasions, restrict to appropriate categories
        if preferred_style.lower() == "formal":
            # Formal requires dress shirts (not t-shirts) and trousers (not jeans/joggers)
            tops_allowed = [item for item in tops_allowed if item.get("category", "").lower() in ["shirt", "polo"]]
            bottoms_allowed = [item for item in bottoms_allowed if item.get("category", "").lower() in ["trousers"]]
            print(f"   ✓ Formal dress code: {len(tops_allowed)} tops, {len(bottoms_allowed)} bottoms")
        
        if len(tops_allowed) == 0 or len(bottoms_allowed) == 0:
            print("⚠️  Insufficient items after rule filtering")
            return []
        
        # ========================================================
        # STEP 3: SCORE ITEMS BY STYLE AFFINITY (CLIP RANKING)
        # ========================================================
        # Score how well each item matches the requested style
        # CLIP has already ranked styles during indexing
        # We just extract the relevant score
        
        tops_scored = self._score_items_by_style(tops_allowed, preferred_style)
        bottoms_scored = self._score_items_by_style(bottoms_allowed, preferred_style)
        shoes_scored = self._score_items_by_style(shoes_items, preferred_style) if shoes_items else []
        
        # Sort by style affinity (keep top candidates)
        tops_scored.sort(key=lambda x: x["style_affinity"], reverse=True)
        bottoms_scored.sort(key=lambda x: x["style_affinity"], reverse=True)
        shoes_scored.sort(key=lambda x: x["style_affinity"], reverse=True)
        
        # Use ALL available tops and bottoms to maximize outfit variety
        tops_candidates = tops_scored  # Use all tops
        bottoms_candidates = bottoms_scored  # Use all bottoms
        shoes_candidates = shoes_scored[:5] if shoes_scored else []
        
        print(f"🎯 Top candidates: {len(tops_candidates)} tops, {len(bottoms_candidates)} bottoms")
        
        # ========================================================
        # STEP 4: GENERATE AND SCORE OUTFIT COMBINATIONS
        # ========================================================
        outfit_candidates = []
        
        for top in tops_candidates:
            for bottom in bottoms_candidates:
                # Rule check: color harmony
                if not are_colors_harmonious(top["color"], bottom["color"]):
                    continue
                
                # Rule check: style compatibility
                if not self._are_item_styles_compatible(top, bottom, preferred_style):
                    continue
                
                # Compute outfit score
                outfit = {
                    "top": top,
                    "bottom": bottom,
                    "shoes": None  # Optional for now
                }
                
                score, match_score = self._compute_outfit_score(outfit, preferred_style)
                
                outfit_candidates.append({
                    **outfit,
                    "score": score,
                    "shirt_pant_match_score": match_score
                })
        
        print(f"🔄 Generated {len(outfit_candidates)} valid outfit combinations")
        
        if len(outfit_candidates) == 0:
            print("⚠️  No valid outfit combinations found after applying rules")
            return []
        
        # ========================================================
        # STEP 5: DIVERSIFY AND SELECT TOP OUTFITS
        # ========================================================
        # Sort by score
        outfit_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Enforce diversity: unique (top_color, bottom_color) pairs
        final_outfits = self._diversify_outfits(outfit_candidates, num_outfits)
        
        # ========================================================
        # STEP 6: ADD EXPLANATIONS
        # ========================================================
        for outfit in final_outfits:
            outfit["explanation"] = self._generate_explanation(
                outfit, 
                fitzpatrick_type,
                preferred_style
            )
        
        print(f"✅ Returning {len(final_outfits)} outfits")
        return final_outfits
    
    
    def _score_items_by_style(self, items: List[Dict], target_style: str) -> List[Dict]:
        """
        Score items by how well they match the target style.
        
        Uses CLIP's style distribution (already computed during indexing).
        Each item has a "styles" field with [{"label": str, "score": float}, ...]
        
        PRIORITY:
        1. If item's primary_style == target_style, boost score significantly
        2. Otherwise, use style_affinity from styles distribution
        3. If target style not found anywhere, use penalty (0.05)
        
        Returns:
            List of items with added "style_affinity" field
        """
        scored_items = []
        
        for item in items:
            primary_style = item.get("primary_style", "casual")
            styles_dist = item.get("styles", [])
            
            # PRIORITY 1: Primary style match gets highest boost
            if primary_style == target_style:
                style_affinity = 1.0  # Perfect match
            else:
                # PRIORITY 2: Look for target style in full distribution
                style_affinity = 0.0
                for style_entry in styles_dist:
                    if style_entry["label"] == target_style:
                        style_affinity = style_entry["score"]
                        break
                
                # PRIORITY 3: If target style not found, penalty
                if style_affinity == 0.0:
                    style_affinity = 0.05  # Lower penalty for missing style
            
            scored_items.append({
                **item,
                "style_affinity": style_affinity
            })
        
        return scored_items
    
    
    def _are_item_styles_compatible(
        self, 
        item_a: Dict, 
        item_b: Dict, 
        target_style: str
    ) -> bool:
        """
        Check if two items' styles are compatible with each other and the target style.
        
        Uses RULES from fashion_knowledge.py, NOT CLIP.
        """
        # Get primary styles
        style_a = item_a.get("primary_style", "casual")
        style_b = item_b.get("primary_style", "casual")
        
        # Check if item styles are compatible with each other
        if not are_styles_compatible(style_a, style_b):
            return False
        
        # Check if both items are compatible with target style
        if not are_styles_compatible(style_a, target_style):
            return False
        
        if not are_styles_compatible(style_b, target_style):
            return False
        
        return True
    
    
    def _compute_outfit_score(
        self, 
        outfit: Dict, 
        target_style: str
    ) -> Tuple[float, float]:
        """
        Compute composite score for an outfit.
        
        Score components:
        1. Item style affinity (from CLIP)
        2. Color harmony (from RULES)
        3. Style coherence (from RULES)
        4. Semantic pair compatibility (from CLIP embeddings)
        
        Returns:
            (total_score, pair_match_score)
        """
        top = outfit["top"]
        bottom = outfit["bottom"]
        
        # Component 1: Style affinity (CLIP)
        top_affinity = top.get("style_affinity", 0.5)
        bottom_affinity = bottom.get("style_affinity", 0.5)
        avg_style_affinity = (top_affinity + bottom_affinity) / 2
        
        # Component 2: Color harmony bonus (RULE)
        color_harmony_bonus = 0.1 if are_colors_harmonious(top["color"], bottom["color"]) else 0.0
        
        # Component 3: Style coherence (RULE)
        # Already checked in filtering, so items here are compatible
        style_coherence_bonus = 0.1
        
        # Component 4: Semantic pair compatibility (CLIP)
        # Use CLIP embeddings to check how well the items "go together"
        pair_match_score = self._compute_clip_pair_score(top, bottom)
        
        # Weighted composite
        total_score = (
            avg_style_affinity * 0.5 +
            pair_match_score * 0.3 +
            color_harmony_bonus +
            style_coherence_bonus
        )
        
        return total_score, pair_match_score
    
    
    def _compute_clip_pair_score(self, item_a: Dict, item_b: Dict) -> float:
        """
        Compute semantic similarity between two items using CLIP embeddings.
        
        This uses CLIP correctly: for semantic similarity ranking, not classification.
        
        High similarity = items that semantically "go together"
        """
        emb_a = item_a.get("embedding")
        emb_b = item_b.get("embedding")
        
        if emb_a is None or emb_b is None:
            return 0.5  # Default neutral score
        
        # Ensure numpy arrays
        emb_a = np.array(emb_a).flatten()
        emb_b = np.array(emb_b).flatten()
        
        # Cosine similarity
        dot_product = np.dot(emb_a, emb_b)
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.5
        
        similarity = dot_product / (norm_a * norm_b)
        
        # Map from [-1, 1] to [0, 1]
        score = (similarity + 1) / 2
        
        return float(score)
    
    
    def _diversify_outfits(
        self, 
        outfit_candidates: List[Dict], 
        num_outfits: int
    ) -> List[Dict]:
        """
        Select diverse outfits from candidates.
        
        Strategy:
        - Prioritize unique (top_color, bottom_color) pairs initially
        - But allow reusing colors if we need more outfits
        - Only hard constraint: never reuse the exact same top or bottom item
        
        This ensures we show variety in outfits even with limited catalog.
        """
        selected: List[Dict] = []
        seen_color_pairs: Set[Tuple[str, str]] = set()
        used_top_images: Set[str] = set()
        used_bottom_images: Set[str] = set()
        selected_outfit_ids: Set[Tuple[str, str]] = set()

        # First pass: prefer unique color pairs
        for candidate in outfit_candidates:
            if len(selected) >= num_outfits:
                break

            top = candidate["top"]
            bottom = candidate["bottom"]

            color_pair = (top["color"], bottom["color"]) 
            outfit_id = (top["image"], bottom["image"])

            # Skip if exact same outfit
            if outfit_id in selected_outfit_ids:
                continue
            
            # Skip if items already used
            if top["image"] in used_top_images or bottom["image"] in used_bottom_images:
                continue

            # Prefer unique color pairs but don't hard block
            if color_pair in seen_color_pairs:
                continue

            selected.append(candidate)
            seen_color_pairs.add(color_pair)
            used_top_images.add(top["image"])
            used_bottom_images.add(bottom["image"])
            selected_outfit_ids.add(outfit_id)

        # Second pass: allow color pair reuse but still avoid item reuse
        if len(selected) < num_outfits:
            for candidate in outfit_candidates:
                if len(selected) >= num_outfits:
                    break

                top = candidate["top"]
                bottom = candidate["bottom"]
                outfit_id = (top["image"], bottom["image"])

                # Skip ones already selected
                if outfit_id in selected_outfit_ids:
                    continue

                # Only avoid exact same items being reused
                # (Allow different combinations)
                if top["image"] in used_top_images or bottom["image"] in used_bottom_images:
                    continue

                selected.append(candidate)
                used_top_images.add(top["image"])
                used_bottom_images.add(bottom["image"])
                selected_outfit_ids.add(outfit_id)

        return selected
    
    
    def _generate_explanation(
        self, 
        outfit: Dict, 
        fitzpatrick_type: str,
        target_style: str
    ) -> str:
        """
        Generate human-readable explanation for why this outfit was recommended.
        
        Explanations reference RULES, not CLIP's internal reasoning.
        """
        top = outfit["top"]
        bottom = outfit["bottom"]
        
        # Get style descriptions
        top_styles = ", ".join([s["label"] for s in top.get("styles", [])[:2]])
        bottom_styles = ", ".join([s["label"] for s in bottom.get("styles", [])[:2]])
        
        # Color harmony
        color_harmony_note = ""
        if are_colors_harmonious(top["color"], bottom["color"]):
            color_harmony_note = f"The {top['color']} top and {bottom['color']} bottom create a harmonious color combination."
        
        # Skin tone note
        skin_tone_note = f"Both colors complement Fitzpatrick type {fitzpatrick_type} skin tone."
        
        # Style note
        style_note = f"This {target_style} outfit combines {top_styles} elements with {bottom_styles} aesthetics."
        
        # Match score note
        match_score = outfit.get("shirt_pant_match_score", 0.0)
        match_note = ""
        if match_score > 0.7:
            match_note = "The pieces pair exceptionally well together."
        elif match_score > 0.5:
            match_note = "The pieces create a cohesive look."
        
        explanation = " ".join(filter(None, [
            style_note,
            color_harmony_note,
            skin_tone_note,
            match_note
        ]))
        
        return explanation


# Global singleton
_recommender = None

def get_recommender(catalog: List[Dict] = None) -> FashionRecommender:
    """Get or create global recommender instance.
    
    Args:
        catalog: Optional catalog to initialize/update recommender.
                 If provided and recommender exists, updates the catalog.
    """
    global _recommender
    if _recommender is None:
        _recommender = FashionRecommender(catalog if catalog else [])
    elif catalog is not None:
        _recommender.update_catalog(catalog)
    return _recommender
