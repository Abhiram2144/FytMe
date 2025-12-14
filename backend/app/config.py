"""
Configuration file for fashion recommendation system.
Contains mappings for skin tones, colors, styles, and outfit rules.
"""

# ============================================
# CLIP Zero-Shot Inference Prompts
# ============================================

# Category inference prompts
CATEGORY_PROMPTS = {
    "shirt": "a photo of a shirt",
    "t-shirt": "a photo of a t-shirt",
    "pants": "a photo of pants",
    "jeans": "a photo of jeans",
    "shoes": "a photo of shoes",
    "jacket": "a photo of a jacket",
    "dress": "a photo of a dress",
    "skirt": "a photo of a skirt",
    "shorts": "a photo of shorts",
    "hoodie": "a photo of a hoodie",
    "sweater": "a photo of a sweater"
}

# Color inference (will be contextualized: "a {color} {category}")
COLOR_PROMPTS = {
    "black": "black",
    "white": "white",
    "gray": "gray",
    "navy": "navy blue",
    "beige": "beige",
    "brown": "brown",
    "olive": "olive green",
    "burgundy": "burgundy",
    "khaki": "khaki",
    "cream": "cream",
    "blue": "blue",
    "red": "red",
    "green": "green",
    "yellow": "yellow",
    "pink": "pink"
}

# Style/vibe inference prompts
STYLE_PROMPTS = {
    "old money": "old money style",
    "casual": "casual style",
    "streetwear": "streetwear style",
    "formal": "formal professional attire"
}

# ============================================
# Skin Tone & Color Compatibility
# ============================================

# Fitzpatrick skin tone to compatible color mapping
SKIN_TONE_COLOR_MAP = {
    "I": {
        "colors": ["black", "navy", "charcoal", "white", "cream", "burgundy", "emerald", "royal blue"],
        "avoid": ["yellow", "orange", "bright pink"]
    },
    "II": {
        "colors": ["black", "navy", "gray", "white", "beige", "burgundy", "forest green", "cobalt"],
        "avoid": ["neon", "bright yellow"]
    },
    "III": {
        "colors": ["olive", "white", "brown", "camel", "rust", "teal", "burgundy", "navy", "khaki"],
        "avoid": ["pastel pink", "pale yellow"]
    },
    "IV": {
        "colors": ["olive", "brown", "terracotta", "white", "cream", "navy", "forest green", "gold", "burgundy"],
        "avoid": ["very pale colors"]
    },
    "V": {
        "colors": ["beige", "emerald", "camel", "gold", "rust", "cobalt", "burgundy", "white", "coral"],
        "avoid": ["pale pastels"]
    },
    "VI": {
        "colors": ["bright colors", "white", "yellow", "coral", "turquoise", "fuchsia", "emerald", "royal blue"],
        "avoid": ["brown", "olive"]
    }
}

# Fitzpatrick type descriptions
FITZPATRICK_DESCRIPTIONS = {
    "I": "very fair",
    "II": "fair",
    "III": "medium",
    "IV": "olive",
    "V": "brown",
    "VI": "dark brown"
}

# Valid style/vibe categories
VALID_STYLES = [
    "old money",
    "casual",
    "streetwear",
    "formal"
]

# Outfit composition rules
# Maps category types to actual category names in the dataset
OUTFIT_CATEGORIES = {
    "top": ["shirt", "t-shirt", "tshirts", "blouse", "sweater", "jacket", "jackets", 
            "hoodie", "longsleeve", "tops", "shirts"],
    "bottom": ["pants", "jeans", "skirt", "skirts", "shorts", "trousers", "dresses"],
    "shoes": ["shoes", "sneakers", "boots", "loafers", "sandals"]
}

# Style compatibility matrix (some styles work better together)
STYLE_COMPATIBILITY = {
    "old money": ["formal"],
    "casual": ["streetwear"],
    "streetwear": ["casual"],
    "formal": ["old money"]
}

# Color harmony rules (colors that work well together in an outfit)
COLOR_HARMONY = {
    "black": ["white", "gray", "navy", "beige", "cream"],
    "white": ["black", "navy", "gray", "beige", "brown"],
    "navy": ["white", "beige", "gray", "khaki", "cream"],
    "gray": ["black", "white", "navy", "burgundy"],
    "beige": ["white", "brown", "navy", "olive"],
    "brown": ["beige", "cream", "white", "khaki"],
    "olive": ["beige", "brown", "khaki", "white"],
    "burgundy": ["gray", "black", "navy", "cream"]
}

# Scoring weights for recommendation algorithm
SCORING_WEIGHTS = {
    "style_match": 0.40,      # Primary style match
    "color_compatibility": 0.30,  # Skin tone color compatibility
    "clip_similarity": 0.20,   # CLIP semantic similarity
    "color_harmony": 0.10      # Outfit color coordination
}

# CLIP model configuration
CLIP_MODEL_NAME = "ViT-B/32"
CLIP_DEVICE = "cuda" if __name__ == "__main__" else "cpu"  # Will check at runtime
