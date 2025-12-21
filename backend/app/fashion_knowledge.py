"""
Fashion Knowledge Base - Domain Rules and Embedding Spaces

This module defines the foundational fashion logic that governs the recommendation system.

PHILOSOPHY:
- CLIP ranks semantic similarity
- RULES enforce reality

CLIP is a similarity ranking engine, NOT a classifier.
It must never override domain rules defined here.
"""

# ========================================================
# PART 1: SKIN TONE → COLOR COMPATIBILITY (HARD RULES)
# ========================================================

# Based on the Fitzpatrick scale (I–VI)
# These are deterministic, rule-based mappings that CLIP cannot override.
# CLIP may only rank items WITHIN the allowed colors.

SKIN_TONE_COLOR_MAP = {
    "I": {  # Very Fair (pale white, blond or red hair, blue eyes, freckles)
        "recommended": [
            "navy", "charcoal", "forest green", "burgundy", "cream",
            "slate blue", "deep purple", "chocolate brown", "emerald",
            "soft pink", "lavender", "grey"
        ],
        "avoid": [
            "neon", "bright orange", "mustard yellow", "lime green",
            "harsh white", "black" # can be too harsh against very pale skin
        ],
        "reasoning": "Avoid overwhelming pale complexions; prefer rich, deep colors that create contrast without harshness"
    },
    
    "II": {  # Fair (white skin, light eyes, blonde/light brown hair)
        "recommended": [
            "navy", "charcoal", "forest green", "burgundy", "cream",
            "taupe", "slate", "maroon", "teal", "dusty rose",
            "camel", "olive", "grey", "denim blue"
        ],
        "avoid": [
            "neon", "bright orange", "overly pale pastels"
        ],
        "reasoning": "Similar to Type I but can handle slightly bolder colors; still avoid harsh neons"
    },
    
    "III": {  # Medium (cream white, any eye/hair color)
        "recommended": [
            "olive", "camel", "maroon", "teal", "off-white",
            "rust", "khaki", "navy", "brown", "burnt orange",
            "sage green", "plum", "denim", "charcoal", "beige"
        ],
        "avoid": [
            "overly pale colors that wash out", "harsh neon"
        ],
        "reasoning": "Most versatile skin tone; can handle both warm and cool colors"
    },
    
    "IV": {  # Olive (moderate brown, Mediterranean/Asian descent)
        "recommended": [
            "khaki", "brown", "rust", "navy", "cream",
            "olive", "terracotta", "warm grey", "burgundy",
            "mustard", "burnt sienna", "forest green", "camel",
            "coral", "white"
        ],
        "avoid": [
            "neon pink", "lime green", "overly cool tones"
        ],
        "reasoning": "Warm undertones suit earthy, warm colors; can handle rich jewel tones"
    },
    
    "V": {  # Brown (dark brown skin, dark eyes/hair)
        "recommended": [
            "white", "camel", "emerald", "cobalt", "mustard",
            "burgundy", "bright red", "royal blue", "cream",
            "gold", "orange", "fuchsia", "turquoise", "purple",
            "coral", "lime", "bright pink"
        ],
        "avoid": [
            "muddy browns", "dull olive", "washed out grey"
        ],
        "reasoning": "Rich skin tones look stunning in bright, bold colors and clean whites"
    },
    
    "VI": {  # Dark (deeply pigmented dark brown to black)
        "recommended": [
            "white", "camel", "emerald", "cobalt", "mustard",
            "bright red", "electric blue", "hot pink", "lime",
            "orange", "purple", "gold", "cream", "turquoise",
            "coral", "magenta", "yellow"
        ],
        "avoid": [
            "very dark colors that blend with skin", "muddy tones"
        ],
        "reasoning": "Deepest skin tones shine in vibrant, saturated colors and crisp whites that create contrast"
    }
}


# ========================================================
# PART 2: EMBEDDING SPACES (SEPARATE ATTRIBUTE INFERENCE)
# ========================================================

# Each attribute must be inferred independently using CLIP similarity ranking.
# These prompts define distinct semantic spaces.

# ------------------------------------------------------
# 1. CATEGORY SPACE (What the item IS)
# ------------------------------------------------------

CATEGORY_PROMPTS = {
    "shirt": [
        "a men's dress shirt",
        "a button-up shirt",
        "a formal shirt"
    ],
    "t-shirt": [
        "a men's t-shirt",
        "a casual tee",
        "a short sleeve shirt"
    ],
    "polo": [
        "a polo shirt",
        "a collared casual shirt"
    ],
    "hoodie": [
        "a hoodie",
        "a hooded sweatshirt"
    ],
    "sweater": [
        "a sweater",
        "a pullover",
        "a knit top"
    ],
    "jacket": [
        "a jacket",
        "an outerwear piece",
        "a blazer or coat"
    ],
    "trousers": [
        "dress trousers",
        "formal pants",
        "chinos"
    ],
    "jeans": [
        "jeans",
        "denim pants"
    ],
    "joggers": [
        "joggers",
        "sweatpants",
        "athletic pants"
    ],
    "shorts": [
        "shorts",
        "short pants"
    ],
    "shoes": [
        "shoes",
        "footwear",
        "men's shoes"
    ]
}

# ------------------------------------------------------
# 2. COLOR SPACE (What color it APPEARS)
# ------------------------------------------------------

COLOR_PROMPTS = {
    "black": [
        "black clothing",
        "a black garment"
    ],
    "white": [
        "white clothing",
        "a white garment"
    ],
    "grey": [
        "grey clothing",
        "gray clothing",
        "a grey garment"
    ],
    "navy": [
        "navy blue clothing",
        "dark blue clothing",
        "a navy garment"
    ],
    "blue": [
        "blue clothing",
        "a blue garment"
    ],
    "light blue": [
        "light blue clothing",
        "sky blue clothing",
        "a light blue garment"
    ],
    "brown": [
        "brown clothing",
        "a brown garment"
    ],
    "beige": [
        "beige clothing",
        "tan clothing",
        "a beige garment"
    ],
    "khaki": [
        "khaki clothing",
        "khaki colored garment"
    ],
    "cream": [
        "cream colored clothing",
        "off-white clothing",
        "an ivory garment"
    ],
    "olive": [
        "olive green clothing",
        "an olive garment"
    ],
    "green": [
        "green clothing",
        "a green garment"
    ],
    "burgundy": [
        "burgundy clothing",
        "maroon clothing",
        "wine colored clothing"
    ],
    "red": [
        "red clothing",
        "a red garment"
    ],
    "pink": [
        "pink clothing",
        "a pink garment"
    ],
    "purple": [
        "purple clothing",
        "a purple garment"
    ],
    "orange": [
        "orange clothing",
        "an orange garment"
    ],
    "yellow": [
        "yellow clothing",
        "a yellow garment"
    ],
    "teal": [
        "teal clothing",
        "turquoise clothing"
    ]
}

# ------------------------------------------------------
# 3. STYLE SPACE (What VIBE it gives) — CLUSTERS
# ------------------------------------------------------

# Styles are defined as CLUSTERS with multiple prompt variants.
# Items can belong to multiple styles with different confidence scores.
# Store as distributions, NOT single labels.

STYLE_CLUSTERS = {
    "formal": {
        "prompts": [
            "business formal menswear",
            "office professional outfit",
            "corporate dress code clothing",
            "formal business attire"
        ],
        "description": "Business formal, office appropriate, corporate"
    },
    
    "smart_casual": {
        "prompts": [
            "smart casual menswear",
            "business casual outfit",
            "semi-formal clothing",
            "polished casual attire"
        ],
        "description": "Business casual, polished but relaxed"
    },
    
    "old_money": {
        "prompts": [
            "classic preppy menswear",
            "heritage style clothing",
            "old money aesthetic",
            "ivy league fashion",
            "timeless elegant menswear"
        ],
        "description": "Classic, preppy, heritage, timeless elegance"
    },
    
    "casual": {
        "prompts": [
            "casual everyday menswear",
            "relaxed comfortable outfit",
            "weekend casual clothing"
        ],
        "description": "Everyday casual, comfortable, relaxed"
    },
    
    "streetwear": {
        "prompts": [
            "streetwear fashion",
            "urban style clothing",
            "contemporary street fashion",
            "hypebeast outfit"
        ],
        "description": "Urban, contemporary, street-inspired"
    },
    
    "athletic": {
        "prompts": [
            "athletic wear",
            "sportswear outfit",
            "gym clothing",
            "activewear"
        ],
        "description": "Athletic, sporty, performance wear"
    },
    
    "minimalist": {
        "prompts": [
            "minimalist fashion",
            "clean simple menswear",
            "understated clothing",
            "minimal aesthetic outfit"
        ],
        "description": "Clean, simple, understated"
    }
}

# ------------------------------------------------------
# 4. PATTERN/STRUCTURE SPACE (Optional refinement)
# ------------------------------------------------------

PATTERN_PROMPTS = {
    "solid": [
        "plain solid colored clothing",
        "solid color garment without patterns"
    ],
    "striped": [
        "striped clothing",
        "a garment with stripes"
    ],
    "checked": [
        "checkered clothing",
        "plaid clothing",
        "a checked garment"
    ],
    "graphic": [
        "clothing with graphic print",
        "printed graphic design clothing"
    ],
    "textured": [
        "textured fabric clothing",
        "clothing with visible texture"
    ]
}


# ========================================================
# PART 3: OUTFIT COMBINATION RULES
# ========================================================

# Define which categories form valid outfit combinations
VALID_OUTFIT_STRUCTURES = [
    {"top": True, "bottom": True, "shoes": False},  # 2-piece
    {"top": True, "bottom": True, "shoes": True},   # 3-piece
]

# Categories that count as "top" and "bottom"
TOP_CATEGORIES = ["shirt", "t-shirt", "polo", "hoodie", "sweater", "jacket"]
BOTTOM_CATEGORIES = ["trousers", "jeans", "joggers", "shorts"]
SHOE_CATEGORIES = ["shoes"]


# ========================================================
# PART 4: STYLE COMPATIBILITY RULES
# ========================================================

# Define which style combinations are coherent vs conflicting
# Format: {"style_A": {"compatible": [...], "avoid": [...]}}

STYLE_COMPATIBILITY = {
    "formal": {
        "compatible": ["smart_casual", "old_money", "minimalist"],
        "avoid": ["streetwear", "athletic"],
        "reasoning": "Formal items don't pair well with casual athletic or street styles"
    },
    
    "smart_casual": {
        "compatible": ["formal", "old_money", "minimalist", "casual"],
        "avoid": ["streetwear", "athletic"],
        "reasoning": "Smart casual bridges formal and casual but not street/athletic"
    },
    
    "old_money": {
        "compatible": ["formal", "smart_casual", "casual", "minimalist"],
        "avoid": ["streetwear", "athletic"],
        "reasoning": "Classic preppy aesthetic conflicts with urban street styles"
    },
    
    "casual": {
        "compatible": ["smart_casual", "old_money", "minimalist", "streetwear"],
        "avoid": [],
        "reasoning": "Casual is versatile and can mix with most styles except extreme formal"
    },
    
    "streetwear": {
        "compatible": ["casual", "athletic", "minimalist"],
        "avoid": ["formal", "smart_casual", "old_money"],
        "reasoning": "Urban street aesthetic conflicts with traditional formal styles"
    },
    
    "athletic": {
        "compatible": ["streetwear", "casual"],
        "avoid": ["formal", "smart_casual", "old_money"],
        "reasoning": "Athletic wear is too casual for any formal context"
    },
    
    "minimalist": {
        "compatible": ["formal", "smart_casual", "old_money", "casual", "streetwear"],
        "avoid": [],
        "reasoning": "Clean minimal aesthetic is versatile across most styles"
    }
}


# ========================================================
# PART 5: COLOR HARMONY RULES
# ========================================================

# Define which colors naturally harmonize for outfit pairing
# This is independent of skin tone and applies to item-to-item matching

COLOR_HARMONY = {
    "black": ["white", "grey", "beige", "cream", "burgundy", "navy", "brown", "blue", "red", "pink", "green"],
    "white": ["black", "navy", "grey", "blue", "khaki", "beige", "brown", "burgundy", "red", "green", "pink"],
    "grey": ["black", "white", "navy", "blue", "burgundy", "pink", "brown", "beige", "cream"],
    "navy": ["white", "beige", "cream", "grey", "khaki", "burgundy", "black", "brown", "blue"],
    "blue": ["white", "beige", "cream", "grey", "khaki", "brown", "black", "navy", "burgundy"],
    "brown": ["cream", "beige", "white", "olive", "khaki", "burgundy", "black", "grey", "navy"],
    "beige": ["navy", "brown", "white", "olive", "burgundy", "grey", "black", "blue", "cream"],
    "khaki": ["navy", "white", "brown", "olive", "burgundy", "black", "grey", "blue"],
    "cream": ["navy", "brown", "burgundy", "olive", "grey", "black", "white", "beige"],
    "olive": ["cream", "beige", "brown", "khaki", "burgundy", "navy", "white", "grey"],
    "burgundy": ["navy", "grey", "cream", "beige", "black", "white", "brown", "blue"],
    "red": ["black", "white", "navy", "grey", "beige", "cream", "brown"],
    "pink": ["grey", "navy", "white", "beige", "brown", "black", "cream"],
    "green": ["beige", "brown", "cream", "white", "grey", "navy", "black"],
    "purple": ["grey", "black", "white", "beige", "cream"],
    "orange": ["navy", "brown", "cream", "white", "grey"],
    "yellow": ["navy", "grey", "white", "black", "brown"],
    "teal": ["white", "beige", "grey", "navy", "black", "brown"],
    "charcoal": ["white", "cream", "beige", "grey", "navy", "burgundy"],
    "forest green": ["cream", "beige", "white", "brown", "grey"],
    "slate": ["white", "cream", "beige", "grey"],
    "maroon": ["white", "grey", "cream", "beige"],
    "dusty rose": ["grey", "white", "cream", "beige"],
    "camel": ["navy", "brown", "white", "grey"],
    "denim blue": ["white", "cream", "beige", "grey", "brown"]
}


# ========================================================
# PART 6: FORMAL DRESS CODE RULES
# ========================================================

# What categories are acceptable in different formality contexts
FORMALITY_RULES = {
    "formal": {
        "allowed_tops": ["shirt"],  # Only dress shirts
        "allowed_bottoms": ["trousers"],  # Only dress pants
        "allowed_shoes": ["shoes"],  # Assuming dress shoes
        "avoid_patterns": ["graphic"],
        "reasoning": "Formal contexts require traditional business attire"
    },
    
    "smart_casual": {
        "allowed_tops": ["shirt", "polo", "sweater"],
        "allowed_bottoms": ["trousers", "jeans"],
        "allowed_shoes": ["shoes"],
        "avoid_patterns": ["graphic"],
        "reasoning": "Business casual allows more flexibility but still polished"
    },
    
    "casual": {
        "allowed_tops": ["shirt", "t-shirt", "polo", "hoodie", "sweater"],
        "allowed_bottoms": ["trousers", "jeans", "joggers", "shorts"],
        "allowed_shoes": ["shoes"],
        "avoid_patterns": [],
        "reasoning": "Casual allows all categories"
    }
}


# ========================================================
# HELPER FUNCTIONS
# ========================================================

def get_allowed_colors_for_skin_tone(skin_tone: str) -> list:
    """
    Returns the list of allowed colors for a given skin tone.
    This is a HARD CONSTRAINT that CLIP cannot override.
    """
    if skin_tone not in SKIN_TONE_COLOR_MAP:
        # Default to Type III (medium) if unknown
        skin_tone = "III"
    
    return SKIN_TONE_COLOR_MAP[skin_tone]["recommended"]


def get_avoided_colors_for_skin_tone(skin_tone: str) -> list:
    """
    Returns the list of colors to avoid for a given skin tone.
    """
    if skin_tone not in SKIN_TONE_COLOR_MAP:
        skin_tone = "III"
    
    return SKIN_TONE_COLOR_MAP[skin_tone]["avoid"]


def are_styles_compatible(style_a: str, style_b: str) -> bool:
    """
    Checks if two styles can be paired in the same outfit.
    Returns True if compatible, False if they conflict.
    """
    if style_a not in STYLE_COMPATIBILITY or style_b not in STYLE_COMPATIBILITY:
        return True  # Unknown styles default to compatible
    
    # Check if style_b is in style_a's avoid list
    if style_b in STYLE_COMPATIBILITY[style_a]["avoid"]:
        return False
    
    # Check if style_a is in style_b's avoid list
    if style_a in STYLE_COMPATIBILITY[style_b]["avoid"]:
        return False
    
    return True


def are_colors_harmonious(color_a: str, color_b: str) -> bool:
    """
    Checks if two colors harmonize well together.
    Returns True if they create a pleasing combination.
    """
    if color_a not in COLOR_HARMONY:
        return True  # Unknown colors default to compatible
    
    return color_b in COLOR_HARMONY[color_a]


def is_outfit_structure_valid(has_top: bool, has_bottom: bool, has_shoes: bool) -> bool:
    """
    Validates if the outfit structure is acceptable.
    """
    outfit = {"top": has_top, "bottom": has_bottom, "shoes": has_shoes}
    return outfit in VALID_OUTFIT_STRUCTURES


def get_category_type(category: str) -> str:
    """
    Returns whether a category is a 'top', 'bottom', 'shoes', or 'other'.
    """
    if category in TOP_CATEGORIES:
        return "top"
    elif category in BOTTOM_CATEGORIES:
        return "bottom"
    elif category in SHOE_CATEGORIES:
        return "shoes"
    else:
        return "other"


def validate_formality_match(items: list, target_formality: str = None) -> bool:
    """
    Checks if all items in an outfit match the target formality level.
    If no target is specified, checks if items are compatible with each other.
    """
    if not target_formality:
        # Infer from items' styles (would need item style distributions)
        return True
    
    if target_formality not in FORMALITY_RULES:
        return True
    
    rules = FORMALITY_RULES[target_formality]
    
    for item in items:
        category = item.get("category", "")
        cat_type = get_category_type(category)
        
        if cat_type == "top" and category not in rules["allowed_tops"]:
            return False
        elif cat_type == "bottom" and category not in rules["allowed_bottoms"]:
            return False
        elif cat_type == "shoes" and category not in rules["allowed_shoes"]:
            return False
    
    return True
