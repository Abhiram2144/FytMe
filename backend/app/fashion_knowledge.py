"""
Fashion Knowledge Base - Domain Rules and Embedding Spaces

RULE PRIORITY ORDER (CRITICAL):
================================
1. Formality rules (HARD BLOCK) - category-type cannot be used in formality level
2. Category-type rules (HARD BLOCK) - jeans/hoodies/cargo forbid certain contexts
3. Skin-tone color constraints (HARD BLOCK) - must be in allowed colors
4. Pattern formality rules (HARD BLOCK) - graphic patterns forbidden in formal/smart_casual
5. Style compatibility (SOFT BLOCK) - styles can conflict but don't hard-fail
6. Color harmony (SOFT BONUS) - prefers harmonious but allows fallback
7. CLIP similarity (RANKING ONLY) - ranks within all constraints, never overrides rules

PHILOSOPHY:
-----------
In fashion, the opposite of "allowed" is "forbidden".

CLIP provides perception (similarity ranking).
Rules provide reasoning (real-world appropriateness).

If a rule blocks an item, CLIP CANNOT resurrect it.
"""

# ========================================================
# PART 1.5: CATEGORY-TYPE STRICTNESS RULES
# ========================================================

# Define which specific item types are appropriate for each formality level
# This is STRICTER than just category—it's the ACTUAL item subtype

TOP_TYPE_RULES = {
    "shirt": {  # Dress shirts (solid, formal fabrics)
        "formal": True,
        "smart_casual": True,
        "casual": True,
        "streetwear": False,
        "athletic": False,
        "old_money": True,
        "minimalist": True
    },
    "t-shirt": {  # Casual t-shirts
        "formal": False,
        "smart_casual": False,
        "casual": True,
        "streetwear": True,
        "athletic": True,
        "old_money": False,
        "minimalist": True
    },
    "polo": {  # Polo shirts
        "formal": False,
        "smart_casual": True,
        "casual": True,
        "streetwear": False,
        "athletic": False,
        "old_money": True,
        "minimalist": True
    },
    "hoodie": {  # Hoodies
        "formal": False,
        "smart_casual": False,
        "casual": True,
        "streetwear": True,
        "athletic": True,
        "old_money": False,
        "minimalist": False
    },
    "sweater": {  # Knit sweaters
        "formal": False,
        "smart_casual": True,
        "casual": True,
        "streetwear": False,
        "athletic": False,
        "old_money": True,
        "minimalist": True
    },
    "jacket": {  # Blazers/outerwear
        "formal": True,
        "smart_casual": True,
        "casual": True,
        "streetwear": True,
        "athletic": False,
        "old_money": True,
        "minimalist": True
    }
}

BOTTOM_TYPE_RULES = {
    "trousers": {  # Dress trousers
        "formal": True,
        "smart_casual": True,
        "casual": True,
        "streetwear": False,
        "athletic": False,
        "old_money": True,
        "minimalist": True
    },
    "jeans": {  # Denim (typically requires dark, clean in formal/smart_casual)
        "formal": False,  # Never in formal
        "smart_casual": True,  # Only if dark and pristine
        "casual": True,
        "streetwear": True,
        "athletic": False,
        "old_money": False,  # Preppy doesn't wear regular jeans
        "minimalist": True,
        "notes": "Only dark, clean denim in smart_casual. Ripped/distressed forbidden."
    },
    "cargo_pants": {  # Cargo/utility pants
        "formal": False,
        "smart_casual": False,
        "casual": True,
        "streetwear": True,
        "athletic": False,
        "old_money": False,
        "minimalist": False,
        "notes": "Cargo pants never formal or smart casual"
    },
    "joggers": {  # Sweatpants/joggers
        "formal": False,
        "smart_casual": False,
        "casual": True,
        "streetwear": True,
        "athletic": True,
        "old_money": False,
        "minimalist": False,
        "notes": "Joggers only casual/athletic"
    },
    "shorts": {  # Shorts
        "formal": False,
        "smart_casual": False,
        "casual": True,
        "streetwear": True,
        "athletic": True,
        "old_money": False,
        "minimalist": False,
        "notes": "Shorts never formal"
    }
}

SHOE_TYPE_RULES = {
    "dress_shoes": {  # Formal shoes (oxfords, loafers, brogues)
        "formal": True,
        "smart_casual": True,
        "casual": False,
        "streetwear": False,
        "athletic": False,
        "old_money": True,
        "minimalist": True
    },
    "loafers": {  # Casual loafers
        "formal": True,
        "smart_casual": True,
        "casual": True,
        "streetwear": False,
        "athletic": False,
        "old_money": True,
        "minimalist": True
    },
    "sneakers": {  # Casual sneakers/trainers
        "formal": False,
        "smart_casual": False,
        "casual": True,
        "streetwear": True,
        "athletic": True,
        "old_money": False,
        "minimalist": True,
        "notes": "Sneakers never formal or smart_casual"
    },
    "running_shoes": {  # Athletic shoes
        "formal": False,
        "smart_casual": False,
        "casual": False,
        "streetwear": False,
        "athletic": True,
        "old_money": False,
        "minimalist": False,
        "notes": "Running shoes athletic only"
    }
}

PATTERN_FORMALITY_RULES = {
    "formal": {
        "allowed": ["solid", "subtle_texture", "fine_stripe"],
        "forbidden": ["graphic", "large_pattern", "loud_stripe", "bold_check"],
        "notes": "Formal demands subtlety"
    },
    "smart_casual": {
        "allowed": ["solid", "fine_stripe", "subtle_check"],
        "forbidden": ["graphic", "loud_stripe", "large_pattern"],
        "notes": "Graphics too casual for business contexts"
    },
    "casual": {
        "allowed": ["solid", "striped", "checked", "graphic", "textured"],
        "forbidden": [],
        "notes": "Casual allows anything"
    },
    "streetwear": {
        "allowed": ["solid", "graphic", "large_pattern", "bold"],
        "forbidden": [],
        "notes": "Urban street allows bold patterns"
    },
    "athletic": {
        "allowed": ["solid", "stripe", "graphic"],
        "forbidden": [],
        "notes": "Athletic allows practical patterns"
    },
    "old_money": {
        "allowed": ["solid", "fine_stripe", "subtle_check"],
        "forbidden": ["graphic", "loud_pattern"],
        "notes": "Heritage preppy is understated"
    },
    "minimalist": {
        "allowed": ["solid", "subtle_texture"],
        "forbidden": ["graphic", "large_pattern", "loud"],
        "notes": "Minimalism avoids visual noise"
    }
}


# ========================================================
# PART 2: SKIN TONE → COLOR COMPATIBILITY (HARD RULES)

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
# PART 7: COLOR MATCHING FLEXIBILITY
# ========================================================

# Map catalog colors to recommended color families
# This allows "navy" to match "royal blue" or "cobalt", etc.
COLOR_EQUIVALENTS = {
    "navy": ["navy", "royal blue", "cobalt", "blue"],
    "blue": ["blue", "royal blue", "cobalt", "navy"],
    "light blue": ["light blue", "blue", "cobalt"],
    "black": ["black"],
    "white": ["white", "cream"],
    "cream": ["cream", "white", "beige"],
    "beige": ["beige", "cream", "camel", "khaki"],
    "brown": ["brown", "chocolate brown", "camel"],
    "grey": ["grey", "charcoal", "warm grey"],
    "burgundy": ["burgundy", "maroon", "wine"],
    "red": ["red", "bright red"],
    "pink": ["pink", "bright pink", "hot pink", "dusty rose", "coral"],
    "purple": ["purple", "deep purple", "lavender", "plum"],
    "green": ["green", "emerald", "forest green", "olive", "sage green"],
    "yellow": ["yellow", "mustard", "gold"],
    "orange": ["orange", "burnt orange", "coral", "rust", "terracotta"],
    "teal": ["teal", "turquoise"],
    "khaki": ["khaki", "beige", "camel"]
}


# ========================================================
# HELPER FUNCTIONS
# ========================================================

def matches_allowed_color(catalog_color: str, allowed_colors: list) -> bool:
    """
    Check if a catalog color matches any of the allowed colors.
    Uses flexible matching via COLOR_EQUIVALENTS.
    
    Args:
        catalog_color: Color detected in the catalog item (e.g., "navy")
        allowed_colors: List of recommended colors for skin tone (e.g., ["royal blue", "cobalt"])
        
    Returns:
        True if the catalog color is equivalent to any allowed color
    """
    catalog_color_lower = catalog_color.lower()
    
    # Get equivalent colors for this catalog color
    equiv = COLOR_EQUIVALENTS.get(catalog_color_lower, [catalog_color_lower])
    
    # Check if any equivalent matches an allowed color
    for allowed in allowed_colors:
        allowed_lower = allowed.lower()
        if allowed_lower in equiv or allowed_lower == catalog_color_lower:
            return True
        # Also check reverse: if allowed color has equivalents that include catalog color
        allowed_equiv = COLOR_EQUIVALENTS.get(allowed_lower, [allowed_lower])
        if catalog_color_lower in allowed_equiv:
            return True
    
    return False


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


def is_category_type_allowed(category: str, style: str) -> bool:
    """
    HARD RULE: Check if a specific category type is allowed in a specific style.
    
    This enforces "jeans are never formal", "hoodies are not smart_casual", etc.
    
    Args:
        category: Item category (e.g., "jeans", "hoodie", "trousers")
        style: Target style (e.g., "formal", "casual", "smart_casual")
        
    Returns:
        True if the category is allowed in this style, False otherwise
        
    NOTE: If a rule forbids it, CLIP CANNOT resurrect it.
    """
    # Check tops
    if category in TOP_TYPE_RULES:
        allowed = TOP_TYPE_RULES[category].get(style, None)
        if allowed is False:  # Explicitly forbidden
            return False
        if allowed is True:  # Explicitly allowed
            return True
    
    # Check bottoms
    if category in BOTTOM_TYPE_RULES:
        allowed = BOTTOM_TYPE_RULES[category].get(style, None)
        if allowed is False:  # Explicitly forbidden
            return False
        if allowed is True:  # Explicitly allowed
            return True
    
    # Check shoes
    if category in SHOE_TYPE_RULES:
        allowed = SHOE_TYPE_RULES[category].get(style, None)
        if allowed is False:  # Explicitly forbidden
            return False
        if allowed is True:  # Explicitly allowed
            return True
    
    # Default: if not in rules, allow it (backward compatibility)
    return True


def is_pattern_allowed_in_style(pattern: str, style: str) -> bool:
    """
    HARD RULE: Check if a pattern is allowed in a specific style.
    
    This forbids graphic patterns in formal, enforces subtlety in smart_casual, etc.
    
    Args:
        pattern: Pattern type (e.g., "solid", "graphic", "striped", "checked")
        style: Target style (e.g., "formal", "casual", "smart_casual")
        
    Returns:
        True if pattern is allowed, False if explicitly forbidden
        
    NOTE: If a rule forbids it, CLIP CANNOT resurrect it.
    """
    if style not in PATTERN_FORMALITY_RULES:
        return True  # Unknown style defaults to allow
    
    rules = PATTERN_FORMALITY_RULES[style]
    pattern_lower = pattern.lower()
    
    # Check forbidden list first (more important than allowed)
    if pattern_lower in [p.lower() for p in rules.get("forbidden", [])]:
        return False
    
    # Check allowed list (if not in allowed, still allow for compatibility)
    if pattern_lower in [p.lower() for p in rules.get("allowed", [])]:
        return True
    
    # If not explicitly mentioned, default to allow
    return True

