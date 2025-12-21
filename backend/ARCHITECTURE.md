# Fashion Recommendation System Architecture

## Philosophy: Rules Enforce Reality, CLIP Ranks Within Constraints

This system implements a **hybrid rule-based + ML ranking** approach to fashion recommendations, treating CLIP as a **similarity ranking engine**, not a classifier.

---

## Core Principles

### 1. **CLIP's Role: Ranking, Not Classification**
- ❌ **CLIP is NOT** a ground truth classifier
- ❌ **CLIP is NOT** a single-label decider
- ✅ **CLIP IS** a semantic similarity ranking engine
- ✅ **CLIP IS** a provider of top-k confidence distributions

### 2. **Rules > CLIP**
- Hard constraints are enforced by domain rules (fashion_knowledge.py)
- CLIP ranks options **within** those constraints
- CLIP never overrides skin tone suitability or style compatibility rules

### 3. **Separate Embedding Spaces**
- Each attribute (category, color, style, pattern) has its own prompt space
- Attributes are inferred **independently**
- No mixed prompts like "blue formal shirt" (antipattern)

---

## System Architecture

### Phase 1: Indexing (image_indexer.py)

```
Image → CLIP Encoder → Separate Ranking for Each Attribute
  ↓
  Category Ranking (top-1 with confidence)
  Color Ranking (top-1 with confidence)
  Style Ranking (top-3 distribution)
  Pattern Ranking (top-1 with confidence)
  ↓
  Store as multi-attribute item with distributions
```

**Key Innovation:**
- Styles are **multi-label distributions**, not single labels
- Example: `{"formal": 0.72, "old_money": 0.68, "minimalist": 0.45}`

### Phase 2: Recommendation (recommender.py)

```
User Request (skin_tone, style, num_outfits)
  ↓
  STEP 1: HARD FILTER (Rules)
    ├─ Skin tone → color mapping (Fitzpatrick scale)
    └─ Category separation (tops, bottoms, shoes)
  ↓
  STEP 2: SCORE ITEMS (CLIP Ranking)
    └─ Extract style affinity from pre-computed distributions
  ↓
  STEP 3: GENERATE PAIRS
    ├─ Color harmony check (rules)
    ├─ Style compatibility check (rules)
    └─ CLIP semantic similarity (pair match score)
  ↓
  STEP 4: DIVERSIFY (Rules)
    ├─ Unique color combinations
    └─ No repeated items
  ↓
  RETURN: Top-k outfits with explanations
```

---

## Module Breakdown

### 1. fashion_knowledge.py (Domain Rules)

**Purpose:** Central repository of fashion domain knowledge

**Contents:**
- `SKIN_TONE_COLOR_MAP`: Fitzpatrick I-VI → recommended/avoid colors
- `CATEGORY_PROMPTS`: Separate prompt spaces for each category
- `COLOR_PROMPTS`: Color inference prompts
- `STYLE_CLUSTERS`: Multi-prompt style definitions
- `PATTERN_PROMPTS`: Pattern detection
- `STYLE_COMPATIBILITY`: Which styles can be paired
- `COLOR_HARMONY`: Which colors harmonize together
- `FORMALITY_RULES`: Dress code enforcement

**Helper Functions:**
- `get_allowed_colors_for_skin_tone(type)`: Returns allowed colors (HARD CONSTRAINT)
- `are_styles_compatible(a, b)`: Checks style pairing validity
- `are_colors_harmonious(a, b)`: Checks color pairing
- `get_category_type(cat)`: Returns "top", "bottom", or "shoes"

### 2. image_indexer.py (CLIP Ranking)

**Purpose:** Index clothing images using CLIP as a ranking engine

**Methodology:**
1. Load image
2. Extract CLIP embedding
3. Rank against **separate** prompt spaces:
   - Category: top-1 with score
   - Color: top-1 with score
   - Style: **top-3 distribution** with scores
   - Pattern: top-1 with score
4. Rename file to descriptive slug: `{category}-{color}-{style1}-{style2}-{hash}.jpg`
5. Cache results with hash-based validation

**Key Functions:**
- `_rank_top_1()`: Single-label ranking
- `_rank_top_k_clusters()`: Multi-label style ranking with prompt variants

**Output Structure:**
```python
{
  "image": "shirt-navy-formal-old_money-abc12345.jpg",
  "category": "shirt",
  "category_score": 0.95,
  "color": "navy",
  "color_score": 0.89,
  "styles": [
    {"label": "formal", "score": 0.72},
    {"label": "old_money", "score": 0.68},
    {"label": "minimalist", "score": 0.45}
  ],
  "primary_style": "formal",
  "pattern": "solid",
  "pattern_score": 0.91,
  "hash": "abc123...",
  "embedding": [...]
}
```

### 3. recommender.py (Rule-Based + CLIP Ranking)

**Purpose:** Generate outfits by filtering with rules, ranking with CLIP

**Workflow:**

#### Step 1: Hard Filter (Rules)
```python
allowed_colors = get_allowed_colors_for_skin_tone(fitzpatrick_type)
filtered_catalog = [item for item in catalog if item["color"] in allowed_colors]
```

#### Step 2: Score Items (CLIP)
```python
style_affinity = item.styles[target_style]  # Pre-computed by indexer
```

#### Step 3: Generate Pairs (Rules + CLIP)
```python
for top in tops:
  for bottom in bottoms:
    # Rule checks
    if not are_colors_harmonious(top.color, bottom.color):
      continue
    if not are_styles_compatible(top.style, bottom.style):
      continue
    
    # CLIP semantic similarity
    pair_score = cosine_similarity(top.embedding, bottom.embedding)
    
    outfit_score = weighted_sum(item_scores, pair_score, rules_bonuses)
```

#### Step 4: Diversify (Rules)
```python
seen_color_pairs = set()
for outfit in sorted(candidates, key=score):
  if (top.color, bottom.color) in seen_color_pairs:
    continue
  selected.append(outfit)
```

**Output Structure:**
```python
{
  "top": {...},
  "bottom": {...},
  "shoes": {...} or None,
  "score": 0.87,
  "explanation": "This formal outfit combines...",
  "shirt_pant_match_score": 0.79
}
```

---

## Example Flow

### User Input
```json
{
  "fitzpatrick_type": "IV",
  "preferred_style": "old_money"
}
```

### Step 1: Filter by Skin Tone
```
Type IV allowed colors: ["khaki", "brown", "rust", "navy", "cream", ...]
Filtered: 45 items → 28 items
```

### Step 2: Score by Style
```
Navy shirt: old_money=0.68, formal=0.72
Brown trousers: old_money=0.71, casual=0.58
```

### Step 3: Check Compatibility
```
✓ navy + brown = harmonious
✓ old_money + old_money = compatible
CLIP pair score = 0.79
```

### Step 4: Generate Explanation
```
"This old_money outfit combines formal and old_money elements with 
old_money aesthetics. The navy top and brown bottom create a harmonious 
color combination. Both colors complement Fitzpatrick type IV skin tone. 
The pieces create a cohesive look."
```

---

## Why This Approach Works

### Traditional ML Approach (❌)
- CLIP as classifier
- Filename-based assumptions
- Single-label predictions
- No explainability
- No domain constraints

### Our Approach (✅)
- CLIP as ranking engine
- Multi-label distributions
- Rule-based constraints
- Full explainability
- Domain expert knowledge encoded

---

## Summary

This system embodies the principle:

> **"CLIP ranks semantic similarity. Rules enforce reality."**

By separating domain logic (rules) from ML ranking (CLIP), we achieve:
- ✅ Explainable recommendations
- ✅ Domain expert knowledge encoded
- ✅ Correct usage of CLIP (ranking, not classification)
- ✅ Multi-label style distributions
- ✅ Extensible architecture
- ✅ Production-ready constraints

**The result:** A fashion stylist system, not a naive image classifier.
