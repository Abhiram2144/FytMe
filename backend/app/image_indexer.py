"""
Automatic image indexing using CLIP as a similarity ranking engine.

PHILOSOPHY:
- CLIP ranks semantic similarity across separate embedding spaces
- Each attribute (category, color, style) is inferred INDEPENDENTLY
- Results are stored as top-k distributions with confidence scores
- NO filename-based assumptions
- Styles are MULTI-LABEL, not single labels

This module implements CLIP correctly:
- NOT as a classifier
- NOT as ground truth
- BUT as a similarity ranking engine providing top-k results
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from PIL import Image
import clip
import hashlib
import cv2
import numpy as np

# Import fashion domain knowledge
from .fashion_knowledge import (
    CATEGORY_PROMPTS,
    COLOR_PROMPTS,
    STYLE_CLUSTERS,
    PATTERN_PROMPTS
)


class ClothingIndexer:
    """Automatically indexes clothing images using CLIP inference."""
    
    def __init__(self, assets_dir: str = "assets/clothes", cache_file: str = "assets/catalog.json"):
        self.assets_dir = Path(assets_dir)
        self.cache_file = Path(cache_file)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.catalog = []
        self.debug = os.getenv("INDEXER_DEBUG", "0") == "1"
        
    def load_model(self):
        """Load CLIP model once."""
        print(f"Loading CLIP model on {self.device}...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        print("CLIP model loaded successfully")
        
    def _get_image_hash(self, img_path: Path) -> str:
        """Get hash of image file for change detection."""
        with open(img_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _extract_dominant_color(self, img_path: Path) -> Tuple[int, int, int] | None:
        """Estimate dominant clothing color using pixel analysis.

        Strategy:
        - Center-crop to reduce background influence
        - Ignore near-white and near-black pixels
        - KMeans (k=3) to find dominant cluster
        - Return BGR color tuple for the largest cluster
        """
        try:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                return None

            h, w = img_bgr.shape[:2]
            # Center crop (70% area)
            y1 = int(h * 0.15)
            y2 = int(h * 0.85)
            x1 = int(w * 0.15)
            x2 = int(w * 0.85)
            crop = img_bgr[y1:y2, x1:x2]

            if crop.size == 0:
                crop = img_bgr

            # Mask out near-white and near-black background
            white_mask = (crop[:, :, 0] > 230) & (crop[:, :, 1] > 230) & (crop[:, :, 2] > 230)
            black_mask = (crop[:, :, 0] < 20) & (crop[:, :, 1] < 20) & (crop[:, :, 2] < 20)
            valid_mask = ~(white_mask | black_mask)
            pixels = crop[valid_mask]

            # Fallback if mask too strict
            if pixels.shape[0] < 500:
                pixels = crop.reshape(-1, 3)

            pixels = np.float32(pixels)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 3
            _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
            counts = np.bincount(labels.flatten())
            dominant = centers[np.argmax(counts)].astype(np.uint8)
            # BGR tuple
            return int(dominant[0]), int(dominant[1]), int(dominant[2])
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Dominant color extraction error: {e}")
            return None

    def _map_bgr_to_color_label(self, bgr: Tuple[int, int, int]) -> str:
        """Map a BGR color to one of our domain color labels."""
        b, g, r = bgr
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
        h = float(hsv[0]) / 180.0  # 0..1
        s = float(hsv[1]) / 255.0
        v = float(hsv[2]) / 255.0

        hue_deg = h * 360.0

        # Neutral/achromatic first
        if v < 0.15:
            return "black"
        if s < 0.12:
            # Preserve very light blues instead of falling to grey
            if 180.0 <= hue_deg <= 260.0 and v > 0.55:
                return "light blue"
            # High value → white/cream, else grey
            if v > 0.92:
                # Slight yellow tint considered cream
                if 20.0 <= hue_deg <= 60.0 and s < 0.25:
                    return "cream"
                return "white"
            return "grey"

        # Blues
        if 190.0 <= hue_deg <= 260.0:
            if v < 0.35:
                return "navy"
            # Sky/light blues: allow moderate saturation too
            if v > 0.65 and s < 0.40:
                return "light blue"
            return "blue"

        # Browns / Beiges
        if 20.0 <= hue_deg <= 50.0:
            if v > 0.80 and s < 0.40:
                return "beige"
            return "brown"

        # Reds → burgundy when dark
        if hue_deg >= 340.0 or hue_deg < 15.0:
            if v < 0.45:
                return "burgundy"
            # fallback not commonly used; treat as brownish
            return "brown"

        # Greens
        if 60.0 <= hue_deg <= 150.0:
            if v < 0.60 and 70.0 <= hue_deg <= 100.0:
                return "olive"
            return "teal"

        # Fallback
        return "grey"

    def _slugify(self, text: str) -> str:
        """Make a filesystem- and URL-friendly slug."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", "-", text)
        return re.sub(r"-+", "-", text).strip("-")

    def _unique_filename(self, base: str, ext: str, original_name: str) -> str:
        """Generate a unique filename within assets_dir, avoiding collisions."""
        candidate = f"{base}{ext}"
        if candidate.lower() == original_name.lower():
            return original_name
        i = 1
        while (self.assets_dir / candidate).exists():
            candidate = f"{base}-{i}{ext}"
            i += 1
        return candidate
    
    def _load_cache(self) -> Dict:
        """Load cached catalog if it exists."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Failed to load cache: {e}")
                return None
        return None
    
    def _save_cache(self, catalog: List[Dict]):
        """Save catalog to JSON cache file."""
        # Ensure directory exists
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        cache_data = []
        for item in catalog:
            cache_item = {
                "image": item["image"],
                "category": item["category"],
                "category_score": item.get("category_score", 0.0),
                "color": item["color"],
                "color_score": item.get("color_score", 0.0),
                "styles": item.get("styles", []),  # List of {"label": str, "score": float}
                "primary_style": item.get("primary_style", "casual"),
                "pattern": item.get("pattern", "solid"),
                "pattern_score": item.get("pattern_score", 0.0),
                "hash": item.get("hash", "")
            }
            cache_data.append(cache_item)
        
        with open(self.cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"✅ Catalog saved to {self.cache_file}")
        
    def scan_and_index(self) -> List[Dict]:
        """
        Scan assets directory and build catalog with CLIP inference.
        Uses cache if available and images haven't changed.
        
        Returns:
            List of dicts with image metadata and embeddings
        """
        if not self.assets_dir.exists():
            print(f"⚠️  Creating {self.assets_dir}...")
            self.assets_dir.mkdir(parents=True, exist_ok=True)
            return []
        
        # Find all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(self.assets_dir.glob(ext))
        
        if len(image_files) == 0:
            print(f"⚠️  No images found in {self.assets_dir}")
            return []
        
        # Check cache and hashes
        cache = self._load_cache()
        current_images = {f.name for f in image_files}

        if cache is not None:
            cached_images = {item.get("image") for item in cache}

            # If same filenames, verify file hashes to ensure color relabeling triggers when content changes
            if current_images == cached_images:
                # Verify no hash change
                hash_changed = False
                current_hash_map = {}
                for f in image_files:
                    try:
                        current_hash_map[f.name] = self._get_image_hash(f)
                    except Exception:
                        current_hash_map[f.name] = None

                for item in cache:
                    if item.get("hash") and current_hash_map.get(item.get("image")) and current_hash_map[item["image"]] != item["hash"]:
                        hash_changed = True
                        break

                if not hash_changed:
                    print(f"✅ Using cached catalog ({len(cache)} items)")
                    self.catalog = cache
                    return cache
                else:
                    print("♻️  File hashes changed. Re-indexing to refresh labels and filenames.")
            else:
                print(f"📍 Images changed. Detected: {current_images - cached_images} new, {cached_images - current_images} removed")
        
        # Need to re-index
        if self.model is None:
            self.load_model()
        
        print(f"🔄 Indexing {len(image_files)} images...")
        
        self.catalog = []
        for img_path in sorted(image_files):
            try:
                item = self._index_image(img_path)
                self.catalog.append(item)
                print(f"✓ {item['image']} -> {item['category']}, {item['color']}, {item['primary_style']}")
            except Exception as e:
                print(f"✗ Failed to index {img_path.name}: {e}")
        
        # Save to cache
        if len(self.catalog) > 0:
            self._save_cache(self.catalog)
        
        print(f"✅ Indexing complete: {len(self.catalog)} items")
        return self.catalog
    
    def _index_image(self, img_path: Path) -> Dict:
        """
        Use CLIP as a RANKING ENGINE to infer attributes across separate embedding spaces.
        
        METHODOLOGY:
        1. Parse filename for category hint (GROUND TRUTH)
        2. Extract image embedding once
        3. Rank against CATEGORY prompts → if filename hints category, use it (override CLIP)
        4. Rank against COLOR prompts → pick top-1 with score  
        5. Rank against STYLE prompts → keep top-3 with scores (multi-label distribution)
        6. Optionally rank against PATTERN prompts → pick top-1 with score
        
        FILENAME FORMAT (mandatory for clarity):
        {category}-{color}-{style1}-{style2}-{hash}.jpeg
        
        Example: jeans-dark-blue-casual-minimalist-abc123.jpeg
                 trousers-black-formal-minimalist-def456.jpeg
                 cargo-khaki-casual-streetwear-ghi789.jpeg
        
        CLIP provides similarity rankings. Domain rules (in recommender) enforce constraints.
        
        Returns:
            Dict with:
            - image: filename
            - category: top category label (from filename hint if available, else CLIP)
            - category_score: confidence
            - color: top color label
            - color_score: confidence
            - styles: list of {"label": str, "score": float} (top-3 distribution)
            - primary_style: highest scoring style
            - pattern: optional pattern label
            - hash: file hash for cache validation
            - embedding: image vector
        """
        # ========================================================
        # STEP 0: EXTRACT CATEGORY HINT FROM FILENAME
        # ========================================================
        # Filename format: {category}-{color}-{style}-{style}-{hash}.ext
        # Parse the first part as category hint
        filename_without_ext = img_path.stem
        parts = filename_without_ext.split('-')
        
        category_hint = None
        if len(parts) > 0:
            potential_category = parts[0].lower()
            # Check if it's a known category
            if potential_category in CATEGORY_PROMPTS:
                category_hint = potential_category
                if self.debug:
                    print(f"[DEBUG] Category hint from filename: {category_hint}")
        
        # Load and preprocess image
        with Image.open(img_path).convert('RGB') as image:
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        # Get image embedding (used for all attribute spaces)
        with torch.no_grad():
            image_embedding = self.model.encode_image(image_input)
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        # ========================================================
        # STEP 1: CATEGORY INFERENCE (use filename hint as ground truth)
        # ========================================================
        if category_hint:
            # Use filename hint as ground truth
            category = category_hint
            category_score = 1.0  # High confidence from filename
            if self.debug:
                print(f"[DEBUG] Using filename category: {category}")
        else:
            # Fall back to CLIP inference
            category_prompts_list = []
            category_labels = []
            for cat, prompts in CATEGORY_PROMPTS.items():
                category_prompts_list.append(prompts[0])
                category_labels.append(cat)
            
            category, category_score = self._rank_top_1(
                image_embedding, 
                category_prompts_list, 
                category_labels
            )
            if self.debug:
                print(f"[DEBUG] CLIP inferred category: {category} ({category_score:.3f})")

        # ========================================================
        # STEP 2: COLOR INFERENCE (independent space)
        # ========================================================
        color_prompts_list = []
        color_labels = []
        for col, prompts in COLOR_PROMPTS.items():
            # Use first prompt variant for each color
            color_prompts_list.append(prompts[0])
            color_labels.append(col)
        
        color, color_score = self._rank_top_1(
            image_embedding,
            color_prompts_list,
            color_labels
        )

        # Pixel-based override for neutral/achromatic or obvious mismatches
        dominant_bgr = self._extract_dominant_color(img_path)
        if dominant_bgr is not None:
            pixel_color = self._map_bgr_to_color_label(dominant_bgr)
            if pixel_color and pixel_color != color:
                # Prefer pixel-derived color for neutrals or strong mismatches
                if pixel_color in {"grey", "black", "white", "cream", "beige", "navy"} or color in {"brown", "blue"}:
                    if self.debug:
                        print(f"[DEBUG] Overriding CLIP color: {color} -> {pixel_color} (pixel dominant)")
                    color = pixel_color
                    color_score = 0.99

        # ========================================================
        # STEP 3: STYLE INFERENCE (multi-label distribution)
        # ========================================================
        # Each style cluster has multiple prompt variants
        # Average similarity across all prompts in the cluster for robustness
        style_clusters_list = []
        style_labels = []
        for style_name, style_config in STYLE_CLUSTERS.items():
            style_clusters_list.append(style_config["prompts"])
            style_labels.append(style_name)
        
        styles = self._rank_top_k_clusters(
            image_embedding,
            style_clusters_list,
            style_labels,
            top_k=3
        )
        
        primary_style = styles[0]["label"] if styles else "casual"

        # ========================================================
        # STEP 4: PATTERN INFERENCE (optional, for future use)
        # ========================================================
        pattern_prompts_list = []
        pattern_labels = []
        for pat, prompts in PATTERN_PROMPTS.items():
            pattern_prompts_list.append(prompts[0])
            pattern_labels.append(pat)
        
        pattern, pattern_score = self._rank_top_1(
            image_embedding,
            pattern_prompts_list,
            pattern_labels
        )

        # ========================================================
        # STEP 5: GENERATE DESCRIPTIVE FILENAME
        # ========================================================
        # Create a human-readable, model-friendly filename
        digest = self._get_image_hash(img_path)[:8]
        ext = img_path.suffix.lower()
        
        # Include top-2 styles for richer filename
        style_slug = "-".join([s["label"] for s in styles[:2]])
        base = self._slugify(f"{category}-{color}-{style_slug}-{digest}")
        new_name = self._unique_filename(base, ext, img_path.name)
        
        # Rename file if different
        if new_name != img_path.name:
            new_path = img_path.with_name(new_name)
            try:
                img_path.rename(new_path)
                print(f"↪ Renamed '{img_path.name}' → '{new_name}'")
                img_path = new_path
            except Exception as e:
                print(f"⚠️  Rename failed for {img_path.name}: {e}")

        # Debug output
        if self.debug:
            debug_styles = ", ".join([f"{s['label']}={s['score']:.2f}" for s in styles[:3]])
            print(f"[DEBUG] {img_path.name}")
            print(f"  Category: {category} ({category_score:.3f})")
            print(f"  Color: {color} ({color_score:.3f})")
            print(f"  Styles: {debug_styles}")
            print(f"  Pattern: {pattern} ({pattern_score:.3f})")

        return {
            "image": img_path.name,
            "category": category,
            "category_score": float(category_score),
            "color": color,
            "color_score": float(color_score),
            "styles": styles,  # List of {"label": str, "score": float}
            "primary_style": primary_style,
            "pattern": pattern,
            "pattern_score": float(pattern_score),
            "hash": self._get_image_hash(img_path),
            "embedding": image_embedding.cpu().numpy()
        }
    
    def _rank_top_1(
        self, 
        image_embedding: torch.Tensor, 
        prompts: List[str], 
        labels: List[str]
    ) -> Tuple[str, float]:
        """
        Rank image against prompts and return top-1 label with confidence score.
        
        This is CLIP used correctly: as a similarity ranking engine, not a classifier.
        """
        text_inputs = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_embeddings = self.model.encode_text(text_inputs)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        # Cosine similarity
        similarities = (image_embedding @ text_embeddings.T).squeeze(0)
        
        # Convert to probabilities for interpretability
        probs = torch.softmax(similarities * 100, dim=0)  # Scale for sharper softmax
        
        best_idx = similarities.argmax().item()
        return labels[best_idx], float(probs[best_idx])
    
    def _rank_top_k_clusters(
        self,
        image_embedding: torch.Tensor,
        style_clusters: List[List[str]],  # Each cluster is a list of prompt variants
        labels: List[str],
        top_k: int = 3
    ) -> List[Dict]:
        """
        Rank image against style clusters using multiple prompt variants per cluster.
        Returns top-k styles with confidence scores as a distribution.
        
        Each style cluster has multiple prompts. We average similarity across prompts
        within each cluster for more robust style inference.
        
        This implements CLIP correctly: multi-label with top-k distribution, not single label.
        """
        cluster_scores = []
        
        for prompts in style_clusters:
            # Encode all prompts in this cluster
            text_inputs = clip.tokenize(prompts).to(self.device)
            with torch.no_grad():
                text_embeddings = self.model.encode_text(text_inputs)
                text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            
            # Compute similarities
            similarities = (image_embedding @ text_embeddings.T).squeeze(0)
            
            # Average similarity across all prompts in this cluster
            avg_similarity = similarities.mean().item()
            cluster_scores.append(avg_similarity)
        
        # Convert to tensor for top-k
        cluster_scores_tensor = torch.tensor(cluster_scores)
        
        # Softmax for interpretability
        probs = torch.softmax(cluster_scores_tensor * 100, dim=0)
        
        # Get top-k
        topk_scores, topk_idx = torch.topk(probs, k=min(top_k, len(labels)))
        
        styles = []
        for score, idx in zip(topk_scores.tolist(), topk_idx.tolist()):
            styles.append({
                "label": labels[idx],
                "score": float(score)
            })
        
        return styles


# Global singleton
_indexer = None

def get_indexer() -> ClothingIndexer:
    """Get or create global indexer instance."""
    global _indexer
    if _indexer is None:
        _indexer = ClothingIndexer()
    return _indexer
