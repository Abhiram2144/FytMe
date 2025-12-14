"""
Automatic image indexing using CLIP zero-shot inference.
Scans assets/clothes/ directory and infers category, color, and style.
Caches results to catalog.json for efficiency.
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
                "color": item["color"],
                "style": item["style"],
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
                print(f"✓ {item['image']} -> {item['category']}, {item['color']}, {item['style']}")
            except Exception as e:
                print(f"✗ Failed to index {img_path.name}: {e}")
        
        # Save to cache
        if len(self.catalog) > 0:
            self._save_cache(self.catalog)
        
        print(f"✅ Indexing complete: {len(self.catalog)} items")
        return self.catalog
    
    def _index_image(self, img_path: Path) -> Dict:
        """
        Use CLIP to infer all attributes for a single image.
        
        Returns:
            Dict with image, category, color, styles (with scores), primary_style, hash, embedding
        """
        # Load and preprocess image
        with Image.open(img_path).convert('RGB') as image:
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        # Get image embedding
        with torch.no_grad():
            image_embedding = self.model.encode_image(image_input)
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        # Attribute prompts
        category_prompts = [
            "a photo of a shirt",
            "a photo of pants",
            "a photo of trousers",
            "a photo of joggers",
            "a photo of hoodie",
            "a photo of jacket",
            "a photo of shoes",
        ]
        category_labels = ["shirt", "pants", "pants", "joggers", "hoodie", "jacket", "shoes"]

        color_prompts = [
            "a black clothing item",
            "a navy clothing item",
            "a beige clothing item",
            "a white clothing item",
            "a brown clothing item",
            "a gray clothing item",
        ]
        color_labels = ["black", "navy", "beige", "white", "brown", "gray"]

        style_prompts = [
            "formal office outfit",
            "classic preppy outfit",
            "old money aesthetic outfit",
            "casual everyday outfit",
            "streetwear outfit",
            "sportswear athletic outfit",
        ]
        style_labels = [
            "formal",
            "preppy",
            "old money",
            "casual",
            "streetwear",
            "sportswear",
        ]

        # Infer category and color (single label)
        category, category_score, _ = self._infer_single(image_embedding, category_prompts, category_labels)
        color, color_score, _ = self._infer_single(image_embedding, color_prompts, color_labels)

        # Infer styles (multi-label top-2 with scores)
        styles = self._infer_styles(image_embedding, style_prompts, style_labels, top_k=3)
        styles = self._apply_style_adjustments(styles, category)
        primary_style = styles[0]["label"] if styles else "unknown"

        # Sanitize and optionally rename file to model-friendly name
        digest = self._get_image_hash(img_path)[:8]
        ext = img_path.suffix.lower()
        base = self._slugify(f"{category}-{color}-{primary_style}-{digest}")
        new_name = self._unique_filename(base, ext, img_path.name)
        if new_name != img_path.name:
            new_path = img_path.with_name(new_name)
            try:
                img_path.rename(new_path)
                print(f"↪ Renamed '{img_path.name}' → '{new_name}'")
                img_path = new_path
            except Exception as e:
                print(f"⚠️  Rename failed for {img_path.name}: {e}")

        if self.debug:
            debug_styles = ", ".join([f"{s['label']}={s['score']:.2f}" for s in styles[:3]])
            print(f"[DEBUG] {img_path.name} | cat={category} ({category_score:.2f}) | color={color} ({color_score:.2f}) | styles: {debug_styles}")

        return {
            "image": img_path.name,
            "category": category,
            "category_score": category_score,
            "color": color,
            "color_score": color_score,
            "style": primary_style,
            "styles": styles,
            "hash": self._get_image_hash(img_path),
            "embedding": image_embedding.cpu().numpy()
        }
    
    def _infer_single(self, image_embedding, prompts: List[str], labels: List[str]) -> Tuple[str, float, torch.Tensor]:
        text_inputs = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_embeddings = self.model.encode_text(text_inputs)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        similarities = (image_embedding @ text_embeddings.T).squeeze(0)
        probs = torch.softmax(similarities, dim=0)
        best_idx = similarities.argmax().item()
        return labels[best_idx], float(probs[best_idx]), similarities

    def _infer_styles(self, image_embedding, prompts: List[str], labels: List[str], top_k: int = 2) -> List[Dict]:
        text_inputs = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_embeddings = self.model.encode_text(text_inputs)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        similarities = (image_embedding @ text_embeddings.T).squeeze(0)
        probs = torch.softmax(similarities, dim=0)

        topk_scores, topk_idx = torch.topk(probs, k=min(top_k, len(labels)))
        styles = []
        for score, idx in zip(topk_scores.tolist(), topk_idx.tolist()):
            styles.append({"label": labels[idx], "score": float(score)})
        return styles

    def _apply_style_adjustments(self, styles: List[Dict], category: str) -> List[Dict]:
        adjusted = []
        for entry in styles:
            label = entry["label"]
            score = entry["score"]

            # Formal is strict: penalize for athleisure or casual garments
            if label == "formal" and category in {"joggers", "hoodie", "shoes"}:
                score *= 0.5

            # Old money is heritage/preppy but not formal: slight penalty if category is overtly sporty
            if label == "old money" and category in {"joggers", "hoodie"}:
                score *= 0.8

            adjusted.append({"label": label, "score": score})

        # Re-normalize to keep scores interpretable (not strictly sum to 1, just scale max to 1)
        if adjusted:
            max_score = max(a["score"] for a in adjusted)
            if max_score > 0:
                adjusted = [{"label": a["label"], "score": a["score"] / max_score} for a in adjusted]
        return adjusted


# Global singleton
_indexer = None

def get_indexer() -> ClothingIndexer:
    """Get or create global indexer instance."""
    global _indexer
    if _indexer is None:
        _indexer = ClothingIndexer()
    return _indexer
