"""
CLIP utilities for semantic similarity computation.
Uses OpenAI's CLIP (ViT-B/32) for text embeddings.
"""

import torch
import clip
import numpy as np
from typing import List, Union
from functools import lru_cache

from app.config import CLIP_MODEL_NAME


class CLIPWrapper:
    """Wrapper class for CLIP model operations."""
    
    def __init__(self):
        """Initialize CLIP model and preprocessing."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model on {self.device}...")
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(CLIP_MODEL_NAME, device=self.device)
        self.model.eval()  # Set to evaluation mode
        
        print("CLIP model loaded successfully.")
    
    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text(s) into CLIP embeddings.
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            Normalized embeddings as numpy array (N x 512)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize text
        text_tokens = clip.tokenize(texts).to(self.device)
        
        # Generate embeddings
        text_features = self.model.encode_text(text_tokens)
        
        # Normalize embeddings
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        # Ensure embeddings are 1D
        if embedding1.ndim > 1:
            embedding1 = embedding1.squeeze()
        if embedding2.ndim > 1:
            embedding2 = embedding2.squeeze()
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2)
        
        # Clip to [0, 1] range (sometimes slight numerical errors can push beyond 1)
        return float(np.clip(similarity, 0, 1))
    
    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score (0 to 1)
        """
        embeddings = self.encode_text([text1, text2])
        return self.compute_similarity(embeddings[0], embeddings[1])
    
    def rank_by_similarity(self, query_text: str, candidate_texts: List[str]) -> List[tuple]:
        """
        Rank candidate texts by similarity to query text.
        
        Args:
            query_text: Query text to compare against
            candidate_texts: List of candidate texts to rank
            
        Returns:
            List of (index, similarity_score) tuples, sorted by score descending
        """
        # Encode all texts
        query_embedding = self.encode_text(query_text)
        candidate_embeddings = self.encode_text(candidate_texts)
        
        # Compute similarities
        similarities = []
        for idx, candidate_embedding in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate_embedding)
            similarities.append((idx, similarity))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities


# Global CLIP model instance (singleton pattern)
_clip_model = None


def get_clip_model() -> CLIPWrapper:
    """
    Get or initialize the global CLIP model instance.
    Uses singleton pattern to avoid reloading model multiple times.
    
    Returns:
        CLIPWrapper instance
    """
    global _clip_model
    if _clip_model is None:
        _clip_model = CLIPWrapper()
    return _clip_model


def generate_clothing_description(item: dict) -> str:
    """
    Generate a natural language description for a clothing item.
    Used for CLIP similarity comparison.
    
    Args:
        item: Dictionary with keys like category, color, pattern, style
        
    Returns:
        Natural language description string
    """
    parts = []
    
    if 'predicted_color' in item and item['predicted_color']:
        parts.append(item['predicted_color'])
    
    if 'predicted_pattern' in item and item['predicted_pattern']:
        parts.append(item['predicted_pattern'])
    
    if 'category' in item and item['category']:
        parts.append(item['category'])
    
    if 'predicted_style' in item and item['predicted_style']:
        parts.append(f"with {item['predicted_style']} style")
    
    description = " ".join(parts)
    
    # If description is empty, use category as fallback
    if not description and 'category' in item:
        description = item['category']
    
    return description if description else "clothing item"


def generate_user_intent(fitzpatrick_type: str, preferred_style: str) -> str:
    """
    Generate a user intent description for CLIP comparison.
    
    Args:
        fitzpatrick_type: Fitzpatrick skin tone type (I-VI)
        preferred_style: User's preferred style/vibe
        
    Returns:
        User intent description string
    """
    from app.config import FITZPATRICK_DESCRIPTIONS
    
    skin_tone_desc = FITZPATRICK_DESCRIPTIONS.get(fitzpatrick_type, "medium")
    
    intent = f"{preferred_style} style outfit for {skin_tone_desc} skin tone"
    
    return intent


# Precompute and cache common embeddings to improve performance
@lru_cache(maxsize=128)
def get_cached_text_embedding(text: str) -> np.ndarray:
    """
    Get cached text embedding for frequently used texts.
    
    Args:
        text: Text to embed
        
    Returns:
        CLIP text embedding
    """
    model = get_clip_model()
    return model.encode_text(text)


def test_clip_similarity():
    """Test function for CLIP similarity computation."""
    model = get_clip_model()
    
    # Test examples
    text1 = "old money style beige shirt"
    text2 = "preppy beige button-up shirt"
    text3 = "streetwear oversized hoodie"
    
    sim_12 = model.compute_text_similarity(text1, text2)
    sim_13 = model.compute_text_similarity(text1, text3)
    
    print(f"Similarity between '{text1}' and '{text2}': {sim_12:.3f}")
    print(f"Similarity between '{text1}' and '{text3}': {sim_13:.3f}")


if __name__ == "__main__":
    test_clip_similarity()
