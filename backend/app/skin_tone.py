"""
Skin tone detection module using computer vision.
Maps detected skin tone to Fitzpatrick scale (I-VI).
Uses MediaPipe for face detection and KMeans clustering for accurate skin tone extraction.
"""

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import io
from typing import Dict
from sklearn.cluster import KMeans

from app.config import FITZPATRICK_DESCRIPTIONS

# Initialize MediaPipe Face Detection once at module scope (singleton pattern)
mp_face_detection = mp.solutions.face_detection
FACE_DETECTION = mp_face_detection.FaceDetection(
    model_selection=1,  # Full-range model (better for selfies)
    min_detection_confidence=0.5
)


def predict_skin_tone(image_bytes: bytes) -> Dict[str, str]:
    """
    Predict skin tone from a selfie image using MediaPipe and KMeans clustering.
    
    Args:
        image_bytes: Raw image bytes from uploaded file
        
    Returns:
        Dictionary with fitzpatrick_type and skin_tone_label
        Example: {"fitzpatrick_type": "III", "skin_tone_label": "medium"}
    """
    try:
        # Convert bytes to numpy array
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image.convert('RGB'))
        
        # MediaPipe expects RGB format
        img_rgb = image_np
        
        # Detect face using MediaPipe
        results = FACE_DETECTION.process(img_rgb)
        
        if not results.detections:
            # Fallback: analyze center region if no face detected
            h, w = img_rgb.shape[:2]
            face_crop = img_rgb[h//4:3*h//4, w//4:3*w//4]
            skin_pixels = face_crop.reshape(-1, 3)
        else:
            # Use first detected face
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = img_rgb.shape
            
            # Convert relative coordinates to absolute
            x1 = int(bboxC.xmin * w)
            y1 = int(bboxC.ymin * h)
            x2 = x1 + int(bboxC.width * w)
            y2 = y1 + int(bboxC.height * h)
            
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            face_crop = img_rgb[y1:y2, x1:x2]
            
            # Extract skin tone from specific facial regions (forehead + cheeks)
            # This avoids hair, beard, and background contamination
            skin_pixels = extract_facial_skin_regions(face_crop)
        
        # Use KMeans clustering to find dominant skin color
        dominant_color = get_dominant_color(skin_pixels)
        
        # Map to Fitzpatrick scale
        fitzpatrick_type = rgb_to_fitzpatrick(dominant_color)
        
        return {
            "fitzpatrick_type": fitzpatrick_type,
            "skin_tone_label": FITZPATRICK_DESCRIPTIONS[fitzpatrick_type]
        }
        
    except Exception as e:
        # Fallback to Type III (medium) if detection fails
        print(f"Skin tone detection error: {e}")
        return {
            "fitzpatrick_type": "III",
            "skin_tone_label": "medium"
        }


def extract_facial_skin_regions(face_crop: np.ndarray) -> np.ndarray:
    """
    Extract skin pixels from specific facial regions (forehead + cheeks).
    Avoids hair, beard, and background contamination.
    
    Args:
        face_crop: RGB face crop array
        
    Returns:
        Array of skin pixel RGB values from forehead and cheek regions
    """
    h_fc, w_fc = face_crop.shape[:2]
    
    # Extract forehead region (top 25% of face, center 50% horizontally)
    forehead = face_crop[
        0:int(h_fc * 0.25),
        int(w_fc * 0.25):int(w_fc * 0.75)
    ]
    
    # Extract left cheek region
    cheek_left = face_crop[
        int(h_fc * 0.4):int(h_fc * 0.7),
        int(w_fc * 0.15):int(w_fc * 0.4)
    ]
    
    # Extract right cheek region
    cheek_right = face_crop[
        int(h_fc * 0.4):int(h_fc * 0.7),
        int(w_fc * 0.6):int(w_fc * 0.85)
    ]
    
    # Combine all regions into a single pixel array
    skin_pixels = np.vstack([
        forehead.reshape(-1, 3),
        cheek_left.reshape(-1, 3),
        cheek_right.reshape(-1, 3)
    ])
    
    return skin_pixels


def get_dominant_color(pixels: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    """
    Use KMeans clustering to find the dominant color in pixel array.
    
    Args:
        pixels: Array of RGB pixel values (Nx3)
        n_clusters: Number of color clusters to find
        
    Returns:
        RGB array of the most dominant color
    """
    # Run KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Get cluster centers (colors)
    colors = kmeans.cluster_centers_.astype(int)
    
    # Find the most frequent cluster (dominant color)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_idx = labels[np.argmax(counts)]
    dominant_color = colors[dominant_idx]
    
    return dominant_color


def rgb_to_fitzpatrick(rgb: np.ndarray) -> str:
    """
    Map RGB skin color to Fitzpatrick scale (I-VI) based on brightness.
    
    Args:
        rgb: RGB color array [R, G, B]
        
    Returns:
        Fitzpatrick type as string ("I" through "VI")
    """
    r, g, b = rgb
    
    # Calculate brightness (average of RGB channels)
    brightness = (int(r) + int(g) + int(b)) / 3
    
    # Map brightness to Fitzpatrick scale
    # Higher brightness = lighter skin tone
    if brightness > 220:
        return "I"    # Very fair (pale white)
    elif brightness > 180:
        return "II"   # Fair (white)
    elif brightness > 150:
        return "III"  # Medium (light brown)
    elif brightness > 120:
        return "IV"   # Olive (moderate brown)
    elif brightness > 90:
        return "V"    # Brown (dark brown)
    else:
        return "VI"   # Dark brown (deeply pigmented)


def test_skin_tone_detection():
    """Test function for skin tone detection."""
    # This function can be used for testing with sample images
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'rb') as f:
            image_bytes = f.read()
        result = predict_skin_tone(image_bytes)
        print(f"Detected skin tone: {result}")


if __name__ == "__main__":
    test_skin_tone_detection()
