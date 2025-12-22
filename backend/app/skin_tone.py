"""
Skin tone detection module.
Uses MediaPipe face detection + KMeans clustering on forehead & cheeks
to map skin tone to Fitzpatrick scale (I–VI).

This implementation matches the verified Google Colab prototype.
"""

import numpy as np
from PIL import Image
import io
from typing import Dict
from sklearn.cluster import KMeans
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .config import FITZPATRICK_DESCRIPTIONS

# Initialize MediaPipe Face Detector once at module scope (new API for 0.10.30+)
base_options = python.BaseOptions(model_asset_path='')
options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5
)
FACE_DETECTOR = vision.FaceDetector.create_from_options(options)


def predict_skin_tone(image_bytes: bytes) -> Dict[str, str]:
    """
    Predict skin tone from a selfie image using MediaPipe face detection
    and KMeans clustering on forehead and cheek regions.

    Args:
        image_bytes: Raw image bytes from uploaded file

    Returns:
        Dictionary with fitzpatrick_type and skin_tone_label
        Example: {"fitzpatrick_type": "III", "skin_tone_label": "medium"}
    """
    try:
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_rgb = np.array(image)
        
        # Convert to MediaPipe Image format
        mp_image = python.Image(image_format=python.ImageFormat.SRGB, data=img_rgb)

        # Face detection using MediaPipe (new API)
        detection_result = FACE_DETECTOR.detect(mp_image)

        if not detection_result.detections:
            # No face detected - return default
            print("[SkinTone] No face detected, returning default")
            return {
                "fitzpatrick_type": "III",
                "skin_tone_label": FITZPATRICK_DESCRIPTIONS["III"]
            }

        # Use first detected face
        detection = detection_result.detections[0]
        bbox = detection.bounding_box
        h, w, _ = img_rgb.shape

        # Extract bounding box coordinates (already in absolute pixels)
        x1 = max(0, bbox.origin_x)
        y1 = max(0, bbox.origin_y)
        x2 = min(w, bbox.origin_x + bbox.width)
        y2 = min(h, bbox.origin_y + bbox.height)

        # Extract face region
        face_crop = img_rgb[y1:y2, x1:x2]

        # Extract skin pixels from forehead and cheeks only
        skin_pixels = extract_facial_skin_regions(face_crop)

        # Find dominant color using KMeans clustering
        dominant_rgb = get_dominant_color(skin_pixels)

        # Map to Fitzpatrick scale
        fitzpatrick = rgb_to_fitzpatrick(dominant_rgb)

        return {
            "fitzpatrick_type": fitzpatrick,
            "skin_tone_label": FITZPATRICK_DESCRIPTIONS[fitzpatrick]
        }

    except Exception as e:
        # Return default on any error
        print(f"[SkinTone] Error: {e}, returning default")
        return {
            "fitzpatrick_type": "III",
            "skin_tone_label": FITZPATRICK_DESCRIPTIONS["III"]
        }


def extract_facial_skin_regions(face_crop: np.ndarray) -> np.ndarray:
    """
    Extract skin pixels from specific facial regions (forehead + cheeks).
    Avoids hair, beard, lips, and background.

    Args:
        face_crop: RGB face crop array

    Returns:
        Nx3 array of skin pixel RGB values
    """
    h_fc, w_fc, _ = face_crop.shape

    # Forehead: top 25% of face, center 50% horizontally
    forehead = face_crop[
        0:int(h_fc * 0.25),
        int(w_fc * 0.25):int(w_fc * 0.75)
    ]

    # Left cheek: 40-70% height, 15-40% width
    cheek_left = face_crop[
        int(h_fc * 0.4):int(h_fc * 0.7),
        int(w_fc * 0.15):int(w_fc * 0.4)
    ]

    # Right cheek: 40-70% height, 60-85% width
    cheek_right = face_crop[
        int(h_fc * 0.4):int(h_fc * 0.7),
        int(w_fc * 0.6):int(w_fc * 0.85)
    ]

    # Combine all regions into single pixel array
    pixels = np.vstack([
        forehead.reshape(-1, 3),
        cheek_left.reshape(-1, 3),
        cheek_right.reshape(-1, 3)
    ])

    return pixels


def get_dominant_color(pixels: np.ndarray) -> np.ndarray:
    """
    Find dominant RGB color using KMeans clustering.

    Args:
        pixels: Nx3 array of RGB pixel values

    Returns:
        RGB array [R, G, B] of the most dominant color
    """
    # Run KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # Get cluster centers
    colors = kmeans.cluster_centers_.astype(int)

    # Find most frequent cluster
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_idx = np.argmax(counts)
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
    brightness = (int(r) + int(g) + int(b)) / 3

    if brightness > 220:
        return "I"
    elif brightness > 180:
        return "II"
    elif brightness > 150:
        return "III"
    elif brightness > 120:
        return "IV"
    elif brightness > 90:
        return "V"
    else:
        return "VI"


def test_skin_tone_detection():
    """Test function for skin tone detection."""
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'rb') as f:
            image_bytes = f.read()
        result = predict_skin_tone(image_bytes)
        print(f"Detected skin tone: {result}")


if __name__ == "__main__":
    test_skin_tone_detection()
