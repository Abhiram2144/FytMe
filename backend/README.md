# Fashion Recommendation Backend

AI-powered fashion recommendation system using skin tone analysis and CLIP embeddings.

## 🎯 Features

- **Skin Tone Analysis**: Detects skin tone from selfie images and maps to Fitzpatrick scale (I-VI)
- **Style-Based Recommendations**: Filters clothing by user's preferred style/vibe
- **Color Compatibility**: Recommends colors that complement detected skin tone
- **CLIP Semantic Matching**: Uses OpenAI's CLIP for semantic similarity between preferences and clothing
- **Explainable AI**: Provides clear explanations for each outfit recommendation
- **Complete Outfit Assembly**: Returns coordinated outfits (top + bottom + shoes)

## 🧱 Tech Stack

- **Framework**: FastAPI
- **ML Models**: PyTorch, OpenAI CLIP (ViT-B/32)
- **Computer Vision**: OpenCV, PIL
- **Data Processing**: pandas, numpy

## 📁 Project Structure

```
backend/
├── app/
│   ├── main.py           # FastAPI app entry point
│   ├── api.py            # API route definitions
│   ├── recommender.py    # Outfit recommendation engine
│   ├── skin_tone.py      # Skin tone detection module
│   ├── clip_utils.py     # CLIP embeddings utilities
│   └── config.py         # Configuration and mappings
├── data/
│   └── 1000_class/
│       ├── 1000img.csv          # Clothing dataset (10K+ items)
│       └── 1000img_class/       # Image files (gitignored)
├── requirements.txt
└── README.md
```

## 🚀 Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Verify data files**:
Ensure `data/1000_class/1000img.csv` exists with required columns:
- image (image filename)
- label (category: Tshirts, Jeans, Shoes, etc.)

The system auto-detects this dataset and creates default values for color/pattern/style.

## 🏃 Running the API

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at:
- API: `http://localhost:8000`
- Interactive Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📡 API Endpoints

### 1. Analyze Skin Tone
```
POST /api/analyze-skin-tone
```

**Request**: Multipart form with image file

**Response**:
```json
{
  "fitzpatrick_type": "III",
  "skin_tone_label": "medium"
}
```

### 2. Recommend Outfits
```
POST /api/recommend-outfits
```

**Request**:
```json
{
  "fitzpatrick_type": "III",
  "preferred_style": "old money"
}
```

**Response**:
```json
{
  "outfits": [
    {
      "items": [
        {
          "image": "shirt_001.jpg",
          "category": "shirt",
          "color": "white",
          "pattern": "solid",
          "style": "old money"
        },
        {
          "image": "pants_001.jpg",
          "category": "pants",
          "color": "navy",
          "pattern": "solid",
          "style": "old money"
        },
        {
          "image": "shoes_002.jpg",
          "category": "shoes",
          "color": "brown",
          "pattern": "solid",
          "style": "old money"
        }
      ],
      "score": 0.87,
      "explanation": "This outfit perfectly matches your old money aesthetic. The white top and navy bottom complement your skin tone beautifully. The colors harmonize well together for a cohesive look."
    }
  ]
}
```

### 3. Get Available Styles
```
GET /api/styles
```

**Response**:
```json
{
  "styles": ["old money", "casual", "streetwear", "minimalist", "bohemian", "athletic", "professional", "vintage", "preppy", "edgy"]
}
```

### 4. Health Check
```
GET /api/health
```

## 🧠 How It Works

### Skin Tone Detection
1. Detects face using OpenCV Haar Cascade
2. Extracts skin pixels using HSV color space filtering
3. Computes average skin color in LAB space
4. Maps luminance to Fitzpatrick scale (I-VI)

### Recommendation Algorithm
1. **Filtering**: Filters clothing by style and skin-tone-compatible colors
2. **Ranking**: Scores items using:
   - Style match (40%)
   - Color compatibility (30%)
   - CLIP semantic similarity (20%)
   - Color harmony (10%)
3. **Assembly**: Combines top + bottom + shoes into complete outfits
4. **Diversification**: Ensures variety in top recommendations
5. **Explanation**: Generates human-readable explanations

### CLIP Integration
- Uses CLIP ViT-B/32 for text embeddings
- Computes semantic similarity between:
  - User preferences (e.g., "old money style for medium skin tone")
  - Clothing descriptions (e.g., "beige shirt with old money style")
- No training required - uses pretrained model

## 🎨 Supported Styles

- **Old Money**: Classic, timeless, sophisticated
- **Casual**: Relaxed, everyday wear
- **Streetwear**: Urban, trendy, bold
- **Minimalist**: Simple, clean lines
- **Bohemian**: Free-spirited, artistic
- **Athletic**: Sporty, performance-oriented
- **Professional**: Business, formal
- **Vintage**: Retro, classic styles
- **Preppy**: Collegiate, polished
- **Edgy**: Bold, unconventional

## 🔧 Configuration

Edit `app/config.py` to customize:
- Skin tone to color mappings
- Style compatibility rules
- Color harmony definitions
- Scoring weights

## 🌐 Frontend Integration

The API is designed to work with a React frontend:

1. **Upload selfie** → POST `/api/analyze-skin-tone`
2. **Receive skin tone result**
3. **User selects style preference**
4. **Request recommendations** → POST `/api/recommend-outfits`
5. **Display outfits with scores and explanations**

CORS is enabled for all origins in development. Configure `app/main.py` for production.

## ⚠️ Important Notes

- **No Training**: This backend does NOT train any ML models at runtime
- **Preprocessed Data**: `1000img.csv` contains 10K+ clothing items, preprocessed offline
- **CLIP Model**: Downloads ViT-B/32 model (~350MB) on first run
- **OpenCV Data**: Uses Haar Cascade for face detection (included in OpenCV)

## 🚀 Demo-Ready Features

- Deterministic outputs for stable demos
- Fast inference (no training overhead)
- Clear explanations for transparency
- Modular, maintainable code structure
- Comprehensive error handling
- Interactive API documentation

## 📝 License

MIT License - See LICENSE file for details
