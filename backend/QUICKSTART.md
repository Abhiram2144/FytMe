# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### 1. Install Dependencies

```powershell
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Start the Server

```powershell
# Run the FastAPI server
uvicorn app.main:app --reload
```

Or use the provided startup scripts:
- Windows CMD: `start.bat`
- PowerShell: `.\start.ps1`

### 3. Test the API

Open your browser and visit:
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧪 Testing the Backend

Run the test script to verify all components:

```powershell
python test_backend.py
```

## 📡 API Endpoints

### Analyze Skin Tone
```bash
POST /api/analyze-skin-tone
Content-Type: multipart/form-data

# Upload a selfie image
# Returns: { "fitzpatrick_type": "III", "skin_tone_label": "medium" }
```

### Get Available Styles
```bash
GET /api/styles

# Returns list of available style categories
```

### Recommend Outfits
```bash
POST /api/recommend-outfits
Content-Type: application/json

{
  "fitzpatrick_type": "III",
  "preferred_style": "old money"
}

# Returns 3 outfit recommendations with scores and explanations
```

## 🧪 Example Using curl

### 1. Analyze Skin Tone
```bash
curl -X POST "http://localhost:8000/api/analyze-skin-tone" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/selfie.jpg"
```

### 2. Get Recommendations
```bash
curl -X POST "http://localhost:8000/api/recommend-outfits" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "fitzpatrick_type": "III",
    "preferred_style": "old money"
  }'
```

## 🧪 Example Using Python

```python
import requests

# 1. Analyze skin tone
with open('selfie.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/analyze-skin-tone',
        files={'file': f}
    )
    skin_tone = response.json()
    print(f"Skin tone: {skin_tone}")

# 2. Get recommendations
response = requests.post(
    'http://localhost:8000/api/recommend-outfits',
    json={
        'fitzpatrick_type': skin_tone['fitzpatrick_type'],
        'preferred_style': 'old money'
    }
)
outfits = response.json()
print(f"Got {len(outfits['outfits'])} recommendations")
```

## 🔧 Configuration

Edit `app/config.py` to customize:
- Skin tone to color mappings
- Style compatibility rules
- Color harmony definitions
- Scoring weights

## 🐛 Troubleshooting

### CLIP Model Download
On first run, CLIP will download ~350MB model files. This is normal and only happens once.

### OpenCV Face Detection
If face detection fails, the system will analyze the center region of the image.

### Missing Dependencies
If you see import errors:
```bash
pip install -r requirements.txt --upgrade
```

### Port Already in Use
If port 8000 is busy:
```bash
uvicorn app.main:app --reload --port 8001
```

## 📚 Next Steps

1. **Frontend Integration**: Connect a React frontend to these APIs
2. **Add Real Images**: Replace placeholder images in `clothes_metadata.csv`
3. **Expand Dataset**: Add more clothing items to the CSV
4. **Fine-tune Rules**: Adjust color and style mappings in config
5. **Deploy**: Deploy to a cloud platform (Heroku, AWS, etc.)

## 🎨 Available Styles

- Old Money
- Casual
- Streetwear
- Minimalist
- Bohemian
- Athletic
- Professional
- Vintage
- Preppy
- Edgy

## 📖 Documentation

For detailed API documentation, visit:
- Interactive Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

Full README: [README.md](README.md)
