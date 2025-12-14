# FytMe Frontend

React-based frontend for the AI-powered fashion recommendation system.

## 🚀 Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) to view it in the browser.

## 📋 Prerequisites

- Node.js (v16 or higher)
- Running backend at `http://localhost:8000`

## 🎯 Features

- **Selfie Upload**: Upload your photo for skin tone analysis
- **Skin Tone Detection**: Automatic Fitzpatrick scale classification
- **Style Selection**: Choose from 10+ fashion styles
- **Outfit Recommendations**: Get 3 personalized outfit suggestions
- **Visual Results**: See clothing images, scores, and explanations

## 🏗️ Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ImageUpload.jsx      # File upload component
│   │   ├── StyleSelector.jsx    # Style dropdown
│   │   ├── OutfitCard.jsx       # Outfit display card
│   │   └── Loader.jsx           # Loading spinner
│   │
│   ├── pages/
│   │   └── Home.jsx             # Main page
│   │
│   ├── services/
│   │   └── api.js               # Backend API calls
│   │
│   ├── App.jsx                  # Root component
│   └── main.jsx                 # Entry point
│
└── public/
    └── demo_images/             # Clothing images folder
```

## 🔌 Backend Integration

The frontend connects to these endpoints:

- `POST /api/analyze-skin-tone` - Analyzes uploaded selfie
- `GET /api/styles` - Fetches available fashion styles
- `POST /api/recommend-outfits` - Gets outfit recommendations

## 🖼️ Adding Demo Images

Place clothing images in `public/demo_images/` folder. The backend returns filenames that match images in this directory.

## 📦 Dependencies

- **React 19** - UI library
- **Vite** - Build tool
- **Axios** - HTTP client

## 🎨 User Flow

1. Upload selfie image
2. System detects skin tone (Fitzpatrick type)
3. Select preferred style from dropdown
4. Click "Get Recommendations"
5. View 3 personalized outfits with scores

## 🐛 Troubleshooting

**Backend connection error:**
- Ensure backend is running at `http://localhost:8000`
- Check CORS is enabled in backend

**Images not loading:**
- Verify images exist in `public/demo_images/`
- Check image filenames match backend response

**Styles not loading:**
- Confirm backend `/api/styles` endpoint is working
- Check browser console for errors
