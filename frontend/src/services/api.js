import axios from 'axios';

// Configure API endpoints via Vite environment variables
// VITE_API_BASE_URL example: https://your-backend.onrender.com/api
// VITE_IMAGE_BASE_URL example: https://your-backend.onrender.com/api/images
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';
const IMAGE_BASE_URL = import.meta.env.VITE_IMAGE_BASE_URL || `${API_BASE_URL}/images`;

// Analyze skin tone from uploaded image
export const analyzeSkinTone = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await axios.post(`${API_BASE_URL}/analyze-skin-tone`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

// Get available styles
export const getStyles = async () => {
  const response = await axios.get(`${API_BASE_URL}/styles`);
  return response.data.styles;
};

// Get outfit recommendations
export const getRecommendations = async (fitzpatrickType, preferredStyle) => {
  const response = await axios.post(`${API_BASE_URL}/recommend-outfits`, {
    fitzpatrick_type: fitzpatrickType,
    preferred_style: preferredStyle,
  });

  return response.data.outfits;
};

// Get image URL from backend
export const getImageUrl = (filename) => {
  if (!filename) return '';
  return `${IMAGE_BASE_URL}/${encodeURIComponent(filename)}`;
};
