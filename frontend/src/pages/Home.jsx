import { useState } from 'react';
import ImageUpload from '../components/ImageUpload';
import StyleSelector from '../components/StyleSelector';
import OutfitCard from '../components/OutfitCard';
import Loader from '../components/Loader';
import { analyzeSkinTone, getRecommendations } from '../services/api';

const Home = () => {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [skinToneData, setSkinToneData] = useState(null);
  const [selectedStyle, setSelectedStyle] = useState('');
  const [outfits, setOutfits] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Step 1: Handle file upload and analyze skin tone
  const handleFileUpload = async (file) => {
    setUploadedFile(file);
    setLoading(true);
    setError(null);
    setSkinToneData(null);
    setOutfits([]);

    try {
      const result = await analyzeSkinTone(file);
      setSkinToneData(result);
    } catch (err) {
      setError('Failed to analyze skin tone. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Step 2: Handle style selection
  const handleStyleSelect = (style) => {
    setSelectedStyle(style);
  };

  // Step 3: Get recommendations
  const handleGetRecommendations = async () => {
    if (!skinToneData || !selectedStyle) {
      alert('Please complete all steps before getting recommendations.');
      return;
    }

    setLoading(true);
    setError(null);
    setOutfits([]);

    try {
      const recommendations = await getRecommendations(
        skinToneData.fitzpatrick_type,
        selectedStyle
      );
      setOutfits(recommendations);
    } catch (err) {
      const apiMessage = err?.response?.data?.detail;
      setError(apiMessage || 'Failed to get recommendations. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header className="header">
        <h1>AI Fashion Recommendation System</h1>
        <p>Upload a selfie, choose your style, and get personalized outfit recommendations</p>
      </header>

      <main className="main-content">
        {/* Step 1: Upload Image */}
        <ImageUpload onUpload={handleFileUpload} disabled={loading} />

        {/* Step 2: Show Skin Tone Result */}
        {skinToneData && (
          <div className="result-section">
            <h2>Step 2: Skin Tone Detected</h2>
            <div className="skin-tone-result">
              <p>
                <strong>Fitzpatrick Type:</strong> {skinToneData.fitzpatrick_type}
              </p>
              <p>
                <strong>Skin Tone:</strong> {skinToneData.skin_tone_label}
              </p>
            </div>
          </div>
        )}

        {/* Step 3: Style Selector */}
        {skinToneData && (
          <StyleSelector
            onStyleSelect={handleStyleSelect}
            disabled={loading}
            selectedStyle={selectedStyle}
          />
        )}

        {/* Step 4: Get Recommendations Button */}
        {skinToneData && selectedStyle && (
          <div className="action-section">
            <button
              onClick={handleGetRecommendations}
              disabled={loading}
              className="recommend-button"
            >
              Get Recommendations
            </button>
          </div>
        )}

        {/* Loading State */}
        {loading && <Loader message="Processing..." />}

        {/* Error State */}
        {error && <div className="error-message">{error}</div>}

        {/* Step 5: Show Outfits */}
        {outfits.length > 0 && (
          <div className="outfits-section">
            <h2>Your Personalized Outfits</h2>
            <div className="outfits-grid">
              {outfits.map((outfit, index) => (
                <OutfitCard key={index} outfit={outfit} rank={index + 1} />
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default Home;
