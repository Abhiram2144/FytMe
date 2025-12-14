import { useState } from 'react';

const ImageUpload = ({ onUpload, disabled }) => {
  const [preview, setPreview] = useState(null);
  const [fileName, setFileName] = useState('');

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setFileName(file.name);
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);

      // Pass file to parent
      onUpload(file);
    }
  };

  return (
    <div className="upload-section">
      <h2>Step 1: Upload Your Selfie</h2>
      
      <div className="upload-box">
        <input
          type="file"
          id="file-upload"
          accept="image/*"
          onChange={handleFileChange}
          disabled={disabled}
          className="file-input"
        />
        <label htmlFor="file-upload" className="upload-label">
          {fileName || 'Choose a file'}
        </label>
      </div>

      {preview && (
        <div className="preview-container">
          <img src={preview} alt="Preview" className="preview-image" />
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
