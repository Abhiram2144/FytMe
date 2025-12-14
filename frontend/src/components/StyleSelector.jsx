import { useEffect, useState } from 'react';
import { getStyles } from '../services/api';

const StyleSelector = ({ onStyleSelect, disabled, selectedStyle }) => {
  const [styles, setStyles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchStyles = async () => {
      try {
        const styleList = await getStyles();
        setStyles(styleList);
      } catch (err) {
        setError('Failed to load styles');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchStyles();
  }, []);

  if (loading) return <p>Loading styles...</p>;
  if (error) return <p className="error">{error}</p>;

  return (
    <div className="style-section">
      <h2>Step 3: Select Your Style</h2>
      
      <select
        value={selectedStyle}
        onChange={(e) => onStyleSelect(e.target.value)}
        disabled={disabled}
        className="style-dropdown"
      >
        <option value="">-- Choose a style --</option>
        {styles.map((style) => (
          <option key={style} value={style}>
            {style}
          </option>
        ))}
      </select>
    </div>
  );
};

export default StyleSelector;
