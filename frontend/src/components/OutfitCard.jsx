import { getImageUrl } from '../services/api';

const OutfitCard = ({ outfit, rank }) => {
  const { items, score, explanation } = outfit;

  const buyUrl = () => {
    const ids = items.map(i => encodeURIComponent(i.image)).join(',');
    return `https://shop.example.com/outfit?items=${ids}`;
  };

  const handleImageError = (e) => {
    e.target.src = 'https://via.placeholder.com/150?text=No+Image';
  };

  return (
    <div className="outfit-card">
      <div className="card-header">
        <h3>Outfit #{rank}</h3>
        <span className="score">{(score * 100).toFixed(0)}%</span>
      </div>

      <div className="items-grid">
        {items.map((item, index) => (
          <div key={index} className="item">
            <img
              src={getImageUrl(item.image)}
              alt={item.category}
              onError={handleImageError}
            />
            <div className="item-details">
              <p className="category">{item.category}</p>
              <p className="color">{item.color}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="explanation">
        <p>{explanation}</p>
      </div>

      <div className="actions">
        <a
          href={buyUrl()}
          target="_blank"
          rel="noopener noreferrer"
          className="buy-link"
        >
          Link to Buy
        </a>
      </div>
    </div>
  );
};

export default OutfitCard;
