import torch
import clip
from PIL import Image
import pandas as pd
from tqdm import tqdm
import os

# --------------------
# 1. Setup
# --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# --------------------
# 2. Define Categories (Vibe-based styles)
# --------------------
patterns = ["solid", "striped", "checked", "floral", "graphic", "polka dot", "camouflage", "animal print"]

# Better prompts for "vibe" aesthetics
styles = [
    "an outfit with old money aesthetic",
    "a casual everyday outfit",
    "a flashy star boy outfit aesthetic",
    "a sports wear outfit",
    "a formal outfit",
    "a gym wear outfit"
]

style_labels = ["old money", "casual", "star boy", "sports wear", "formals", "gym wear"]  # for clean csv output

# --------------------
# 3. Load Dataset
# --------------------
df = pd.read_csv("clothes_with_colors.csv")  # 👈 use the color-labeled file
image_folder = r"C:\Users\satya\Downloads\lambi\1000_class\1000img_class"

predicted_patterns = []
predicted_styles   = []

# --------------------
# 4. Process Each Item
# --------------------
for _, row in tqdm(df.iterrows(), total=len(df), desc="Pattern+Vibe labeling"):
    img_name, cat = row["image"], row["label"].lower()
    img_path = os.path.join(image_folder, img_name)

    try:
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            # -------- Pattern Prediction --------
            pattern_prompts = [f"a {p} {cat}" for p in patterns]
            pattern_tokens = clip.tokenize(pattern_prompts).to(device)

            img_features = model.encode_image(image)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            pattern_features = model.encode_text(pattern_tokens)
            pattern_features /= pattern_features.norm(dim=-1, keepdim=True)

            sims = (img_features @ pattern_features.T).squeeze(0)
            pattern_idx = sims.argmax().item()
            predicted_patterns.append(patterns[pattern_idx])

            # -------- Style Prediction (Vibe-based) --------
            style_prompts = [f"{s}" for s in styles]  # already aesthetic prompts
            style_tokens = clip.tokenize(style_prompts).to(device)

            style_features = model.encode_text(style_tokens)
            style_features /= style_features.norm(dim=-1, keepdim=True)

            sims = (img_features @ style_features.T).squeeze(0)
            style_idx = sims.argmax().item()
            predicted_styles.append(style_labels[style_idx])

    except Exception as e:
        print(f"⚠️ Error with {img_path}: {e}")
        predicted_patterns.append("unknown")
        predicted_styles.append("unknown")

# --------------------
# 5. Save Results
# --------------------
df["predicted_pattern"] = predicted_patterns
df["predicted_style"]   = predicted_styles

df.to_csv("clothes_with_pattern_vibe.csv", index=False)
print("✅ Done! Results saved to clothes_with_pattern_vibe.csv")
