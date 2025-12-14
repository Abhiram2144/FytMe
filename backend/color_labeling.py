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
# 2. Define Colors + Categories
# --------------------
colors = ["red", "blue", "green", "black", "white", "gray", "yellow", "brown", "pink", "purple", "orange", "beige"]
categories = ["tshirts", "shirts", "tops", "pants", "shorts", "jackets", "dresses", "skirts", "jeans", "longsleeves"]

# --------------------
# 3. Load Dataset
# --------------------
df = pd.read_csv("dataset.csv")  # must have 'image' and 'label' columns
image_folder = r"C:\Users\satya\Downloads\lambi\1000_class\1000img_class"

predicted_colors = []

# --------------------
# 4. Process Each Item
# --------------------
for _, row in tqdm(df.iterrows(), total=len(df), desc="Color labeling"):
    img_name, cat = row["image"], row["label"].lower()
    img_path = os.path.join(image_folder, img_name)

    try:
        # load image
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

        # make prompts like "a red shirt", "a blue pants"
        prompts = [f"a {c} {cat}" for c in colors]
        tokens = clip.tokenize(prompts).to(device)

        with torch.no_grad():
            img_features = model.encode_image(image)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            text_features = model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # cosine similarity
            sims = (img_features @ text_features.T).squeeze(0)
            best_idx = sims.argmax().item()

        predicted_colors.append(colors[best_idx])

    except Exception as e:
        print(f"⚠️ Error with {img_path}: {e}")
        predicted_colors.append("unknown")

# --------------------
# 5. Save Results
# --------------------
df["predicted_color"] = predicted_colors
df.to_csv("clothes_with_colors.csv", index=False)
print("✅ Done! Colors saved to clothes_with_colors.csv")
