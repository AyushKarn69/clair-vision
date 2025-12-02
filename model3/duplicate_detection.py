import os
import torch
import clip
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from shutil import copy2

# === CONFIGURATION ===
input_folder = "input_images"   # your image set
output_folder = "duplicateeeeeeee_sorted"
similarity_threshold = 0.93            # increase to make stricter grouping (0.9–0.95 works best)
device = "cuda" if torch.cuda.is_available() else "cpu"

# === LOAD CLIP MODEL ===
print("Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=device)

# === EMBEDDING EXTRACTION ===
def get_image_embedding(image_path):
    try:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            return model.encode_image(image).cpu().numpy()
    except Exception as e:
        print(f"⚠️ Skipping {image_path}: {e}")
        return None

# === LOAD ALL IMAGES ===
image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Found {len(image_files)} images. Computing embeddings...")

embeddings, valid_paths = [], []
for img_path in tqdm(image_files):
    emb = get_image_embedding(img_path)
    if emb is not None:
        embeddings.append(emb)
        valid_paths.append(img_path)

embeddings = np.vstack(embeddings)
print("Embeddings computed.")

# === DUPLICATE DETECTION ===
print("Detecting duplicates...")
similarity_matrix = cosine_similarity(embeddings)
groups, visited = [], set()

for i in range(len(similarity_matrix)):
    if i in visited:
        continue
    group = [i]
    visited.add(i)
    for j in range(i + 1, len(similarity_matrix)):
        if similarity_matrix[i, j] > similarity_threshold:
            group.append(j)
            visited.add(j)
    if len(group) > 1:
        groups.append(group)

print(f"Found {len(groups)} duplicate groups.")

# === SAVE RESULTS ===
os.makedirs(output_folder, exist_ok=True)

for idx, group in enumerate(groups, start=1):
    group_folder = os.path.join(output_folder, f"group_{idx}")
    os.makedirs(group_folder, exist_ok=True)
    for img_idx in group:
        copy2(valid_paths[img_idx], group_folder)

print("✅ Duplicates grouped successfully.")
print(f"Results saved in: {output_folder}")
