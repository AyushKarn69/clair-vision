import os
import numpy as np
import cv2
import torch
import faiss
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


# -----------------------------
# 1) DEVICE SETUP
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n>>> Running on: {device} ({'GPU' if device.type=='cuda' else 'CPU'})\n")


# -----------------------------
# 2) LOAD MODELS GPU-ORIENTED
# -----------------------------
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Use FP16 on GPU for speed
if device.type == "cuda":
    model.half()


# -----------------------------
# 3) EMBEDDING FUNCTION
# -----------------------------
@torch.no_grad()
def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    face = mtcnn(img)

    if face is None:
        return None

    face = face.unsqueeze(0)

    # Push to GPU in FP16
    if device.type == "cuda":
        face = face.half().to(device, non_blocking=True)
    else:
        face = face.to(device)

    embedding = model(face).cpu().numpy().flatten()
    return embedding


# -----------------------------
# 4) BUILD INDEX FROM FOLDER
# -----------------------------
def create_index(image_folder):
    embeddings = []
    image_paths = []

    print("📌 Extracting embeddings...")
    for img_file in tqdm(os.listdir(image_folder)):
        img_path = os.path.join(image_folder, img_file)

        emb = get_embedding(img_path)
        if emb is not None:
            embeddings.append(emb)
            image_paths.append(img_path)

    if len(embeddings) == 0:
        raise ValueError("❌ No valid faces found in DB images.")

    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]

    print(f"\n📌 Building FAISS GPU index... (Dimension: {dim})")

    if device.type == "cuda":
        res = faiss.StandardGpuResources()
        faiss_index = faiss.GpuIndexFlatL2(res, dim)
    else:
        faiss_index = faiss.IndexFlatL2(dim)

    faiss_index.add(embeddings)

    print(f"✔ Indexed {len(image_paths)} images.\n")

    return faiss_index, image_paths


# -----------------------------
# 5) SEARCH FUNCTION
# -----------------------------
def search(query_img, index, paths, top_k=5):
    print("\n🔍 Searching...")

    query_embed = get_embedding(query_img)

    if query_embed is None:
        print("❌ No face detected in query.")
        return

    query_embed = np.array(query_embed).astype("float32").reshape(1, -1)

    distances, indices = index.search(query_embed, top_k)

    print("\n🎯 Top Matches:")
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        print(f"#{rank} → {paths[idx]} (distance: {dist:.4f})")


# -----------------------------
# 6) RUN
# -----------------------------
if __name__ == "__main__":
    DB_FOLDER = "gallery"
    QUERY_IMAGE = "query/test.jpg"

    index, image_list = create_index(DB_FOLDER)
    search(QUERY_IMAGE, index, image_list, top_k=5)

    print("\n✔ Done.")
