import os
import torch
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------
# Initialize Detection + Embedding Model
# ---------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(image_size=160, margin=20).to(device)
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)


# ---------------------------
# Function: Extract face embedding
# ---------------------------

def get_embedding(image_path):
    """Extracts embedding from image. Returns None if no face detected."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect + align face using MTCNN
    face = mtcnn(img_rgb)
    if face is None:
        print(f"⚠ No face detected in: {image_path}")
        return None

    # Convert tensor for model input
    face = face.unsqueeze(0).to(device)

    # Generate embedding (128-D vector)
    embedding = model(face).detach().cpu().numpy()
    return embedding.flatten()


# ---------------------------
# Load Gallery Embeddings
# ---------------------------

def load_gallery_embeddings(gallery_folder):
    gallery_db = {}
    for filename in os.listdir(gallery_folder):
        path = os.path.join(gallery_folder, filename)

        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            emb = get_embedding(path)
            if emb is not None:
                gallery_db[filename] = emb
                print(f"✔ Stored: {filename}")
    return gallery_db


# ---------------------------
# Perform Face Search
# ---------------------------

def face_search(query_image, gallery_db, threshold=0.65):
    """Compare query embedding against stored gallery embeddings."""

    query_emb = get_embedding(query_image)
    if query_emb is None:
        print("❌ Query image has no detectable face.")
        return

    best_match = None
    best_score = -1

    for img_name, emb in gallery_db.items():
        score = cosine_similarity([query_emb], [emb])[0][0]  # similarity score
        
        # Higher score = more similar
        if score > best_score:
            best_score = score
            best_match = img_name

    print("\n===== Face Search Result =====")
    print(f"Query Image: {os.path.basename(query_image)}")
    print(f"Best Match: {best_match}")
    print(f"Similarity Score: {best_score:.4f}")

    # Threshold check
    if best_score >= threshold:
        print("🔍 Result: MATCH")
    else:
        print("🛑 Result: NO CONFIDENT MATCH")


# ---------------------------
# Run Search
# ---------------------------

if __name__ == "__main__":
    GALLERY_FOLDER = "gallery"
    QUERY_IMAGE = "query/test.jpg"

    print("\n📂 Indexing Gallery...")
    gallery_db = load_gallery_embeddings(GALLERY_FOLDER)

    print("\n🔎 Running Face Search...")
    face_search(QUERY_IMAGE, gallery_db)
