# to be used later for advanced duplicate detection


# import os, torch, clip
# from PIL import Image
# from tqdm import tqdm
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from efficientnet_pytorch import EfficientNet
# from torchvision import transforms
# from skimage import io, color
# from skimage.metrics import structural_similarity as ssim
# import cv2

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # --- Load Models ---
# clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
# eff_model = EfficientNet.from_pretrained("efficientnet-b3").to(device).eval()

# eff_transform = transforms.Compose([
#     transforms.Resize((300,300)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
# ])

# # --- Fns ---
# def get_clip_emb(path):
#     img = clip_preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
#     with torch.no_grad(): emb = clip_model.encode_image(img)
#     return emb.cpu().numpy()

# def get_eff_emb(path):
#     img = eff_transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
#     with torch.no_grad(): emb = eff_model.extract_features(img).mean([2,3])
#     return emb.cpu().numpy()

# def get_blur_score(path):
#     gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
#     return cv2.Laplacian(gray, cv2.CV_64F).var()

# # --- Load images ---
# folder = "input_images"
# files = [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith(('jpg','jpeg','png'))]
# print(f"Found {len(files)} images")

# clip_embs, eff_embs = [], []
# for f in tqdm(files, desc="Extracting embeddings"):
#     clip_embs.append(get_clip_emb(f))
#     eff_embs.append(get_eff_emb(f))
# clip_embs, eff_embs = np.vstack(clip_embs), np.vstack(eff_embs)

# # --- Combine embeddings ---
# alpha = 0.6  # weight CLIP
# beta = 0.4   # weight EfficientNet
# hybrid_emb = (alpha*clip_embs/np.linalg.norm(clip_embs,axis=1,keepdims=True)
#               + beta*eff_embs/np.linalg.norm(eff_embs,axis=1,keepdims=True))

# # --- Duplicate detection ---
# sim = cosine_similarity(hybrid_emb)
# thr = 0.93
# groups, seen = [], set()
# for i in range(len(sim)):
#     if i in seen: continue
#     g = [i]; seen.add(i)
#     for j in range(i+1,len(sim)):
#         if sim[i,j]>thr: g.append(j); seen.add(j)
#     if len(g)>1: groups.append(g)

# os.makedirs("duplicates_adv_sorted", exist_ok=True)
# for idx,g in enumerate(groups,1):
#     grpdir=f"duplicates_adv_sorted/group_{idx}"
#     os.makedirs(grpdir,exist_ok=True)
#     sharpest = max(g,key=lambda k:get_blur_score(files[k]))
#     for k in g:
#         dest = os.path.join(grpdir,("BEST_" if k==sharpest else "")+os.path.basename(files[k]))
#         os.system(f'copy "{files[k]}" "{dest}"')

# print(f"✅ Done. {len(groups)} groups saved to duplicates_sorted/")
