import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
import streamlit as st # Used only for caching

# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================
class ForensicHandwritingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super(ForensicHandwritingNet, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim) 
        )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

# ==========================================
# 2. HELPER: LOAD MODEL (Cached)
# ==========================================
@st.cache_resource
def load_model():
    """Loads the model once and keeps it in memory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForensicHandwritingNet().to(device)
    
    # If you have trained weights, load them here:
    # if os.path.exists("handwriting_expert.pth"):
    #     model.load_state_dict(torch.load("handwriting_expert.pth", map_location=device))
    
    model.eval()
    return model, device

# ==========================================
# 3. PREPROCESSING
# ==========================================
def extract_handwriting_patches(image_path, min_area=500):
    img = cv2.imread(image_path)
    if img is None: return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5)) 
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    patches = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > min_area:
            roi = img[y:y+h, x:x+w]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            patches.append(Image.fromarray(roi_rgb))
    return patches

# ==========================================
# 4. DEEP LEARNING COMPARISON (Metrics)
# ==========================================
def compare_students_deep(path1, path2):
    model, device = load_model()
    
    transform = transforms.Compose([
        transforms.Resize((128, 256)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    patches1 = extract_handwriting_patches(path1)
    patches2 = extract_handwriting_patches(path2)
    
    min_samples = min(len(patches1), len(patches2), 25)
    if min_samples < 3:
        return None # Not enough data

    distances = []
    with torch.no_grad():
        for i in range(min_samples):
            t1 = transform(patches1[i]).unsqueeze(0).to(device)
            t2 = transform(patches2[i]).unsqueeze(0).to(device)
            emb1 = model(t1)
            emb2 = model(t2)
            distances.append(torch.dist(emb1, emb2).item())

    distances = np.array(distances)
    
    # Calculate Metrics
    median_dist = np.median(distances)
    std_dev = np.std(distances)
    threshold = 0.75
    
    # Verdict Logic
    if median_dist < threshold:
        confidence = min(99.9, (1 - (median_dist / threshold)) * 100 + 40)
        verdict = "MATCH DETECTED"
        is_match = True
    else:
        confidence = max(0.1, (1 - (median_dist / 2.0)) * 100)
        verdict = "NO MATCH"
        is_match = False

    # Return structured data for the UI
    return {
        "verdict": verdict,
        "is_match": is_match,
        "confidence": confidence,
        "avg_dist": np.mean(distances),
        "median_dist": median_dist,
        "best_match": np.min(distances),
        "consistency": max(0, (1 - (std_dev * 2)) * 100),
        "raw_distances": distances.tolist() # For histogram
    }

# ==========================================
# 5. VISUAL EVIDENCE (SIFT/RANSAC)
# ==========================================
def generate_visual_evidence(img1_path, img2_path):
    # (Same RANSAC code as before)
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    # Preprocessing
    _, bin1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, bin2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(bin1, None)
    kp2, des2 = sift.detectAndCompute(bin2, None)

    if des1 is None or des2 is None: return 0, None

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
        result = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
        return np.sum(matchesMask), cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    return 0, None