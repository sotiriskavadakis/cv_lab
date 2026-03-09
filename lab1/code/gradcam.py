"""
Grad-CAM visualization for MobileNetV3-Small (bonus, standalone).
Run independently: python code/gradcam.py
Does not modify or depend on part3.py.
"""

import os
import numpy as np
import torch
import torchvision
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import matplotlib.pyplot as plt
import cv2

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'part3')
classes = ['car', 'person', 'bike']

# Load model (same as part3, but we need gradients → no torch.no_grad here)
weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
model = mobilenet_v3_small(weights=weights)
model.eval()
preprocess = weights.transforms()

# --- Grad-CAM hooks ---
# Target: last conv block before avgpool (features.12)
target_layer = model.features[12]

gradients = []
activations = []

def save_gradient(grad):
    gradients.append(grad)

def forward_hook(module, input, output):
    activations.append(output)
    if output.requires_grad:
        output.register_hook(save_gradient)

target_layer.register_forward_hook(forward_hook)

# --- Run Grad-CAM for one sample per class ---
for cls in classes:
    cls_dir = os.path.join(DATA_DIR, cls)
    img_name = sorted(os.listdir(cls_dir))[0]
    img_path = os.path.join(cls_dir, img_name)

    # Load original image for display
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Preprocess for model
    img_tensor = torchvision.io.read_image(img_path)
    if img_tensor.shape[0] == 1:
        img_tensor = img_tensor.repeat(3, 1, 1)
    elif img_tensor.shape[0] == 4:
        img_tensor = img_tensor[:3]
    input_tensor = preprocess(img_tensor).unsqueeze(0)  # (1, 3, H, W)

    # Clear buffers
    gradients.clear()
    activations.clear()

    # Forward pass (gradients needed → no no_grad)
    output = model(input_tensor)                   # (1, 1000)
    pred_class = output.argmax(dim=1).item()
    score = output[0, pred_class]

    # Backward pass for the predicted class
    model.zero_grad()
    score.backward()

    # Grad-CAM computation
    grad = gradients[0].squeeze(0)        # (C, H, W)
    act  = activations[0].squeeze(0)      # (C, H, W)

    weights_cam = grad.mean(dim=(1, 2))   # global average pooling of gradients → (C,)
    cam = (weights_cam[:, None, None] * act).sum(dim=0)  # weighted sum → (H, W)
    cam = torch.relu(cam).detach().numpy()

    # Normalize and resize to original image size
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    h, w = img_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    # Overlay heatmap on original image
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * img_rgb + 0.5 * heatmap).astype(np.uint8)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'Grad-CAM — class: {cls}  (predicted ImageNet class: {pred_class})',
                 fontsize=13, fontweight='bold')

    axes[0].imshow(img_rgb);        axes[0].set_title('Original');  axes[0].axis('off')
    axes[1].imshow(cam_resized, cmap='jet'); axes[1].set_title('Grad-CAM'); axes[1].axis('off')
    axes[2].imshow(overlay);        axes[2].set_title('Overlay');   axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'gradcam_{cls}.jpg', bbox_inches='tight')
    plt.show()
    print(f"[{cls}] predicted ImageNet class index: {pred_class}")

# =============================================================================
# BONUS: Grad-CAM on SVM-misclassified samples
# Reproduces the part3 pipeline to find which test images the SVM got wrong,
# then shows what the CNN was "looking at" for those images.
# =============================================================================

from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Load all images and keep track of their file paths
all_images, all_labels, all_paths = [], [], []
for label, cls in enumerate(classes):
    cls_dir = os.path.join(DATA_DIR, cls)
    for img_name in sorted(os.listdir(cls_dir)):
        img_path = os.path.join(cls_dir, img_name)
        img_tensor = torchvision.io.read_image(img_path)
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        elif img_tensor.shape[0] == 4:
            img_tensor = img_tensor[:3]
        all_images.append(preprocess(img_tensor))
        all_labels.append(label)
        all_paths.append(img_path)

# 2. Extract avgpool features (same as part3)
feature_extractor = create_feature_extractor(model, return_nodes={'avgpool': 'avgpool'})
features = []
with torch.no_grad():
    for img in all_images:
        out = feature_extractor(img.unsqueeze(0))
        features.append(out['avgpool'].squeeze().numpy())

features = np.array(features)
labels   = np.array(all_labels)
paths    = np.array(all_paths)

# 3. Same train/test split and SVM as part3 (identical random_state → same split)
indices = np.arange(len(features))
_, test_idx, _, y_test = train_test_split(
    indices, labels, test_size=0.3, random_state=42, stratify=labels
)
X_train_full, X_test, y_train_full, _ = train_test_split(
    features, labels, test_size=0.3, random_state=42, stratify=labels
)

scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

svm = SVC(kernel='rbf', C=10, gamma='scale', decision_function_shape='ovr')
svm.fit(X_train_full, y_train_full)
y_pred = svm.predict(X_test_scaled)

print(f"\n[misclassified Grad-CAM] SVM accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# 4. Find misclassified test samples
wrong_mask   = y_pred != y_test
wrong_indices = test_idx[wrong_mask]
wrong_true    = y_test[wrong_mask]
wrong_pred    = y_pred[wrong_mask]

if len(wrong_indices) == 0:
    print("No misclassified samples — perfect SVM accuracy on test set!")
else:
    print(f"Found {len(wrong_indices)} misclassified sample(s). Running Grad-CAM...")

    def run_gradcam(img_tensor):
        """Return normalized CAM for a single preprocessed image tensor."""
        gradients.clear()
        activations.clear()
        out   = model(img_tensor.unsqueeze(0))
        score = out[0, out.argmax(dim=1).item()]
        model.zero_grad()
        score.backward()
        grad = gradients[0].squeeze(0)
        act  = activations[0].squeeze(0)
        cam  = torch.relu((grad.mean(dim=(1, 2))[:, None, None] * act).sum(dim=0))
        cam  = cam.detach().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    for idx, true_lbl, pred_lbl in zip(wrong_indices, wrong_true, wrong_pred):
        img_path = paths[idx]
        img_bgr  = cv2.imread(img_path)
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        cam = run_gradcam(all_images[idx])
        h, w = img_rgb.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (0.5 * img_rgb + 0.5 * heatmap).astype(np.uint8)

        true_name = classes[true_lbl]
        pred_name = classes[pred_lbl]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(
            f'Grad-CAM — MISCLASSIFIED\nTrue: {true_name}  |  SVM predicted: {pred_name}',
            fontsize=13, fontweight='bold', color='red'
        )
        axes[0].imshow(img_rgb);                      axes[0].set_title('Original');  axes[0].axis('off')
        axes[1].imshow(cam_resized, cmap='jet');      axes[1].set_title('Grad-CAM'); axes[1].axis('off')
        axes[2].imshow(overlay);                      axes[2].set_title('Overlay');   axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(f'gradcam_misclassified_{true_name}_as_{pred_name}.jpg', bbox_inches='tight')
        plt.show()
