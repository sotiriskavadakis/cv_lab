import os
import copy
import random
import numpy as np
import cv2
import torch
import torchvision
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR     = os.path.join(os.path.dirname(__file__), '..', 'data', 'part3')
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'results', 'part3')
PICTURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'pictures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PICTURES_DIR, exist_ok=True)

def save_fig(filename: str) -> None:
    for directory in (RESULTS_DIR, PICTURES_DIR):
        plt.savefig(os.path.join(directory, filename), bbox_inches='tight')

classes = ['car', 'person', 'bike']

# 3.2.1 Load MobileNetV3-Small and its pretrained weights
weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
model = torchvision.models.mobilenet_v3_small(weights=weights)
model.eval()  # removes dropout and batch normalization layers for inference

preprocess = weights.transforms()  # resizing, cropping, normalization used during training

images, labels, paths = [], [], []
for label, cls in enumerate(classes):
    cls_dir = os.path.join(DATA_DIR, cls)
    for img_name in sorted(os.listdir(cls_dir)):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(cls_dir, img_name)
        img = torchvision.io.read_image(img_path)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4:
            img = img[:3]
        images.append(preprocess(img))
        labels.append(label)
        paths.append(img_path)

labels = np.array(labels)
paths  = np.array(paths)

# 3.2.2 Extract features from the 'avgpool' layer of MobileNetV3-Small
feature_extractor = create_feature_extractor(model, return_nodes={'avgpool': 'avgpool'})

features = []
with torch.no_grad():
    for img in images:
        output = feature_extractor(img.unsqueeze(0))   # {'avgpool': (1, 576, 1, 1)}
        features.append(output['avgpool'].squeeze().numpy())  # (576,)

features = np.array(features)

# 3.2.3 Split the dataset, train an RBF SVM, and evaluate its performance
indices = np.arange(len(features))
idx_train, idx_test, y_train, y_test = train_test_split(
    indices, labels, test_size=0.3, random_state=42, stratify=labels
)

# Normalize features — SVMs are sensitive to input scale
scaler  = StandardScaler()
X_train = scaler.fit_transform(features[idx_train])
X_test  = scaler.transform(features[idx_test])

svm = SVC(kernel='rbf', C=10, gamma='scale', decision_function_shape='ovr')
svm.fit(X_train, y_train)

y_pred   = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"CNN (MobileNetV3) + SVM accuracy: {accuracy*100:.2f}%")
print(f"Confusion matrix:\n{conf_matrix}")

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title(f'Confusion Matrix — MobileNet + SVM ({accuracy*100:.2f}%)')
plt.tight_layout()
save_fig('confusion_matrix.jpg')
plt.close()

# 3.2.4 Visualize feature maps from intermediate layers
layer_nodes = {
    'features.1':  'layer_1',
    'features.3':  'layer_3',
    'features.6':  'layer_6',
    'features.12': 'layer_12',
}
intermediate_extractor = create_feature_extractor(model, return_nodes=layer_nodes)

sample_per_class = {}
for img, lbl in zip(images, labels):
    cls = classes[lbl]
    if cls not in sample_per_class:
        sample_per_class[cls] = img
    if len(sample_per_class) == len(classes):
        break

for cls, img in sample_per_class.items():
    with torch.no_grad():
        feat_maps = intermediate_extractor(img.unsqueeze(0))

    fig, axes = plt.subplots(len(layer_nodes), 3, figsize=(10, 3 * len(layer_nodes)))
    fig.suptitle(f'Feature Maps — class: {cls}', fontsize=14, fontweight='bold')

    for row, (node, name) in enumerate(layer_nodes.items()):
        fmap = feat_maps[name].squeeze(0)   # (C, H, W)
        for col in range(3):
            axes[row, col].imshow(fmap[col].numpy(), cmap='viridis')
            axes[row, col].set_title(f'{name} ch{col}', fontsize=9)
            axes[row, col].axis('off')

    plt.tight_layout()
    save_fig(f'feature_maps_{cls}.jpg')
    plt.close()

# 3.2.5 Robustness experiments

def add_gaussian_noise(img_tensor, std=0.1):
    """Add Gaussian noise to a preprocessed tensor."""
    noise = torch.randn_like(img_tensor) * std
    return torch.clamp(img_tensor + noise, 0, 1)

def random_rotation(img_tensor):
    """Rotate tensor by a random angle in [-45, 45] degrees."""
    angle = random.uniform(-45, 45)
    return torchvision.transforms.functional.rotate(img_tensor, angle)

def load_and_augment(augment_fn=None):
    """Load images with optional augmentation."""
    imgs, lbls = [], []
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(DATA_DIR, cls)
        for img_name in sorted(os.listdir(cls_dir)):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img = torchvision.io.read_image(os.path.join(cls_dir, img_name))
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            elif img.shape[0] == 4:
                img = img[:3]
            img = preprocess(img)
            if augment_fn:
                img = augment_fn(img)
            imgs.append(img)
            lbls.append(label)
    return imgs, np.array(lbls)

def evaluate_pipeline(imgs, lbls, label):
    """Extract features, split, train SVM, print accuracy."""
    feats = []
    with torch.no_grad():
        for img in imgs:
            out = feature_extractor(img.unsqueeze(0))
            feats.append(out['avgpool'].squeeze().numpy())
    feats = np.array(feats)

    Xtr, Xte, ytr, yte = train_test_split(feats, lbls, test_size=0.3,
                                           random_state=42, stratify=lbls)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)

    clf = SVC(kernel='rbf', C=10, gamma='scale', decision_function_shape='ovr')
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    print(f"{label} accuracy: {acc*100:.2f}%")

imgs_noise, lbls_noise = load_and_augment(augment_fn=add_gaussian_noise)
imgs_rot,   lbls_rot   = load_and_augment(augment_fn=random_rotation)

evaluate_pipeline(imgs_noise, lbls_noise, "Gaussian noise")
evaluate_pipeline(imgs_rot,   lbls_rot,   "Random rotation")

# 3.2.5b Robustness curve: accuracy vs noise std
base_imgs, base_lbls = load_and_augment(augment_fn=None)
noise_levels, noise_accuracies = np.linspace(0.0, 1.0, 10), []

for std in noise_levels:
    imgs = [add_gaussian_noise(img, std) if std > 0.0 else img for img in base_imgs]
    feats = []
    with torch.no_grad():
        for img in imgs:
            out = feature_extractor(img.unsqueeze(0))
            feats.append(out['avgpool'].squeeze().numpy())
    feats = np.array(feats)

    Xtr, Xte, ytr, yte = train_test_split(feats, base_lbls, test_size=0.3,
                                           random_state=42, stratify=base_lbls)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)

    clf = SVC(kernel='rbf', C=10, gamma='scale', decision_function_shape='ovr')
    clf.fit(Xtr, ytr)
    noise_accuracies.append(accuracy_score(yte, clf.predict(Xte)) * 100)
    print(f"  noise std={std:.2f} → {noise_accuracies[-1]:.2f}%")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(noise_levels, noise_accuracies, marker='o', linewidth=2, color='#e74c3c')
ax.axhline(y=noise_accuracies[0], linestyle='--', color='gray', alpha=0.6, label='No noise baseline')
ax.set_xlabel('Gaussian noise std')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Robustness of MobileNet+SVM to Gaussian Noise')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_fig('robustness_curve.jpg')
plt.close()

# 3.2.6 t-SNE / PCA visualization of avgpool features
colors = ['#e74c3c', '#3498db', '#2ecc71']

n_pca     = min(50, features.shape[1], features.shape[0] - 1)
pca_feats = PCA(n_components=n_pca).fit_transform(features)
pca_2d    = PCA(n_components=2).fit_transform(features)

tsne     = TSNE(n_components=2, perplexity=min(30, len(features) // 3),
                random_state=42, max_iter=1000)
embedded = tsne.fit_transform(pca_feats)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, proj, title, xlabel, ylabel in [
    (axes[0], pca_2d,   'PCA (linear)',       'PC 1',        'PC 2'),
    (axes[1], embedded, 't-SNE (non-linear)', 't-SNE dim 1', 't-SNE dim 2'),
]:
    for lbl, cls, color in zip(range(len(classes)), classes, colors):
        mask = labels == lbl
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=color, label=cls, s=60, alpha=0.85, edgecolors='white', linewidths=0.5)
    ax.legend(fontsize=11)
    ax.set_title(f'{title}\nMobileNetV3-Small avgpool features (576-dim → 2-dim)', fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
save_fig('tsne_pca_features.jpg')
plt.close()

# 3.2.7 Fine-tuning: unfreeze last blocks of MobileNet + 3-class head
device   = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Fine-tuning on: {device}")

ft_model = copy.deepcopy(model).to(device)

for param in ft_model.parameters():
    param.requires_grad = False
for param in ft_model.features[9:].parameters():
    param.requires_grad = True
ft_model.classifier[-1] = torch.nn.Linear(1024, len(classes)).to(device)

X_ft = torch.stack(images)
y_ft = torch.tensor(labels, dtype=torch.long)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, ft_model.parameters()), lr=1e-4
)
criterion = torch.nn.CrossEntropyLoss()

BATCH, EPOCHS = 32, 20
train_idx_tensor = torch.tensor(idx_train)

ft_model.train()
for epoch in range(EPOCHS):
    perm = train_idx_tensor[torch.randperm(len(train_idx_tensor))]
    for start in range(0, len(perm), BATCH):
        batch_idx = perm[start:start + BATCH]
        x = X_ft[batch_idx].to(device)
        y = y_ft[batch_idx].to(device)
        optimizer.zero_grad()
        criterion(ft_model(x), y).backward()
        optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"  epoch {epoch+1}/{EPOCHS} done")

ft_model.eval()
with torch.no_grad():
    preds = ft_model(X_ft[idx_test].to(device)).argmax(dim=1).cpu()
ft_acc = (preds == y_ft[idx_test]).float().mean().item()

print(f"\nFrozen MobileNet + SVM accuracy : {accuracy*100:.2f}%")
print(f"Fine-tuned MobileNet accuracy   : {ft_acc*100:.2f}%")

# 3.2.8 Grad-CAM
# Target: last conv block before avgpool (features.12)
# Hooks must be registered on the original model (not ft_model) since we use it for feature extraction
target_layer = model.features[12]
gradients, activations = [], []

def _save_gradient(grad):
    gradients.append(grad)

def _forward_hook(module, input, output):
    activations.append(output)
    if output.requires_grad:
        output.register_hook(_save_gradient)

target_layer.register_forward_hook(_forward_hook)

def run_gradcam(img_tensor):
    """Return a normalized (0-1) CAM for a single preprocessed image tensor."""
    gradients.clear()
    activations.clear()
    out   = model(img_tensor.unsqueeze(0))
    score = out[0, out.argmax(dim=1).item()]
    model.zero_grad()
    score.backward()
    grad = gradients[0].squeeze(0)   # (C, H, W)
    act  = activations[0].squeeze(0) # (C, H, W)
    cam  = torch.relu((grad.mean(dim=(1, 2))[:, None, None] * act).sum(dim=0))
    cam  = cam.detach().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam

def plot_gradcam(img_rgb, cam, title, filename):
    h, w = img_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * img_rgb + 0.5 * heatmap).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    axes[0].imshow(img_rgb);                   axes[0].set_title('Original'); axes[0].axis('off')
    axes[1].imshow(cam_resized, cmap='jet');   axes[1].set_title('Grad-CAM'); axes[1].axis('off')
    axes[2].imshow(overlay);                   axes[2].set_title('Overlay');  axes[2].axis('off')
    plt.tight_layout()
    save_fig(filename)
    plt.close()

# Per-class Grad-CAM on the first sample of each class
for cls, img in sample_per_class.items():
    img_path = paths[labels == classes.index(cls)][0]
    img_bgr  = cv2.imread(img_path)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cam      = run_gradcam(img)
    plot_gradcam(img_rgb, cam,
                 title=f'Grad-CAM — class: {cls}',
                 filename=f'gradcam_{cls}.jpg')
    print(f"[{cls}] Grad-CAM saved.")

# Grad-CAM on SVM-misclassified test samples
wrong_mask    = y_pred != y_test
wrong_indices = idx_test[wrong_mask]
wrong_true    = y_test[wrong_mask]
wrong_pred    = y_pred[wrong_mask]

if len(wrong_indices) == 0:
    print("No misclassified samples — perfect SVM accuracy on test set!")
else:
    print(f"Found {len(wrong_indices)} misclassified sample(s). Running Grad-CAM...")
    for idx, true_lbl, pred_lbl in zip(wrong_indices, wrong_true, wrong_pred):
        img_bgr   = cv2.imread(paths[idx])
        img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        cam       = run_gradcam(images[idx])
        true_name = classes[true_lbl]
        pred_name = classes[pred_lbl]
        plot_gradcam(img_rgb, cam,
                     title=f'Grad-CAM — MISCLASSIFIED\nTrue: {true_name}  |  SVM predicted: {pred_name}',
                     filename=f'gradcam_misclassified_{true_name}_as_{pred_name}.jpg')
        print(f"  misclassified {true_name} → {pred_name}: Grad-CAM saved.")
