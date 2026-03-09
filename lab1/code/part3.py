import os 
import numpy as np
import torch
import torchvision
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms.functional import rotate
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import random

DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data', 'part3')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
classes = ['car', 'person', 'bike']

# 3.2.1 Load MobileNetV3-Small and its pretrained weights
weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
model = torchvision.models.mobilenet_v3_small(weights=weights)
model.eval()  # removes dropout and batch normalization layers for inference

preprocess = weights.transforms() # returns the preprocessing transforms used during training, including resizing, cropping, and normalization

images, labels = [], []
for label, cls in enumerate(classes):
    cls_dir = os.path.join(DATA_DIR, cls)
    for img_name in sorted(os.listdir(cls_dir)):
        img_path = os.path.join(cls_dir, img_name)
        img = torchvision.io.read_image(img_path)  # reads the image as a tensor
        img = preprocess(img)  # apply the same preprocessing as used during training
        images.append(img)
        labels.append(label)

# 3.2.2 Extract features from the 'avgpool' layer of MobileNetV3-Small
# Remove the classifier head, keep avgpool output
feature_extractor = create_feature_extractor(model, return_nodes={'avgpool': 'avgpool'})

features = []
with torch.no_grad():  # disable gradient computation for inference
    for img in images:
        tensor = img.unsqueeze(0)                          # add batch dimension (1,3,H,W)
        output = feature_extractor(tensor)                 # {'avgpool': (1, 576, 1, 1)}
        features.append(output['avgpool'].squeeze().numpy())  # (576,)
    
features = np.array(features)
labels = np.array(labels)

# 3.2.3 Split the dataset, train a linear SVM, and evaluate its performance

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, 
    test_size=0.3, 
    random_state=42, 
    stratify=labels
)

# Normalize the features to improve SVM performance, as SVMs are sensitive to the scale of the input features. 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train an rbf kernel SVM
svm = SVC(kernel='rbf', C=10, gamma='scale', decision_function_shape='ovr')
svm.fit(X_train, y_train)

#evaluate the SVM on the test set
y_pred = svm.predict(X_test)
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
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.jpg'), dpi=150, bbox_inches='tight')
plt.show()

# 3.2.4 Visualize feature maps from intermediate layers

layer_nodes = {
    'features.1': 'layer_1',
    'features.3': 'layer_3',
    'features.6': 'layer_6',
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
    plt.savefig(os.path.join(RESULTS_DIR, f'feature_maps_{cls}.jpg'))
    plt.show()

# 3.2.5 run experiments with different noise levels and edge detection parameters

def add_gaussian_noise(img_tensor, std=0.1):
    """Add Gaussian noise to a preprocessed tensor."""
    noise = torch.randn_like(img_tensor) * std
    return torch.clamp(img_tensor + noise, 0, 1)

def random_rotation(img_tensor):
    """Rotate tensor by random angle in [-45, 45] degrees."""
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
            out  = feature_extractor(img.unsqueeze(0))
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

# 3.2.5 Run experiments
imgs_noise, lbls_noise = load_and_augment(augment_fn=add_gaussian_noise)
imgs_rot,   lbls_rot   = load_and_augment(augment_fn=random_rotation)

evaluate_pipeline(imgs_noise, lbls_noise, "Gaussian noise")
evaluate_pipeline(imgs_rot,   lbls_rot,   "Random rotation")

# 3.2.5b Robustness curve: accuracy vs noise level
# Sweeps Gaussian noise std over 10 levels → shows at which point the CNN starts to fail

noise_levels = np.linspace(0.0, 1.0, 10)
noise_accuracies = []

# Pre-load images once, apply noise in-memory each iteration
base_imgs, base_lbls = load_and_augment(augment_fn=None)

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
plt.savefig(os.path.join(RESULTS_DIR, 'robustness_curve.jpg'), dpi=150, bbox_inches='tight')
plt.show()

# 3.2.6 t-SNE visualization of avgpool features
# Shows why CNN+SVM works: well-separated clusters → easy hyperplane for SVM

colors = ['#e74c3c', '#3498db', '#2ecc71']

n_pca = min(50, features.shape[1], features.shape[0] - 1)
pca_feats = PCA(n_components=n_pca).fit_transform(features)
pca_2d    = PCA(n_components=2).fit_transform(features)

tsne = TSNE(n_components=2, perplexity=min(30, len(features) // 3),
            random_state=42, max_iter=1000)
embedded = tsne.fit_transform(pca_feats)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, proj, title, xlabel, ylabel in [
    (axes[0], pca_2d,  'PCA (linear)',  'PC 1',        'PC 2'),
    (axes[1], embedded,'t-SNE (non-linear)', 't-SNE dim 1', 't-SNE dim 2'),
]:
    for label, cls, color in zip(range(len(classes)), classes, colors):
        mask = labels == label
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=color, label=cls, s=60, alpha=0.85, edgecolors='white', linewidths=0.5)
    ax.legend(fontsize=11)
    ax.set_title(f'{title}\nMobileNetV3-Small avgpool features (576-dim → 2-dim)', fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'tsne_pca_features.jpg'), dpi=150, bbox_inches='tight')
plt.show()

# 3.2.7 Fine-tuning: unfreeze last 3 blocks of MobileNet + 3-class head
# Compare against frozen feature extractor + SVM (3.2.3)

import copy

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Fine-tuning on: {device}")

ft_model = copy.deepcopy(model).to(device)

for param in ft_model.parameters():
    param.requires_grad = False
for param in ft_model.features[9:].parameters():
    param.requires_grad = True
ft_model.classifier[-1] = torch.nn.Linear(1024, len(classes)).to(device)

idx = np.arange(len(images))
idx_train, idx_test = train_test_split(idx, test_size=0.3, random_state=42, stratify=labels)

X_ft = torch.stack(images)                     # (N, 3, H, W)
y_ft = torch.tensor(labels, dtype=torch.long)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, ft_model.parameters()), lr=1e-4
)
criterion = torch.nn.CrossEntropyLoss()

BATCH = 32
EPOCHS = 20
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