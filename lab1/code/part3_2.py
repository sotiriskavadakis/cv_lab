import os
import copy # for deep copying the model before fine-tuning
import random # for random rotations
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights # load pretrained weights and transforms
from torchvision.models.feature_extraction import create_feature_extractor # extract features from intermediate layers
from sklearn.model_selection import train_test_split
import scipy.io
import cv26_lab1_part3_utils as p3
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.decomposition import PCA
from peft import LoraConfig, get_peft_model
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

classes = ['person', 'car', 'bike']  

# 3.2.1 Load MobileNetV3-Small and its pretrained weights

weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 # load pretrained weights trained on ImageNet-1K
model = torchvision.models.mobilenet_v3_small(weights=weights) # load the model architecture and initialize with pretrained weights
model.eval()  # removes dropout and batch normalization layers: we only want inference mode

print("MobileNetV3-Small architecture:")
for name, module in model.named_modules():
    print(f"  {name}: {module.__class__.__name__}")

# this should be applied to every input image before feeding it to the model, to ensure it has the same size and normalization as the training data
preprocess = weights.transforms()  # resizing, cropping, normalization used during training

images, labels, paths = [], [], []
for label, cls in enumerate(classes): # for each class take corresponding label (0, 1, 2) and class name
    cls_dir = os.path.join(DATA_DIR, cls) # path to the class directory
    for img_name in sorted(os.listdir(cls_dir)): # for each image in the class directory
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(cls_dir, img_name)
        img = torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.RGB) # read image as RGB tensor (C, H, W) with values in [0, 255]
        images.append(preprocess(img)) # preprocess the image and add to list (C, H, W) with values in [0, 1]
        labels.append(label) # add label of image to list
        paths.append(img_path)

labels = np.array(labels) # convert to numpy array for easier indexing later
paths  = np.array(paths)

# 3.2.2 Extract features from the 'avgpool' layer of MobileNetV3-Small

# create a feature extractor that returns the output of the avgpool layer (after global average pooling, before the final classifier)
feature_extractor = create_feature_extractor(model, return_nodes={'avgpool': 'avgpool'})

features = [] # feature vectors will be stored here
with torch.no_grad(): # dont compute gradients
    for img in images: # for each preprocessed image tensor (C, H, W)
        output = feature_extractor(img.unsqueeze(0)) # forward pass through the network and get the avgpool output, unsqueeze to add batch dimension (1, C, H, W)
        features.append(output['avgpool'].squeeze().numpy())  # add feature vector to list

features = np.array(features)

# 3.2.3 Split the dataset, train an SVM, and evaluate its performance
# 5-fold cross-validation using createTrainTest and svm from part 3.1 utils

# Group features by class (createTrainTest expects features[class][image])
features_by_class = [
    [features[i] for i in range(len(features)) if labels[i] == c]
    for c in range(len(classes))
]
# Track original flat indices per class (needed for downstream LoRA/Grad-CAM)
_orig_idx_by_class = [np.where(labels == c)[0] for c in range(len(classes))]

_orig_cwd = os.getcwd()
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))  # Fold_Indices.mat is at project root

fold_accuracies = []
conf_matrix = None
for k in range(5):
    data_tr, lbl_tr, data_te, lbl_te = p3.createTrainTest(features_by_class, k)
    X_tr, X_te = np.array(data_tr), np.array(data_te)
    y_tr, y_te = np.array(lbl_tr), np.array(lbl_te)

    sc = StandardScaler()
    X_tr, X_te = sc.fit_transform(X_tr), sc.transform(X_te)

    fold_acc, y_pred_k, _ = p3.svm(X_tr, y_tr, X_te, y_te)
    fold_accuracies.append(fold_acc)
    print(f"  Fold {k+1}: {fold_acc*100:.2f}%")

    fold_cm = confusion_matrix(y_te, y_pred_k, labels=list(range(len(classes))))
    conf_matrix = fold_cm if conf_matrix is None else conf_matrix + fold_cm

conf_matrix = (conf_matrix / 5).round().astype(int)
accuracy = np.mean(fold_accuracies)
print(f"CNN (MobileNetV3) + SVM 5-fold accuracy: {accuracy*100:.2f}% (±{np.std(fold_accuracies)*100:.2f}%)")
print(f"Confusion matrix (averaged over folds):\n{conf_matrix}")

# Keep last fold for downstream use (LoRA, Grad-CAM)
mat = scipy.io.loadmat('./Fold_Indices.mat')
_fold_idx = mat['Indices'].flatten()[4].flatten()  # last fold (k=4)
idx_train_list, idx_test_list = [], []
for c in range(len(classes)):
    ic = _fold_idx[c].flatten()
    orig = _orig_idx_by_class[c][ic]
    lim = int(round(0.7 * len(ic)))
    idx_train_list.extend(orig[:lim])
    idx_test_list.extend(orig[lim:])
idx_train = np.array(idx_train_list)
idx_test = np.array(idx_test_list)
y_train, y_test = labels[idx_train], labels[idx_test]
scaler = StandardScaler()
X_train = scaler.fit_transform(features[idx_train])
X_test  = scaler.transform(features[idx_test])
accuracy_last_fold, y_pred, _ = p3.svm(X_train, y_train, X_test, y_test)
print(f"Last fold (baseline for comparison table): {accuracy_last_fold*100:.2f}%")

os.chdir(_orig_cwd)

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.tight_layout()
save_fig('confusion_matrix.jpg')
plt.close()

# 3.2.4 Visualize feature maps from intermediate layers

# which intermediate layers of the network to extract feature maps from
layer_nodes = {
    'features.1':  'layer_1',
    'features.3':  'layer_3',
    'features.6':  'layer_6',
    'features.12': 'layer_12',
}
intermediate_extractor = create_feature_extractor(model, return_nodes=layer_nodes)

# select one sample from each class to visualize the feature maps
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
    return img_tensor + torch.randn_like(img_tensor) * std

def random_rotation(img_tensor):
    angle = random.uniform(-45, 45)
    return torchvision.transforms.functional.rotate(img_tensor, angle)

def load_and_augment(augment_fn=None):
    imgs, lbls = [], []
    for label, cls in enumerate(classes):
        for img_name in sorted(os.listdir(os.path.join(DATA_DIR, cls))):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img = torchvision.io.read_image(
                os.path.join(DATA_DIR, cls, img_name),
                mode=torchvision.io.ImageReadMode.RGB,
            )
            img = preprocess(img)
            if augment_fn:
                img = augment_fn(img)
            imgs.append(img)
            lbls.append(label)
    return imgs, np.array(lbls)

def evaluate_pipeline(imgs, lbls, label):
    feats = []
    with torch.no_grad():
        for img in imgs:
            out = feature_extractor(img.unsqueeze(0))
            feats.append(out['avgpool'].squeeze().numpy())
    feats = np.array(feats)
    Xtr, Xte, ytr, yte = train_test_split(
        feats, lbls, test_size=0.3, random_state=42, stratify=lbls
    )
    sc = StandardScaler()
    Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)
    acc, _, _ = p3.svm(Xtr, ytr, Xte, yte)
    print(f"{label} accuracy: {acc*100:.2f}%")

evaluate_pipeline(*load_and_augment(add_gaussian_noise), "Gaussian noise")
evaluate_pipeline(*load_and_augment(random_rotation),    "Random rotation")

# 3.2.6 (Bonus) Four-way comparison: Frozen+SVM / LoRA+SVM / Frozen+MLP / LoRA+MLP

# for MacOS with Apple Silicon we use mps device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"\nTraining device: {device}")

X_all = torch.stack(images)
y_all = torch.tensor(labels, dtype=torch.long)
train_idx_tensor = torch.tensor(idx_train)


def train_mlp(feat_train, y_train_t, feat_test, y_test_t,
              epochs=500, batch_size=32, lr=1e-3):
    """Train MLP head on pre-extracted features."""
    head = nn.Sequential(
        nn.Linear(576, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, len(classes)),
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    X_tr = torch.tensor(feat_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train_t, dtype=torch.long)

    head.train()
    t0 = time.time()
    for epoch in range(epochs):
        perm = torch.randperm(len(X_tr))
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            criterion(head(X_tr[idx].to(device)), y_tr[idx].to(device)).backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch+1}/{epochs}")
    train_time = time.time() - t0

    head.eval()
    with torch.no_grad():
        preds = head(torch.tensor(feat_test, dtype=torch.float32).to(device)).argmax(1).cpu().numpy()
    acc = accuracy_score(y_test_t, preds)
    f1 = f1_score(y_test_t, preds, average='macro')
    return head, acc, f1, preds, train_time


# (A) Baseline: frozen backbone + linear SVM (already trained above)
svm_f1 = f1_score(y_test, y_pred, average='macro')
t0 = time.time()
p3.svm(X_train, y_train, X_test, y_test)
svm_time = time.time() - t0
print(f"\n[Baseline] Frozen+Linear SVM  — Acc: {accuracy_last_fold*100:.2f}%  F1: {svm_f1:.4f}")

# (A2) Frozen backbone + RBF SVM (C=10)
rbf_svm = SVC(kernel='rbf', C=10, gamma='scale', decision_function_shape='ovr')
t0 = time.time()
rbf_svm.fit(X_train, y_train)
rbf_svm_time = time.time() - t0
rbf_y_pred = rbf_svm.predict(X_test)
rbf_svm_acc = accuracy_score(y_test, rbf_y_pred)
rbf_svm_f1 = f1_score(y_test, rbf_y_pred, average='macro')
print(f"[Frozen+RBF SVM] Acc: {rbf_svm_acc*100:.2f}%  F1: {rbf_svm_f1:.4f}  (C=10, gamma=scale)")

# (B) LoRA backbone adaptation (train end-to-end to adapt features)
print("\n--- Training: LoRA backbone adaptation ---")
lora_backbone = copy.deepcopy(model)
for p in lora_backbone.parameters():
    p.requires_grad = False
lora_backbone = get_peft_model(lora_backbone, LoraConfig(
    r=8, lora_alpha=16, target_modules="all-linear",
    lora_dropout=0.1, bias='none',
))
lora_backbone.print_trainable_parameters()
lora_head = nn.Sequential(
    nn.Linear(576, 256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, len(classes)),
).to(device)
lora_backbone.to(device)

optimizer = torch.optim.AdamW([
    {'params': lora_head.parameters(), 'lr': 5e-4},
    {'params': [p for p in lora_backbone.parameters() if p.requires_grad], 'lr': 5e-5},
], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
criterion = nn.CrossEntropyLoss()
augment = torchvision.transforms.RandomHorizontalFlip(p=0.5)

lora_head.train()
lora_backbone.eval()  # keep BN stats frozen
t0 = time.time()
for epoch in range(50):
    perm = train_idx_tensor[torch.randperm(len(train_idx_tensor))]
    for start in range(0, len(perm), 8):
        idx = perm[start : start + 8]
        x = augment(X_all[idx]).to(device)
        feat = lora_backbone.avgpool(lora_backbone.features(x)).flatten(1)
        optimizer.zero_grad()
        criterion(lora_head(feat), y_all[idx].to(device)).backward()
        optimizer.step()
    scheduler.step()
    if (epoch + 1) % 10 == 0:
        print(f"  epoch {epoch+1}/50")
lora_backbone_time = time.time() - t0

# Extract LoRA-adapted features
lora_backbone.eval()
lora_feats = []
with torch.no_grad():
    for img in images:
        feat = lora_backbone.avgpool(lora_backbone.features(img.unsqueeze(0).to(device))).flatten(1)
        lora_feats.append(feat.cpu().squeeze().numpy())
lora_feats = np.array(lora_feats)
sc_lora = StandardScaler()
X_tr_lora = sc_lora.fit_transform(lora_feats[idx_train])
X_te_lora = sc_lora.transform(lora_feats[idx_test])

# (C) LoRA + SVM
print("\n--- Evaluating: LoRA + SVM ---")
t0 = time.time()
lora_svm_acc, lora_svm_preds, _ = p3.svm(X_tr_lora, y_train, X_te_lora, y_test)
lora_svm_time = lora_backbone_time + (time.time() - t0)
lora_svm_f1 = f1_score(y_test, lora_svm_preds, average='macro')
print(f"[LoRA+SVM] Acc: {lora_svm_acc*100:.2f}%  F1: {lora_svm_f1:.4f}  Time: {lora_svm_time:.1f}s")

# (D) LoRA + MLP: train MLP on LoRA-adapted features
print("\n--- Training: LoRA + MLP (on pre-extracted features) ---")
lora_head, lora_acc, lora_f1, lora_preds, lora_mlp_time = train_mlp(
    X_tr_lora, y_train, X_te_lora, y_test,
)
lora_time = lora_backbone_time + lora_mlp_time
print(f"[LoRA+MLP] Acc: {lora_acc*100:.2f}%  F1: {lora_f1:.4f}  Time: {lora_time:.1f}s")

# (E) Frozen + MLP: train MLP on frozen features
print("\n--- Training: Frozen + MLP (on pre-extracted features) ---")
frozen_head, nh_acc, nh_f1, nh_preds, nh_time = train_mlp(
    X_train, y_train, X_test, y_test,
)
print(f"[Frozen+MLP] Acc: {nh_acc*100:.2f}%  F1: {nh_f1:.4f}  Time: {nh_time:.1f}s")

# --- Comparison table ---------------------------------------------------------
results = [
    ("Frozen + Linear SVM",     accuracy_last_fold, svm_f1,     svm_time),
    ("Frozen + RBF SVM",        rbf_svm_acc,        rbf_svm_f1, rbf_svm_time),
    ("LoRA + SVM",              lora_svm_acc, lora_svm_f1, lora_svm_time),
    ("Frozen + MLP",            nh_acc,       nh_f1,       nh_time),
    ("LoRA + MLP",              lora_acc,     lora_f1,     lora_time),
]

print("\n" + "=" * 65)
print(f"{'Method':<28} {'Accuracy':>10} {'F1-Score':>10} {'Time (s)':>10}")
print("-" * 65)
for name, a, f, t in results:
    print(f"{name:<28} {a*100:>9.2f}% {f:>10.4f} {t:>10.1f}")
print("=" * 65)

fig, ax = plt.subplots(figsize=(9, 3))
ax.axis('off')
table = ax.table(
    cellText=[[n, f"{a*100:.2f}%", f"{f:.4f}", f"{t:.1f}s"] for n, a, f, t in results],
    colLabels=["Method", "Accuracy", "F1-Score", "Training Time"],
    cellLoc='center', loc='center',
)
table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.2, 1.6)
ax.set_title("4-Way Comparison: Feature Extractor × Classifier",
             fontsize=13, fontweight='bold', pad=20)
plt.tight_layout(); save_fig('comparison_table.jpg'); plt.close()

# 3.2.7 Grad-CAM: frozen vs LoRA comparison

# implementing the findings of this paper https://arxiv.org/pdf/1610.02391
# used for explainability reasons

class BackboneWithHead(nn.Module):
    """Combines a MobileNetV3 backbone (or LoRA-wrapped) with a 3-class MLP head
    so that Grad-CAM gradients flow through the task-specific classifier,
    not the original ImageNet head."""
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        feat = self.backbone.avgpool(self.backbone.features(x)).flatten(1)
        return self.head(feat)


def setup_gradcam(target_model, layer_path):
    grads, acts = [], []

    def _hook(_module, _input, output):
        acts.append(output)
        if output.requires_grad:
            output.register_hook(lambda g: grads.append(g))

    # Resolve dotted layer path (e.g. 'backbone.base_model.model.features.12')
    layer = target_model
    for p in layer_path.split('.'):
        layer = layer[int(p)] if p.isdigit() else getattr(layer, p)
    layer.register_forward_hook(_hook)

    def run(img_tensor):
        grads.clear(); acts.clear()
        dev = next(target_model.parameters()).device
        inp = img_tensor.unsqueeze(0).to(dev).requires_grad_(True)
        out = target_model(inp)
        target_model.zero_grad()
        out[0, out.argmax(1).item()].backward()
        g, a = grads[0].squeeze(0), acts[0].squeeze(0)
        cam = torch.relu((g.mean(dim=(1, 2))[:, None, None] * a).sum(0))
        cam = cam.detach().cpu().numpy()
        cam -= cam.min()
        return cam / cam.max() if cam.max() > 0 else cam

    return run


def plot_gradcam_comparison(img_rgb, cam_frozen, cam_lora, title, filename):
    h, w = img_rgb.shape[:2]
    cam_f = cv2.resize(cam_frozen, (w, h))
    cam_l = cv2.resize(cam_lora, (w, h))
    heatmap_l = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(255 * cam_l), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB
    )
    overlay = (0.5 * img_rgb + 0.5 * heatmap_l).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    for ax, data, cmap, lbl in [
        (axes[0], img_rgb, None,  'Original'),
        (axes[1], cam_f,   'jet', 'Frozen Backbone'),
        (axes[2], cam_l,   'jet', 'LoRA + MLP'),
        (axes[3], overlay, None,  'LoRA Overlay'),
    ]:
        ax.imshow(data, cmap=cmap); ax.set_title(lbl); ax.axis('off')
    plt.tight_layout(); save_fig(filename); plt.close()


# Wrap each backbone with its trained 3-class head so Grad-CAM gradients
# flow through the task-specific classifier (not the 1000-class ImageNet head).
frozen_gradcam_model = BackboneWithHead(model.to(device), frozen_head)
lora_gradcam_model   = BackboneWithHead(lora_backbone, lora_head)

# Layer paths are relative to BackboneWithHead (backbone.* prefix)
run_gradcam_frozen = setup_gradcam(frozen_gradcam_model, 'backbone.features.12')
run_gradcam_lora   = setup_gradcam(lora_gradcam_model,   'backbone.base_model.model.features.12')

# Per-class Grad-CAM comparison
for cls, img in sample_per_class.items():
    img_rgb = cv2.cvtColor(cv2.imread(paths[labels == classes.index(cls)][0]), cv2.COLOR_BGR2RGB)
    plot_gradcam_comparison(
        img_rgb, run_gradcam_frozen(img), run_gradcam_lora(img),
        f'Grad-CAM Comparison — class: {cls}', f'gradcam_comparison_{cls}.jpg',
    )
    print(f"[{cls}] Grad-CAM comparison saved.")

# Misclassified samples Grad-CAM
wrong = y_pred != y_test
if not wrong.any():
    print("No misclassified samples!")
else:
    print(f"Found {wrong.sum()} misclassified sample(s). Running Grad-CAM...")
    for i, (idx, t_lbl, p_lbl) in enumerate(
        zip(idx_test[wrong], y_test[wrong], y_pred[wrong])
    ):
        img_rgb = cv2.cvtColor(cv2.imread(paths[idx]), cv2.COLOR_BGR2RGB)
        t_name, p_name = classes[t_lbl], classes[p_lbl]
        plot_gradcam_comparison(
            img_rgb, run_gradcam_frozen(images[idx]), run_gradcam_lora(images[idx]),
            f'Grad-CAM — MISCLASSIFIED\nTrue: {t_name}  |  Predicted: {p_name}',
            f'gradcam_misclassified_{i}_{t_name}_as_{p_name}.jpg',
        )
        print(f"  {t_name} → {p_name}: saved.")

# 3.2.8 PCA: frozen vs LoRA-adapted features
colors = ['red', 'green', 'blue']
pca_frozen = PCA(n_components=2).fit_transform(features)
pca_lora   = PCA(n_components=2).fit_transform(lora_feats)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, proj, title in [
    (axes[0], pca_frozen, 'PCA — Frozen Backbone'),
    (axes[1], pca_lora,   'PCA — LoRA Backbone'),
]:
    for lbl, cls, color in zip(range(len(classes)), classes, colors):
        mask = labels == lbl
        ax.scatter(proj[mask, 0], proj[mask, 1], c=color, label=cls,
                   s=60, alpha=0.85, edgecolors='white', linewidths=0.5)
    ax.legend(fontsize=11)
    ax.set_title(f'{title}\navgpool features (576-dim → 2-dim)', fontsize=12)
    ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2'); ax.grid(True, alpha=0.3)

plt.suptitle('Feature Clustering: Frozen vs LoRA-adapted Backbone',
             fontsize=15, fontweight='bold')
plt.tight_layout(); save_fig('pca_frozen_vs_lora.jpg'); plt.close()

print("\nAll done. Results saved to:", RESULTS_DIR)
