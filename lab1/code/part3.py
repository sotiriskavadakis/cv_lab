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
import matplotlib.pyplot as plt
import random

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'part3')
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
    plt.savefig(f'feature_maps_{cls}.jpg')
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