import os
import sys
import copy
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.metrics import accuracy_score, confusion_matrix
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt
import seaborn as sns

# Add utils directory to path
UTILS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'cv26_lab2_part2_3')
sys.path.insert(0, os.path.abspath(UTILS_DIR))
from cv26_lab2_utils import read_video, svm_train_test

DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data', 'cv26_lab2_part2_3', 'UCF11')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'part3')
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASSES      = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

# x3d normalisation constants (ImageNet-video statistics)
_X3D_MEAN = torch.tensor([0.45,  0.45,  0.45 ]).view(1, 3, 1, 1, 1)
_X3D_STD  = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.1.1  Dataset loading
# ═══════════════════════════════════════════════════════════════════════════════

def create_train_test_ucf11(data_dir, classes, class_to_idx):
    """
    Build train/test split for UCF11 based on training_videos.txt.
    Returns file paths + integer labels (no pixel data loaded).
    """
    train_txt = os.path.join(data_dir, 'training_videos.txt')
    with open(train_txt) as f:
        train_set = set(line.strip() for line in f if line.strip())

    paths_train, labels_train = [], []
    paths_test,  labels_test  = [], []

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        label   = class_to_idx[cls]
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.endswith('.mpg'):
                continue
            fpath = os.path.join(cls_dir, fname)
            if fname in train_set:
                paths_train.append(fpath)
                labels_train.append(label)
            else:
                paths_test.append(fpath)
                labels_test.append(label)

    return paths_train, labels_train, paths_test, labels_test


# ═══════════════════════════════════════════════════════════════════════════════
# 3.1.2  MobileNet_v3_small feature extraction
# ═══════════════════════════════════════════════════════════════════════════════

def video_to_mobilenet_feat(fpath, extractor, preprocess):
    """
    Load all frames of a video, extract MobileNet avgpool (576-d) per frame,
    and return the temporal average → (576,).
    """
    video = read_video(fpath, gray=True, num_frames=-1)
    frame_feats = []
    for t in range(video.shape[2]):
        rgb    = np.stack([video[:, :, t]] * 3, axis=0)        # (3, H, W) uint8
        tensor = preprocess(torch.from_numpy(rgb))
        with torch.no_grad():
            feat = extractor(tensor.unsqueeze(0))['avgpool'].squeeze().numpy()
        frame_feats.append(feat)
    return np.mean(frame_feats, axis=0)


def extract_mobilenet_features(paths, extractor, preprocess):
    """Extract temporal-average MobileNet features for a list of video paths."""
    features = []
    for i, path in enumerate(paths, 1):
        features.append(video_to_mobilenet_feat(path, extractor, preprocess))
        if i % 50 == 0 or i == len(paths):
            print(f'    {i}/{len(paths)}')
    return np.array(features)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.1  LoRA MobileNet adaptation + feature extraction
# ═══════════════════════════════════════════════════════════════════════════════

def adapt_lora_mobilenet(mobilenet, preprocess, paths_train, labels_train,
                         device, epochs=50):
    """
    LoRA-adapt a MobileNet_v3_small backbone on the training videos.
    A temporary linear head provides the gradient signal; only LoRA delta
    matrices are updated (original weights stay frozen).
    Returns the adapted model (lora_mn).
    """
    # Pre-load center frame per video to avoid re-reading every epoch
    print('  Pre-loading training frames...')
    frames, label_list = [], []
    for path, lbl in zip(paths_train, labels_train):
        video = read_video(path, gray=True, num_frames=16)
        t     = video.shape[2] // 2
        rgb   = np.stack([video[:, :, t]] * 3, axis=0)
        frames.append(preprocess(torch.from_numpy(rgb)))
        label_list.append(lbl)
    X = torch.stack(frames).to(device)
    y = torch.tensor(label_list, dtype=torch.long).to(device)

    # Inject LoRA into all linear layers
    lora_mn = copy.deepcopy(mobilenet)
    for p in lora_mn.parameters():
        p.requires_grad = False
    lora_mn = get_peft_model(lora_mn, LoraConfig(
        r=8, lora_alpha=16, target_modules='all-linear', lora_dropout=0.1, bias='none',
    ))
    lora_mn.print_trainable_parameters()
    lora_mn.to(device)

    head      = nn.Linear(576, len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW([
        {'params': head.parameters(),                                      'lr': 5e-4},
        {'params': [p for p in lora_mn.parameters() if p.requires_grad],  'lr': 5e-5},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    lora_mn.eval()   # keep BN stats frozen
    head.train()
    print(f'  Training LoRA MobileNet ({epochs} epochs)...')
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        for start in range(0, len(perm), 8):
            idx  = perm[start:start + 8]
            feat = lora_mn.avgpool(lora_mn.features(X[idx])).flatten(1)
            optimizer.zero_grad()
            criterion(head(feat), y[idx]).backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f'    epoch {epoch+1}/{epochs}')

    return lora_mn


def extract_lora_mobilenet_features(paths, lora_mn, preprocess, device):
    """
    Extract temporal-average features from a LoRA-adapted MobileNet backbone.
    Same avg-pool protocol as the frozen version.
    """
    features = []
    for i, path in enumerate(paths, 1):
        video = read_video(path, gray=True, num_frames=-1)
        frame_feats = []
        for t in range(video.shape[2]):
            rgb    = np.stack([video[:, :, t]] * 3, axis=0)
            tensor = preprocess(torch.from_numpy(rgb)).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = lora_mn.avgpool(lora_mn.features(tensor)).flatten(1).squeeze().cpu().numpy()
            frame_feats.append(feat)
        features.append(np.mean(frame_feats, axis=0))
        if i % 50 == 0 or i == len(paths):
            print(f'    {i}/{len(paths)}')
    return np.array(features)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.2.2  x3d_xs video pre-processing
# ═══════════════════════════════════════════════════════════════════════════════

def video_to_x3d_clips(fpath, T=4, H=182, W=182, stride=4):
    """
    Load all frames, resize to H×W, split into non-overlapping clips of T
    frames (stride=4). Last clip padded by repeating its final frame.
    Returns a list of (1, 3, T, H, W) normalised float tensors.
    """
    video = read_video(fpath, gray=True, num_frames=-1)
    frames = []
    for t in range(video.shape[2]):
        frame = cv2.resize(video[:, :, t], (W, H))
        frames.append(np.stack([frame] * 3, axis=0))           # (3, H, W)

    clips = []
    for start in range(0, len(frames), stride):
        clip = list(frames[start:start + T])
        while len(clip) < T:
            clip.append(clip[-1])                              # pad last clip
        arr    = np.stack(clip, axis=1).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)            # (1, 3, T, H, W)
        clips.append((tensor - _X3D_MEAN) / _X3D_STD)
    return clips


# ═══════════════════════════════════════════════════════════════════════════════
# 3.2.3  x3d_xs feature extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_x3d_features(paths, model, dev=torch.device('cpu')):
    """
    For each video: run every 4-frame clip through x3d_xs, capture the
    blocks[5].output_pool vector via a forward hook, then average over clips
    → one feature vector per video.
    """
    buf  = {}
    hook = model.blocks[5].output_pool.register_forward_hook(
        lambda _, __, out: buf.update({'feat': out})
    )
    features = []
    for i, path in enumerate(paths, 1):
        clips      = video_to_x3d_clips(path)
        clip_feats = []
        for clip in clips:
            with torch.no_grad():
                model(clip.to(dev))
            clip_feats.append(buf['feat'].squeeze().cpu().numpy())
        features.append(np.mean(clip_feats, axis=0))
        if i % 50 == 0 or i == len(paths):
            print(f'    {i}/{len(paths)}')
    hook.remove()
    return np.array(features)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.2  LoRA x3d_xs adaptation
# ═══════════════════════════════════════════════════════════════════════════════

def adapt_lora_x3d(model_x3d, paths_train, labels_train, feat_dim,
                   device, epochs=50):
    """
    LoRA-adapt an x3d_xs backbone on the training videos.
    The center clip of each video is pre-loaded for efficiency.
    A forward hook on blocks[5].output_pool captures features with gradients
    so they flow back into the LoRA layers during training.
    Returns the adapted model (lora_x3d).
    """
    # Pre-load center clip per video
    print('  Pre-loading training clips...')
    train_clips = []
    for path in paths_train:
        clips = video_to_x3d_clips(path)
        train_clips.append(clips[len(clips) // 2])

    # Inject LoRA
    lora_x3d = copy.deepcopy(model_x3d)
    for p in lora_x3d.parameters():
        p.requires_grad = False
    lora_x3d = get_peft_model(lora_x3d, LoraConfig(
        r=8, lora_alpha=16, target_modules='all-linear', lora_dropout=0.1, bias='none',
    ))
    lora_x3d.print_trainable_parameters()
    lora_x3d.to(device)

    head      = nn.Linear(feat_dim, len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW([
        {'params': head.parameters(),                                       'lr': 5e-4},
        {'params': [p for p in lora_x3d.parameters() if p.requires_grad],  'lr': 5e-5},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Hook captures blocks[5].output_pool output while keeping it in the
    # computation graph so gradients flow into the LoRA layers
    buf = {}
    lora_x3d.blocks[5].output_pool.register_forward_hook(
        lambda _, __, out: buf.update({'feat': out})
    )

    labels_arr = np.array(labels_train)
    lora_x3d.eval()   # keep BN stats frozen
    head.train()
    print(f'  Training LoRA x3d_xs ({epochs} epochs)...')
    for epoch in range(epochs):
        perm = torch.randperm(len(train_clips))
        for i in perm:
            clip  = train_clips[i].to(device)
            label = torch.tensor([labels_arr[i]], dtype=torch.long).to(device)
            lora_x3d(clip)
            feat  = buf['feat'].flatten(1)
            optimizer.zero_grad()
            criterion(head(feat), label).backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f'    epoch {epoch+1}/{epochs}')

    return lora_x3d


# ═══════════════════════════════════════════════════════════════════════════════
# Classifiers
# ═══════════════════════════════════════════════════════════════════════════════

def run_svm(feat_train, labels_train, feat_test, labels_test, svm_type='linear'):
    """
    Wrapper around the provided svm_train_test utility.
    svm_type: 'linear' or 'chi2' (default: 'linear').
    Returns (accuracy, predictions, confusion_matrix).
    """
    acc, preds = svm_train_test(feat_train, labels_train, feat_test, labels_test,
                                svm_type=svm_type)
    return acc, preds, confusion_matrix(labels_test, preds)


def train_mlp(feat_train, labels_train, feat_test, labels_test,
              num_classes, device, epochs=300, batch_size=32):
    """
    Train a 3-layer MLP on pre-extracted features.
    Uses AdamW + CosineAnnealingLR + CrossEntropyLoss.
    Returns (accuracy, predictions, confusion_matrix).
    """
    input_dim = feat_train.shape[1]
    head = nn.Sequential(
        nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 128),       nn.ReLU(),
        nn.Linear(128, num_classes),
    ).to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    X_tr = torch.tensor(feat_train, dtype=torch.float32)
    y_tr = torch.tensor(labels_train, dtype=torch.long)

    head.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(X_tr))
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            criterion(head(X_tr[idx].to(device)), y_tr[idx].to(device)).backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 100 == 0:
            print(f'    epoch {epoch+1}/{epochs}')

    head.eval()
    with torch.no_grad():
        preds = head(
            torch.tensor(feat_test, dtype=torch.float32).to(device)
        ).argmax(1).cpu().numpy()

    return accuracy_score(labels_test, preds), preds, confusion_matrix(labels_test, preds)


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def save_confusion_matrix(cm, classes, title, filepath):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {filepath}')


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment runner
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── 3.1.1  Load dataset ───────────────────────────────────────────────────
    print('\n=== 3.1.1  Loading UCF11 dataset ===')
    paths_train, labels_train, paths_test, labels_test = create_train_test_ucf11(
        DATA_DIR, CLASSES, CLASS_TO_IDX
    )
    labels_train_arr = np.array(labels_train)
    labels_test_arr  = np.array(labels_test)

    print(f'Classes ({len(CLASSES)}): {CLASSES}')
    print(f'Train: {len(paths_train)}  |  Test: {len(paths_test)}\n')
    print(f'  {"Class":<22} {"Train":>6} {"Test":>6} {"Total":>6}')
    print(f'  {"-"*22} {"-"*6} {"-"*6} {"-"*6}')
    for cls, idx in CLASS_TO_IDX.items():
        n_tr = labels_train.count(idx)
        n_ts = labels_test.count(idx)
        print(f'  {cls:<22} {n_tr:>6} {n_ts:>6} {n_tr+n_ts:>6}')

  

    # ── 3.1.2  MobileNet frozen features ─────────────────────────────────────
    print('\n=== 3.1.2  MobileNet_v3_small frozen features ===')
    mn_weights    = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    mobilenet     = mobilenet_v3_small(weights=mn_weights).eval()
    mn_preprocess = mn_weights.transforms()
    mn_extractor  = create_feature_extractor(mobilenet, return_nodes={'avgpool': 'avgpool'})

    print('  Extracting train features...')
    feat_train_mn = extract_mobilenet_features(paths_train, mn_extractor, mn_preprocess)
    print('  Extracting test features...')
    feat_test_mn  = extract_mobilenet_features(paths_test,  mn_extractor, mn_preprocess)
    print(f'  Feature shape: {feat_train_mn.shape}')

    # ── 3.1.3  Frozen MobileNet — SVM vs MLP ─────────────────────────────────
    print('\n=== 3.1.3  Frozen MobileNet + SVM ===')
    acc_mn_svm, _, cm_mn_svm = run_svm(feat_train_mn, labels_train_arr,
                                        feat_test_mn,  labels_test_arr)
    print(f'  Test accuracy: {acc_mn_svm * 100:.2f}%')
    save_confusion_matrix(cm_mn_svm, CLASSES,
                          f'Frozen MobileNet + SVM  (acc={acc_mn_svm*100:.1f}%)',
                          os.path.join(RESULTS_DIR, 'cm_frozen_mn_svm.png'))

    print('\n=== 3.1.3  Frozen MobileNet + MLP ===')
    acc_mn_mlp, _, cm_mn_mlp = train_mlp(feat_train_mn, labels_train_arr,
                                          feat_test_mn,  labels_test_arr,
                                          len(CLASSES), device)
    print(f'  Test accuracy: {acc_mn_mlp * 100:.2f}%')
    save_confusion_matrix(cm_mn_mlp, CLASSES,
                          f'Frozen MobileNet + MLP  (acc={acc_mn_mlp*100:.1f}%)',
                          os.path.join(RESULTS_DIR, 'cm_frozen_mn_mlp.png'))

    # ── LoRA MobileNet — adaptation + features ────────────────────────────────
    print('\n=== LoRA MobileNet — adaptation ===')
    lora_mn = adapt_lora_mobilenet(mobilenet, mn_preprocess,
                                   paths_train, labels_train, device)

    print('  Extracting LoRA MobileNet train features...')
    feat_train_lora_mn = extract_lora_mobilenet_features(
        paths_train, lora_mn, mn_preprocess, device)
    print('  Extracting LoRA MobileNet test features...')
    feat_test_lora_mn  = extract_lora_mobilenet_features(
        paths_test,  lora_mn, mn_preprocess, device)

    print('\n=== LoRA MobileNet + SVM ===')
    acc_lora_mn_svm, _, cm_lora_mn_svm = run_svm(feat_train_lora_mn, labels_train_arr,
                                                   feat_test_lora_mn,  labels_test_arr)
    print(f'  Test accuracy: {acc_lora_mn_svm * 100:.2f}%')
    save_confusion_matrix(cm_lora_mn_svm, CLASSES,
                          f'LoRA MobileNet + SVM  (acc={acc_lora_mn_svm*100:.1f}%)',
                          os.path.join(RESULTS_DIR, 'cm_lora_mn_svm.png'))

    print('\n=== LoRA MobileNet + MLP ===')
    acc_lora_mn_mlp, _, cm_lora_mn_mlp = train_mlp(feat_train_lora_mn, labels_train_arr,
                                                     feat_test_lora_mn,  labels_test_arr,
                                                     len(CLASSES), device)
    print(f'  Test accuracy: {acc_lora_mn_mlp * 100:.2f}%')
    save_confusion_matrix(cm_lora_mn_mlp, CLASSES,
                          f'LoRA MobileNet + MLP  (acc={acc_lora_mn_mlp*100:.1f}%)',
                          os.path.join(RESULTS_DIR, 'cm_lora_mn_mlp.png'))

    # ── 3.2.1  Load x3d_xs ────────────────────────────────────────────────────
    print('\n=== 3.2.1  Loading pretrained x3d_xs ===')
    model_x3d = torch.hub.load(
        'facebookresearch/pytorchvideo', 'x3d_xs', pretrained=True
    )
    model_x3d.eval()

    # ── 3.2.3  Frozen x3d_xs features ────────────────────────────────────────
    print('\n=== 3.2.3  Frozen x3d_xs features ===')
    print('  Extracting train features...')
    feat_train_x3d = extract_x3d_features(paths_train, model_x3d)
    print('  Extracting test features...')
    feat_test_x3d  = extract_x3d_features(paths_test,  model_x3d)
    print(f'  Feature shape: {feat_train_x3d.shape}')

    print('\n=== Frozen x3d_xs + SVM ===')
    acc_x3d_svm, _, cm_x3d_svm = run_svm(feat_train_x3d, labels_train_arr,
                                           feat_test_x3d,  labels_test_arr)
    print(f'  Test accuracy: {acc_x3d_svm * 100:.2f}%')
    save_confusion_matrix(cm_x3d_svm, CLASSES,
                          f'Frozen x3d_xs + SVM  (acc={acc_x3d_svm*100:.1f}%)',
                          os.path.join(RESULTS_DIR, 'cm_frozen_x3d_svm.png'))

    print('\n=== Frozen x3d_xs + MLP ===')
    acc_x3d_mlp, _, cm_x3d_mlp = train_mlp(feat_train_x3d, labels_train_arr,
                                             feat_test_x3d,  labels_test_arr,
                                             len(CLASSES), device)
    print(f'  Test accuracy: {acc_x3d_mlp * 100:.2f}%')
    save_confusion_matrix(cm_x3d_mlp, CLASSES,
                          f'Frozen x3d_xs + MLP  (acc={acc_x3d_mlp*100:.1f}%)',
                          os.path.join(RESULTS_DIR, 'cm_frozen_x3d_mlp.png'))

    # ── LoRA x3d_xs — adaptation + features ──────────────────────────────────
    print('\n=== LoRA x3d_xs — adaptation ===')
    x3d_feat_dim = feat_train_x3d.shape[1]
    lora_x3d = adapt_lora_x3d(model_x3d, paths_train, labels_train,
                               x3d_feat_dim, device)

    print('  Extracting LoRA x3d train features...')
    feat_train_lora_x3d = extract_x3d_features(paths_train, lora_x3d, dev=device)
    print('  Extracting LoRA x3d test features...')
    feat_test_lora_x3d  = extract_x3d_features(paths_test,  lora_x3d, dev=device)

    print('\n=== LoRA x3d_xs + SVM ===')
    acc_lora_x3d_svm, _, cm_lora_x3d_svm = run_svm(feat_train_lora_x3d, labels_train_arr,
                                                     feat_test_lora_x3d,  labels_test_arr)
    print(f'  Test accuracy: {acc_lora_x3d_svm * 100:.2f}%')
    save_confusion_matrix(cm_lora_x3d_svm, CLASSES,
                          f'LoRA x3d_xs + SVM  (acc={acc_lora_x3d_svm*100:.1f}%)',
                          os.path.join(RESULTS_DIR, 'cm_lora_x3d_svm.png'))

    print('\n=== LoRA x3d_xs + MLP ===')
    acc_lora_x3d_mlp, _, cm_lora_x3d_mlp = train_mlp(feat_train_lora_x3d, labels_train_arr,
                                                       feat_test_lora_x3d,  labels_test_arr,
                                                       len(CLASSES), device)
    print(f'  Test accuracy: {acc_lora_x3d_mlp * 100:.2f}%')
    save_confusion_matrix(cm_lora_x3d_mlp, CLASSES,
                          f'LoRA x3d_xs + MLP  (acc={acc_lora_x3d_mlp*100:.1f}%)',
                          os.path.join(RESULTS_DIR, 'cm_lora_x3d_mlp.png'))

    # ── 3.2.4  Comparison ─────────────────────────────────────────────────────
    results = [
        # (label,               acc_svm,          cm_svm,          acc_mlp,           cm_mlp)
        ('Frozen MobileNet', acc_mn_svm,       cm_mn_svm,       acc_mn_mlp,       cm_mn_mlp),
        ('LoRA MobileNet',   acc_lora_mn_svm,  cm_lora_mn_svm,  acc_lora_mn_mlp,  cm_lora_mn_mlp),
        ('Frozen x3d_xs',    acc_x3d_svm,      cm_x3d_svm,      acc_x3d_mlp,      cm_x3d_mlp),
        ('LoRA x3d_xs',      acc_lora_x3d_svm, cm_lora_x3d_svm, acc_lora_x3d_mlp, cm_lora_x3d_mlp),
    ]

    print('\n=== 3.2.4  Comparison ===')
    print(f'\n  {"Method":<22} {"SVM":>10} {"MLP":>10}')
    print(f'  {"-"*22} {"-"*10} {"-"*10}')
    for name, a_svm, _, a_mlp, __ in results:
        print(f'  {name:<22} {a_svm*100:>9.2f}% {a_mlp*100:>9.2f}%')

    # 4×2 grid: rows = feature set, cols = SVM | MLP
    fig, axes = plt.subplots(4, 2, figsize=(14, 24))
    for row, (name, a_svm, cm_svm, a_mlp, cm_mlp) in enumerate(results):
        for col, (acc, cm, clf) in enumerate([
            (a_svm, cm_svm, 'SVM'),
            (a_mlp, cm_mlp, 'MLP'),
        ]):
            ax = axes[row, col]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'{name} + {clf}\nAcc = {acc*100:.1f}%')

    plt.suptitle('UCF11 — 2D/3D × Frozen/LoRA × SVM/MLP',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    grid_path = os.path.join(RESULTS_DIR, 'cm_comparison_grid.png')
    plt.savefig(grid_path, bbox_inches='tight')
    plt.close()
    print(f'\n  Saved 4×2 grid: {grid_path}')
    print('\nDone.')
