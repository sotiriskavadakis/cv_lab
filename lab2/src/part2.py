import sys
import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import cv2
import glob
import pickle
from PIL import Image
from scipy.ndimage import convolve1d, maximum_filter
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

np.random.seed(42)  # for reproducibility

lab2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(lab2_dir, 'data', 'cv26_lab2_part2_3'))
from cv26_lab2_utils import read_video, show_detection, orientation_histogram, bag_of_words, svm_train_test

data_dir     = os.path.join(lab2_dir, 'data', 'cv26_lab2_part2_3', 'KTH')
results_dir  = os.path.join(lab2_dir, 'results')
pictures_dir = os.path.join(lab2_dir, 'docs', 'pictures')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(pictures_dir, exist_ok=True)

# clean up old outputs from this script
for old in glob.glob(os.path.join(results_dir, '*.png')) + glob.glob(os.path.join(results_dir, '*.gif')):
    os.remove(old)

# ─── 2.1: interest point detectors (Harris3D, Gabor) ──────────────────────────────────────────────

def gaussian_smooth_3d(volume, sigma_s, sigma_t):
    n_s = int(np.ceil(3 * sigma_s)) * 2 + 1
    n_t = int(np.ceil(3 * sigma_t)) * 2 + 1
    k_s = cv2.getGaussianKernel(n_s, sigma_s).flatten()
    k_t = cv2.getGaussianKernel(n_t, sigma_t).flatten()
    out = convolve1d(volume, k_s, axis=0)
    out = convolve1d(out, k_s, axis=1)
    out = convolve1d(out, k_t, axis=2)
    return out

#def harris_detector(video, sigma=4, tau=1.5, k=0.00005, s=1, thresh=0.0001, top_n=500):
def harris_detector(video, sigma=4, tau=1.5, k=0.005, s=2, thresh=0.1, top_n=500):
    """Spatiotemporal Harris interest point detector (3D Harris-Stephens).
    Returns: (points, H) where points is Nx4 [x,y,t,sigma] and H is the response volume."""
    video = video.astype(np.float64) / 255.0

    L = gaussian_smooth_3d(video, sigma, tau)

    # spatiotemporal derivatives via central differences [-1, 0, 1]
    d = np.array([-1.0, 0.0, 1.0])
    Lx = convolve1d(L, d, axis=1)
    Ly = convolve1d(L, d, axis=0)
    Lt = convolve1d(L, d, axis=2)

    # structure tensor M products
    Lx2  = Lx * Lx
    Ly2  = Ly * Ly
    Lt2  = Lt * Lt
    LxLy = Lx * Ly
    LxLt = Lx * Lt
    LyLt = Ly * Lt

    # smooth each product of the structure tensor
    Lx2  = gaussian_smooth_3d(Lx2,  s * sigma, s * tau)
    Ly2  = gaussian_smooth_3d(Ly2,  s * sigma, s * tau)
    Lt2  = gaussian_smooth_3d(Lt2,  s * sigma, s * tau)
    LxLy = gaussian_smooth_3d(LxLy, s * sigma, s * tau)
    LxLt = gaussian_smooth_3d(LxLt, s * sigma, s * tau)
    LyLt = gaussian_smooth_3d(LyLt, s * sigma, s * tau)

    # Harris criterion: H = det(M) - k * trace(M)^3
    trace_M = Lx2 + Ly2 + Lt2
    det_M = (Lx2 * (Ly2 * Lt2 - LyLt ** 2)
             - LxLy * (LxLy * Lt2 - LyLt * LxLt)
             + LxLt * (LxLy * LyLt - Ly2 * LxLt))
    H = det_M - k * (trace_M ** 3)

    # 2.1.3: keep only points with H > thresh * H_max (reject smooth/low-response regions)
    # 2.1.4: keep local maxima (3x3x3 NMS), return top-N strongest
    threshold = thresh * H.max()
    H_nms = maximum_filter(H, size=3)
    flat = H.ravel().copy()
    flat[~((H == H_nms) & (H > threshold)).ravel()] = 0
    top_idx = np.argsort(flat)[::-1][:top_n]
    top_idx = top_idx[flat[top_idx] > 0]
    coords = np.array(np.unravel_index(top_idx, H.shape)).T  # (N, 3): y, x, t

    if len(coords) == 0:
        return np.empty((0, 4)), H

    points = np.column_stack([
        coords[:, 1], coords[:, 0], coords[:, 2],
        np.full(len(coords), sigma)
    ])
    return points, H

def gabor_detector(video, sigma=4, tau=1.5, thresh=0.1, top_n=500):
    """Spatiotemporal Gabor interest point detector."""
    video = video.astype(np.float64) / 255.0

    # gabor filters on the window of [-2*τ, 2*τ] with frequency 4/τ
    omega = 4.0 / tau
    t = np.arange(-int(2 * tau), int(2 * tau) + 1, dtype=np.float64) # temporal window
    gaussian = np.exp(-t ** 2 / (2 * tau ** 2)) # 
    hev = np.cos(2 * np.pi * t * omega) * gaussian  # even (cos)
    hod = np.sin(2 * np.pi * t * omega) * gaussian  # odd  (sin)
    # L1 normalisation
    hev = hev / np.abs(hev).sum()
    hod = hod / np.abs(hod).sum()

    # spatial smoothing (x,y plane) with 2D Gaussian kernel
    n_s = int(np.ceil(3 * sigma)) * 2 + 1 # kernel size
    k_s = cv2.getGaussianKernel(n_s, sigma).flatten() # 1D Gaussian kernel
    I_smooth = convolve1d(video, k_s, axis=0) # smooth along y-axis
    I_smooth = convolve1d(I_smooth, k_s, axis=1) # smooth along x-axis

    # convolve the smoothed video with the gabor filters along the temporal axis
    resp_ev = convolve1d(I_smooth, hev, axis=2)
    resp_od = convolve1d(I_smooth, hod, axis=2)

    # calculate the gabor response magnitude
    H = resp_ev ** 2 + resp_od ** 2

    # 2.1.3: keep only points with H > thresh * H_max (reject smooth/low-response regions)
    # 2.1.4: keep local maxima (3x3x3 NMS), return top-N strongest
    threshold = thresh * H.max()
    H_nms = maximum_filter(H, size=3)
    flat = H.ravel().copy()
    flat[~((H == H_nms) & (H > threshold)).ravel()] = 0
    top_idx = np.argsort(flat)[::-1][:top_n]
    top_idx = top_idx[flat[top_idx] > 0]
    coords = np.array(np.unravel_index(top_idx, H.shape)).T

    if len(coords) == 0:
        return np.empty((0, 4)), H

    points = np.column_stack([
        coords[:, 1], coords[:, 0], coords[:, 2],
        np.full(len(coords), sigma)
    ])
    return points, H

# ─── 2.2: HOG/HOF descriptors ───────────────────────────────────────────────

def compute_descriptors(video, points, nbins=8, ncells=3):
    """Compute HOG/HOF descriptors for each interest point.

    2.2.1: gradient (Sobel) and TV-L1 optical flow per frame.
    2.2.2: orientation_histogram on a 4*sigma patch around each point,
           concatenate HOG and HOF into a single descriptor.

    Args:
        video: H x W x T uint8 grayscale video
        points: Nx4 array [x, y, t, sigma]
        nbins: number of histogram bins
        ncells: grid size for orientation_histogram (ncells x ncells)

    Returns:
        descriptors: Nx(2 * ncells * ncells * nbins) array
    """
    H, W, T = video.shape # video dimensions (height, width, frames)
    desc_dim = 2 * ncells * ncells * nbins # HOG + HOF dimensions (concatenated)
    descriptors = np.zeros((len(points), desc_dim)) # this is what we will return

    # 2.2.1: precompute gradient (Gx, Gy) for each frame via central differences [-1, 0, 1] (for HOG)
    d = np.array([-1.0, 0.0, 1.0]) # central difference kernel 
    video_f = video.astype(np.float32) # convert to float for convolution
    Gx_all = convolve1d(video_f, d, axis=1) # gradient along x-axis
    Gy_all = convolve1d(video_f, d, axis=0) # gradient along y-axis

    # 2.2.1: precompute TV-L1 optical flow 
    # input must be uint8!!
    # we compute the optical flow for each pixel as a vector (fx, fy), meaning how much it moves in x and y from frame t to t+1.
    oflow = cv2.optflow.DualTVL1OpticalFlow_create(nscales=1) # TV-L1 optical flow object
    flow_all = {}  # here we store the optical flow vectors for each frame pair (t, t+1)
    for fr in range(T - 1): # for each frame except the last one
        flow_all[fr] = oflow.calc(video[:, :, fr], video[:, :, fr + 1], None) # calc the optical flow between frame fr and fr+1
    # now flow_all[fr] is an H x W x 2 array where flow_all[fr][:, :, 0] is fx and flow_all[fr][:, :, 1] is fy for frame fr
    # and we also have the gradients Gx_all and Gy_all for each frame
    # and we will go ahead and compute the HOG and HOF descriptors for each interest point in the next step

    # 2.2.2: extract HOG/HOF descriptors per interest point
    for i, (x, y, t, sigma) in enumerate(points): # for each interest point:
        x, y, t = int(x), int(y), int(t) # they were originally floats -> convert to integers
        half = int(2 * sigma)  # the patch is +/- 2*sigma around the point

        # make sure we dont go out of bounds
        y0 = max(0, y - half)
        y1 = min(H, y + half)
        x0 = max(0, x - half)
        x1 = min(W, x + half)

        # skip degenerate patches
        if y1 - y0 < 2 or x1 - x0 < 2:
            continue

        # HOG: gradient patch
        gx_patch = Gx_all[y0:y1, x0:x1, t].astype(np.float32)
        gy_patch = Gy_all[y0:y1, x0:x1, t].astype(np.float32)
        hog = orientation_histogram(gx_patch, gy_patch, nbins, np.array([ncells, ncells]))

        # HOF: optical flow patch
        if t < T - 1 and t in flow_all:
            flow = flow_all[t]
            fx_patch = flow[y0:y1, x0:x1, 0].astype(np.float32)
            fy_patch = flow[y0:y1, x0:x1, 1].astype(np.float32)
        else:
            # last frame: no forward flow, use zeros
            fx_patch = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
            fy_patch = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        hof = orientation_histogram(fx_patch, fy_patch, nbins, np.array([ncells, ncells]))

        # concatenate HOG and HOF
        descriptors[i] = np.concatenate([hog, hof])

    return descriptors

# one sample video per action for visualization (2.1.4)
sample_videos = {
    'walking':      os.path.join(data_dir, 'walking',      'person04_walking_d1_uncomp.avi'),
    'running':      os.path.join(data_dir, 'running',      os.listdir(os.path.join(data_dir, 'running'))[0]),
    'handclapping': os.path.join(data_dir, 'handclapping', os.listdir(os.path.join(data_dir, 'handclapping'))[0]),
}
selected_frames = [10, 25, 40]

# ─── 2.1.4: visualize H map + detections (always regenerated) ────────────────

viz_configs = [
    ('harris', harris_detector, 2, 1.5, 0.1, 500),
    ('gabor',  gabor_detector,  2, 1.5, 0.1, 500),
]
print("\n=== 2.1.4: Visualization ===")
for det_name, det_func, sigma, tau, thresh, top_n in viz_configs:
    tag = f"{det_name}_s{sigma}_t{tau}_thr{thresh}"
    for s_action, s_path in sample_videos.items():
        s_video = read_video(s_path, gray=True, num_frames=50)
        s_pts, s_H = det_func(s_video, sigma=sigma, tau=tau, thresh=thresh, top_n=top_n)
        print(f"  {det_name} {s_action}: {len(s_pts)} pts")

        for frame_idx in selected_frames:
            if frame_idx >= s_video.shape[2]:
                continue

            # H map
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(s_H[:, :, frame_idx], cmap='hot')
            ax.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(results_dir, f"{s_action}_{tag}_frame{frame_idx}_H.png"),
                        dpi=150, bbox_inches='tight')
            plt.close()

            # detections
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(s_video[:, :, frame_idx], cmap='gray')
            frame_pts = s_pts[s_pts[:, 2] == frame_idx] if len(s_pts) > 0 else []
            for p in frame_pts:
                ax.add_patch(Circle((int(p[0]), int(p[1])), 2 * int(p[3]),
                                    edgecolor='g', fill=False, linewidth=1.5))
            ax.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(results_dir, f"{s_action}_{tag}_frame{frame_idx}_det.png"),
                        dpi=150, bbox_inches='tight')
            plt.close()

# ─── 2.3: Bag of Visual Words + SVM classification ──────────────────────────

actions = ['running', 'handclapping', 'walking']
label_map = {a: i for i, a in enumerate(actions)}

# 2.3.1: train/test split based on provided file
train_file = os.path.join(data_dir, 'traininng_videos.txt')
with open(train_file) as f:
    train_names = set(line.strip() for line in f if line.strip())

all_videos = []
for action in actions:
    for fname in sorted(os.listdir(os.path.join(data_dir, action))):
        if fname.endswith('.avi'):
            all_videos.append((action, fname, os.path.join(data_dir, action, fname)))

train_videos = [(a, n, p) for a, n, p in all_videos if n in train_names]
test_videos  = [(a, n, p) for a, n, p in all_videos if n not in train_names]
print(f"\n=== 2.3: BoVW + SVM ===")
print(f"Train: {len(train_videos)}, Test: {len(test_videos)}")

cache_dir = os.path.join(lab2_dir, 'cache')
os.makedirs(cache_dir, exist_ok=True)

# delete old-format cache files (naming without _s{sigma}_t{tau}_n{top_n})
for f in glob.glob(os.path.join(cache_dir, '*.pkl')):
    if '_s' not in os.path.basename(f):
        os.remove(f)
        print(f"Removed old cache: {os.path.basename(f)}")

# 2.3.4: all experiments to run
# each entry: (det_name, det_func, sigma, tau, thresh, top_n, desc_type, K)
# det_name 'harris+gabor' pools interest points from both detectors
# desc_type: 'HOG+HOF' = all 144 dims, 'HOG' = first 72, 'HOF' = last 72
experiments = [
    # baseline
    ('harris',       harris_detector, 2, 1.5, 0.1,  500,  'HOG+HOF', 50),
    ('gabor',        gabor_detector,  2, 1.5, 0.1,  500,  'HOG+HOF', 50),
    # descriptor type
    ('gabor',        gabor_detector,  2, 1.5, 0.1,  500,  'HOG',     50),
    ('gabor',        gabor_detector,  2, 1.5, 0.1,  500,  'HOF',     50),
    ('harris',       harris_detector, 2, 1.5, 0.1,  500,  'HOG',     50),
    ('harris',       harris_detector, 2, 1.5, 0.1,  500,  'HOF',     50),
    # combined detector
    ('harris+gabor', None,            2, 1.5, 0.1,  500,  'HOG+HOF', 50),
    # K variation
    ('gabor',        gabor_detector,  2, 1.5, 0.1,  500,  'HOG+HOF', 100),
    ('gabor',        gabor_detector,  2, 1.5, 0.1,  500,  'HOG+HOF', 200),
    # sigma variation
    ('gabor',        gabor_detector,  3, 1.5, 0.1,  500,  'HOG+HOF', 50),
    ('gabor',        gabor_detector,  4, 1.5, 0.1,  500,  'HOG+HOF', 50),
    # tau variation
    ('gabor',        gabor_detector,  2, 1.0, 0.1,  500,  'HOG+HOF', 50),
    ('gabor',        gabor_detector,  2, 2.0, 0.1,  500,  'HOG+HOF', 50),
    # thresh variation (gabor)
    ('gabor',        gabor_detector,  2, 1.5, 0.05, 500,  'HOG+HOF', 50),
    ('gabor',        gabor_detector,  2, 1.5, 0.2,  500,  'HOG+HOF', 50),
    ('gabor',        gabor_detector,  2, 1.5, 0.3,  500,  'HOG+HOF', 50),
    # thresh variation (harris)
    ('harris',       harris_detector, 2, 1.5, 0.05, 500,  'HOG+HOF', 50),
    # top_n variation
    ('gabor',        gabor_detector,  2, 1.5, 0.1,  200,  'HOG+HOF', 50),
    ('gabor',        gabor_detector,  2, 1.5, 0.1,  1000, 'HOG+HOF', 50),
    ('gabor',        gabor_detector,  2, 1.5, 0.1,  2000, 'HOG+HOF', 50),
    # N=1000 follow-up: descriptor type and K
    ('gabor',        gabor_detector,  2, 1.5, 0.1,  1000, 'HOG',     50),
    ('gabor',        gabor_detector,  2, 1.5, 0.1,  1000, 'HOG+HOF', 100),
    ('gabor',        gabor_detector,  2, 1.5, 0.1,  1000, 'HOG+HOF', 200),
    # harris τ variation
    ('harris',       harris_detector, 2, 1.0, 0.1,  500,  'HOG+HOF', 50),
    # harris with reference-paper params (σ=1, τ=0.7)
    ('harris',       harris_detector, 1, 0.7, 0.1,  500,  'HOG+HOF', 50),
    ('harris',       harris_detector, 1, 0.7, 0.1,  500,  'HOG',     50),
]

# cache dict: key = (det_name, sigma, tau, thresh, top_n), value = loaded cache dict
loaded = {}

print(f"\n{'Detector':<15} {'σ':<4} {'τ':<5} {'thr':<6} {'N':<6} {'Desc':<10} {'K':<6} Accuracy")
print("-" * 64)

acc_rows = []

for det_name, det_func, sigma, tau, thresh, top_n, desc_type, K in experiments:

    # 2.3.2: load or compute descriptors, cache to disk
    if det_name == 'harris+gabor':
        for name, func in [('harris', harris_detector), ('gabor', gabor_detector)]:
            if (name, sigma, tau, thresh, top_n) not in loaded:
                cache_path = os.path.join(cache_dir, f'desc_{name}_s{sigma}_t{tau}_thr{thresh}_n{top_n}.pkl')
                if os.path.exists(cache_path):
                    with open(cache_path, 'rb') as f:
                        loaded[(name, sigma, tau, thresh, top_n)] = pickle.load(f)
                else:
                    print(f"  Computing {name} s={sigma} t={tau} thr={thresh} n={top_n}...")
                    desc_train, desc_test, tr_lbl, ts_lbl = [], [], [], []
                    for action, n, path in train_videos:
                        video = read_video(path, gray=True, num_frames=100)
                        pts, _ = func(video, sigma=sigma, tau=tau, thresh=thresh, top_n=top_n)
                        desc_train.append(compute_descriptors(video, pts))
                        tr_lbl.append(label_map[action])
                    for action, n, path in test_videos:
                        video = read_video(path, gray=True, num_frames=100)
                        pts, _ = func(video, sigma=sigma, tau=tau, thresh=thresh, top_n=top_n)
                        desc_test.append(compute_descriptors(video, pts))
                        ts_lbl.append(label_map[action])
                    loaded[(name, sigma, tau, thresh, top_n)] = {'desc_train': desc_train, 'desc_test': desc_test,
                                                                  'train_labels': tr_lbl, 'test_labels': ts_lbl}
                    with open(cache_path, 'wb') as f:
                        pickle.dump(loaded[(name, sigma, tau, thresh, top_n)], f)

        h = loaded[('harris', sigma, tau, thresh, top_n)]
        g = loaded[('gabor',  sigma, tau, thresh, top_n)]
        desc_train = [np.vstack([h['desc_train'][i], g['desc_train'][i]]) for i in range(len(train_videos))]
        desc_test  = [np.vstack([h['desc_test'][i],  g['desc_test'][i]])  for i in range(len(test_videos))]
        train_labels = np.array(g['train_labels'])
        test_labels  = np.array(g['test_labels'])
    else:
        if (det_name, sigma, tau, thresh, top_n) not in loaded:
            cache_path = os.path.join(cache_dir, f'desc_{det_name}_s{sigma}_t{tau}_thr{thresh}_n{top_n}.pkl')
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    loaded[(det_name, sigma, tau, thresh, top_n)] = pickle.load(f)
            else:
                print(f"  Computing {det_name} s={sigma} t={tau} thr={thresh} n={top_n}...")
                desc_train, desc_test, tr_lbl, ts_lbl = [], [], [], []
                for action, name, path in train_videos:
                    video = read_video(path, gray=True, num_frames=100)
                    pts, _ = det_func(video, sigma=sigma, tau=tau, thresh=thresh, top_n=top_n)
                    desc_train.append(compute_descriptors(video, pts))
                    tr_lbl.append(label_map[action])
                    print(f"    train {name}: {len(pts)} pts")
                for action, name, path in test_videos:
                    video = read_video(path, gray=True, num_frames=100)
                    pts, _ = det_func(video, sigma=sigma, tau=tau, thresh=thresh, top_n=top_n)
                    desc_test.append(compute_descriptors(video, pts))
                    ts_lbl.append(label_map[action])
                loaded[(det_name, sigma, tau, thresh, top_n)] = {'desc_train': desc_train, 'desc_test': desc_test,
                                                                  'train_labels': tr_lbl, 'test_labels': ts_lbl}
                with open(cache_path, 'wb') as f:
                    pickle.dump(loaded[(det_name, sigma, tau, thresh, top_n)], f)

        c = loaded[(det_name, sigma, tau, thresh, top_n)]
        desc_train   = c['desc_train']
        desc_test    = c['desc_test']
        train_labels = np.array(c['train_labels'])
        test_labels  = np.array(c['test_labels'])

    # slice descriptor dimensions based on desc_type
    if desc_type == 'HOG':
        tr = [d[:, :72] for d in desc_train]
        ts = [d[:, :72] for d in desc_test]
    elif desc_type == 'HOF':
        tr = [d[:, 72:] for d in desc_train]
        ts = [d[:, 72:] for d in desc_test]
    else:
        tr = desc_train
        ts = desc_test

    # 2.3.3: BoVW + SVM
    total_pts = sum(len(d) for d in tr)
    if total_pts == 0 or total_pts // 2 < K:
        print(f"{det_name:<15} {sigma:<4} {tau:<5} {thresh:<6} {top_n:<6} {desc_type:<10} {K:<6} skipped ({total_pts} pts)")
        continue

    bow_tr, bow_ts = bag_of_words(tr, ts, num_centers=K)
    acc, _ = svm_train_test(bow_tr, train_labels, bow_ts, test_labels)
    print(f"{det_name:<15} {sigma:<4} {tau:<5} {thresh:<6} {top_n:<6} {desc_type:<10} {K:<6} {acc:.2%}")
    acc_rows.append((det_name, sigma, tau, thresh, top_n, desc_type, K, acc))

# Generate GIFs using best-accuracy params per detector (harris and gabor)
harris_rows = [r for r in acc_rows if r[0] == 'harris']
gabor_rows = [r for r in acc_rows if r[0] == 'gabor']

best_harris = max(harris_rows, key=lambda r: r[7]) if harris_rows else None
best_gabor = max(gabor_rows, key=lambda r: r[7]) if gabor_rows else None

print("\n=== 2.1.4: GIFs (best params per detector) ===")
for best_row in [best_harris, best_gabor]:
    if best_row is None:
        continue

    det_name, sigma, tau, thresh, top_n, desc_type, K, acc = best_row
    det_func = harris_detector if det_name == 'harris' else gabor_detector
    print(
        f"  best {det_name}: sigma={sigma}, tau={tau}, thr={thresh}, "
        f"N={top_n}, desc={desc_type}, K={K}, acc={acc:.2%}"
    )

    for s_action, s_path in sample_videos.items():
        s_video = read_video(s_path, gray=True, num_frames=50)
        s_pts, _ = det_func(s_video, sigma=sigma, tau=tau, thresh=thresh, top_n=top_n)

        gif_frames = []
        for fr in range(s_video.shape[2]):
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(s_video[:, :, fr], cmap='gray')
            frame_pts = s_pts[s_pts[:, 2] == fr] if len(s_pts) > 0 else []
            for p in frame_pts:
                ax.add_patch(Circle((int(p[0]), int(p[1])), 2 * int(p[3]),
                                    edgecolor='g', fill=False, linewidth=1.5))
            ax.axis('off')
            fig.canvas.draw()
            gif_frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
            plt.close()

        pil_frames = [Image.fromarray(f) for f in gif_frames]
        gif_name = f"{s_action}_{det_name}_best.gif"
        gif_path = os.path.join(results_dir, gif_name)
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                           duration=40, loop=0)
        print(f"  Saved {gif_name}")
