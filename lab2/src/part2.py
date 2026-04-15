import sys
import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import cv2
import glob
from scipy.ndimage import convolve1d, maximum_filter
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data', 'cv26_lab2_part2_3'))
from cv26_lab2_utils import read_video, show_detection, orientation_histogram, bag_of_words, svm_train_test

data_dir = os.path.join(os.path.dirname(__file__), 'data', 'cv26_lab2_part2_3', 'KTH')
results_dir = os.path.join(os.path.dirname(__file__), 'results')
pictures_dir = os.path.join(os.path.dirname(__file__), 'docs', 'pictures')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(pictures_dir, exist_ok=True)

# clean up old outputs from this script
for old in glob.glob(os.path.join(results_dir, '*.png')) + glob.glob(os.path.join(results_dir, '*.gif')):
    os.remove(old)

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
    H, W, T = video.shape
    desc_dim = 2 * ncells * ncells * nbins
    descriptors = np.zeros((len(points), desc_dim))

    # 2.2.1: precompute gradient (Gx, Gy) for each frame
    Gx_all = np.zeros_like(video, dtype=np.float32)
    Gy_all = np.zeros_like(video, dtype=np.float32)
    for fr in range(T):
        Gx_all[:, :, fr] = cv2.Sobel(video[:, :, fr], cv2.CV_32F, 1, 0, ksize=3)
        Gy_all[:, :, fr] = cv2.Sobel(video[:, :, fr], cv2.CV_32F, 0, 1, ksize=3)

    # 2.2.1: precompute TV-L1 optical flow for consecutive frames
    # input must be uint8
    oflow = cv2.optflow.DualTVL1OpticalFlow_create(nscales=1)
    flow_all = {}  # frame t -> flow from frame t to t+1
    for fr in range(T - 1):
        flow_all[fr] = oflow.calc(video[:, :, fr], video[:, :, fr + 1], None)

    # 2.2.2: extract HOG/HOF descriptors per interest point
    for i, (x, y, t, sigma) in enumerate(points):
        x, y, t = int(x), int(y), int(t)
        half = int(2 * sigma)  # patch is 4*sigma, half-size is 2*sigma

        # clip patch to image boundaries
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


sample_videos = {
    'walking':      os.path.join(data_dir, 'walking',      'person04_walking_d1_uncomp.avi'),
    'running':      os.path.join(data_dir, 'running',      os.listdir(os.path.join(data_dir, 'running'))[0]),
    'handclapping': os.path.join(data_dir, 'handclapping', os.listdir(os.path.join(data_dir, 'handclapping'))[0]),
}

# parameter sets to experiment with
param_sets = [
    {'sigma': 2, 'tau': 1.5},
    {'sigma': 4, 'tau': 1.5},
    {'sigma': 2, 'tau': 1.0},
]

selected_frames = [10, 25, 40]

for action, path in sample_videos.items():
    print(f"\n=== {action} ===")
    video = read_video(path, gray=True, num_frames=50)

    for params in param_sets:
        sigma, tau = params['sigma'], params['tau']
        tag = f"s{sigma}_t{tau}"
        print(f"  params: sigma={sigma}, tau={tau}")

        pts_harris, H_harris = harris_detector(video, sigma=sigma, tau=tau)
        print(f"    Harris: {len(pts_harris)} points")

        pts_gabor, H_gabor = gabor_detector(video, sigma=sigma, tau=tau)
        print(f"    Gabor:  {len(pts_gabor)} points")

        # Save H maps and detections for selected frames
        for frame_idx in selected_frames:
            if frame_idx >= video.shape[2]:
                continue

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'{action} — frame {frame_idx} (σ={sigma}, τ={tau})', fontsize=14)

            axes[0, 0].imshow(video[:, :, frame_idx], cmap='gray')
            axes[0, 0].set_title('Original frame')

            axes[0, 1].imshow(H_harris[:, :, frame_idx], cmap='hot')
            axes[0, 1].set_title('Harris response H(x,y,t)')

            axes[1, 0].imshow(H_gabor[:, :, frame_idx], cmap='hot')
            axes[1, 0].set_title('Gabor response H(x,y,t)')

            # detections overlay
            axes[1, 1].imshow(video[:, :, frame_idx], cmap='gray')
            for pts, color, label in [(pts_harris, 'g', 'Harris'), (pts_gabor, 'r', 'Gabor')]:
                frame_pts = pts[pts[:, 2] == frame_idx] if len(pts) > 0 else []
                for p in frame_pts:
                    circ = Circle((int(p[0]), int(p[1])), 2 * int(p[3]),
                                  edgecolor=color, fill=False, linewidth=1.5)
                    axes[1, 1].add_patch(circ)
            axes[1, 1].set_title('Detections (green=Harris, red=Gabor)')

            for ax in axes.flat:
                ax.axis('off')
            plt.tight_layout()
            fname = f"{action}_{tag}_frame{frame_idx}.png"
            plt.savefig(os.path.join(results_dir, fname), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved {fname}")

        # Save detection frames as GIF for the default params
        if params == param_sets[0]:
            from PIL import Image
            for detector_name, pts in [('harris', pts_harris), ('gabor', pts_gabor)]:
                gif_frames = []
                for fr in range(video.shape[2]):
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.imshow(video[:, :, fr], cmap='gray')
                    ax.set_title(f'{action} — {detector_name} (σ={sigma}, τ={tau}) — frame {fr}')
                    frame_pts = pts[pts[:, 2] == fr] if len(pts) > 0 else []
                    for p in frame_pts:
                        circ = Circle((int(p[0]), int(p[1])), 2 * int(p[3]),
                                      edgecolor='g', fill=False, linewidth=1.5)
                        ax.add_patch(circ)
                    ax.axis('off')
                    fig.canvas.draw()
                    buf = fig.canvas.buffer_rgba()
                    img = np.asarray(buf)[:, :, :3].copy()
                    gif_frames.append(img)
                    plt.close()
                pil_frames = [Image.fromarray(f) for f in gif_frames]
                gif_path = os.path.join(results_dir, f"{action}_{detector_name}.gif")
                pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                                   duration=40, loop=0)
                print(f"    Saved {action}_{detector_name}.gif")

# ─── 2.3: Bag of Visual Words + SVM classification ──────────────────────────
import pickle

actions = ['running', 'handclapping', 'walking']
label_map = {a: i for i, a in enumerate(actions)}

# 2.3.1: train/test split based on provided file
train_file = os.path.join(data_dir, 'traininng_videos.txt')
with open(train_file) as f:
    train_names = set(line.strip() for line in f if line.strip())

# collect all videos with labels, split into train/test
all_videos = []
for action in actions:
    action_dir = os.path.join(data_dir, action)
    for fname in sorted(os.listdir(action_dir)):
        if not fname.endswith('.avi'):
            continue
        all_videos.append((action, fname, os.path.join(action_dir, fname)))

train_videos = [(a, n, p) for a, n, p in all_videos if n in train_names]
test_videos  = [(a, n, p) for a, n, p in all_videos if n not in train_names]
print(f"\n=== 2.3: BoVW + SVM ===")
print(f"Train: {len(train_videos)}, Test: {len(test_videos)}")

# 2.3.2 & 2.3.3: run pipeline for each detector
cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(cache_dir, exist_ok=True)

for det_name, det_func in [('harris', harris_detector), ('gabor', gabor_detector)]:
    print(f"\n--- Detector: {det_name} ---")

    # extract descriptors (or load from cache)
    cache_path = os.path.join(cache_dir, f'desc_{det_name}.pkl')
    if os.path.exists(cache_path):
        print(f"  Loading cached descriptors from {cache_path}")
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        desc_train = cached['desc_train']
        desc_test = cached['desc_test']
        train_labels = cached['train_labels']
        test_labels = cached['test_labels']
    else:
        desc_train = []
        train_labels = []
        for action, name, path in train_videos:
            video = read_video(path, gray=True, num_frames=100)
            pts, _ = det_func(video, sigma=2, tau=1.5)
            desc = compute_descriptors(video, pts)
            desc_train.append(desc)
            train_labels.append(label_map[action])
            print(f"  train {name}: {len(pts)} pts, desc {desc.shape}")

        desc_test = []
        test_labels = []
        for action, name, path in test_videos:
            video = read_video(path, gray=True, num_frames=100)
            pts, _ = det_func(video, sigma=2, tau=1.5)
            desc = compute_descriptors(video, pts)
            desc_test.append(desc)
            test_labels.append(label_map[action])
            print(f"  test  {name}: {len(pts)} pts, desc {desc.shape}")

        with open(cache_path, 'wb') as f:
            pickle.dump({
                'desc_train': desc_train, 'desc_test': desc_test,
                'train_labels': train_labels, 'test_labels': test_labels
            }, f)
        print(f"  Saved cache to {cache_path}")

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # 2.3.2: Bag of Visual Words
    bow_train, bow_test = bag_of_words(desc_train, desc_test, num_centers=50)
    print(f"  BoW: train={bow_train.shape}, test={bow_test.shape}")

    # 2.3.3: SVM classification
    accuracy, predictions = svm_train_test(bow_train, train_labels, bow_test, test_labels)
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Predictions: {predictions}")
    print(f"  True labels: {test_labels}")

# ─── 2.3.4: Experiments ──────────────────────────────────────────────────────
print("\n=== 2.3.4: Experiments ===")

def load_cache(name):
    with open(os.path.join(cache_dir, f'desc_{name}.pkl'), 'rb') as f:
        return pickle.load(f)

h = load_cache('harris')
g = load_cache('gabor')

combined_train = [np.vstack([h['desc_train'][i], g['desc_train'][i]]) for i in range(len(h['desc_train']))]
combined_test  = [np.vstack([h['desc_test'][i],  g['desc_test'][i]])  for i in range(len(h['desc_test']))]

train_labels = np.array(g['train_labels'])
test_labels  = np.array(g['test_labels'])

# (detector name, train descriptors, test descriptors)
experiments = [
    ('harris',       h['desc_train'], h['desc_test']),
    ('gabor',        g['desc_train'], g['desc_test']),
    ('harris+gabor', combined_train,  combined_test),
]

# HOG = first 72 dims, HOF = last 72 (ncells*ncells*nbins = 3*3*8 = 72)
desc_slices = [('HOG+HOF', slice(None)), ('HOG', slice(0, 72)), ('HOF', slice(72, None))]

print(f"\n{'Detector':<15} {'Desc':<10} {'K':<6} Accuracy")
print("-" * 42)
for det, d_tr, d_ts in experiments:
    for desc_name, slc in desc_slices:
        for K in (50, 100, 200):
            tr = [d[slc] for d in d_tr]
            ts = [d[slc] for d in d_ts]
            total_pts = sum(len(d) for d in tr)
            if any(len(d) == 0 for d in tr) or any(len(d) == 0 for d in ts) or total_pts < K:
                print(f"{det:<15} {desc_name:<10} {K:<6} skipped (only {total_pts} pts < K)")
                continue
            bow_tr, bow_ts = bag_of_words(tr, ts, num_centers=K)
            acc, _ = svm_train_test(bow_tr, train_labels, bow_ts, test_labels)
            print(f"{det:<15} {desc_name:<10} {K:<6} {acc:.2%}")

# σ variation: Gabor, K=50, HOG+HOF
print("\n--- σ variation (Gabor, K=50) ---")
for sigma_val in (2, 3, 4):
    cache_name = 'gabor' if sigma_val == 2 else f'gabor_s{sigma_val}'
    cache_path_s = os.path.join(cache_dir, f'desc_{cache_name}.pkl')
    if os.path.exists(cache_path_s):
        c = load_cache(cache_name)
        d_tr_s, d_ts_s = c['desc_train'], c['desc_test']
    else:
        d_tr_s, tr_lbl_s = [], []
        for action, name, path in train_videos:
            video = read_video(path, gray=True, num_frames=100)
            pts, _ = gabor_detector(video, sigma=sigma_val, tau=1.5)
            d_tr_s.append(compute_descriptors(video, pts))
            tr_lbl_s.append(label_map[action])
        d_ts_s, ts_lbl_s = [], []
        for action, name, path in test_videos:
            video = read_video(path, gray=True, num_frames=100)
            pts, _ = gabor_detector(video, sigma=sigma_val, tau=1.5)
            d_ts_s.append(compute_descriptors(video, pts))
            ts_lbl_s.append(label_map[action])
        with open(cache_path_s, 'wb') as f:
            pickle.dump({'desc_train': d_tr_s, 'desc_test': d_ts_s,
                         'train_labels': tr_lbl_s, 'test_labels': ts_lbl_s}, f)

    bow_tr, bow_ts = bag_of_words(d_tr_s, d_ts_s, num_centers=50)
    acc, _ = svm_train_test(bow_tr, train_labels, bow_ts, test_labels)
    print(f"  σ={sigma_val}: {acc:.2%}")
