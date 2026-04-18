import sys
import os
import shutil
import tempfile
import matplotlib
matplotlib.use('Agg')
import numpy as np
import cv2
import glob
import pickle
from PIL import Image
from scipy.ndimage import convolve1d, maximum_filter
from matplotlib import pyplot as plt

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

def harris_detector(video, sigma, tau, k, s, thresh, top_n):
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

    # NMS: spatial window scales with sigma, temporal with tau
    # threshold: per-frame relative + global floor (suppresses noise in static/blank frames)
    nms_xy = 2 * int(np.ceil(sigma)) + 1
    nms_t  = 2 * int(np.ceil(tau))   + 1
    frame_max = H.max(axis=(0, 1), keepdims=True)
    global_floor = 0.005 * H.max()  # suppresses noise in static/blank frames
    mask = (H == maximum_filter(H, size=(nms_xy, nms_xy, nms_t))) & \
           (H > thresh * frame_max) & (H > global_floor)
    ys, xs, ts = np.where(mask)
    if len(ys) == 0:
        return np.empty((0, 4)), H
    order = np.argsort(H[ys, xs, ts])[::-1][:top_n]
    points = np.column_stack([xs[order], ys[order], ts[order],
                               np.full(len(order), sigma)])
    return points, H

def gabor_detector(video, sigma, tau, thresh, top_n):
    """Spatiotemporal Gabor interest point detector."""
    video = video.astype(np.float64) / 255.0

    # gabor filters on the window of [-2*τ, 2*τ] with frequency 4/τ
    omega = 4.0 / tau
    t = np.arange(-int(2 * tau), int(2 * tau) + 1, dtype=np.float64) # temporal window
    gaussian = np.exp(-t ** 2 / (2 * tau ** 2))
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

    # NMS: spatial window scales with sigma, temporal with tau
    # per-frame relative threshold + global floor (suppresses background-noise frames)
    nms_xy = 2 * int(np.ceil(sigma)) + 1
    nms_t  = 2 * int(np.ceil(tau))   + 1
    frame_max = H.max(axis=(0, 1), keepdims=True)
    global_floor = 0.05 * H.max()
    mask = (H == maximum_filter(H, size=(nms_xy, nms_xy, nms_t))) & \
           (H > thresh * frame_max) & (H > global_floor)
    ys, xs, ts = np.where(mask)
    if len(ys) == 0:
        return np.empty((0, 4)), H
    order = np.argsort(H[ys, xs, ts])[::-1][:top_n]
    points = np.column_stack([xs[order], ys[order], ts[order],
                               np.full(len(order), sigma)])
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
    descriptors = np.zeros((len(points), 2 * ncells * ncells * nbins))

    # 2.2.1: precompute gradient (Gx, Gy) for each frame via central differences [-1, 0, 1] (for HOG)
    d = np.array([-1.0, 0.0, 1.0]) # central difference kernel
    video_f = video.astype(np.float32) # convert to float for convolution
    Gx_all = convolve1d(video_f, d, axis=1) # gradient along x-axis
    Gy_all = convolve1d(video_f, d, axis=0) # gradient along y-axis

    # 2.2.1: precompute TV-L1 optical flow (input must be uint8)
    # flow_all[fr] is an H x W x 2 array: [:,:,0] = fx, [:,:,1] = fy between frame fr and fr+1
    oflow = cv2.optflow.DualTVL1OpticalFlow_create(nscales=1) # TV-L1 optical flow object
    flow_all = {fr: oflow.calc(video[:, :, fr], video[:, :, fr + 1], None) for fr in range(T - 1)}

    # 2.2.2: extract HOG/HOF descriptors per interest point
    for i, (x, y, t, sigma) in enumerate(points): # for each interest point:
        x, y, t = int(x), int(y), int(t) # they were originally floats -> convert to integers
        half = int(2 * sigma) # the patch is +/- 2*sigma around the point

        # make sure we dont go out of bounds
        y0, y1 = max(0, y - half), min(H, y + half)
        x0, x1 = max(0, x - half), min(W, x + half)

        # skip degenerate patches
        if y1 - y0 < 2 or x1 - x0 < 2:
            continue

        # HOG: gradient patch
        gx_patch = Gx_all[y0:y1, x0:x1, t].astype(np.float32)
        gy_patch = Gy_all[y0:y1, x0:x1, t].astype(np.float32)
        hog = orientation_histogram(gx_patch, gy_patch, nbins, np.array([ncells, ncells]))

        # HOF: optical flow patch
        if t < T - 1:
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
selected_frames = [12, 27, 42]

# ─── 2.1.4: visualize H map + detections ─────────────────────────────────────

viz_configs = [
    ('harris', harris_detector, dict(sigma=2, tau=1.5, k=0.00001, s=1, thresh=0.01, top_n=500)),
    ('gabor',  gabor_detector,  dict(sigma=2, tau=1.5,             thresh=0.1,       top_n=500)),
]
def _save_gif(video, pts, gif_path, title=None):
    """Use show_detection to render all frames, then stitch into a GIF."""
    from PIL import ImageDraw, ImageFont
    tmpdir = tempfile.mkdtemp()
    show_detection(video, pts, save_path=tmpdir)
    frames = []
    for i in range(video.shape[2]):
        img = Image.open(os.path.join(tmpdir, f'frame{i}.png')).convert('RGB')
        if title:
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 14)
            except Exception:
                font = ImageFont.load_default()
            draw.rectangle([0, 0, img.width, 18], fill=(0, 0, 0))
            draw.text((4, 2), title, fill=(255, 255, 255), font=font)
        frames.append(img)
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=40, loop=0)
    shutil.rmtree(tmpdir)

def _save_png(src_path, fname):
    """Copy/move a PNG to both results_dir and docs/pictures/."""
    dst_results = os.path.join(results_dir, fname)
    dst_pictures = os.path.join(pictures_dir, fname)
    shutil.copy(src_path, dst_results)
    shutil.copy(src_path, dst_pictures)

def _savefig_png(fname):
    """Save current matplotlib figure to both results_dir and docs/pictures/."""
    dst_results = os.path.join(results_dir, fname)
    dst_pictures = os.path.join(pictures_dir, fname)
    plt.savefig(dst_results, dpi=150, bbox_inches='tight')
    shutil.copy(dst_results, dst_pictures)

print("\n=== 2.1.4: Visualization ===")
for det_name, det_func, det_kw in viz_configs:
    tag = f"{det_name}_s{det_kw['sigma']}_t{det_kw['tau']}_thr{det_kw['thresh']}"
    for s_action, s_path in sample_videos.items():
        s_video = read_video(s_path, gray=True, num_frames=50)
        s_pts, s_H = det_func(s_video, **det_kw)
        print(f"  {det_name} {s_action}: {len(s_pts)} pts")

        # save detection frames via show_detection, then pick selected ones
        tmpdir = tempfile.mkdtemp()
        show_detection(s_video, s_pts, save_path=tmpdir)
        for frame_idx in selected_frames:
            if frame_idx >= s_video.shape[2]:
                continue
            # H map
            if s_H is not None:
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.imshow(s_H[:, :, frame_idx], cmap='hot')
                ax.axis('off')
                plt.tight_layout(pad=0)
                _savefig_png(f"{s_action}_{tag}_frame{frame_idx}_H.png")
                plt.close()
            # detection frame
            _save_png(os.path.join(tmpdir, f'frame{frame_idx}.png'),
                      f"{s_action}_{tag}_frame{frame_idx}_det.png")
        shutil.rmtree(tmpdir)

# ─── GIFs ────────────────────────────────────────────────────────────────────
print("\n=== 2.1.4: GIFs ===")
for det_name, det_func, det_kw in viz_configs:
    for s_action, s_path in sample_videos.items():
        s_video = read_video(s_path, gray=True, num_frames=50)
        s_pts, _ = det_func(s_video, **det_kw)
        gif_name = f"{s_action}_{det_name}_s{det_kw['sigma']}_t{det_kw['tau']}.gif"
        title = f"{det_name} σ={det_kw['sigma']} τ={det_kw['tau']} thr={det_kw['thresh']}"
        _save_gif(s_video, s_pts, os.path.join(results_dir, gif_name), title=title)
        print(f"  Saved {gif_name}")

# ─── 2.3: Bag of Visual Words + SVM classification ───────────────────────────

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
    if not os.path.basename(f).startswith('desc_'):
        os.remove(f)
        print(f"Removed old cache: {os.path.basename(f)}")

# 2.3.4: all experiments to run
# each entry: (det_name, det_func, det_kw, desc_type, K)
# det_name 'harris+gabor' pools interest points from both detectors
# desc_type: 'HOG+HOF' = all 144 dims, 'HOG' = first 72, 'HOF' = last 72
experiments = [
    # baseline (single-scale)
    ('harris',       harris_detector,       dict(sigma=2, tau=1.5, k=0.0001, s=1, thresh=0.1,  top_n=500),            'HOG+HOF', 50),
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,            thresh=0.1,  top_n=500),                 'HOG+HOF', 50),
    # descriptor type
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,            thresh=0.1,  top_n=500),                 'HOG',     50),
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,            thresh=0.1,  top_n=500),                 'HOF',     50),
    ('harris',       harris_detector,       dict(sigma=2, tau=1.5, k=0.0001, s=1, thresh=0.1,  top_n=500),            'HOG',     50),
    ('harris',       harris_detector,       dict(sigma=2, tau=1.5, k=0.0001, s=1, thresh=0.1,  top_n=500),            'HOF',     50),
    # combined detector
    ('harris+gabor', None,                  dict(sigma=2, tau=1.5,            thresh=0.1,  top_n=500),                 'HOG+HOF', 50),
    # K variation
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,            thresh=0.1,  top_n=500),                 'HOG+HOF', 100),
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,            thresh=0.1,  top_n=500),                 'HOG+HOF', 200),
    # sigma variation (gabor)
    ('gabor',        gabor_detector,        dict(sigma=3, tau=1.5,            thresh=0.1,  top_n=500),                 'HOG+HOF', 50),
    ('gabor',        gabor_detector,        dict(sigma=4, tau=1.5,            thresh=0.1,  top_n=500),                 'HOG+HOF', 50),
    # Harris lab indicative params (σ=4, s=2, τ=1.5, k=0.005)
    ('harris',       harris_detector,       dict(sigma=4, tau=1.5, k=0.005,  s=2, thresh=0.1,  top_n=500),            'HOG+HOF', 50),
    # tau variation
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.0,            thresh=0.1,  top_n=500),                 'HOG+HOF', 50),
    ('gabor',        gabor_detector,        dict(sigma=2, tau=2.0,            thresh=0.1,  top_n=500),                 'HOG+HOF', 50),
    # thresh variation (gabor)
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,            thresh=0.05, top_n=500),                 'HOG+HOF', 50),
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,            thresh=0.2,  top_n=500),                 'HOG+HOF', 50),
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,            thresh=0.3,  top_n=500),                 'HOG+HOF', 50),
    # thresh variation (harris)
    ('harris',       harris_detector,       dict(sigma=2, tau=1.5, k=0.0001, s=1, thresh=0.05, top_n=500),            'HOG+HOF', 50),
    # top_n variation
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,            thresh=0.1,  top_n=200),                 'HOG+HOF', 50),
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,            thresh=0.1,  top_n=1000),                'HOG+HOF', 50),
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,            thresh=0.1,  top_n=2000),                'HOG+HOF', 50),
    # N=1000 follow-up: descriptor type and K
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,            thresh=0.1,  top_n=1000),                'HOG',     50),
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,            thresh=0.1,  top_n=1000),                'HOG+HOF', 100),
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,            thresh=0.1,  top_n=1000),                'HOG+HOF', 200),
    # harris τ variation
    ('harris',       harris_detector,       dict(sigma=2, tau=1.0, k=0.0001, s=1, thresh=0.1,  top_n=500),            'HOG+HOF', 50),
    # harris with reference params (σ=1, τ=0.7)
    ('harris',       harris_detector,       dict(sigma=1, tau=0.7, k=0.0001, s=1, thresh=0.1,  top_n=500),            'HOG+HOF', 50),
    ('harris',       harris_detector,       dict(sigma=1, tau=0.7, k=0.0001, s=1, thresh=0.1,  top_n=500),            'HOG',     50),
]

# cache dict: key = (det_name, sigma, tau, thresh, top_n)
loaded = {}

print(f"\n{'Detector':<15} {'σ':<4} {'τ':<5} {'thr':<6} {'Npts':<6} {'Desc':<10} {'K':<6} Accuracy")
print("-" * 65)

acc_rows = []

for det_name, det_func, det_kw, desc_type, K in experiments:
    sigma  = det_kw['sigma']
    tau    = det_kw['tau']
    thresh = det_kw['thresh']
    top_n  = det_kw['top_n']

    def _cache_key(name, sg, ta, thr, tn):
        return (name, sg, ta, thr, tn)

    def _cache_path(name, sg, ta, thr, tn):
        return os.path.join(cache_dir,
            f'desc_{name}_s{sg}_t{ta}_thr{thr}_n{tn}.pkl')

    def _compute_and_cache(name, func, kw):
        key = _cache_key(name, kw['sigma'], kw['tau'], kw['thresh'], kw['top_n'])
        if key in loaded:
            return
        path = _cache_path(*key)
        if os.path.exists(path):
            with open(path, 'rb') as fh:
                loaded[key] = pickle.load(fh)
            return
        print(f"  Computing {name} {kw}...")
        dtr, dts, trl, tsl = [], [], [], []
        for action, n, vpath in train_videos:
            vid = read_video(vpath, gray=True, num_frames=100)
            pts, _ = func(vid, **kw)
            dtr.append(compute_descriptors(vid, pts))
            trl.append(label_map[action])
            print(f"    train {n}: {len(pts)} pts")
        for action, n, vpath in test_videos:
            vid = read_video(vpath, gray=True, num_frames=100)
            pts, _ = func(vid, **kw)
            dts.append(compute_descriptors(vid, pts))
            tsl.append(label_map[action])
        loaded[key] = {'desc_train': dtr, 'desc_test': dts,
                       'train_labels': trl, 'test_labels': tsl}
        with open(path, 'wb') as fh:
            pickle.dump(loaded[key], fh)

    # 2.3.2: load or compute descriptors, cache to disk
    if det_name == 'harris+gabor':
        _compute_and_cache('harris', harris_detector,
                           dict(sigma=sigma, tau=tau, k=0.0001, s=1, thresh=thresh, top_n=top_n))
        _compute_and_cache('gabor',  gabor_detector,
                           dict(sigma=sigma, tau=tau, thresh=thresh, top_n=top_n))
        hk = _cache_key('harris', sigma, tau, thresh, top_n)
        gk = _cache_key('gabor',  sigma, tau, thresh, top_n)
        h, g = loaded[hk], loaded[gk]
        desc_train = [np.vstack([h['desc_train'][i], g['desc_train'][i]]) for i in range(len(train_videos))]
        desc_test  = [np.vstack([h['desc_test'][i],  g['desc_test'][i]])  for i in range(len(test_videos))]
        train_labels = np.array(g['train_labels'])
        test_labels  = np.array(g['test_labels'])
    else:
        _compute_and_cache(det_name, det_func, det_kw)
        c = loaded[_cache_key(det_name, sigma, tau, thresh, top_n)]
        desc_train   = c['desc_train']
        desc_test    = c['desc_test']
        train_labels = np.array(c['train_labels'])
        test_labels  = np.array(c['test_labels'])

    # slice descriptor dimensions based on desc_type
    if desc_type == 'HOG': # first 72 dims are HOG, last 72 are HOF
        tr = [d[:, :72] for d in desc_train]
        ts = [d[:, :72] for d in desc_test]
    elif desc_type == 'HOF':
        tr = [d[:, 72:] for d in desc_train]
        ts = [d[:, 72:] for d in desc_test]
    else: # take all 144 dims
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
    acc_rows.append((det_name, det_kw, desc_type, K, acc))

# Generate GIFs using best-accuracy params per detector family
_det_func_map = {
    'harris': harris_detector,
    'gabor':  gabor_detector,
}
harris_rows = [r for r in acc_rows if r[0] == 'harris']
gabor_rows  = [r for r in acc_rows if r[0] == 'gabor']

best_harris = max(harris_rows, key=lambda r: r[4]) if harris_rows else None
best_gabor  = max(gabor_rows,  key=lambda r: r[4]) if gabor_rows  else None

print("\n=== 2.1.4: GIFs (best params per detector) ===")
for best_row in [best_harris, best_gabor]:
    if best_row is None:
        continue

    det_name, det_kw, desc_type, K, acc = best_row
    det_func = _det_func_map[det_name]
    print(f"  best {det_name}: {det_kw}, desc={desc_type}, K={K}, acc={acc:.2%}")

    for s_action, s_path in sample_videos.items():
        s_video = read_video(s_path, gray=True, num_frames=50)
        s_pts, _ = det_func(s_video, **det_kw)
        gif_name = f"{s_action}_{det_name}_best.gif"
        title = f"{det_name} σ={det_kw['sigma']} τ={det_kw['tau']} thr={det_kw['thresh']} acc={acc:.0%}"
        _save_gif(s_video, s_pts, os.path.join(results_dir, gif_name), title=title)
        print(f"  Saved {gif_name}")
