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
from scipy.ndimage import convolve1d
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

#clean up old outputs from this script
for old in glob.glob(os.path.join(results_dir, '*.png')) + glob.glob(os.path.join(results_dir, '*.gif')):
    os.remove(old)

#2.1: interest point detectors (Harris3D, Gabor) 

def gaussian_smooth_3d(volume, sigma_s, sigma_t):
    n_s = int(np.ceil(3 * sigma_s)) * 2 + 1 #kernel size for spatial smoothing
    n_t = int(np.ceil(3 * sigma_t)) * 2 + 1 #the same just for temporal smoothing
    k_s = cv2.getGaussianKernel(n_s, sigma_s).flatten() #1D Gaussian kernel for spatial smoothing
    k_t = cv2.getGaussianKernel(n_t, sigma_t).flatten() #1D Gaussian kernel for temporal smoothing
    out = convolve1d(volume, k_s, axis=0) #smooth along y-axis
    out = convolve1d(out, k_s, axis=1) #now along x-axis
    out = convolve1d(out, k_t, axis=2) #finally along temporal axis
    return out

def harris_detector(video, sigma, tau, k, s, thresh, top_n):

    video = video.astype(np.float64) / 255.0 #convert to float in [0,1] for processing

    L = gaussian_smooth_3d(video, sigma, tau) #smooth with (σ,τ) to get the smoothed video L

    #spatiotemporal derivatives via central differences [-1, 0, 1]
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

    #smooth each product of the structure tensor
    Lx2  = gaussian_smooth_3d(Lx2,  s * sigma, s * tau)
    Ly2  = gaussian_smooth_3d(Ly2,  s * sigma, s * tau)
    Lt2  = gaussian_smooth_3d(Lt2,  s * sigma, s * tau)
    LxLy = gaussian_smooth_3d(LxLy, s * sigma, s * tau)
    LxLt = gaussian_smooth_3d(LxLt, s * sigma, s * tau)
    LyLt = gaussian_smooth_3d(LyLt, s * sigma, s * tau)

    # we now implement the Harris criterion: H = det(M) - k * trace(M)^3
    trace_M = Lx2 + Ly2 + Lt2
    det_M = (Lx2 * (Ly2 * Lt2 - LyLt ** 2)
             - LxLy * (LxLy * Lt2 - LyLt * LxLt)
             + LxLt * (LxLy * LyLt - Ly2 * LxLt))
    H = det_M - k * (trace_M ** 3)

    H[H <= thresh * H.max()] = 0 # thresholding by global max
    top_indices = np.argsort(H.ravel())[-top_n:][::-1] #get top N indices in descending order
    ys, xs, ts = np.unravel_index(top_indices, H.shape) #convert flat indices to 3D coordinates
    points = np.column_stack([xs, ys, ts, np.full(len(xs), sigma)]) #stack into Nx4 array [x,y,t,sigma]
    return points, H

def gabor_detector(video, sigma, tau, thresh, top_n):

    video = video.astype(np.float64) / 255.0

    #gabor filters on the window of [-2*τ, 2*τ] with frequency 4/τ
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

    # 
    H[H <= thresh * H.max()] = 0 
    top_indices = np.argsort(H.ravel())[-top_n:][::-1]
    ys, xs, ts = np.unravel_index(top_indices, H.shape)
    points = np.column_stack([xs, ys, ts, np.full(len(xs), sigma)])
    return points, H

#  2.2: HOG/HOF descriptors ──

def compute_descriptors(video, points, nbins=8, ncells=3):

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

#  2.1.4: visualize H map + detections 

viz_configs = [
    ('harris', harris_detector, dict(sigma=4, tau=1.5, k=0.005, s=2, thresh=0.05, top_n=500)),
    ('gabor',  gabor_detector,  dict(sigma=4, tau=1.5,             thresh=0.05,       top_n=500)),
    ('harris', harris_detector, dict(sigma=2, tau=1.5, k=0.005, s=1.5, thresh=0.05, top_n=500)),
    ('gabor',  gabor_detector,  dict(sigma=2, tau=1.5,             thresh=0.05,       top_n=500)),
]
def _save_gif(video, pts, gif_path): # helper function to save a GIF of the detections on the video
    from PIL import ImageDraw, ImageFont
    tmpdir = tempfile.mkdtemp()
    show_detection(video, pts, save_path=tmpdir) 
    frames = [Image.open(os.path.join(tmpdir, f'frame{i}.png')).convert('RGB')
              for i in range(video.shape[2])]
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=40, loop=0)
    shutil.rmtree(tmpdir) 

def _save_png(src, fname): # helper func to save png file in both results and pictures dirs
    for d in [results_dir, pictures_dir]:
        shutil.copy(src, os.path.join(d, fname))

def _savefig_png(fname): # helper func to save matplotlib fig as PNG in res + pictures dirs
    path = os.path.join(results_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    shutil.copy(path, os.path.join(pictures_dir, fname))

print("\n=== 2.1.4: Visualization + GIFs ===")
for det_name, det_func, det_kw in viz_configs:
    tag = f"{det_name}_s{det_kw['sigma']}_t{det_kw['tau']}_thr{det_kw['thresh']}"
    for s_action, s_path in sample_videos.items():
        s_video = read_video(s_path, gray=True, num_frames=50)
        s_pts, s_H = det_func(s_video, **det_kw)
        print(f"  {det_name} {s_action}: {len(s_pts)} pts")

        # pick 3 frames with most detections
        t_counts = np.bincount(s_pts[:, 2].astype(int), minlength=s_video.shape[2])
        chosen = sorted(np.argsort(t_counts)[-3:].tolist())

        tmpdir = tempfile.mkdtemp()
        show_detection(s_video, s_pts, save_path=tmpdir)
        for f in chosen:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(s_H[:, :, f], cmap='hot')
            ax.axis('off')
            plt.tight_layout(pad=0)
            _savefig_png(f"{s_action}_{tag}_frame{f}_H.png")
            plt.close()
            _save_png(os.path.join(tmpdir, f'frame{f}.png'), f"{s_action}_{tag}_frame{f}_det.png")

        gif_path = os.path.join(results_dir, f"{s_action}_{det_name}_s{det_kw['sigma']}_t{det_kw['tau']}.gif")
        _save_gif(s_video, s_pts, gif_path)
        shutil.rmtree(tmpdir)
        print(f"  Saved GIF: {os.path.basename(gif_path)}")

#2.3: Bag of Visual Words + SVM classification 

actions = ['running', 'handclapping', 'walking']
label_map = {a: i for i, a in enumerate(actions)}

#2.3.1: train/test split based on provided file
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

# 2.3.4: all experiments to run
# each entry: (det_name, det_func, det_kw, desc_type, K)
# desc_type: 'HOG+HOF' = all 144 dims, 'HOG' = first 72, 'HOF' = last 72
experiments = [
    # A. which detector x descriptor combo wins? (baseline params) 
    ('harris',       harris_detector,       dict(sigma=2, tau=1.5, k=0.005, s=1.5, thresh=0.02, top_n=500),  'HOG+HOF', 50),
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,                  thresh=0.08, top_n=500),  'HOG+HOF', 50),
    ('harris',       harris_detector,       dict(sigma=2, tau=1.5, k=0.005, s=1.5, thresh=0.02, top_n=500),  'HOG',     50),
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,                  thresh=0.08, top_n=500),  'HOG',     50),
    ('harris',       harris_detector,       dict(sigma=2, tau=1.5, k=0.005, s=1.5, thresh=0.02, top_n=500),  'HOF',     50),
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,                  thresh=0.08, top_n=500),  'HOF',     50),

    # B. only K varies 
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,                  thresh=0.08, top_n=500),  'HOG+HOF', 20),
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,                  thresh=0.08, top_n=500),  'HOG+HOF', 100),
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,                  thresh=0.08, top_n=500),  'HOG+HOF', 200),
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,                  thresh=0.08, top_n=500),  'HOG+HOF', 500),
    ('harris',       harris_detector,       dict(sigma=2, tau=1.5, k=0.005, s=1.5, thresh=0.02, top_n=500),  'HOG+HOF', 100),
    ('harris',       harris_detector,       dict(sigma=2, tau=1.5, k=0.005, s=1.5, thresh=0.02, top_n=500),  'HOG+HOF', 200),

    # C. only σ varies 
    ('gabor',        gabor_detector,        dict(sigma=1, tau=1.5,                  thresh=0.08, top_n=500),  'HOG+HOF', 50),
    ('gabor',        gabor_detector,        dict(sigma=3, tau=1.5,                  thresh=0.08, top_n=500),  'HOG+HOF', 50),
    ('gabor',        gabor_detector,        dict(sigma=4, tau=1.5,                  thresh=0.08, top_n=500),  'HOG+HOF', 50),
    ('harris',       harris_detector,       dict(sigma=1, tau=1.5, k=0.005, s=1.5, thresh=0.02, top_n=500),  'HOG+HOF', 50),
    ('harris',       harris_detector,       dict(sigma=3, tau=1.5, k=0.005, s=1.5, thresh=0.02, top_n=500),  'HOG+HOF', 50),
    ('harris',       harris_detector,       dict(sigma=4, tau=1.5, k=0.005, s=2,   thresh=0.02, top_n=500),  'HOG+HOF', 50),

    # D. only τ varies 
    ('gabor',        gabor_detector,        dict(sigma=2, tau=0.5,                  thresh=0.08, top_n=500),  'HOG+HOF', 50),
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.0,                  thresh=0.08, top_n=500),  'HOG+HOF', 50),
    ('gabor',        gabor_detector,        dict(sigma=2, tau=2.0,                  thresh=0.08, top_n=500),  'HOG+HOF', 50),
    ('harris',       harris_detector,       dict(sigma=2, tau=1.0, k=0.005, s=1.5, thresh=0.02, top_n=500),  'HOG+HOF', 50),
    ('harris',       harris_detector,       dict(sigma=2, tau=2.0, k=0.005, s=1.5, thresh=0.02, top_n=500),  'HOG+HOF', 50),

    # E. only threshhold varies
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,                  thresh=0.02, top_n=500),  'HOG+HOF', 50),
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,                  thresh=0.05, top_n=500),  'HOG+HOF', 50),
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,                  thresh=0.15, top_n=500),  'HOG+HOF', 50),
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,                  thresh=0.30, top_n=500),  'HOG+HOF', 50),
    ('harris',       harris_detector,       dict(sigma=2, tau=1.5, k=0.005, s=1.5, thresh=0.005,top_n=500),  'HOG+HOF', 50),
    ('harris',       harris_detector,       dict(sigma=2, tau=1.5, k=0.005, s=1.5, thresh=0.01, top_n=500),  'HOG+HOF', 50),
    ('harris',       harris_detector,       dict(sigma=2, tau=1.5, k=0.005, s=1.5, thresh=0.05, top_n=500),  'HOG+HOF', 50),
    ('harris',       harris_detector,       dict(sigma=2, tau=1.5, k=0.005, s=1.5, thresh=0.10, top_n=500),  'HOG+HOF', 50),

    # F. only top_n varies
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,                  thresh=0.08, top_n=100),  'HOG+HOF', 50),
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,                  thresh=0.08, top_n=200),  'HOG+HOF', 50),
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,                  thresh=0.08, top_n=1000), 'HOG+HOF', 50),
    ('gabor',        gabor_detector,        dict(sigma=2, tau=1.5,                  thresh=0.08, top_n=2000), 'HOG+HOF', 50),

    # G. only k varies (Harris only)
    ('harris', harris_detector, dict(sigma=2, tau=1.5, k=0.001, s=1.5, thresh=0.02, top_n=200), 'HOG+HOF', 50),
    ('harris', harris_detector, dict(sigma=2, tau=1.5, k=0.005, s=1.5, thresh=0.02, top_n=200), 'HOG+HOF', 50),
    ('harris', harris_detector, dict(sigma=2, tau=1.5, k=0.01,  s=1.5, thresh=0.02, top_n=200), 'HOG+HOF', 50),
    ('harris', harris_detector, dict(sigma=2, tau=1.5, k=0.02,  s=1.5, thresh=0.02, top_n=200), 'HOG+HOF', 50)
]

loaded = {}  # cache: key -> descriptors dict

def _get_descriptors(name, func, kw): # compute decriptors for all videos with caching
    key = (name, kw['sigma'], kw['tau'], kw['thresh'], kw['top_n'])
    if key in loaded:
        return loaded[key]
    path = os.path.join(cache_dir, f"desc_{name}_s{kw['sigma']}_t{kw['tau']}_thr{kw['thresh']}_n{kw['top_n']}.pkl") 
    if os.path.exists(path):
        with open(path, 'rb') as fh:
            loaded[key] = pickle.load(fh)
        return loaded[key]
    print(f"  Computing {name} descriptors...")
    desc_train, train_labels, desc_test, test_labels = [], [], [], []
    for action, n, vpath in train_videos: # for each training video read descriptors 
        vid = read_video(vpath, gray=True, num_frames=100)
        pts, _ = func(vid, **kw)
        desc_train.append(compute_descriptors(vid, pts))
        train_labels.append(label_map[action])
        print(f"    train {n}: {len(pts)} pts")
    for action, n, vpath in test_videos: # for each test video read the descriptors 
        vid = read_video(vpath, gray=True, num_frames=100)
        pts, _ = func(vid, **kw)
        desc_test.append(compute_descriptors(vid, pts))
        test_labels.append(label_map[action])
    loaded[key] = dict(desc_train=desc_train, desc_test=desc_test, train_labels=train_labels, test_labels=test_labels)
    with open(path, 'wb') as fh:
        pickle.dump(loaded[key], fh)
    return loaded[key]

print(f"\n{'Detector':<15} {'σ':<4} {'τ':<5} {'thr':<6} {'N':<6} {'Desc':<10} {'K':<6} Accuracy")
print("-" * 65)
acc_rows = []

for det_name, det_func, det_kw, desc_type, K in experiments:
    c = _get_descriptors(det_name, det_func, det_kw) # use descriptors from cache 
    tr, ts = c['desc_train'], c['desc_test']
    train_labels = np.array(c['train_labels'])
    test_labels  = np.array(c['test_labels'])

    if desc_type == 'HOG':
        tr, ts = [d[:, :72] for d in tr], [d[:, :72] for d in ts] # take first 72 dims for HOG
    elif desc_type == 'HOF':
        tr, ts = [d[:, 72:] for d in tr], [d[:, 72:] for d in ts] # take last 72 dims slices for HOF

    total_pts = sum(len(d) for d in tr)
    sigma, tau, thresh, top_n = det_kw['sigma'], det_kw['tau'], det_kw['thresh'], det_kw['top_n']
    if total_pts == 0 or total_pts // 2 < K:
        print(f"{det_name:<15} {sigma:<4} {tau:<5} {thresh:<6} {top_n:<6} {desc_type:<10} {K:<6} skipped ({total_pts} pts)")
        continue

    bow_tr, bow_ts = bag_of_words(tr, ts, num_centers=K) # compute the BoVW histograms for train and test sets with K centers
    acc, _ = svm_train_test(bow_tr, train_labels, bow_ts, test_labels) # train SVM on train BoVW and evaluate on test BoVW
    print(f"{det_name:<15} {sigma:<4} {tau:<5} {thresh:<6} {top_n:<6} {desc_type:<10} {K:<6} {acc:.2%}")
    acc_rows.append((det_name, det_kw, desc_type, K, acc))

print("\n=== Best accuracy + GIFs per detector ===") # right here we pick the best accuracy for each detector and show the corresponding GIFs
det_func_map = {'harris': harris_detector, 'gabor': gabor_detector}
for name in ['harris', 'gabor']:
    rows = [r for r in acc_rows if r[0] == name]

    best_det, best_kw, best_desc, best_K, best_acc = max(rows, key=lambda r: r[4])
    print(f"  {name}: σ={best_kw['sigma']} τ={best_kw['tau']} thr={best_kw['thresh']} desc={best_desc} K={best_K} → {best_acc:.2%}")
    for s_action, s_path in sample_videos.items():
        s_video = read_video(s_path, gray=True, num_frames=50)
        s_pts, _ = det_func_map[name](s_video, **best_kw)
        gif_path = os.path.join(results_dir, f"{s_action}_{name}_best.gif")
        _save_gif(s_video, s_pts, gif_path)
        print(f"  Saved {os.path.basename(gif_path)}")
