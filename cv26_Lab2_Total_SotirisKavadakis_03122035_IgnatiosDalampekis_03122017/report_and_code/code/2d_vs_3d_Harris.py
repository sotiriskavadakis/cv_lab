"""
harris_comparison.py

Multi-scale (automatic scale selection) versions of both 2D and 3D Harris.

2D Harris-Laplacian:
  Detect corners at scales σ₀…σ_{N-1}.  Keep a point (x,y,σᵢ) only if the
  normalised LoG response at (x,y) is a local maximum across neighbouring
  scales — i.e. the detector itself "votes" for the right scale per point.

3D Harris multi-scale:
  Compute the H volume at scale pairs (σ₀,τ₀)…(σ_{N-1},τ_{N-1}).  Keep a
  voxel (x,y,t,σᵢ,τᵢ) only if H(x,y,t) at that scale is strictly greater
  than H(x,y,t) at every other tested scale — same idea, three dimensions.

Output (no titles, no axes, yellow circles whose radius encodes σ):
  harris2d_multiscale.png
  harris3d_multiscale.png
"""

import sys, os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import convolve1d, maximum_filter

# ── paths ──────────────────────────────────────────────────────────────────────
lab2_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pictures_dir = os.path.join(lab2_dir, 'docs', 'pictures')
results_dir  = os.path.join(lab2_dir, 'results')
os.makedirs(pictures_dir, exist_ok=True)
os.makedirs(results_dir,  exist_ok=True)

sys.path.insert(0, os.path.join(lab2_dir, 'data', 'cv26_lab2_part2_3'))
from cv26_lab2_utils import read_video

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def gaussian_kernel(sigma):
    n = int(np.ceil(3 * sigma)) * 2 + 1
    g = cv2.getGaussianKernel(n, sigma)
    return (g @ g.T).astype(np.float64)

def smooth2d(I, sigma):
    return cv2.filter2D(I.astype(np.float64), cv2.CV_64F,
                        gaussian_kernel(sigma), borderType=cv2.BORDER_REFLECT)

def log_response(I, sigma):
    """Normalised (σ²·|LoG|) response — used for 2D scale selection."""
    n = int(np.ceil(3 * sigma)) * 2 + 1
    x, y = np.meshgrid(np.arange(-(n//2), n//2+1), np.arange(-(n//2), n//2+1))
    r2 = x**2 + y**2
    s2 = sigma**2
    kernel = ((r2 - 2*s2) / (2*np.pi*s2**3)) * np.exp(-r2 / (2*s2))
    kernel -= kernel.mean()
    resp = cv2.filter2D(I.astype(np.float64), cv2.CV_64F,
                        kernel.astype(np.float64), borderType=cv2.BORDER_REFLECT)
    return (sigma**2) * np.abs(resp)

def gaussian_smooth_3d(vol, ss, st):
    ns = int(np.ceil(3 * ss)) * 2 + 1
    nt = int(np.ceil(3 * st)) * 2 + 1
    ks = cv2.getGaussianKernel(ns, ss).flatten()
    kt = cv2.getGaussianKernel(nt, st).flatten()
    out = convolve1d(vol.astype(np.float64), ks, axis=0)
    out = convolve1d(out, ks, axis=1)
    return convolve1d(out, kt, axis=2)

# ─────────────────────────────────────────────────────────────────────────────
# 2D Harris-Laplacian  (automatic scale selection via LoG)
# ─────────────────────────────────────────────────────────────────────────────
def harris_response_2d(I, sigma, rho, k=0.04):
    """Returns R map (Harris cornerness) for one scale."""
    I_s = smooth2d(I, sigma)
    Iy, Ix = np.gradient(I_s)
    def s(x): return smooth2d(x, rho)
    J1, J2, J3 = s(Ix*Ix), s(Ix*Iy), s(Iy*Iy)
    disc = np.sqrt(np.maximum((J1-J3)**2 + 4*J2**2, 0.0))
    lm = 0.5*(J1+J3-disc)
    lp = 0.5*(J1+J3+disc)
    return lm*lp - k*(lm+lp)**2

def harris2d_multiscale(frame, sigmas, rho_factor=1.5, k=0.04, theta=0.0005):
    """
    Harris-Laplacian: detect corners at each scale, keep only those where
    the normalised LoG is a local maximum across neighbouring scales.
    Returns Nx3 array [x, y, sigma].
    """
    I = frame.astype(np.float64) / 255.0
    N = len(sigmas)

    R_maps  = [harris_response_2d(I, s, s*rho_factor, k) for s in sigmas]
    LoG_maps = [log_response(I, s) for s in sigmas]

    points = []
    for i, sigma in enumerate(sigmas):
        R = R_maps[i]
        # spatial NMS
        nms_size = int(np.ceil(3*sigma))*2 + 1
        R_nms = maximum_filter(R, size=nms_size)
        spatial_max = (R == R_nms) & (R > theta * R.max())
        ys, xs = np.nonzero(spatial_max)

        for y, x in zip(ys, xs):
            # scale NMS: LoG at this scale must be >= neighbours
            log_here = LoG_maps[i][y, x]
            log_prev = LoG_maps[i-1][y, x] if i > 0   else -np.inf
            log_next = LoG_maps[i+1][y, x] if i < N-1 else -np.inf
            if log_here >= log_prev and log_here >= log_next:
                points.append((x, y, sigma))

    return np.array(points) if points else np.empty((0, 3))

# ─────────────────────────────────────────────────────────────────────────────
# 3D Harris multi-scale  (automatic scale selection across (σ,τ) pairs)
# ─────────────────────────────────────────────────────────────────────────────
def harris3d_hmap(video, sigma, tau, k=0.005, s=2):
    """Compute raw H volume (no thresholding) for one (σ,τ) scale."""
    V = video.astype(np.float64) / 255.0
    L = gaussian_smooth_3d(V, sigma, tau)
    d = np.array([-1.0, 0.0, 1.0])
    Lx = convolve1d(L, d, axis=1)
    Ly = convolve1d(L, d, axis=0)
    Lt = convolve1d(L, d, axis=2)
    def gs(x): return gaussian_smooth_3d(x, s*sigma, s*tau)
    Lx2, Ly2, Lt2 = gs(Lx*Lx), gs(Ly*Ly), gs(Lt*Lt)
    LxLy, LxLt, LyLt = gs(Lx*Ly), gs(Lx*Lt), gs(Ly*Lt)
    tr  = Lx2 + Ly2 + Lt2
    det = (Lx2*(Ly2*Lt2 - LyLt**2)
         - LxLy*(LxLy*Lt2 - LyLt*LxLt)
         + LxLt*(LxLy*LyLt - Ly2*LxLt))
    return det - k * tr**3

def harris3d_multiscale(video, scale_pairs, k=0.005, s=2, thresh=0.02, top_n=400):
    """
    Compute H at each (σ,τ).  Normalise each H map to [0,1] so responses are
    comparable across scales, then keep a voxel (x,y,t) at scale i only if:
      1. it is a 3×3×3 local maximum within that scale's H map, AND
      2. its normalised H is the largest among all tested scales at (x,y,t).
    Returns Nx4 [x, y, t, sigma].
    """
    print("  Computing H maps …")
    H_maps = []
    for sigma, tau in scale_pairs:
        print(f"    σ={sigma}, τ={tau}")
        H_maps.append(harris3d_hmap(video, sigma, tau, k=k, s=s))

    # normalise to [0,1] per scale so magnitudes are comparable
    H_norm = [H / (H.max() + 1e-12) for H in H_maps]
    H_norm_stack = np.stack(H_norm, axis=0)          # (S, H, W, T)
    best_scale_idx = np.argmax(H_norm_stack, axis=0)  # winning scale per voxel

    points = []
    for i, (sigma, tau) in enumerate(scale_pairs):
        H = H_maps[i]
        threshold = thresh * H.max()
        H_nms = maximum_filter(H, size=3)
        mask = (H == H_nms) & (H > threshold) & (best_scale_idx == i)
        flat = H.ravel().copy()
        flat[~mask.ravel()] = 0
        top_idx = np.argsort(flat)[::-1][:top_n]
        top_idx = top_idx[flat[top_idx] > 0]
        if len(top_idx) == 0:
            continue
        coords = np.array(np.unravel_index(top_idx, H.shape)).T  # y,x,t
        pts = np.column_stack([coords[:,1], coords[:,0], coords[:,2],
                                np.full(len(coords), sigma)])
        points.append(pts)
        print(f"    σ={sigma} τ={tau}: {len(pts)} points")

    return np.vstack(points) if points else np.empty((0, 4))

# ─────────────────────────────────────────────────────────────────────────────
# Load video
# ─────────────────────────────────────────────────────────────────────────────
data_dir   = os.path.join(lab2_dir, 'data', 'cv26_lab2_part2_3', 'KTH')
video_path = os.path.join(data_dir, 'walking', 'person04_walking_d1_uncomp.avi')
video      = read_video(video_path, gray=True, num_frames=50)
frame_idx  = 25
frame      = video[:, :, frame_idx]

# ─────────────────────────────────────────────────────────────────────────────
# Detect
# ─────────────────────────────────────────────────────────────────────────────
sigmas_2d    = [1.5, 2.0, 3.0, 4.0]
scale_pairs  = [(1.5, 1.0), (2.0, 1.5), (3.0, 2.0)]

print("=== 2D Harris-Laplacian ===")
pts2d = harris2d_multiscale(frame, sigmas_2d, theta=0.0001)
print(f"  {len(pts2d)} corners total")

print("\n=== 3D Harris multi-scale ===")
pts3d_all = harris3d_multiscale(video, scale_pairs, thresh=0.1, top_n=2000)
# show all points within a temporal window around the chosen frame
t_window = 8
if len(pts3d_all) > 0:
    mask_t = np.abs(pts3d_all[:,2] - frame_idx) <= t_window
    pts3d = pts3d_all[mask_t]
else:
    pts3d = np.empty((0,4))
print(f"  {len(pts3d)} points in window [{frame_idx-t_window},{frame_idx+t_window}] (total {len(pts3d_all)})")

# ─────────────────────────────────────────────────────────────────────────────
# Save images
# ─────────────────────────────────────────────────────────────────────────────
def save_detection(frame, points, out_name, radius_mult=3):
    H, W = frame.shape
    dpi = 100
    fig, ax = plt.subplots(figsize=(W/dpi, H/dpi), dpi=dpi)
    ax.imshow(frame, cmap='gray', vmin=0, vmax=255)
    for pt in points:
        x, y = int(pt[0]), int(pt[1])
        r = radius_mult * float(pt[-1])
        ax.add_patch(Circle((x, y), r, edgecolor='yellow',
                             facecolor='none', linewidth=1.2))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for d in (pictures_dir, results_dir):
        plt.savefig(os.path.join(d, out_name), dpi=dpi*3,
                    bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved {out_name}")

save_detection(frame, pts2d,  'harris2d_multiscale.png')
save_detection(frame, pts3d,  'harris3d_multiscale.png')
