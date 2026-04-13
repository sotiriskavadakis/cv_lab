import sys
import os
import numpy as np
from scipy.ndimage import gaussian_filter, convolve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data', 'cv26_lab2_part2_3'))
from cv26_lab2_utils import read_video, show_detection


def harris_detector(video, sigma, tau, k=0.005, s=1.5, threshold=None, top_n=500):
    """
    Spatiotemporal Harris interest point detector (3D extension).

    Args:
        video: H x W x T numpy array (grayscale video)
        sigma: spatial scale for derivative computation
        tau: temporal scale for derivative computation
        k: Harris sensitivity parameter
        s: integration scale factor (smoothing window = s*sigma, s*tau)
        threshold: absolute threshold on H values (if None, use top_n)
        top_n: number of top points to keep if threshold is None

    Returns:
        points: N x 4 array [x, y, t, sigma] for show_detection compatibility
    """
    video = video.astype(np.float64)

    # 1. Smooth video at (sigma, tau) scale
    L = gaussian_filter(video, sigma=[sigma, sigma, tau])

    # 2. Compute spatiotemporal derivatives via central differences
    d = np.array([-1, 0, 1]).reshape(1, 1, 3)
    Lx = convolve(L, d.reshape(1, 3, 1))  # spatial x (width axis=1)
    Ly = convolve(L, d.reshape(3, 1, 1))  # spatial y (height axis=0)
    Lt = convolve(L, d.reshape(1, 1, 3))  # temporal  (axis=2)

    # 3. Products of derivatives
    Lx2 = Lx * Lx
    Ly2 = Ly * Ly
    Lt2 = Lt * Lt
    LxLy = Lx * Ly
    LxLt = Lx * Lt
    LyLt = Ly * Lt

    # 4. Smooth each product with integration-scale Gaussian g(sσ, sσ, sτ)
    int_sigma = [s * sigma, s * sigma, s * tau]
    Lx2  = gaussian_filter(Lx2,  sigma=int_sigma)
    Ly2  = gaussian_filter(Ly2,  sigma=int_sigma)
    Lt2  = gaussian_filter(Lt2,  sigma=int_sigma)
    LxLy = gaussian_filter(LxLy, sigma=int_sigma)
    LxLt = gaussian_filter(LxLt, sigma=int_sigma)
    LyLt = gaussian_filter(LyLt, sigma=int_sigma)

    # 5. Harris criterion: H = det(M) - k * trace(M)^3
    trace_M = Lx2 + Ly2 + Lt2
    det_M = (Lx2 * (Ly2 * Lt2 - LyLt ** 2)
             - LxLy * (LxLy * Lt2 - LyLt * LxLt)
             + LxLt * (LxLy * LyLt - Ly2 * LxLt))
    H = det_M - k * (trace_M ** 3)

    # 6. Non-maximum suppression (3D, 3x3x3 neighbourhood)
    from scipy.ndimage import maximum_filter
    H_max = maximum_filter(H, size=3)
    is_local_max = (H == H_max) & (H > 0)

    # 7. Select interest points
    if threshold is not None:
        coords = np.argwhere(is_local_max & (H > threshold))
    else:
        # Keep top_n strongest responses
        candidate_coords = np.argwhere(is_local_max)
        if len(candidate_coords) == 0:
            return np.empty((0, 4))
        candidate_vals = H[is_local_max]
        top_idx = np.argsort(candidate_vals)[::-1][:top_n]
        coords = candidate_coords[top_idx]

    # coords columns: [row(y), col(x), frame(t)]
    points = np.column_stack([
        coords[:, 1],                          # x
        coords[:, 0],                          # y
        coords[:, 2],                          # t
        np.full(len(coords), sigma)             # sigma (for circle radius)
    ])
    return points


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__),
                            'data', 'cv26_lab2_part2_3', 'KTH')

    # Load a sample video from each class
    sample_videos = {
        'walking':     os.path.join(data_dir, 'walking',     'person04_walking_d1_uncomp.avi'),
        'running':     os.path.join(data_dir, 'running',     os.listdir(os.path.join(data_dir, 'running'))[0]),
        'handclapping': os.path.join(data_dir, 'handclapping', os.listdir(os.path.join(data_dir, 'handclapping'))[0]),
    }

    sigma = 3
    tau = 1.5

    for action, path in sample_videos.items():
        print(f"Processing {action}: {os.path.basename(path)}")
        video = read_video(path, gray=True, num_frames=50)
        points = harris_detector(video, sigma=sigma, tau=tau, k=0.005)
        print(f"  Detected {len(points)} interest points")
        show_detection(video, points)
