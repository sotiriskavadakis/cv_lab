import numpy as np
import cv2
import os
from scipy.ndimage import map_coordinates

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'cv26_lab2_part1')

# Initial bounding boxes [x, y, width, height] from the lab sheet
BOUNDING_BOXES = {
    'face':       [154, 102, 67, 115],
    'left_hand':  [93,  272, 56,  83],
    'right_hand': [201, 270, 56,  83],
}


def lk(I1, I2, features, rho, epsilon, d_x0, d_y0):
    """Lucas-Kanade optical flow (single-scale).

    Follows the lab-sheet guide exactly:
      - images are normalised to [0, 1]  (so epsilon in [0.01, 0.1] makes sense)
      - I1(x + d_i) is computed with map_coordinates on a full meshgrid
      - Gaussian weighting G_rho is applied as a filter2D convolution
      - the 2x2 system is evaluated at each feature location

    Notation
    --------
      I1 = I_{n-1},  I2 = I_n
      A1 = dI1/dx at (x + d_i)    [Eq. 5]
      A2 = dI1/dy at (x + d_i)    [Eq. 5]
      E  = I2(x) - I1(x + d_i)    [Eq. 6]
      u  = (u_x, u_y)  correction  d_{i+1} = d_i + u

    Parameters
    ----------
    I1, I2    : (H, W) uint8 or float – consecutive frames cropped to bounding box
    features  : (N, 2) – interest-point coordinates  (col=x, row=y)
    rho       : Gaussian window std rho  (pixels)
    epsilon   : Tikhonov regularisation epsilon
    d_x0, d_y0: initial displacement estimate (scalar)

    Returns
    -------
    d_x, d_y : (N,) per-feature displacement arrays
    """
    # Normalise to [0, 1]  — epsilon range [0.01, 0.1] is calibrated for this
    I1 = I1.astype(np.float64) / 255.0
    I2 = I2.astype(np.float64) / 255.0
    H = I1.shape[0]   # number of rows    = height
    W = I1.shape[1]   # number of columns = width
    N = len(features)

    # ── Full-image coordinate grid (as in the guide) ──────────────────────────
    # x_0[r,c] = c  (column index),  y_0[r,c] = r  (row index)
    x_0, y_0 = np.meshgrid(np.arange(W), np.arange(H))   # both shape (H, W)

    # ── Precompute gradient images of I1 ─────────────────────────────────────
    # np.gradient returns (d/d_row, d/d_col)
    dI1_dy, dI1_dx = np.gradient(I1)    # A2 base image,  A1 base image

    # ── Build Gaussian kernel G_rho for filter2D  (guide: getGaussianKernel) ──
    kernel_size = int(np.ceil(3 * rho)) * 2 + 1
    k1d         = cv2.getGaussianKernel(kernel_size, rho)
    G_kernel    = (k1d @ k1d.T).astype(np.float64)       # 2-D Gaussian kernel

    # ── Per-feature iterative refinement  d_{i+1} = d_i + u ─────────────────
    d_x = np.full(N, float(d_x0))
    d_y = np.full(N, float(d_y0))

    for n in range(N):
        # Feature n pixel location: col (x) and row (y)
        fc = int(np.clip(features[n, 0], 0, W - 1))
        fr = int(np.clip(features[n, 1], 0, H - 1))

        dx_n = d_x[n]   # this feature's current displacement estimate
        dy_n = d_y[n]

        for _ in range(200):

            # I1(x + d_i) for every pixel using THIS feature's current estimate
            I1_shifted = map_coordinates(
                I1,
                [np.ravel(y_0 + dy_n), np.ravel(x_0 + dx_n)],
                order=1, mode='nearest'
            ).reshape(H, W)

            # A1 = dI1/dx at (x + d_i)                                [Eq. 5]
            A1 = map_coordinates(
                dI1_dx,
                [np.ravel(y_0 + dy_n), np.ravel(x_0 + dx_n)],
                order=1, mode='nearest'
            ).reshape(H, W)

            # A2 = dI1/dy at (x + d_i)                                [Eq. 5]
            A2 = map_coordinates(
                dI1_dy,
                [np.ravel(y_0 + dy_n), np.ravel(x_0 + dx_n)],
                order=1, mode='nearest'
            ).reshape(H, W)

            # E = I2(x) - I1(x + d_i)   residual error               [Eq. 6]
            E = I2 - I1_shifted

            # Convolve products with G_rho to apply the Gaussian window [Eq. 4]
            M11_img = cv2.filter2D(A1 * A1, -1, G_kernel) + epsilon   # Gρ*A1² + ε
            M12_img = cv2.filter2D(A1 * A2, -1, G_kernel)             # Gρ*A1*A2
            M22_img = cv2.filter2D(A2 * A2, -1, G_kernel) + epsilon   # Gρ*A2² + ε
            b1_img  = cv2.filter2D(A1 * E,  -1, G_kernel)             # Gρ*A1*E
            b2_img  = cv2.filter2D(A2 * E,  -1, G_kernel)             # Gρ*A2*E

            # Read off the 2x2 system values at THIS feature's location
            M11 = M11_img[fr, fc]
            M12 = M12_img[fr, fc]
            M22 = M22_img[fr, fc]
            b1  = b1_img [fr, fc]
            b2  = b2_img [fr, fc]

            # Solve  M * u = b  via Cramer's rule
            det = M11 * M22 - M12 * M12
            if abs(det) < 1e-12:
                break

            u_x = (M22 * b1 - M12 * b2) / det
            u_y = (M11 * b2 - M12 * b1) / det

            # Update  d_{i+1} = d_i + u
            dx_n += u_x
            dy_n += u_y

            # Convergence: stop when the correction is negligible
            if u_x**2 + u_y**2 < 1e-4:
                break

        d_x[n] = dx_n
        d_y[n] = dy_n

    return -d_x, -d_y


# ── LK vs TV-L1: norm error over all consecutive frame pairs ─────────────────
if __name__ == '__main__':
    NUM_FRAMES = 2
    tvl1_estimator = cv2.optflow.createOptFlow_DualTVL1()

    # per-box list that will collect the mean norm error of each frame pair
    frame_errors = {name: [] for name in BOUNDING_BOXES}

    for frame_idx in range(1, NUM_FRAMES):        # pairs: 1→2, 2→3, …, 69→70
        I1 = cv2.imread(os.path.join(DATA_DIR, f'{frame_idx}.png'),   cv2.IMREAD_GRAYSCALE)
        I2 = cv2.imread(os.path.join(DATA_DIR, f'{frame_idx + 1}.png'), cv2.IMREAD_GRAYSCALE)

        print(f'frame {frame_idx:02d}→{frame_idx+1:02d}', end='  ')

        for name, (x, y, w, h) in BOUNDING_BOXES.items():
            crop_I1 = I1[y:y+h, x:x+w]
            crop_I2 = I2[y:y+h, x:x+w]

            # Shi-Tomasi corners extracted from crop_I1
            pts = cv2.goodFeaturesToTrack(crop_I1, maxCorners=200,
                                          qualityLevel=0.01, minDistance=5)
            if pts is None:
                print(f'{name}: no features  ', end='')
                continue
            features = pts.reshape(-1, 2)   # (N, 2)  (col, row)

            # ── Our LK ───────────────────────────────────────────────────────
            lk_dx, lk_dy = lk(crop_I1, crop_I2, features,
                               rho=5, epsilon=0.1, d_x0=0.0, d_y0=0.0)

            # ── OpenCV TV-L1 (dense) sampled at the same feature locations ───
            flow = tvl1_estimator.calc(crop_I1, crop_I2, None)  # (H, W, 2)
            fc = np.clip(features[:, 0].astype(int), 0, crop_I1.shape[1] - 1)
            fr = np.clip(features[:, 1].astype(int), 0, crop_I1.shape[0] - 1)
            tvl1_dx = flow[fr, fc, 0]   # (N,)
            tvl1_dy = flow[fr, fc, 1]   # (N,)

            # ── Norm error per feature, then mean over features ───────────────
            # LK uses backward-flow convention (d such that I1(x+d) ≈ I2(x))
            # TV-L1 uses forward-flow convention (d such that I2(x+d) ≈ I1(x))
            # They are negatives of each other, so we negate TV-L1 before comparing
            norm_per_feature = np.sqrt((lk_dx - (-tvl1_dx))**2 +
                                       (lk_dy - (-tvl1_dy))**2)   # (N,)
            mean_error = float(norm_per_feature.mean())
            frame_errors[name].append(mean_error)

        print()   # newline after each frame pair

    # ── Summary: mean norm error per box across all frame pairs ──────────────
    print('\n── Mean norm error (LK vs TV-L1, sign-corrected) per box ──')
    for name, errs in frame_errors.items():
        if errs:
            print(f'  {name:>12s}:  {np.mean(errs):.4f} pixels  '
                  f'(min {np.min(errs):.4f}  max {np.max(errs):.4f})')







import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data', 'cv26_lab2_part1')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'part1')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Initial bounding boxes [x, y, width, height] from the lab sheet
BOUNDING_BOXES = {
    'face':       [154, 102, 67, 115],
    'left_hand':  [93,  272, 56,  83],
    'right_hand': [201, 270, 56,  83],
}


def lk(I1, I2, features, rho, epsilon, d_x0, d_y0):
    """Lucas-Kanade optical flow (single-scale).

    Follows the lab-sheet guide exactly:
      - images are normalised to [0, 1]  (so epsilon in [0.01, 0.1] makes sense)
      - I1(x + d_i) is computed with map_coordinates on a full meshgrid
      - Gaussian weighting G_rho is applied as a filter2D convolution
      - the 2x2 system is evaluated at each feature location

    Notation
    --------
      I1 = I_{n-1},  I2 = I_n
      A1 = dI1/dx at (x + d_i)    [Eq. 5]
      A2 = dI1/dy at (x + d_i)    [Eq. 5]
      E  = I2(x) - I1(x + d_i)    [Eq. 6]
      u  = (u_x, u_y)  correction  d_{i+1} = d_i + u

    Parameters
    ----------
    I1, I2    : (H, W) uint8 or float – consecutive frames cropped to bounding box
    features  : (N, 2) – interest-point coordinates  (col=x, row=y)
    rho       : Gaussian window std rho  (pixels)
    epsilon   : Tikhonov regularisation epsilon
    d_x0, d_y0: initial displacement estimate (scalar)

    Returns
    -------
    d_x, d_y : (N,) per-feature displacement arrays
    """
    # Normalise to [0, 1]  — epsilon range [0.01, 0.1] is calibrated for this
    I1 = I1.astype(np.float64) / 255.0
    I2 = I2.astype(np.float64) / 255.0
    H = I1.shape[0]   # number of rows    = height
    W = I1.shape[1]   # number of columns = width

    # ── Full-image coordinate grid (as in the guide) ──────────────────────────
    # x_0[r,c] = c  (column index),  y_0[r,c] = r  (row index)
    x_0, y_0 = np.meshgrid(np.arange(W), np.arange(H))   # both shape (H, W)

    # ── Precompute gradient images of I1 ─────────────────────────────────────
    # np.gradient returns (d/d_row, d/d_col)
    dI1_dy, dI1_dx = np.gradient(I1)    # A2 base image,  A1 base image

    # ── Build Gaussian kernel G_rho for filter2D  (guide: getGaussianKernel) ──
    kernel_size = int(np.ceil(3 * rho)) * 2 + 1
    k1d         = cv2.getGaussianKernel(kernel_size, rho)
    G_kernel    = (k1d @ k1d.T).astype(np.float64)       # 2-D Gaussian kernel

    # ── Initialise a dense displacement field  (one value per pixel) ────────────
    d_x_img = np.full((H, W), float(d_x0))   # (H, W)
    d_y_img = np.full((H, W), float(d_y0))   # (H, W)

    # ── Iterative refinement  d_{i+1} = d_i + u ──────────────────────────────
    for _ in range(100):

        # Each pixel uses its OWN current displacement estimate:
        # x_0 + d_x_img and y_0 + d_y_img are both (H, W) — same shape
        shifted_cols = np.ravel(x_0 + d_x_img)   # (H*W,)
        shifted_rows = np.ravel(y_0 + d_y_img)   # (H*W,)

        # I1(x + d_i) for every pixel                                 [Eq. 6]
        I1_shifted = map_coordinates(
            I1,
            [shifted_rows, shifted_cols],
            order=1, mode='nearest'
        ).reshape(H, W)

        # A1 = dI1/dx at (x + d_i)                                    [Eq. 5]
        A1 = map_coordinates(
            dI1_dx,
            [shifted_rows, shifted_cols],
            order=1, mode='nearest'
        ).reshape(H, W)

        # A2 = dI1/dy at (x + d_i)                                    [Eq. 5]
        A2 = map_coordinates(
            dI1_dy,
            [shifted_rows, shifted_cols],
            order=1, mode='nearest'
        ).reshape(H, W)

        # E = I2(x) - I1(x + d_i)   residual error                   [Eq. 6]
        E = I2 - I1_shifted

        # Convolve products with G_rho to apply the Gaussian window    [Eq. 4]
        M11_img = cv2.filter2D(A1 * A1, -1, G_kernel) + epsilon   # Gρ*A1² + ε
        M12_img = cv2.filter2D(A1 * A2, -1, G_kernel)             # Gρ*A1*A2
        M22_img = cv2.filter2D(A2 * A2, -1, G_kernel) + epsilon   # Gρ*A2² + ε
        b1_img  = cv2.filter2D(A1 * E,  -1, G_kernel)             # Gρ*A1*E
        b2_img  = cv2.filter2D(A2 * E,  -1, G_kernel)             # Gρ*A2*E
        
        
        # Solve  M * u = b  for EVERY pixel via Cramer's rule
        det      = M11_img * M22_img - M12_img * M12_img            # (H, W)
        valid    = np.abs(det) > 1e-12
        safe_det = np.where(valid, det, 1.0)

        u_x_img = np.where(valid, (M22_img * b1_img - M12_img * b2_img) / safe_det, 0.0)
        u_y_img = np.where(valid, (M11_img * b2_img - M12_img * b1_img) / safe_det, 0.0)

        # Update dense displacement field  d_{i+1} = d_i + u
        d_x_img += u_x_img
        d_y_img += u_y_img
        fc = np.clip(features[:, 0].astype(int), 0, W - 1)   # col (x)
        fr = np.clip(features[:, 1].astype(int), 0, H - 1)   # row (y)

        # Convergence: stop when the largest per-pixel correction is negligible
        u_x_features = u_x_img[fr, fc]
        u_y_features = u_y_img[fr, fc]
        
        if np.max(u_x_features**2 + u_y_features**2) < 1e-4:
            break

    # ── Sample the final dense field at the requested feature locations ───────
    # Keep feature coordinates in floating point to preserve subpixel accuracy.
    
    d_x = map_coordinates(d_x_img, [fr, fc], order=1, mode='nearest')
    d_y = map_coordinates(d_y_img, [fr, fc], order=1, mode='nearest')

    return d_x, d_y


# ── Quiver plots for all consecutive frame pairs, saved to results/part1 ──────
if __name__ == '__main__':
    from scipy.ndimage import shift as ndimage_shift

    RHO     = 3
    EPSILON = 0.05

    # ── Synthetic test: I2 = I1 shifted by (1, 1) — expected d_x≈-1, d_y≈-1 ──
    I1_full = cv2.imread(os.path.join(DATA_DIR, '1.png'), cv2.IMREAD_GRAYSCALE)
    I2_full = ndimage_shift(I1_full.astype(np.float64),
                            shift=[1, 1], mode='nearest').astype(np.uint8)

    print('── Synthetic test (shift +1, +1) ──')
    for name, (x, y, w, h) in BOUNDING_BOXES.items():
        crop_I1 = I1_full[y:y+h, x:x+w]
        crop_I2 = I2_full[y:y+h, x:x+w]

        pts = cv2.goodFeaturesToTrack(crop_I2, maxCorners=200,
                                      qualityLevel=0.01, minDistance=5)
        if pts is None:
            print(f'  {name}: no features found')
            continue

        features = pts.reshape(-1, 2)
        d_x, d_y = lk(crop_I1, crop_I2, features,
                       rho=RHO, epsilon=EPSILON, d_x0=0.0, d_y0=0.0)

        print(f'  {name}:  d_x={d_x.mean():+.4f} (expected -1.0)'
              f'   d_y={d_y.mean():+.4f} (expected -1.0)')

    # ── Quiver plots for all consecutive frame pairs ───────────────────────────
    






def lk(I1, I2, features, rho, epsilon, d_x0, d_y0):
    """Lucas-Kanade optical flow (single-scale).

    Follows the lab-sheet guide exactly:
      - images are normalised to [0, 1]  (so epsilon in [0.01, 0.1] makes sense)
      - I1(x + d_i) is computed with map_coordinates on a full meshgrid
      - Gaussian weighting G_rho is applied as a filter2D convolution
      - the 2x2 system is evaluated at each feature location

    Notation
    --------
      I1 = I_{n-1},  I2 = I_n
      A1 = dI1/dx at (x + d_i)    [Eq. 5]
      A2 = dI1/dy at (x + d_i)    [Eq. 5]
      E  = I2(x) - I1(x + d_i)    [Eq. 6]
      u  = (u_x, u_y)  correction  d_{i+1} = d_i + u

    Parameters
    ----------
    I1, I2    : (H, W) uint8 or float – consecutive frames cropped to bounding box
    features  : (N, 2) – interest-point coordinates  (col=x, row=y)
    rho       : Gaussian window std rho  (pixels)
    epsilon   : Tikhonov regularisation epsilon
    d_x0, d_y0: initial displacement estimate (scalar)

    Returns
    -------
    d_x, d_y : (N,) per-feature displacement arrays
    """
    # Normalise to [0, 1]  — epsilon range [0.01, 0.1] is calibrated for this
    I1 = I1.astype(np.float64) / 255.0
    I2 = I2.astype(np.float64) / 255.0
    H = I1.shape[0]   # number of rows    = height
    W = I1.shape[1]   # number of columns = width
    N = len(features)

    # ── Full-image coordinate grid (as in the guide) ──────────────────────────
    # x_0[r,c] = c  (column index),  y_0[r,c] = r  (row index)
    x_0, y_0 = np.meshgrid(np.arange(W), np.arange(H))   # both shape (H, W)

    # ── Precompute gradient images of I1 ─────────────────────────────────────
    # np.gradient returns (d/d_row, d/d_col)
    dI1_dy, dI1_dx = np.gradient(I1)    # A2 base image,  A1 base image

    # ── Build Gaussian kernel G_rho for filter2D  (guide: getGaussianKernel) ──
    kernel_size = int(np.ceil(3 * rho)) * 2 + 1
    k1d         = cv2.getGaussianKernel(kernel_size, rho)
    G_kernel    = (k1d @ k1d.T).astype(np.float64)       # 2-D Gaussian kernel

    # ── Initialise per-feature displacements ─────────────────────────────────
    d_x = np.full(N, float(d_x0))
    d_y = np.full(N, float(d_y0))

    # ── Iterative refinement  d_{i+1} = d_i + u ──────────────────────────────
    for _ in range(100):

        # Common shift for this iteration (mean of per-feature estimates)
        dx_i = float(np.mean(d_x))
        dy_i = float(np.mean(d_y))

        # I1(x + d_i) for every pixel  — guide:
        #   map_coordinates(I1, [ravel(y_0+dy_i), ravel(x_0+dx_i)], order=1)
        I1_shifted = map_coordinates(
            I1,
            [np.ravel(y_0 + dy_i), np.ravel(x_0 + dx_i)],
            order=1, mode='nearest'
        ).reshape(H, W)

        # A1 = dI1/dx at (x + d_i)                                    [Eq. 5]
        A1 = map_coordinates(
            dI1_dx,
            [np.ravel(y_0 + dy_i), np.ravel(x_0 + dx_i)],
            order=1, mode='nearest'
        ).reshape(H, W)

        # A2 = dI1/dy at (x + d_i)                                    [Eq. 5]
        A2 = map_coordinates(
            dI1_dy,
            [np.ravel(y_0 + dy_i), np.ravel(x_0 + dx_i)],
            order=1, mode='nearest'
        ).reshape(H, W)

        # E = I2(x) - I1(x + d_i)   residual error                   [Eq. 6]
        E = I2 - I1_shifted

        # ── Apply Gaussian weighting via convolution  (guide: filter2D) ───────
        # Each convolved image gives Gρ * (product) at every pixel
        M11_img = cv2.filter2D(A1 * A1, -1, G_kernel) + epsilon   # Gρ*A1² + ε
        M12_img = cv2.filter2D(A1 * A2, -1, G_kernel)             # Gρ*A1*A2
        M22_img = cv2.filter2D(A2 * A2, -1, G_kernel) + epsilon   # Gρ*A2² + ε
        b1_img  = cv2.filter2D(A1 * E,  -1, G_kernel)             # Gρ*A1*E
        b2_img  = cv2.filter2D(A2 * E,  -1, G_kernel)             # Gρ*A2*E

        # ── Sample at each feature location ───────────────────────────────────
        fc = np.clip(features[:, 0].astype(int), 0, W - 1)   # col (x)
        fr = np.clip(features[:, 1].astype(int), 0, H - 1)   # row (y)

        M11 = M11_img[fr, fc]   # (N,)
        M12 = M12_img[fr, fc]
        M22 = M22_img[fr, fc]
        b1  = b1_img [fr, fc]
        b2  = b2_img [fr, fc]

        # ── Solve  M * u = b  per feature via Cramer's rule ──────────────────
        det      = M11 * M22 - M12 * M12
        valid    = np.abs(det) > 1e-12
        safe_det = np.where(valid, det, 1.0)

        u_x = np.where(valid, (M22 * b1 - M12 * b2) / safe_det, 0.0)
        u_y = np.where(valid, (M11 * b2 - M12 * b1) / safe_det, 0.0)

        # ── Update  d_{i+1} = d_i + u ────────────────────────────────────────
        d_x += u_x
        d_y += u_y

        # Convergence: stop when all corrections are negligible
        if np.max(u_x**2 + u_y**2) < 1e-4:
            break

    return -d_x, -d_y