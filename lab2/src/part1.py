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
BOX_COLORS = {'face': (0, 0,255), 'left_hand': (0, 255, 0), 'right_hand': (255, 0, 0)}


import numpy as np
import cv2
from scipy.ndimage import map_coordinates

def lk(I1, I2, features, rho, epsilon, d_x0, d_y0):
    """
    Lucas-Kanade optical flow (Dense Vectorized approach)
    """
    # 1. Κανονικοποίηση (Απαραίτητο για να έχει νόημα το epsilon)
    I1 = I1.astype(np.float64) / 255.0
    I2 = I2.astype(np.float64) / 255.0
    
    H, W = I1.shape
    
    # Προ-υπολογίζουμε τις ακέραιες συντεταγμένες των features που ψάχνουμε
    fx = np.clip(features[:, 0].astype(int), 0, W - 1)  # Στήλες (x)
    fy = np.clip(features[:, 1].astype(int), 0, H - 1)  # Γραμμές (y)

    # 2. Φτιάχνουμε το πλέγμα για όλη την εικόνα
    x_0, y_0 = np.meshgrid(np.arange(W), np.arange(H))

    # 3. Προ-υπολογισμός Παραγώγων (Gradients) του I1
    dI1_dy, dI1_dx = np.gradient(I1) 

    # 4. Δημιουργία του Γκαουσιανού Πυρήνα (Το "παράθυρο" G_rho)
    kernel_size = int(np.ceil(3 * rho)) * 2 + 1
    k1d = cv2.getGaussianKernel(kernel_size, rho)
    G_kernel = (k1d @ k1d.T).astype(np.float64)

    # 5. Αρχικοποίηση Πυκνού Πεδίου Κίνησης (Ένα d_x, d_y για ΚΑΘΕ pixel)
    # d_x0/d_y0 can be a scalar (uniform init) or a 2D map (e.g. from coarser scale)
    if np.ndim(d_x0) == 2:
        d_x_img = d_x0.astype(np.float64)
        d_y_img = d_y0.astype(np.float64)
    else:
        d_x_img = np.full((H, W), float(d_x0))
        d_y_img = np.full((H, W), float(d_y0))

    # --- Επαναληπτική Διαδικασία (Σύγκλιση) ---
    for _ in range(100):
        
        # Υπολογισμός των νέων (μετατοπισμένων) συντεταγμένων
        shifted_cols = np.ravel(x_0 + d_x_img)
        shifted_rows = np.ravel(y_0 + d_y_img)

        # 6. Παρεμβολή (Warping): Υπολογισμός I1, A1, A2 στις νέες θέσεις
        I1_shifted = map_coordinates(I1, [shifted_rows, shifted_cols], order=1, mode='nearest').reshape(H, W)
        A1 = map_coordinates(dI1_dx, [shifted_rows, shifted_cols], order=1, mode='nearest').reshape(H, W)
        A2 = map_coordinates(dI1_dy, [shifted_rows, shifted_cols], order=1, mode='nearest').reshape(H, W)

        # 7. Το Σφάλμα φωτεινότητας
        E = I2 - I1_shifted

        # 8. Εφαρμογή του Γκαουσιανού παραθύρου σε όλα τα pixels (Συνέλιξη)
        M11_img = cv2.filter2D(A1 * A1, -1, G_kernel) + epsilon
        M12_img = cv2.filter2D(A1 * A2, -1, G_kernel)
        M22_img = cv2.filter2D(A2 * A2, -1, G_kernel) + epsilon
        b1_img  = cv2.filter2D(A1 * E,  -1, G_kernel)
        b2_img  = cv2.filter2D(A2 * E,  -1, G_kernel)

        # 9. Επίλυση του Συστήματος 2x2 για ΚΑΘΕ pixel ταυτόχρονα (Cramer's Rule)
        det = M11_img * M22_img - M12_img * M12_img
        valid = np.abs(det) > 1e-12
        safe_det = np.where(valid, det, 1.0) # Αποφυγή διαίρεσης με το 0

        u_x_img = np.where(valid, (M22_img * b1_img - M12_img * b2_img) / safe_det, 0.0)
        u_y_img = np.where(valid, (M11_img * b2_img - M12_img * b1_img) / safe_det, 0.0)

        # 10. Ανανέωση του διανύσματος κίνησης
        d_x_img += u_x_img
        d_y_img += u_y_img

        # 11. Έλεγχος Σύγκλισης (ΜΟΝΟ στα features που μας ενδιαφέρουν)
        u_x_features = u_x_img[fy, fx]
        u_y_features = u_y_img[fy, fx]
        #breaking condition: if the maximum movement of features is very small, we consider it converged
        threshold = 1e-4
        max_update = np.max(np.sqrt(u_x_features**2 + u_y_features**2))
        if max_update < threshold:
            break # Αν τα features σταμάτησαν να κουνιούνται, τελειώσαμε!
    d_x=d_x_img[fy, fx]
    d_y=d_y_img[fy, fx]
    # 12. Επιστρέφουμε τα διανύσματα κίνησης ΜΟΝΟ για τα features (με αντεστραμμένο πρόσημο)
    return d_x, d_y


def lk_multiscale(I1, I2, features, rho, epsilon, d_x0, d_y0, num_scales):
    """
    Multiscale Lucas-Kanade optical flow via Gaussian pyramid.

    Builds a coarse-to-fine pyramid of num_scales levels.
    At each level (coarsest → finest):
      - runs single-scale lk() with the current d as initial condition
      - doubles d before passing it to the next finer level

    Parameters mirror lk(), plus num_scales (number of pyramid levels).
    Returns d_x, d_y at the finest (original) scale for the given features.
    """
    # ── 1. Build Gaussian pyramids (index 0 = finest, index -1 = coarsest) ──
    pyramid1 = [I1]
    pyramid2 = [I2]
    for _ in range(num_scales - 1):
        # Blur with sigma=3 before downsampling to avoid aliasing
        blurred1 = cv2.GaussianBlur(pyramid1[-1], (0, 0), sigmaX=3, sigmaY=3)
        blurred2 = cv2.GaussianBlur(pyramid2[-1], (0, 0), sigmaX=3, sigmaY=3)
        pyramid1.append(blurred1[::2, ::2])
        pyramid2.append(blurred2[::2, ::2])

    # ── 2. Initialise dense d map at coarsest scale (scalar → uniform map) ───
    H_c, W_c = pyramid1[-1].shape[:2]
    scale_factor = 2 ** (num_scales - 1)
    d_x_map = np.full((H_c, W_c), d_x0 / scale_factor)
    d_y_map = np.full((H_c, W_c), d_y0 / scale_factor)

    # ── 3. Coarse-to-fine refinement ─────────────────────────────────────────
    for level in range(num_scales - 1, -1, -1):
        I1_level = pyramid1[level]
        I2_level = pyramid2[level]
        scale     = 2 ** level

        # Scale features to this pyramid level
        features_level = features / scale
        H_l, W_l = I1_level.shape[:2]
        features_level[:, 0] = np.clip(features_level[:, 0], 0, W_l - 1)
        features_level[:, 1] = np.clip(features_level[:, 1], 0, H_l - 1)

        # Pass the full 2D displacement map as initial condition
        dx_level, dy_level = lk(
            I1_level, I2_level,
            features_level,
            rho, epsilon,
            d_x_map, d_y_map      # 2D maps — lk() resizes if needed
        )

        # Double d map and upsample when moving to the next finer level
        if level > 0:
            H_next, W_next = pyramid1[level - 1].shape[:2]

            x_next, y_next = np.meshgrid(np.arange(W_next), np.arange(H_next))
            x_coarse = x_next / 2.0
            y_coarse = y_next / 2.0

            d_x_map = map_coordinates(
                d_x_map,
                [np.ravel(y_coarse), np.ravel(x_coarse)],
                order=1, mode='nearest'
            ).reshape(H_next, W_next)

            d_y_map = map_coordinates(
                d_y_map,
                [np.ravel(y_coarse), np.ravel(x_coarse)],
                order=1, mode='nearest'
            ).reshape(H_next, W_next)

            d_x_map *= 2.0
            d_y_map *= 2.0
        else:
            # At finest level: update map at feature locations for completeness
            H_l, W_l = I1_level.shape[:2]
            fi = np.clip(features_level[:, 0].astype(int), 0, W_l - 1)
            fj = np.clip(features_level[:, 1].astype(int), 0, H_l - 1)
            d_x_map[fj, fi] = dx_level
            d_y_map[fj, fi] = dy_level

    return dx_level, dy_level


def displ(d_x, d_y):
        """
        Compute the total bounding-box displacement from per-feature flow vectors.
        Uses energy-based filtering: keep only features where ||d||^2 > threshold,
        then return the mean of the survivors.
        Falls back to plain mean if no feature survives the threshold.
        """
        energy_threshold=5
        energy = d_x ** 2 + d_y ** 2
        mask = energy > energy_threshold
        if mask.sum() > 0:
            return np.median(d_x[mask]), np.median(d_y[mask])
        # fallback: all features filtered out → plain mean
        return np.median(d_x), np.median(d_y)

# ── Synthetic test: I2 = I1 shifted by (1, 1) — expected output dx≈1, dy≈1 ───
if __name__ == '__main__':
    # from scipy.ndimage import shift as ndimage_shift
    # tvl1_estimator = cv2.optflow.DualTVL1OpticalFlow_create(nscales=1)

    # I1 = cv2.imread(os.path.join(DATA_DIR, '1.png'), cv2.IMREAD_GRAYSCALE)

    # # Create I2 by shifting I1 by exactly +1 pixel in x (col) and +1 pixel in y (row)
    # # ndimage_shift(image, [shift_row, shift_col])
    # shft_y, shft_x = 1.0, 1.0
    # shift=[shft_y, shft_x]
    # I2 = ndimage_shift(I1.astype(np.float64), shift=shift, mode='nearest').astype(np.uint8)
    
    # for name, (x, y, w, h) in BOUNDING_BOXES.items():
    #     crop_I1 = I1[y:y+h, x:x+w]
    #     crop_I2 = I2[y:y+h, x:x+w]

    #     pts = cv2.goodFeaturesToTrack(crop_I2, maxCorners=2000,
    #                                   qualityLevel=0.01, minDistance=5)
        

    #     features = pts.reshape(-1, 2)        # (N, 2)  (col, row)

    #     d_x_lk, d_y_lk = lk(crop_I1, crop_I2, features,
    #                            rho=5, epsilon=0.001, d_x0=0.0, d_y0=0.0)
    #     displacement_x, displacement_y = displ(-d_x_lk, -d_y_lk)  # Αντιστροφή για να ταιριάζει με την κατεύθυνση της μετατόπισης
    #     d_x_mlk, d_y_mlk = lk_multiscale(crop_I1, crop_I2, features,
    #                                     rho=5, epsilon=0.001, d_x0=0.0, d_y0=0.0, num_scales=3)
    #     displacement_x_mlk, displacement_y_mlk = displ(-d_x_mlk, -d_y_mlk)
    
    #     lk_dx = -d_x_lk  # Αντιστροφή πρόσημου για να ταιριάζει με την κατεύθυνση της μετατόπισης
    #     lk_dy = -d_y_lk
    #     mlk_dx = -d_x_mlk
    #     mlk_dy = -d_y_mlk
    #     flow = tvl1_estimator.calc(crop_I1, crop_I2, None)  # (H, W, 2)
    #     fx = np.clip(features[:, 0].astype(int), 0, crop_I1.shape[1] - 1)
    #     fy = np.clip(features[:, 1].astype(int), 0, crop_I1.shape[0] - 1)
    #     tvl1_dx = flow[fy, fx, 0]   # (N,)
    #     tvl1_dy = flow[fy, fx, 1]   # (N,)
    #     displacement_x_tvl1, displacement_y_tvl1 = displ(tvl1_dx, tvl1_dy)

    #     print(f'{name}:')
    #     print(f'  features      : {len(lk_dx)}')
    #     print(f' Uni scale LK ')
    #     print(f'  mean d_x = {lk_dx.mean():.4f}  (expected  {shft_x:.1f})')
    #     print(f'  mean d_y = {lk_dy.mean():.4f}  (expected  {shft_y:.1f})')
    #     print(f' diplacement: dx = {displacement_x:.4f}  dy = {displacement_y:.4f}  (expected {shft_x:.1f}, {shft_y:.1f})')
    #     print(f' Multi scale LK ')
    #     print(f'  mean d_x = {mlk_dx.mean():.4f}  (expected  {shft_x:.1f})')
    #     print(f'  mean d_y = {mlk_dy.mean():.4f}  (expected  {shft_y:.1f}  )')
    #     print(f' diplacement: dx = {displacement_x_mlk:.4f}  dy = {displacement_y_mlk:.4f}  (expected {shft_x:.1f}, {shft_y:.1f})')
    #     print(f' TV-L1 ')
    #     print(f'  TV-L1 mean d_x = {tvl1_dx.mean():.4f}  (expected  {shft_x:.1f})')
    #     print(f'  TV-L1 mean d_y = {tvl1_dy.mean():.4f}  (expected  {shft_y:.1f})')
    #     print(f' diplacement: dx = {displacement_x_tvl1:.4f}  dy = {displacement_y_tvl1:.4f}  (expected {shft_x:.1f}, {shft_y:.1f})')
    # # # ── Grid plots: I1 + features | I2 + features | optical flow ─────────────
    # print('\n── Grid plots ──')
    frame_files = sorted(
         [f for f in os.listdir(DATA_DIR) if f.endswith('.png')],
         key=lambda f: int(os.path.splitext(f)[0])
     )
    NUM_FRAMES = len(frame_files)

    # for idx in range(NUM_FRAMES - 1):
    #     f1 = cv2.imread(os.path.join(DATA_DIR, frame_files[idx]),     cv2.IMREAD_GRAYSCALE)
    #     f2 = cv2.imread(os.path.join(DATA_DIR, frame_files[idx + 1]), cv2.IMREAD_GRAYSCALE)

    #     # 3 bounding boxes × 3 columns (I1 | I2+pts | flow)
    #     fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    #     fig.suptitle(f'LK optical flow — frames {idx+1} → {idx+2}', fontsize=12)

    #     col_titles = ['I1', 'I2 (features)', 'Optical flow']
    #     for col, title in enumerate(col_titles):
    #         axes[0, col].set_title(title, fontsize=10)

    #     for row, (name, (x, y, w, h)) in enumerate(BOUNDING_BOXES.items()):
    #         crop1 = f1[y:y+h, x:x+w]
    #         crop2 = f2[y:y+h, x:x+w]

           
    #         pts = cv2.goodFeaturesToTrack(crop2, maxCorners=200,
    #                                        qualityLevel=0.01, minDistance=5)

    #         # ── column 0: I1 ───────────────────────────────────────────────────
    #         axes[row, 0].imshow(crop1, cmap='gray', origin='upper')
    #         axes[row, 0].set_ylabel(name, fontsize=9)
    #         axes[row, 0].axis('off')

    #         # ── column 1: I2 with detected feature points ─────────────────────
    #         axes[row, 1].imshow(crop2, cmap='gray', origin='upper')
    #         if pts is not None:
    #             features = pts.reshape(-1, 2)
    #             axes[row, 1].scatter(features[:, 0], features[:, 1],
    #                                  s=12, c='lime', marker='o',
    #                                  edgecolors='black', linewidths=0.3)
    #         axes[row, 1].axis('off')

    #         # ── column 2: optical flow quiver without image background ────────
    #         axes[row, 2].set_xlim(0, w)
    #         axes[row, 2].set_ylim(h, 0)
    #         axes[row, 2].set_aspect('equal')
    #         axes[row, 2].axis('off')
    #         if pts is not None:
    #             features = pts.reshape(-1, 2)
    #             d_x, d_y = lk(crop1, crop2, features,
    #                           rho=5, epsilon=0.001, d_x0=0.0, d_y0=0.0)
    #             axes[row, 2].quiver(
    #                 features[:, 0], features[:, 1],
    #                 -d_x, -d_y,
    #                 angles='xy', scale_units='xy', scale=1,
    #                 color='black', width=0.006
    #             )

    #     out = os.path.join(RESULTS_DIR, f'grid_lk_{idx+1:03d}.jpg')
    #     fig.savefig(out, dpi=120)
    #     plt.close(fig)
    #     print(f'  saved {os.path.basename(out)}')

    # print(f'\nDone — {NUM_FRAMES-1} grids saved to {os.path.abspath(RESULTS_DIR)}')

    # ── Section 1.1.2: Bounding-box tracking ─────────────────────────────────
    # print('\n── Bounding-box tracking ──') 

    

    # One figure with all three boxes tracked together

    # Initialise box state for each region
    box_state = {name: list(map(float, bb)) for name, bb in BOUNDING_BOXES.items()}

    # Get frame size from first frame
    sample = cv2.imread(os.path.join(DATA_DIR, frame_files[0]))
    H_vid, W_vid = sample.shape[:2]
    #out_vid = os.path.join(RESULTS_DIR, 'tracking_tvl.mp4')

    #out_vid = os.path.join(RESULTS_DIR, 'tracking_uni.mp4')
    out_vid = os.path.join(RESULTS_DIR, 'tracking_multi.mp4')

    writer = cv2.VideoWriter(
        out_vid,
        cv2.VideoWriter_fourcc(*'mp4v'),
        10,              # fps — slow enough to follow the tracking
        (W_vid, H_vid)
    )

    fig_track, axes_track = plt.subplots(
        1, NUM_FRAMES, figsize=(3 * NUM_FRAMES, 4), constrained_layout=True
    )
    fig_track.suptitle('Tracking: face + hands', fontsize=11)
    tvl1_estimator = cv2.optflow.DualTVL1OpticalFlow_create(nscales=1)
    for idx, fname in enumerate(frame_files):
        frame_bgr  = cv2.imread(os.path.join(DATA_DIR, fname))
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        vis = frame_bgr.copy()

        # Draw all three boxes on the same frame
        for name, (bx, by, bw, bh) in box_state.items():
            ix, iy, iw, ih = int(round(bx)), int(round(by)), int(round(bw)), int(round(bh))
            cv2.rectangle(vis, (ix, iy), (ix + iw, iy + ih), BOX_COLORS[name], 2)
            cv2.putText(vis, name, (ix, max(iy - 4, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, BOX_COLORS[name], 1)

        writer.write(vis)
        axes_track[idx].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes_track[idx].set_title(f'frame {idx+1}', fontsize=8)
        axes_track[idx].axis('off')

        # Propagate each box to the next frame
        if idx < NUM_FRAMES - 1:
            next_gray = cv2.imread(
                os.path.join(DATA_DIR, frame_files[idx + 1]), cv2.IMREAD_GRAYSCALE
            )
            for name in list(box_state.keys()):
                bx, by, bw, bh = box_state[name]
                cx  = max(0, min(int(round(bx)), frame_gray.shape[1] - 1))
                cy  = max(0, min(int(round(by)), frame_gray.shape[0] - 1))
                cw  = min(int(round(bw)), frame_gray.shape[1] - cx)
                ch  = min(int(round(bh)), frame_gray.shape[0] - cy)

                crop_I1 = frame_gray[cy:cy+ch, cx:cx+cw]
                crop_I2 = next_gray[cy:cy+ch, cx:cx+cw]

                pts = cv2.goodFeaturesToTrack(crop_I2, maxCorners=200,
                                              qualityLevel=0.01, minDistance=5)
                if pts is not None and len(pts) > 0:
                    features = pts.reshape(-1, 2)
                    #d_x, d_y = lk(crop_I1, crop_I2, features, rho=10, epsilon=0.001, d_x0=0.0, d_y0=0.0)
                    #flow = tvl1_estimator.calc(crop_I1, crop_I2, None)  # (H, W, 2)
    #               fx = np.clip(features[:, 0].astype(int), 0, crop_I1.shape[1] - 1)
    #               fy = np.clip(features[:, 1].astype(int), 0, crop_I1.shape[0] - 1)
    #               d_x = flow[fy, fx, 0]   # (N,)
    #               d_y = flow[fy, fx, 1]   # (N,)
                    d_x, d_y = lk_multiscale(crop_I1, crop_I2, features, rho=10, epsilon=0.001, d_x0=0.0, d_y0=0.0, num_scales=4)
                    dx_box, dy_box = displ(-d_x, -d_y)
                    box_state[name][0] += dx_box
                    box_state[name][1] += dy_box

    writer.release()
    print(f'  saved {os.path.basename(out_vid)}')
    #out_img = os.path.join(RESULTS_DIR, 'tracking_tvl.jpg')
    #out_img = os.path.join(RESULTS_DIR, 'tracking_uni.jpg')
    out_img = os.path.join(RESULTS_DIR, 'tracking_multi.jpg')

    fig_track.savefig(out_img, dpi=120)
    plt.close(fig_track)
    print(f'  saved {os.path.basename(out_img)}')

    print(f'\nTracking done — results in {os.path.abspath(RESULTS_DIR)}')
