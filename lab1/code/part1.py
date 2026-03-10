import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'part1')
PICTURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'pictures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PICTURES_DIR, exist_ok=True)

def save_fig(filename: str) -> None:
    for directory in (RESULTS_DIR, PICTURES_DIR):
        plt.savefig(os.path.join(directory, filename))

# PART 1: Edge Detection in Grayscale Images

# 1.1.1 Load the image and ensure it is in grayscale format

I0_raw = cv2.imread(os.path.join(DATA_DIR, 'edgetest_26.png'), cv2.IMREAD_GRAYSCALE)
if I0_raw is None:
    raise FileNotFoundError("Could not load 'edgetest_26.png'.")
I0 = I0_raw.astype(np.float64) # convert it into a numpy array of type float64

# 1.1.2 Add Gaussian noise based on PSNR to the images

def add_gaussian_noise(image: np.ndarray, psnr_db: float) -> np.ndarray:
    """
    Add Gaussian noise to an image based on a specified PSNR value.

    Parameters:
    - image: Input grayscale image as a 2D numpy array.
    - psnr_db: Desired PSNR value in decibels.

    Returns:
    - Noisy image as a 2D numpy array.
    """
    i_max, i_min = np.max(image), np.min(image) # find max and min pixel values
    sigma_n = (i_max - i_min) / (10 ** (psnr_db / 20)) # compute noise std
    noise = np.random.normal(loc=0.0, scale=sigma_n, size=image.shape) # generate Gaussian noise, mean=0, std=sigma_n

    return image + noise # add noise to the original image

I_20 = add_gaussian_noise(I0, psnr_db=20)
I_10 = add_gaussian_noise(I0, psnr_db=10)

# Display the original and noisy images side by side for comparison
# Image with PSNR=20dB should have less noise than the one with PSNR=10dB as the noise power is lower.

fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
for ax, img, title in zip(
    axes,
    [I0, I_20, I_10],
    ['Original ($I_0$)', 'Noisy PSNR=20dB', 'Noisy PSNR=10dB']
):
    ax.imshow(img, cmap='gray', vmin=0, vmax=255) # black is 0, white is 255
    ax.set_title(title)
    ax.axis('off')

save_fig('noisy_images.jpg')
plt.close()

# 1.2.1 Filter Kernels

# The Gaussian kernel is used for smoothing the image before edge detection

def gaussian_kernel(sigma: float) -> np.ndarray:
    """
    The rule for n ensures the kernel covers a significant portion of the Gaussian distribution
    """
    n = int(np.ceil(3 * sigma)) * 2 + 1 
    k1d = cv2.getGaussianKernel(n, sigma) # this returns a 1D kernel
    return (k1d @ k1d.T).astype(np.float64) # create a 2D kernel by taking the outer product

# The LoG kernel is used for edge detection by highlighting regions of rapid intensity changes

def log_kernel(sigma: float) -> np.ndarray:
    """
    LoG kernel via direct meshgrid formula 
    """
    n = int(np.ceil(3 * sigma)) * 2 + 1
    x, y = np.meshgrid(np.arange(-n//2, n//2 + 1),
                       np.arange(-n//2, n//2 + 1))
    
    r_squared = x**2 + y**2
    s_squared = sigma**2

    kernel = ((r_squared - 2*s_squared) / (2 * np.pi * s_squared**3)) * np.exp(-r_squared / (2*s_squared))
    kernel -= kernel.mean() # zero-mean for better edge detection
    return kernel

# 1.2.2 

def EdgeDetect(I: np.ndarray, sigma: float, theta_edge: float, 
               laplacian_type: str = 'linear') -> np.ndarray:
    """
    Detect edges using LoG zero-crossings with gradient thresholding.

    Args:
        image:          Input grayscale image (float64).
        sigma:          Std dev for Gaussian/LoG filters.
        theta_edge:     Threshold as fraction of max gradient magnitude [0-1].
        laplacian_type: 'linear'    → L1 = LoG convolution (eq. 2)
                        'nonlinear' → L2 = morphological approx (eq. 3)
    Returns:
        D: Binary edge map (bool), True at detected edge pixels.
    """

    # smooth the image with a Gaussian kernel first to reduce noise and avoid false edges
    I_smooth = cv2.filter2D(I, ddepth=-1, kernel=gaussian_kernel(sigma),
                        borderType=cv2.BORDER_REFLECT)
    B = np.array([[0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]], dtype=np.uint8)
        
    if laplacian_type == 'linear':
        # take the convolution of the original image with the LoG kernel (smoothing kernel)
        L = cv2.filter2D(I, ddepth=-1, kernel=log_kernel(sigma),
                         borderType=cv2.BORDER_REFLECT)
    elif laplacian_type == 'nonlinear':
        # we must first smooth the image with a Gaussian kernel, and then apply dilation and erosion to get the morphological Laplacian
        dilated = cv2.dilate(I_smooth, B).astype(np.float64)
        eroded = cv2.erode(I_smooth, B).astype(np.float64)
        L = dilated + eroded - 2 * I_smooth
        # this is a non linear approximation of the laplacian
    else:
        raise ValueError("laplacian_type must be 'linear' or 'nonlinear'")
    
    # 1.2.3 Zero-crossing detection

    # First we find the Binary sign Image (1 where L>=0, 0 where L<0)
    X = (L >= 0).astype(np.uint8)
    # Then we find the morphological gradient of the sign image
    Y = cv2.dilate(X, B) - cv2.erode(X, B) # 1 only at zero-crossings

    # 1.2.4 Rejection of weak zero-crossings based on gradient magnitude
    # Keep only pixels where grad(I_smooth) > theta_edge * max_grad

    grad_y, grad_x = np.gradient(I_smooth)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    D = (Y == 1) & (grad_magnitude > theta_edge * grad_magnitude.max()) # bitwise AND
    return D

# 1.3 Evaluation of Edge Detection results

# 1.3.1 Find Ground-Truth Edges from clean I0

def find_ground_truth_edges(image: np.ndarray, theta_real_edge: float) -> np.ndarray:
    """
    To compute the ground-truth edge map from the clean image I0, 
    we first apply a simple morphological edge detection method to get an edge map.
    Then we keep only the pixels where they exceed a certain threshold based.
    """
    B = np.array([[0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]], dtype=np.uint8)
    M = cv2.dilate(image, B).astype(np.float64) - cv2.erode(image, B).astype(np.float64)
    return M > theta_real_edge 

# 1.3.2 Evaluate the detected edge map D against the ground-truth edge map GT

def evaluate_edges(D: np.ndarray, T: np.ndarray) -> dict:
    """
    Evaluate the detected edge map D against the ground-truth edge map GT.
    """
    intersection = (D & T).sum() # True Positives
    precision = intersection / D.sum() if D.sum() > 0 else 0
    recall = intersection / T.sum() if T.sum() > 0 else 0
    C = (precision + recall) / 2 if (precision + recall) > 0 else 0 

    return {'precision': precision, 'recall': recall, 'C': C}

# 1.3.3 
T = find_ground_truth_edges(I0, theta_real_edge=0.1)

# Suggested parameter values
experiments = [
    (I_20, 'PSNR=20dB', 1.5, 0.2),
    (I_10, 'PSNR=10dB', 3.0, 0.2),
]

for img, label, sigma, theta_edge in experiments:
    D_lin = EdgeDetect(img, sigma, theta_edge, laplacian_type='linear')
    D_nln = EdgeDetect(img, sigma, theta_edge, laplacian_type='nonlinear')

    m_lin = evaluate_edges(D_lin, T)
    m_nln = evaluate_edges(D_nln, T)

    print(f"\n{label} | sigma={sigma}, theta_edge={theta_edge}")
    print(f"  Linear: P={m_lin['precision']:.3f}, R={m_lin['recall']:.3f}, C={m_lin['C']:.3f}")
    print(f"  Nonlinear: P={m_nln['precision']:.3f}, R={m_nln['recall']:.3f}, C={m_nln['C']:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)

    axes[0].imshow(T, cmap='gray')
    axes[0].set_title('Ground Truth', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(D_lin, cmap='gray')
    axes[1].set_title(f'Linear (L1)\n'
                      f'P={m_lin["precision"]:.3f}  R={m_lin["recall"]:.3f}  C={m_lin["C"]:.3f}',
                      fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(D_nln, cmap='gray')
    axes[2].set_title(f'Nonlinear (L2)\n'
                      f'P={m_nln["precision"]:.3f}  R={m_nln["recall"]:.3f}  C={m_nln["C"]:.3f}',
                      fontsize=12)
    axes[2].axis('off')

    fname = f"edges_{label.replace('=','').replace(' ','_')}_sigma{sigma}_theta{theta_edge}.jpg"
    save_fig(fname)
    plt.close()

# 1.3.4 Precision-Recall Curves over theta_edge sweep

theta_values = np.linspace(0.01, 0.99, 50)

pr_experiments = [
    (I_20, 'PSNR=20dB', 1.5),
    (I_10, 'PSNR=10dB', 3.0),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (img, label, sigma) in zip(axes, pr_experiments):
    for lap_type, color in [('linear', 'blue'), ('nonlinear', 'orange')]:
        precisions, recalls = [], []
        for theta in theta_values:
            D = EdgeDetect(img, sigma, theta, laplacian_type=lap_type)
            m = evaluate_edges(D, T)
            precisions.append(m['precision'])
            recalls.append(m['recall'])
        # Average Precision: area under the PR curve (sort by recall for correct integration)
        sorted_pairs = sorted(zip(recalls, precisions))
        sorted_recalls, sorted_precisions = zip(*sorted_pairs)
        ap = np.trapezoid(sorted_precisions, sorted_recalls) if len(set(recalls)) > 1 else 0
        ax.plot(recalls, precisions, color=color, label=f'{lap_type} (AP={ap:.3f})')
        best = np.argmax([(p + r) / 2 for p, r in zip(precisions, recalls)])
        ax.scatter(recalls[best], precisions[best], color=color, marker='*', s=200, zorder=5)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{label}, σ={sigma}')
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

plt.tight_layout()
save_fig('precision_recall_curves.jpg')
plt.close()

# 1.4 Edge Detection on a real Image

I_real = cv2.imread(os.path.join(DATA_DIR, 'ermoupoli.jpg'), cv2.IMREAD_GRAYSCALE)

if I_real is None:
    raise FileNotFoundError("Could not load 'ermoupoli.jpg'.")

I_real = I_real.astype(np.float64)

# No ground truth for real images, qualitative (visual) evaluation only
real_experiments = [
    (1.5, 0.1, 'low σ, low θ'),
    (1.5, 0.3, 'low σ, high θ'),
    (3.0, 0.1, 'high σ, low θ'),
    (3.0, 0.3, 'high σ, high θ'),
]

for sigma, theta_edge, desc in real_experiments:
    D_lin = EdgeDetect(I_real, sigma, theta_edge, laplacian_type='linear')
    D_nln = EdgeDetect(I_real, sigma, theta_edge, laplacian_type='nonlinear')

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5), constrained_layout=True)
    axes[0].imshow(I_real, cmap='gray', aspect='equal')
    axes[0].set_title('Original', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(D_lin, cmap='gray', aspect='equal')
    axes[1].set_title('Linear (L1)', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(D_nln, cmap='gray', aspect='equal')
    axes[2].set_title('Nonlinear (L2)', fontsize=12)
    axes[2].axis('off')

    fname = f"real_edges_sigma{sigma}_theta{theta_edge}.jpg"
    save_fig(fname)
    plt.close()

# 1.5 2D heatmap: C metric over (sigma, theta_edge) grid

sigmas      = np.linspace(0.5, 5.0, 10)
theta_values_sweep = np.linspace(0.05, 0.95, 10)

datasets = [(I_20, 'PSNR=20dB'), (I_10, 'PSNR=10dB')]
lap_types = ['linear', 'nonlinear']

fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

for col, (img, label) in enumerate(datasets):
    for row, lap_type in enumerate(lap_types):
        # C[i, j] = C score at theta_values_sweep[i], sigmas[j]
        C_grid = np.array([
            [evaluate_edges(EdgeDetect(img, s, th, lap_type), T)['C']
             for s in sigmas]
            for th in theta_values_sweep
        ])

        ax = axes[row][col]
        im = ax.imshow(C_grid, origin='lower', aspect='auto',
                       extent=[sigmas[0], sigmas[-1],
                               theta_values_sweep[0], theta_values_sweep[-1]],
                       vmin=0, vmax=1, cmap='viridis')
        fig.colorbar(im, ax=ax, label='C score')
        ax.set_xlabel('σ')
        ax.set_ylabel('θ_edge')
        ax.set_title(f'{label} — {lap_type}')

plt.suptitle('C score heatmap over (σ, θ_edge)', fontsize=14)
save_fig('sigma_sweep.jpg')
plt.close()