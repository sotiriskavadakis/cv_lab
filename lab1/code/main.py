import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# PART 1: Edge Detection in Grayscale Images

# 1.1.1 Load the image and ensure it is in grayscale format

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

I0 = cv2.imread(os.path.join(DATA_DIR, 'edgetest_26.png'), cv2.IMREAD_GRAYSCALE).astype(np.float64)

if I0 is None:
    raise FileNotFoundError("Could not load 'edgetest_26.png.")

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
    i_max, i_min = np.max(image), np.min(image)
    sigma_n = (i_max - i_min) / (10 ** (psnr_db / 20))
    noise = np.random.normal(loc=0.0, scale=sigma_n, size=image.shape)

    return image + noise

I_20 = add_gaussian_noise(I0, psnr_db=20)
I_10 = add_gaussian_noise(I0, psnr_db=10)

# Display the original and noisy images side by side for comparison
# Image with PSNR=20dB should have less noise than the one with PSNR=10dB as the noise power is lower.

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, img, title in zip(
    axes,
    [I0, I_20, I_10],
    ['Original (I0)', 'Noisy PSNR=20dB', 'Noisy PSNR=10dB']
):
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()

# 1.2.1 Filter Kernels

def gaussian_kernel(sigma: float) -> np.ndarray:
    """
    The rule for n ensures the kernel covers a significant portion of the Gaussian distribution
    """
    n = int(np.ceil(3 * sigma)) * 2 + 1 
    k1d = cv2.getGaussianKernel(n, sigma) # this returns a 1D kernel
    return (k1d @ k1d.T).astype(np.float64) # create a 2D kernel by taking the outer product
