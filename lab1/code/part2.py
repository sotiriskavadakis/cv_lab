import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ============================================================
# Part 2.1.1 only - Compute J1, J2, J3 of the structure tensor J
# ============================================================
# Paper equations:
#   I_sigma = G_sigma * I
#   J1 = G_rho * (Ix * Ix)
#   J2 = G_rho * (Ix * Iy)
#   J3 = G_rho * (Iy * Iy)
# where Ix, Iy are the first derivatives of I_sigma.


def gaussian_kernel(sigma: float) -> np.ndarray:
    """Return a normalized 2D Gaussian kernel G_sigma."""
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    n = int(np.ceil(3 * sigma) * 2 + 1)
    g1d = cv2.getGaussianKernel(n, sigma)
    return (g1d @ g1d.T).astype(np.float64)


def ensure_gray_float(I: np.ndarray) -> np.ndarray:
    """Convert an input image to a single-channel float64 intensity image."""
    if I.ndim == 3:
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    return I.astype(np.float64)


def convolve(I: np.ndarray, K: np.ndarray) -> np.ndarray:
    """2D convolution with reflective borders to avoid artificial edges."""
    return cv2.filter2D(I, ddepth=cv2.CV_64F, kernel=K, borderType=cv2.BORDER_REFLECT)


def compute_J1_J2_J3(I: np.ndarray, sigma: float = 2.0, rho: float = 2.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Part 2.1.1 implementation of the second-moment matrix components.
    Input:
      I     : grayscale or BGR image
      sigma : differentiation scale
      rho   : integration scale
    Output:
      J1, J2, J3 : structure tensor components
    """
    I = ensure_gray_float(I)

    if rho <= 0:
        raise ValueError("rho must be positive.")

    # Step 1: I_sigma = G_sigma * I
    G_sigma = gaussian_kernel(sigma)
    I_sigma = convolve(I, G_sigma)

    # Step 2: Compute first derivatives Ix, Iy of the smoothed image I_sigma.
    Iy, Ix = np.gradient(I_sigma)

    # Step 3: Form the pointwise products that appear in the structure tensor.
    Ix2 = Ix * Ix
    IxIy = Ix * Iy
    Iy2 = Iy * Iy

    # Step 4: Integrate the derivative products with G_rho.
    G_rho = gaussian_kernel(rho)
    J1 = convolve(Ix2, G_rho)
    J2 = convolve(IxIy, G_rho)
    J3 = convolve(Iy2, G_rho)

    return J1, J2, J3


def run_2_1_1_demo() -> None:
    """Visualize J1, J2, J3 for the two images requested in Part 2.1.1."""
    sigma = 2.0
    rho = 2.5

    for name in ["solar.jpg", "blood_cells.jpg"]:
        path = os.path.join(DATA_DIR, name)
        I_color = cv2.imread(path, cv2.IMREAD_COLOR)
        if I_color is None:
            raise FileNotFoundError(f"Could not load '{name}'.")

        J1, J2, J3 = compute_J1_J2_J3(I_color, sigma=sigma, rho=rho)

        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(
            f"Part 2.1.1 - {name} (sigma={sigma}, rho={rho})",
            fontsize=13,
            fontweight="bold",
        )

        ax[0].imshow(J1, cmap="gray")
        ax[0].set_title("J1")
        ax[0].axis("off")

        ax[1].imshow(J2, cmap="gray")
        ax[1].set_title("J2")
        ax[1].axis("off")

        ax[2].imshow(J3, cmap="gray")
        ax[2].set_title("J3")
        ax[2].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()


if __name__ == "__main__":
    run_2_1_1_demo()
