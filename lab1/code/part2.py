import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    from .cv26_lab1_part2_utils import disk_strel, interest_points_visualization
except ImportError:
    from cv26_lab1_part2_utils import disk_strel, interest_points_visualization


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ============================================================
# Shared helper functions for Part 2
# ============================================================


# ============================================================
# Part 2.1.1 - Compute J1, J2, J3 of the structure tensor J
# ============================================================
# Paper equations for each pixel (x, y):
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

    # 2.1.1(a): Compute the smoothed image I_sigma = G_sigma * I.
    G_sigma = gaussian_kernel(sigma)
    I_sigma = convolve(I, G_sigma)

    # 2.1.1(b): Compute the first derivatives Ix and Iy of I_sigma.
    Iy, Ix = np.gradient(I_sigma)

    # 2.1.1(c): Form the derivative products Ix*Ix, Ix*Iy, Iy*Iy.
    Ix2 = Ix * Ix
    IxIy = Ix * Iy
    Iy2 = Iy * Iy

    # 2.1.1(d): Integrate the products with G_rho to obtain J1, J2, J3.
    G_rho = gaussian_kernel(rho)
    J1 = convolve(Ix2, G_rho)
    J2 = convolve(IxIy, G_rho)
    J3 = convolve(Iy2, G_rho)

    return J1, J2, J3


# ============================================================
# Part 2.1.2 - Compute the eigenvalues lambda_- and lambda_+
# ============================================================
# Paper equation:
#   lambda_(+/-) = 0.5 * (J1 + J3 +/- sqrt((J1 - J3)^2 + 4 * J2^2))


def compute_lambda_minus_plus(
    J1: np.ndarray, J2: np.ndarray, J3: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Part 2.1.2 implementation.

    Compute the two eigenvalues of the 2x2 structure tensor:
        J = [[J1, J2],
             [J2, J3]]

    The closed-form expressions are:
        lambda_minus = 0.5 * (J1 + J3 - sqrt((J1 - J3)^2 + 4 * J2^2))
        lambda_plus  = 0.5 * (J1 + J3 + sqrt((J1 - J3)^2 + 4 * J2^2))
    """
    # 2.1.2(a): Compute the discriminant term of the eigenvalue formula.
    discriminant = np.sqrt(np.maximum((J1 - J3) ** 2 + 4.0 * (J2 ** 2), 0.0))

    # 2.1.2(b): Compute the small and large eigenvalue at each pixel.
    lambda_minus = 0.5 * (J1 + J3 - discriminant)
    lambda_plus = 0.5 * (J1 + J3 + discriminant)
    return lambda_minus, lambda_plus


# ============================================================
# Part 2.1.3 - Harris cornerness criterion and interest points
# ============================================================
# Paper equation:
#   R(x, y) = lambda_- * lambda_+ - k * (lambda_- + lambda_+)^2
# Keep pixels that:
#   1. are local maxima of R
#   2. satisfy R(x, y) > theta_corn * R_max


def compute_harris_response(
    lambda_minus: np.ndarray, lambda_plus: np.ndarray, k: float = 0.05
) -> np.ndarray:
    """Compute the Harris-Stephens cornerness response R for each pixel."""
    return lambda_minus * lambda_plus - k * (lambda_minus + lambda_plus) ** 2


def detect_harris_corners(
    I: np.ndarray,
    sigma: float = 2.0,
    rho: float = 2.5,
    k: float = 0.05,
    theta_corn: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Part 2.1.3 implementation.

    Returns:
      R: Harris response image
      corners: Nx3 array with columns [x, y, sigma]
    """
    # 2.1.3(a): Compute J1, J2, J3 from Part 2.1.1.
    J1, J2, J3 = compute_J1_J2_J3(I, sigma=sigma, rho=rho)

    # 2.1.3(b): Compute the tensor eigenvalues from Part 2.1.2.
    lambda_minus, lambda_plus = compute_lambda_minus_plus(J1, J2, J3)

    # 2.1.3(c): Compute the Harris cornerness criterion R(x, y).
    R = compute_harris_response(lambda_minus, lambda_plus, k=k)

    # 2.1.3(d): Keep only local maxima inside a disk of radius ceil(3*sigma).
    ns = int(np.ceil(3 * sigma) * 2 + 1)
    B_sq = disk_strel(ns)
    cond1 = R == cv2.dilate(R, B_sq)

    # 2.1.3(e): Keep only strong responses above theta_corn * R_max.
    R_max = float(R.max())
    cond2 = R > theta_corn * R_max

    ys, xs = np.nonzero(cond1 & cond2)
    corners = np.column_stack((xs, ys, np.full(xs.shape, sigma, dtype=np.float64)))
    return R, corners.astype(np.float64)


# ============================================================
# Visualization helper for Part 2.1.1
# ============================================================


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


# ============================================================
# Visualization helper for Part 2.1.2
# ============================================================


def run_2_1_2_demo() -> None:
    """Visualize the structure tensor eigenvalues lambda_- and lambda_+."""
    sigma = 2.0
    rho = 2.5

    for name in ["solar.jpg", "blood_cells.jpg"]:
        path = os.path.join(DATA_DIR, name)
        I_color = cv2.imread(path, cv2.IMREAD_COLOR)
        if I_color is None:
            raise FileNotFoundError(f"Could not load '{name}'.")

        J1, J2, J3 = compute_J1_J2_J3(I_color, sigma=sigma, rho=rho)
        lambda_minus, lambda_plus = compute_lambda_minus_plus(J1, J2, J3)

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(
            f"Part 2.1.2 - {name} (sigma={sigma}, rho={rho})",
            fontsize=13,
            fontweight="bold",
        )

        ax[0].imshow(lambda_minus, cmap="gray")
        ax[0].set_title("lambda_-")
        ax[0].axis("off")

        ax[1].imshow(lambda_plus, cmap="gray")
        ax[1].set_title("lambda_+")
        ax[1].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()


# ============================================================
# Visualization helper for Part 2.1.3
# ============================================================


def run_2_1_3_demo() -> None:
    """Visualize Harris responses and detected corners for the requested images."""
    sigma = 2.0
    rho = 2.5
    k = 0.05
    theta_corn = 0.005

    for name in ["solar.jpg", "blood_cells.jpg"]:
        path = os.path.join(DATA_DIR, name)
        I_color = cv2.imread(path, cv2.IMREAD_COLOR)
        if I_color is None:
            raise FileNotFoundError(f"Could not load '{name}'.")

        I_rgb = cv2.cvtColor(I_color, cv2.COLOR_BGR2RGB)
        R, corners = detect_harris_corners(
            I_color, sigma=sigma, rho=rho, k=k, theta_corn=theta_corn
        )

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Part 2.1.3 - {name} (sigma={sigma}, rho={rho}, k={k}, theta_corn={theta_corn})",
            fontsize=13,
            fontweight="bold",
        )

        ax[0].imshow(R, cmap="gray")
        ax[0].set_title("Harris response R")
        ax[0].axis("off")

        interest_points_visualization(I_rgb, corners, ax=ax[1])
        ax[1].set_title(f"Detected corners: {len(corners)}")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()


if __name__ == "__main__":
    run_2_1_1_demo()
    run_2_1_2_demo()
    run_2_1_3_demo()
