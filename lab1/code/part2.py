import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

try:
    from .cv26_lab1_part2_utils import disk_strel, interest_points_visualization
except ImportError:
    from cv26_lab1_part2_utils import disk_strel, interest_points_visualization


# PART 2: Interest Point Detection in Images

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "part2")
PICTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "pictures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PICTURES_DIR, exist_ok=True)


def save_fig(filename: str) -> None:
    for directory in (RESULTS_DIR, PICTURES_DIR):
        plt.savefig(os.path.join(directory, filename), dpi=150, bbox_inches="tight")


# Part 2 helper functions


# 2.1.1 Compute J1, J2, J3 of the structure tensor J
# Paper equations for each pixel (x, y):
#   I_sigma = G_sigma * I
#   J1 = G_rho * (Ix * Ix)
#   J2 = G_rho * (Ix * Iy)
#   J3 = G_rho * (Iy * Iy)
# where Ix, Iy are the first derivatives of I_sigma.


def gaussian_kernel(sigma: float) -> np.ndarray:
    """Return a normalized 2D Gaussian kernel G_sigma."""
  
    n = int(np.ceil(3 * sigma) * 2 + 1)
    g1d = cv2.getGaussianKernel(n, sigma)
    return (g1d @ g1d.T).astype(np.float64)


def log_kernel(sigma: float) -> np.ndarray:
    """Return the LoG kernel using the same direct formula as Part 1."""
    n = int(np.ceil(3 * sigma) * 2 + 1)
    x, y = np.meshgrid(
        np.arange(-n // 2, n // 2 + 1),
        np.arange(-n // 2, n // 2 + 1),
    )

    r_squared = x**2 + y**2
    sigma_squared = sigma**2

    kernel = (
        (r_squared - 2 * sigma_squared)
        / (2 * np.pi * sigma_squared**3)
        * np.exp(-r_squared / (2 * sigma_squared))
    )
    kernel -= kernel.mean()
    return kernel.astype(np.float64)


def convolve(I: np.ndarray, K: np.ndarray) -> np.ndarray:
    """2D convolution with reflective borders to avoid artificial edges."""
    return cv2.filter2D(I, ddepth=cv2.CV_64F, kernel=K, borderType=cv2.BORDER_REFLECT)


def compute_J1_J2_J3(I: np.ndarray, sigma: float = 2.0, rho: float = 2.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Part 2.1.1 implementation of the second-moment matrix components.
    Input:
      I     : grayscale image
      sigma : differentiation scale
      rho   : integration scale
    Output:
      J1, J2, J3 : structure tensor components
    """
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


# 2.1.2 Compute the eigenvalues lambda_- and lambda_+
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


# 2.1.3 Harris cornerness criterion and interest points
# Paper equation:
#   R(x, y) = lambda_- * lambda_+ - k * (lambda_- + lambda_+)^2
# Keep pixels that:
#   1. are local maxima of R
#   2. satisfy R(x, y) > theta_corn * R_max


def harris_cornerness_criterion(
    lambda_minus: np.ndarray, lambda_plus: np.ndarray, k: float = 0.05
) -> np.ndarray:
    """Compute the Harris-Stephens cornerness response R for each pixel."""
    return lambda_minus * lambda_plus - k * (lambda_minus + lambda_plus) ** 2


def harris_corners_detector(
    I: np.ndarray,
    sigma: float = 2.0,
    rho: float = 2.5,
    k: float = 0.05,
    theta_corn: float = 0.005,
) -> np.ndarray:
    """
    Part 2.1.3 implementation.

    Returns:
      corners: Nx3 array with columns [x, y, sigma]
    """
    # 2.1.3(a): Compute J1, J2, J3 from Part 2.1.1.
    J1, J2, J3 = compute_J1_J2_J3(I, sigma=sigma, rho=rho)

    # 2.1.3(b): Compute the tensor eigenvalues from Part 2.1.2.
    lambda_minus, lambda_plus = compute_lambda_minus_plus(J1, J2, J3)

    # 2.1.3(c): Compute the Harris cornerness criterion R(x, y).
    R = harris_cornerness_criterion(lambda_minus, lambda_plus, k=k)

    # 2.1.3(d): Keep only local maxima inside a disk of radius ceil(3*sigma).
    ns = int(np.ceil(3 * sigma) * 2 + 1)
    B_sq = disk_strel(ns)
    cond1 = R == cv2.dilate(R, B_sq)

    # 2.1.3(e): Keep only strong responses above theta_corn * R_max.
    R_max = float(R.max())
    cond2 = R > theta_corn * R_max

    ys, xs = np.nonzero(cond1 & cond2)
    corners = np.column_stack((xs, ys, np.full(xs.shape, sigma, dtype=np.float64)))
    return corners.astype(np.float64)


def select_harris_laplacian_points(
    I: np.ndarray,
    sigma_0: float = 2.0,
    rho_0: float = 2.5,
    s: float = 1.5,
    N: int = 4,
    k: float = 0.05,
    theta_corn: float = 0.005,
) -> tuple[list[dict], np.ndarray]:
    """
    Part 2.2.2 implementation.

    For each Harris point detected at scale i, keep it only if its normalized LoG
    response is a local maximum over the neighboring scales i-1, i, i+1.

    Returns:
      updated_scale_results: same per-scale data with LoG responses attached
      selected_points: Nx3 array with columns [x, y, sigma]
    """
    scale_results = []
    for i in range(N):
        sigma_i = float((s ** i) * sigma_0)
        rho_i = float((s ** i) * rho_0)
        J1_i, J2_i, J3_i = compute_J1_J2_J3(I, sigma=sigma_i, rho=rho_i)
        lambda_minus_i, lambda_plus_i = compute_lambda_minus_plus(J1_i, J2_i, J3_i)
        R_i = harris_cornerness_criterion(lambda_minus_i, lambda_plus_i, k=k)
        corners_i = harris_corners_detector(I, sigma=sigma_i, rho=rho_i, k=k, theta_corn=theta_corn)
        scale_results.append(
            {
                "scale_index": i,
                "sigma": sigma_i,
                "rho": rho_i,
                "R": R_i,
                "corners": corners_i,
            }
        )

    # 2.2.2(d): Compute the normalized LoG response image at every scale.
    log_responses = []
    for scale_result in scale_results:
        sigma = scale_result["sigma"]
        LoG_sigma = log_kernel(sigma)
        log_response = convolve(I, LoG_sigma)
        log_responses.append((sigma**2) * np.abs(log_response))

    for scale_result, log_response in zip(scale_results, log_responses):
        scale_result["LoG"] = log_response

    selected_points: list[list[float]] = []

    # 2.2.2(e): Keep only points that are scale-space maxima of the LoG response.
    for i, scale_result in enumerate(scale_results):
        corners_i = scale_result["corners"]

        log_i = log_responses[i]
        log_prev = log_responses[i - 1] if i > 0 else None
        log_next = log_responses[i + 1] if i < len(log_responses) - 1 else None

        selected_at_scale: list[list[float]] = []
        for x, y, sigma_i in corners_i:
            x_int = int(round(x))
            y_int = int(round(y))
            value = log_i[y_int, x_int]

            prev_ok = log_prev is None or value >= log_prev[y_int, x_int]
            next_ok = log_next is None or value >= log_next[y_int, x_int]

            if prev_ok and next_ok:
                point = [float(x), float(y), float(sigma_i)]
                selected_at_scale.append(point)
                selected_points.append(point)

        scale_result["selected_corners"] = np.array(selected_at_scale, dtype=np.float64)

    selected_points_array = np.array(selected_points, dtype=np.float64).reshape(-1, 3)
    return scale_results, selected_points_array


# 2.3.1 Hessian determinant response for blob detection
# Paper equations:
#   H(x, y) = [[Lxx(x, y; sigma), Lxy(x, y; sigma)],
#              [Lxy(x, y; sigma), Lyy(x, y; sigma)]]
#   R(x, y) = det(H(x, y))


def compute_Lxx_Lxy_Lyy(
    I: np.ndarray, sigma: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Part 2.3.1 implementation.

    Compute the second-order partial derivatives of the smoothed image:
        I_sigma = G_sigma * I
        Lxx = d^2 I_sigma / dx^2
        Lxy = d^2 I_sigma / dx dy
        Lyy = d^2 I_sigma / dy^2
    """
    # 2.3.1(a): Smooth the input image at the selected scale sigma.
    G_sigma = gaussian_kernel(sigma)
    I_sigma = convolve(I, G_sigma)

    # 2.3.1(b): Estimate first derivatives, then differentiate once more.
    Iy, Ix = np.gradient(I_sigma)
    Ixy_from_x = np.gradient(Ix, axis=0)
    Ixy_from_y = np.gradient(Iy, axis=1)
    Lxx = np.gradient(Ix, axis=1)
    Lyy = np.gradient(Iy, axis=0)

    # Average both mixed-derivative estimates for better symmetry.
    Lxy = 0.5 * (Ixy_from_x + Ixy_from_y)
    return Lxx, Lxy, Lyy


def hessian_blobness_criterion(
    Lxx: np.ndarray, Lxy: np.ndarray, Lyy: np.ndarray
) -> np.ndarray:
    """Compute the determinant of the Hessian matrix at each pixel."""
    return Lxx * Lyy - Lxy * Lxy


# 2.3.2 Blob interest points from Hessian local maxima


def hessian_blobs_detector(
    I: np.ndarray, sigma: float = 2.0, theta_blob: float = 0.005
) -> np.ndarray:
    """
    Part 2.3.2 implementation.

    Returns:
      blobs: Nx3 array with columns [x, y, sigma]
    """
    # 2.3.2(a): Build the Hessian determinant response of Part 2.3.1.
    Lxx, Lxy, Lyy = compute_Lxx_Lxy_Lyy(I, sigma=sigma)
    R = hessian_blobness_criterion(Lxx, Lxy, Lyy)

    # 2.3.2(b): Keep local maxima inside a disk neighborhood.
    ns = int(np.ceil(3 * sigma) * 2 + 1)
    B_sq = disk_strel(ns)
    cond1 = R == cv2.dilate(R, B_sq)

    # 2.3.2(c): Keep only strong determinant values.
    R_max = float(R.max())
    cond2 = R > theta_blob * R_max

    ys, xs = np.nonzero(cond1 & cond2)
    blobs = np.column_stack((xs, ys, np.full(xs.shape, sigma, dtype=np.float64)))
    return blobs.astype(np.float64)


def select_hessian_laplacian_blobs(
    I: np.ndarray,
    sigma_0: float = 2.0,
    s: float = 1.5,
    N: int = 4,
    theta_blob: float = 0.005,
) -> tuple[list[dict], np.ndarray]:
    """
    Part 2.4.1 implementation of the Hessian-Laplace selection stage.

    For each blob detected at scale i, keep it only if its normalized LoG
    response is maximal over neighboring scales i-1, i, i+1.
    """
    scale_results = []
    for i in range(N):
        sigma_i = float((s ** i) * sigma_0)
        Lxx_i, Lxy_i, Lyy_i = compute_Lxx_Lxy_Lyy(I, sigma=sigma_i)
        R_i = hessian_blobness_criterion(Lxx_i, Lxy_i, Lyy_i)
        blobs_i = hessian_blobs_detector(I, sigma=sigma_i, theta_blob=theta_blob)
        scale_results.append(
            {
                "scale_index": i,
                "sigma": sigma_i,
                "R": R_i,
                "blobs": blobs_i,
            }
        )

    log_responses = []
    for scale_result in scale_results:
        sigma = scale_result["sigma"]
        LoG_sigma = log_kernel(sigma)
        log_response = convolve(I, LoG_sigma)
        log_responses.append((sigma**2) * np.abs(log_response))

    for scale_result, log_response in zip(scale_results, log_responses):
        scale_result["LoG"] = log_response

    selected_points: list[list[float]] = []

    for i, scale_result in enumerate(scale_results):
        blobs_i = scale_result["blobs"]

        log_i = log_responses[i]
        log_prev = log_responses[i - 1] if i > 0 else None
        log_next = log_responses[i + 1] if i < len(log_responses) - 1 else None

        selected_at_scale: list[list[float]] = []
        for x, y, sigma_i in blobs_i:
            x_int = int(round(x))
            y_int = int(round(y))
            value = log_i[y_int, x_int]

            prev_ok = log_prev is None or value >= log_prev[y_int, x_int]
            next_ok = log_next is None or value >= log_next[y_int, x_int]

            if prev_ok and next_ok:
                point = [float(x), float(y), float(sigma_i)]
                selected_at_scale.append(point)
                selected_points.append(point)

        scale_result["selected_blobs"] = np.array(selected_at_scale, dtype=np.float64)

    selected_points_array = np.array(selected_points, dtype=np.float64).reshape(-1, 3)
    return scale_results, selected_points_array


def harris_laplacian_corners_detector(
    I: np.ndarray,
    sigma_0: float = 2.0,
    rho_0: float = 2.5,
    s: float = 1.5,
    N: int = 4,
    k: float = 0.05,
    theta_corn: float = 0.005,
) -> np.ndarray:
    return select_harris_laplacian_points(
        I, sigma_0=sigma_0, rho_0=rho_0, s=s, N=N, k=k, theta_corn=theta_corn
    )[1]


def hessian_laplacian_blobs_detector(
    I: np.ndarray,
    sigma_0: float = 2.0,
    s: float = 1.5,
    N: int = 4,
    theta_blob: float = 0.005,
) -> np.ndarray:
    return select_hessian_laplacian_blobs(
        I, sigma_0=sigma_0, s=s, N=N, theta_blob=theta_blob
    )[1]


def calculate_repeatability(
    pts1: np.ndarray,
    pts2: np.ndarray,
    H: np.ndarray,
    dist_thresh: float = 3,
    scale_tol: float | None = None,
) -> tuple[float, int]:
    """
    Compute repeatability between two sets of interest points.

    Args:
        pts1: (N, 3) array of (x, y, sigma) from image 1
        pts2: (M, 3) array of (x, y, sigma) from image 2
        H: homography that maps image 1 points to image 2
        dist_thresh: maximum Euclidean distance for a match
        scale_tol: optional tolerance for |sigma1 / sigma2 - 1|

    Returns:
        repeatability: n_matches / min(len(pts1), len(pts2))
        n_matches: number of unique matches
    """
    if len(pts1) == 0 or len(pts2) == 0:
        return 0.0, 0

    xy1 = pts1[:, :2]
    sigma1 = pts1[:, 2]
    xy2 = pts2[:, :2]
    sigma2 = pts2[:, 2]

    xy1_h = np.column_stack((xy1, np.ones(len(xy1), dtype=np.float64)))
    xy1_proj_h = (H @ xy1_h.T).T

    valid = np.abs(xy1_proj_h[:, 2]) > 1e-12
    xy1_proj = xy1_proj_h[valid, :2] / xy1_proj_h[valid, 2:3]
    sigma1 = sigma1[valid]

    matched_pts2 = np.zeros(len(pts2), dtype=bool)
    n_matches = 0

    for i, point in enumerate(xy1_proj):
        distances = np.linalg.norm(xy2 - point, axis=1)
        candidates = np.where((distances < dist_thresh) & (~matched_pts2))[0]

        if len(candidates) == 0:
            continue

        if scale_tol is not None:
            scale_ratio = sigma1[i] / sigma2[candidates]
            candidates = candidates[np.abs(scale_ratio - 1.0) <= scale_tol]

            if len(candidates) == 0:
                continue

        best_match = candidates[np.argmin(distances[candidates])]
        matched_pts2[best_match] = True
        n_matches += 1

    repeatability = n_matches / min(len(pts1), len(pts2))
    return repeatability, n_matches


if __name__ == "__main__":
    sigma = 2.0
    rho = 2.5
    sigma_0 = 2.0
    rho_0 = 2.5
    s = 1.5
    N = 4
    k = 0.05
    theta_corn = 0.005
    theta_blob = 0.005
    image_names = ["solar.jpg", "blood_cells.jpg"]

    images = {}
    for name in image_names:
        image_gray_raw = cv2.imread(os.path.join(DATA_DIR, name), cv2.IMREAD_GRAYSCALE)
        image_rgb_raw = cv2.imread(os.path.join(DATA_DIR, name), cv2.IMREAD_COLOR)
        images[name] = {
            "gray": image_gray_raw.astype(np.float64),
            "rgb": cv2.cvtColor(image_rgb_raw, cv2.COLOR_BGR2RGB),
        }

    for name in image_names:
        I_gray = images[name]["gray"]
        I_rgb = images[name]["rgb"]

        # 2.1.1 Visualize J1, J2, J3
        J1, J2, J3 = compute_J1_J2_J3(I_gray, sigma=sigma, rho=rho)

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
        save_fig(f"part2_2_1_1_{os.path.splitext(name)[0]}.jpg")
        plt.close()

        # 2.1.2 Visualize lambda_- and lambda_+
        J1, J2, J3 = compute_J1_J2_J3(I_gray, sigma=sigma, rho=rho)
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
        save_fig(f"part2_2_1_2_{os.path.splitext(name)[0]}.jpg")
        plt.close()

        # 2.1.3 Visualize Harris responses and corners
        J1, J2, J3 = compute_J1_J2_J3(I_gray, sigma=sigma, rho=rho)
        lambda_minus, lambda_plus = compute_lambda_minus_plus(J1, J2, J3)
        R = harris_cornerness_criterion(lambda_minus, lambda_plus, k=k)
        corners = harris_corners_detector(I_gray, sigma=sigma, rho=rho, k=k, theta_corn=theta_corn)

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
        save_fig(f"part2_2_1_3_{os.path.splitext(name)[0]}.jpg")
        plt.close()

        # 2.2.1 Visualize multi-scale Harris corners
        scale_results = []
        for i in range(N):
            sigma_i = float((s ** i) * sigma_0)
            rho_i = float((s ** i) * rho_0)
            J1_i, J2_i, J3_i = compute_J1_J2_J3(I_gray, sigma=sigma_i, rho=rho_i)
            lambda_minus_i, lambda_plus_i = compute_lambda_minus_plus(J1_i, J2_i, J3_i)
            R_i = harris_cornerness_criterion(lambda_minus_i, lambda_plus_i, k=k)
            corners_i = harris_corners_detector(I_gray, sigma=sigma_i, rho=rho_i, k=k, theta_corn=theta_corn)
            scale_results.append(
                {
                    "scale_index": i,
                    "sigma": sigma_i,
                    "rho": rho_i,
                    "R": R_i,
                    "corners": corners_i,
                }
            )

        fig, axes = plt.subplots(2, N, figsize=(4 * N, 8))
        fig.suptitle(
            f"Part 2.2.1 - {name} (sigma_0={sigma_0}, rho_0={rho_0}, s={s}, N={N})",
            fontsize=13,
            fontweight="bold",
        )

        for ax_col, scale_result in enumerate(scale_results):
            R_i = scale_result["R"]
            corners_i = scale_result["corners"]
            sigma_i = scale_result["sigma"]
            rho_i = scale_result["rho"]

            axes[0, ax_col].imshow(R_i, cmap="gray")
            axes[0, ax_col].set_title(
                f"R, i={ax_col}\nsigma={sigma_i:.2f}, rho={rho_i:.2f}"
            )
            axes[0, ax_col].axis("off")

            interest_points_visualization(I_rgb, corners_i, ax=axes[1, ax_col])
            axes[1, ax_col].set_title(f"Corners: {len(corners_i)}")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        save_fig(f"part2_2_2_1_{os.path.splitext(name)[0]}.jpg")
        plt.close()

        # 2.2.2 Visualize Harris-Laplacian scale selection
        scale_results, selected_points = select_harris_laplacian_points(
            I_gray,
            sigma_0=sigma_0,
            rho_0=rho_0,
            s=s,
            N=N,
            k=k,
            theta_corn=theta_corn,
        )

        fig, axes = plt.subplots(2, N, figsize=(4 * N, 8))
        fig.suptitle(
            f"Part 2.2.2 - {name} (sigma_0={sigma_0}, rho_0={rho_0}, s={s}, N={N})",
            fontsize=13,
            fontweight="bold",
        )

        for ax_col, scale_result in enumerate(scale_results):
            axes[0, ax_col].imshow(scale_result["LoG"], cmap="gray")
            axes[0, ax_col].set_title(
                f"|LoG|, i={ax_col}\nsigma={scale_result['sigma']:.2f}"
            )
            axes[0, ax_col].axis("off")

            selected = scale_result.get(
                "selected_corners", np.empty((0, 3), dtype=np.float64)
            )
            interest_points_visualization(I_rgb, selected, ax=axes[1, ax_col])
            axes[1, ax_col].set_title(f"Selected: {len(selected)}")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        save_fig(f"part2_2_2_2_grid_{os.path.splitext(name)[0]}.jpg")
        plt.close()

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        fig2.suptitle(f"Part 2.2.2 - Final Harris-Laplacian points on {name}")
        interest_points_visualization(I_rgb, selected_points, ax=ax2)
        ax2.set_title(f"Total selected points: {len(selected_points)}")
        plt.tight_layout()
        save_fig(f"part2_2_2_2_final_{os.path.splitext(name)[0]}.jpg")
        plt.close()

        # 2.3 Visualize Hessian determinant blobs
        Lxx, Lxy, Lyy = compute_Lxx_Lxy_Lyy(I_gray, sigma=sigma)
        R = hessian_blobness_criterion(Lxx, Lxy, Lyy)
        blobs = hessian_blobs_detector(I_gray, sigma=sigma, theta_blob=theta_blob)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Part 2.3 - {name} (sigma={sigma}, theta_blob={theta_blob})",
            fontsize=13,
            fontweight="bold",
        )

        ax[0].imshow(R, cmap="gray")
        ax[0].set_title("det(H)")
        ax[0].axis("off")

        interest_points_visualization(I_rgb, blobs, ax=ax[1])
        ax[1].set_title(f"Detected blobs: {len(blobs)}")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        save_fig(f"part2_2_3_{os.path.splitext(name)[0]}.jpg")
        plt.close()

        # 2.4 Visualize Hessian-Laplace multi-scale blobs
        scale_results, selected_blobs = select_hessian_laplacian_blobs(
            I_gray,
            sigma_0=sigma_0,
            s=s,
            N=N,
            theta_blob=theta_blob,
        )

        fig, axes = plt.subplots(2, N, figsize=(4 * N, 8))
        fig.suptitle(
            f"Part 2.4 - {name} (sigma_0={sigma_0}, s={s}, N={N})",
            fontsize=13,
            fontweight="bold",
        )

        for ax_col, scale_result in enumerate(scale_results):
            axes[0, ax_col].imshow(scale_result["LoG"], cmap="gray")
            axes[0, ax_col].set_title(
                f"|LoG|, i={ax_col}\nsigma={scale_result['sigma']:.2f}"
            )
            axes[0, ax_col].axis("off")

            selected = scale_result.get(
                "selected_blobs", np.empty((0, 3), dtype=np.float64)
            )
            interest_points_visualization(I_rgb, selected, ax=axes[1, ax_col])
            axes[1, ax_col].set_title(f"Selected: {len(selected)}")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        save_fig(f"part2_2_4_grid_{os.path.splitext(name)[0]}.jpg")
        plt.close()

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        fig2.suptitle(f"Part 2.4 - Final Hessian-Laplace blobs on {name}")
        interest_points_visualization(I_rgb, selected_blobs, ax=ax2)
        ax2.set_title(f"Total selected blobs: {len(selected_blobs)}")
        plt.tight_layout()
        save_fig(f"part2_2_4_final_{os.path.splitext(name)[0]}.jpg")
        plt.close()

        # 2.5 Repeatability evaluation under transformations
        height, width = I_gray.shape
        center = (width / 2.0, height / 2.0)

        rotation_2x3 = cv2.getRotationMatrix2D(center, 45.0, 1.0)
        scaling_2x3 = cv2.getRotationMatrix2D(center, 0.0, 0.5)
        transformations = {
            "translation": np.array([
                [1.0, 0.0, 20.0],
                [0.0, 1.0, 15.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64),
            "rotation": np.vstack([rotation_2x3, [0.0, 0.0, 1.0]]),
            "scaling": np.vstack([scaling_2x3, [0.0, 0.0, 1.0]]),
        }

        print(f"\nRepeatability for {name}")
        print("Transformations: translation=(20, 15), rotation=45 deg, scaling=0.5 around image center")

        detector_results = [
            (
                "Harris",
                f"sigma={sigma}, rho={rho}, k={k}, theta_corn={theta_corn}",
                harris_corners_detector(I_gray, sigma=sigma, rho=rho, k=k, theta_corn=theta_corn),
            ),
            (
                "Harris-Laplacian",
                f"sigma_0={sigma_0}, rho_0={rho_0}, s={s}, N={N}, k={k}, theta_corn={theta_corn}",
                select_harris_laplacian_points(
                    I_gray, sigma_0=sigma_0, rho_0=rho_0, s=s, N=N, k=k, theta_corn=theta_corn
                )[1],
            ),
            (
                "Hessian",
                f"sigma={sigma}, theta_blob={theta_blob}",
                hessian_blobs_detector(I_gray, sigma=sigma, theta_blob=theta_blob),
            ),
            (
                "Hessian-Laplacian",
                f"sigma_0={sigma_0}, s={s}, N={N}, theta_blob={theta_blob}",
                select_hessian_laplacian_blobs(
                    I_gray, sigma_0=sigma_0, s=s, N=N, theta_blob=theta_blob
                )[1],
            ),
        ]

        for detector_name, params_text, pts1 in detector_results:
            print(f"\n  {detector_name} [{params_text}]")

            for transform_name, H in transformations.items():
                I_transformed = cv2.warpPerspective(I_gray, H, (width, height))

                if detector_name == "Harris":
                    pts2 = harris_corners_detector(
                        I_transformed, sigma=sigma, rho=rho, k=k, theta_corn=theta_corn
                    )
                elif detector_name == "Harris-Laplacian":
                    pts2 = select_harris_laplacian_points(
                        I_transformed, sigma_0=sigma_0, rho_0=rho_0, s=s, N=N, k=k, theta_corn=theta_corn
                    )[1]
                elif detector_name == "Hessian":
                    pts2 = hessian_blobs_detector(I_transformed, sigma=sigma, theta_blob=theta_blob)
                else:
                    pts2 = select_hessian_laplacian_blobs(
                        I_transformed, sigma_0=sigma_0, s=s, N=N, theta_blob=theta_blob
                    )[1]

                repeatability, n_matches = calculate_repeatability(pts1, pts2, H, dist_thresh=3)
                print(
                    f"    {transform_name}: repeatability={repeatability:.3f}, "
                    f"matches={n_matches}, pts1={len(pts1)}, pts2={len(pts2)}"
                )
