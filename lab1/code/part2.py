import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

try:
    from .cv26_lab1_part2_utils import disk_strel, interest_points_visualization
except ImportError:
    from cv26_lab1_part2_utils import disk_strel, interest_points_visualization


# PART 2: Interest Point Detection in Images
#directories for saving results and loading data
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "part2")
PICTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "pictures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PICTURES_DIR, exist_ok=True)

#function to save coutput figures 
def save_fig(filename: str) -> None:
    for directory in (RESULTS_DIR, PICTURES_DIR):
        plt.savefig(os.path.join(directory, filename), dpi=150, bbox_inches="tight")


# Part 2 helper functions

#fucntion for creating a Gaussian kernel
def gaussian_kernel(sigma: float) -> np.ndarray:
    n = int(np.ceil(3 * sigma) * 2 + 1)
    g1d = cv2.getGaussianKernel(n, sigma)
    return (g1d @ g1d.T).astype(np.float64)

#log kernel for laplacian criterion , from part 1
def log_kernel(sigma: float) -> np.ndarray:
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

#convolution function with cv2.filter2D and reflective borders
def convolve(I: np.ndarray, K: np.ndarray) -> np.ndarray:
    return cv2.filter2D(I, ddepth=cv2.CV_64F, kernel=K, borderType=cv2.BORDER_REFLECT)

#helper fucntion for computing the elements of the J tensor for the Harris cornerness criterion for 2.1.1
def compute_J1_J2_J3(I: np.ndarray, sigma: float = 2.0, rho: float = 2.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Part 2.1.1 implementation of thestructure tensor J.
    Input:
      I     : grayscale image
      sigma : differentiation scale
      rho   : integration scale
    Output:
      J1, J2, J3 : structure tensor components
    """
   #firstly we smooth the grayscale image
    G_sigma = gaussian_kernel(sigma)
    I_sigma = convolve(I, G_sigma)
    # then Compute the first derivatives Ix and Iy of I_sigma.
    Iy, Ix = np.gradient(I_sigma)
    #take derivative products Ix*Ix, Ix*Iy, Iy*Iy that the tensor wants, multiplication element by element
    Ix2 = Ix * Ix
    IxIy = Ix * Iy
    Iy2 = Iy * Iy
    #apply the integrration kernel to the multiplication derivatives  with convolve to get the J1, J2, J3 elements of the structure tensor
    G_rho = gaussian_kernel(rho)
    J1 = convolve(Ix2, G_rho)
    J2 = convolve(IxIy, G_rho)
    J3 = convolve(Iy2, G_rho)

    return J1, J2, J3
#helper function for computing the eigenvalues of the J tensor for 2.1.2
def compute_lambda_minus_plus(
    J1: np.ndarray, J2: np.ndarray, J3: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Part 2.1.2
    Compute the two eigenvalues of the 2x2 structure tensor:
        J = [[J1, J2],
             [J2, J3]]
        lambda_minus = 0.5 * (J1 + J3 - sqrt((J1 - J3)^2 + 4 * J2^2))
        lambda_plus  = 0.5 * (J1 + J3 + sqrt((J1 - J3)^2 + 4 * J2^2))
    """
    #Compute the discriminant of th sstructure tensor
    discriminant = np.sqrt(np.maximum((J1 - J3) ** 2 + 4.0 * (J2 ** 2), 0.0))

    #Compute the eigenvalues lambda_- and lambda_+.
    lambda_minus = 0.5 * (J1 + J3 - discriminant)
    lambda_plus = 0.5 * (J1 + J3 + discriminant)
    return lambda_minus, lambda_plus



#helper function for harris function R(x,y) for 2.1.3
def harris_cornerness_criterion(
    lambda_minus: np.ndarray, lambda_plus: np.ndarray, k: float = 0.05
) -> np.ndarray:
    #   R(x, y) = lambda_- * lambda_+ - k * (lambda_- + lambda_+)^2
    return lambda_minus * lambda_plus - k * (lambda_minus + lambda_plus) ** 2

#function for detecting harris corners for 2.1.3 takes as input the grayscaale image and the parameters of scales and for the criterion and returns the interest points as an  Nx3 array of [x,y,sigma]
def harris_corners_detector(
    #default values for parameters
    I: np.ndarray,
    sigma: float = 2.0,
    rho: float = 2.5,
    k: float = 0.05,
    theta_corn: float = 0.005,
) -> np.ndarray:
    J1, J2, J3 = compute_J1_J2_J3(I, sigma=sigma, rho=rho)
    lambda_minus, lambda_plus = compute_lambda_minus_plus(J1, J2, J3)
    R = harris_cornerness_criterion(lambda_minus, lambda_plus, k=k)
    # Keep pixels that:
    #   1. are local maxima of R
    ns = int(np.ceil(3 * sigma) * 2 + 1)
    B_sq = disk_strel(ns)
    cond1 = R == cv2.dilate(R, B_sq)
    #2. satisfy R(x, y) > theta_corn * R_max.
    R_max = float(R.max())
    cond2 = R > theta_corn * R_max
    # keep only the points that satisfy both conditions and return them as an Nx3 array of [x, y, sigma]
    ys, xs = np.nonzero(cond1 & cond2)
    corners = np.column_stack((xs, ys, np.full(xs.shape, sigma, dtype=np.float64)))
    return corners.astype(np.float64)

#function for 2.2.2 applying the harris corners detector and the laplacian criterion for harris-laplacian scale space interest point selection
#again it takes as input the grayscale image and the parameters for the scales and the criterion and returns a tuple of two elements : a list of per-scale data with the detected corners and their LoG responses just for visualizationcauses and an Nx3 array of the selected points with columns [x,y,sigma]
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
    Part 2.2.2
    For each Harris point detected at scale i, keep it only if its normalized LoG
    response is a local maximum over the neighboring scales i-1, i, i+1.
    """
    scale_results = []
    for i in range(N):
        #compute the Harris response and corners at scale i
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

    #compute log responses at each scale with the log kernel function
    log_responses = []
    for scale_result in scale_results:
        sigma = scale_result["sigma"]
        LoG_sigma = log_kernel(sigma)
        log_response = convolve(I, LoG_sigma)
        log_responses.append((sigma**2) * np.abs(log_response))
    
    for scale_result, log_response in zip(scale_results, log_responses):
        scale_result["LoG"] = log_response

    selected_points: list[list[float]] = []

    #Keep only points that are scale-space maxima of the LoG response.
    for i, scale_result in enumerate(scale_results):
        corners_i = scale_result["corners"]
        #for each corner at scale i we check the log criterion
        log_i = log_responses[i]
        log_prev = log_responses[i - 1] if i > 0 else None
        log_next = log_responses[i + 1] if i < len(log_responses) - 1 else None

        selected_at_scale: list[list[float]] = []
        for x, y, sigma_i in corners_i:
            x_int = int(round(x))
            y_int = int(round(y))
            value = log_i[y_int, x_int]
            #the neighbors are prev and next scales for the same point
            prev_ok = log_prev is None or value >= log_prev[y_int, x_int]
            next_ok = log_next is None or value >= log_next[y_int, x_int]
            #only if it is maximum in his neighbors only then i keep it
            if prev_ok and next_ok:
                point = [float(x), float(y), float(sigma_i)]
                selected_at_scale.append(point)
                selected_points.append(point)

        scale_result["selected_corners"] = np.array(selected_at_scale, dtype=np.float64)

    selected_points_array = np.array(selected_points, dtype=np.float64).reshape(-1, 3)
    return scale_results, selected_points_array


#helper function for computing second derivatives and hessian matrix for 2.3.1
def compute_Lxx_Lxy_Lyy(
    I: np.ndarray, sigma: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    H(x, y) = [[Lxx(x, y; sigma), Lxy(x, y; sigma)],
#              [Lxy(x, y; sigma), Lyy(x, y; sigma)]]
        I_sigma = G_sigma * I
        Lxx = d^2 I_sigma / dx^2
        Lxy = d^2 I_sigma / dx dy
        Lyy = d^2 I_sigma / dy^2
    """
    #smooth the image firstly with a gaussian kernel
    G_sigma = gaussian_kernel(sigma)
    I_sigma = convolve(I, G_sigma)
    #find first derivatives and then repeat for second derivatives
    Iy, Ix = np.gradient(I_sigma)
    Ixy_from_x = np.gradient(Ix, axis=0)
    Ixy_from_y = np.gradient(Iy, axis=1)
    Lxx = np.gradient(Ix, axis=1)
    Lyy = np.gradient(Iy, axis=0)
    #take the average of second mixed derivatives to get a more accurate estimation of Lxy
    Lxy = 0.5 * (Ixy_from_x + Ixy_from_y)
    return Lxx, Lxy, Lyy

#helper function for computing the determinant of the Hessian matrix for 2.3.1
def hessian_blobness_criterion(
    Lxx: np.ndarray, Lxy: np.ndarray, Lyy: np.ndarray
) -> np.ndarray:
    #   R(x, y) = det(H(x, y))
    return Lxx * Lyy - Lxy * Lxy
#function for detecting hessian blobs for 2.3.2 takes as input the grayscale image and the parameters of scales and for the criterion and returns the interest points as an  Nx3 array of [x,y,sigma]
def hessian_blobs_detector(
    I: np.ndarray, sigma: float = 2.0, theta_blob: float = 0.005
) -> np.ndarray:
    #appy the hessian blobness criterion
    Lxx, Lxy, Lyy = compute_Lxx_Lxy_Lyy(I, sigma=sigma)
    R = hessian_blobness_criterion(Lxx, Lxy, Lyy)
    #apply the same process with harris to keep only the local maxima and the strong responses of the determinant of the hessian matrix
    #condition 1 :keep the local maxima
    ns = int(np.ceil(3 * sigma) * 2 + 1)
    B_sq = disk_strel(ns)
    cond1 = R == cv2.dilate(R, B_sq)

    #condition 2:Keep only strong determinant values.
    R_max = float(R.max())
    cond2 = R > theta_blob * R_max
    #return as a Nx3 array
    ys, xs = np.nonzero(cond1 & cond2)
    blobs = np.column_stack((xs, ys, np.full(xs.shape, sigma, dtype=np.float64)))
    return blobs.astype(np.float64)

#function for 2.4. for multiscale analysis of the hessian-laplacian blob detector method retirns Nx3 as well as log responses for visualization purposes
def select_hessian_laplacian_blobs(
    I: np.ndarray,
    sigma_0: float = 2.0,
    s: float = 1.5,
    N: int = 4,
    theta_blob: float = 0.005,
) -> tuple[list[dict], np.ndarray]:
    """
    Part 2.4.1 
    For each blob detected at scale i, keep it only if its normalized LoG
    response is maximal over neighboring scales i-1, i, i+1.
    """
    #similar process with the harris-laplacian selection but now with the hessian blobness criterion and the same log criterion for selection
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
        #log criterion for each blob at scale i
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

#final function that takes the grayscale image and the needed parameters and and returns only the neecessary Nx3 matrix for harris-laplace , as asked for the part3.1
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
    )[1] # keep only the selected points array from the tuple returned by the selection function and not the visualaization data that is not needed for part 3.1

#final function that takes the grayscale image and the needed parameters and and returns only the neecessary Nx3 matrix for hessian-laplace , as asked for the part3.1
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

# bonus function for computing the repeatability score between two sets of interest points given the homography that maps one image to the other and the parameters for distance threshold and scale tolerance for matching
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
    #find the candidate matches for each projected point from the first set to the second
    #then with the distance threshold and the scale tolerance count the matches
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
        #final repetability function after having found the matches
    repeatability = n_matches / min(len(pts1), len(pts2))
    return repeatability, n_matches


if __name__ == "__main__":
    #default parameters for the different methods and the images to test on
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
    #load the asked images and convert them to grayscale for the detectors and to rgb for visualization purposes since the interest points will be visualized on the rgb images
    images = {}
    #for each image apply the already built helper function for clean main code 
    #and visualize for report and save them for reproducability reasons
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
        J1, J2, J3 = compute_J1_J2_J3(I_gray, sigma=sigma, rho=rho)
        lambda_minus, lambda_plus = compute_lambda_minus_plus(J1, J2, J3)
        # 2.1.1 Visualize J1, J2, J3
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

        #2.1.2 Visualize lambda_- and lambda_+
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

        #2.1.3 Visualize Harris responses and corners
        harris_response = harris_cornerness_criterion(lambda_minus, lambda_plus, k=k)
        harris_corners = harris_corners_detector(
            I_gray, sigma=sigma, rho=rho, k=k, theta_corn=theta_corn
        )
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Part 2.1.3 - {name} (sigma={sigma}, rho={rho}, k={k}, theta_corn={theta_corn})",
            fontsize=13,
            fontweight="bold",
        )

        ax[0].imshow(harris_response, cmap="gray")
        ax[0].set_title("Harris response R")
        ax[0].axis("off")

        interest_points_visualization(I_rgb, harris_corners, ax=ax[1])
        ax[1].set_title(f"Detected corners: {len(harris_corners)}")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        save_fig(f"part2_2_1_3_{os.path.splitext(name)[0]}.jpg")
        plt.close()

        #2.2.1 Visualize multi-scale Harris corners for the different scales and the corresponding Harris responses at each scale
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

        #2.2.2 Visualize Harris-Laplacian scale selection
        harris_laplacian_scale_results, harris_laplacian_points = (
            select_harris_laplacian_points(
                I_gray,
                sigma_0=sigma_0,
                rho_0=rho_0,
                s=s,
                N=N,
                k=k,
                theta_corn=theta_corn,
            )
        )
        fig, axes = plt.subplots(2, N, figsize=(4 * N, 8))
        fig.suptitle(
            f"Part 2.2.2 - {name} (sigma_0={sigma_0}, rho_0={rho_0}, s={s}, N={N})",
            fontsize=13,
            fontweight="bold",
        )

        for ax_col, scale_result in enumerate(harris_laplacian_scale_results):
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
        interest_points_visualization(I_rgb, harris_laplacian_points, ax=ax2)
        ax2.set_title(f"Total selected points: {len(harris_laplacian_points)}")
        plt.tight_layout()
        save_fig(f"part2_2_2_2_final_{os.path.splitext(name)[0]}.jpg")
        plt.close()

        #2.3 Visualize Hessian determinant blobs
        Lxx, Lxy, Lyy = compute_Lxx_Lxy_Lyy(I_gray, sigma=sigma)
        hessian_response = hessian_blobness_criterion(Lxx, Lxy, Lyy)
        hessian_blobs = hessian_blobs_detector(I_gray, sigma=sigma, theta_blob=theta_blob)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Part 2.3 - {name} (sigma={sigma}, theta_blob={theta_blob})",
            fontsize=13,
            fontweight="bold",
        )

        ax[0].imshow(hessian_response, cmap="gray")
        ax[0].set_title("det(H)")
        ax[0].axis("off")

        interest_points_visualization(I_rgb, hessian_blobs, ax=ax[1])
        ax[1].set_title(f"Detected blobs: {len(hessian_blobs)}")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        save_fig(f"part2_2_3_{os.path.splitext(name)[0]}.jpg")
        plt.close()

        #2.4 Visualize Hessian-Laplace multi-scale blobs
        hessian_laplacian_scale_results, hessian_laplacian_blobs = (
            select_hessian_laplacian_blobs(
                I_gray,
                sigma_0=sigma_0,
                s=s,
                N=N,
                theta_blob=theta_blob,
            )
        )

        fig, axes = plt.subplots(2, N, figsize=(4 * N, 8))
        fig.suptitle(
            f"Part 2.4.1 - {name} (sigma_0={sigma_0}, s={s}, N={N})",
            fontsize=13,
            fontweight="bold",
        )

        for ax_col, scale_result in enumerate(hessian_laplacian_scale_results):
            R_i = scale_result["R"]
            blobs_i = scale_result["blobs"]
            sigma_i = scale_result["sigma"]

            axes[0, ax_col].imshow(R_i, cmap="gray")
            axes[0, ax_col].set_title(
                f"det(H), i={ax_col}\nsigma={sigma_i:.2f}"
            )
            axes[0, ax_col].axis("off")

            interest_points_visualization(I_rgb, blobs_i, ax=axes[1, ax_col])
            axes[1, ax_col].set_title(f"Blobs: {len(blobs_i)}")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        save_fig(f"part2_2_4_1_{os.path.splitext(name)[0]}.jpg")
        plt.close()

        fig, axes = plt.subplots(2, N, figsize=(4 * N, 8))
        fig.suptitle(
            f"Part 2.4 - {name} (sigma_0={sigma_0}, s={s}, N={N})",
            fontsize=13,
            fontweight="bold",
        )

        for ax_col, scale_result in enumerate(hessian_laplacian_scale_results):
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
        interest_points_visualization(I_rgb, hessian_laplacian_blobs, ax=ax2)
        ax2.set_title(f"Total selected blobs: {len(hessian_laplacian_blobs)}")
        plt.tight_layout()
        save_fig(f"part2_2_4_final_{os.path.splitext(name)[0]}.jpg")
        plt.close()

        #2.5 Repeatability evaluation under transformations
        height, width = I_gray.shape
        center = (width / 2.0, height / 2.0)
        # as possible transformations we tested
        #1. translation by (20, 15) pixels
        #2. rotation by 45 degrees around the image center
        #3. scaling by a factor of 0.5 around the image center
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
        #test the repeatablitiy test for each detecotr
        detector_results = [
            (
                "Harris",
                f"sigma={sigma}, rho={rho}, k={k}, theta_corn={theta_corn}",
                harris_corners,
            ),
            (
                "Harris-Laplacian",
                f"sigma_0={sigma_0}, rho_0={rho_0}, s={s}, N={N}, k={k}, theta_corn={theta_corn}",
                harris_laplacian_points,
            ),
            (
                "Hessian",
                f"sigma={sigma}, theta_blob={theta_blob}",
                hessian_blobs,
            ),
            (
                "Hessian-Laplacian",
                f"sigma_0={sigma_0}, s={s}, N={N}, theta_blob={theta_blob}",
                hessian_laplacian_blobs,
            ),
        ]
        # apply the transformations to the images and apply the repeatability function to all combinations
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

                repeatability, n_matches = calculate_repeatability(
                    pts1, pts2, H, dist_thresh=3, scale_tol=0.50
                )
                print(
                    f"    {transform_name}: repeatability={repeatability:.3f}, "
                    f"matches={n_matches}, pts1={len(pts1)}, pts2={len(pts2)}"
                )