import argparse
import os
import time

import cv26_lab1_part3_utils as p3
import cv2
import matplotlib.pyplot as plt
import numpy as np
import part2 as p2
import scipy.io
from matplotlib.patches import Circle
from sklearn.metrics import confusion_matrix


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "part3")
ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
FOLD_INDICES_PATH = os.path.join(ROOT_DIR, "Fold_Indices.mat")
DATA_DIR = os.path.join(ROOT_DIR, "Data")
CLASS_INFO = [
    ("person", "TUGraz_person"),
    ("car", "TUGraz_cars"),
    ("bike", "TUGraz_bike"),
]
SAMPLE_IMAGE_PATHS = {
    "bike": os.path.join(ROOT_DIR, "Data", "TUGraz_bike", "bike_001.png"),
    "person": os.path.join(ROOT_DIR, "Data", "TUGraz_person", "person_001.png"),
    "car": os.path.join(ROOT_DIR, "Data", "TUGraz_cars", "carsgraz_001.png"),
}
os.makedirs(RESULTS_DIR, exist_ok=True)


def summarize_features(features) -> tuple[int, int]:
    num_images = sum(len(class_features) for class_features in features)
    num_descriptors = 0

    for class_features in features:
        for descriptors in class_features:
            if descriptors is None:
                continue
            num_descriptors += len(descriptors)

    return num_images, num_descriptors


def summarize_split(data_train, label_train, data_test, label_test) -> None:
    print(f"Train images: {len(data_train)}")
    print(f"Test images: {len(data_test)}")
    print(f"Train labels: {len(label_train)}")
    print(f"Test labels: {len(label_test)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run part 3.1 feature extraction and classification."
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore cached feature files and extract features again.",
    )
    return parser.parse_args()


def load_sample_images():
    images = {}
    for label, path in SAMPLE_IMAGE_PATHS.items():
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Sample image not found: {path}")

        images[label] = cv2.resize(
            image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
        )

    return images


def load_dataset_entries():
    dataset_entries = []
    for class_name, class_dir in CLASS_INFO:
        class_path = os.path.join(DATA_DIR, class_dir)
        image_names = sorted(os.listdir(class_path))
        class_entries = []
        for image_name in image_names:
            if class_name not in image_name:
                continue
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            resized = cv2.resize(
                image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
            )
            class_entries.append(
                {
                    "class_name": class_name,
                    "path": image_path,
                    "image_name": image_name,
                    "image": resized,
                }
            )
        dataset_entries.append(class_entries)
    return dataset_entries


def get_test_entries_for_fold(dataset_entries, fold_idx: int):
    fold_mat = scipy.io.loadmat(FOLD_INDICES_PATH)
    indices = fold_mat["Indices"].flatten()[fold_idx].flatten()

    test_entries = []
    for class_idx in range(indices.shape[0]):
        idx_class = indices[class_idx].flatten()
        ordered_entries = [dataset_entries[class_idx][i] for i in idx_class]
        split_idx = int(round(0.7 * idx_class.shape[0]))
        test_entries.extend(ordered_entries[split_idx:])

    return test_entries


def extract_patch(image: np.ndarray, keypoint: np.ndarray) -> np.ndarray:
    x, y, scale = keypoint
    radius = int(round(5 * scale))
    y0 = max(int(round(y)) - radius, 0)
    y1 = min(int(round(y)) + radius + 1, image.shape[0])
    x0 = max(int(round(x)) - radius, 0)
    x1 = min(int(round(x)) + radius + 1, image.shape[1])
    return image[y0:y1, x0:x1]


def extract_sift_support_patch(image: np.ndarray, keypoint: np.ndarray) -> np.ndarray:
    x, y, scale = keypoint
    keypoint_size = 2 * np.ceil(3 * scale) + 1
    radius = int(np.ceil(keypoint_size / 2.0))
    y0 = max(int(round(y)) - radius, 0)
    y1 = min(int(round(y)) + radius + 1, image.shape[0])
    x0 = max(int(round(x)) - radius, 0)
    x1 = min(int(round(x)) + radius + 1, image.shape[1])
    return image[y0:y1, x0:x1]


def compute_descriptor_visual(
    image: np.ndarray,
    descriptor_name: str,
    detect_fun,
    desc_fun,
):
    keypoints = detect_fun(image.astype(float) / 255.0)
    if keypoints is None or len(keypoints) == 0:
        return None

    selected_idx = int(np.argmax(keypoints[:, 2]))
    selected_kp = keypoints[selected_idx]
    descriptor = desc_fun(image, keypoints[[selected_idx]])
    if descriptor is None or len(descriptor) == 0:
        return None

    descriptor_vector = np.asarray(descriptor[0], dtype=np.float64)
    if descriptor_name == "hog":
        descriptor_image = descriptor_vector.reshape(4, 9)
        cmap = "magma"
        descriptor_title = "HOG"
        patch = extract_patch(image, selected_kp)
        support_radius = float(round(5 * selected_kp[2]))
    else:
        descriptor_image = descriptor_vector.reshape(16, 8)
        cmap = "viridis"
        descriptor_title = "SIFT descriptor"
        patch = extract_sift_support_patch(image, selected_kp)
        support_radius = float(np.ceil((2 * np.ceil(3 * selected_kp[2]) + 1) / 2.0))

    return {
        "keypoint": selected_kp,
        "patch": patch,
        "descriptor_image": descriptor_image,
        "cmap": cmap,
        "descriptor_title": descriptor_title,
        "support_radius": support_radius,
    }


def save_combined_descriptor_visualizations(
    sample_images: dict[str, np.ndarray],
    visualization_results: list[tuple[str, str, dict[str, dict[str, object] | None]]],
) -> None:
    for label, image in sample_images.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        colorbar_mappable = None

        for ax, (detector_name, descriptor_name, result_by_label) in zip(
            axes, visualization_results
        ):
            result = result_by_label.get(label)
            if result is None:
                ax.axis("off")
                ax.set_title(f"{detector_name} + {descriptor_name}\nno descriptor")
                continue

            ax.imshow(image, cmap="gray")
            keypoint = result["keypoint"]
            ax.add_patch(
                Circle(
                    (keypoint[0], keypoint[1]),
                    radius=result["support_radius"],
                    edgecolor="lime",
                    fill=False,
                    linewidth=2,
                )
            )

            inset_ax = ax.inset_axes([0.62, 0.05, 0.33, 0.33])
            inset_ax.imshow(result["patch"], cmap="gray")
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.set_title("Patch", fontsize=8)

            descriptor_ax = ax.inset_axes([0.62, 0.46, 0.33, 0.44])
            colorbar_mappable = descriptor_ax.imshow(
                result["descriptor_image"],
                cmap=result["cmap"],
                aspect="auto",
            )
            descriptor_ax.set_title(result["descriptor_title"], fontsize=8)
            descriptor_ax.set_xlabel("Bins", fontsize=8)
            descriptor_ax.set_ylabel("Cells", fontsize=8)
            descriptor_ax.tick_params(labelsize=7)

            ax.set_title(f"{detector_name} + {descriptor_name}")
            ax.axis("off")

        if colorbar_mappable is not None:
            fig.colorbar(colorbar_mappable, ax=axes.tolist(), fraction=0.025, pad=0.02)
        fig.suptitle(f"Descriptor comparison for {label}_001", fontsize=14)
        fig.tight_layout()

        output_path = os.path.join(
            RESULTS_DIR, f"descriptor_visual_all_combinations_{label}.png"
        )
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved combined descriptor visualizations to: {output_path}")


def save_combined_confusion_matrices(
    confusion_results: list[tuple[str, str, list[int], list[int]]],
) -> None:
    class_names = ["person", "car", "bike"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    max_value = 0
    matrices = []

    for detector_name, descriptor_name, y_true, y_pred in confusion_results:
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        matrices.append((detector_name, descriptor_name, matrix))
        if matrix.size > 0:
            max_value = max(max_value, int(matrix.max()))

    for ax, (detector_name, descriptor_name, matrix) in zip(axes, matrices):
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max_value if max_value > 0 else None)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"{detector_name} + {descriptor_name}")

        threshold = matrix.max() / 2 if matrix.size > 0 else 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    str(matrix[i, j]),
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > threshold else "black",
                )

    fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02)
    fig.suptitle("Confusion matrices for all detector/descriptor combinations", fontsize=14)
    fig.tight_layout()
    output_path = os.path.join(RESULTS_DIR, "confusion_matrix_all_combinations.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined confusion matrices to: {output_path}")


def save_failure_analysis_figure(
    detector_name: str,
    descriptor_name: str,
    failure_case,
) -> None:
    if failure_case is None:
        print(f"No misclassification found for {detector_name} + {descriptor_name}.")
        return

    image = failure_case["entry"]["image"]
    result = compute_descriptor_visual(
        image,
        descriptor_name,
        failure_case["detect_fun"],
        failure_case["desc_fun"],
    )
    if result is None:
        print(
            f"Skipping failure analysis figure for {detector_name} + {descriptor_name}: "
            "descriptor visualization failed."
        )
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].imshow(image, cmap="gray")
    keypoint = result["keypoint"]
    axes[0].add_patch(
        Circle(
            (keypoint[0], keypoint[1]),
            radius=result["support_radius"],
            edgecolor="lime",
            fill=False,
            linewidth=2,
        )
    )
    axes[0].set_title(
        f"Original image\ntrue: {failure_case['true_name']} | pred: {failure_case['pred_name']}"
    )
    axes[0].axis("off")

    axes[1].imshow(result["patch"], cmap="gray")
    axes[1].set_title("Patch used for descriptor view")
    axes[1].axis("off")

    im = axes[2].imshow(
        result["descriptor_image"],
        cmap=result["cmap"],
        aspect="auto",
    )
    axes[2].set_title(
        f"{result['descriptor_title']} visualization\n"
        f"wrong-class confidence: {100.0 * failure_case['confidence']:.2f}%"
    )
    axes[2].set_xlabel("Bins")
    axes[2].set_ylabel("Cells")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        (
            f"Failure analysis: {detector_name} + {descriptor_name}\n"
            f"{failure_case['entry']['image_name']} ({failure_case['entry']['class_name']})"
        ),
        fontsize=12,
    )
    fig.tight_layout()
    output_path = os.path.join(
        RESULTS_DIR, f"failure_analysis_{detector_name}_{descriptor_name}.png"
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved failure analysis to: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    os.chdir(ROOT_DIR)
    sample_images = load_sample_images()
    dataset_entries = load_dataset_entries()
    confusion_results = []
    descriptor_visualization_results = []

    detector_configs = [
        (
            "harris_laplacian",
            lambda I: p2.harris_laplacian_corners_detector(
                I, sigma_0=2.0, rho_0=2.5, s=1.5, N=4, k=0.05, theta_corn=0.005
            ),
        ),
        (
            "hessian_laplacian",
            lambda I: p2.hessian_laplacian_blobs_detector(
                I, sigma_0=2.0, s=1.5, N=4, theta_blob=0.005
            ),
        ),
    ]

    descriptor_configs = [
        ("surf", lambda I, kp: p3.featuresSURF(I, kp)),
        ("hog", lambda I, kp: p3.featuresHOG(I, kp)),
    ]

    for detector_name, detect_fun in detector_configs:
        for descriptor_name, desc_fun in descriptor_configs:
            save_path = os.path.join(
                RESULTS_DIR,
                f"features_{detector_name}_{descriptor_name}.pkl",
            )

            if os.path.exists(save_path) and not args.recompute:
                print(f"\nLoading cached features for {detector_name} + {descriptor_name}")
                feats = p3.FeatureExtraction(
                    detector_fun=detect_fun,
                    descriptor_fun=desc_fun,
                    loadFile=save_path,
                )
                extraction_time_ms = None
            else:
                print(f"\nExtracting features with {detector_name} + {descriptor_name}")
                start_time = time.perf_counter()
                feats = p3.FeatureExtraction(
                    detector_fun=detect_fun,
                    descriptor_fun=desc_fun,
                    saveFile=save_path,
                )
                extraction_time_ms = (time.perf_counter() - start_time) * 1000.0

            num_images, num_descriptors = summarize_features(feats)
            print(f"Feature cache: {save_path}")
            print(f"Images processed: {num_images}")
            print(f"Total descriptors: {num_descriptors}")
            if extraction_time_ms is None:
                print("Feature extraction time: skipped (loaded from cache)")
            else:
                avg_time_per_image_ms = extraction_time_ms / num_images
                print(f"Feature extraction time: {extraction_time_ms:.2f} ms total")
                print(
                    f"Average feature extraction time: "
                    f"{avg_time_per_image_ms:.2f} ms/image"
                )

            result_by_label = {}
            for label, image in sample_images.items():
                result_by_label[label] = compute_descriptor_visual(
                    image,
                    descriptor_name,
                    detect_fun,
                    desc_fun,
                )
            descriptor_visualization_results.append(
                (detector_name, descriptor_name, result_by_label)
            )

            if not os.path.exists(FOLD_INDICES_PATH):
                print(f"Skipping train/test split because {FOLD_INDICES_PATH} was not found.")
                continue

            accs = []
            all_true = []
            all_pred = []
            best_failure = None
            for k in range(5):
                data_train, label_train, data_test, label_test = p3.createTrainTest(feats, k)
                test_entries = get_test_entries_for_fold(dataset_entries, k)
                print(f"Fold {k}")
                summarize_split(data_train, label_train, data_test, label_test)

                BOF_tr, BOF_ts = p3.BagOfWords(data_train, data_test)
                acc, preds, probas = p3.svm(BOF_tr, label_train, BOF_ts, label_test)
                accs.append(acc)
                all_true.extend(label_test)
                all_pred.extend(preds)

                for sample_idx, (true_label, pred_label, probs) in enumerate(
                    zip(label_test, preds, probas)
                ):
                    if pred_label == true_label:
                        continue
                    failure_case = {
                        "entry": test_entries[sample_idx],
                        "true_label": int(true_label),
                        "pred_label": int(pred_label),
                        "true_name": CLASS_INFO[int(true_label)][0],
                        "pred_name": CLASS_INFO[int(pred_label)][0],
                        "confidence": float(probs[int(pred_label)]),
                        "detect_fun": detect_fun,
                        "desc_fun": desc_fun,
                    }
                    if (
                        best_failure is None
                        or failure_case["confidence"] > best_failure["confidence"]
                    ):
                        best_failure = failure_case

            print(
                f"Mean accuracy for {detector_name} with {descriptor_name}: "
                f"{100.0 * np.mean(accs):.3f}%"
            )
            save_failure_analysis_figure(
                detector_name,
                descriptor_name,
                best_failure,
            )
            confusion_results.append(
                (detector_name, descriptor_name, all_true, all_pred)
            )

    save_combined_descriptor_visualizations(
        sample_images,
        descriptor_visualization_results,
    )
    save_combined_confusion_matrices(confusion_results)
