import argparse
import json
import os
import time

import cv26_lab1_part3_utils as p3
import matplotlib.pyplot as plt
import numpy as np
import part2 as p2
from sklearn.metrics import confusion_matrix


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "part3")
ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
FOLD_INDICES_PATH = os.path.join(ROOT_DIR, "Fold_Indices.mat")
S_SWEEP_VALUES = [1.0, 1.2, 1.5, 1.8, 2.0]
TIME_CACHE_PATH = os.path.join(RESULTS_DIR, "extraction_time_cache.json")
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run part 3.1 feature extraction and accuracy evaluation."
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore cached feature files and extract features again.",
    )
    return parser.parse_args()


def load_time_cache() -> dict[str, float]:
    if not os.path.exists(TIME_CACHE_PATH):
        return {}

    with open(TIME_CACHE_PATH, "r", encoding="ascii") as f:
        data = json.load(f)

    return {str(key): float(value) for key, value in data.items()}


def save_time_cache(time_cache: dict[str, float]) -> None:
    with open(TIME_CACHE_PATH, "w", encoding="ascii") as f:
        json.dump(time_cache, f, indent=2, sort_keys=True)


def plot_confusion_matrices_for_s(
    s_value: float,
    confusion_results: dict[str, np.ndarray],
    class_labels: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    for ax, (combo_name, conf_mat) in zip(axes.flat, confusion_results.items()):
        image = ax.imshow(conf_mat, cmap="Blues")
        ax.set_title(combo_name)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks(range(len(class_labels)))
        ax.set_yticks(range(len(class_labels)))
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)

        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                ax.text(j, i, int(conf_mat[i, j]), ha="center", va="center", color="black")

    fig.colorbar(image, ax=axes, shrink=0.8)
    fig.suptitle(f"Confusion Matrices for s = {s_value:.1f}")
    plot_path = os.path.join(
        RESULTS_DIR,
        f"confusion_matrices_s{str(s_value).replace('.', 'p')}.png",
    )
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    os.chdir(ROOT_DIR)

    if not os.path.exists(FOLD_INDICES_PATH):
        raise FileNotFoundError(
            f"Expected fold splits at {FOLD_INDICES_PATH}. "
            "part3_1.py follows createTrainTest() from cv26_lab1_part3_utils.py, "
            "which loads ./Fold_Indices.mat from the project root."
        )

    descriptor_configs = [
        ("surf", lambda I, kp: p3.featuresSURF(I, kp)),
        ("hog", lambda I, kp: p3.featuresHOG(I, kp)),
    ]

    sweep_results = {
        "harris_laplacian + surf": [],
        "harris_laplacian + hog": [],
        "hessian_laplacian + surf": [],
        "hessian_laplacian + hog": [],
        "late fusion: harris_surf + hessian_surf": [],
        "late fusion: harris_hog + hessian_surf": [],
        "late fusion: hessian_surf + hessian_hog": [],
    }
    extraction_time_results = {
        "harris_laplacian + surf": [],
        "harris_laplacian + hog": [],
        "hessian_laplacian + surf": [],
        "hessian_laplacian + hog": [],
        "late fusion: harris_surf + hessian_surf": [],
        "late fusion: harris_hog + hessian_surf": [],
        "late fusion: hessian_surf + hessian_hog": [],
    }
    time_cache = load_time_cache()

    for s_value in S_SWEEP_VALUES:
        print(f"\n=== Sweep for s = {s_value:.1f} ===")
        confusion_results = {}
        class_labels = None
        fusion_harris_surf_features = None
        fusion_harris_hog_features = None
        fusion_hessian_surf_features = None
        fusion_hessian_hog_features = None

        detector_configs = [
            (
                "harris_laplacian",
                lambda I, s_value=s_value: p2.harris_laplacian_corners_detector(
                    I, sigma_0=2.0, rho_0=2.5, s=s_value, N=4, k=0.05, theta_corn=0.005
                ),
            ),
            (
                "hessian_laplacian",
                lambda I, s_value=s_value: p2.hessian_laplacian_blobs_detector(
                    I, sigma_0=2.0, s=s_value, N=4, theta_blob=0.005
                ),
            ),
        ]

        for detector_name, detect_fun in detector_configs:
            for descriptor_name, desc_fun in descriptor_configs:
                save_path = os.path.join(
                    RESULTS_DIR,
                    f"features_{detector_name}_{descriptor_name}_s{str(s_value).replace('.', 'p')}.pkl",
                )

                if os.path.exists(save_path) and not args.recompute:
                    print(
                        f"\nLoading cached features for "
                        f"{detector_name} + {descriptor_name} at s={s_value:.1f}"
                    )
                    feats = p3.FeatureExtraction(
                        detect_fun,
                        desc_fun,
                        loadFile=save_path,
                    )
                    extraction_time_s = None
                else:
                    print(
                        f"\nExtracting features with "
                        f"{detector_name} + {descriptor_name} at s={s_value:.1f}"
                    )
                    start_time = time.perf_counter()
                    feats = p3.FeatureExtraction(
                        detect_fun,
                        desc_fun,
                        saveFile=save_path,
                    )
                    extraction_time_s = time.perf_counter() - start_time

                combo_name = f"{detector_name} + {descriptor_name}"
                time_cache_key = f"{detector_name}|{descriptor_name}|{s_value:.1f}"
                if extraction_time_s is not None:
                    extraction_time_value = float(extraction_time_s)
                    time_cache[time_cache_key] = extraction_time_value
                    save_time_cache(time_cache)
                else:
                    extraction_time_value = time_cache.get(time_cache_key, np.nan)
                extraction_time_results[combo_name].append(extraction_time_value)
                if np.isnan(extraction_time_value):
                    print(f"Feature extraction time for {combo_name}: unavailable")
                elif extraction_time_s is None:
                    print(
                        f"Feature extraction time for {combo_name}: "
                        f"{extraction_time_value:.2f} s (cached)"
                    )
                else:
                    print(
                        f"Feature extraction time for {combo_name}: "
                        f"{extraction_time_s:.2f} s"
                    )

                accs = []
                combo_confusion = None
                for k in range(5):
                    data_train, label_train, data_test, label_test = p3.createTrainTest(feats, k)
                    if class_labels is None:
                        class_labels = np.unique(np.concatenate((label_train, label_test)))

                    bof_train, bof_test = p3.BagOfWords(data_train, data_test)
                    acc, predictions, _ = p3.svm(
                        bof_train, label_train, bof_test, label_test
                    )
                    accs.append(acc)
                    fold_confusion = confusion_matrix(
                        label_test, predictions, labels=class_labels
                    )
                    if combo_confusion is None:
                        combo_confusion = fold_confusion
                    else:
                        combo_confusion += fold_confusion

                mean_acc = float(np.mean(accs))
                sweep_results[combo_name].append(mean_acc)
                confusion_results[combo_name] = combo_confusion
                print(
                    f"Mean accuracy for {detector_name} with {descriptor_name} "
                    f"at s={s_value:.1f}: {100.0 * mean_acc:.3f}%"
                )

                if detector_name == "harris_laplacian" and descriptor_name == "surf":
                    fusion_harris_surf_features = feats
                elif detector_name == "harris_laplacian" and descriptor_name == "hog":
                    fusion_harris_hog_features = feats
                elif detector_name == "hessian_laplacian" and descriptor_name == "surf":
                    fusion_hessian_surf_features = feats
                elif detector_name == "hessian_laplacian" and descriptor_name == "hog":
                    fusion_hessian_hog_features = feats

        if (
            fusion_harris_surf_features is not None
            and fusion_hessian_surf_features is not None
        ):
            accs = []

            for k in range(5):
                harris_surf_train, label_train, harris_surf_test, label_test = p3.createTrainTest(
                    fusion_harris_surf_features, k
                )
                hessian_surf_train, label_train_2, hessian_surf_test, label_test_2 = p3.createTrainTest(
                    fusion_hessian_surf_features, k
                )

                if label_train != label_train_2 or label_test != label_test_2:
                    raise ValueError("Late fusion requires identical train/test splits.")

                bof_train_harris_surf, bof_test_harris_surf = p3.BagOfWords(
                    harris_surf_train, harris_surf_test
                )
                bof_train_hessian_surf, bof_test_hessian_surf = p3.BagOfWords(
                    hessian_surf_train, hessian_surf_test
                )

                bof_train_fused = np.hstack(
                    (bof_train_harris_surf, bof_train_hessian_surf)
                )
                bof_test_fused = np.hstack(
                    (bof_test_harris_surf, bof_test_hessian_surf)
                )

                acc, _, _ = p3.svm(
                    bof_train_fused, label_train, bof_test_fused, label_test
                )
                accs.append(acc)

            mean_acc = float(np.mean(accs))
            fusion_name = "late fusion: harris_surf + hessian_surf"
            sweep_results[fusion_name].append(mean_acc)
            harris_surf_time = extraction_time_results["harris_laplacian + surf"][-1]
            hessian_surf_time = extraction_time_results["hessian_laplacian + surf"][-1]
            if np.isnan(harris_surf_time) or np.isnan(hessian_surf_time):
                fusion_time = np.nan
            else:
                fusion_time = harris_surf_time + hessian_surf_time
            extraction_time_results[fusion_name].append(fusion_time)
            print(
                "Mean accuracy for late fusion "
                "(harris_laplacian + surf) + (hessian_laplacian + surf) "
                f"at s={s_value:.1f}: {100.0 * mean_acc:.3f}%"
            )
            if np.isnan(fusion_time):
                print(f"Feature extraction time for {fusion_name}: unavailable")
            else:
                print(
                    f"Feature extraction time for {fusion_name}: "
                    f"{fusion_time:.2f} s"
                )
        else:
            sweep_results["late fusion: harris_surf + hessian_surf"].append(np.nan)
            extraction_time_results["late fusion: harris_surf + hessian_surf"].append(np.nan)

        if (
            fusion_harris_hog_features is not None
            and fusion_hessian_surf_features is not None
        ):
            accs = []

            for k in range(5):
                harris_hog_train, label_train, harris_hog_test, label_test = p3.createTrainTest(
                    fusion_harris_hog_features, k
                )
                hessian_surf_train, label_train_2, hessian_surf_test, label_test_2 = p3.createTrainTest(
                    fusion_hessian_surf_features, k
                )

                if label_train != label_train_2 or label_test != label_test_2:
                    raise ValueError("Late fusion requires identical train/test splits.")

                bof_train_harris_hog, bof_test_harris_hog = p3.BagOfWords(
                    harris_hog_train, harris_hog_test
                )
                bof_train_hessian_surf, bof_test_hessian_surf = p3.BagOfWords(
                    hessian_surf_train, hessian_surf_test
                )

                bof_train_fused = np.hstack(
                    (bof_train_harris_hog, bof_train_hessian_surf)
                )
                bof_test_fused = np.hstack(
                    (bof_test_harris_hog, bof_test_hessian_surf)
                )

                acc, _, _ = p3.svm(
                    bof_train_fused, label_train, bof_test_fused, label_test
                )
                accs.append(acc)

            mean_acc = float(np.mean(accs))
            fusion_name = "late fusion: harris_hog + hessian_surf"
            sweep_results[fusion_name].append(mean_acc)
            harris_hog_time = extraction_time_results["harris_laplacian + hog"][-1]
            hessian_surf_time = extraction_time_results["hessian_laplacian + surf"][-1]
            if np.isnan(harris_hog_time) or np.isnan(hessian_surf_time):
                fusion_time = np.nan
            else:
                fusion_time = harris_hog_time + hessian_surf_time
            extraction_time_results[fusion_name].append(fusion_time)
            print(
                "Mean accuracy for late fusion "
                "(harris_laplacian + hog) + (hessian_laplacian + surf) "
                f"at s={s_value:.1f}: {100.0 * mean_acc:.3f}%"
            )
            if np.isnan(fusion_time):
                print(f"Feature extraction time for {fusion_name}: unavailable")
            else:
                print(
                    f"Feature extraction time for {fusion_name}: "
                    f"{fusion_time:.2f} s"
                )
        else:
            sweep_results["late fusion: harris_hog + hessian_surf"].append(np.nan)
            extraction_time_results["late fusion: harris_hog + hessian_surf"].append(np.nan)

        if (
            fusion_hessian_hog_features is not None
            and fusion_hessian_surf_features is not None
        ):
            accs = []

            for k in range(5):
                hessian_surf_train, label_train, hessian_surf_test, label_test = p3.createTrainTest(
                    fusion_hessian_surf_features, k
                )
                hessian_hog_train, label_train_2, hessian_hog_test, label_test_2 = p3.createTrainTest(
                    fusion_hessian_hog_features, k
                )

                if label_train != label_train_2 or label_test != label_test_2:
                    raise ValueError("Late fusion requires identical train/test splits.")

                bof_train_hessian_surf, bof_test_hessian_surf = p3.BagOfWords(
                    hessian_surf_train, hessian_surf_test
                )
                bof_train_hessian_hog, bof_test_hessian_hog = p3.BagOfWords(
                    hessian_hog_train, hessian_hog_test
                )

                bof_train_fused = np.hstack(
                    (bof_train_hessian_surf, bof_train_hessian_hog)
                )
                bof_test_fused = np.hstack(
                    (bof_test_hessian_surf, bof_test_hessian_hog)
                )

                acc, _, _ = p3.svm(
                    bof_train_fused, label_train, bof_test_fused, label_test
                )
                accs.append(acc)

            mean_acc = float(np.mean(accs))
            fusion_name = "late fusion: hessian_surf + hessian_hog"
            sweep_results[fusion_name].append(mean_acc)
            hessian_surf_time = extraction_time_results["hessian_laplacian + surf"][-1]
            hessian_hog_time = extraction_time_results["hessian_laplacian + hog"][-1]
            if np.isnan(hessian_surf_time) or np.isnan(hessian_hog_time):
                fusion_time = np.nan
            else:
                fusion_time = hessian_surf_time + hessian_hog_time
            extraction_time_results[fusion_name].append(fusion_time)
            print(
                "Mean accuracy for late fusion "
                "(hessian_laplacian + surf) + (hessian_laplacian + hog) "
                f"at s={s_value:.1f}: {100.0 * mean_acc:.3f}%"
            )
            if np.isnan(fusion_time):
                print(f"Feature extraction time for {fusion_name}: unavailable")
            else:
                print(
                    f"Feature extraction time for {fusion_name}: "
                    f"{fusion_time:.2f} s"
                )
        else:
            sweep_results["late fusion: hessian_surf + hessian_hog"].append(np.nan)
            extraction_time_results["late fusion: hessian_surf + hessian_hog"].append(np.nan)

        if class_labels is not None:
            plot_confusion_matrices_for_s(s_value, confusion_results, class_labels)

    fig, ax = plt.subplots(figsize=(9, 6))
    for combo_name, accuracies in sweep_results.items():
        ax.plot(S_SWEEP_VALUES, accuracies, marker="o", linewidth=2, label=combo_name)

    ax.set_xlabel("Scale ratio s")
    ax.set_ylabel("Mean 5-fold accuracy")
    ax.set_title("Accuracy sweep over s")
    ax.set_xticks(S_SWEEP_VALUES)
    ax.set_ylim(0.5, 0.8)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "accuracy_vs_s_all_combinations.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved accuracy sweep plot to: {plot_path}")

    fig, ax = plt.subplots(figsize=(9, 6))
    for combo_name, extraction_times in extraction_time_results.items():
        ax.plot(S_SWEEP_VALUES, extraction_times, marker="o", linewidth=2, label=combo_name)

    ax.set_xlabel("Scale ratio s")
    ax.set_ylabel("Feature extraction time (s)")
    ax.set_title("Feature extraction time sweep over s")
    ax.set_xticks(S_SWEEP_VALUES)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "extraction_time_vs_s_all_combinations.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved extraction time sweep plot to: {plot_path}")
