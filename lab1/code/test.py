import os
import argparse
import numpy as np
import cv26_lab1_part3_utils as p3
import part2 as p2


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "part3")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run part 3.2.5 local-feature robustness experiments."
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore cached feature files and extract features again.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.chdir(ROOT_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    detector_configs = [
        (
            "harris-laplacian",
            lambda image: p2.harris_laplacian_corners_detector(
                image,
                sigma_0=2.0,
                rho_0=2.5,
                s=1.5,
                N=4,
                k=0.05,
                theta_corn=0.005,
            ),
        ),
        (
            "hessian-laplacian",
            lambda image: p2.hessian_laplacian_blobs_detector(
                image,
                sigma_0=2.0,
                s=1.5,
                N=4,
                theta_blob=0.005,
            ),
        ),
    ]

    def surf_descriptors(image, keypoints):
        if keypoints is None or len(keypoints) == 0:
            return np.zeros((1, 128), dtype=np.float32)
        descriptors = p3.featuresSURF(image, keypoints)
        if descriptors is None or len(descriptors) == 0:
            return np.zeros((1, 128), dtype=np.float32)
        return np.asarray(descriptors, dtype=np.float32)

    def hog_descriptors(image, keypoints):
        if keypoints is None or len(keypoints) == 0:
            return np.zeros((1, 36), dtype=np.float32)
        descriptors = p3.featuresHOG(image, keypoints)
        if descriptors is None or len(descriptors) == 0:
            return np.zeros((1, 36), dtype=np.float32)
        return np.asarray(descriptors, dtype=np.float32)

    descriptor_configs = [
        ("surf", surf_descriptors),
        ("hog", hog_descriptors),
    ]

    augmentation_configs = [
        ("gaussian_noise", "gaussian-noise"),
        ("random_rotation", "random-rotation"),
    ]

    summary = {}
    for distortion_mode, report_name in augmentation_configs:
        print(f"\n=== Part 3.2.5: {report_name}, s=1.5 ===")
        summary[report_name] = {}

        for detector_name, detector_fun in detector_configs:
            for descriptor_name, descriptor_fun in descriptor_configs:
                save_path = os.path.join(
                    RESULTS_DIR,
                    f"features_{detector_name}_{descriptor_name}_{distortion_mode}_s1p5.pkl",
                )

                if os.path.exists(save_path) and not args.recompute:
                    print(
                        f"Loading cached features for "
                        f"{detector_name} + {descriptor_name} with {report_name}"
                    )
                    features = p3.FeatureExtraction(
                        detector_fun,
                        descriptor_fun,
                        loadFile=save_path,
                    )
                else:
                    print(
                        f"Extracting features for "
                        f"{detector_name} + {descriptor_name} with {report_name}"
                    )
                    features = p3.FeatureExtraction(
                        detector_fun,
                        descriptor_fun,
                        saveFile=save_path,
                        distort=distortion_mode,
                    )

                accuracies = []
                for k in range(5):
                    data_train, label_train, data_test, label_test = p3.createTrainTest(
                        features, k
                    )
                    bof_train, bof_test = p3.BagOfWords(data_train, data_test)
                    accuracy, _, _ = p3.svm(
                        bof_train, label_train, bof_test, label_test
                    )
                    accuracies.append(float(accuracy))

                mean_accuracy = float(np.mean(accuracies))
                std_accuracy = float(np.std(accuracies))
                print(
                    f"{report_name} | {detector_name} + {descriptor_name}: "
                    f"{100.0 * mean_accuracy:.3f}% (+/- {100.0 * std_accuracy:.3f}%)"
                )
                summary[report_name][f"{detector_name} + {descriptor_name}"] = mean_accuracy

    print("\n=== Summary ===")
    for augmentation_name, results in summary.items():
        print(augmentation_name)
        for combo_name, accuracy in results.items():
            print(f"  {combo_name}: {100.0 * accuracy:.3f}%")
