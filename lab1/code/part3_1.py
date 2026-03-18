import os

import cv26_lab1_part3_utils as p3
import numpy as np
import part2 as p2


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "part3")
ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
FOLD_INDICES_PATH = os.path.join(ROOT_DIR, "Fold_Indices.mat")
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


if __name__ == "__main__":
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

            print(f"\nExtracting features with {detector_name} + {descriptor_name}")
            feats = p3.FeatureExtraction(
                detector_fun=detect_fun,
                descriptor_fun=desc_fun,
                saveFile=save_path,
            )

            num_images, num_descriptors = summarize_features(feats)
            print(f"Saved to: {save_path}")
            print(f"Images processed: {num_images}")
            print(f"Total descriptors: {num_descriptors}")

            if not os.path.exists(FOLD_INDICES_PATH):
                print(f"Skipping train/test split because {FOLD_INDICES_PATH} was not found.")
                continue

            accs = []
            for k in range(5):
                data_train, label_train, data_test, label_test = p3.createTrainTest(feats, k)
                print(f"Fold {k}")
                summarize_split(data_train, label_train, data_test, label_test)

                BOF_tr, BOF_ts = p3.BagOfWords(data_train, data_test)
                acc, preds, probas = p3.svm(BOF_tr, label_train, BOF_ts, label_test)
                accs.append(acc)

            print(
                f"Mean accuracy for {detector_name} with {descriptor_name}: "
                f"{100.0 * np.mean(accs):.3f}%"
            )
