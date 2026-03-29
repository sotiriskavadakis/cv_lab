import argparse
import json
import os
import time
#load the needed given classes and functions from the utils file for part 3 
#note that i have added some functions to the utils file for distortimage for 3.2.5 and genrally added random seed (42) for reproducability for every randomness in the given functions
import cv26_lab1_part3_utils as p3
import matplotlib.pyplot as plt
import numpy as np
#import part 2 for the detectors used in part 3.1 and part 3.2.5
import part2 as p2
from sklearn.metrics import confusion_matrix

#This file contains the code fr 3.1 and 3.2.5 for the local descriptors of 3.1


#directories and paths for loading and saving the results and features for part 3.1 and part 3.2.5
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "part3")
ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
FOLD_INDICES_PATH = os.path.join(ROOT_DIR, "Fold_Indices.mat")
S_SWEEP_VALUES = [1.0, 1.2, 1.5, 1.8, 2.0]
TIME_CACHE_PATH = os.path.join(RESULTS_DIR, "extraction_time_cache.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

#this functions helps to parse the arguments for the script, whether to run part 3.1 or part 3.2.5 and whether to recompute the features or load them from cache if they exist
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run either Part 3.1 or Part 3.2.5 local-feature experiments."
    )
    #run 3.1 or 3.2.5
    parser.add_argument(
        "--mode",
        choices=["part3_1", "part3_2_5"],
        default="part3_1",
        help="Choose which experiment block to run.",
    )
    #whether to recompute the features or load them from cache if they exist
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore cached feature files and extract features again.",
    )
    return parser.parse_args()

#in order to run 3.1: run the script with the command line argument --mode part3_1 and for 3.2.5 run it with --mode part3_2_5
#for both 3.1 and 3.2.5 if you want to recompute the features instead of loading them from cache if they exist run the script with the command line argument run with --recompute
 #python3 part3_1.py  runs part 3.1  as defaultwith loading features from cache if they exist and no recomputation                                                                                                                                                             
 # python3 part3_1.py --mode part3_1  runs part 3.1 with loading features from cache if they exist and no recomputation                                                                                                                                              
  #python3 part3_1.py --mode part3_1 --recompute runs part 3.1 with recomputation of features and no loading from cache                                                                                                                            
  #python3 part3_1.py --mode part3_2_5  runs part 3.2.5 with loading features from cache if they exist and no recomputation                                                                                                                                  
  #python3 part3_1.py --mode part3_2_5 --recompute runs part 3.2.5 with recomputation of features and no loading from cache


#function to load the extraction time cache from a json file where they are stored if they are stored and norecomputation is asked for 3.1
def load_time_cache() -> dict[str, float]:
    if not os.path.exists(TIME_CACHE_PATH):
        return {}

    with open(TIME_CACHE_PATH, "r", encoding="ascii") as f:
        data = json.load(f)

    return {str(key): float(value) for key, value in data.items()}

#function to save feature extraction time into json file for 3.1
def save_time_cache(time_cache: dict[str, float]) -> None:
    with open(TIME_CACHE_PATH, "w", encoding="ascii") as f:
        json.dump(time_cache, f, indent=2, sort_keys=True)

#function to create confusion matrices for 3.1
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

#trivial function for 3.2.5 in case distortion causes no  keypoints to be detected or no descriptors to be computed so no error shuts the process
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


if __name__ == "__main__":
    args = parse_args()#call the argument parsing function from the cli command line arguments
    os.chdir(ROOT_DIR)
    #createtraintest function for 3.1 and 3.2.5 expects the foldinidices mat into a specific path so ckech you have the right path while running
    if not os.path.exists(FOLD_INDICES_PATH):
        raise FileNotFoundError(
            f"Expected fold splits at {FOLD_INDICES_PATH}. "
            "Please ensure you have the correct directory structure and files."
        )
    #case 1 we call part3_1 to run
    if args.mode == "part3_1":
        #possible descriptor functions for 3.1
        descriptor_configs = [
            ("surf", lambda I, kp: p3.featuresSURF(I, kp)),
            ("hog", lambda I, kp: p3.featuresHOG(I, kp)),
        ]
        #results for all combunations tested for 3.1 for accuracy storing and time storing for the sweep of s values for 3.1
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
        #sweep for different values of s for all combinations
        for s_value in S_SWEEP_VALUES:
            print(f"\n=== Sweep for s = {s_value:.1f} ===")
            confusion_results = {}
            class_labels = None
            fusion_harris_surf_features = None
            fusion_harris_hog_features = None
            fusion_hessian_surf_features = None
            fusion_hessian_hog_features = None
             #define the two detecotors from the helping functions in part2 for 3.1 with the current s value in the sweep
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
            #for each s calue and for each combination we find the mean accuracy and extraction time
            #this is for the simple combinations
            for detector_name, detect_fun in detector_configs:
                for descriptor_name, desc_fun in descriptor_configs:
                    save_path = os.path.join(
                        RESULTS_DIR,
                        f"features_{detector_name}_{descriptor_name}_s{str(s_value).replace('.', 'p')}.pkl",
                    )
                    #depending on the argument into the cli command we either load the features from cache if they exist , or we commadn the feature extraction prosecc to run again and store the new results
                    if os.path.exists(save_path) and not args.recompute:
                        print(
                            f"\nLoading cached features for "
                            f"{detector_name} + {descriptor_name} at s={s_value:.1f}"
                        )
                        feats = p3.FeatureExtraction(
                            detect_fun,
                            desc_fun,
                            loadFile=save_path, #load the features from cache if they exist and no recomputation is asked for
                        )
                        extraction_time_s = None
                    else:
                        #cfeature extraction method for a combination using the featureextraction functio from the utils
                        #with rguments the detecotr function , the descriptor function and the save path 
                        print(
                            f"\nExtracting features with "
                            f"{detector_name} + {descriptor_name} at s={s_value:.1f}"
                        )
                        start_time = time.perf_counter()
                        feats = p3.FeatureExtraction(
                            detect_fun,
                            desc_fun,
                            saveFile=save_path, #save the features to cache after extraction for future use if  recomputation is asked for
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
                    #k-fold cross valiation with k=5 with helping function createtraintest
                    for k in range(5):
                        data_train, label_train, data_test, label_test = p3.createTrainTest(
                            feats, k
                        )
                        if class_labels is None:
                            class_labels = np.unique(np.concatenate((label_train, label_test)))
                        #create the optical lexicon based on the features of the training set and then represent both training feats and test feats into histograms of words of the lexicon
                        bof_train, bof_test = p3.BagOfWords(data_train, data_test)
                        #for every fold compute the accuracy
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
                    #compute mean accuracy for this combination
                    mean_acc = float(np.mean(accs))
                    sweep_results[combo_name].append(mean_acc)
                    confusion_results[combo_name] = combo_confusion
                    print(
                        f"Mean accuracy for {detector_name} with {descriptor_name} "
                        f"at s={s_value:.1f}: {100.0 * mean_acc:.3f}%"
                    )
                 #after having computed the simple combinations for a s value, we will concatenate the different features already computed,for the late fusion methods 
                    if detector_name == "harris_laplacian" and descriptor_name == "surf":
                        fusion_harris_surf_features = feats
                    elif detector_name == "harris_laplacian" and descriptor_name == "hog":
                        fusion_harris_hog_features = feats
                    elif detector_name == "hessian_laplacian" and descriptor_name == "surf":
                        fusion_hessian_surf_features = feats
                    elif detector_name == "hessian_laplacian" and descriptor_name == "hog":
                        fusion_hessian_hog_features = feats

            fusion_configs = [ #the three late fusion combinations we will test for 3.1 by concatenating the features of the simple combinations already computed
                (
                    "late fusion: harris_surf + hessian_surf",
                    fusion_harris_surf_features,
                    fusion_hessian_surf_features,
                    "harris_laplacian + surf",
                    "hessian_laplacian + surf",
                    "(harris_laplacian + surf) + (hessian_laplacian + surf)",
                ),
                (
                    "late fusion: harris_hog + hessian_surf",
                    fusion_harris_hog_features,
                    fusion_hessian_surf_features,
                    "harris_laplacian + hog",
                    "hessian_laplacian + surf",
                    "(harris_laplacian + hog) + (hessian_laplacian + surf)",
                ),
                (
                    "late fusion: hessian_surf + hessian_hog",
                    fusion_hessian_surf_features,
                    fusion_hessian_hog_features,
                    "hessian_laplacian + surf",
                    "hessian_laplacian + hog",
                    "(hessian_laplacian + surf) + (hessian_laplacian + hog)",
                ),
            ]

            for fusion_name, first_features, second_features, first_key, second_key, printed_name in fusion_configs:
                if first_features is None or second_features is None:
                    sweep_results[fusion_name].append(np.nan)
                    extraction_time_results[fusion_name].append(np.nan)
                    continue

                accs = []
                for k in range(5):
                    first_train, label_train, first_test, label_test = p3.createTrainTest(
                        first_features, k
                    )
                    second_train, label_train_2, second_test, label_test_2 = p3.createTrainTest(
                        second_features, k
                    )

                    if label_train != label_train_2 or label_test != label_test_2:
                        raise ValueError("Late fusion requires identical train/test splits.")
                     #from the already feats computed from the simple combinations we represent them with the bag of words 
                    bof_train_first, bof_test_first = p3.BagOfWords(first_train, first_test)
                    bof_train_second, bof_test_second = p3.BagOfWords(second_train, second_test)
                     #then we concateate the representations of the two combinations to have a fused representation for both train and test sets with th np.hstack function
                    bof_train_fused = np.hstack((bof_train_first, bof_train_second))
                    bof_test_fused = np.hstack((bof_test_first, bof_test_second))
                        #then we compute the accuracy for this fold with the fused features
                    acc, _, _ = p3.svm(
                        bof_train_fused, label_train, bof_test_fused, label_test
                    )
                    accs.append(acc)
                #and mean accuracy
                mean_acc = float(np.mean(accs))
                sweep_results[fusion_name].append(mean_acc)
                first_time = extraction_time_results[first_key][-1]
                second_time = extraction_time_results[second_key][-1]
                if np.isnan(first_time) or np.isnan(second_time):
                    fusion_time = np.nan
                else:
                    fusion_time = first_time + second_time
                extraction_time_results[fusion_name].append(fusion_time)
                print(
                    f"Mean accuracy for late fusion {printed_name} "
                    f"at s={s_value:.1f}: {100.0 * mean_acc:.3f}%"
                )
                if np.isnan(fusion_time):
                    print(f"Feature extraction time for {fusion_name}: unavailable")
                else:
                    print(f"Feature extraction time for {fusion_name}: {fusion_time:.2f} s")

            if class_labels is not None:
                plot_confusion_matrices_for_s(s_value, confusion_results, class_labels)
        #create the sweep plots for accuracy and extraction time for all combinations tested in 3.1
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


        
    #case 2 we call part3_2_5 to run 
    else:
        #again we define the two detectors for 3.2.5 with the helping functions in part2 but with fixed s value of 1.5 since this was the most robust value
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

        descriptor_configs = [
            ("surf", surf_descriptors),
            ("hog", hog_descriptors),
        ]
        #types of distortions we will test for 3.2.5
        augmentation_configs = [
            ("gaussian_noise", "gaussian-noise"),
            ("random_rotation", "random-rotation"),
        ]

        summary = {}
        #we ru for all combinations of distortion , detectors ,and descriptors
        for distortion_mode, report_name in augmentation_configs:
            print(f"\n=== Part 3.2.5: {report_name}, s=1.5 ===")
            summary[report_name] = {}
            fusion_harris_surf_features = None
            fusion_harris_hog_features = None
            fusion_hessian_surf_features = None
            fusion_hessian_hog_features = None
            #for the simple combinations
            for detector_name, detector_fun in detector_configs:
                for descriptor_name, descriptor_fun in descriptor_configs:
                    save_path = os.path.join(
                        RESULTS_DIR,
                        f"features_{detector_name}_{descriptor_name}_{distortion_mode}_s1p5.pkl",
                    )
                    #again we use the same recompute logic to either load the existing featurs or recompute them and save them to cache for future use depending on the cli arguments
                    if os.path.exists(save_path) and not args.recompute:
                        print(
                            f"Loading cached features for "
                            f"{detector_name} + {descriptor_name} with {report_name}"
                        )
                        features = p3.FeatureExtraction(
                            detector_fun,
                            descriptor_fun,
                            loadFile=save_path, #if we load them we do not need the distortion argument
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
                            distort=distortion_mode, #if we ask to recompute the features we need to pass the distortion mode as argument into the given function to apply the distortion to the images during feature extraction
                        #we set the distortion mode to true and this calls the distorte image function from file cv26_lab1_part3_utils which applies the distortion to the images during feature extraction depending on the mode provided as an argument , gaussian noise or random rotation
                        )
                        
                    #again k fold validation and mean accuracy
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
                        #store the features for late fusion combinations after having computed the simple ones for the current distortion mode and s value of 1.5
                    if detector_name == "harris-laplacian" and descriptor_name == "surf":
                        fusion_harris_surf_features = features
                    elif detector_name == "harris-laplacian" and descriptor_name == "hog":
                        fusion_harris_hog_features = features
                    elif detector_name == "hessian-laplacian" and descriptor_name == "surf":
                        fusion_hessian_surf_features = features
                    elif detector_name == "hessian-laplacian" and descriptor_name == "hog":
                        fusion_hessian_hog_features = features

            fusion_configs = [
                (
                    "late fusion: harris_surf + hessian_surf",
                    fusion_harris_surf_features,
                    fusion_hessian_surf_features,
                ),
                (
                    "late fusion: harris_hog + hessian_surf",
                    fusion_harris_hog_features,
                    fusion_hessian_surf_features,
                ),
                (
                    "late fusion: hessian_surf + hessian_hog",
                    fusion_hessian_surf_features,
                    fusion_hessian_hog_features,
                ),
            ]

            for fusion_name, first_features, second_features in fusion_configs:
                if first_features is None or second_features is None:
                    continue

                accuracies = []
                for k in range(5):
                    first_train, label_train, first_test, label_test = p3.createTrainTest(
                        first_features, k
                    )
                    second_train, label_train_2, second_test, label_test_2 = p3.createTrainTest(
                        second_features, k
                    )

                    if label_train != label_train_2 or label_test != label_test_2:
                        raise ValueError("Late fusion requires identical train/test splits.")

                    bof_train_first, bof_test_first = p3.BagOfWords(first_train, first_test)
                    bof_train_second, bof_test_second = p3.BagOfWords(second_train, second_test)
                    bof_train_fused = np.hstack((bof_train_first, bof_train_second))
                    bof_test_fused = np.hstack((bof_test_first, bof_test_second))

                    accuracy, _, _ = p3.svm(
                        bof_train_fused, label_train, bof_test_fused, label_test
                    )
                    accuracies.append(float(accuracy))

                mean_accuracy = float(np.mean(accuracies))
                std_accuracy = float(np.std(accuracies))
                print(
                    f"{report_name} | {fusion_name}: "
                    f"{100.0 * mean_accuracy:.3f}% (+/- {100.0 * std_accuracy:.3f}%)"
                )
                summary[report_name][fusion_name] = mean_accuracy

        print("\n=== Summary ===")
        for augmentation_name, results in summary.items():
            print(augmentation_name)
            for combo_name, accuracy in results.items():
                print(f"  {combo_name}: {100.0 * accuracy:.3f}%")
