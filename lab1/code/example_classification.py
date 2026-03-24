# Assuming detectors are in file "cv25_lab1_part2.py", replace with your filename.
import cv25_lab1_part3_utils as p3
import cv25_lab1_part2 as p2

import numpy as np

if __name__ == '__main__':
    # Change with your own detectors here!
    detect_fun = lambda I: p2.harrisLaplaceDetector(I, 2, 2.5, 0.05, 0.005, 1.5, 4)

    desc_fun = lambda I, kp: p3.featuresSURF(I,kp)

    # Extract features from the provided dataset.
    feats = p3.FeatureExtraction(detect_fun, desc_fun)

    # If the above code takes too long, you can use the following extra parameters of Feature extraction:
    #   saveFile = <filename>: Save the extracted features in a file with the provided name.
    #   loadFile = <filename>: Load the extracted features from a given file (which MUST exist beforehand).


    accs = []
    for k in range(5):
        # Split into a training set and a test set.
        data_train, label_train, data_test, label_test = p3.createTrainTest(feats, k)

        # Perform Kmeans to find centroids for clusters.
        BOF_tr, BOF_ts = p3.BagOfWords(data_train, data_test)

        # Train an svm on the training set and make predictions on the test set
        acc, preds, probas = p3.svm(BOF_tr, label_train, BOF_ts, label_test)
        accs.append(acc)

    print('Mean accuracy for Harris-Laplace with SURF descriptors: {:.3f}%'.format(100.0*np.mean(accs)))
