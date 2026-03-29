from matplotlib import pyplot as plt
import scipy.io
import cv2
import numpy as np
import os
import pickle
import time
import multiprocessing as mp
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.multiclass import OneVsRestClassifier
from scipy.spatial.distance import cdist
#set random seed for reproducibility
RANDOM_SEED = 42

# Part 3.1

def featuresSURF(I, kp):
    # I = I.astype(np.float32)/255
    kp_cv2 = [cv2.KeyPoint(x,y,2*np.ceil(3*scale)+1) for x,y,scale in kp]
    fex = cv2.SIFT_create()
    _, des = fex.compute(I, kp_cv2)
    return des

def featuresHOG(I, points):
    I = I.astype(float)/255
    N = points.shape[0]
    cells = np.array([2,2])
    nbins = 9
    ovrlp = .5
    [h,w] = I.shape

    desc = []
    for i in range(N):
        pi = int(points[i,1])
        pj = int(points[i,0])
        scale = points[i,2]
        rad = int(round(5*scale))
        img_patch = I[max(0,pi-rad):min(pi+rad,h)+1,max(0,pj-rad):min(pj+rad,w)+1]
        desc.append(simple_hog(img_patch,nbins,cells,ovrlp))

    return np.float32(np.array(desc))

def simple_hog(Im,bins,cells,overlap=0.3,signed=0):

    # % basic hog function
    # % Im : the input image
    # % bins : the number of bins
    # % cells : image segmentation. [cellsi cellsj] or one number for rectangular
    # % overlap : range : 0 - .5 for overlapping cells
    # % signed : 0 for unsigned , 1 for signed
    # % gradient_option : input to function imgradient.

    if len(cells.shape) == 1:
        cellsi = cells[0]
        cellsj = cells[0]
    else:
        cellsi = cells[0]
        cellsj = cells[1]

    N, M = Im.shape
    GY, GX = np.gradient(Im)
    magn, angle = cv2.cartToPolar(GX, GY)
    angle[angle < 0] = (signed+1)*np.pi + angle[angle < 0]
    p = (signed+1)*np.pi/bins
    ind_angle = np.mod(np.floor(angle/p).astype(int), bins)

    P,patch_i,patch_j = rectangular_grid(N,M,cellsi,cellsj,overlap)

    local_desc = []
    for i in range(cellsi*cellsj):
        i_start = int(max(0,P[i,1]-patch_i))
        i_end = int(min(N-1,P[i,1]+patch_i))
        j_start = int(max(0,P[i,0]-patch_j))
        j_end = int(min(M-1,P[i,0]+patch_j))

        hist = np.zeros((1,bins))

        if (i_start != i_end and j_start != j_end):
            m_part = magn[i_start:i_end+1,j_start:j_end+1]
            a_part = ind_angle[i_start:i_end+1,j_start:j_end+1]
            mm = m_part.flatten()
            aa = a_part.flatten()
            hist[0,0:max(aa)+1] = np.bincount(aa,weights=mm)

            hist=hist/(np.linalg.norm(hist)+np.finfo(float).eps)

        local_desc.append(hist)

    return np.concatenate(local_desc,axis=1).flatten()


def rectangular_grid(N,M,cellsi,cellsj,overlap):

    step_i = N/(cellsi*(1-overlap)+overlap)
    patch_i = round(step_i/2)
    step_j = M/(cellsj*(1-overlap)+overlap)
    patch_j = round(step_j/2)

    xr = np.linspace(patch_j,M-1-patch_j,cellsj)
    yr = np.linspace(patch_i,N-1-patch_i,cellsi)

    X, Y = np.meshgrid(xr,yr)
    P = np.concatenate([X.reshape((-1,1)), Y.reshape((-1,1))], axis=1).round().astype(int)
    return (P, patch_i, patch_j)

# Part 3.2
#function for adding gaussian noise
def add_gaussian_noise(I, rng=None, std=1):
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    # Match the effective noise scale used in part3_2 after ImageNet normalization.
    imagenet_gray_std = 0.225
    noise = rng.normal(loc=0.0, scale=std * 255.0 * imagenet_gray_std, size=I.shape)
    noisy = I.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

#function for adding random rotation
def add_random_rotation(I, rng=None, angle_range=(-45.0, 45.0)):
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    angle = float(rng.uniform(angle_range[0], angle_range[1]))
    height, width = I.shape
    center = (width / 2.0, height / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        I,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

#function for applying the distortions to the images depending on the mode provided as an argument , gaussian noise or random rotation , is called from feature extraction  function when the argument passed to it is true
def distort_image(I, mode, rng=None):
    if not mode:
        return I
    if mode is True:
        mode = 'gaussian_noise'

    if mode == 'gaussian_noise':
        return add_gaussian_noise(I, rng=rng)
    if mode == 'random_rotation':
        return add_random_rotation(I, rng=rng)

    raise ValueError(
        "distort must be one of False, True, 'gaussian_noise', or 'random_rotation'."
    )


def FeatureExtraction(detector_fun, descriptor_fun, loadFile=None, saveFile=None, distort=False):
    ''' Extract features using the descriptor provided in constructor'''

    if loadFile is not None:
        X_full = pickle.load(open(loadFile,'rb'))
        return X_full

    # The following should correspond to the data
    dataDir = './Data'
    categories = [
        ('person','TUGraz_person'),
        ('cars','TUGraz_cars'),
        ('bike','TUGraz_bike')
    ]

    num_cpus = os.cpu_count()
    if num_cpus is None:
        num_procs = 1
    else:
        num_procs = min(3,num_cpus-1)

    rng = np.random.default_rng(RANDOM_SEED)
    im_full = []
    for name, catDir in categories:
        img_list = sorted(os.listdir(os.path.join(dataDir, catDir)))
        im_class = []
        for img_file in tqdm(img_list, total=len(img_list), desc=f"Reading {name} images"):
            if name not in img_file:
                continue
            I = cv2.cvtColor(cv2.imread(os.path.join(dataDir, catDir, img_file)), cv2.COLOR_BGR2GRAY)
            if distort:
                I = distort_image(I, distort, rng=rng)
            I = cv2.resize(I, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            im_class.append(I)

        im_full.append((detector_fun,descriptor_fun,im_class))

    # print('Time for reading all images from disk: {:.3f}'.format(time.time()-start_time))

    start_time = time.time()

    X_full = list(map(FeatureExtractionThread, im_full))

    print('Time for feature extraction: {:.3f}'.format(time.time()-start_time))

    if saveFile is not None:
        pickle.dump(X_full, open(saveFile,'wb'))

    return X_full


def FeatureExtractionThread(tup):
    detect_fun, desc_fun, im_class = tup
    X_class = []
    for I in im_class:
        kp = detect_fun(I.astype(float)/255)
        descriptors = desc_fun(I,kp)
        X_class.append(descriptors)
    
    return X_class


def createTrainTest(features, k=None):
    r = 0.7
    if k is not None:
        mat = scipy.io.loadmat('./Fold_Indices.mat')
        indices = mat['Indices'].flatten()[k].flatten()
    else:
        raise ValueError('createTrainTest: Please provide fold index (1-5).')

    data_train = []
    data_test = []
    label_train = []
    label_test = []
    for c in range(indices.shape[0]):
        idx_class = indices[c].flatten()
        feats_class = [features[c][i] for i in idx_class]
        lim = int(round(r*idx_class.shape[0]))

        data_train.extend(feats_class[:lim])
        data_test.extend(feats_class[lim:])
        label_train.extend([c for i in range(len(feats_class[:lim]))])
        label_test.extend([c for i in range(len(feats_class[lim:]))])

    return (data_train, label_train, data_test, label_test)

def BagOfWords(data_train, data_test):
    slice_ratio = 0.5
    num_centers = 500

    train_all = np.concatenate(data_train, axis = 0)
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(train_all)
    # print(train_all.shape)

    clf = KMeans(
        n_clusters=num_centers,
        n_init=1,
        random_state=RANDOM_SEED,
        verbose=False,
    )

    clf.fit(train_all[:round(slice_ratio*train_all.shape[0])])
    C = clf.cluster_centers_
    BOF_tr = np.zeros((len(data_train), num_centers))
    BOF_ts = np.zeros((len(data_test), num_centers))

    for img_idx in range(len(data_train)):
        closest = np.argmin(cdist(data_train[img_idx], C), axis=1)
        hist = np.zeros(num_centers)
        hist[0:np.max(closest)+1] = np.bincount(closest)
        BOF_tr[img_idx,:] = hist/(np.linalg.norm(hist) + np.finfo(float).eps)

    for img_idx in range(len(data_test)):
        closest = np.argmin(cdist(data_test[img_idx], C), axis=1)
        hist = np.zeros(num_centers)
        hist[0:np.max(closest)+1] = np.bincount(closest)
        BOF_ts[img_idx,:] = hist/(np.linalg.norm(hist) + np.finfo(float).eps)

    return (BOF_tr, BOF_ts)

def svm(train_data, train_labels, test_data, test_labels, cost=1, svm_type='linear'):

    numClasses = len(np.unique(train_labels))

    if svm_type == 'chi2':
        raise NotImplementedError()

    if svm_type == 'linear':
        base_clf = SVC(C=cost, kernel='linear', probability=True,  verbose=0)
    elif svm_type == 'chi2':
        raise NotImplementedError()
    else:
        raise ValueError('svm: Unsupported Kernel Type.')

    multisvm = OneVsRestClassifier(base_clf)

    multisvm.fit(train_data, train_labels)

    predictions = multisvm.predict(test_data)
    probs = multisvm.predict_proba(test_data)

    acc = np.sum(predictions == test_labels)/len(test_labels)

    return (acc, predictions, probs)
