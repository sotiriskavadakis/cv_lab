import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data', 'cv26_lab2_part1')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'part1')
os.makedirs(RESULTS_DIR, exist_ok=True)

#Initial bounding boxes [x, y, width, height] from the lab sheet
BOUNDING_BOXES = {
    'face':       [154, 102, 67, 115],
    'left_hand':  [93,  272, 56,  83],
    'right_hand': [201, 270, 56,  83],
}
#Colors for drawing the boxes (BGR format for OpenCV)
BOX_COLORS = {'face': (0, 0,255), 'left_hand': (0, 255, 0), 'right_hand': (255, 0, 0)}


import numpy as np
import cv2
from scipy.ndimage import map_coordinates

def lk(I1, I2, features, rho, epsilon, d_x0, d_y0):
    """
    Lucas-Kanade optical flow (Dense Vectorized approach)
    """
    #Ι1 and I2 are the cropped windows of the intial image  around the bounding boxes 
    #features are computed inside the cropped window of I2 and we want to track them back to I1
    #normalize images to [0, 1] for better numerical stability
    I1 = I1.astype(np.float64) / 255.0
    I2 = I2.astype(np.float64) / 255.0
    #we find the dimensions of the cropped image 
    H, W = I1.shape
    
    #we find the coordinates of the featurees 
    fx = np.clip(features[:, 0].astype(int), 0, W - 1)  #columns (x)
    fy = np.clip(features[:, 1].astype(int), 0, H - 1)  # rows (y)
    #build the grid of pixel coordinates (x_0, y_0) for the whole image
    x_0, y_0 = np.meshgrid(np.arange(W), np.arange(H))
    #compute all the gradients of the image 1
    dI1_dy, dI1_dx = np.gradient(I1) 

    #create the gaussian kernel with standard deviation rho and large enough kernel size
    kernel_size = int(np.ceil(3 * rho)) * 2 + 1
    k1d = cv2.getGaussianKernel(kernel_size, rho)
    G_kernel = (k1d @ k1d.T).astype(np.float64)
    #initialize the optical flow maps d_x_img and d_y_img for the WHOLE CROPPEDIMAGE NOT JUST THE FEATURES, using the initial conditions d_x0 and d_y0
    #initial condition may be either a scalar or a per-pixel 2D array used for multiscale , both wayw we create a map of displacements for all pixels for the iterations of the algorithm
    #IN GENERAL we decided to compute and update the displacements for all picel in the cropped image
    #in order to get better and more accurate results even though this makes our algorithm computationally more ecpensive
    #at the end we return only the indexed displacements at the feature locations for the porpuses of the lab
    #SO WE  INITIALLY COPMUTE THE DENSE OPTICAL FLOW AND THEN KEEP ONLYTHE DISPLACEMENETS OF THE FEATURES
    if np.ndim(d_x0) == 2:
        d_x_img = d_x0.astype(np.float64)
        d_y_img = d_y0.astype(np.float64)
    else:
        d_x_img = np.full((H, W), float(d_x0))
        d_y_img = np.full((H, W), float(d_y0))
    #we perform the iterations of the Lucas-Kanade algorithm , 150 iterations were enough for convergence in our tests
    for _ in range(150):
        
        # creates the shifted coordinates of the columns and rows by adding the current displacemtn and making it to a 1D array
        shifted_cols = np.ravel(x_0 + d_x_img)
        shifted_rows = np.ravel(y_0 + d_y_img)

        #in order to estmate the value In-1 at x+di (In-1(x+dxi))we use the map_coordinates function  to  perform interpolation at the shifted coordinates for the image I1 and its gradients, we reshape the result back to the original image shape (H, W)
        I1_shifted = map_coordinates(I1, [shifted_rows, shifted_cols], order=1, mode='nearest').reshape(H, W)
        #compute the gradients of In-1 at x+di with the same function
        A1 = map_coordinates(dI1_dx, [shifted_rows, shifted_cols], order=1, mode='nearest').reshape(H, W)#In-1(x+dx_i)/dx
        A2 = map_coordinates(dI1_dy, [shifted_rows, shifted_cols], order=1, mode='nearest').reshape(H, W)#In-1(x+dx_i)/dy
        #E(x) = In(x) - In-1(x+di)
        E = I2 - I1_shifted
        #now we have to compute the system Mu=B  so we compute the elements of the matrix M and the vector b by 
        #convolving the products of A1, A2, and E with the Gaussian kernel G_kernel, we add epsilon to the diagonal elements of M to ensure numerical stability
        #as in equation 2 of the lab sheet
        #this system is solved for each pixel simultaneously, we will solve it only for the features later by indexing the resulting u_x_img and u_y_img at the feature locations
        M11_img = cv2.filter2D(A1 * A1, -1, G_kernel) + epsilon
        M12_img = cv2.filter2D(A1 * A2, -1, G_kernel)
        M22_img = cv2.filter2D(A2 * A2, -1, G_kernel) + epsilon
        b1_img  = cv2.filter2D(A1 * E,  -1, G_kernel)
        b2_img  = cv2.filter2D(A2 * E,  -1, G_kernel)


        #now in order to get the solutions we will applyt he least square method and we will solve the system of equationsfor each pixel
        # in order to solve the system we used Cramer's Rule for a 2x2  system
        #so we compute the determinant of M and check if it's not too close to zero to avoid numerical instability, we use np.where to handle the case where the determinant is too small by setting the updates to zero in those cases
        det = M11_img * M22_img - M12_img * M12_img
        valid = np.abs(det) > 1e-12
        safe_det = np.where(valid, det, 1.0) # Αποφυγή διαίρεσης με το 0
        #cramers solution for u_x and u_y at each pixel, we will later index these at the feature locations to get the updates for the features
        u_x_img = np.where(valid, (M22_img * b1_img - M12_img * b2_img) / safe_det, 0.0)
        u_y_img = np.where(valid, (M11_img * b2_img - M12_img * b1_img) / safe_det, 0.0)

        #we update each pixel with the computed above displacement 
        d_x_img += u_x_img
        d_y_img += u_y_img

        #at the end we kept only the updates for the features points since those are the ones we care in the lab 
        #and for tracking of the important objectsin the video
        u_x_features = u_x_img[fy, fx]
        u_y_features = u_y_img[fy, fx]

        #as a breaking condition we check if the maximum update among the features is below a certain threshold,
        # if it is we break the loop since we have converged to a solution where the features are not moving significantly anymore
        threshold = 1e-4
        max_update = np.max(np.sqrt(u_x_features**2 + u_y_features**2))
        if max_update < threshold:
            break 
    #after convergence we return the final displacements for the features by indexing the final d_x_img and d_y_img 
    #at the feature locations
    d_x=d_x_img[fy, fx]
    d_y=d_y_img[fy, fx]
    #we return the final displacements for the features , but these are not the real one we will have to take the negative of them 
    #since they represent the movement from I2 to I1 since as we will se in the main code
    #we compute the features in the I2 image while we want the movement from I1 to I2 for the tracking of the objects in the video
    return d_x, d_y , d_x_img, d_y_img #we return also the dense dispacement for the multiscale version as we have to put the dense result as initialization to the next finr level


def lk_multiscale(I1, I2, features, rho, epsilon, d_x0, d_y0, num_scales):
    """
    Multiscale Lucas-Kanade optical flow via Gaussian pyramid.

    Builds a coarse-to-fine pyramid of num_scales levels.
    At each level (coarsest → finest):
      - runs single-scale lk() with the current d as initial condition
      - doubles d before passing it to the next finer level

    Parameters mirror lk(), plus num_scales (number of pyramid levels).
    Returns d_x, d_y at the finest (original) scale for the given features.
    """
    #here we will build the gaussian pyramids of the two images in order to start from a very coarse version
    #and progressively refine the optical flow at larger resolutions
    #index 0 is the original image and index -1 will be the coarsest-smallest level
    pyramid1 = [I1]
    pyramid2 = [I2]
    for _ in range(num_scales - 1):
        #before downsampling we blur the image in order to avoid aliasing effects
        #so we apply a gaussian blur with a standard deviation of 3 which is a common choice for anti-aliasing before downsampling
        blurred1 = cv2.GaussianBlur(pyramid1[-1], (0, 0), sigmaX=3, sigmaY=3)
        blurred2 = cv2.GaussianBlur(pyramid2[-1], (0, 0), sigmaX=3, sigmaY=3)
        #we downsample by a factor of 2 each time so the next level is half the size of the previous one
        pyramid1.append(blurred1[::2, ::2])
        pyramid2.append(blurred2[::2, ::2])

    #we initialize the displacement maps at the coarsest scale
    #the initial displacement is scalar usually zero and we create a uniform map for the whole image as we were doing in the simple lk
    #we divide by the total scale factor because at the coarsest image the displacement should be divided accordingly to the number of scales
    #since the motion in the coarsest level is smaller by a factor of 2^(num_scales-1) compared to the original scale, we have to divide the initial displacement by this factor to get the correct initial condition for the coarsest level
    H_c, W_c = pyramid1[-1].shape[:2] 
    scale_factor = 2 ** (num_scales - 1)
    d_x_map = np.full((H_c, W_c), d_x0 / scale_factor)#so we create a map of displacementes for all pixels at the coarest level
    d_y_map = np.full((H_c, W_c), d_y0 / scale_factor)

    #now we start the coarse to fine refinement
    #we begin from the coarsest level and we move towards the original resolution
    for level in range(num_scales - 1, -1, -1):
        I1_level = pyramid1[level]
        I2_level = pyramid2[level]
        scale     = 2 ** level

        #the features were originally computed in the full resolution image I2
        #so for each pyramid level we have to scale them down accordingly in order to refer to the correct positions in that level
        features_level = features / scale
        H_l, W_l = I1_level.shape[:2]
        #we clip them in order to ensure that no feature lies outside the image boundaries of the current level
        features_level[:, 0] = np.clip(features_level[:, 0], 0, W_l - 1)
        features_level[:, 1] = np.clip(features_level[:, 1], 0, H_l - 1)

        #we now run the simple lk at the current scale
        #instead of starting from zero each time we pass as initial condition the dense displacement map computed from the coarser level
        #this is the key idea of multiscale lucas kanade because the coarse level gives a good initial guess for the finer level
        dx_level, dy_level , d_x_map, d_y_map = lk( #we return both the dense map in order to use it as initialization to the next level and the feature displacements at the current level that we need to compute for the motion of the object
            I1_level, I2_level,
            features_level,
            rho, epsilon,
            d_x_map, d_y_map      # 2D maps that is we built lk so that it can handels 2d initialization
        )

        #if we are not yet at the finest level we have to move to the next finer image
        #in order to do that we upsample the dense displacement maps and then multiply by 2
        #because one pixel motion in the coarse image corresponds to two pixels motion in the next finer image
        if level > 0:
            H_next, W_next = pyramid1[level - 1].shape[:2]#we get the dimensions of the next finer image

            #build the coordinates of the next finer image
            # x_next, y_next = np.meshgrid(np.arange(W_next), np.arange(H_next))
            # #map these coordinates back to the coarse grid in order to interpolate the displacement maps
            # x_coarse = x_next / 2.0
            # y_coarse = y_next / 2.0

            # #interpolate the dense dx map from the coarse level to the finer level
            # d_x_map = map_coordinates(
            #     d_x_map,
            #     [np.ravel(y_coarse), np.ravel(x_coarse)],
            #     order=1, mode='nearest'
            # ).reshape(H_next, W_next)#map to next level

            # #interpolate similarly the dense dy map
            # d_y_map = map_coordinates(
            #     d_y_map,
            #     [np.ravel(y_coarse), np.ravel(x_coarse)],
            #     order=1, mode='nearest'
            # ).reshape(H_next, W_next)
     
            # #after interpolation we multiply by 2 to express the motion in the coordinate system of the finer level
            # d_x_map *= 2.0
            # d_y_map *= 2.0

            d_x_map = cv2.resize(d_x_map, (W_next, H_next), interpolation=cv2.INTER_LINEAR) * 2.0
            d_y_map = cv2.resize(d_y_map, (W_next, H_next), interpolation=cv2.INTER_LINEAR) * 2.0
        else:
            #at the finest level for completeness we store the final per-feature displacements back into the dense maps
            H_l, W_l = I1_level.shape[:2]
            fi = np.clip(features_level[:, 0].astype(int), 0, W_l - 1)
            fj = np.clip(features_level[:, 1].astype(int), 0, H_l - 1)
            d_x_map[fj, fi] = dx_level
            d_y_map[fj, fi] = dy_level

    #finally we return the displacements of the features at the finest original scale
    return dx_level, dy_level

#ENERGY MASK+MEDIAN APPROACH FOR DISPLACEMENT COMPUTATION
def displ(d_x, d_y):
        """
        in order to compute the total displacement of the bounding box from the different displacements of the features , 
        as well as rejecting outliers, we did not just take the mean of all feature's displacements,
        but we applied an energy-mask that filters all the displacements of all the features
        and it let passes the displacements of the features whose energy (which is the square of the magnitude of the displacement vector ||d||^2)
        is above a certain threshold which we were changing in order to test our results
        
        After computing the survivor feature points we take the median of these displacements as a better alternative from the mean
        in order to deal with remaining outliers and that was the total displacement of the box
        Falls back to plain mean if no feature survives the threshold.
        """
        energy_threshold=1
        energy = d_x ** 2 + d_y ** 2
        mask = energy > energy_threshold
        if mask.sum() > 0: #if there are surviving features
            return np.median(d_x[mask]), np.median(d_y[mask])#apply the mask so we keep only the displacemenets with a hiigh energy and then compute the median fo them
#         #in case no survivng features we just return the mean of all displacements without applying any mask
        return np.median(d_x), np.median(d_y)

#ALTERNATIVE MEAN APPROACH FOR DISPLACEMENT COMPUTATION WITH ENERGY MASK+MEAN
# def displ(d_x, d_y):
      
#       energy_threshold = 1
#       energy = d_x ** 2 + d_y ** 2
#       mask = energy > energy_threshold

#       if mask.sum() > 0:
#           return np.mean(d_x[mask]), np.mean(d_y[mask])

#       return np.mean(d_x), np.mean(d_y)
# Synthetic test: I2 = I1 shifted by (1, 1) — expected output dx=1, dy=1 
if __name__ == '__main__':
    from scipy.ndimage import shift as ndimage_shift
    tvl1_estimator = cv2.optflow.DualTVL1OpticalFlow_create(nscales=1)#tvl 

    I1 = cv2.imread(os.path.join(DATA_DIR, '1.png'), cv2.IMREAD_GRAYSCALE)
    
    #Create I2 by shifting I1 by exactly +1 pixel in x (col) and +1 pixel in y (row)
    shft_y, shft_x = 1.0, 1.0
    shift=[shft_y, shft_x]
    I2 = ndimage_shift(I1.astype(np.float64), shift=shift, mode='nearest').astype(np.uint8)
    #we run the lk for all boxes in the cropped images
    print('Sanity check on synthetic shift (I2 = I1 shifted by (1, 1))')
    for name, (x, y, w, h) in BOUNDING_BOXES.items():
        crop_I1 = I1[y:y+h, x:x+w]
        crop_I2 = I2[y:y+h, x:x+w]#cropped pathces

        pts = cv2.goodFeaturesToTrack(crop_I2, maxCorners=200, #feature detection in I2
                                      qualityLevel=0.01, minDistance=5)
        

        features = pts.reshape(-1, 2)        #reshape for our lk function

        d_x_lk, d_y_lk, _, _ = lk(crop_I1, crop_I2, features, #dx dy of lk
                               rho=10, epsilon=0.001, d_x0=0.0, d_y0=0.0)
        displacement_x, displacement_y = displ(-d_x_lk, -d_y_lk)  #real displacements are the negtive of the ones computed because  we find the featres in I2
        d_x_mlk, d_y_mlk = lk_multiscale(crop_I1, crop_I2, features, #same for multiscale lk
                                        rho=5, epsilon=0.001, d_x0=0.0, d_y0=0.0, num_scales=3)
        displacement_x_mlk, displacement_y_mlk = displ(-d_x_mlk, -d_y_mlk)
    
        lk_dx = -d_x_lk  # engative displacements
        lk_dy = -d_y_lk
        mlk_dx = -d_x_mlk
        mlk_dy = -d_y_mlk
        flow = tvl1_estimator.calc(crop_I2, crop_I1, None)  # (H, W, 2) #again we keep the right order for i1 i2 based on where the features are computed
        fx = np.clip(features[:, 0].astype(int), 0, crop_I1.shape[1] - 1)
        fy = np.clip(features[:, 1].astype(int), 0, crop_I1.shape[0] - 1)
        tvl1_dx = flow[fy, fx, 0]   # (N,)
        tvl1_dy = flow[fy, fx, 1]   # (N,)
        tvl1_dx = -tvl1_dx  #for the same reason revert the sign of the displacements as we did for lk and mlk
        tvl1_dy = -tvl1_dy  
        displacement_x_tvl1, displacement_y_tvl1 = displ(tvl1_dx, tvl1_dy)
       #results for our sanity check
        print(f'{name}:')
        print(f'  features      : {len(lk_dx)}')
        print(f' Uni scale LK ')
        print(f'  mean d_x = {lk_dx.mean():.4f}  (expected  {shft_x:.1f})')
        print(f'  mean d_y = {lk_dy.mean():.4f}  (expected  {shft_y:.1f})')
        print(f' diplacement: dx = {displacement_x:.4f}  dy = {displacement_y:.4f}  (expected {shft_x:.1f}, {shft_y:.1f})')
        print(f' Multi scale LK ')
        print(f'  mean d_x = {mlk_dx.mean():.4f}  (expected  {shft_x:.1f})')
        print(f'  mean d_y = {mlk_dy.mean():.4f}  (expected  {shft_y:.1f}  )')
        print(f' diplacement: dx = {displacement_x_mlk:.4f}  dy = {displacement_y_mlk:.4f}  (expected {shft_x:.1f}, {shft_y:.1f})')
        print(f' TV-L1 ')
        print(f'  TV-L1 mean d_x = {tvl1_dx.mean():.4f}  (expected  {shft_x:.1f})')
        print(f'  TV-L1 mean d_y = {tvl1_dy.mean():.4f}  (expected  {shft_y:.1f})')
        print(f' diplacement: dx = {displacement_x_tvl1:.4f}  dy = {displacement_y_tvl1:.4f}  (expected {shft_x:.1f}, {shft_y:.1f})')
    
    
    #  Grid plots: I1 + features | I2 + features | optical flow LK | optical flow TV-L1 for all the frames 
    frame_files = sorted(
         [f for f in os.listdir(DATA_DIR) if f.endswith('.png')],
         key=lambda f: int(os.path.splitext(f)[0])
     )
    NUM_FRAMES = len(frame_files)
    print('\nGrid plots: I1 + features | I2 + features | optical flow LK | optical flow TV-L1 for all the frames')

    # tvl1_estimator = cv2.optflow.DualTVL1OpticalFlow_create(nscales=1)

    # for idx in range(NUM_FRAMES - 1):
    #     f1 = cv2.imread(os.path.join(DATA_DIR, frame_files[idx]),     cv2.IMREAD_GRAYSCALE)
    #     f2 = cv2.imread(os.path.join(DATA_DIR, frame_files[idx + 1]), cv2.IMREAD_GRAYSCALE)

    #     #3 bounding boxes × 4 columns (I1 | I2+pts | LK flow | TV-L1 flow)
    #     fig, axes = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)
    #     fig.suptitle(f'Optical flow comparison — frames {idx+1} to {idx+2}', fontsize=12)

    #     col_titles = ['I1', 'I2 (features)', 'LK flow', 'TV-L1 flow']
    #     for col, title in enumerate(col_titles):
    #         axes[0, col].set_title(title, fontsize=10)

    #     for row, (name, (x, y, w, h)) in enumerate(BOUNDING_BOXES.items()):
    #         crop1 = f1[y:y+h, x:x+w]
    #         crop2 = f2[y:y+h, x:x+w]

           
    #         pts = cv2.goodFeaturesToTrack(crop2, maxCorners=200,
    #                                        qualityLevel=0.01, minDistance=5)

    #         #column 0: I1
    #         axes[row, 0].imshow(crop1, cmap='gray', origin='upper')
    #         axes[row, 0].set_ylabel(name, fontsize=9)
    #         axes[row, 0].axis('off')

    #         #column 1: I2 with detected feature points
    #         axes[row, 1].imshow(crop2, cmap='gray', origin='upper')
    #         if pts is not None:
    #             features = pts.reshape(-1, 2)
    #             axes[row, 1].scatter(features[:, 0], features[:, 1],
    #                                  s=12, c='lime', marker='o',
    #                                  edgecolors='black', linewidths=0.3)
    #         axes[row, 1].axis('off')

    #         #column 2: Lucas-Kanade quiver without image background
    #         axes[row, 2].set_xlim(0, w)
    #         axes[row, 2].set_ylim(h, 0)
    #         axes[row, 2].set_aspect('equal')
    #         axes[row, 2].axis('off')
    #         if pts is not None:
    #             features = pts.reshape(-1, 2)
    #             d_x, d_y, _, _ = lk(crop1, crop2, features,
    #                           rho=10, epsilon=0.001, d_x0=0.0, d_y0=0.0)
    #             axes[row, 2].quiver( #visualiation with quiver plot
    #                 features[:, 0], features[:, 1],
    #                 -d_x, -d_y,
    #                 angles='xy', scale_units='xy', scale=1,
    #                 color='black', width=0.006
    #             )

    #         #column 3: TV-L1 quiver without image background
    #         axes[row, 3].set_xlim(0, w)
    #         axes[row, 3].set_ylim(h, 0)
    #         axes[row, 3].set_aspect('equal')
    #         axes[row, 3].axis('off')
    #         if pts is not None:
    #             features = pts.reshape(-1, 2)
    #             flow = tvl1_estimator.calc(crop1, crop2, None)
    #             fx = np.clip(features[:, 0].astype(int), 0, crop1.shape[1] - 1)
    #             fy = np.clip(features[:, 1].astype(int), 0, crop1.shape[0] - 1)
    #             tvl1_dx = flow[fy, fx, 0]
    #             tvl1_dy = flow[fy, fx, 1]
    #             axes[row, 3].quiver(
    #                 features[:, 0], features[:, 1],
    #                 tvl1_dx, tvl1_dy,
    #                 angles='xy', scale_units='xy', scale=1,
    #                 color='darkred', width=0.006
    #             )

    #     out = os.path.join(RESULTS_DIR, f'grid_r10e001_{idx+1:03d}.jpg')
    #     fig.savefig(out, dpi=120)
    #     plt.close(fig)
    #     print(f'  saved {os.path.basename(out)}')


    #Section 1.1.2: Bounding-box tracking
    print('\nBounding-box tracking') 
    # One figure   per frame with all three boxes tracked together

    #Initialise box state for each region
    box_state = {name: list(map(float, bb)) for name, bb in BOUNDING_BOXES.items()}

    #Get frame size from first frame
    sample = cv2.imread(os.path.join(DATA_DIR, frame_files[0]))
    H_vid, W_vid = sample.shape[:2]
    
    out_vid = os.path.join(RESULTS_DIR, 'tracking_mlk.mp4')

    writer = cv2.VideoWriter(#synthesize the frames to a video
        out_vid,
        cv2.VideoWriter_fourcc(*'mp4v'),
        5,              # fps — slow enough to follow the tracking
        (W_vid, H_vid)
    )

    fig_track, axes_track = plt.subplots(
        1, NUM_FRAMES, figsize=(3 * NUM_FRAMES, 4), constrained_layout=True
    )
    fig_track.suptitle('Tracking: face + hands', fontsize=11)
    tvl1_estimator = cv2.optflow.DualTVL1OpticalFlow_create(nscales=1)
    for idx, fname in enumerate(frame_files):
        frame_bgr  = cv2.imread(os.path.join(DATA_DIR, fname))
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        vis = frame_bgr.copy()

        #Draw all three boxes on the same frame
        for name, (bx, by, bw, bh) in box_state.items():
            ix, iy, iw, ih = int(round(bx)), int(round(by)), int(round(bw)), int(round(bh))
            cv2.rectangle(vis, (ix, iy), (ix + iw, iy + ih), BOX_COLORS[name], 2)
            cv2.putText(vis, name, (ix, max(iy - 4, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, BOX_COLORS[name], 1)

        writer.write(vis)
        axes_track[idx].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes_track[idx].set_title(f'frame {idx+1}', fontsize=8)
        axes_track[idx].axis('off')

        #Propagate each box to the next frame
        if idx < NUM_FRAMES - 1:
            next_gray = cv2.imread(
                os.path.join(DATA_DIR, frame_files[idx + 1]), cv2.IMREAD_GRAYSCALE
            )
            for name in list(box_state.keys()):
                bx, by, bw, bh = box_state[name] #current box position and size
                cx  = max(0, min(int(round(bx)), frame_gray.shape[1] - 1))
                cy  = max(0, min(int(round(by)), frame_gray.shape[0] - 1))
                cw  = min(int(round(bw)), frame_gray.shape[1] - cx)
                ch  = min(int(round(bh)), frame_gray.shape[0] - cy)
                #cropped images for the current frame
                crop_I1 = frame_gray[cy:cy+ch, cx:cx+cw]
                crop_I2 = next_gray[cy:cy+ch, cx:cx+cw]

                pts = cv2.goodFeaturesToTrack(crop_I2, maxCorners=200,
                                              qualityLevel=0.01, minDistance=5)
                if name == 'left_hand' and idx >= NUM_FRAMES - 3:
                    num_pts = 0 if pts is None else len(pts)
                    print(f'[debug] frame {idx+1} -> {idx+2} left_hand features: {num_pts}')
                if pts is not None and len(pts) > 0:
                    features = pts.reshape(-1, 2)

                    #SIPLE SCALE LUCAS-KANADE APPROACH
                    #d_x, d_y, _, _ = lk(crop_I1, crop_I2, features, rho=5, epsilon=0.001, d_x0=0.0, d_y0=0.0)

                    #TVL 
                    #flow = tvl1_estimator.calc(crop_I2, crop_I1, None)  # (H, W, 2)
                    #TVL DENSE OPTICAL FLOW APPROACH
                    # d_x_img = -flow[:, :, 0]
                    # d_y_img = -flow[:, :, 1]
                    # dx_box, dy_box = displ(d_x_img.ravel(), d_y_img.ravel())
                    # #TVL FEATURE-BASED APPROACH
                    # fx = np.clip(features[:, 0].astype(int), 0, crop_I1.shape[1] - 1)
                    # fy = np.clip(features[:, 1].astype(int), 0, crop_I1.shape[0] - 1)
                    # d_x = flow[fy, fx, 0]   # (N,)
                    # d_y = flow[fy, fx, 1]   # (N,)

                    #MULTI-SCALE LUCAS-KANADE APPROACH
                    d_x, d_y = lk_multiscale(crop_I1, crop_I2, features, rho=5, epsilon=0.01, d_x0=0.0, d_y0=0.0, num_scales=3)
                    dx_box, dy_box = displ(d_x, d_y)
                    dx_box, dy_box = -dx_box, -dy_box #each box displacement
                    box_state[name][0] += dx_box #box new position with the computed displacement
                    box_state[name][1] += dy_box
                    
    writer.release()
    print(f'  saved {os.path.basename(out_vid)}')
    #out_img = os.path.join(RESULTS_DIR, 'tracking_tvl.jpg')
    #out_img = os.path.join(RESULTS_DIR, 'tracking_uni.jpg')
    out_img = os.path.join(RESULTS_DIR, 'tracking_mlk.jpg')

    fig_track.savefig(out_img, dpi=120)
    plt.close(fig_track)
    print(f'  saved {os.path.basename(out_img)}')

    print(f'\nTracking done — results in {os.path.abspath(RESULTS_DIR)}')
