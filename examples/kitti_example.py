import yaml, sys, os
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.Entropy import EntropyOnline
from scripts.Spaciousness import SpaciousnessOnline
from scripts.FixedDistance import FixedDistanceOnline
from scripts.RMIP import RMIP_Online
from optkey_utils.OT_Handler import OT_Handler
from optkey_utils.KITTI_Handler import KITTI_Handler
from optkey_utils.descriptors.OverlapTransformer.modules.overlap_transformer import *
from optkey_utils.descriptors.OverlapTransformer.tools.overlap_utils.overlap_utils import *
# from optkey_utils.PGO import PGO
from scipy.spatial.transform import Rotation as R

## Path to config file
config_file = '/home/niksta/python_projects/msa/config/kitti_config.yaml'
## Load config file
with open(config_file) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
## Load descriptors from file or compute now
FROM_FILE = config['from_file']
## Define the dataset 
DATASET = config['dataset']
## Path to the dataset
PATH = config['dataset_path']
## List of sequences 
SEQUENCES = config['kitti_sequences']
## Path to the dataset
DATASET_PATH = PATH + '/' + DATASET
## Define the sequence path
SEQUENCE_PATH = DATASET_PATH + '/sequences/'
## Define descriptor method
DESCRIPTOR_METHOD = config['descriptor']
## PATH TO WEIGHTS
PATH_TO_WEIGHTS = config['weights_path']
## PATH TO SAVE RESULTS
PATH_TO_SAVE = config['save_path']
## Device: cpu or cuda
DEVICE = config['device']
## Optimization parameters
ALPHA = config['alpha']
BETA = config['beta']
WINDOW_SIZE = config['window_size']
SAMPLING_METHODS = config['methods']
## PGO parameters
ODOM_NOISE = config['odom_noise']
LOOP_NOISE = config['loop_noise']
GT_NOISE = config['gt_noise']
USE_GT = config['use_gt']
RADIUS = config['loop_closure_radius']
MIN_SEPARATION = config['min_separation']
N_NEIGHBORS = config['n_neighbors']
THRESHOLD = config['descriptor_threshold']
PLOT = config['plot']
SAVE = config['save']

def save_results_to_file(filename, method, num_frames,
                         icp_ate_t, icp_ate_r, pgo_ate_t, pgo_ate_r, 
                         icp_rpe_t, icp_rpe_r, pgo_rpe_t, pgo_rpe_r, 
                         toc, tic, loop_closures, pgo_poses):
    box_width = 60
    # Prepare the formatted output
    output = []
    output.append(" " + "-" * box_width)
    output.append(f"|{method.center(box_width - 2)}  |")
    output.append(" " + "-" * box_width)
    output.append(f"| KISS-ICP: ATE in Translation:                 {format_number(icp_ate_t, 10)} m |")
    output.append(f"| KISS-ICP: ATE in Rotation:                    {format_number(icp_ate_r * 180 / np.pi, 10)} ∘ |")
    output.append(f"| PGO: ATE in Translation:                      {format_number(pgo_ate_t, 10)} m |")
    output.append(f"| PGO: ATE in Rotation:                         {format_number(pgo_ate_r * 180 / np.pi, 10)} ∘ |")
    output.append(f"| Improvement in Translation:                   {format_number((icp_ate_t - pgo_ate_t) / icp_ate_t * 100, 10)} % |")
    output.append(f"| Improvement in Rotation:                      {format_number(((icp_ate_r * 180 / np.pi) - (pgo_ate_r * 180 / np.pi)) / (icp_ate_r * 180 / np.pi) * 100, 10)} % |")
    output.append(" " + "-" * box_width)
    output.append(f"| KISS-ICP: RPE in Translation:                 {format_number(icp_rpe_t, 10)} m |")
    output.append(f"| KISS-ICP: RPE in Rotation:                    {format_number(icp_rpe_r * 180 / np.pi, 10)} ∘ |")
    output.append(f"| PGO: RPE in Translation:                      {format_number(pgo_rpe_t, 10)} m |")
    output.append(f"| PGO: RPE in Rotation:                         {format_number(pgo_rpe_r * 180 / np.pi, 10)} ∘ |")
    output.append(f"| Improvement in Translation:                   {format_number((icp_rpe_t - pgo_rpe_t) / icp_rpe_t * 100, 10)} % |")
    output.append(f"| Improvement in Rotation:                      {format_number(((icp_rpe_r * 180 / np.pi) - (pgo_rpe_r * 180 / np.pi)) / (icp_rpe_r * 180 / np.pi) * 100, 10)} % |")
    output.append(f"| Time:                                         {format_number(toc - tic, 10)} s |")
    output.append(f"| Number of Loop Closures:                      {format_number(len(loop_closures), precision=0)}   |")
    output.append(" " + "-" * box_width)
    output.append(f'')
    ## Save to file
    with open(filename, 'a') as file:
        file.write("\n".join(output))
    ## Save to txt without formatting for easy plotting
    ates = np.array([icp_ate_t, icp_ate_r, pgo_ate_t, pgo_ate_r, icp_rpe_t, icp_rpe_r, pgo_rpe_t, pgo_rpe_r, toc-tic, num_frames, len(loop_closures)])
    with open(filename.replace('.txt', '_raw.txt'), 'a') as file:
        ## Write the method name first
        file.write(f'{method}\n')
        ## Write the ATE and RPE values
        np.savetxt(file, ates, fmt='%1.4f')
        file.write('\n')
    ## Save the loop closures for each method in different file as npy
    np.save(filename.replace('.txt', f'_{method}_loop_closures.npy'), loop_closures, allow_pickle=True)
    ## Save the pgo poses in a txt file
    np.savetxt(filename.replace('.txt', f'_{method}_pgo_poses.txt'), pgo_poses.reshape(-1, 16))
    return

def format_number(num, width=10, precision=4):
        return f"{num:>{width}.{precision}f}"

def print_ate_ape(icp_ate_t, icp_ate_r, pgo_ate_t, pgo_ate_r, 
                  icp_rpe_t, pgo_rpe_t, icp_rpe_r, pgo_rpe_r, 
                  loop_closures, tic, toc, method, true_positives, false_positives):
    box_width = 60
    # Helper function to format numbers with a fixed width and precision
    print(f'')
    # Print the formatted output
    print(" " + "-" * box_width)
    print(f"|{method.center(box_width - 2)}  |")
    print(" " + "-" * box_width)
    print(f"| KISS-ICP: ATE in Translation:                 {format_number(icp_ate_t, 10)} m |")
    print(f"| KISS-ICP: ATE in Rotation:                    {format_number(icp_ate_r*180/np.pi, 10)} ∘ |")
    print(f"| PGO: ATE in Translation:                      {format_number(pgo_ate_t, 10)} m |")
    print(f"| PGO: ATE in Rotation:                         {format_number(pgo_ate_r*180/np.pi, 10)} ∘ |")
    print(f"| Improvement in Translation:                   {format_number((icp_ate_t - pgo_ate_t) / icp_ate_t * 100, 10)} % |")
    print(f"| Improvement in Rotation:                      {format_number(((icp_ate_r*180/np.pi) - (pgo_ate_r *180/np.pi)) / (icp_ate_r *180/np.pi) * 100, 10)} % |")
    print(" " + "-" * box_width)
    # print(f"| KISS-ICP: RPE in Translation:                 {format_number(icp_rpe_t, 10)} m |")
    # print(f"| KISS-ICP: RPE in Rotation:                    {format_number(icp_rpe_r*180/np.pi, 10)} ∘ |")
    # print(f"| PGO: RPE in Translation:                      {format_number(pgo_rpe_t, 10)} m |")
    # print(f"| PGO: RPE in Rotation:                         {format_number(pgo_rpe_r*180/np.pi, 10)} ∘ |")
    print(f"| Improvement in Translation:                   {format_number((icp_rpe_t - pgo_rpe_t) / icp_rpe_t * 100, 10)} % |")
    print(f"| Improvement in Rotation:                      {format_number(((icp_rpe_r*180/np.pi) - (pgo_rpe_r *180/np.pi)) / (icp_rpe_r *180/np.pi) * 100, 10)} % |")
    print(f"| Time:                                         {format_number(toc - tic, 10)} s |")
    print(f"| Number of Loop Closures:                      {format_number(len(loop_closures), precision=0)}   | ")
    print(f"| False Positive Rate:                          {format_number(false_positives/(false_positives+true_positives) * 100, 10, 4)} %")
    print(" " + "-" * box_width)
    return

def compute_ate(ground_truth, estimated):
    """
    Compute Absolute Trajectory Error (ATE) for both translation and rotation 
    between ground truth and estimated trajectories.
    
    Parameters:
    - ground_truth (numpy.ndarray): Ground truth poses of shape (N, 4, 4).
    - estimated (numpy.ndarray): Estimated poses of shape (N, 4, 4).
    
    Returns:
    - translation_ate (float): RMSE of the translation absolute trajectory error.
    - rotation_ate (float): RMSE of the rotation absolute trajectory error.
    """
    ## Number of poses
    N = ground_truth.shape[0]
    
    ## Align estimated trajectory to ground truth using SVD (Umeyama method)
    mean_gt = np.mean(ground_truth[:, :3, 3], axis=0)
    mean_est = np.mean(estimated[:, :3, 3], axis=0)
    centered_gt = ground_truth[:, :3, 3] - mean_gt
    centered_est = estimated[:, :3, 3] - mean_est
    W = np.dot(centered_est.T, centered_gt)
    U, _, Vt = np.linalg.svd(W)
    
    ## Compute rotation and translation
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:  # Ensure a proper rotation matrix
        Vt[-1, :] *= -1
        R = np.dot(U, Vt)
    t = mean_gt - np.dot(R, mean_est)
    
    ## Apply the alignment to each pose
    aligned_estimated = estimated.copy()
    aligned_estimated[:, :3, :3] = np.einsum('ij,njk->nik', R, estimated[:, :3, :3])
    aligned_estimated[:, :3, 3] = np.dot(R, estimated[:, :3, 3].T).T + t
    
    ## Compute the translation ATE as RMSE
    translation_diff = ground_truth[:, :3, 3] - aligned_estimated[:, :3, 3]
    translation_ate = np.sqrt(np.mean(np.sum(translation_diff ** 2, axis=1)))

    ## Compute the rotation ATE as RMSE
    rotation_diff = np.array([np.linalg.norm(np.eye(3) - np.dot(gt[:3, :3].T, est[:3, :3]), 'fro') 
                              for gt, est in zip(ground_truth, aligned_estimated)])
    rotation_ate = np.sqrt(np.mean(rotation_diff ** 2))
    
    return translation_ate, rotation_ate

def compute_rpe(ground_truth, estimated, delta=1):
    """
    Compute Relative Pose Error (RPE) for both translation and rotation 
    between ground truth and estimated trajectories.
    
    Parameters:
    - ground_truth (numpy.ndarray): Ground truth poses of shape (N, 4, 4).
    - estimated (numpy.ndarray): Estimated poses of shape (N, 4, 4).
    - delta (int): Interval over which the RPE is calculated (default is 1 for consecutive poses).
    
    Returns:
    - translation_rpe (float): RMSE of the translation relative pose error.
    - rotation_rpe (float): RMSE of the rotation relative pose error.
    """
    ## Number of poses
    N = ground_truth.shape[0]
    
    ## Initialize lists to store translation and rotation errors
    translation_errors = []
    rotation_errors = []
    
    for i in range(N - delta):
        ## Ground truth relative motion
        gt_delta_pose = np.dot(np.linalg.inv(ground_truth[i]), ground_truth[i + delta])
        ## Estimated relative motion
        est_delta_pose = np.dot(np.linalg.inv(estimated[i]), estimated[i + delta])
        ## Compute relative transformation error
        error_transform = np.dot(np.linalg.inv(gt_delta_pose), est_delta_pose)
        
        ## Translation error
        error_translation = error_transform[:3, 3]
        translation_error = np.linalg.norm(error_translation)
        translation_errors.append(translation_error)
        
        ## Rotation error: Compute the rotation matrix difference
        error_rotation = error_transform[:3, :3]
        ## Clip the value for numerical stability in arccos
        trace_error_rotation = np.clip((np.trace(error_rotation) - 1) / 2, -1.0, 1.0)
        angle_error = np.arccos(trace_error_rotation)  # Convert rotation matrix to angle
        rotation_errors.append(angle_error)
    
    ## Compute RMSE for translation and rotation RPE
    translation_rpe = np.sqrt(np.mean(np.array(translation_errors) ** 2))
    rotation_rpe = np.sqrt(np.mean(np.array(rotation_errors) ** 2))
    
    return translation_rpe, rotation_rpe

def run_pgo(gt_poses:np.ndarray, descriptors:np.ndarray, 
            icp_poses:np.ndarray, scans:np.ndarray, min_time) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    ## Perform the loop closures
    pgo = PGO(optimizer='LM', 
                use_gt=USE_GT, 
                gt_noise=GT_NOISE, 
                loop_noise=LOOP_NOISE, 
                odom_noise=ODOM_NOISE) # Levenberg-Marquardt
    ## Create the graph
    pgo.create_factor_graph(poses=icp_poses)
    ## Add the edges
    pgo.add_edges(poses=icp_poses)
    ## Perform loop closure detection
    loop_closures, true_positives, false_positives = pgo.loop_closure_detection(poses=icp_poses, 
                                                gt_poses=gt_poses,
                                                descriptors=descriptors, 
                                                radius=RADIUS, 
                                                min_time_separation=min_time, 
                                                threshold=THRESHOLD)
    ## Add the loop closures to the graph
    loop_closures = pgo.add_loop_closures(poses=icp_poses, 
                            gt_poses=gt_poses, 
                            scans=scans, 
                            loop_closure=loop_closures)
    ## Optimize the graph
    result = pgo.optimize_graph()
    ## Get the optimized poses
    pgo_poses = []
    for i in range(len(gt_poses)):
        pose = result.atPose3(i)
        pgo_poses.append(pose.matrix())
    ## Plot the gt and optimized poses
    pgo_poses = np.array(pgo_poses)
    return pgo_poses, loop_closures, true_positives, false_positives

def test_method(poses:np.ndarray, descriptors:np.ndarray, 
                icp_poses:np.ndarray, scans:np.ndarray, filename:str,
                plot:bool, save:bool, method:str) -> None:
    tic = time.time()
    pgo_poses, loop_closures, true_positives, false_positives = run_pgo(gt_poses=poses, 
                        descriptors=descriptors, 
                        icp_poses=icp_poses, 
                        scans=scans,
                        min_time=MIN_SEPARATION)
    toc = time.time()
    ## Compute the ATE
    icp_ate_t, icp_ate_r = compute_ate(poses, icp_poses)
    pgo_ate_t, pgo_ate_r = compute_ate(poses, pgo_poses)
    ## Compute the RPE
    icp_rpe_t, icp_rpe_r = compute_rpe(poses, icp_poses)
    pgo_rpe_t, pgo_rpe_r = compute_rpe(poses, pgo_poses)
    ## Print the results
    print_ate_ape(icp_ate_t, icp_ate_r, pgo_ate_t, pgo_ate_r, 
                    icp_rpe_t, pgo_rpe_t, icp_rpe_r, pgo_rpe_r, 
                    loop_closures, tic, toc, method, true_positives, false_positives)
    if plot:
        ## Plot the trajectories
        plot_trajectories(gt_poses=poses, 
                          icp_poses=icp_poses, 
                          pgo_poses=pgo_poses, 
                          loop_closures=loop_closures)
    
    if save:
        ## Save the results to file
        save_results_to_file(filename=filename,
                            method=method,
                            num_frames=len(poses),
                            icp_ate_t=icp_ate_t, icp_ate_r=icp_ate_r, 
                            pgo_ate_t=pgo_ate_t, pgo_ate_r=pgo_ate_r, 
                            icp_rpe_t=icp_rpe_t, icp_rpe_r=icp_rpe_r, 
                            pgo_rpe_t=pgo_rpe_t, pgo_rpe_r=pgo_rpe_r, 
                            toc=toc, tic=tic, loop_closures=loop_closures,
                            pgo_poses=pgo_poses)

    return 

def plot_trajectories(gt_poses:np.ndarray, icp_poses:np.ndarray, 
                      pgo_poses:np.ndarray, loop_closures:List[Tuple[int, int]]) -> None:
    ## Plot the poses
    plt.figure()
    plt.plot(gt_poses[:, 0, 3], gt_poses[:, 1, 3], '-o', label='GT Poses', linewidth=1.5, markersize=1.5)
    plt.plot(icp_poses[:, 0, 3], icp_poses[:, 1, 3],'-o', label='KISS-ICP Poses', linewidth=1.5, markersize=1.5)
    plt.plot(pgo_poses[:, 0, 3], pgo_poses[:, 1, 3], '-o', label='PGO Poses', linewidth=1.5, markersize=1.5)
    ## Add the loop closures to the plot
    for i, j in loop_closures:
        plt.plot([gt_poses[i, 0, 3], icp_poses[j, 0, 3]], [gt_poses[i, 1, 3], icp_poses[j, 1, 3]], 'r-', alpha=0.25, linewidth=1)
    plt.legend()
    plt.grid(linestyle='--', alpha=0.5, linewidth=0.5)
    plt.show()
    return

def main():
    kitti_handler = KITTI_Handler(path_to_dataset=DATASET_PATH, verbose=True)
    ## Initialize the descriptor handler
    descriptor_handler = OT_Handler(dist_threshold=3.0,
                                    theta_threshold=0.8,
                                    device=DEVICE,
                                    verbose=True)
    ot = descriptor_handler.get_extracter(path_to_weights=PATH_TO_WEIGHTS)
    ## Loop through the sequences
    # SEQUENCES = ['08'] # override for testing
    for sequence in tqdm(SEQUENCES, desc='Dataset sequences' f'{SEQUENCES}', total=len(SEQUENCES)):
        print(f'Processing sequence: ' f'{sequence}')
        ## Path to depth maps
        path_to_depth = SEQUENCE_PATH + f'{sequence}' + '/depth_map/depth'
        ## Path to descriptors
        path_to_descriptors = SEQUENCE_PATH + f'{sequence}' + '/' + DESCRIPTOR_METHOD + '_descriptors_' + f'{sequence}' + '.npy'
        ## Load the KITTI dataset
        poses, scans, T_cam_to_velo = kitti_handler.load_kitti(sequence=sequence)
        ## Get the descriptors
        descriptors = descriptor_handler.get_descriptors(depth_maps_path=path_to_depth,
                                                            featureExtracter=ot,
                                                            from_file=FROM_FILE,
                                                            descriptor_path=path_to_descriptors)
        ## Chek if from file or optimize keyframes now
        if FROM_FILE:
            ## Load keyframe indices for every method
            spaciousness_keyframes = np.load(PATH_TO_SAVE + '/indices/' + DATASET + '_' + sequence + '_' + DESCRIPTOR_METHOD + '_spaciousness' + '.npy', allow_pickle=True)
            print(f'Spaciousness keyframes: ' f'{len(spaciousness_keyframes)}')
            entropy_keyframes = np.load(PATH_TO_SAVE + '/indices/' + DATASET + '_' + sequence + '_' + DESCRIPTOR_METHOD + '_entropy' + '.npy', allow_pickle=True)
            print(f'Entropy keyframes: ' f'{len(entropy_keyframes)}')
            fixed_1_keyframes = np.load(PATH_TO_SAVE + '/indices/' + DATASET + '_' + sequence + '_' + DESCRIPTOR_METHOD + '_fixed_distance_1' + '.npy', allow_pickle=True)
            print(f'Fixed distance 1 keyframes: ' f'{len(fixed_1_keyframes)}')
            fixed_2_keyframes = np.load(PATH_TO_SAVE + '/indices/' + DATASET + '_' + sequence + '_' + DESCRIPTOR_METHOD + '_fixed_distance_2' + '.npy', allow_pickle=True)
            print(f'Fixed distance 2 keyframes: ' f'{len(fixed_2_keyframes)}')
            fixed_3_keyframes = np.load(PATH_TO_SAVE + '/indices/' + DATASET + '_' + sequence + '_' + DESCRIPTOR_METHOD + '_fixed_distance_3' + '.npy', allow_pickle=True)
            print(f'Fixed distance 3 keyframes: ' f'{len(fixed_3_keyframes)}')
            msa_keyframes = np.load(PATH_TO_SAVE + '/indices/' + DATASET + '_' + sequence + '_' + DESCRIPTOR_METHOD + '_msa' + '.npy', allow_pickle=True)
            print(f'Optimized keyframes: ' f'{len(msa_keyframes)}')
            all_frames = np.arange(0, len(poses))
            print(f'All frames: ' f'{len(all_frames)}') 
        else:
            ## Initialize the methods
            spaciousness = SpaciousnessOnline(alpha=0.9,
                                        beta=0.1,
                                        delta_min=1.0,
                                        delta_mid=2.0,
                                        delta_max=3.0,
                                        theta_min=5.0,
                                        theta_mid=10.0,
                                        theta_max=15.0,
                                        queue_size=10)
            entropy = EntropyOnline(entropy_threshold= 0.03, #0.035, 
                                        dist_threshold=3.0)
            fixed_distance_1 = FixedDistanceOnline(dist_threshold=1)
            fixed_distance_2 = FixedDistanceOnline(dist_threshold=2)
            fixed_distance_3 = FixedDistanceOnline(dist_threshold=3)
            msa = RMIP_Online(window_size=WINDOW_SIZE,
                                    n_neighbours=2,
                                    delta_min=1.0, #1.0
                                    delta_max=2.0, #3.0
                                    alpha=ALPHA,
                                    beta=BETA)
            all_frames = np.arange(0, len(poses))
            ## Compute the keyframes
            for i in tqdm(range(len(poses)), desc='Computing keyframes', total=len(poses)):
                ## Get the current pose
                pose = poses[i]
                ## Get the current scan
                scan = scans[i]
                ## Get the current descriptor
                descriptor = descriptors[i]
                ## Remove nan values from the point cloud
                scan = scan[~np.isnan(scan).any(axis=1)]
                ## Check if the keyframe is valid with different methods
                if SAMPLING_METHODS[0]:
                    entropy.sample(pose=pose, scan=scan, descriptor=descriptor, index=i)
                if SAMPLING_METHODS[1]:
                    spaciousness.sample(pose=pose, scan=scan, descriptor=descriptor, index=i)
                if SAMPLING_METHODS[2]:
                    fixed_distance_1.sample(pose=pose, descriptor=descriptor, index=i)
                if SAMPLING_METHODS[3]:
                    fixed_distance_2.sample(pose=pose, descriptor=descriptor, index=i)
                if SAMPLING_METHODS[4]:
                    fixed_distance_3.sample(pose=pose, descriptor=descriptor, index=i)
                if SAMPLING_METHODS[-1]:
                    msa.sample_rmip(pose=pose, scan=scan, descriptor=descriptor, index=i)
            ## Save the keyframes
            if SAMPLING_METHODS[1]:
                spaciousness_keyframes = [spaciousness.keyframes[i].index for i in range(len(spaciousness.keyframes))]
                np.save(PATH_TO_SAVE + '/indices/' + DATASET + '_' + sequence + '_' + DESCRIPTOR_METHOD + '_spaciousness' + '.npy', spaciousness_keyframes)
                print(f'Spaciousness keyframes: ' f'{len(spaciousness_keyframes)}')
            ## Print the results
            if SAMPLING_METHODS[0]:
                entropy_keyframes = [entropy.keyframes[i].index for i in range(len(entropy.keyframes))]
                np.save(PATH_TO_SAVE + '/indices/' + DATASET + '_' + sequence + '_' + DESCRIPTOR_METHOD + '_entropy' + '.npy', entropy_keyframes)
                print(f'Entropy keyframes: ' f'{len(entropy_keyframes)}')
            if SAMPLING_METHODS[2]:
                fixed_1_keyframes = [fixed_distance_1.keyframes[i].index for i in range(len(fixed_distance_1.keyframes))]
                np.save(PATH_TO_SAVE + '/indices/' + DATASET + '_' + sequence + '_' + DESCRIPTOR_METHOD + '_fixed_distance_1' + '.npy', fixed_1_keyframes)
                print(f'Fixed distance 1 keyframes: ' f'{len(fixed_1_keyframes)}')
            if SAMPLING_METHODS[3]:
                fixed_2_keyframes = [fixed_distance_2.keyframes[i].index for i in range(len(fixed_distance_2.keyframes))]
                np.save(PATH_TO_SAVE + '/indices/' + DATASET + '_' + sequence + '_' + DESCRIPTOR_METHOD + '_fixed_distance_2' + '.npy', fixed_2_keyframes)
                print(f'Fixed distance 2 keyframes: ' f'{len(fixed_2_keyframes)}')
            if SAMPLING_METHODS[4]:
                fixed_3_keyframes = [fixed_distance_3.keyframes[i].index for i in range(len(fixed_distance_3.keyframes))]
                np.save(PATH_TO_SAVE + '/indices/' + DATASET + '_' + sequence + '_' + DESCRIPTOR_METHOD + '_fixed_distance_3' + '.npy', fixed_3_keyframes)
                print(f'Fixed distance 3 keyframes: ' f'{len(fixed_3_keyframes)}')
            ## Get keyframe indices to save
            if SAMPLING_METHODS[-1]:
                msa_keyframes = [msa.keyframes[i].index for i in range(len(msa.keyframes))]
                np.save(PATH_TO_SAVE + '/indices/' + DATASET + '_' + sequence + '_' + DESCRIPTOR_METHOD + '_msa' + '.npy', msa_keyframes)
                print(f'Optimized keyframes: ' f'{len(msa_keyframes)}')
        ## Load the KISS-ICP poses
        icp_poses = np.load(SEQUENCE_PATH + sequence + '/poses/velodyne_poses.npy')
        ## Transform the gt poses to the 0,0,0 frame
        poses = np.array([np.linalg.inv(poses[0]) @ pose for pose in poses])
        
        #############################
        ## Run PGO for All Samples ##
        #############################
        test_method(poses=poses, descriptors=descriptors, 
                    icp_poses=icp_poses, scans=scans, 
                    filename=PATH_TO_SAVE + '/results/' + DATASET + '_' + sequence + '_' + DESCRIPTOR_METHOD + '_' + str(RADIUS) + '.txt',
                    save=SAVE, plot=PLOT, method='All Samples')

        ##################################
        ## Repeat PGO for MSA keyframes ## 
        ##################################
        msa_keyframes = np.append(msa_keyframes, len(poses) - 1)
        msa_scans = [scans[i] for i in msa_keyframes]
        test_method(poses=poses[msa_keyframes], descriptors=descriptors[msa_keyframes],
                    icp_poses=icp_poses[msa_keyframes], scans=msa_scans, 
                    filename=PATH_TO_SAVE + '/results/' + DATASET + '_' + sequence + '_' + DESCRIPTOR_METHOD + '_' + str(RADIUS) + '.txt',
                    save=SAVE, plot=PLOT, method='MSA')
        
        ##########################################
        ## Repeat PGO for the Entropy keyframes ##
        ##########################################
        entropy_scans = [scans[i] for i in entropy_keyframes]
        test_method(poses=poses[entropy_keyframes], descriptors=descriptors[entropy_keyframes],
                    icp_poses=icp_poses[entropy_keyframes], scans=entropy_scans, 
                    filename=PATH_TO_SAVE + '/results/' + DATASET + '_' + sequence + '_' + DESCRIPTOR_METHOD + '_' + str(RADIUS) + '.txt',
                    save=SAVE, plot=PLOT, method='Entropy')
        
        ###############################################
        ## Repeat PGO for the Spaciousness keyframes ##
        ###############################################
        spaciousness_scans = [scans[i] for i in spaciousness_keyframes]
        test_method(poses=poses[spaciousness_keyframes], descriptors=descriptors[spaciousness_keyframes],
                    icp_poses=icp_poses[spaciousness_keyframes], scans=spaciousness_scans, 
                    filename=PATH_TO_SAVE + '/results/' + DATASET + '_' + sequence + '_' + DESCRIPTOR_METHOD + '_' + str(RADIUS) + '.txt',
                    save=SAVE, plot=PLOT, method='Spaciousness')

        ##############################################
        ## Repeat PGO for the Constant 1m keyframes ##
        ##############################################
        fixed_1_scans = [scans[i] for i in fixed_1_keyframes]
        test_method(poses=poses[fixed_1_keyframes], descriptors=descriptors[fixed_1_keyframes],
                    icp_poses=icp_poses[fixed_1_keyframes], scans=fixed_1_scans, 
                    filename=PATH_TO_SAVE + '/results/' + DATASET + '_' + sequence + '_' + DESCRIPTOR_METHOD + '_' + str(RADIUS) + '.txt',
                    save=SAVE, plot=PLOT, method='Constant 1m')
        
        ##############################################
        ## Repeat PGO for the Constant 2m keyframes ##
        ##############################################
        fixed_2_scans = [scans[i] for i in fixed_2_keyframes]
        test_method(poses=poses[fixed_2_keyframes], descriptors=descriptors[fixed_2_keyframes],
                    icp_poses=icp_poses[fixed_2_keyframes], scans=fixed_2_scans, 
                    filename=PATH_TO_SAVE + '/results/' + DATASET + '_' + sequence + '_' + DESCRIPTOR_METHOD + '_' + str(RADIUS) + '.txt',
                    save=SAVE, plot=PLOT, method='Constant 2m')
        
        ##############################################
        ## Repeat PGO for the Constant 3m keyframes ##
        ##############################################
        fixed_3_scans = [scans[i] for i in fixed_3_keyframes]
        test_method(poses=poses[fixed_3_keyframes], descriptors=descriptors[fixed_3_keyframes],
                    icp_poses=icp_poses[fixed_3_keyframes], scans=fixed_3_scans, 
                    filename=PATH_TO_SAVE + '/results/' + DATASET + '_' + sequence + '_' + DESCRIPTOR_METHOD + '_' + str(RADIUS) + '.txt',
                    save=SAVE, plot=PLOT, method='Constant 3m')

if __name__ == '__main__':
    main()