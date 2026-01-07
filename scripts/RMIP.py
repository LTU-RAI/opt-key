"""
    This script contains the RMIP_Online class used to sample keyframes based on 
        redundancy minimization and information preservation.
"""

import itertools, sys, time
import numpy as np
from typing import List, Tuple
sys.path.append('.')
from optkey_utils.Keyframe import Keyframe, Keyframes

class RMIP_Online:
    def __init__(self, window_size:int=10,
                       n_neighbours:int=10,
                       delta_min:float=1.0,
                       delta_max:float=5.0,
                       alpha:float=1.0,
                       beta:float=1.0,
                       min_mult:float=0.1,
                       max_mult:float=5.0,
                       verbose:bool=False):
        """
            This class is used to handle the sliding window algorithm in an online fashion.
            Args:
                desc_handler: A DescriptorHandler object
                window_size: An integer with the size of the window
                delta_min: A float with the minimum distance between poses
                delta_max: A float with the maximum distance between poses
                alpha: A float with the weight for the AUC
                beta: A float with the weight for the memory
                verbose: A boolean to indicate if the class should print information
            Returns:
                A bool if the keyframe should be added
        """
        self.window_size = window_size
        self.n_neighbours = n_neighbours
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.min_mult = min_mult
        self.max_mult = max_mult
        self.alpha = alpha
        self.beta = beta
        self.verbose = verbose
        self.curr_windows = []
        self.keyframes = Keyframes()
        self.keyframe = Keyframe()
        self.window_keyframes = Keyframes()
        self.index = 0
        self.window_index = 0
        self.global_avg_distance = 0.0

    ## Calculate the average distance between poses in the window.
    def average_distance(self, poses:np.ndarray) -> float:
        distances = []
        for i in range(1, len(poses)):
            pose_diff = np.linalg.norm(poses[i][0:3, 3] - poses[i-1][0:3, 3])
            distances.append(pose_diff)
        return np.mean(distances)

    ## Check if the constraints are satisfied.
    def constraints(self, poses:np.ndarray) -> bool:
        for i in range(1, len(poses)):
            pose_diff = np.linalg.norm(poses[i][0:3, 3] - poses[i-1][0:3, 3])
            if pose_diff < self.delta_min or pose_diff > self.delta_max: # or psi_diff > self.psi:
                return False
        return True

    ## Generate all of the subsets of the window.
    def generate_subsets(self, method:str) -> List:
        num_poses = self.window_index
        subsets_indexes = []
        if method == 'bounded':
            for r in range(2, num_poses + 1):
                for subset_indexes in itertools.combinations(range(1, num_poses - 1), r - 2):
                    subset_indexes = [0] + list(subset_indexes) + [num_poses - 1]
                    subset_poses = [self.window_keyframes[i].pose for i in subset_indexes]
                    if self.constraints(subset_poses):
                        subsets_indexes.append(subset_indexes)
        elif method == 'adaptive':
            for r in range(2, num_poses):
                for subset_indexes in itertools.combinations(range(1, num_poses), r - 1):
                    subset_indexes = [0] + list(subset_indexes)
                    subset_poses = [self.window_keyframes[i].pose for i in subset_indexes]
                    if self.constraints(subset_poses):
                        subsets_indexes.append(subset_indexes)
        ## If no subsets are found, return the first and last pose
        if len(subsets_indexes) == 0:
            subsets_indexes = [[0, num_poses - 1]]
        return subsets_indexes

    ## Objective function for the subset of poses.
    def objective_function(self, rmt:float, ipt) -> float:
        # return (self.alpha + rmt) / (ipt - self.beta)
        return (self.alpha + rmt) / (self.beta + ipt)
    
    ## Calculate similarities.
    def calculate_similarity(self, query_descr:np.ndarray, map_descr:np.ndarray) -> np.ndarray:
        ## If query and map descriptors are one dimensional, calculate the similarity
        if query_descr.ndim == 1 and map_descr.ndim == 1:
            return 1 / (1 + np.linalg.norm(query_descr - map_descr))
        ## If query and map descriptors are multi dimensional, calculate the similarity
        elif query_descr.ndim > 1 and map_descr.ndim > 1:
            return 1 / (1 + np.linalg.norm(query_descr - map_descr, axis=1))

    ## Calculate the redundancy minimization term.
    def redundancy_minimization_term(self, subset:np.array) -> float:
        subset_desc = self.window_keyframes._get_descriptors(subset)
        ## Get the similarity between consecutive descriptors
        subset_similarity = np.zeros(len(subset_desc) - 1)
        ## Calculate similarity between consecutive descriptors
        subset_similarity = self.calculate_similarity(subset_desc[1:], subset_desc[:-1])
        ## Get the Redundancy Minimization Term
        return np.sum(subset_similarity)/len(subset_similarity)
    
    ## Compute the Jacobian of the current window.
    def jacobian(self, curr_window:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
            Compute the Jacobian of the current window. 
            J = dD/dx where D is the descriptor and x is the distance between poses.
            D is a matrix of size (n, m) where n is the number of features and m is the window size.
            P is a vector of size m and is the 4x4 transformation matrix.
            x is a vector of size m and is the distance between poses.
            Args:
                curr_window: A numpy array with the current window of poses
            Returns:
                J: A numpy array with the Jacobian of the current window
                C: A numpy array with the covariance matrix
                lamda: A numpy array with the eigenvalues of the covariance matrix
                v: A numpy array with the eigenvectors of the covariance matrix
        """
        ## Get the descriptors
        D = np.transpose(self.window_keyframes._get_descriptors(curr_window))
        ## If the descriptors are LxMxN, flatten them to L*MxN
        if D.ndim > 2:
            D = np.reshape(D.T, (D.T.shape[0], -1)).T
        ## Get the poses
        P = self.window_keyframes._get_poses(curr_window)
        ## Get distance between poses keep same size as D
        x = np.linalg.norm(P[:, 0:3, 3] - P[0, 0:3, 3], axis=1)
        # x = np.linalg.norm(np.diff(P[:, 0:3, 3], axis=0), axis=1)
        # x = np.insert(x, 0, 0)
        ## Get the jacobian
        J = np.zeros((np.shape(D)[0], np.shape(P)[0]))
        try:
            J = np.gradient(D, x, axis=1)
        except:
            dx = np.random.rand(len(x))
            x = x + 1e-4*dx
            J = np.gradient(D, x, axis=1)
        ## Compute the J^T*J to get the covariance matrix C
        C = np.dot(J.T, J)
        ## Compute the eigenvalues and eigenvectors of the J^T*J matrix
        try:
            ## Check if the matrix contains infs or nans
            if np.isnan(C).any() or np.isinf(C).any():
                C = np.nan_to_num(C)
                # print(np.shape(C))
                C = np.clip(C, -1e6, 1e6)
            lamda, v = np.linalg.eigh(C)
        except np.linalg.LinAlgError:
            # print(f'Error in eigenvalues computation')
            lamda = np.zeros(np.shape(C)[0])
            v = np.zeros(np.shape(C))
        return J, C, lamda, v

    ## Calculate the information preservation term.
    def information_preservation_term(self, subset:np.array) -> float:
        subset_desc = self.window_keyframes._get_descriptors(subset)
        ## If descriptors are NxLxM flatten them to NxL*M
        if subset_desc.ndim > 2:
            subset_desc = np.reshape(subset_desc, (subset_desc.shape[0], -1))
        ## Get the Jacobian, covariance matrix, eigenvalues and eigenvectors
        J, C, lamda, v = self.jacobian(subset)
        ## Set negative eigenvalues to zero
        lamda[lamda < 0] = 0
        ## Add a small value to avoid division by zero
        lamda += 1e-6
        ## Get the Information Preservation Term
        # ipt = np.sum(lamda * np.sum(v**2, axis=0))/len(lamda)
        ## Transform the descriptors using the eigenvectors
        transformed_descriptors = (np.sqrt(lamda)*np.dot(subset_desc.T, v))
        # transformed_descriptors = (np.sqrt(lamda)*np.dot(v, subset_desc))
        ## Calculate the distances between the transformed descriptors
        # distances = np.linalg.norm(transformed_descriptors[:, :, np.newaxis] - transformed_descriptors[:, np.newaxis, :], axis=0)
        distances = np.linalg.norm(transformed_descriptors[:, 1:] - transformed_descriptors[:, :-1], axis=0)
        # Normalize the distances
        distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances) + 1e-6)
        ## Calculate transformed descriptors similarity
        # transformed_similarity = np.zeros(len(transformed_descriptors) - 1)
        # ## Calculate similarity between transformed descriptors
        # transformed_similarity = self.calculate_similarity(transformed_descriptors[1:], transformed_descriptors[:-1])
        ## Sum distances to obtain information preservation term
        ipt = np.sum(distances)/len(distances)
        # ipt = np.sum(transformed_similarity)#/len(transformed_similarity)
        return ipt

    ## Optimize the objective function for the current window.
    def optimize_objective(self, method:str) -> np.ndarray:
        """
            Optimize the objective function for the current window.
        """
        ## Generate subsets
        subsets = self.generate_subsets(method=method)
        ## Check the distances in every subset and remove the ones that do not satisfy the constraints
        subsets = [subset for subset in subsets if self.constraints(self.window_keyframes._get_poses(subset))]
        ## If no subsets are found, return the first pose only
        if len(subsets) == 0:
            return [0,self.window_size-1]
        ## Initialize the rates
        _rmt = []
        _ipt = []
        ## Loop through the subsets and get the rate of feature change
        for subset in subsets:
            ## Get the Redundancy Minimization Term
            rmt = self.redundancy_minimization_term(subset)
            _rmt.append(rmt)
            ## Get the Information Preservation Term
            ipt = self.information_preservation_term(subset)
            _ipt.append(ipt)
            ## Objective score
            # subset_scores.append(self.objective_function(rmt, -ipt))
            if self.verbose:
                print(f'Redundancy Minimization Term: {rmt}')
                print(f'Information Preservation Term: {ipt}')
        ## Normalize each subset rate with respect to the maximum and minimum values
        _rmt = np.asarray(_rmt)
        if not np.max(_rmt) - np.min(_rmt) == 0:
            _rmt = (_rmt - np.min(_rmt)) / (np.max(_rmt) - np.min(_rmt))
        else:
            _rmt = np.ones(len(_rmt))
        ## Normalize each subset rate with respect to the maximum and minimum values
        _ipt = np.asarray(_ipt)
        if not np.max(_ipt) - np.min(_ipt) == 0:
            _ipt = (_ipt - np.min(_ipt)) / (np.max(_ipt) - np.min(_ipt))
        else:
            _ipt = np.ones(len(_ipt))
        ## Get the best subset
        optimal_subset = subsets[np.argmin(self.objective_function(_rmt, _ipt))]
        return optimal_subset

    ## Redundancy minimization and Information preservationusing main.
    def sample_rmip(self, pose:np.ndarray, scan:np.ndarray, descriptor:np.ndarray, index:int) -> float:
        """
            Add keyframes to the current window until it's full and the optimize it.
            Args:
                pose: A numpy array with the pose (4x4 transformation matrix)
                scan: A numpy array with the LiDAR scan (Nx3 matrix)
                descriptor: A numpy array with the descriptor (Mx1 vector)
            Returns:
                time: A float with the time
        """
        toc = time.time()
        # print((f'---------------------------------'))
        # print(f'Current sample index: {index}')
        ## Add the first keyframe
        if self.index == 0:
            self.keyframe = Keyframe(index=index,
                                    time=0.0, 
                                    imu=None, 
                                    pose=pose, 
                                    descriptor=descriptor, 
                                    pointcloud=scan)
            self.keyframes._append(self.keyframe)
            self.index += 1
            self.window_keyframes._append(self.keyframe)
            self.window_index += 1
        ## Check window size
        if self.window_index < self.window_size:#-1:
            ## Add the keyframe to the window
            self.keyframe = Keyframe(index=index,
                                    time=0.0, 
                                    imu=None, 
                                    pose=pose, 
                                    descriptor=descriptor, 
                                    pointcloud=scan)
            ## Check if distance between the current keyframe and the last keyframe is not zero
            if np.linalg.norm(self.keyframe.pose[0:3, 3] - self.window_keyframes[-1].pose[0:3, 3]) >= 0.01:
                self.window_keyframes._append(self.keyframe)
                self.window_index += 1
                # print(f'Current window idnex: {self.window_index}')
                # print(f'Window keyframes: {self.window_keyframes}')
            ## Check if the are neighbours to include in the window
            # if self.index >= 10:
                # temp_keyframes = self.keyframes._get_neighbours(self.window_keyframes[-1].pose, n=self.n_neighbours)
                # ## Check if there are any neighbours that are within delta_max and they are not already in the window
                # for keyframe in temp_keyframes:
                #     if np.linalg.norm(keyframe.pose[0:3, 3] - self.window_keyframes[-1].pose[0:3, 3]) <= self.delta_max:
                #         if  keyframe.index not in [key.index for key in self.window_keyframes]:
                #             self.window_keyframes._append(keyframe)
                #             self.window_index += 1
                #             # print(f'Found neighbour: {keyframe.index}')
                #             # print(f'Added to current window with idnex: {self.window_index}')
                #             # print(f'New Window keyframes: {self.window_keyframes}')
        # print(f'Window index: {self.window_index}, Window size: {self.window_size}')
        if self.window_index >= self.window_size:#-1:
            
            # Get adaptive bounds
            avg_dist = self.average_distance(self.window_keyframes._get_poses(range(len(self.window_keyframes))))
            if self.global_avg_distance == 0.0:
                self.global_avg_distance = avg_dist
            else:
                # if avg_dist > 0.01:
                self.global_avg_distance = (self.global_avg_distance + avg_dist) / 2
            
            self.delta_min = self.min_mult * self.global_avg_distance if self.global_avg_distance > 0.01 else 0.01
            self.delta_max = self.max_mult * self.global_avg_distance if self.global_avg_distance < 5.00 else 5.00
            # self.delta_max = 3.0 * self.global_avg_distance #if self.global_avg_distance < 5.00 else 5.00
            # self.delta_max = 2.0 * self.global_avg_distance #if self.global_avg_distance < 0.5 else 0.5
            # if self.delta_max < 2.00: self.delta_max = 2.00
            
            # print(f'Current avg distance: {np.round(avg_dist,3)}')
            # print(f'Global average distance: {np.round(self.global_avg_distance,3)}')
            # print(f'Adaptive bounds: delta_min: {np.round(self.delta_min, 2)}, delta_max: {np.round(self.delta_max, 2)}')
            
            # print(f'Optimizing window')
            ## Get the optimal subset from the latest window
            tic = time.time()
            optimal_subset = self.optimize_objective(method='adaptive')
            toc = time.time()
            print(f'Time to optimize window: {toc - tic}')
            # print(f'Optimal subset: {optimal_subset}')
            ## Add the optimal subset to the keyframes except the first one
            for i in optimal_subset[1:]:
                # print(f'Optimal subset index to add: {i}')
                # print(f'Keyframe to add: {self.window_keyframes}')
                ## Check if the keyframe is already in the keyframes
                # print(f'Adding keyframe {self.window_keyframes[i].index} to keyframes')
                if self.window_keyframes[i].index not in [key.index for key in self.keyframes]:
                    self.keyframes._append(self.window_keyframes[i])
            ## Clear the window keyframes before the last optimized keyframe
            for i in range(optimal_subset[-1]):
                self.window_keyframes._delete(0)
            self.window_index = len(self.window_keyframes) - 1
            self.index = len(self.keyframes) - 1
            # print(f'Window keyframes after optimization: {self.window_keyframes}')
            # print(f'New window index: {self.window_index}')
            # print(f'Current Global Index: {self.index}')
            # print(f'All Keyframes: {self.keyframes}')
            # print((f'---------------------------------'))
        tic = time.time()
        # print(f'RMIP time: {tic - toc}')
        return tic - toc