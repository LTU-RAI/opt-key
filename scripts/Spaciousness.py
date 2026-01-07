"""
    This script contains the Spaciousness class that calculates the spaciousness of the environment
    and samples keyframes based on it.
"""

import numpy as np
from tqdm import tqdm
from typing import List
from collections import deque
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from optkey_utils.Keyframe import Keyframe, Keyframes

class SpaciousnessOnline:
    def __init__(self, alpha:float=0.9,
                       beta:float=0.1,
                       delta_min:float=1.0,
                       delta_mid:float=3.0,
                       delta_max:float=5.0,
                       theta_min:float=3.0,
                       theta_mid:float=5.0,
                       theta_max:float=10.0,
                       queue_size:int=10):
        """
            This class is used to calculate the spaciousness of the environment and sample keyframes.
            Args:
                poses: A numpy array with the poses
                scans: A List with the LiDAR scans
                alpha: A float with the alpha parameter
                beta: A float with the beta parameter
                delta_min: A float with the minimum delta parameter
                delta_mid: A float with the middle delta parameter
                delta_max: A float with the maximum delta parameter
                theta_min: A float with the minimum theta parameter
                theta_mid: A float with the middle theta parameter
                theta_max: A float with the maximum theta parameter
                queue_size: An integer with the size of the queue
                verbose: A boolean to indicate if the class should print information
            Returns:
                A bool if the keyframe should be added
        """
        self.alpha = alpha
        self.beta = beta
        self.delta_min = delta_min
        self.delta_mid = delta_mid
        self.delta_max = delta_max
        self.theta_min = theta_min
        self.theta_mid = theta_mid
        self.theta_max = theta_max
        self.queue_size = queue_size
        self.spaciousness = deque(maxlen=self.queue_size)
        self.keyframes = Keyframes() 
        self.keyframe = Keyframe()
        self.index = 0

    def median_eucledian_distance(self, points:np.array) -> np.float16:
        eucledian_dist = np.linalg.norm(points, axis=1)
        return np.mean(eucledian_dist)
    
    def get_threshold(self, spaciousness:np.float16) -> int:
        ## Maybe try an adaptive controller with the information gain
        if spaciousness > self.theta_max: 
            return self.delta_max
        if spaciousness > self.theta_mid and spaciousness <= self.theta_max:
            return self.delta_mid
        if spaciousness > self.theta_min and spaciousness <= self.theta_mid: 
            return self.delta_min
        else: 
            return spaciousness
        
    def sample(self, pose:np.array, scan:np.ndarray, descriptor:np.ndarray, index:int) -> bool:
        """ Sample the poses by keeping a distance threshold between them based on the entropy of the LiDAR scans.
            Args:
                poses: A numpy array with the poses (4x4 transformation matrices)
                scans: A numpy array with the LiDAR scans (Nx3 matrices)
            Returns:
                A bool if the keyframe should be added
        """
        ## Check if the keyframes are empty
        if self.index == 0:
            ## Add the first keyframe
            self.keyframe = Keyframe(index=index,
                                     time=0.0, 
                                     imu=None, 
                                     pose=pose, 
                                     descriptor=descriptor, 
                                     pointcloud=scan)
            self.keyframes._append(self.keyframe)
            self.index += 1
            self.spaciousness.append(self.median_eucledian_distance(scan))
            return True
        ## Check if the entropy of the current and the previous scan is greater than the threshold
        ## Get previous scan from the nearest keyframe
        prev_space = self.median_eucledian_distance(self.keyframes[self.index-1].pointcloud)
        curr_space = self.median_eucledian_distance(scan)
        ## Get distance to the nearest keyframe
        distance = np.linalg.norm(pose[:3, 3] - self.keyframes[self.index-1].pose[:3, 3])
        ## Append the spaciousness
        self.spaciousness.append(self.alpha*self.spaciousness[-1] + self.beta*curr_space)
        ## Get corresponding threshold
        threshold = self.get_threshold(self.spaciousness[-1])
        ## Check the information gain
        if distance > threshold:
            ## Add the keyframe
            self.keyframe = Keyframe(index=index,
                                     time=0.0, 
                                     imu=None, 
                                     pose=pose, 
                                     descriptor=descriptor, 
                                     pointcloud=scan)
            self.keyframes._append(self.keyframe)
            self.index += 1
            return True
        return False


class Spaciousness:
    def __init__(self, poses:np.ndarray,
                       scans:List[np.ndarray],
                       alpha:float=0.9,
                       beta:float=0.1,
                       delta_min:float=1.0,
                       delta_mid:float=3.0,
                       delta_max:float=5.0,
                       theta_min:float=3.0,
                       theta_mid:float=5.0,
                       theta_max:float=10.0,
                       queue_size:int=10):
        """
            This class is used to calculate the spaciousness of the environment and sample keyframes.
            Args:
                poses: A numpy array with the poses
                scans: A List with the LiDAR scans
                alpha: A float with the alpha parameter
                beta: A float with the beta parameter
                delta_min: A float with the minimum delta parameter
                delta_mid: A float with the middle delta parameter
                delta_max: A float with the maximum delta parameter
                theta_min: A float with the minimum theta parameter
                theta_mid: A float with the middle theta parameter
                theta_max: A float with the maximum theta parameter
                queue_size: An integer with the size of the queue
                verbose: A boolean to indicate if the class should print information
            Returns:
                A numpy array with the keyframe indices
        """
        self.poses = poses
        self.scans = scans
        self.alpha = alpha
        self.beta = beta
        self.delta_min = delta_min
        self.delta_mid = delta_mid
        self.delta_max = delta_max
        self.theta_min = theta_min
        self.theta_mid = theta_mid
        self.theta_max = theta_max
        self.queue_size = queue_size
        self.num_poses = len(self.poses)

    ## Clear variables.
    def clear_variables(self):
        del self.poses, self.scans

    ## Calculate median Eucledian distance from 3D point list
    def median_eucledian_distance(self, points:np.array) -> np.float16:
        eucledian_dist = np.linalg.norm(points, axis=1)
        return np.mean(eucledian_dist)

    ## Calculate median Eucledian d10sed on spaciousness
    def get_threshold(self, spaciousness:np.float16) -> int:
        ## Maybe try an adaptive controller with the information gain
        if spaciousness > self.theta_max: 
            return self.delta_max
        if spaciousness > self.theta_mid and spaciousness <= self.theta_max:
            return self.delta_mid
        if spaciousness > self.theta_min and spaciousness <= self.theta_mid: 
            return self.delta_min
        else: 
            return spaciousness
        
    def sample_keyframes(self) -> np.ndarray:
        """ Sample the poses by keeping a distance threshold between them based on the entropy of the LiDAR scans.
            Returns:
                A list with the indices of the keyframes
        """
        ## Initialize the keyframes
        indices = [0]
        ## Initialize spaciousness
        spaciousness = deque(maxlen=self.queue_size)
        ## Append the first spaciousness
        spaciousness.append(self.median_eucledian_distance(self.scans[0]))
        ## Loop through the poses
        for i in tqdm(range(1, self.num_poses), desc='Spaciousness: Sampling keyframes', total=self.num_poses):
            ## Get the distance between the poses
            distance = self.median_eucledian_distance(self.scans[i])
            ## Append to the spaciousness
            spaciousness.append(self.alpha*spaciousness[-1] + self.beta*distance)
            ## Get corresponding threshold
            threshold = self.get_threshold(spaciousness[-1])
            ## If the distance between previous and current pose is greater than the threshold, add the keyframe
            if np.linalg.norm(self.poses[i, :3, 3] - self.poses[indices[-1], :3, 3]) > threshold:
                indices.append(i)
        return indices