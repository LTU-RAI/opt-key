"""
    This script contains the Entropy and EntropyOnline classes used to sample keyframes based on entropy thresholds.
"""

import numpy as np
from tqdm import tqdm
from typing import List
from scipy import stats
from utils.Keyframe import Keyframe, Keyframes


class EntropyOnline:
    def __init__(self, entropy_threshold:float=0.05,
                       dist_threshold:float=5.0):
        """
            This class is used to calculate the entropy of the environment and sample keyframes.
            Args:
                entropy_threshold: A float with the entropy threshold
                dist_threshold: A float with the distance threshold            
        """
        self.entropy_threshold = entropy_threshold
        self.dist_threshold = dist_threshold
        self.keyframes = Keyframes()
        self.keyframe = Keyframe()   
        self.index = 0 
        
    ## Calculate information gain based on entropy
    def information_gain(self, current:np.array, relative:np.array=None) -> np.float16:
        # entropy = sum(-point*np.log(point))
        return -np.log(stats.entropy(pk=current, qk=relative, axis=0))

    ## Calculate point cloud entropy
    def get_entropy(self, points:np.array) -> np.float16:
        hist = np.histogramdd(np.array(points))[0]
        hist /= hist.sum()
        hist = hist.flatten()
        hist = hist[hist.nonzero()]
        return -0.5 * np.sum(hist * np.log2(hist))
        
    ## Sample keyframes based on entropy
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
            return True
        ## Check if the entropy of the current and the previous scan is greater than the threshold
        ## Get previous scan from the nearest keyframe
        try:
            prev_entropy = self.get_entropy(self.keyframes[self.index-1].pointcloud)
            curr_entropy = self.get_entropy(scan)
        except:
            return False
        ## Get distance to the nearest keyframe
        distance = np.linalg.norm(pose[:3, 3] - self.keyframes[self.index-1].pose[:3, 3])
        ## Check the information gain
        if np.abs(1 - curr_entropy/prev_entropy) > self.entropy_threshold or distance > self.dist_threshold:
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

class Entropy:
    def __init__(self, poses:np.ndarray,
                       scans:List[np.ndarray],
                       entropy_threshold:float=0.05,
                       dist_threshold:float=5.0):
        """
            This class is used to calculate the entropy of the environment and sample keyframes.
            Args:
                poses: A numpy array with the poses
                scans: A List with the LiDAR scans
                entropy_threshold: A float with the entropy threshold
                dist_threshold: A float with the distance threshold
                verbose: A boolean to indicate if the class should print information
            Returns:
                A numpy array with the keyframe indices
            
        """
        self.poses = poses
        self.scans = scans
        self.entropy_threshold = entropy_threshold
        self.dist_threshold = dist_threshold
        self.num_poses = len(self.poses)
    
    ## Clear variables.
    def clear_variables(self):
        del self.poses, self.scans
        
    ## Calculate information gain based on entropy
    def information_gain(self, current:np.array, relative:np.array=None) -> np.float16:
        # entropy = sum(-point*np.log(point))
        return -np.log(stats.entropy(pk=current, qk=relative, axis=0))

    ## Calculate point cloud entropy
    def get_entropy(self, points:np.array) -> np.float16:
        hist = np.histogramdd(np.array(points))[0]
        hist /= hist.sum()
        hist = hist.flatten()
        hist = hist[hist.nonzero()]
        return -0.5 * np.sum(hist * np.log2(hist))
        
    ## Sample keyframes based on entropy
    def sample_keyframes(self) -> List[int]:
        """ Sample the poses by keeping a distance threshold between them based on the entropy of the LiDAR scans.
            Args:
                poses: A numpy array with the poses
                scans: A numpy array with the LiDAR scans
            Returns:
                A list with the indices of the keyframes
        """
        ## Initialize the keyframes
        indices = [0]
        ## Get the first entropy
        last_entropy = self.get_entropy(self.scans[0])
        ## Loop through the poses
        for i in tqdm(range(1, self.num_poses), desc='Entropy: Sampling keyframes', total=self.num_poses):
            ## Get the distance between the poses
            curr_entropy = self.get_entropy(self.scans[i])
            ## Compare with previous entropy
            if np.abs(1 - curr_entropy/last_entropy) > self.entropy_threshold or np.linalg.norm(self.poses[i, :3, 3] - self.poses[indices[-1], :3, 3]) > self.dist_threshold:
                indices.append(i)
                last_entropy = curr_entropy
        return indices