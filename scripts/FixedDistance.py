"""
    This script contains the FixedDistance class used to sample keyframes based on a fixed distance threshold.
"""

import numpy as np
from tqdm import tqdm
from typing import List
from utils.Keyframe import Keyframe, Keyframes

class FixedDistanceOnline:
    def __init__(self, dist_threshold:float):
        """
            This class is used to sample keyframes based on a fixed distance threshold.
            Args:
                dist_threshold: A float with the distance threshold
        """
        self.dist_threshold = dist_threshold
        self.keyframes = Keyframes()
        self.keyframe = Keyframe()   
        self.index = 0

    ## Sample keyframes based on entropy
    def sample(self, pose:np.array, descriptor:np.ndarray, index:int) -> bool:
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
                                     pointcloud=None)
            self.keyframes._append(self.keyframe)
            self.index += 1
        else:
            ## Check the distance between the poses
            if np.linalg.norm(pose[:3, 3] - self.keyframes[-1].pose[:3, 3]) >= self.dist_threshold:
                ## Add the keyframe
                self.keyframe = Keyframe(index=index,
                                         time=0.0, 
                                         imu=None, 
                                         pose=pose, 
                                         descriptor=descriptor, 
                                         pointcloud=None)
                self.keyframes._append(self.keyframe)
                self.index += 1
                return True
        return False


class FixedDistance:
    def __init__(self, poses:np.ndarray,
                       scans:List[np.ndarray],
                       dist_threshold:float):
        """
            
            
        """
        self.poses = poses
        self.scans = scans
        self.dist_threshold = dist_threshold
        self.num_poses = len(self.poses)
        
    ## Clear variables.
    def clear_variables(self):
        del self.poses, self.scans

    ## Sample keyframes based on distance
    def sample_keyframes(self) -> List[int]:
        ## Initialize the keyframes
        indices = [0]
        ## Loop through the poses
        for i in tqdm(range(1, self.num_poses), desc='Fixed Distance: Sampling keyframes', total=self.num_poses):
            ## If the distance is greater than the threshold, add the keyframe
            if np.linalg.norm(self.poses[i, :3, 3] - self.poses[indices[-1], :3, 3]) > self.dist_threshold:
                indices.append(i)
        return indices