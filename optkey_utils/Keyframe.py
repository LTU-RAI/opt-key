"""
    This module contains the Keyframe and Keyframes classes for managing keyframe data.
    TODO: Add support for images.
"""

from dataclasses import dataclass
import numpy as np
from typing import List
from scipy.spatial import KDTree


@dataclass
class Keyframe:
    index: int = None
    time: float = 0.0
    imu: np.ndarray = None
    pose: np.ndarray = None ## 4x4 matrix
    descriptor: np.ndarray = None ## Mx1 vector
    pointcloud: np.ndarray = None  ## Nx3 matrix

    def __str__(self) -> str:
        return f'Keyframe(time={self.time}, imu={self.imu}, pose={self.pose}, pointcloud={self.pointcloud.shape}, desriptor={self.desriptor})'

@dataclass
class Keyframes:
    ## A class that holds the keyframes
    keyframes = []

    def __post_init__(self):
        self.keyframes = []

    def __str__(self) -> str:
        ## Print a list of the indices of the keyframes
        return f'Keyframes: {[keyframe.index for keyframe in self.keyframes]}'
    
    def __len__(self) -> int:
        return len(self.keyframes)
    
    def __getitem__(self, index:int) -> Keyframe:
        return self.keyframes[index]
    
    def __setitem__(self, index:int, keyframe:Keyframe) -> None:
        self.keyframes[index] = keyframe
        return None
    
    def _delete(self, index:int) -> None:
        del self.keyframes[index]
        return None
    
    def _append(self, keyframe:Keyframe) -> None:
        self.keyframes.append(keyframe)
        return None
    
    def _get_descriptors(self, indices:int) -> np.ndarray:
        return np.array([self.keyframes[i].descriptor for i in indices])
    
    def _get_poses(self, indices:int) -> np.ndarray:
        return np.array([self.keyframes[i].pose for i in indices])
    
    def _get_pointclouds(self, indices:int) -> np.ndarray:
        return np.array([self.keyframes[i].pointcloud for i in indices])
    
    def _get_neighbours(self, pose:np.ndarray, n:int=1) -> List[Keyframe]:
        ## Based on the index check the distance between the keyframes and return the n neighbours using a kd-tree
        ## Initialize the tree
        tree = KDTree(np.array([keyframe.pose[:3, 3] for keyframe in self.keyframes]))
        ## Get the distances and indices
        distances, indices = tree.query(pose[:3, 3], k=n)
        ## Return the neighbours
        return [self.keyframes[i] for i in indices[1:]]
    


