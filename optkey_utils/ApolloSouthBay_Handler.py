"""
    This is a class to handle the Apollo-SouthBay dataset.
"""

import numpy as np
import pypcd4
from tqdm import tqdm
import os, cv2
from typing import List
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


class ApolloSouthBay_Handler:
    def __init__(self, path_to_dataset:str, 
                       verbose:bool=False):
        """
            This is a class to handle the Apollo-SouthBay dataset, containing any number of sequences.
            Args:
                path_to_dataset: str, path to the Apollo-SouthBay dataset
                sequences: List[str], list of sequences to be used
                verbose: bool, whether to print information or not
            Note:
                The Apollo-SouthBay dataset is expected to have the following structure:
                path_to_dataset
                ├─Apollo-SouthBay
                |   |── MapData
                |   |   ├─BaylandsToSeafood
                |   |   |   |── 2018-09-26
                |   |   |   |   |── poses
                |   |   |   |   |   |── gt_poses.txt
                |   |   |   |   |── pcds
                |   |   |   |   |   |── 000000.pcd
                |   |   |   |   |   |── ...
                |   |   |   |   |── depth_map
                |   |   |   |   |   |── depth
                |   |   |   |   |   |   |── 000000.png
                |   |   |   |   |   |   |── ...
                |   |   |   |   |── ot_descriptors_BaylandsToSeafood.npy
                |   |   |   |   |── ...
                |   |   |   |── ... 
                |   |   |── ColumbiaPark
                |   |   |   |── 2018-09-21
                |   |   |   |   |── 1
                |   |   |   |   |   |── poses
                |   |   |   |   |   |   |── gt_poses.txt
                |   |   |   |   |   |── pcds
                |   |   |   |   |   |   |── 000000.pcd
                |   |   |   |   |   |   |── ...
                |   |   |   |   |   |── ...
                |   |   |   |   |── ...
                |   |   |   |── ...
                |   |── TrainData
                |   |   |── ...
                |   |── TestData
                |   |   |── ...
        """
        self.path_to_dataset = path_to_dataset
        self.verbose = verbose

    ## Load the Apollo-SouthBay dataset.
    def load_apollo_southbay(self, session:str, sequence:str, type_data:str='MapData'):
        """ Read the Apollo-SouthBay dataset and return the pointclouds, poses, depth maps, and calibration data.
            Includes calibration data to properly project poses and pointclouds.
            Args:
                session: str, the session to be loaded
                sequence: str, the sequence to be loaded
                type_data: str, the type of data to be loaded -> MapData, TrainData, TestData
            Returns:
                poses: np.ndarray, the ground truth poses
                scans: List, the pointclouds
                timestamps: List, the timestamps
        """
        ## Load the ground truth poses
        poses, timestamps = self.load_poses(self.path_to_dataset + f'/{type_data}/{sequence}/{session}/poses/gt_poses.txt')
        ## Load the pointclouds
        scans = self.load_scans(self.path_to_dataset + f'/{type_data}/{sequence}/{session}/pcds')
        ## Load the depth maps
        # depth_maps = self.load_depth_map(self.path_to_dataset + f'/{sequence}/depth_map')
        ## Print the number of pointclouds, poses, and depth maps
        if self.verbose:
            print(f'Number of pointclouds: {len(scans)}')
            print(f'Number of poses: {len(poses)}')
            # print(f'Number of depth maps: {len(depth_maps)}')
        ## Return the pointclouds, poses, depth maps, and calibration data
        return poses, scans, timestamps

    ## Load the Apollo-SouthBay ground truth poses.
    def load_poses(self, pose_path:str) -> np.ndarray:
        """ Load ground truth poses .txt from file.
            Args:
                pose_path: (Complete) filename for the pose file
            Returns:
                A numpy array of size nx4x4 with n poses as 4x4 transformation matrices
        """
        ## Read and parse the poses. The poses are in the following format
        ## timestamp x, y, z, q_x, q_y, q_z, q_w
        poses = []
        timestamps = []
        try:
            if '.txt' in pose_path:
                with open(pose_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip().split()
                        timestamps.append(float(line[1]))
                        pose = np.eye(4)
                        pose[:3, 3] = np.array([float(x) for x in line[2:5]])
                        pose[:3, :3] = R.from_quat([float(x) for x in line[5:]]).as_matrix()
                        poses.append(pose)
            else:
                print('The pose file is not in .txt format.')
        except FileNotFoundError:
            if self.verbose:
                print('Ground truth poses are not available.')
                ## Print the wrong path to the pose file
                print(f'Path to pose file: {pose_path}')
        return np.array(poses), timestamps

    ## Load the Apollo-SouthBay depth maps.
    def load_depth_map(self, depth_map_path:str, split:str) -> List[np.ndarray]:
        """ Load depth maps from file. The format is .png files.
            Args:
                depth_map_path: The path to the depth map folder
            Returns:
                A list of depth maps
        """
        ## Get the list of depth map files
        depth_map_files = sorted(os.listdir(depth_map_path), key=lambda x: int(x.split('.')[0]))
        ## Create an empty list to store the depth maps
        depth_maps = []
        ## Loop through the depth map files
        for depth_map_file in tqdm(depth_map_files, desc='Loading depth maps', total=len(os.listdir(depth_map_path))):
            print(depth_map_file)
            ## Read the depth map file
            depth_map =  np.array(cv2.imread(depth_map_path + f'/{depth_map_file}', cv2.IMREAD_GRAYSCALE))
            ## Append the depth map to the list
            depth_maps.append(depth_map)
        ## Return the depth maps
        return depth_maps
    
    ## Load the Apollo-SouthBay point cloud scans.
    def load_scans(self, pointcloud_path:str) -> List:
        """ Load the pointclouds from the Apollo-SouthBay dataset. The format is .bin files.
            Args:
                pointcloud_path: The path to the pointcloud folder
            Returns:
                A nx4 numpy array with n pointclouds as 4D points
        """
        ## Get the list of pointcloud files
        pointcloud_files = sorted(os.listdir(pointcloud_path), key=lambda x: int(x.split('.')[0]))
        ## Create an empty list to store the pointclouds
        pointclouds = []
        ## Loop through the pointcloud files
        for pointcloud_file in tqdm(pointcloud_files, desc='Loading pointclouds', total=len(pointcloud_files)):
            ## Read the pointcloud file
            curr_scan = pypcd4.PointCloud.from_path(os.path.join(pointcloud_path, pointcloud_file)).numpy()
            ## Reshape the pointcloud to nx4
            # curr_scan = curr_scan.reshape(-1, 4)
            # curr_scan = np.asarray(pcd.points)
            ## Remove intensity
            homogeneous_scan = curr_scan[:, 0:3]
            ## Add homogeneous coordinates
            curr_scan = np.ones((homogeneous_scan.shape[0], homogeneous_scan.shape[1]+1))
            curr_scan[:, :-1] = homogeneous_scan
            ## Append the pointcloud to the list
            pointclouds.append(curr_scan)
        ## Return the pointclouds
        return pointclouds
    
    ## Load te Apollo-SouthBay point cloud scan.
    def load_scan(self, pointcloud_file:str) -> np.ndarray:
        """ Load a pointcloud from the Apollo-SouthBay dataset. The format is .bin files.
            Args:
                pointcloud_file: The path to the pointcloud file
            Returns:
                A nx4 numpy array with n points as 4D points
        """
        ## Read the pointcloud file (.pcd) using open3d
        curr_scan = pypcd4.PointCloud.from_path(pointcloud_file).numpy()
        # curr_scan = np.asarray(pcd.points)
        ## Remove intensity
        homogeneous_scan = curr_scan[:, 0:3]
        ## Add homogeneous coordinates
        curr_scan = np.ones((homogeneous_scan.shape[0], homogeneous_scan.shape[1]+1))
        curr_scan[:, :-1] = homogeneous_scan
        ## Return the pointcloud
        return curr_scan

    ## Visulize the Apollo-SouthBay ground truth poses.
    def visualize_poses(self, poses:np.ndarray, poses2:np.ndarray=None, save:bool=False) -> None:
        """ Visualize the ground truth poses using matplotlib.
            Args:
                poses: A numpy array of size nx4x4 with n poses as 4x4 transformation matrices
            Returns:
                None
        """
        ## Create a 2D plot
        fig, ax = plt.subplots()
        ## Plot the x and y coordinates
        ax.plot(poses[:, 0, 3], poses[:, 1, 3], 'o-', linewidth=2, label='gt poses')
        ## Set the x and y labels
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ## Set the title
        ax.set_title('Apollo-SouthBay ground truth poses')
        ## Add grid
        # plt.grid(alpha=0.5, linestyle='--', linewidth=1.0)
        if poses2 is not None:
            ax.plot(poses2[:, 0, 3], poses2[:, 1, 3], 'x-', linewidth=1, label='gt poses 2')
        ## Add limit
        # plt.xlim(-50, 500)
        # plt.ylim(-350, 350)
        ## Add legend
        plt.legend()
        ## Save the plot
        if save:
            plt.savefig('gt_poses.png')
        ## Show the plot
        plt.show()
        return

    ## Visualize a Apollo-SouthBay scan.
    def plot_scan(self, scan:np.ndarray, save:bool=False) -> None:
        """ Visualize a Apollo-SouthBay scan using matplotlib.
            Args:
                scan: A numpy array of size nx4 with n points as 3D points with homogeneous coordinates.
            Returns:
                None
        """
        ## Visualize in 2D
        fig, ax = plt.subplots()
        ## Plot the x and y coordinates
        ax.scatter(scan[:, 0], scan[:, 1], s=1, label='scan')
        ## Set the x and y labels
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ## Set the title
        ax.set_title('Apollo-SouthBay scan')
        ## Add grid
        plt.grid(alpha=0.5, linestyle='--', linewidth=1.0)
        ## Add limit
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        ## Add legend
        plt.legend()
        ## Save the plot
        if save:
            plt.savefig('scan.png')
        ## Show the plot
        plt.show()
        return
    

# ## Test run
# if __name__ == '__main__':
#     ## Define the path to the Apollo-SouthBay dataset
#     path_to_dataset = '/home/niksta/python_projects/datasets/Apollo-SouthBay'
#     ## Create an instance of the ApolloSouthBay_Handler class
#     apollo_southbay_handler = ApolloSouthBay_Handler(path_to_dataset, verbose=True)
#     ## Load the Apollo-SouthBay dataset
#     poses, scans, timestamps = apollo_southbay_handler.load_apollo_southbay('HighWay237', '2018-10-12', 'TestData')
#     ## Plot the ground truth poses
#     apollo_southbay_handler.visualize_poses(poses, save=True)
#     ## Plot the first scan
#     apollo_southbay_handler.plot_scan(scans[0], save=True)
#     ## Check the distance between every two consecutive poses
#     distances = []
#     for i in range(1, len(poses)):
#         distances.append(np.linalg.norm(poses[i-1, :3, 3] - poses[i, :3, 3]))
#     print('Mean distance between consecutive poses:', np.mean(distances))
#     print('Max distance between consecutive poses:', np.max(distances))
#     print('Min distance between consecutive poses:', np.min(distances))
    