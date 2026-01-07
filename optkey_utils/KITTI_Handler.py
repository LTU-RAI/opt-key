"""
    This is a class to handle the KITTI dataset.
"""

import numpy as np
from tqdm import tqdm
# import open3d as o3d
import os, cv2
from typing import List
import matplotlib.pyplot as plt


class KITTI_Handler:
    def __init__(self, path_to_dataset:str, 
                       verbose:bool=False):
        """
            This is a class to handle the KITTI dataset, containing any number of sequences.
            Args:
                path_to_dataset: str, path to the KITTI dataset
                sequences: List[str], list of sequences to be used
                verbose: bool, whether to print information or not
            Note:
                The KITTI dataset is expected to have the following structure:
                path_to_dataset
                KITTI
                ├── calib.txt
                |── sequences
                    ├── 00
                    │   ├── depth_map (contains the range images)
                    │   │   |── depth
                    │   │   │   ├── 000000.png
                    │   │   |   ├── ...
                    │   ├── poses (contains the txt file, 4x4 transformation matrices)
                    │   │   ├── 00.txt
                    │   ├── velodyne (contains the point clouds)
                    │   │   ├── 000000.bin
                    │   |   ├── ...
                    ├── 01
                    │   ├── ...
        """
        self.path_to_dataset = path_to_dataset
        self.verbose = verbose

    ## Load the KITTI dataset.
    def load_kitti(self, sequence:str):
        """ Read the KITTI dataset and return the pointclouds, poses, depth maps, and calibration data.
            Includes calibration data to properly project poses and pointclouds.
        """
        ## Load the calibration data
        T_cam_velo = self.load_calib(self.path_to_dataset + '/calib.txt')
        ## Load the ground truth poses
        poses = self.load_poses(self.path_to_dataset + f'/sequences/{sequence}/poses/{sequence}.txt')
        ## Transform the poses to the velodyne frame
        poses = self.project_poses_to_velo(poses, T_cam_velo)
        ## Load the pointclouds
        scans = self.load_scans(self.path_to_dataset + f'/sequences/{sequence}/velodyne/')
        ## Load the depth maps
        # depth_maps = self.load_depth_map(self.path_to_dataset + f'/{sequence}/depth_map')
        ## Print the number of pointclouds, poses, and depth maps
        if self.verbose:
            print(f'Number of pointclouds: {len(scans)}')
            print(f'Number of poses: {len(poses)}')
            # print(f'Number of depth maps: {len(depth_maps)}')
        ## Return the pointclouds, poses, depth maps, and calibration data
        return poses, scans, T_cam_velo

    ## Load the KITTI calibration data.
    def load_calib(self, calib_path:str) -> np.ndarray:
        """ Load calibrations (from camera to velodyne frame) from file.
            Args:
                calib_path: (Complete) filename for the calibration file
            Returns:
                A numpy array of size 4x4 with the transformation matrix from camera to velodyne frame
        """
        ## Read and parse calibrations
        T_cam_velo = []
        try:
            with open(calib_path, 'r') as f:
                lines = f.readlines()
                for line in (lines):
                    if 'Tr:' in line:
                        line = line.replace('Tr:', '')
                        T_cam_velo = np.fromstring(line, dtype=float, sep=' ')
                        T_cam_velo = T_cam_velo.reshape(3, 4)
                        T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))  
        except FileNotFoundError:
            if self.verbose:
                print('Calibration file not found.')
                ## Print the wrong path to the calibration file
                print(f'Path to calibration file: {calib_path}')
        ## Return the calibration data as a 4x4 transformation matrix
        return np.array(T_cam_velo)
    
    ## Load the KITTI timestamps.
    def load_timestamps(self, timestamp_path:str) -> List:
        """ Load timestamps from file.
            Args:
                timestamp_path: (Complete) filename for the timestamp file
            Returns:
                A list of timestamps
        """
        ## Read and parse the timestamps
        timestamps = []
        try:
            with open(timestamp_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    timestamps.append(float(line.strip()))
        except FileNotFoundError:
            if self.verbose:
                print('Timestamp file not found.')
                ## Print the wrong path to the timestamp file
                print(f'Path to timestamp file: {timestamp_path}')
        ## Return the timestamps
        return timestamps

    ## Load the KITTI ground truth poses.
    def load_poses(self, pose_path:str) -> np.ndarray:
        """ Load ground truth poses (T_w_cam0) from file.
            Args:
                pose_path: (Complete) filename for the pose file
            Returns:
                A numpy array of size nx4x4 with n poses as 4x4 transformation matrices
        """
        ## Read and parse the poses
        poses = []
        try:
            if '.txt' in pose_path:
                with open(pose_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                        T_w_cam0 = T_w_cam0.reshape(3, 4)
                        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                        poses.append(T_w_cam0)
            else:
                poses = np.load(pose_path)['arr_0']
        except FileNotFoundError:
            if self.verbose:
                print('Ground truth poses are not available.')
                ## Print the wrong path to the pose file
                print(f'Path to pose file: {pose_path}')
        return np.array(poses)

    ## Load the KITTI depth maps.
    def load_depth_map(self, depth_map_path:str) -> List[np.ndarray]:
        """ Load depth maps from file. The format is .png files.
            Args:
                depth_map_path: The path to the depth map folder
            Returns:
                A list of depth maps
        """
        ## Create an empty list to store the depth maps
        depth_maps = []
        ## Loop through the depth map files
        for depth_map_file in tqdm(np.sort(os.listdir(depth_map_path)), desc='Loading depth maps', total=len(os.listdir(depth_map_path))):
            ## Read the depth map file
            depth_map =  np.array(cv2.imread(depth_map_file, cv2.IMREAD_GRAYSCALE))
            ## Append the depth map to the list
            depth_maps.append(depth_map)
        ## Return the depth maps
        return depth_maps

    ## Load the KITTI point cloud scans.
    def load_scans(self, pointcloud_path:str) -> List:
        """ Load the pointclouds from the KITTI dataset. The format is .bin files.
            Args:
                pointcloud_path: The path to the pointcloud folder
            Returns:
                A nx4 numpy array with n pointclouds as 4D points
        """
        ## Get the list of pointcloud files
        pointcloud_files = sorted(os.listdir(pointcloud_path))
        ## Create an empty list to store the pointclouds
        pointclouds = []
        ## Loop through the pointcloud files
        for pointcloud_file in tqdm(pointcloud_files, desc='Loading pointclouds', total=len(pointcloud_files)):
            ## Read the pointcloud file
            curr_scan = np.fromfile(os.path.join(pointcloud_path, pointcloud_file), 
                                    dtype=np.float32).reshape(-1, 4)
            ## Remove intensity
            homogeneous_scan = curr_scan[:, 0:3]
            ## Add homogeneous coordinates
            curr_scan = np.ones((homogeneous_scan.shape[0], homogeneous_scan.shape[1]+1))
            curr_scan[:, :-1] = homogeneous_scan
            ## Append the pointcloud to the list
            pointclouds.append(curr_scan)
        ## Return the pointclouds
        return pointclouds

    ## Convert frame to pointcloud from Camera.
    def project_poses_to_velo(self, poses:np.ndarray, T_cam_velo:np.ndarray) -> np.ndarray:
        """ Project the ground truth poses from camera to velodyne frame.
            Args:
                poses: A numpy array of size nx4x4 with n poses as 4x4 transformation matrices
                calib: A numpy array of size 4x4 with the transformation matrix from camera to velodyne frame
            Returns:
                A numpy array of size nx4x4 with n poses as 4x4 transformation matrices in the velodyne frame
        """
        ## Transform the poses to the velodyne frame
        inv_pose_0= np.linalg.inv(poses[0])
        T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
        T_velo_cam = np.linalg.inv(T_cam_velo)
        velo_poses = []
        ## Loop through the poses and transform them to the velodyne frame
        for pose in poses:
            velo_poses.append(T_velo_cam.dot(inv_pose_0).dot(pose).dot(T_cam_velo))
        ## Return the poses in the velodyne frame
        return np.array(velo_poses)

    ## Visulize the KITTI ground truth poses.
    def visualize_poses(self, poses:np.ndarray, poses2:np.ndarray=None) -> None:
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
        ax.set_title('KITTI ground truth poses')
        ## Add grid
        # plt.grid(alpha=0.5, linestyle='--', linewidth=1.0)
        if poses2 is not None:
            ax.plot(poses2[:, 0, 3], poses2[:, 1, 3], 'x-', linewidth=1, label='gt poses 2')
        ## Add limit
        # plt.xlim(-50, 500)
        # plt.ylim(-350, 350)
        ## Add legend
        plt.legend()
        ## Show the plot
        plt.show()
        return

    ## Visualize a KITTI scan.
    def plot_scan(self, scan:np.ndarray) -> None:
        """ Visualize a KITTI scan using matplotlib.
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
        ax.set_title('KITTI scan')
        ## Add grid
        plt.grid(alpha=0.5, linestyle='--', linewidth=1.0)
        ## Add limit
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        ## Add legend
        plt.legend()
        ## Show the plot
        plt.show()
        return