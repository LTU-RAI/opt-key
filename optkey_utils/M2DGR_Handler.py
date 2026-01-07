"""
    This is a class to handle the M2DGR dataset.
"""

import numpy as np
from tqdm import tqdm
import open3d as o3d
import os, cv2
from typing import List
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


class M2DGR_Handler:
    def __init__(self, path_to_dataset:str, 
                       verbose:bool=False):
        """
            This is a class to handle the M2GDR dataset, containing any number of sequences.
            Args:
                path_to_dataset: str, path to the M2GDR dataset
                sequences: List[str], list of sequences to be used
                verbose: bool, whether to print information or not
            Note:
                The M2DGR dataset is expected to have the following structure:
                path_to_dataset
                ├─ M2GDR
                |   |── door
                |   |   |   |── door_01
                |   |   |   |   |── poses
                |   |   |   |   |   |── door_01.txt
                |   |   |   |   |── pcds
                |   |   |   |   |   |── 1661159289-317595648.pcd
                |   |   |   |   |   |── ...
                |   |   |   |   |── depth_map
                |   |   |   |   |   |── 1661159289-317595648.png
                |   |   |   |   |   |── ...
                |   |   |   |   |── ot_descriptors_door_01.npy
                |   |   |   |   |── ...
                |   |   |   |── door_02
                |   |   |   |   |── poses
                |   |   |   |   |   |── door_02.txt
                |   |   |   |   |── pcds
                |   |   |   |   |   |── ...
                |   |   |   |   |── depth_map
                |   |   |   |   |   |── ...
                |   |   |   |   |── ot_descriptors_door_02.npy
                |   |   |   |   |── ...
        """
        self.path_to_dataset = path_to_dataset
        self.verbose = verbose

    ## Load the M2DGR dataset.
    def load_m2dgr(self, sequence:str, session:str):
        """ Read the S3E dataset and return the pointclouds, poses and timestamps.
            Includes calibration data to properly project poses and pointclouds.
            Args:
                session: str, the session to be loaded
                robot: str, the namespace of the robot to be loaded
            Returns:
                poses: np.ndarray, the ground truth poses
                pose_timestamps: List, the timestamps
                scans: List, the pointclouds
                scan_timestamps: List, the timestamps
        """
        ## Load the ground truth poses
        poses, pose_timestamps = self.load_poses(self.path_to_dataset + f'/{sequence}/{sequence}_{session}/poses/{sequence}_{session}.txt')
        ## Load the pointclouds
        scans, scan_timestamps = self.load_scans(self.path_to_dataset + f'{sequence}/{sequence}_{session}/pcds')
        ## Load the depth maps
        # depth_maps = self.load_depth_map(self.path_to_dataset + f'/{sequence}/depth_map')
        ## Syncronize the data
        indices = self.sync_data(pose_timestamps, scan_timestamps)
        scans = [scans[i] for i in indices]
        scan_timestamps = [scan_timestamps[i] for i in indices]
        ## Print the number of pointclouds, poses, and depth maps
        if self.verbose:
            print(f'Number of pointclouds: {len(scans)}')
            print(f'Number of poses: {len(poses)}')
            # print(f'Number of depth maps: {len(depth_maps)}')
        ## Return the pointclouds, poses, depth maps, and calibration data
        return poses, pose_timestamps, scans, scan_timestamps

    ## Load the M2DGR ground truth poses.
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
                        timestamps.append(float(line[0]))
                        # print(line)
                        pose = np.eye(4)
                        pose[:3, 3] = np.array([float(x) for x in line[1:4]])
                        ## Replace the last value with 1.0
                        line[7] = '1.0'
                        pose[:3, :3] = R.from_quat([float(x) for x in line[4:]]).as_matrix()
                        poses.append(pose)
            else:
                print('The pose file is not in .txt format.')
        except FileNotFoundError:
            if self.verbose:
                print('Ground truth poses are not available.')
                ## Print the wrong path to the pose file
                print(f'Path to pose file: {pose_path}')
        return np.array(poses), timestamps
    
    ## Load the KISS-ICP generated poses.
    def load_kiss_icp_poses(self, pose_path:str) -> np.ndarray:
        """ Load KISS-ICP generated poses .txt from file.
            Args:
                pose_path: (Complete) filename for the pose file
            Returns:
                A numpy array of size nx4x4 with n poses as 4x4 transformation matrices
        """
        ## Read and parse the poses. The poses are in 
        poses = []
        try:
            if '.txt' in pose_path:
                with open(pose_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip().split()
                        pose = np.eye(4)
                        pose[:3, :] = np.array([float(x) for x in line]).reshape(3, 4)
                        poses.append(pose)        
            else:
                print('The pose file is not in .txt format.')
        except FileNotFoundError:
            if self.verbose:
                print('KISS-ICP generated poses are not available.')
                ## Print the wrong path to the pose file
                print(f'Path to pose file: {pose_path}')
        return np.array(poses)

    ## Load the M2DGR depth maps.
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
    
    ## Load the M2DGR point cloud scans.
    def load_scans(self, pointcloud_path:str) -> List:
        """ Load the pointclouds from the M2DGR dataset. The format is .bin files.
            Args:
                pointcloud_path: The path to the pointcloud folder
            Returns:
                A nx4 numpy array with n pointclouds as 4D points
        """
        ## Get the list of pointcloud files
        pointcloud_files = sorted(os.listdir(pointcloud_path), key=lambda x: float(x.split('-')[0] + '.' + x.split('-')[1].split('.')[0]))
        ## Create an empty list to store the pointclouds
        pointclouds = []
        timestamps = []
        ## Loop through the pointcloud files and the corresponding timestamps
        for pointcloud_file in tqdm(pointcloud_files, desc='Loading pointclouds', total=len(pointcloud_files)):
            # ## Read the pointcloud file
            # curr_scan = np.fromfile(os.path.join(pointcloud_path, pointcloud_file), 
            #                         dtype=np.float32).reshape(-1, 4)
            ## Read the pointcloud file (.pcd) using open3d
            pcd = o3d.io.read_point_cloud(os.path.join(pointcloud_path, pointcloud_file))
            curr_timestamp = float(pointcloud_file.split('-')[0] + '.' + pointcloud_file.split('-')[1].split('.')[0])
            curr_scan = np.asarray(pcd.points)
            ## Remove intensity
            homogeneous_scan = curr_scan[:, 0:3]
            ## Add homogeneous coordinates
            curr_scan = np.ones((homogeneous_scan.shape[0], homogeneous_scan.shape[1]+1))
            curr_scan[:, :-1] = homogeneous_scan
            ## Append the pointcloud to the list
            pointclouds.append(curr_scan)
            timestamps.append(curr_timestamp)
        ## Return the pointclouds
        return pointclouds, timestamps
    
    ## Load te M2DGR point cloud scan.
    def load_scan(sellf, pointcloud_file:str) -> np.ndarray:
        """ Load a pointcloud from the M2DGR dataset. The format is .bin files.
            Args:
                pointcloud_file: The path to the pointcloud file
            Returns:
                A nx4 numpy array with n points as 4D points
                A list of timestamps
        """
        ## Read the pointcloud file (.pcd) using open3d
        pcd = o3d.io.read_point_cloud(pointcloud_file)
        pointcloud_file = pointcloud_file.split('/')[-1]
        timestamp = float(pointcloud_file.split('-')[0] + '.' + pointcloud_file.split('-')[1].split('.')[0])
        curr_scan = np.asarray(pcd.points)
        ## Remove intensity
        homogeneous_scan = curr_scan[:, 0:3]
        ## Add homogeneous coordinates
        curr_scan = np.ones((homogeneous_scan.shape[0], homogeneous_scan.shape[1]+1))
        curr_scan[:, :-1] = homogeneous_scan
        ## Return the pointcloud
        return curr_scan, timestamp

    ## Syncronize the LiDAR scans and poses.
    def sync_data(self, pose_timestamps:np.ndarray, scan_timestamps:np.ndarray) -> np.ndarray:
        """ Synchronize two data streams based on their timestamps.
            Args:
                pose_timestamps: np.ndarray, timestamps of the poses
                scan_timestamps: np.ndarray, timestamps of the scans
            Returns:
                indices: np.ndarray, synchronized pose indices
        """
        ## For each scan timestamp find the nearest pose timestamp and keep the index
        idx = []
        for i in range(len(pose_timestamps)):
            idx.append(np.argmin(np.abs(np.asarray(scan_timestamps) - pose_timestamps[i])))
        ## Return the synchronized data
        return idx

    ## Visulize the M2DGR ground truth poses.
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
        ax.set_title('M2DGR ground truth poses')
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

    ## Visualize a M2DGR scan.
    def plot_scan(self, scan:np.ndarray, save:bool=False) -> None:
        """ Visualize a M2DGR scan using matplotlib.
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
        ax.set_title('M2DGR scan')
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

    ## Visualize a M2DGR pointcloud.
    def visualize_scan(self, scan:np.ndarray) -> None:
        """ Visualize a M2DGR pointcloud using open3d.
            Args:
                scan: A numpy array of size nx4 with n points as 3D points with homogeneous coordinates.
            Returns:
                None
        """
        ## Create a pointcloud
        pcd = o3d.geometry.PointCloud()
        ## Set the pointcloud points
        pcd.points = o3d.utility.Vector3dVector(scan[:, 0:3])
        ## Visualize the pointcloud
        o3d.visualization.draw_geometries([pcd])
        return

## Test run
# if __name__ == '__main__':
#     ## Define the path to the M2DGR dataset
#     path_to_dataset = '/home/niksta/Documents/datasets/M2DGR/'
#     ## Create an instance of the M2DGR_Handler class
#     m2dgr_handler = M2DGR_Handler(path_to_dataset, verbose=True)
#     ## Load the M2DGR dataset
#     poses, pose_timestamps, scans, scan_timestamps = m2dgr_handler.load_m2dgr('door', '02')
#     ## Sync poses and scans
#     indices = m2dgr_handler.sync_data(pose_timestamps, scan_timestamps)
#     print('Number of synchronized indices:', len(indices))
#     scans = [scans[i] for i in indices]
#     scan_timestamps = [scan_timestamps[i] for i in indices]
#     print('Number of synchronized poses:', len(poses))
#     print('Number of synchronized scans:', len(scans))
#     ## Plot the ground truth poses
#     m2dgr_handler.visualize_poses(poses, save=True)
#     ## Plot the first scan
#     m2dgr_handler.plot_scan(scans[0], save=True)
#     ## Check the distance between every two consecutive poses
#     distances = []
#     for i in range(1, len(poses)):
#         distances.append(np.linalg.norm(poses[i-1, :3, 3] - poses[i, :3, 3]))
#     print('Mean distance between consecutive poses:', np.mean(distances))
#     print('Max distance between consecutive poses:', np.max(distances))
#     print('Min distance between consecutive poses:', np.min(distances))
    