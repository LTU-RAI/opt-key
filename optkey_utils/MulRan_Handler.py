"""
    This is a class to handle the MulRan dataset.
"""

import numpy as np
from tqdm import tqdm
# import open3d as o3d
# import pypcd4
import os, cv2
from typing import List, Tuple
import matplotlib.pyplot as plt


class MulRan_Handler:
    def __init__(self, path_to_dataset:str, 
                       verbose:bool=False):
        """
            This is a class to handle the MulRan dataset
            Args:
                path_to_dataset: str, path to the MulRan dataset
                verbose: bool, whether to print information or not
            Note:
                The MulRan dataset is expected to have the following structure:
                /path_to_dataset/
                /MulRan/
                |── DCC
                    ├── DCC01
                    │   ├── depth_map (contains the range images)
                    │   │   |── depth
                    │   │   │   ├── 000000.png
                    │   │   |   ├── ...
                    │   ├── global_pose.csv (contains the csv file, timestamp and 3x4 transformation matrices)
                    │   ├── Ouster (contains the point clouds)
                    │   │   ├── 1567496149757432877.bin
                    │   |   ├── ...
                    ├── DCC02
                    │   ├── ...
                |── KAIST
                    ├── K01
                    │   ├── ...
                |── SejongCity
                    ├── SC01
                    │   ├── ...
        """
        self.path_to_dataset = path_to_dataset
        self.verbose = verbose

    ## Load the MulRan dataset.
    def load_mulran(self, sequence:str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Read the MulRan dataset and return the pointclouds, poses and depth maps."""
        ## Load the ground truth poses
        poses, pose_timestamps = self.load_poses(self.path_to_dataset + f'/{sequence}/global_pose.csv')
        ## Load the pointclouds
        scans, scan_timestamps = self.load_scans(self.path_to_dataset + f'/{sequence}/Ouster/')
        ## Synchronize the data
        idx = self.sync_data(pose_timestamps, scan_timestamps)
        ## Load the depth maps
        # depth_maps = self.load_depth_map(self.path_to_dataset + f'/{sequence}/depth_map')
        ## Print the number of pointclouds, poses, and depth maps
        if self.verbose:
            print(f'Number of pointclouds: {len(scans)}')
            print(f'Number of poses: {len(poses[idx])}')
            # print(f'Number of depth maps: {len(depth_maps)}')
        ## Return the pointclouds, poses, depth maps, and calibration data
        return poses[idx], scans, pose_timestamps[idx], scan_timestamps
    
    ##
    def sync_data(self, pose_timestamps:np.ndarray, scan_timestamps:np.ndarray) -> np.ndarray:
        """ Synchronize two data streams based on their timestamps.
            Args:
                timestamps1: np.ndarray, timestamps of the first data stream
                timestamps2: np.ndarray, timestamps of the second data stream
                data1: np.ndarray, data of the first data stream
                data2: np.ndarray, data of the second data stream
            Returns:
                indices: np.ndarray, synchronized pose indices
        """
        ## For each scan timestamp find the nearest pose timestamp and keep the index
        idx = []
        for i in range(len(scan_timestamps)):
            idx.append(np.argmin(np.abs(pose_timestamps - scan_timestamps[i])))
        ## Return the synchronized data
        return idx

    ## Load the MulRan ground truth poses.
    def load_poses(self, pose_path:str) -> Tuple[np.ndarray, np.ndarray]:
        """ Load ground truth poses from file.
            Args:
                pose_path: (Complete) filename for the pose file, CSV foramted
                            Each line contains 13 numbers. The first is the timestamp
                            and the next 12 are the elements of the 3x4 matrix
            Returns:
                A numpy array of size nx4x4 with n poses as 4x4 transformation matrices
        """
        ## Read and parse the poses
        poses = []
        timestamps = []
        try:
            if '.csv' in pose_path:
                with open(pose_path, 'r') as f:
                    for line in f:  
                        ## Parse the line
                        line = line.strip().split(',')
                        ## Get timestamp
                        timestamp = float(line[0])*1e-9
                        ## Get the pose
                        pose = np.array(line[1:], dtype=np.float32).reshape(3, 4)
                        ## Create a 4x4 transformation matrix
                        pose = np.vstack((pose, np.array([0, 0, 0, 1], dtype=np.float32)))
                        ## Append the pose to the list
                        poses.append(pose)
                        timestamps.append(timestamp)
            else:
                raise ValueError('Invalid pose file format')
        except:
            raise ValueError('Error reading pose file')
        ## Return the poses
        return np.array(poses), np.array(timestamps)

    ## Load the MulRan depth maps.
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

    ## Load the MulRan point cloud scans.
    def load_scans(self, pointcloud_path:str) -> Tuple[np.ndarray, np.ndarray]:
        """ Load the pointclouds from the MulRan dataset. The format is .bin files.
            Args:
                pointcloud_path: The path to the pointcloud folder
            Returns:
                A nx4 numpy array with n pointclouds as 4D points
        """
        ## Get the list of pointcloud files
        pointcloud_files = sorted(os.listdir(pointcloud_path))
        ## Create an empty list to store the pointclouds
        pointclouds = []
        timestamps = [int(file.split('.')[0])*1e-9 for file in pointcloud_files]
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
        return pointclouds, timestamps
    
    ## Load single LiDAR scan.
    def load_scan(self, pointcloud_path:str) -> np.ndarray:
        """ Load a single LiDAR scan from a .bin file.
            Args:
                pointcloud_path: The path to the pointcloud file
            Returns:
                A nx4 numpy array with n points as 4D points
        """
        ## Read the pointcloud file
        scan = np.fromfile(pointcloud_path, dtype=np.float32).reshape(-1, 4)
        ## Remove intensity
        homogeneous_scan = scan[:, 0:3]
        ## Add homogeneous coordinates
        scan = np.ones((homogeneous_scan.shape[0], homogeneous_scan.shape[1]+1))
        scan[:, :-1] = homogeneous_scan
        ## Return the scan
        return scan
        
    ## Visulize the MulRan ground truth poses.
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
        ax.set_title('MulRan ground truth poses')
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
        plt.savefig('gt_poses.png')
        return

    ## Visualize a MulRan scan.
    def plot_scan(self, scan:np.ndarray) -> None:
        """ Visualize a MulRan scan using matplotlib.
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
        ax.set_title('MulRan scan')
        ## Add grid
        plt.grid(alpha=0.5, linestyle='--', linewidth=1.0)
        ## Add limit
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        ## Add legend
        plt.legend()
        ## Show the plot
        plt.savefig('scan.png')
        return

    # ## Visualize a MulRan pointcloud.
    # def visualize_scan(self, scan:np.ndarray) -> None:
    #     """ Visualize a MulRan pointcloud using open3d.
    #         Args:
    #             scan: A numpy array of size nx4 with n points as 3D points with homogeneous coordinates.
    #         Returns:
    #             None
    #     """
    #     ## Create a pointcloud
    #     pcd = o3d.geometry.PointCloud()
    #     ## Set the pointcloud points
    #     pcd.points = o3d.utility.Vector3dVector(scan[:, 0:3])
    #     ## Visualize the pointcloud
    #     o3d.visualization.draw_geometries([pcd])
    #     return


## Test loading the MulRan dataset
if __name__ == '__main__':
    ## Path to the MulRan dataset
    path_to_dataset = '/home/niksta/python_projects/datasets/MulRan/DCC/'
    ## Create a MulRan_Handler object
    mulran_handler = MulRan_Handler(path_to_dataset, verbose=True)
    ## Load the MulRan dataset
    poses, scans, pose_timestamps, scan_timestamps = mulran_handler.load_mulran('DCC03')
    ## Visualize the poses
    mulran_handler.visualize_poses(poses)
    ## Visualize the first scan
    mulran_handler.plot_scan(scans[0])
    ## Plot the pose to pose distances
    pose_distances = np.linalg.norm(poses[1:, 0:3, 3] - poses[:-1, 0:3, 3], axis=1)
    plt.figure()
    plt.plot(pose_distances)
    plt.xlabel('Pose index')
    plt.ylabel('Distance [m]')
    plt.title('Pose to pose distances')
    plt.grid(alpha=0.5, linestyle='--', linewidth=1.0)
    plt.savefig('pose_distances.png')
    ## Print min and max timestamp difference between poses and scans
    print(f'Min timestamp difference between poses and scans: {np.min(np.abs(pose_timestamps - scan_timestamps))}')
    print(f'Max timestamp difference between poses and scans: {np.max(np.abs(pose_timestamps - scan_timestamps))}')    