"""
    This file contains the class for handling the ScanContext descriptor.
    TODO:
    - Load KITTI data with KITTI_Handler and test genSCs
    - Add Distance_SC.py to the methods
    - Make genSCs to get_descriptors
    - Test the distance calculation
    - Add the SC_Handler to the main pipeline
"""

import os, cv2, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np 
from typing import Tuple, List
from optkey_utils.DescriptorHandler import DescriptorHandler
from optkey_utils.KITTI_Handler import KITTI_Handler
import open3d as o3d
from tqdm import tqdm
import multiprocessing

class SC_Handler(DescriptorHandler):
    def __init__(self, downcell_size:float=0.5,
                       lidar_height:float=2.0,
                       sector_res:int=60,
                       ring_res:int=20,
                       max_length:int=80,
                       num_threads:int=8,
                       verbose:bool=False):
        """
            This class is used to handle the ScanContext descriptor.
            Args:
                downcell_size: A float with the downcell size
                lidar_height: A float with the lidar height
                sector_res: An array with the sector resolution
                ring_res: An array with the ring resolution
                max_length: An integer with the max length of the LiDAR range
                num_threads: An integer with the number of threads
                verbose: A boolean to indicate if the class should print information
        """
        self.downcell_size = downcell_size
        self.lidar_height = lidar_height
        self.sector_res = sector_res
        self.ring_res = ring_res
        self.max_length = max_length
        self.num_threads = num_threads
        self.verbose = verbose

    ## Get the thetas.
    def xy2theta(self, x:float, y:float) -> float:
        if (x >= 0 and y >= 0): 
            theta = 180/np.pi * np.arctan(y/x);
        if (x < 0 and y >= 0): 
            theta = 180 - ((180/np.pi) * np.arctan(y/(-x)));
        if (x < 0 and y < 0): 
            theta = 180 + ((180/np.pi) * np.arctan(y/x));
        if ( x >= 0 and y < 0):
            theta = 360 - ((180/np.pi) * np.arctan((-y)/x));
        return theta
    
    ## Get the ring and sector from a point.
    def pt2rs(self, point:np.ndarray, gap_ring:float, gap_sector:float) -> Tuple[int, int]:
        x = point[0]
        y = point[1]
        z = point[2]
        
        if(x == 0.0):
            x = 0.001
        if(y == 0.0):
            y = 0.001
     
        theta = self.xy2theta(x, y)
        faraway = np.sqrt(x*x + y*y)
        
        idx_ring = np.divmod(faraway, gap_ring)[0]       
        idx_sector = np.divmod(theta, gap_sector)[0]

        if(idx_ring >= self.ring_res):
            idx_ring = self.ring_res-1 # python starts with 0 and ends with N-1
        
        return int(idx_ring), int(idx_sector)
    
    ## Get the ScanContext descriptor from a point cloud.
    def ptcloud2sc(self, ptcloud:np.ndarray, max_length:int) -> np.ndarray:
        
        num_points = ptcloud.shape[0]
       
        gap_ring = max_length/self.ring_res
        gap_sector = 360/self.sector_res
        
        enough_large = 1000
        sc_storage = np.zeros([enough_large, self.ring_res, self.sector_res])
        sc_counter = np.zeros([self.ring_res, self.sector_res])
        
        for pt_idx in range(num_points):

            point = ptcloud[pt_idx, :]
            point_height = point[2] + self.lidar_height
            
            idx_ring, idx_sector = self.pt2rs(point, gap_ring, gap_sector)
            
            if sc_counter[idx_ring, idx_sector] >= enough_large:
                continue
            sc_storage[int(sc_counter[idx_ring, idx_sector]), idx_ring, idx_sector] = point_height
            sc_counter[idx_ring, idx_sector] = sc_counter[idx_ring, idx_sector] + 1

        sc = np.amax(sc_storage, axis=0)
            
        return sc
    
    ## Load the velodyne scan
    def genSC(self, ptcloud_xyz:np.ndarray) -> np.ndarray:
        ptcloud_xyz = ptcloud_xyz[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ptcloud_xyz)
        downpcd = pcd.voxel_down_sample(voxel_size=self.downcell_size)
        ptcloud_xyz_downed = np.asarray(downpcd.points)        
        return np.array(self.ptcloud2sc(ptcloud_xyz_downed, self.max_length))
    
    ## Get ScanContext descriptors.
    def get_descriptors(self, scans:List[np.ndarray], 
                        from_file:bool, descriptor_path:str) -> np.ndarray:
        """ Get ScanContext descriptors from a list of scans.
            Args:
                scans: A list of numpy arrays with the scans
                from_file: A boolean to indicate if the descriptors should be loaded from file
                descriptor_path: A string with the path to the descriptor file
            Returns:
                descriptors: A numpy array with the descriptors
        """
        ## Check if the descriptors should be loaded from file
        if from_file:
            try:
                ## Load the descriptors from file
                descriptors = np.load(descriptor_path)
            except FileNotFoundError:
                ## Print that the descriptor file was not found
                print('Descriptor file not found.')
                ## Print the wrong path to the descriptor file
                print(f'Path to descriptor file given: {descriptor_path}')
                return None
        else:
            ## Compute the descriptors
            descriptors = []
            for scan in tqdm(scans, desc='Generating ScanContext descriptors', total=len(scans)):
                descriptors.append(self.genSC(np.asarray(scan)))
            descriptors = np.array(descriptors)
            ## Save the descriptors to file
            np.save(descriptor_path, descriptors, allow_pickle=True)
        ## Return the descriptors
        return descriptors
    
    ## Get single ScanContext descriptor.
    def get_descriptor(self, scan:np.ndarray) -> np.ndarray:
        """ Get the ScanContext descriptor from a single scan.
            Args:
                scan: A numpy array with the scan
            Returns:
                A numpy array with the descriptor
        """
        return np.asarray(self.genSC(scan))

    ## Query ScanContext descriptors.
    def query_map(self, map_descr:np.ndarray, query_descr:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Query the map with the query keyframes.
            Args:
                map_descriptors: A numpy array with the map descriptors
                query_descriptors: A numpy array with the query descriptors
            Returns:
                A numpy array with the similarities
        """
        ## Create an empty array to store similarities and indexes
        similarities = np.zeros(len(query_descr))
        indexes = np.zeros(len(query_descr), dtype=int)
        ## Create a ThreadPoolExecutor for multithreading
        # with ThreadPoolExecutor(max_workers=1) as executor:
        #     ## Compute similarities concurrently for each query descriptor
        #     futures = [executor.submit(self.find_best_candidate, query_desc, map_descr) 
        #                for query_desc in query_descr]
        #     ## Retrieve computed similarities and indexes
        #     for o, future in  enumerate(futures):
        #         similarities[o], indexes[o] = future.result()
        # for i, query_desc in tqdm(enumerate(query_descr), desc='Querying map', total=len(query_descr)):
        #     similarities[i], indexes[i] = self.find_best_candidate(query_desc, map_descr)
        # return similarities, indexes
    
        """ Query the map with the query keyframes.
                Args:
                    map_descriptors: A numpy array with the map descriptors
                    query_descriptors: A numpy array with the query descriptors
                Returns:
                    A numpy array with the similarities
        """
        # Create a multiprocessing pool
        with multiprocessing.Pool(processes=8) as pool:
            # Map the query descriptors to the find_best_candidate function using multiprocessing
            results = pool.starmap(self.find_best_candidate_wrapper, zip(query_descr, [map_descr]*len(query_descr)))
        # Unpack the results
        similarities, indexes = zip(*results)
        return np.array(similarities), np.array(indexes)
    
    def find_best_candidate_wrapper(self, query_desc, map_descr):
        # Wrapper function to call find_best_candidate with two arguments
        return self.find_best_candidate(query_desc, map_descr)
    
    ## Find best candidate match.
    def find_best_candidate(self, query_descr:np.ndarray, map_descr:np.ndarray) -> Tuple[float, int]:
        ## Get the similarity between the query descr and all map descr and return the best match
        similarities = []
        for map_desc in map_descr:
            similarities.append(self.calculate_similarity(query_descr, map_desc))
        return np.max(similarities), np.argmax(similarities)

    ## Calculate the similarity between two descriptors.    
    def calculate_similarity(self, query_descr:np.ndarray, map_descr:np.ndarray) -> np.ndarray:
        sc1 = query_descr
        sc2 = map_descr
        num_sectors = self.sector_res
        # repeate to move 1 columns
        sim_for_each_cols = np.zeros(num_sectors)
        for i in range(num_sectors):
            # Shift
            one_step = 1 # const
            sc1 = np.roll(sc1, one_step, axis=1) #  columne shift
            # compare
            norm_sc1 = np.linalg.norm(sc1, axis=0)
            norm_sc2 = np.linalg.norm(sc2, axis=0)
            nonzero_norm_indices = np.where((norm_sc1 != 0) & (norm_sc2 != 0))[0]
            if len(nonzero_norm_indices) == 0:
                continue
            cos_similarities = np.sum(sc1[:, nonzero_norm_indices] * sc2[:, nonzero_norm_indices], axis=0) / (norm_sc1[nonzero_norm_indices] * norm_sc2[nonzero_norm_indices])
            sim_for_each_cols[i] = np.mean(cos_similarities)
        sim = np.max(sim_for_each_cols)
        ## Return similarity
        return sim