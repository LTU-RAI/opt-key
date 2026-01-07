"""
    This file contains the OverlapTransformerHandler class, which is used to handle the OverlapTransformer.
"""

import os, cv2, tqdm, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor
from optkey_utils.DescriptorHandler import DescriptorHandler
from optkey_utils.descriptors.OverlapTransformer.modules.overlap_transformer import *
from optkey_utils.descriptors.OverlapTransformer.tools.overlap_utils.overlap_utils import *
from matplotlib import pyplot as plt

class OT_Handler(DescriptorHandler):
    def __init__(self, channels:int=1,
                       use_transformer:bool=True, 
                       device:str=None,
                       dist_threshold:float=3.0,
                       theta_threshold:float=0.8,
                       num_threads:int=8,
                       verbose:bool=False):
        """
            This class is used to handle the OverlapTransformer descriptor.
            Args:
                channels: An integer with the number of channels of the input image
                use_transformer: A boolean to indicate if the OverlapTransformer model should be used
                device: A string with the device to use (e.g. 'cuda' or 'cpu')
                verbose: A boolean to indicate if the class should print information
        """
        self.channels = channels
        self.use_transformer = use_transformer
        self.device = device
        self.dist_threshold = dist_threshold
        self.theta_threshold = theta_threshold
        self.num_threads = num_threads
        self.verbose = verbose

    ## Get the Overlap Transformer descriptor model.
    def get_extracter(self, path_to_weights:str) -> featureExtracter:
        ## Build extracter
        feature_extracter = featureExtracter(channels=self.channels, use_transformer=self.use_transformer)
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.device)
        if self.verbose:
            ## Print the device
            print(f'Device: {self.device}')
        feature_extracter.to(self.device)
        ## Check if weights exist
        if not os.path.exists(path_to_weights):
            if self.verbose:
                ## Print that the weights file does not exist
                print('Weights file does not exist.')
                ## Print the wrong path to the weights file
                print(f'Path to weights file given: {path_to_weights}')
            return None
        ## Load the weights
        checkpoint = torch.load(path_to_weights, map_location=self.device)
        feature_extracter.load_state_dict(checkpoint['state_dict'])
        feature_extracter.eval()
        return feature_extracter 
    
    ## Get OverlapTransformer descriptors.
    def get_descriptors(self, depth_maps_path:str, featureExtracter:featureExtracter,
                    from_file:bool, descriptor_path:str, split:str=None) -> np.ndarray:
        """ Get OverlapTransformer descriptors from the dataset.
            Args:
                depth_maps_path: A string with the path to the depth maps
                featureExtracter: An OverlapTransformer model
                from_file: A boolean to indicate if the descriptors should be loaded from file
                descriptor_path: A string with the path to the descriptor file
            Returns:
                descriptors: A numpy array with the descriptors
        """
        ## Check if the descriptors should be loaded from file
        if from_file:
            try:
                ## Load the descriptors from file
                descriptors = np.load(descriptor_path, allow_pickle=True)
            except FileNotFoundError:
                if self.verbose:
                    ## Print that the descriptor file was not found
                    print('Descriptor file not found.')
                    ## Print the wrong path to the descriptor file
                    print(f'Path to descriptor file given: {descriptor_path}')
        else:
            ## Create the descriptor list
            descriptor_files = sorted(os.listdir(depth_maps_path), key=lambda x: int(x.split('.')[0]))
            ## Compute the descriptors
            descriptors = []
            for depth_map in tqdm.tqdm(np.sort(descriptor_files), desc=f'Computing descriptors', total=len(descriptor_files)):
                ## Load depth map
                range_image = np.array(cv2.imread(os.path.join(depth_maps_path, depth_map), cv2.IMREAD_GRAYSCALE))
                range_image_tensor = torch.from_numpy(range_image).type(torch.FloatTensor)
                range_image_tensor = torch.unsqueeze(range_image_tensor, dim=0)
                range_image_tensor = torch.unsqueeze(range_image_tensor, dim=0)
                range_image_tensor = range_image_tensor.to(self.device)
                descriptors.append(featureExtracter(range_image_tensor).cpu().detach().numpy())
                # print(f'Processed {depth_map}')
            descriptors = np.array(descriptors)
            ## Save the descriptors to file
            np.save(descriptor_path, descriptors, allow_pickle=True)
        ## Return the descriptors
        return descriptors

    def range_projection(self, current_vertex, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50, normalize=False):
        """ Project a pointcloud into a spherical projection, range image.
            Args:
                current_vertex: raw point clouds
            Returns: 
                proj_range: projected range image with depth, each pixel contains the corresponding depth
                proj_vertex: each pixel contains the corresponding point (x, y, z, 1)
                proj_intensity: each pixel contains the corresponding intensity
                proj_idx: each pixel contains the corresponding index of the point in the raw point cloud
        """
        # laser parameters
        fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
        fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians
        
        # get depth of all points
        depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
        current_vertex = current_vertex[(depth > 0) & (depth < max_range)]  # get rid of [0, 0, 0] points
        depth = depth[(depth > 0) & (depth < max_range)]
        
        # get scan components
        scan_x = current_vertex[:, 0]
        scan_y = current_vertex[:, 1]
        scan_z = current_vertex[:, 2]
        intensity = current_vertex[:, 3]
        
        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)
        
        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
        
        # scale to image size using angular resolution
        proj_x *= proj_W  # in [0.0, W]
        proj_y *= proj_H  # in [0.0, H]
        
        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        
        proj_y = np.floor(proj_y)
        proj_y = np.minimum(proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        
        # order in decreasing depth
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        intensity = intensity[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        scan_x = scan_x[order]
        scan_y = scan_y[order]
        scan_z = scan_z[order]
        
        indices = np.arange(depth.shape[0])
        indices = indices[order]
        
        proj_range = np.full((proj_H, proj_W), -1,
                            dtype=np.float32)  # [H,W] range (-1 is no data)
        proj_vertex = np.full((proj_H, proj_W, 4), -1,
                                dtype=np.float32)  # [H,W] index (-1 is no data)
        proj_idx = np.full((proj_H, proj_W), -1,
                            dtype=np.int32)  # [H,W] index (-1 is no data)
        proj_intensity = np.full((proj_H, proj_W), -1,
                            dtype=np.float32)  # [H,W] index (-1 is no data)
        
        proj_range[proj_y, proj_x] = depth
        proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
        proj_idx[proj_y, proj_x] = indices
        proj_intensity[proj_y, proj_x] = intensity

        # normalize the image
        if normalize:
            proj_range = proj_range / np.max(proj_range)
        
        return proj_range, proj_vertex, proj_intensity, proj_idx
    
    ## Get single OverlapTransformer descriptor.
    def get_descriptor(self, scan:np.ndarray, featureExtracter:featureExtracter) -> np.ndarray:
        """ Get a single OverlapTransformer descriptor.
            Args:
                scan: A np.ndarray with the point cloud
                featureExtracter: An OverlapTransformer model
            Returns:
                descriptor: A numpy array with the descriptor
        """
        ## Project the scan
        proj_range, _, _, _ = range_projection(scan)
        ## Convert the range image to tensor
        range_image_tensor = torch.from_numpy(proj_range).type(torch.FloatTensor)
        range_image_tensor = torch.unsqueeze(range_image_tensor, dim=0)
        range_image_tensor = torch.unsqueeze(range_image_tensor, dim=0)
        range_image_tensor = range_image_tensor.to(self.device)
        ## Compute the descriptor
        descriptor = featureExtracter(range_image_tensor).cpu().detach().numpy()
        return descriptor

    ## Query OverlapTransformer descriptors.
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
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            ## Compute similarities concurrently for each query descriptor
            futures = [executor.submit(self.find_best_candidate, query_desc, map_descr) for query_desc in query_descr]
            ## Retrieve computed similarities and indexes
            for o, future in enumerate(futures):
                similarities[o], indexes[o] = future.result()
        return similarities, indexes

    ## Find best candidate match.
    def find_best_candidate(self, query_descr:np.ndarray, map_descr:np.ndarray) -> Tuple[float, int]:
        return np.max(1 / (1 + np.linalg.norm(query_descr - map_descr, axis=1))), \
                np.argmax(1 / (1 + np.linalg.norm(query_descr - map_descr, axis=1)))
    
    ## Calculate similarities.
    def calculate_similarity(self, query_descr:np.ndarray, map_descr:np.ndarray) -> np.ndarray:
        return 1 / (1 + np.linalg.norm(query_descr - map_descr))