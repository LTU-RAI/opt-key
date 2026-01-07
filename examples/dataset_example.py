import yaml, sys, os
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.Entropy import EntropyOnline
from scripts.Spaciousness import SpaciousnessOnline
from scripts.FixedDistance import FixedDistanceOnline
from scripts.RMIP import RMIP_Online
from optkey_utils.OT_Handler import OT_Handler
from optkey_utils.SC_Handler import SC_Handler
from optkey_utils.KITTI_Handler import KITTI_Handler
from optkey_utils.Keyframe import Keyframes, Keyframe
from optkey_utils.ApolloSouthBay_Handler import ApolloSouthBay_Handler
from optkey_utils.descriptors.OverlapTransformer.modules.overlap_transformer import *
from optkey_utils.descriptors.OverlapTransformer.tools.overlap_utils.overlap_utils import *
# from optkey_utils.PGO import PGO
from scipy.spatial.transform import Rotation as R

## Path to config file
path_to_config = '/home/niksta/python_projects/opt-key/config/dataset_config.yaml'
## Load the config file
with open(path_to_config) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
## Define the dataset
DATASET_CFG = config['dataset']
DATASET_NAME = DATASET_CFG['name']
## Path to the dataset
PATH = config['dataset_path']
## Dataset sequence (single string)
SEQ = DATASET_CFG['sequence']
## Additional dataset info (used for Apollo-SouthBay)
SESSION = DATASET_CFG.get('session')
DATA_TYPE = DATASET_CFG.get('data_type', 'MapData')
## Path to dataset
PATH_TO_DATASET = os.path.join(PATH, DATASET_NAME)
## Define descriptor method
DESCRIPTOR_METHOD = config['descriptor']
## PATH TO WEIGHTS
PATH_TO_WEIGHTS = config['weights_path']
## PATH TO SAVE RESULTS
PATH_TO_SAVE = config['save_path']
## Optimization parameters
ALPHA = config['alpha']
BETA = config['beta']
WINDOW_SIZE = config['window_size']
## Sampling method to evaluate
METHOD = config['method']
## Distance to classify as a true positive
DELTA = config['delta']
## Number of candidates to query
N_CANDIDATES = config['n_candidates']


class EvaluateOnline:
    def __init__(self, 
                 path_to_dataset:str, 
                 session:str,
                 sequence:str,
                 descriptor_method:str,
                 sampling_method:str,
                 data_type:str='MapData',
                 path_to_weights:str=None
                ) -> None:
        """
            This class is used to evaluate the keyframe selection methods online
                (place recognition) on any session on KITTI or Apollo-SouthBay dataset.
            Args:
                path_to_dataset: A string with the path to the dataset
                session: A string with the session to evaluate (Apollo-SouthBay only)
                sequence: String with the sequence to evaluate
                descriptor_method: A string with the descriptor method to use
                sampling_method: A string with the sampling method to evaluate
                    Options: 'entropy', 'spaciousness', 'fixed_1', 'fixed_3', 'fixed_5', 'optimized'
                data_type: A string with the data split (Apollo-SouthBay only) -> 'MapData', 'TrainData', 'TestData'
                path_to_weights: A string with the path to the descriptor weights
        """
        ## Initialize the parameters
        self.path_to_dataset = path_to_dataset
        self.session = session
        self.sequence = sequence
        self.descriptor_method = descriptor_method
        self.sampling_method = sampling_method
        ## Check if is part of the 'fixed' and get the fixed distance 
        if 'fixed' in sampling_method:
            self.fixed_distance = float(sampling_method.split('_')[1])
            self.sampling_method = 'fixed'
        self.data_type = data_type
        self.path_to_weights = path_to_weights
        ## Initialize the descriptor handler
        self.init_descriptor_handler()
        ## Initialize the keyframe selection methods
        self.init_keyframe_selection_methods()
        ## Initialize the dataset handler
        self.init_dataset_handler()
        ## Initialize lists for metrics
        self.similarities = []
        self.distances = []
        return None

    def init_descriptor_handler(self) -> None:
        ## Initialize Descritor handler
        if self.descriptor_method == 'ot':
            ## Initialize the OverlapTransformer handler
            self.desc_handler = OT_Handler(dist_threshold=3.0, 
                                           theta_threshold=0.8,
                                           verbose=True)
            ## Get the OT feature extracter
            self.feature_extracter = self.desc_handler.get_extracter(self.path_to_weights)
        elif self.descriptor_method == 'sc':
            ## Initialize the Scan Context handler
            self.desc_handler = SC_Handler(downcell_size=0.5, 
                                            lidar_height=2.0, 
                                            sector_res=60, 
                                            ring_res=20, 
                                            max_length=80, 
                                            verbose=True)
        else:
            raise ValueError('Descriptor method not supported')
        return None
    
    def init_keyframe_selection_methods(self) -> None:
        ## Initialize the keyframe selection method based on config
        if self.sampling_method == 'entropy':
            self.sampler = EntropyOnline(entropy_threshold=0.05, dist_threshold=5.0)
        elif self.sampling_method == 'spaciousness':
            self.sampler = SpaciousnessOnline(alpha=0.9, beta=0.1, delta_min=1.0,
                                             delta_mid=3.0, delta_max=5.0,
                                             theta_min=3.0, theta_mid=5.0,
                                             theta_max=10.0, queue_size=10)
        elif self.sampling_method == 'fixed':
            self.sampler = FixedDistanceOnline(dist_threshold=self.fixed_distance)
        elif self.sampling_method == 'optimized':
            self.sampler = RMIP_Online(n_neighbours=2,
                                        alpha=ALPHA,
                                        beta=BETA,
                                        window_size=WINDOW_SIZE)
        else:
            raise ValueError(f'Sampling method "{self.sampling_method}" not supported. '
                           'Options: entropy, spaciousness, fixed, optimized')
        return None
    
    def init_dataset_handler(self) -> None:
        ## Initialize the dataset handler
        if DATASET_NAME in ['KITTI', 'SemanticKitti']:
            ## Initialize the KITTI/SemanticKITTI handler
            self.dataset_handler = KITTI_Handler(path_to_dataset=self.path_to_dataset, verbose=True)
            return None
        elif DATASET_NAME == 'Apollo-SouthBay':
            ## Initialize the Apollo-SouthBay handler
            self.dataset_handler = ApolloSouthBay_Handler(path_to_dataset=self.path_to_dataset, verbose=True)
            return None
        else:
            raise ValueError('Dataset not supported')
        return None

    def query_map(self, curr_pose:np.ndarray, curr_scan:np.ndarray, curr_descriptor:np.ndarray, curr_index:int, keyframes:Keyframes, window_keyframes:Keyframes=None) -> None:
        ## Get the nearest neighbours of the current keyframe
        neighbours = keyframes._get_neighbours(curr_pose, n=N_CANDIDATES)
        if window_keyframes is not None:
            window_neighbours = window_keyframes._get_neighbours(curr_pose, n=2)
            neighbours.extend(window_neighbours)
        ## Get the descriptors of the neighbours
        descriptors = [keyframe.descriptor for keyframe in neighbours]
        ## Get the poses of the neighbours
        poses = [keyframe.pose for keyframe in neighbours]
        ## Get the best candidate
        similarity, index = self.desc_handler.find_best_candidate(curr_descriptor, descriptors)
        # print(f'Index: {index}')    
        # print(f'Similarity: {self.similarities[-1]}')
        ## Get the best candidate
        best_candidate = neighbours[index]
        if np.abs(curr_index - best_candidate.index) > 10:
            self.similarities.append(similarity)
            ## Get the distance to the best candidate
            self.distances.append(np.linalg.norm(curr_pose[:3, 3] - best_candidate.pose[:3, 3]))
            ## Print the index of the best candidate
            # print(f'Best candidate: {best_candidate.index}')
            ## Print the distance to the best candidate
            # print(f'Distance to best candidate: {self.distances[-1]}')
            ## Return the distance
        return 
    
    def get_metrics(self): 
        ## Get the Precision and Recall for every pair of similarities and distances
        precisions = []
        recalls = []
        f1 = []
        thetas = np.arange(0, 100, 1)
        for theta in thetas:
            tp = np.sum( (np.array(self.similarities) >= theta/100) & (np.array(self.distances) <= DELTA) )
            fp = np.sum( (np.array(self.similarities) > theta/100) & (np.array(self.distances) > DELTA) )
            fn = np.sum( (np.array(self.similarities) < theta/100) & (np.array(self.distances) <= DELTA) )
            tn = np.sum( (np.array(self.similarities) < theta/100) & (np.array(self.distances) > DELTA) )
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall)
            precisions.append(precision)
            recalls.append(recall)
            f1.append(f1_score)
        ## Plot the Precision-Recall curve
        plt.plot(recalls, precisions)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.savefig('precision_recall_curve.png')
        ## calculate the area under the curve
        auc = np.trapz(precisions, recalls)
        print(f'AUC: {np.round(-auc*100, 2)}%')
        print(f'F1 score: {np.round(np.max(f1)*100, 2)}%')
        return 

    def main(self):
        if DATASET_NAME in ['KITTI', 'SemanticKitti']:
            ## Prepare KITTI/SemanticKITTI data (stream scans on demand)
            seq_path = os.path.join(self.path_to_dataset, 'sequences', self.sequence)
            T_cam_velo = self.dataset_handler.load_calib(os.path.join(self.path_to_dataset, 'calib.txt'))
            poses_cam = self.dataset_handler.load_poses(os.path.join(seq_path, 'poses', f'{self.sequence}.txt'))
            poses = self.dataset_handler.project_poses_to_velo(poses_cam, T_cam_velo)
            scan_dir = os.path.join(seq_path, 'velodyne')
            scan_files = sorted(os.listdir(scan_dir))
        elif DATASET_NAME == 'Apollo-SouthBay':
            ## Prepare Apollo-SouthBay data (stream scans on demand)
            base_path = os.path.join(self.path_to_dataset, self.data_type, self.sequence, self.session)
            poses, _ = self.dataset_handler.load_poses(os.path.join(base_path, 'poses', 'gt_poses.txt'))
            scan_dir = os.path.join(base_path, 'pcds')
            scan_files = sorted(os.listdir(scan_dir), key=lambda x: int(x.split('.')[0]))
        else:
            raise ValueError(f'Dataset {DATASET_NAME} not supported')

        total_frames = min(len(poses), len(scan_files))

        ## Go through the data and sample the keyframes
        for i in tqdm(range(total_frames), total=total_frames, desc=f'Sampling keyframes ({self.sampling_method})'):
            ## Get the current pose
            pose = poses[i]
            ## Load the current scan on demand
            scan_path = os.path.join(scan_dir, scan_files[i])
            if DATASET_NAME in ['KITTI', 'SemanticKitti']:
                raw_scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
                homogeneous_scan = raw_scan[:, 0:3]
                scan = np.ones((homogeneous_scan.shape[0], homogeneous_scan.shape[1] + 1))
                scan[:, :-1] = homogeneous_scan
            else:  # Apollo-SouthBay
                scan = self.dataset_handler.load_scan(scan_path)

            ## Get descriptor
            if self.descriptor_method == 'ot':
                descriptor = self.desc_handler.get_descriptor(scan, self.feature_extracter)
                descriptor = descriptor.reshape(-1)
            elif self.descriptor_method == 'sc':
                descriptor = self.desc_handler.get_descriptor(scan)
            
            ## Sample keyframe with the selected method
            if self.sampling_method == 'optimized':
                timer = self.sampler.sample_rmip(pose=pose, scan=scan, descriptor=descriptor, index=i)
                if self.sampler.index > N_CANDIDATES:
                    self.query_map(pose, scan, descriptor, i, self.sampler.keyframes, self.sampler.window_keyframes)
            elif self.sampling_method in ['entropy', 'spaciousness']:
                self.sampler.sample(pose=pose, scan=scan, descriptor=descriptor, index=i)
                if self.sampler.index > N_CANDIDATES:
                    self.query_map(pose, scan, descriptor, i, self.sampler.keyframes)
            else:  # fixed distance methods
                self.sampler.sample(pose=pose, descriptor=descriptor, index=i)
                if self.sampler.index > N_CANDIDATES:
                    self.query_map(pose, scan, descriptor, i, self.sampler.keyframes)
        
        ## Print the results
        print(f'{self.sampling_method.capitalize()}: {self.sampler.keyframes}')
        self.get_metrics()
            

if __name__ == '__main__':
    ## Initialize the evaluation
    evaluation = EvaluateOnline(path_to_dataset=PATH_TO_DATASET,
                                session=SESSION,
                                sequence=SEQ,
                                descriptor_method=DESCRIPTOR_METHOD,
                                sampling_method=METHOD,
                                data_type=DATA_TYPE,
                                path_to_weights=PATH_TO_WEIGHTS)
    ## Run the evaluation
    evaluation.main()