"""
    This is a generic class to handle descriptors. It is used to handle the OverlapTransformer descriptor etc.
"""
import numpy as np
from typing import Tuple

class DescriptorHandler:
    def __init__(self, dist_threshold:float, 
                       theta_threshold:float, 
                       verbose:bool=False):
        self.dist_threshold = dist_threshold
        self.theta_threshold = theta_threshold
        self.verbose = verbose

    def query_map(self, map_desc:np.ndarray, query_descr:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def find_best_candidate(self, query_desc:np.ndarray, map_desc:np.ndarray) -> Tuple[float, int]:
        pass

    def get_extracter(self):
        pass
    
    def get_descriptors(self):
        pass

    def calculate_similarity(self):
        pass