"""
    This file contains the InLiER_Handler class, which is used to handle the InLiER descriptor.
    The compact MINT histogram is the keyframe descriptor (used by the samplers, e.g. MSA), while
    place recognition runs the full InLiER pipeline (MINT -> BEAM -> verify) on the cached tokens.
    Requires the InLiER package: pip install -e /path/to/InLiER
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Dict
from optkey_utils.DescriptorHandler import DescriptorHandler
from inlier import InLiER, InLiER_Matcher, InLiER_Tokens, InLiER_Keypoints
from inlier.config import load, resolve
from inlier.eval.encode import encode_frame
from inlier.eval.pipeline import build_matcher, minimal_keypoints, shortlist_one, beam_one, verify_one

class InLiER_Handler(DescriptorHandler):
    def __init__(self, config_path:str=None,
                       verbose:bool=False):
        """
            This class is used to handle the InLiER descriptor.
            Args:
                config_path: A string with the path to an InLiER config yaml (None uses the InLiER defaults)
                verbose: A boolean to indicate if the class should print information
        """
        self.resolved = resolve(load(config_path))
        self.voxel_size = self.resolved.voxel_size
        self.cfg = self.resolved.inlier
        self.eps = self.resolved.shortlist.eps
        self.encoder = InLiER(self.cfg)
        self.verbose = verbose
        self.encodings: Dict[int, Tuple[InLiER_Tokens, InLiER_Keypoints]] = {}

    ## Get the InLiER encoder.
    def get_extracter(self) -> InLiER:
        return self.encoder

    ## Encode a scan into InLiER tokens and keypoints.
    def encode(self, scan:np.ndarray) -> Tuple[InLiER_Tokens, InLiER_Keypoints]:
        tokens, kp_aligned, kp_sensor, T_ground = encode_frame(self.encoder, scan[:, :3], self.voxel_size)
        return tokens, minimal_keypoints(kp_aligned, kp_sensor, T_ground)

    ## Get the compact MINT descriptor from tokens.
    def tokens_to_descriptor(self, tokens:InLiER_Tokens) -> np.ndarray:
        c = self.cfg
        hist = InLiER_Matcher._tokens_to_hist(tokens.token_id, c.N_a, c.N_h * c.N_r * c.N_s, c.N_h, c.N_r * c.N_s)
        return hist.sum(axis=0)

    ## Get InLiER descriptors.
    def get_descriptors(self, scans:List[np.ndarray],
                        from_file:bool, descriptor_path:str) -> np.ndarray:
        """ Get compact MINT descriptors from a list of scans.
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
            for scan in tqdm(scans, desc='Generating InLiER descriptors', total=len(scans)):
                descriptors.append(self.get_descriptor(np.asarray(scan)))
            descriptors = np.array(descriptors)
            ## Save the descriptors to file
            np.save(descriptor_path, descriptors, allow_pickle=True)
        ## Return the descriptors
        return descriptors

    ## Get single InLiER descriptor.
    def get_descriptor(self, scan:np.ndarray, index:int=None) -> np.ndarray:
        """ Get the compact MINT descriptor from a single scan.
            Args:
                scan: A numpy array with the scan (Nx4 homogeneous coordinates)
                index: An integer with the frame index, if given the tokens and keypoints are cached for match_candidates
            Returns:
                A numpy array with the descriptor (N_r*N_s,)
        """
        tokens, keypoints = self.encode(scan)
        if index is not None:
            self.encodings[index] = (tokens, keypoints)
        return self.tokens_to_descriptor(tokens)

    ## Query InLiER descriptors.
    def query_map(self, map_descr:np.ndarray, query_descr:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Query the map with the query keyframes.
            Args:
                map_descriptors: A numpy array with the map descriptors
                query_descriptors: A numpy array with the query descriptors
            Returns:
                A numpy array with the similarities and a numpy array with the indexes
        """
        similarities, indexes = zip(*[self.find_best_candidate(query_desc, map_descr) for query_desc in query_descr])
        return np.array(similarities), np.array(indexes)

    ## Find best candidate match.
    def find_best_candidate(self, query_descr:np.ndarray, map_descr:np.ndarray) -> Tuple[float, int]:
        ## MINT L1 intersection between the query and all map descriptors
        scores = InLiER_Matcher._mint_l1_intersection(np.asarray(query_descr)[None, :], np.asarray(map_descr), self.eps)
        return float(np.max(scores)), int(np.argmax(scores))

    ## Find best candidate match with the full InLiER pipeline.
    def match_candidates(self, query_index:int, candidate_indices:List[int]) -> Tuple[float, int]:
        """ Run MINT -> BEAM -> verify over the candidates, using the cached encodings.
            Args:
                query_index: An integer with the frame index of the query
                candidate_indices: A list with the frame indices of the candidates
            Returns:
                The verification score (keypoint inlier ratio, 0.0 if failed) and the position of the best candidate
        """
        q_tokens, q_keypoints = self.encodings[query_index]
        db_tokens = [self.encodings[i][0] for i in candidate_indices]
        db_keypoints = [self.encodings[i][1] for i in candidate_indices]
        matcher = build_matcher(self.resolved, db_tokens, verbose=False)
        ids, _ = shortlist_one(matcher, q_tokens, topk=len(db_tokens))
        ids, _, shifts = beam_one(matcher, q_tokens, ids)
        scores, verified, _ = verify_one(matcher, q_tokens, q_keypoints, db_tokens, db_keypoints.__getitem__,
                                         ids, dict(zip(ids, shifts)), self.resolved.verify, self.resolved.verify_topv)
        ## Highest verification score, ties keep the retrieval rank order
        best = max(verified, key=scores.get)
        return float(scores[best]), int(best)

    ## Calculate the similarity between two descriptors.
    def calculate_similarity(self, query_descr:np.ndarray, map_descr:np.ndarray) -> float:
        return self.find_best_candidate(query_descr, np.asarray(map_descr)[None, :])[0]
