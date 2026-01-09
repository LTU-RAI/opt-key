import argparse
import yaml, sys, os
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.Entropy import EntropyOnline
from scripts.Spaciousness import SpaciousnessOnline
from scripts.FixedDistance import FixedDistanceOnline
from scripts.RMIP import RMIP_Online
from optkey_utils.OT_Handler import OT_Handler
from optkey_utils.SC_Handler import SC_Handler
from optkey_utils.KITTI_Handler import KITTI_Handler
from optkey_utils.MulRan_Handler import MulRan_Handler
from optkey_utils.Keyframe import Keyframes
from optkey_utils.ApolloSouthBay_Handler import ApolloSouthBay_Handler
from optkey_utils.descriptors.OverlapTransformer.modules.overlap_transformer import *
from optkey_utils.descriptors.OverlapTransformer.tools.overlap_utils.overlap_utils import *


## Default path to config file (can be overridden via CLI)
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../config/dataset_config.yaml')

## Globals populated by load_config()
config: Dict[str, Any] = {}
DATASET_CFG: Dict[str, Any] = {}
DATASET_NAME: str = ''
PATH: str = ''
SEQ: str = ''
SESSION: str = ''
DATA_TYPE: str = 'MapData'
PATH_TO_DATASET: str = ''
DESCRIPTOR_METHOD: str = ''
PATH_TO_WEIGHTS: str = ''
PATH_TO_SAVE: str = ''
ALPHA: float = 1.0
BETA: float = 1.0
WINDOW_SIZE: int = 10
METHODS: Any = []
DELTA: float = 3.0
N_CANDIDATES: int = 10
path_to_config: str = DEFAULT_CONFIG_PATH


def load_config(cfg_path: str) -> None:
    """Load YAML config and populate module-level settings."""
    global config, DATASET_CFG, DATASET_NAME, PATH, SEQ, SESSION, DATA_TYPE
    global PATH_TO_DATASET, DESCRIPTOR_METHOD, PATH_TO_WEIGHTS, PATH_TO_SAVE
    global ALPHA, BETA, WINDOW_SIZE, METHODS, DELTA, N_CANDIDATES, path_to_config

    path_to_config = cfg_path
    with open(path_to_config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    DATASET_CFG = config['dataset']
    DATASET_NAME = DATASET_CFG['name']
    PATH = config['dataset_path']
    SEQ = DATASET_CFG['sequence']
    SESSION = DATASET_CFG.get('session')
    DATA_TYPE = DATASET_CFG.get('data_type', 'MapData')
    PATH_TO_DATASET = os.path.join(PATH, DATASET_NAME)
    DESCRIPTOR_METHOD = config['descriptor']
    PATH_TO_WEIGHTS = config['weights_path']
    PATH_TO_SAVE = config['save_path']
    ALPHA = config['alpha']
    BETA = config['beta']
    WINDOW_SIZE = config['window_size']
    METHODS = config['method']
    DELTA = config['delta']
    N_CANDIDATES = config['n_candidates']


# Load default config on import; may be overridden via CLI in __main__
load_config(DEFAULT_CONFIG_PATH)


def compute_pr_metrics_from_arrays(similarities: List[float], distances: List[float], method_label: str) -> Dict[str, Any]:
    """Compute Precision-Recall metrics for provided similarity/distance lists."""
    precisions = []
    recalls = []
    f1_scores = []
    thetas = np.arange(0, 100, 1)
    sim_array = np.array(similarities)
    dist_array = np.array(distances)

    for theta in thetas:
        threshold = theta / 100
        tp = np.sum((sim_array >= threshold) & (dist_array <= DELTA))
        fp = np.sum((sim_array > threshold) & (dist_array > DELTA))
        fn = np.sum((sim_array < threshold) & (dist_array <= DELTA))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    auc = float(-np.trapz(precisions, recalls))
    best_f1 = float(np.max(f1_scores)) if f1_scores else 0.0
    return {
        'method': method_label,
        'recalls': np.array(recalls),
        'precisions': np.array(precisions),
        'auc': auc,
        'f1': best_f1
    }

def init_descriptor_handler() -> Tuple[Any, Any]:
    """Initialize descriptor handler once for all methods."""
    if DESCRIPTOR_METHOD == 'ot':
        desc_handler = OT_Handler(dist_threshold=3.0, theta_threshold=0.8, verbose=False)
        feature_extracter = desc_handler.get_extracter(PATH_TO_WEIGHTS)
        return desc_handler, feature_extracter
    if DESCRIPTOR_METHOD == 'sc':
        desc_handler = SC_Handler(downcell_size=0.5, lidar_height=2.0, sector_res=60, ring_res=20, max_length=80, verbose=False)
        return desc_handler, None
    raise ValueError('Descriptor method not supported')

def init_dataset_handler() -> Any:
    """Initialize dataset handler once for all methods."""
    if DATASET_NAME in ['KITTI', 'SemanticKitti']:
        return KITTI_Handler(path_to_dataset=PATH_TO_DATASET, verbose=True)
    if DATASET_NAME == 'Apollo-SouthBay':
        return ApolloSouthBay_Handler(path_to_dataset=PATH_TO_DATASET, verbose=True)
    if DATASET_NAME == 'MulRan':
        return MulRan_Handler(path_to_dataset=PATH_TO_DATASET, verbose=True)
    raise ValueError('Dataset not supported')

def load_dataset_files(dataset_handler) -> Tuple[np.ndarray, str, List[str]]:
    """Load poses and scan file paths from the dataset."""
    if DATASET_NAME in ['KITTI', 'SemanticKitti']:
        seq_path = os.path.join(PATH_TO_DATASET, 'sequences', SEQ)
        T_cam_velo = dataset_handler.load_calib(os.path.join(PATH_TO_DATASET, 'calib.txt'))
        poses_cam = dataset_handler.load_poses(os.path.join(seq_path, 'poses', f'{SEQ}.txt'))
        poses = dataset_handler.project_poses_to_velo(poses_cam, T_cam_velo)
        scan_dir = os.path.join(seq_path, 'velodyne')
        scan_files = sorted(os.listdir(scan_dir))
        return poses, scan_dir, scan_files
    if DATASET_NAME == 'Apollo-SouthBay':
        base_path = os.path.join(PATH_TO_DATASET, DATA_TYPE, SEQ, SESSION)
        poses, _ = dataset_handler.load_poses(os.path.join(base_path, 'poses', 'gt_poses.txt'))
        scan_dir = os.path.join(base_path, 'pcds')
        scan_files = sorted(os.listdir(scan_dir), key=lambda x: int(x.split('.')[0]))
        return poses, scan_dir, scan_files
    if DATASET_NAME == 'MulRan':
        seq_path = os.path.join(PATH_TO_DATASET, SEQ, SEQ + SESSION)
        poses, pose_ts = dataset_handler.load_poses(os.path.join(seq_path, 'global_pose.csv'))
        scan_dir = os.path.join(seq_path, 'Ouster')
        scan_files = sorted(os.listdir(scan_dir))
        # derive scan timestamps from filenames to time-sync to nearest pose
        scan_ts = np.array([int(f.split('.')[0]) * 1e-9 for f in scan_files], dtype=float)
        pose_idx = dataset_handler.sync_data(pose_ts, scan_ts)
        poses_aligned = poses[pose_idx]
        return poses_aligned, scan_dir, scan_files
    raise ValueError(f'Dataset {DATASET_NAME} not supported')

def build_sampler_context(method_label: str) -> Dict[str, Any]:
    """Create sampler instance and bookkeeping containers for a method label."""
    sampling_method = method_label
    sampler = None
    if method_label.startswith('fixed_'):
        sampling_method = 'fixed'
        fixed_distance = float(method_label.split('_')[1])
        sampler = FixedDistanceOnline(dist_threshold=fixed_distance)
    elif method_label == 'entropy':
        sampler = EntropyOnline(entropy_threshold=0.05, dist_threshold=5.0)
    elif method_label == 'spaciousness':
        sampler = SpaciousnessOnline(alpha=0.9, beta=0.1, delta_min=1.0,
                                     delta_mid=3.0, delta_max=5.0,
                                     theta_min=3.0, theta_mid=5.0,
                                     theta_max=10.0, queue_size=10)
    elif method_label == 'optimized':
        sampler = RMIP_Online(alpha=ALPHA, beta=BETA, window_size=WINDOW_SIZE, verbose=False)
    else:
        raise ValueError(f'Sampling method "{method_label}" not supported. Options: entropy, spaciousness, fixed_X, optimized')

    return {
        'label': method_label,
        'sampling_method': sampling_method,
        'sampler': sampler,
        'similarities': [],
        'distances': []
    }

def query_map(desc_handler, curr_pose: np.ndarray, curr_descriptor: np.ndarray, curr_index: int, ctx: Dict[str, Any], window_keyframes: Keyframes = None) -> None:
    sampler = ctx['sampler']
    neighbours = sampler.keyframes._get_neighbours(curr_pose, n=N_CANDIDATES)
    if window_keyframes is not None:
        neighbours.extend(window_keyframes._get_neighbours(curr_pose, n=2))
    descriptors = [k.descriptor for k in neighbours]
    similarity, index = desc_handler.find_best_candidate(curr_descriptor, descriptors)
    best_candidate = neighbours[index]
    if np.abs(curr_index - best_candidate.index) > 10:
        ctx['similarities'].append(similarity)
        ctx['distances'].append(np.linalg.norm(curr_pose[:3, 3] - best_candidate.pose[:3, 3]))

def run_sampling_methods(methods: List[str], live: bool = False) -> List[Dict[str, Any]]:
    desc_handler, feature_extracter = init_descriptor_handler()
    dataset_handler = init_dataset_handler()
    poses, scan_dir, scan_files = load_dataset_files(dataset_handler)
    os.makedirs(PATH_TO_SAVE, exist_ok=True)

    contexts = [build_sampler_context(m) for m in methods]
    total_frames = min(len(poses), len(scan_files))

    live_ctx = None
    fig = ax_map = ax_bar = handles = None
    seen_optimization_events = 0
    optimized_window_snapshot = np.empty((0, 2))
    optimized_selected_xy = np.empty((0, 2))
    last_kept_xy = np.empty((0, 2))
    trajectory_xy = np.empty((0, 2))
    if live:
        live_ctx = next((c for c in contexts if c['sampling_method'] == 'optimized'), None)
        if live_ctx is not None:
            fig, (ax_map, ax_bar), handles = create_live_plot()
        else:
            live = False

    if live and live_ctx is not None and handles is not None:
        controls = {'paused': False, 'step': 0}

        def on_key(event):
            key = (event.key or '').lower()
            if key == 'p':
                controls['paused'] = not controls['paused']
            elif key == 'right':
                controls['paused'] = True
                controls['step'] += 1
            elif key == 'left':
                controls['paused'] = True
                controls['step'] = max(controls['step'] - 1, 0)  # backward not supported; keep non-negative

        fig.canvas.mpl_connect('key_press_event', on_key)

        idx = 0
        pbar = tqdm(total=total_frames, desc='Sampling keyframes')
        while idx < total_frames:
            if controls['paused'] and controls['step'] == 0:
                plt.pause(0.05)
                continue

            pose = poses[idx]
            scan_path = os.path.join(scan_dir, scan_files[idx])
            if DATASET_NAME in ['KITTI', 'SemanticKitti']:
                raw_scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
                homogeneous_scan = raw_scan[:, 0:3]
                scan = np.ones((homogeneous_scan.shape[0], homogeneous_scan.shape[1] + 1))
                scan[:, :-1] = homogeneous_scan
            else:
                scan = dataset_handler.load_scan(scan_path)

            if DESCRIPTOR_METHOD == 'ot':
                descriptor = desc_handler.get_descriptor(scan, feature_extracter).reshape(-1)
            elif DESCRIPTOR_METHOD == 'sc':
                descriptor = desc_handler.get_descriptor(scan)
            else:
                raise ValueError('Descriptor method not supported')

            for ctx in contexts:
                sampler = ctx['sampler']
                if ctx['sampling_method'] == 'optimized':
                    time = sampler.sample_rmip(pose=pose, scan=scan, descriptor=descriptor, index=idx)
                    if sampler.index > N_CANDIDATES:
                        query_map(desc_handler, pose, descriptor, idx, ctx, sampler.window_keyframes)
                elif ctx['sampling_method'] in ['entropy', 'spaciousness']:
                    sampler.sample(pose=pose, scan=scan, descriptor=descriptor, index=idx)
                    if sampler.index > N_CANDIDATES:
                        query_map(desc_handler, pose, descriptor, idx, ctx)
                else:  # fixed distance
                    sampler.sample(pose=pose, descriptor=descriptor, index=idx)
                    if sampler.index > N_CANDIDATES:
                        query_map(desc_handler, pose, descriptor, idx, ctx)

            # Transform scan to global coordinates using current pose
            scan_h = np.ones((scan.shape[0], 4))
            scan_h[:, :3] = scan[:, :3]
            scan_global = (pose @ scan_h.T).T
            scan_xy = scan_global[:, :2]
            if len(scan_xy) > 8000:
                scan_xy = scan_xy[::4]

            window_xy = np.array([kf.pose[:2, 3] for kf in live_ctx['sampler'].window_keyframes]) if len(live_ctx['sampler'].window_keyframes) else np.empty((0, 2))

            sampler = live_ctx['sampler']

            # Snapshot optimized window when an optimization event occurs
            if hasattr(sampler, 'optimization_events') and sampler.optimization_events > seen_optimization_events:
                seen_optimization_events = sampler.optimization_events
                if getattr(sampler, 'last_optimized_window_poses', None):
                    optimized_window_snapshot = np.array([
                        pose[:2, 3] for pose in sampler.last_optimized_window_poses
                    ]) if sampler.last_optimized_window_poses else np.empty((0, 2))
                if getattr(sampler, 'last_optimized_selected_pose', None) is not None:
                    pose_sel = sampler.last_optimized_selected_pose
                    optimized_selected_xy = np.array([[pose_sel[0, 3], pose_sel[1, 3]]])

            # Update last kept keyframe (latest in sampler.keyframes)
            if len(sampler.keyframes.keyframes) > 0:
                last_kf_pose = sampler.keyframes.keyframes[-1].pose
                last_kept_xy = np.array([[last_kf_pose[0, 3], last_kf_pose[1, 3]]])
                trajectory_xy = np.array([[kf.pose[0, 3], kf.pose[1, 3]] for kf in sampler.keyframes.keyframes])
            else:
                last_kept_xy = np.empty((0, 2))
                trajectory_xy = np.empty((0, 2))

            kept_pct = (len(sampler.keyframes.keyframes) / float(idx + 1)) * 100.0 if idx + 1 > 0 else 0.0

            update_live_plot(
                ax_map,
                ax_bar,
                handles,
                scan_xy,
                window_xy,
                optimized_window_snapshot,
                optimized_selected_xy,
                last_kept_xy,
                trajectory_xy,
                kept_pct,
                current_pose_xy=(pose[0, 3], pose[1, 3]),
            )

            idx += 1
            pbar.update(1)
            if controls['step'] > 0:
                controls['step'] -= 1
                controls['paused'] = True

        pbar.close()
    else:
        for i in tqdm(range(total_frames), total=total_frames, desc='Sampling keyframes'):
            pose = poses[i]
            scan_path = os.path.join(scan_dir, scan_files[i])
            if DATASET_NAME in ['KITTI', 'SemanticKitti']:
                raw_scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
                homogeneous_scan = raw_scan[:, 0:3]
                scan = np.ones((homogeneous_scan.shape[0], homogeneous_scan.shape[1] + 1))
                scan[:, :-1] = homogeneous_scan
            else:
                scan = dataset_handler.load_scan(scan_path)

            if DESCRIPTOR_METHOD == 'ot':
                descriptor = desc_handler.get_descriptor(scan, feature_extracter).reshape(-1)
            elif DESCRIPTOR_METHOD == 'sc':
                descriptor = desc_handler.get_descriptor(scan)
            else:
                raise ValueError('Descriptor method not supported')

            for ctx in contexts:
                sampler = ctx['sampler']
                if ctx['sampling_method'] == 'optimized':
                    time = sampler.sample_rmip(pose=pose, scan=scan, descriptor=descriptor, index=i)
                    if sampler.index > N_CANDIDATES:
                        query_map(desc_handler, pose, descriptor, i, ctx, sampler.window_keyframes)
                elif ctx['sampling_method'] in ['entropy', 'spaciousness']:
                    sampler.sample(pose=pose, scan=scan, descriptor=descriptor, index=i)
                    if sampler.index > N_CANDIDATES:
                        query_map(desc_handler, pose, descriptor, i, ctx)
                else:  # fixed distance
                    sampler.sample(pose=pose, descriptor=descriptor, index=i)
                    if sampler.index > N_CANDIDATES:
                        query_map(desc_handler, pose, descriptor, i, ctx)

    metrics_list = []
    for ctx in contexts:
        metrics = compute_pr_metrics_from_arrays(ctx['similarities'], ctx['distances'], ctx['label'])
        metrics['num_keyframes'] = len(ctx['sampler'].keyframes)
        print(f"{ctx['label']} | AUC: {metrics['auc']*100:.2f}% | F1-MAX: {metrics['f1']*100:.2f}% | Keyframes: {metrics['num_keyframes']}/{total_frames}")
        metrics_list.append(metrics)
    return metrics_list

def plot_precision_recall_curves(metrics_list: List[Dict[str, Any]], save_dir: str, filename: str = 'precision_recall_curve.png') -> str:
    """Plot PR curves and bar comparison of keyframes kept per method."""
    if not metrics_list:
        raise ValueError('No metrics provided for plotting PR curves.')

    os.makedirs(save_dir, exist_ok=True)
    fig, (ax_pr, ax_bar) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [3, 1]})

    # Precision-Recall curves
    for metrics in metrics_list:
        label = f"{metrics['method']} (AUC: {metrics['auc']*100:.2f}%, F1: {metrics['f1']*100:.2f}%)"
        ax_pr.plot(metrics['recalls'], metrics['precisions'], label=label)
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall curves')
    ax_pr.grid(True)
    ax_pr.legend()

    # Keyframes comparison bar chart
    methods = [m['method'] for m in metrics_list]
    keyframes_counts = [m.get('num_keyframes', 0) for m in metrics_list]
    bars = ax_bar.bar(methods, keyframes_counts, color='C0', alpha=0.8)
    ax_bar.set_ylabel('Keyframes kept')
    ax_bar.set_title('Keyframes per method')
    ax_bar.grid(True, axis='y', linestyle='--', alpha=0.3)
    for bar, count in zip(bars, keyframes_counts):
        ax_bar.text(bar.get_x() + bar.get_width() / 2.0, count + max(1, 0.01 * max(keyframes_counts, default=1)), f'{count}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    output_path = os.path.join(save_dir, filename)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path

def methods_to_list(methods_cfg: Any) -> List[str]:
    """Normalize the method config entry to a list of strings."""
    if isinstance(methods_cfg, (list, tuple)):
        return list(methods_cfg)
    if isinstance(methods_cfg, str):
        return [methods_cfg]
    raise ValueError('`method` in config must be a string or list of strings.')


def create_live_plot():
    """Initialize live matplotlib plot for top-view visualization with kept-percentage subplot."""
    plt.ion()
    fig, (ax_map, ax_bar) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [3, 1]})

    # Map view
    ax_map.set_title('Top-View Live Visualization (MSA Sampling)')
    ax_map.set_xlabel('X')
    ax_map.set_ylabel('Y')
    ax_map.set_aspect('equal', adjustable='datalim')
    scan_scatter = ax_map.scatter([], [], s=2, c='gray', alpha=0.35, label='Scan')
    optimized_window_scatter = ax_map.scatter([], [], s=40, c='k', marker='o', linewidths=1.0, edgecolors='k', label='Prev. Window')
    traj_scatter = ax_map.scatter([], [], s=40, c='C0', marker='o', linewidths=1.0, edgecolors='k', label='Optimized KFs')
    window_scatter = ax_map.scatter([], [], s=40, c='C1', marker='o', linewidths=1.0, edgecolors='k', label='Cur. Window')
    selected_scatter = ax_map.scatter([], [], s=40, c='red', marker='o', linewidths=1.0, edgecolors='k', label='Selected KF')
    ax_map.legend(loc='upper right')

    # Kept percentage bar
    ax_bar.set_title('Keyframes Kept')
    ax_bar.set_xlim(-0.5, 0.5)
    ax_bar.set_ylim(0, 100)
    kept_bar = ax_bar.bar([0], [0], width=0.6, color='C0')[0]
    kept_text = ax_bar.text(0, 5, '0%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax_bar.set_xticks([])
    ax_bar.set_ylabel('Percent')
    ax_bar.grid(True, axis='y', linestyle='--', alpha=0.3)

    return fig, (ax_map, ax_bar), {
        'scan': scan_scatter,
        'window': window_scatter,
        'optimized_window': optimized_window_scatter,
        'selected': selected_scatter,
        'traj': traj_scatter,
        'kept_bar': kept_bar,
        'kept_text': kept_text,
    }


def update_live_plot(
    ax_map,
    ax_bar,
    handles,
    scan_xy: np.ndarray,
    window_xy: np.ndarray,
    optimized_window_xy: np.ndarray,
    optimized_selected_xy: np.ndarray,
    last_kept_xy: np.ndarray,
    trajectory_xy: np.ndarray,
    kept_percent: float,
    current_pose_xy: Tuple[float, float],
) -> None:
    """Update live plot elements and keep view centered on current pose."""
    handles['scan'].set_offsets(scan_xy if scan_xy.size else np.empty((0, 2)))
    handles['window'].set_offsets(window_xy if window_xy.size else np.empty((0, 2)))
    handles['optimized_window'].set_offsets(optimized_window_xy if optimized_window_xy.size else np.empty((0, 2)))
    handles['selected'].set_offsets(optimized_selected_xy if optimized_selected_xy.size else np.empty((0, 2)))
    handles['traj'].set_offsets(trajectory_xy if trajectory_xy.size else np.empty((0, 2)))

    ax_map.relim()
    ax_map.autoscale_view()
    view_span = 80.0
    cx, cy = current_pose_xy
    ax_map.set_xlim(cx - view_span, cx + view_span)
    ax_map.set_ylim(cy - view_span, cy + view_span)

    handles['kept_bar'].set_height(kept_percent)
    handles['kept_text'].set_text(f'{kept_percent:.1f}%')
    handles['kept_text'].set_y(max(kept_percent, 5))
    ax_bar.set_ylim(0, 100)

    plt.pause(0.001)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate keyframe sampling methods.')
    parser.add_argument('-c', '--config', default=path_to_config, help='Path to dataset_config.yaml')
    parser.add_argument('--live', action='store_true', help='Enable live top-view visualization (optimized only)')
    args = parser.parse_args()

    load_config(args.config)

    method_list = methods_to_list(METHODS)
    metrics_list = run_sampling_methods(method_list, live=args.live)
    pr_curve_path = plot_precision_recall_curves(metrics_list, PATH_TO_SAVE, filename=f'pr_curve_{DATASET_NAME}_{SEQ}_{SESSION}.png')
    print(f'Precision-Recall plot saved to: {pr_curve_path}')