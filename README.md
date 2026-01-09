<div align="center">
  
# A Minimal Subset Approach (MSA)
## for Informed Keyframe Sampling in Large-Scale SLAM

[![arXiv](https://img.shields.io/badge/Arxiv-2501.01791-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2501.01791)
[![DOI:10.1109/LRA.2025.3623045](https://img.shields.io/badge/IEEE-10.1109/LRA.2025.3636035-00629B.svg)](https://doi.org/10.1109/LRA.2025.3636035)
  <a href="https://www.youtube.com/watch?v=70CcaRCL-7g"><img src="https://badges.aleen42.com/src/youtube.svg" alt="YouTube" /></a>

![Python](https://img.shields.io/badge/Python-3670A0?logo=python&logoColor=ffdd54)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)

[**Nikolaos Stathoulopoulos**](https://github.com/nstathou) · [**Christoforos Kanellakis**](https://github.com/christoforoskanel) · [**George Nikolakopoulos**](https://github.com/geonikolak)

</div>

<p align=center> <img src="/figures/ral25_pipeline.png" width="75%" height="75%"/> </p>

## 💡 Introduction

**Abstract:** Typical LiDAR SLAM architectures feature a front-end for odometry estimation and a back-end for refining and optimizing the trajectory and map, commonly through loop closures. However, loop closure detection in large-scale missions presents significant computational challenges due to the need to identify, verify, and process numerous candidate pairs for pose graph optimization. Keyframe sampling bridges the front-end and back-end by selecting frames for storing and processing during global optimization. This article proposes an online keyframe sampling approach that constructs the pose graph using the most impactful keyframes for loop closure. We introduce the Minimal Subset Approach (MSA), which optimizes two key objectives: redundancy minimization and information preservation, implemented within a sliding window framework. By operating in the feature space rather than 3-D space, MSA efficiently reduces redundant keyframes while retaining essential information. Evaluations on diverse public datasets show that the proposed approach outperforms naive methods in reducing false positive rates in place recognition, while delivering superior ATE and RPE in metric localization, without the need for manual parameter tuning. Additionally, MSA demonstrates efficiency and scalability by reducing memory usage and computational overhead during loop closure detection and pose graph optimization.

---

## 📋 Table of Contents

- [💡 Introduction](#-introduction)
- [🚀 Setup](#-setup) — [Prerequisites](#prerequisites) • [Installation](#installation) • [Environment Setup](#environment-setup) • [Dataset Setup](#dataset-setup) • [Descriptor Setup](#descriptor-setup)
- [🕹️ Run the Examples](#️-run-the-examples) — [Configuration](#configuration) • [Evaluations](#running-evaluations) • [Visualization](#live-visualization)
- [🔜 Coming Soon](#-coming-soon)
- [📝 Citation](#-citation)

---

## 🚀 Setup

Tested with Python 3.8 and 3.12, should work with other versions since there are minimal packages needed.

### Prerequisites

- Python 3.8 or higher
- At least one supported dataset: KITTI/SemanticKITTI, Apollo-SouthBay, MulRan, or NewerCollege (or adapt examples to your dataset)
- At least one descriptor extraction method: OverlapTransformer or ScanContext (or implement a custom descriptor)

For more information on how to adapt to your dataset or descriptor check [Custom Dataset](#custom-dataset) and [Custom Descriptor](#custom-descriptor)

### Installation

Clone the repository to your desired working directory:

```bash
git clone https://github.com/LTU-RAI/opt-key.git
cd opt-key
```

### Environment Setup

Using a virtual environment is strongly recommended. You can use any virtual environment manager you prefer.

#### Using Conda (Recommended)

```bash
conda create -n opt-key
conda activate opt-key
pip install -r requirements.txt
```

#### Using pip with venv

```bash
python -m venv opt-key-env
source opt-key-env/bin/activate  
pip install -r requirements.txt
```

### Dataset Setup

Supported datasets:

- **[KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)** / **[SemanticKITTI](https://semantic-kitti.org/dataset.html#download)** — Using [KITTI_Handler.py](optkey_utils/KITTI_Handler.py)
- **[Apollo-SouthBay](https://developer.apollo.auto/southbay.html)** — Using [ApolloSouthBay_Handler.py](optkey_utils/ApolloSouthBay_Handler.py)
- **[MulRan](https://sites.google.com/view/mulran-pr/dataset)** — Using [MulRan_Handler.py](optkey_utils/MulRan_Handler.py)
- **[NewerCollege](https://ori-drs.github.io/newer-college-dataset/)** — Using [NewerCollege_Handler.py](optkey_utils/NewerCollege_Handler.py)

Download your dataset and verify its structure matches the handler's expectations. Check the corresponding handler file (e.g., `KITTI_Handler.py`) for the required directory layout.

### Custom Dataset

To add support for a new dataset:

1. Create a handler class (e.g., `MyDataset_Handler.py`) in `/optkey_utils/`
2. Implement the following methods:
   - `load_poses(pose_path: str) -> Tuple[np.ndarray, np.ndarray]` — Returns poses (Nx4x4) and timestamps
   - `load_scan(scan_path: str) -> np.ndarray` — Returns point cloud (Nx4) with homogeneous coordinates
   - `sync_data(pose_timestamps, scan_timestamps) -> np.ndarray` — Synchronizes poses to scans
3. Add the handler to `init_dataset_handler()` in your example script
4. Update the config file with your dataset name, sequence, and paths

See [KITTI_Handler.py](optkey_utils/KITTI_Handler.py) or [MulRan_Handler.py](optkey_utils/MulRan_Handler.py) for reference implementations.

### Descriptor Setup

Supported descriptors:

- **[OverlapTransformer](https://github.com/haomo-ai/OverlapTransformer)** — Using [OT_Handler.py](optkey_utils/OT_Handler.py) (requires pretrained weights)
- **[ScanContext](https://github.com/gisbi-kim/scancontext)** — Using [SC_Handler.py](optkey_utils/SC_Handler.py) (geometric, no training required)

Install descriptor dependencies in `/optkey_utils/descriptors/` following the original repository instructions. Update your config file with the descriptor choice and weights path (if applicable).

### Custom Descriptor

To implement a custom descriptor:

1. Create a handler class extending [DescriptorHandler.py](optkey_utils/DescriptorHandler.py) (e.g., `MyDescriptor_Handler.py`)
2. Implement the following methods:
   - `get_descriptor(scan: np.ndarray) -> np.ndarray` — Extracts descriptor from a single scan
   - `find_best_candidate(query_descriptor, map_descriptors) -> Tuple[float, int]` — Computes similarity and returns (score, best_match_index)
3. Add the handler to `init_descriptor_handler()` in your example script
4. Update the config file to specify your descriptor name

See [OT_Handler.py](optkey_utils/OT_Handler.py) or [SC_Handler.py](optkey_utils/SC_Handler.py) for reference implementations.

---

## 🕹️ Run the Examples

An example script is provided:

- **examples/pr_example.py** — Evaluates keyframe sampling methods with Precision-Recall metrics

### Configuration

Update the config file in `/config/` with your dataset and descriptor paths. Template configs for each dataset are provided (e.g., `pr_config.yaml`, `kitti_config.yaml`, etc.).

### Running Evaluations

```bash
python3 examples/pr_example.py
```

Or specify a custom config:

```bash
python3 examples/pr_example.py --config /path/to/config.yaml
```

### Sampling Methods

Configure which methods to evaluate in your config file:

```yaml
method: ['fixed_0', 'fixed_1', 'fixed_2', 'spaciousness', 'entropy', 'optimized']
```

Options:
- `fixed_X` — Fixed distance threshold (X in meters)
- `entropy` — Entropy-based sampling
- `spaciousness` — Spaciousness-based sampling
- `optimized` — Proposed MSA approach

### Results

After completion, results are printed to console and saved as a plot in your configured output directory (default: `/results/`). Example output from KITTI sequence 00:

<p align=center> <img src="/results/pr_curve_SemanticKitti_00_ot.png" width="75%" height="75%"/> </p>

### Live Visualization

Enable real-time visualization of the sampling process:

```bash
python3 examples/pr_example.py --live
```
<p align=center> <img src="/figures/kitti_00_msa_example.gif" width="75%" height="75%"/> </p>

Key controls:
- **Space bar** — Pause/resume playback
- **Right Arrow** — Step forward one frame

**Note:** Running without visualization is significantly faster for batch evaluation.

---

## 🔜 Coming Soon

TODO:
- Add example with iSAM2 PGO.
- Add ROS2 wrapper.

---

## 📝 Citation

If you found this work useful, please cite the following publication:

```bibtex
@article{stathoulopoulos2025msa,
  author={Stathoulopoulos, Nikolaos and Kanellakis, Christoforos and Nikolakopoulos, George},
  journal={IEEE Robotics and Automation Letters}, 
  title={A Minimal Subset Approach for Informed Keyframe Sampling in Large-Scale SLAM}, 
  year={2026},
  volume={11},
  number={1},
  pages={738-745},
  doi={10.1109/LRA.2025.3636035}
}
```

🏅 An earlier version was presented and won the Best Paper Award at IROS 2024 [Standing the Test of Time Workshop: Retrospective and Future of World Representations for Lifelong Robotics](https://montrealrobotics.ca/test-of-time-workshop/papers/)
