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
- [🕹️ Run the Examples](#-run-the-examples) — [Configuration](#configuration) • [Evaluations](#running-evaluations) • [Visualization](#live-visualization)
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

1. Clone the repository to your desired working directory:

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

There is currently support for the following datasets:

- [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)/[SemanticKITTI](https://semantic-kitti.org/dataset.html#download)
- [Apollo-SouthBay](https://developer.apollo.auto/southbay.html)
- [MulRan](https://sites.google.com/view/mulran-pr/dataset)
- [NewerCollege](https://ori-drs.github.io/newer-college-dataset/)

For the datasets you want to use, download the dataset from the respective site and check the corresponding handler script in the /optkey_utils folder e.g. KITTI_Handler.py, MulRan_Handler.py etc. to make sure that your dataset has the expected directory structure.

### Custom Dataset

If you want to test with another dataset that is currently not supported, all you need to do is make a respective handling script that follows the same structure as the handlers already provided KITTI_Handler.py, MulRan_Handler.py. Mainly functions like load_pose and load_scan.

### Descriptor Setup

There is currently support for the following descriptors:

- [OverlapTransformer](https://github.com/haomo-ai/OverlapTransformer)
- [ScanContext](https://github.com/gisbi-kim/scancontext)

In the /optkey_utils/descriptors install the descriptor you want to use following the instructions by the original developers.

### Custom Descriptor

If you want to test with another descriptor you need to make the handler similar to OT_Handler.py and SC_Handler.py following the DescriptorHandler.py template. Basically you need to specify the descriptor extraction process and the similarity/distance function.

## 🕹️ Run the Examples

An example script is provided:

- **pr_example.py** — Evaluates keyframe sampling methods with Precision-Recall metrics

### Configuration

Update the config file in `/config/` with your dataset and descriptor paths. Template configs for each dataset are provided (e.g., `pr_config.yaml`, `kitti_config.yaml`).

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

<p align=left> <img src="/results/pr_curve_SemanticKitti_00_ot.png" width="75%" height="75%"/> </p>

### Live Visualization

Enable real-time visualization of the sampling process:

```bash
python3 examples/pr_example.py --live
```

Key controls:
- **P** — Pause/resume playback
- **Right Arrow** — Step forward one frame

**Note:** Running without visualization is significantly faster for batch evaluation.

## 🔜 Coming Soon

TODO:
- Add example with iSAM2 PGO.
- Add ROS2 wrapper.

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
