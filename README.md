<div align="center">
  
<h1>A Minimal Subset Approach (MSA)</h1>
<h2>for Informed Keyframe Sampling in Large-scale SLAM</h2>

[![arXiv](https://img.shields.io/badge/Arxiv-2501.01791-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2501.01791)
[![DOI:10.1109/LRA.2025.3623045](https://img.shields.io/badge/IEEE-10.1109/LRA.2025.3636035-00629B.svg)](https://doi.org/10.1109/LRA.2025.3636035)
![Python](https://img.shields.io/badge/Python-3670A0?logo=python&logoColor=ffdd54)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)
    
[**Nikolaos Stathoulopoulos**](https://github.com/nstathou) · [**Christoforos Kanellakis**](https://github.com/christoforoskanel) · [**George Nikolakopoulos**](https://github.com/geonikolak)

</div>

<p align=center> <img src="/ral25_pipeline.png" width="75%" height="75%"/> </p>

<h2>💡 Introduction</h2>

**Abstract:** Typical LiDAR SLAM architectures feature a front-end for odometry estimation and a back-end for refining and optimizing the trajectory and map, commonly through loop closures. However, loop closure detection in large-scale missions presents significant computational challenges due to the need to identify, verify, and process numerous candidate pairs for pose graph optimization. Keyframe sampling bridges the front-end and back-end by selecting frames for storing and processing during global optimization. This article proposes an online keyframe sampling approach that constructs the pose graph using the most impactful keyframes for loop closure. We introduce the Minimal Subset Approach (MSA), which optimizes two key objectives: redundancy minimization and information preservation, implemented within a sliding window framework. By operating in the feature space rather than 3-D space, MSA efficiently reduces redundant keyframes while retaining essential information. Evaluations on diverse public datasets show that the proposed approach outperforms naive methods in reducing false positive rates in place recognition, while delivering superior ATE and RPE in metric localization, without the need for manual parameter tuning. Additionally, MSA demonstrates efficiency and scalability by reducing memory usage and computational overhead during loop closure detection and pose graph optimization.

<!-- <img src="./figures/original.gif" width="49%" height="50%"/> <img src="./figures/decompressed.gif" width="49%" height="50%"/> -->

***Code will be released before February 2026***

<h2>📝 Citation</h2>

If you found this work useful, please cite the following publication (to appear in RA-L25):

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

An earlier version was presented and won the Best Paper Award at IROS 2024 [Standing the Test of Time Workshop: Retrospective and Future of World Representations for Lifelong Robotics](https://montrealrobotics.ca/test-of-time-workshop/papers/)
