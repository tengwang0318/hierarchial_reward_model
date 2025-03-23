This is the official code for the paper [**towards hierarchical multi-step reward models for enhanced reasoning in large language models**](https://arxiv.org/abs/2503.13551). 

This project has two parts for illustrating this paper:

### 1. Training with Manual Annotations and Best-of-N Evaluation

- Manual annotations are utilized to train three reward models:
  - **ORM** (Output Reward Model)
  - **PRM** (Process Reward Model)
  - **HRM** (Hierarchical Reward Model)
- The policy model is evaluated using the **best-of-N** strategy based on these reward models.
- Relevant code:
  - The `construct_dataset` folder is used to generate training data for ORM, PRM, and HRM.
  - The `sft_rw_manual_annotation` folder contains code for fine-tuning the reward models and evaluating the policy model.

### 2. Self-Supervised Training with MCTS and HNC

- **MCTS** (Monte Carlo Tree Search) and **HNC** (Hierachical Node Compression, as described in the paper) are implemented to automatically generate and label PRM and HRM training data.
- The automatically generated **PRM800K** dataset is used to train PRM and HRM models.
- Evaluation is conducted on the following datasets:
  - **PRM800K**
  - **MATH500** (Cross-domain)
  - **GSM8K** (Cross-domain)
- A policy model trained with SFT is included, which incorporates **KL divergence** from a reference model to enhance reasoning ability.
- Relevant code is located in the `self-training` folder.

If you find our work useful, please consider citing it in your research.

```
@article{wang2025towards,
  title={Towards Hierarchical Multi-Step Reward Models for Enhanced Reasoning in Large Language Models},
  author={Wang, Teng and Jiang, Zhangyi and He, Zhenqi and Yang, Wenhan and Zheng, Yanan and Li, Zeyu and He, Zifan and Tong, Shenyang and Gong, Hailei},
  journal={arXiv preprint arXiv:2503.13551},
  year={2025}
}
```

