# DGCWSL

[Dynamic Graph Consistent Weighted Subspace Learning for Cross-Domain Speech Emotion Recognition](https://ieeexplore.ieee.org/document/11373908)

# Abstract

In recent years, cross-domain speech emotion recognition (SER) has attracted considerable interest. Most transfer subspace learning based SER methods lack unified adaptive constraints, making it difficult to balance discriminative capability and domain alignment, which limits their cross-domain generalization. To address these problems, we propose a novel domain adaptation (DA) approach called dynamic graph consistent weighted subspace learning (DGCWSL). Specifically, DGCWSL first projects samples from the source and target domains into a shared low-dimensional discriminative subspace, then performs cross-domain instance reconstruction, representing each target as a weighted combination of source instances. In parallel, a dynamic graph is constructed to capture local structural information between domains while preserving the data manifold. Subsequently, label supervision and discriminative learning between domains are achieved through linear regression. Furthermore, we introduce an adaptive weighted matrix that enforces consistent feature contributions across the distance metric, instance alignment, and discriminative regression, thereby mitigating both overfitting and underfitting. Finally, extensive experiments are conducted on four public datasets. The results confirm the superiority of DGCWSL over several state-of-the-art DA methods.<br>

<img width="700" alt="image" src="https://image.thesky.top/frame7.png" /><br>

# Paper

If you find this project useful for your research, please cite:

```
@ARTICLE{wang2025dynamic,
  author={Wang, Yifan and Song, Peng and Fu, Siqi and Liu, Zhaowei and Wang, Changjia and Zheng, Wenming},
  journal={IEEE Transactions on Computational Social Systems}, 
  title={Dynamic Graph Consistent Weighted Subspace Learning for Cross-Domain Speech Emotion Recognition}, 
  year={2026},
  volume={13},
  number={3},
  pages={3529-3540}
```

If you have any questions, please email the author: yifanfengshun594993@gmail.com

