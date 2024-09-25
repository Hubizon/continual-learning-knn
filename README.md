# Continual Learning with KNN and K-Means

This project aims to improve continual learning models by applying K-Nearest Neighbors (KNN) with K-Means clustering. The ultimate goal is to surpass the performance of FeCAM, a state-of-the-art continual learning method, as outlined in the paper [FeCAM: Few-Shot Class-Incremental Learning via Feature Compression and Augmentation](https://arxiv.org/abs/2309.14062). A future paper will explore the proposed method in detail.

## Introduction

This project leverages KNN combined with K-Means for continual learning tasks. The main datasets include:
- **Visual Transformer Representations**: Dataset 1, which contains representations generated by a visual transformer model.
- **ResNet Representations**: Dataset 2, containing representations from a ResNet model.
  
Note: the datasets are not included in the files.

Preliminary experiments using MNIST and synthetic "moons" datasets helped in testing the initial KNN implementation. The project aims to refine these techniques to outperform FeCAM, with further research to be published in an upcoming paper.