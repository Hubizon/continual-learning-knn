# Continual Learning with Mahalanobis Distance

This still on-going project explores improvements to continual learning by leveraging Mahalanobis Distance in combination with K-Means clustering and other approaches, including K-Nearest Neighbors (KNN), Multi-Layer Perceptron (MLP), and a custom method called GradKNNC. Our ultimate goal is to outperform FeCAM, a state-of-the-art method for few-shot class-incremental learning, as described in the paper [FeCAM: Few-Shot Class-Incremental Learning via Feature Compression and Augmentation](https://arxiv.org/abs/2309.14062). A future paper will explore the proposed methods in detail.

## Introduction

Continual learning addresses the challenge of incrementally learning new tasks without forgetting previously learned knowledge. This project aims to identify robust methods for continual learning by focusing on embedding space representations, including:
- **Visual Transformer Representations**: Captures high-dimensional features from transformer-based models.
- **ResNet Representations**: Extracts features from convolutional networks.

*Note: The datasets are not included in this repository. However, the proposed methods are adaptable to various datasets and will be evaluated on additional datasets in future experiments.*

## Methods
### KNNClassifier
The most successful method to date involves a K-Nearest Neighbors classifier combined with Mahalanobis distance. This approach calculates distances from training samples to centroids (computed via K-Means) and selects the class with the most neighbors using a weighted metric. Hyperparameter tuning with Optuna resulted in a 1-2% improvement over FeCAM's performance.
### MLPClassifier
An alternative method explored was training models (e.g., MLP, SVM, Logistic Regression) to predict classes based on the distances between samples and centroids. However, this approach was abandoned due to challenges in achieving competitive results.
### GradKNNClassifier
The GradKNNC method is an ongoing area of development. It trains four parameters per class using a custom formula and integrates techniques like normalization, regularization, and early stopping. We continue to explore variations and optimizations to enhance its performance.




## Disclaimer:
I am currently working on this project in a GMUM repository. Updates to this repository may be less frequent, but I will commit changes periodically to keep it in sync.
