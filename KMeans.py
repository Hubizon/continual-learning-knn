import torch

import Metrics


class KMeans:
    def __init__(self, n_clusters, max_iter=300, tol=0, metric=Metrics.EuclideanMetric()):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.metric = metric

    def fit(self, D):
        # centroids = self.kmeans_init(D)
        random_indices = torch.randperm(D.size(1))[:self.n_clusters]
        centroids = D[:, random_indices]

        for d_class in range(D.size(0)):
            for i in range(self.max_iter):
                # distances = self.metric.calculate(D[d_class].unsqueeze(1), centroids[d_class].unsqueeze(0))
                distances = torch.cdist(D[d_class], centroids[d_class])
                cluster_labels = torch.argmin(distances, dim=1)
                new_centroids = torch.stack([D[d_class, (cluster_labels == j)].mean(dim=0)
                                             for j in range(self.n_clusters)])

                if torch.norm(new_centroids - centroids[d_class]) <= self.tol:
                    break
                centroids[d_class] = new_centroids

        return centroids
