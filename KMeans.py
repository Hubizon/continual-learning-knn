import torch

import Metrics


class KMeans:
    def __init__(self, n_clusters, max_iter=100, tol=0, metric=Metrics.EuclideanMetric()):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.metric = metric

    def kmeans_plusplus(self, D):
        n_classes, n_samples, n_features = D.shape
        centroids = [D[:, torch.randint(0, n_samples, (1,))]]

        for _ in range(self.n_clusters - 1):
            dists = torch.sum(torch.cat([torch.cdist(D, centroid) for centroid in centroids], dim=-1), dim=-1).to(D.device)
            indices = torch.multinomial(dists, num_samples=1)
            centroids.append(D[torch.arange(D.size(0)), indices.flatten()].unsqueeze(1))

        return torch.stack(centroids, dim=1).squeeze(2)

    def fit_predict(self, D, init='k-means++'):
        if init == 'k-means++':
            centroids = self.kmeans_plusplus(D)
        else:
            random_indices = torch.randperm(D.size(1))[:self.n_clusters]
            centroids = D[:, random_indices]

        for d_class in range(D.size(0)):
            for i in range(self.max_iter):
                distances = self.metric.calculate(D[d_class].unsqueeze(1), centroids[d_class].unsqueeze(0))
                cluster_labels = torch.argmin(distances, dim=1)
                new_centroids = torch.stack([D[d_class, (cluster_labels == j)].mean(dim=0)
                                             for j in range(self.n_clusters)])

                for c in torch.argwhere(torch.any(torch.isnan(new_centroids), axis=1)).flatten():
                    new_centroids[c] = centroids[d_class, c]
                if torch.norm(new_centroids - centroids[d_class]) <= self.tol:
                    break
                centroids[d_class] = new_centroids

        return centroids
