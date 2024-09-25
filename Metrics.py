from abc import ABC, abstractmethod

import torch


class Metric(ABC):
    """ Abstract base class for a distance metric. """

    @abstractmethod
    def preprocess(self, D):
        pass

    @abstractmethod
    def calculate(self, a, b):
        pass


class EuclideanMetric(Metric):
    def preprocess(self, D):
        pass

    def calculate(self, a, b, squared=False):
        res = torch.sum((a.float() - b) ** 2, dim=-1)
        return res if squared else torch.sqrt(res)


class CosineMetric(Metric):
    def preprocess(self, D):
        pass

    def calculate(self, a, b):
        dot_product = torch.sum(a * b, dim=-1)
        norms_a = torch.norm(a.float(), p=2, dim=-1)
        norms_b = torch.norm(b.float(), p=2, dim=-1)
        res = 1 - dot_product / (norms_a * norms_b)
        return torch.clamp(res, 1e-30, 2)


class MahalanobisMetric(Metric):
    def __init__(self, shrinkage=False, gamma_1=1, gamma_2=1, normalization=False):
        self.shrinkage = shrinkage
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.normalization = normalization
        self.preprocessed = False

    def preprocess(self, D):
        # D (torch.Tensor): Dataset tensor of shape [n_classes, samples_per_class, n_features].
        if not self.preprocessed:
            self.preprocessed = True
            self.n_classes = D.size(0)
            self.samples_per_class = D.size(1)
            self.n_features = D.size(2)
            self.inv_cov_matrix = self._preprocess(D)
        else:
            self.n_classes += D.size(0)
            task_inv_cov_matrix = self._preprocess(D)
            self.inv_cov_matrix = torch.concat((self.inv_cov_matrix, task_inv_cov_matrix))

    def _covariance_shrinkage(self, D, cov_matrix):
        diag = cov_matrix.diagonal(dim1=1, dim2=2)
        V1 = diag.mean(1).reshape(-1, 1, 1).to(D.device)
        V2 = ((cov_matrix.sum((1, 2)) - diag.sum(1)) / (self.n_features * (self.n_features - 1))).reshape(-1, 1, 1)
        Id = torch.eye(self.n_features).repeat(D.size(0), 1, 1).to(D.device)
        return cov_matrix + self.gamma_1 * V1 * Id + self.gamma_2 * V2 * (1 - Id)

    def _normalization(self, cov_matrix):
        stds = torch.sqrt(self.cov_matrix.diagonal(dim1=1, dim2=2))
        return cov_matrix / torch.einsum('bi,bj->bij', stds, stds)

    def _preprocess(self, D):
        # Compute the Covariance Matrix  [n_classes, n_features, n_features]
        cov_matrix = D - torch.mean(D, dim=1, keepdim=True)
        cov_matrix = torch.matmul(cov_matrix.transpose(1, 2), cov_matrix) / self.n_features

        if self.shrinkage:
            cov_matrix = self._covariance_shrinkage(D, cov_matrix)

        if self.normalization:
            self.cov_matrix = cov_matrix
            cov_matrix = self._normalization(cov_matrix)

        # Calculate the inverse of Covariance Matrix
        return torch.inverse(cov_matrix)

    def calculate(self, a, b):
        """
        Calculate the squared Mahalanobis distance between tensors a and b.

        Parameters:
        a (torch.Tensor): First tensor of shape [n_classes, n_samples_a, n_features].
        b (torch.Tensor): Second tensor of shape [n_samples_b, n_features].

        Returns:
        torch.Tensor: Mahalanobis distance between a and b. Shape [n_samples_b, n_classes, n_examples].
        """
        a = a.reshape(1, self.n_classes, -1, self.n_features)
        b = b.reshape(-1, 1, 1, self.n_features)

        diff = b - a  # [n_samples, n_classes, n_examples, n_features]
        return torch.einsum('abcd,bed,abce->abc', diff, self.inv_cov_matrix, diff)
