import time
import unittest

import sklearn
import torch
import torchvision
from sklearn import datasets, neighbors

import Metrics
from KNNClassifier import KNNClassifier


class TestMetrics(unittest.TestCase):
    def test_euclidean_calculate(self):
        """ Test the Euclidean distance calculation with 3 features. """
        a = torch.tensor([[[2, 3, 4], [3, 4, 0]],
                          [[0, -3, -6], [-2, 6, 4]]])
        b = torch.tensor([2, 3, 4], dtype=torch.float32)

        metric = Metrics.EuclideanMetric()

        dist_test = metric.calculate(a, b)
        dist_correct = torch.tensor([[0, torch.sqrt(torch.tensor(18))], [torch.sqrt(torch.tensor(140)), 5]])

        self.assertTrue(torch.allclose(dist_test, dist_correct, atol=1e-4))

    def test_cosine_calculate(self):
        """ Test the Cosine distance calculation using sklearn function. """
        a = torch.rand((10, 300), dtype=torch.float32)
        b = torch.rand(300, dtype=torch.float32)

        metric = Metrics.CosineMetric()

        dist_test = metric.calculate(a, b)
        dist_correct = torch.tensor(sklearn.metrics.pairwise.cosine_distances(a, b.reshape(1, -1)),
                                    dtype=torch.float32).flatten()

        self.assertTrue(torch.allclose(dist_test, dist_correct, atol=1e-4))

    def test_mahalanobis_distance(self):
        D = torch.tensor([[[1, 2], [2, 3], [3, 5]], [[5, 7], [4, -3], [10, 0]]], dtype=torch.float32)
        a = torch.mean(D, 1).unsqueeze(1)
        b = torch.tensor([[2, 4]])

        metric = Metrics.MahalanobisMetric()
        metric.preprocess(D)

        dist_test = metric.calculate(a, b)
        dist_correct = torch.tensor([[[5.33336], [1.99815]]])

        self.assertTrue(torch.allclose(dist_test, dist_correct, atol=1e-4))

    def test_mahalanobis_distance_2(self):
        D = torch.tensor([[[1, 2], [2, 3], [3, 5]], [[5, 7], [4, -3], [10, 0]]], dtype=torch.float32)
        a = torch.mean(D, 1).unsqueeze(1)
        b = torch.tensor([[2, 4], [-3, 0], [1, 2]])

        metric = Metrics.MahalanobisMetric()
        metric.preprocess(D)

        dist_test = metric.calculate(a, b)
        dist_correct = torch.tensor([[[5.33336], [1.99815]], [[233.33333], [8.64758]], [[1.33333], [2.75284]]])

        self.assertTrue(torch.allclose(dist_test, dist_correct, atol=1e-4))


class TestKNNClassifier(unittest.TestCase):
    def test_get_d(self):
        """ Test the transformation of X and y into the tensor D. """
        X = torch.tensor([[3, 2], [-5, 3], [2, 0], [4, 2], [-1, -1]])
        y = torch.tensor([0, 1, 1, 0, 0])
        D = torch.tensor([[[3, 2], [4, 2]], [[-5, 3], [2, 0]]])

        self.assertTrue(torch.equal(KNNClassifier.getD(X, y), D))

    def moons_sklearn_helper(self, metric, metric_sklearn, device):
        """ Helper function to test the KNN classifier with the moons dataset against sklearn's implementation. """
        # Create a moons dataset (due to how sklearn breaks ties, n_neighbors must be an odd number)
        train_samples, test_samples, n_neighbors = 200, 100, 3
        noise = 0.2
        X_train, y_train = datasets.make_moons(n_samples=train_samples, noise=noise)
        X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train)
        X_test, y_test = datasets.make_moons(n_samples=test_samples, noise=noise)
        X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test)

        # Transform X_train and y_train into D for my KNN
        D = KNNClassifier.getD(X_train, y_train)

        # Initialize my KNN and predict classes of some test samples
        knn1 = KNNClassifier(n_neighbors, metric, device=device).fit(D)
        pred1 = knn1.predict(X_test.to(device))

        if metric_sklearn != 'mahalanobis':
            # Initialize sklearn KNN and predict classes of some test samples
            knn2 = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric_sklearn,
                                                  algorithm='brute').fit(X_train, y_train)
            pred2 = torch.tensor(knn2.predict(X_test), dtype=torch.int32, device=device)

            self.assertTrue(torch.equal(pred1, pred2))

    def test_moons_sklearn_euclidean_cpu(self):
        """ Test the KNN classifier on the moons dataset using the CPU with Euclidean metric. """
        self.moons_sklearn_helper(Metrics.EuclideanMetric(), 'euclidean', 'cpu')

    def test_moons_sklearn_cosine_cpu(self):
        """ Test the KNN classifier on the moons dataset using the CPU with Cosine metric. """
        self.moons_sklearn_helper(Metrics.CosineMetric(), 'cosine', 'cpu')

    def test_moons_sklearn_mahalanobis_cpu(self):
        """ Test the KNN classifier on the moons dataset using the CPU with Cosine metric. """
        self.moons_sklearn_helper(Metrics.MahalanobisMetric(True, True), 'mahalanobis', 'cpu')

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA not available')
    def test_moons_sklearn_euclidean_gpu(self):
        """ Test the KNN classifier on the moons dataset using the GPU with Cosine metric. """
        self.moons_sklearn_helper(Metrics.EuclideanMetric(), 'euclidean', 'cuda')

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA not available')
    def test_moons_sklearn_cosine_gpu(self):
        """ Test the KNN classifier on the moons dataset using the CPU with Cosine metric. """
        self.moons_sklearn_helper(Metrics.CosineMetric(), 'cosine', 'cuda')

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA not available')
    def test_moons_sklearn_mahalanobis_gpu(self):
        """ Test the KNN classifier on the moons dataset using the CPU with Cosine metric. """
        self.moons_sklearn_helper(Metrics.MahalanobisMetric(True, True), 'mahalanobis', 'cuda')

    def mnist_sklearn_helper(self, metric, metric_sklearn, device):
        """ Helper function to test the KNN classifier with the MNIST dataset against sklearn's implementation. """
        # Due to how sklearn breaks ties, n_neighbors must be equal to 1 to get the same results.
        n_neighbors = 1

        # Download MNIST dataset
        train = torchvision.datasets.MNIST('/files/', train=True, download=True)
        X_train = train.data.reshape(-1, 28 * 28).type(torch.float32)
        y_train = train.targets

        test = torchvision.datasets.MNIST('/files/', train=False, download=True)
        X_test = test.data.reshape(-1, 28 * 28).type(torch.float32)[:1000]
        y_test = test.targets[:1000]

        # Transform X_train and y_train into D for my KNN (which requires the same amount of samples per each class)
        D = KNNClassifier.getD(X_train, y_train)

        # Transform D into X_train and y_train for sklearn KNN (so that it has the same samples as my KNN)
        X_train = D.reshape(-1, 28 * 28)
        y_train = torch.tensor([[i] * D.size(1) for i in range(10)]).flatten()

        # Initialize my KNN and predict classes of some test samples
        start = time.time()
        knn1 = KNNClassifier(n_neighbors, metric, device=device).fit(D)
        pred1 = knn1.predict(X_test)
        print('my knn: ', time.time() - start, KNNClassifier.accuracy_score(y_test, pred1))

        if metric_sklearn != 'mahalanobis':
            # Initialize sklearn KNN and predict classes of some test samples
            start = time.time()
            knn2 = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric_sklearn,
                                                  algorithm='brute').fit(X_train, y_train)
            pred2 = torch.tensor(knn2.predict(X_test), dtype=torch.int32)
            print('sklearn knn: ', time.time() - start, sklearn.metrics.accuracy_score(y_test, pred2))

            self.assertTrue(torch.equal(pred1, pred2))

    def test_mnist_sklearn_euclidean_cpu(self):
        """ Test the KNN classifier on the MNIST dataset using the CPU with Euclidean metric. """
        self.mnist_sklearn_helper(Metrics.EuclideanMetric(), 'euclidean', 'cpu')

    def test_mnist_sklearn_cosine_cpu(self):
        """ Test the KNN classifier on the MNIST dataset using the CPU with Cosine metric. """
        self.mnist_sklearn_helper(Metrics.CosineMetric(), 'cosine', 'cpu')

    def test_mnist_sklearn_mahalanobis_cpu(self):
        """ Test the KNN classifier on the MNIST dataset using the CPU with Cosine metric. """
        self.mnist_sklearn_helper(Metrics.MahalanobisMetric(True, True), 'mahalanobis', 'cpu')

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA not available')
    def test_mnist_sklearn_euclidean_gpu(self):
        """ Test the KNN classifier on the MNIST dataset using the GPU with Euclidean metric. """
        self.mnist_sklearn_helper(Metrics.EuclideanMetric(), 'euclidean', 'cuda')

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA not available')
    def test_mnist_sklearn_cosine_gpu(self):
        """ Test the KNN classifier on the MNIST dataset using the GPU with Cosine metric. """
        self.mnist_sklearn_helper(Metrics.CosineMetric(), 'cosine', 'cuda')

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA not available')
    def test_mnist_sklearn_mahalanobis_gpu(self):
        """ Test the KNN classifier on the MNIST dataset using the GPU with Cosine metric. """
        self.mnist_sklearn_helper(Metrics.MahalanobisMetric(True, True), 'mahalanobis', 'cuda')


if __name__ == '__main__':
    unittest.main()
