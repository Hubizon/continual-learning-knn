import torch


class KNNClassifier:
    def __init__(self, n_neighbors, metric, tukey_lambda=1, kmeans=None, device='cpu'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.tukey_lambda = tukey_lambda
        self.kmeans = kmeans
        self.device = device
        self.D = None
        self.n_classes = None
        self.samples_per_class = None
        self.fitted = False

    # Tukey’s Ladder of Powers Transformation
    def _tukey(self, T):
        if self.tukey_lambda != 0:
            return torch.pow(T, self.tukey_lambda)
        else:
            return torch.log(T)

    def fit(self, D):
        # D (torch.Tensor): Training data tensor of shape [n_classes, samples_per_class, n_features].
        if D.ndim == 2:
            D = D.unsqueeze(0)
        D = self._tukey(D.type(torch.float32).to(self.device))

        self.metric.preprocess(D)
        if self.kmeans is not None:
            D = self.kmeans.fit(D)

        if not self.fitted:
            self.fitted = True
            self.D = D
            self.n_classes = self.D.size(0)
            self.samples_per_class = self.D.size(1)
            self.n_features = self.D.size(2)
        else:
            self.D = torch.concat((self.D, D))
            self.n_classes = self.D.size(0)

        return self

    def predict(self, X, batch_size_X=1, batch_size_D=64, batch_size=-1):
        # X (torch.Tensor): Test data tensor of shape [n_samples, n_features].
        X = self._tukey(X)
        if batch_size_X == -1:
            batch_size_X = X.size(0)
        if batch_size_D == -1:
            batch_size_D = self.D.size(1)

        pred = []
        split_D = self.D[None, :, :, :].split(batch_size_D, dim=2)  # Split tensor D to speed up the prediction

        for batch_X in X[:, None, None, :].split(batch_size_X, dim=0):
            # Calculate the distances between the test sample and all training samples
            distances = torch.cat([self.metric.calculate(batch_D, batch_X) for batch_D in split_D],
                                  dim=2).reshape(batch_X.size(0), -1)

            # Get the distances and indices of the k closest training samples
            values, indices = torch.topk(distances, self.n_neighbors, sorted=False, largest=False)

            # Determine the class with the most neighbors
            classes = torch.zeros((batch_X.size(0), self.n_classes), dtype=torch.float32, device=self.device)
            classes.scatter_add_(1, indices // self.samples_per_class, 1. / values)
            pred.append(classes.argmax(1))

        return torch.cat(pred)

    @staticmethod
    def getD(X, y):
        """
        Transforms tensor X and y into tensor D.

        Parameters:
        X (torch.Tensor): Data tensor of shape [n_samples, n_features].
        y (torch.Tensor): Labels tensor of shape [n_samples].

        Returns:
        torch.Tensor: Transformed tensor D of shape [n_classes, samples_per_class, n_features].
        """
        n_classes = len(torch.unique(y))
        D = [[] for _ in range(n_classes)]
        for _X, _y in zip(X, y):
            D[_y].append(_X)

        # Trim tensor D so that there is the same amount of samples for each class
        min_len = min([len(d) for d in D])
        D = [d[:min_len] for d in D]
        return torch.stack([torch.stack(D[i]) for i in range(n_classes)]).type(torch.float32).to(X.device)

    @staticmethod
    def accuracy_score(y_true, pred):
        return torch.sum(torch.eq(y_true, pred)).item() / len(y_true)
