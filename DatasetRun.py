import math
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from hyperopt import fmin, tpe, Trials, STATUS_OK


def get_device(verbose=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print(f'Device used: {device}')
    return device


def train(clf, n_tasks, folder_name, verbose=False):
    device = clf.device

    accuracies = []
    for task_number in range(n_tasks):
        current_file = f"./{folder_name}/task_{task_number}.hdf5"

        with h5py.File(current_file, "r") as f:
            X_train = torch.tensor(f["X_train"][:], dtype=torch.float32, device=device)
            y_train = torch.tensor(f["y_train"][:], dtype=torch.float32, device=device)

            X_test = torch.tensor(f["X_test"][:], dtype=torch.float32, device=device)
            y_test = torch.tensor(f["y_test"][:], dtype=torch.float32, device=device)

            D = torch.concat([X_train[y_train == y_class].unsqueeze(0) for y_class in y_train.unique()])

            if verbose:
                start = time.time()

            clf.fit(D)
            pred = clf.predict(X_test, batch_size_X=1, batch_size_D=-1)
            accuracy = clf.accuracy_score(y_test, pred)
            accuracies.append(accuracy)

            if verbose:
                end = time.time()
                print(f'task {task_number}: (time: {(end - start):.3f})')
                print(
                    f"Paper accuracy: {f['info'].attrs['accuracy']:.3f}; Mahalanobis knn accuracy: {(accuracy * 100):.3f}")

    return accuracies


def grid_search(fn, space, max_evals):
    trials = Trials()
    best = fmin(
        fn=fn,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )

    return trials, best


def plot_accuracy_trials(trials):
    accuracies = -np.array(trials.losses())
    plt.plot(accuracies)
    plt.xlabel('Trial')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over trials')
    plt.show()


def plot_hyperparameter(trials, param, deg=2, ylimit=True):
    vals = trials.vals[param]
    accuracies = -np.array(trials.losses())
    plt.scatter(vals, accuracies, color='blue', s=5, alpha=0.6)

    z = np.polyfit(vals, accuracies, deg)
    p = np.poly1d(z)
    x_range = np.linspace(min(vals), max(vals), 500)
    plt.plot(x_range, p(x_range), "g-", linewidth=2)

    if ylimit:
        acc_mean, acc_std = accuracies.mean(), accuracies.std()
        plt.ylim(acc_mean - acc_std, acc_mean + acc_std)

    plt.xlabel(param)
    plt.ylabel('accuracy')
    plt.title(f'{param} vs accuracy')
    plt.grid(True)


def plot_hyperparameters(trials, columns=3, deg=2, ylimit=True):
    rows = math.ceil(len(trials.vals.keys()) / columns)
    width, height = 7.5, 5.5
    plt.figure(figsize=(columns * width, rows * height))
    for i, param in enumerate(trials.vals.keys()):
        plt.subplot(rows, columns, i + 1)
        plot_hyperparameter(trials, param, deg, ylimit)
    plt.show()


def extract_results(trials):
    trials_list = []
    for trial in trials.trials:
        if trial['result']['status'] == STATUS_OK:
            trials_list.append({
                'loss': trial['result']['loss'],
                'params': trial['misc']['vals']
            })

    # Sort by the loss (objective function value)
    sorted_trials = sorted(trials_list, key=lambda x: x['loss'])

    return sorted_trials


def print_results(trials, k=10):
    best_trials = extract_results(trials)[:k]

    # Pretty print the top 10 results
    print(f"Top {k} Results:")
    print("=" * 30)
    for idx, trial in enumerate(best_trials):
        print(f"Trial {idx + 1}:")
        print(f"  Accuracy: {-trial['loss']:.4f}")
        print(f"  Parameters:")
        for param, value in trial['params'].items():
            # Show the actual value (hyperopt stores lists of values)
            print(f"    {param}: {value[0]:.4f}")
        print("-" * 30)
