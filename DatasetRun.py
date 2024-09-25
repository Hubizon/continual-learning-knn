import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import optuna
import h5py
import sys

OPTUNA_DB_PATH = 'sqlite:///./results/optuna_study.db'


def get_device(verbose=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print(f'Device used: {device}')
    return device


def is_jupyter():
    return 'ipykernel' in sys.modules


def train(clf, folder_name, n_tasks, only_last=False, verbose=False):
    device = clf.device

    for task_number in range(n_tasks):
        current_file = f"{folder_name}/task_{task_number}.hdf5"

        with h5py.File(current_file, "r") as f:
            X_train = torch.tensor(f["X_train"][:], dtype=torch.float32, device=device)
            y_train = torch.tensor(f["y_train"][:], dtype=torch.float32, device=device)

            X_test = torch.tensor(f["X_test"][:], dtype=torch.float32, device=device)
            y_test = torch.tensor(f["y_test"][:], dtype=torch.float32, device=device)

            D = torch.concat([X_train[y_train == y_class].unsqueeze(0) for y_class in y_train.unique()])

            if verbose:
                start = time.time()

            clf.fit(D)
            if not only_last or task_number == n_tasks - 1:
                pred = clf.predict(X_test, batch_size_X=1, batch_size_D=-1)
                accuracy = clf.accuracy_score(y_test, pred)

            if verbose:
                end = time.time()
                print(f'task {task_number}: (time: {(end - start):.3f})')
                print(f"Paper accuracy: {f['info'].attrs['accuracy']:.3f}; My accuracy: {accuracy:.3f}")

    return accuracy


def grid_search(objective, study_name, n_trials, sampler=optuna.samplers.TPESampler(), restart=False, n_jobs=4,
                verbose=3):
    verbose_levels = [optuna.logging.CRITICAL, optuna.logging.ERROR, optuna.logging.WARNING,
                      optuna.logging.INFO, optuna.logging.DEBUG]
    optuna.logging.set_verbosity(verbose_levels[verbose])

    if restart:
        optuna.delete_study(study_name=study_name, storage=OPTUNA_DB_PATH)
    study = optuna.create_study(sampler=sampler,
                                direction='maximize',
                                study_name=study_name,
                                storage=OPTUNA_DB_PATH,
                                load_if_exists=True)

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    if verbose >= 3:
        print("Best hyperparameters: ", study.best_params)
        print("Best accuracy: ", study.best_value)


def save_to_csv(study_name, only_complete=True):
    loaded_study = optuna.load_study(
        study_name=study_name,
        storage=OPTUNA_DB_PATH
    )

    df = loaded_study.trials_dataframe()
    if only_complete:
        df.drop(df[df.state != 'COMPLETE'].index, inplace=True)
    df.to_csv(f"./results/{study_name}.csv", index=False)


def load_from_csv(study_name):
    return pd.read_csv(f"./results/{study_name}.csv")


def plot_accuracy_trials(study_name, ylim=True):
    df = load_from_csv(study_name)
    accuracies = df['value'].values

    plt.plot(accuracies)
    if ylim:
        plt.ylim(bottom=(accuracies.mean() - accuracies.std()))

    plt.xlabel('Trial')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over trials')
    plt.show()


def plot_hyperparameter(param_name, param_vals, accuracies, deg=2, ylim=True):
    plt.scatter(param_vals, accuracies, color='blue', s=5, alpha=0.6)

    if ylim:
        mask = accuracies.mean() - accuracies.std() <= accuracies
        param_vals = param_vals[mask]
        accuracies = accuracies[mask]

    z = np.polyfit(param_vals, accuracies, deg)
    p = np.poly1d(z)
    x_range = np.linspace(min(param_vals), max(param_vals), 500)
    plt.plot(x_range, p(x_range), "g-", linewidth=2)

    plt.xlabel(param_name)
    plt.ylabel('accuracy')
    plt.title(f'{param_name} vs accuracy')
    plt.grid(True)


def plot_hyperparameters(study_name, columns=3, deg=2, ylim=True):
    df = load_from_csv(study_name)
    accuracies = df['value'].values

    params = []
    for key in df.keys():
        if key.startswith('params_'):
            params.append(key)

    rows = math.ceil(len(params) / columns)
    width, height = 7.5, 5.5
    plt.figure(figsize=(columns * width, rows * height))

    for i, param in enumerate(params):
        plt.subplot(rows, columns, i + 1)
        plot_hyperparameter(param[7:], df[param].values, accuracies, deg, ylim)

    plt.show()


def print_results(study_name, only_important=True):
    df = load_from_csv(study_name)
    df_sorted = df.sort_values(by=['value'], ascending=False)
    if only_important:
        for key in df_sorted.keys():
            if key.startswith('params_'):
                df_sorted[key[7:]] = df_sorted[key]
            if key != "value":
                df_sorted.drop(key, axis=1, inplace=True)

    return df_sorted

