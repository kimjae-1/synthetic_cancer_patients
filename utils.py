import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve

import matplotlib as mpl
import matplotlib.pyplot as plt

import utils


def recon_loss(y_true, y_pred, epoch=None, wait_kl=None):
    """
    Arguements:
        y_true -- shape: (N, (x, m), t, d)
        y_pred -- {outputs_recon, q_z_mean, q_z_logvar}
            outputs_recon -- shape: (N, t, d)
            q_z_mean -- shape: (N, t_ref, d_l)
            q_z_logvar -- shape: (N, t_ref, d_l)
    Returns:
        loss -- shape: (N, )
    """

    def log_normal_pdf(x, mean, logvar, mask):
        const = tf.convert_to_tensor(np.array([2 * np.pi]), dtype=tf.float32)
        const = tf.math.log(const)
        pdf = -0.5 * (const + logvar + (x - mean) ** 2 / tf.math.exp(logvar)) * mask

        return pdf

    def kl_divergence_normal(mu1, logvar1, mu2, logvar2):
        var1 = tf.math.exp(logvar1)
        var2 = tf.math.exp(logvar2)

        kl = 0.5 * (tf.math.log(var2 / var1) + (var1 + (mu1 - mu2) ** 2) / var2 - 1)

        return kl

    X_true = y_true[:, 0, :, :]
    m = y_true[:, 1, :, :]

    X_pred = y_pred["outputs_recon"]
    q_z_mean = y_pred["q_z_mean"]
    q_z_logvar = y_pred["q_z_logvar"]

    X_pred_std = 0.01 * tf.ones_like(X_pred, dtype=tf.float32)
    X_pred_logvar = 2 * tf.math.log(X_pred_std)

    p_z_mean = tf.zeros_like(q_z_mean, dtype=tf.float32)
    p_z_logvar = tf.zeros_like(q_z_logvar, dtype=tf.float32)

    logpx = tf.reduce_sum(log_normal_pdf(X_true, X_pred, X_pred_logvar, m), axis=(1, 2))
    kl = tf.reduce_sum(kl_divergence_normal(q_z_mean, q_z_logvar, p_z_mean, p_z_logvar), axis=(1, 2))

    logpx = logpx / tf.reduce_sum(m, axis=(1, 2))
    kl = kl / tf.reduce_sum(m, axis=(1, 2))

    if wait_kl is not None:
        if epoch < wait_kl:
            kl_coef = 0
        else:
            kl_coef = (1 - 0.99 ** (epoch - wait_kl))
    else:
        kl_coef = 1

    loss = - (logpx - kl_coef * kl)

    return loss


def MSE(y_true, y_pred):
    """
    Arguements:
        y_true -- shape: (N, (x, m), t, d)
        y_pred -- {outputs_recon, q_z_mean, q_z_logvar}
            outputs_recon -- shape: (N, t, d)
            q_z_mean -- shape: (N, t_ref, d_l)
            q_z_logvar -- shape: (N, t_ref, d_l)
        data_normalize -- (X_train_min, X_train_max, cat_axis)
    Returns:
        loss -- shape: (N, )
    """

    X_true = y_true[:, 0, :, :]
    m = y_true[:, 1, :, :]

    X_pred = y_pred["outputs_recon"]

    numerator = tf.reduce_sum(tf.math.square((X_true - X_pred) * X_denorm) * m, axis=(1, 2))
    denominator = tf.reduce_sum(m, axis=(1, 2)) + 1e-5
    values = numerator / denominator

    return values


def MAE(y_true, y_pred):
    """
    Arguements:
        y_true -- shape: (N, (x, m), t, d)
        y_pred -- {outputs_recon, q_z_mean, q_z_logvar}
            outputs_recon -- shape: (N, t, d)
            q_z_mean -- shape: (N, t_ref, d_l)
            q_z_logvar -- shape: (N, t_ref, d_l)
    Returns:
        loss -- shape: (N, )
    """

    X_true = y_true[:, 0, :, :]
    m = y_true[:, 1, :, :]

    X_pred = y_pred["outputs_recon"]

    numerator = tf.reduce_sum(tf.math.abs(X_true - X_pred) * m, axis=(1, 2))
    denominator = tf.reduce_sum(m, axis=(1, 2)) + 1e-5
    values = numerator / denominator

    return values


def MRE(y_true, y_pred):
    """
    Arguements:
        y_true -- shape: (N, (x, m), t, d)
        y_pred -- {outputs_recon, q_z_mean, q_z_logvar}
            outputs_recon -- shape: (N, t, d)
            q_z_mean -- shape: (N, t_ref, d_l)
            q_z_logvar -- shape: (N, t_ref, d_l)
    Returns:
        loss -- shape: (N, )
    """

    X_true = y_true[:, 0, :, :]
    m = y_true[:, 1, :, :]

    X_pred = y_pred["outputs_recon"]

    numerator = tf.reduce_sum(tf.math.abs((X_true - X_pred) / (X_true + 1e-4)) * 100 * m, axis=(1, 2))
    denominator = tf.reduce_sum(m, axis=(1, 2)) + 1e-5
    values = numerator / denominator

    return values


def plot_learning(history_train, history_valid, figsize=(15, 20), save=False, filepath='.', filename="Figure"):
    mpl.rcParams['figure.figsize'] = figsize
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.clf()

    name = ['Loss', 'Loss_pred', 'Loss_recon',
            'Accuracy', 'AUROC', 'AUPRC',
            'Precision', 'Recall', 'F1 score',
            'MSE', 'MAE', 'MRE']

    for i in range(len(name)):
        plt.subplot(4, 3, i + 1)
        plt.plot(np.arange(1, len(history_train[i])+1), history_train[i], color=colors[0], label='Train')
        plt.plot(np.arange(1, len(history_valid[i])+1), history_valid[i], color=colors[1], linestyle="--", label='Valid')

        plt.xlabel('Epoch')
        plt.ylabel(name[i])

        plt.legend()

    if save:
        os.makedirs(filepath, exist_ok=True)
        filepath = os.path.join(filepath, filename + '.png')
        plt.savefig(filepath)


def print_results(model, tensor, metrics, loss_weight, stage="Train", plot=False, save=False, filepath='.', filename="Figure"):
    """
    Returns:
        results -- (Loss, Loss_pred, Loss_recon, Accuracy, AUROC, AUPRC, Precision, Recall, F1 Score, MSE, MAE, MRE)
    """

    lw_pred, lw_recon = loss_weight
    Y_pred_full, Y_pred_hat_full = [], []
    results = []

    for step, (X, Y) in enumerate(tensor):
        Y_pred = Y["pred"]
        Y_recon = Y["recon"]

        Y_hat = model(X)
        Y_pred_hat = Y_hat["pred"]
        Y_recon_hat = Y_hat["recon"]

        loss_pred = binary_crossentropy(Y_pred, Y_pred_hat, from_logits=False)
        loss_recon = utils.recon_loss(Y_recon, Y_recon_hat)

        loss = lw_pred * loss_pred + lw_recon * loss_recon

        metric_mse = MAE(Y_recon, Y_recon_hat)
        metric_mae = MSE(Y_recon, Y_recon_hat)
        metric_mre = MRE(Y_recon, Y_recon_hat)

        for metric, loss_type in zip(metrics["loss"], [loss, loss_pred, loss_recon]):
            metric.update_state(loss_type)
        for metric in metrics["pred"]:
            metric.update_state(Y_pred, Y_pred_hat)
        for metric, loss_type in zip(metrics["recon"], [metric_mse, metric_mae, metric_mre]):
            metric.update_state(loss_type)

        Y_pred_full.append(Y_pred)
        Y_pred_hat_full.append(Y_pred_hat)

    for metric in (metrics["loss"] + metrics["pred"] + metrics["recon"]):
        results.append(metric.result().numpy())
        metric.reset_states()

    Y_pred = np.concatenate(Y_pred_full, axis=0)
    Y_pred_hat = np.concatenate(Y_pred_hat_full, axis=0)

    predict_base = (Y_pred_hat >= 0.5).astype('int')

    auroc = results[4]
    auprc = results[5]

    roc_results = roc_curve(Y_pred, Y_pred_hat)
    pr_results = precision_recall_curve(Y_pred, Y_pred_hat)

    print("\n===============[{}]===============".format(stage))
    print("{} {:<10}".format(stage, "MSE"), results[9])
    print("{} {:<10}".format(stage, "MAE"), results[10])
    print("{} {:<10}".format(stage, "MRE4"), results[11])
    print()
    print("{} {:<10}".format(stage, "AUROC"), results[4])
    print("{} {:<10}".format(stage, "AUPRC"), results[5])
    print()

    print("{} Confusion Matrix (Threshold >= 0.5)".format(stage))
    print(confusion_matrix(Y_pred, predict_base))
    print()

    print("{} {:<10}".format(stage, "Accuracy"), results[3])
    print("{} {:<10}".format(stage, "Precision"), results[6])
    print("{} {:<10}".format(stage, "Recall"), results[7])
    print("{} {:<10}".format(stage, "F1"), results[8])
    print()

    optimal_idx = np.argmax(roc_results[1] - roc_results[0])
    optimal_threshold = roc_results[2][optimal_idx]
    predict_optimal = (Y_pred_hat >= optimal_threshold).astype('int')

    print("{} Confusion Matrix (Youden's J statistic)".format(stage))
    print(confusion_matrix(Y_pred, predict_optimal))
    print()

    print("{} {:<10}".format(stage, "Accuracy"), accuracy_score(Y_pred, predict_optimal))
    print("{} {:<10}".format(stage, "Precision"), precision_score(Y_pred, predict_optimal, average='binary'))
    print("{} {:<10}".format(stage, "Recall"), recall_score(Y_pred, predict_optimal, average='binary'))
    print("{} {:<10}".format(stage, "F1"), f1_score(Y_pred, predict_optimal, average='binary'))

    def plot_results(x, y, plot_type='AUROC', save=False, filepath='.', filename="Figure"):
        plt.clf()
        plt.figure(figsize=(10, 10))
        plt.plot(x, y)
        plt.title("{}".format(plot_type))
        if plot_type == 'AUROC':
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
        elif plot_type == 'AUPRC':
            plt.xlabel("Recall")
            plt.ylabel("Precision")

        if save:
            os.makedirs(filepath, exist_ok=True)
            filepath = os.path.join(filepath, filename + '.png')
            plt.savefig(filepath)

    if plot:
        plot_results(roc_results[0], roc_results[1], plot_type="AUROC", save=save, filepath=filepath, filename=filename)
        plot_results(pr_results[1], pr_results[0], plot_type="AUPRC", save=save, filepath=filepath, filename=filename)

    return (auroc, auprc, roc_results, pr_results), results
