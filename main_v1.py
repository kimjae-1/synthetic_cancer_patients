#%%
## >>> [TMP] >>>
import sys, os
sys.path.append('/home/mincheol/git/synthetic_cancer_patients')
os.chdir('/home/mincheol/git/synthetic_cancer_patients')
## <<< [TMP] <<<


#%%
import os
import time
import csv
import pickle

from scipy import stats

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import Mean, BinaryAccuracy, AUC, Precision, Recall
from tensorflow_addons.metrics import F1Score

import utils
import data_pipeline_v1
import models


#%%
def set_gpu(num_gpu=0):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[num_gpu], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[num_gpu], True)


def combine_general_ts(data_train, data_valid, data_test, data_normalize):
    """
    Var time: General Descriptors / Time Series
    Var type: Continuous / Categorical

    Arguments:
        data -- (X_0, X_t, X_info, y)
            X_0 -- shape: (N, (x, m), d_0)
            X_t -- shape: (N, (x, m), t, d_t)
            X_info -- shape: (N, t)
            y -- shape: (N, 1)
        normalize -- (X_0, X_t)
            X -- (min, max, cat_axis)
    Returns:
        data -- (X_0t, X_info, y)
            X_0t -- shape: (N, (x, m), 1+t, d_0t)
            X_info -- shape: (N, 1+t)
            y -- shape: (N, 1)
        normalize -- (X_0t)
            X -- (min, max, cat_axis)
    """

    # data
    X_0_train, X_t_train, X_info_train, y_train = data_train
    X_0_valid, X_t_valid, X_info_valid, y_valid = data_valid
    X_0_test, X_t_test, X_info_test, y_test = data_test

    pad_width_0_train = ((0, 0), (0, 0), (0, X_info_train.shape[-1]), (0, 0))
    pad_width_0_valid = ((0, 0), (0, 0), (0, X_info_valid.shape[-1]), (0, 0))
    pad_width_0_test = ((0, 0), (0, 0), (0, X_info_test.shape[-1]), (0, 0))

    pad_width_t = ((0, 0), (0, 0), (1, 0), (0, 0))
    pad_width_info = ((0, 0), (1, 0))

    X_0_train = np.pad(X_0_train[:, :, np.newaxis, :], pad_width=pad_width_0_train, mode='constant', constant_values=0)
    X_0_valid = np.pad(X_0_valid[:, :, np.newaxis, :], pad_width=pad_width_0_valid, mode='constant', constant_values=0)
    X_0_test = np.pad(X_0_test[:, :, np.newaxis, :], pad_width=pad_width_0_test, mode='constant', constant_values=0)

    X_t_train = np.pad(X_t_train, pad_width=pad_width_t, mode='constant', constant_values=0)
    X_t_valid = np.pad(X_t_valid, pad_width=pad_width_t, mode='constant', constant_values=0)
    X_t_test = np.pad(X_t_test, pad_width=pad_width_t, mode='constant', constant_values=0)

    X_0t_train = np.concatenate((X_0_train, X_t_train), axis=-1)
    X_0t_valid = np.concatenate((X_0_valid, X_t_valid), axis=-1)
    X_0t_test = np.concatenate((X_0_test, X_t_test), axis=-1)

    X_info_train = np.pad(X_info_train, pad_width=pad_width_info, mode='constant', constant_values=0)
    X_info_valid = np.pad(X_info_valid, pad_width=pad_width_info, mode='constant', constant_values=0)
    X_info_test = np.pad(X_info_test, pad_width=pad_width_info, mode='constant', constant_values=0)

    data_train = (X_0t_train, X_info_train, y_train)
    data_valid = (X_0t_valid, X_info_valid, y_valid)
    data_test = (X_0t_test, X_info_test, y_test)

    # normalize
    X_0_train_normalize, X_t_train_normalize = data_normalize
    X_0_train_min, X_0_train_max, X_0_cat_axis = X_0_train_normalize
    X_t_train_min, X_t_train_max, X_t_cat_axis = X_t_train_normalize

    X_train_min = np.concatenate((X_0_train_min, X_t_train_min))
    X_train_max = np.concatenate((X_0_train_max, X_t_train_max))
    X_cat_axis = X_0_cat_axis + X_t_cat_axis

    data_normalize = (X_train_min, X_train_max, X_cat_axis)

    return data_train, data_valid, data_test, data_normalize


def convert_tensor(data_train, data_valid, data_test, batch_size):
    """
    Arguments:
        data -- (X_0t, X_info, y)
            X_0t -- shape: (N, (x, m), t, d_0t)
            X_info -- shape: (N, t)
            y -- shape: (N, 1)
    Returns:
        data -- (inputs, outputs)
            inputs -- {"inputs_t", "inputs_time"}
                inputs_t -- shape: (N, (x, m), t, d_0t)
                inputs_time -- shape: (N, t)
            outputs -- {"pred", "recon"}
                pred -- shape: (N, 1)
                recon -- shape: (N, (x, m), t, d_0t)
    """

    X_0t_train, X_info_train, y_train = data_train
    X_0t_valid, X_info_valid, y_valid = data_valid
    X_0t_test, X_info_test, y_test = data_test

    X_train = {"inputs_t": X_0t_train, "inputs_time": X_info_train}
    X_valid = {"inputs_t": X_0t_valid, "inputs_time": X_info_valid}
    X_test = {"inputs_t": X_0t_test, "inputs_time": X_info_test}

    Y_train = {"pred": y_train, "recon": X_0t_train}
    Y_valid = {"pred": y_valid, "recon": X_0t_valid}
    Y_test = {"pred": y_test, "recon": X_0t_test}

    tensor_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    tensor_train = tensor_train.shuffle(len(y_train)).batch(batch_size)

    tensor_valid = tf.data.Dataset.from_tensor_slices((X_valid, Y_valid))
    tensor_valid = tensor_valid.batch(batch_size)

    tensor_test = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    tensor_test = tensor_test.batch(batch_size)

    return tensor_train, tensor_valid, tensor_test


#%%
NUM_GPU = 0

# Dataset: CEA
dataset = "cea"
dataset_full = "clrc-cea"
encoding = 'CP949'
seed = 42

type_clf = 'full'
mode = 'recon'

epoch = 500
batch_size = 128
learning_rate = 0.0001

lw_pred = 10
lw_recon = 1

num_ref = 128
dim_time = 128
num_heads = 1
dim_attn = dim_time // num_heads
dim_hidden_enc = 256
dim_hidden_dec = 50
dim_ffn = 50
dim_latent = 20
dim_clf = 300


#%%
set_gpu(NUM_GPU)


#%%
# data_filepath = os.path.join('/', 'home', 'mincheol', 'ext', 'hdd1', 'data', 'CONNECT', 'SMC', '202107')
# data_train, data_valid, data_test, data_normalize = data_pipeline_v1.clrc_diag(filepath=data_filepath, encoding=encoding, seed=seed)


#%%
## >>> [TMP] >>>
with open('./data_preprocessed/clrc-cea/data_train.pickle', 'rb') as f:
    data_train = pickle.load(f)
with open('./data_preprocessed/clrc-cea/data_valid.pickle', 'rb') as f:
    data_valid = pickle.load(f)
with open('./data_preprocessed/clrc-cea/data_test.pickle', 'rb') as f:
    data_test = pickle.load(f)
with open('./data_preprocessed/clrc-cea/data_normalize.pickle', 'rb') as f:
    data_normalize = pickle.load(f)
## <<< [TMP] <<<


#%%
data_train, data_valid, data_test, data_normalize = combine_general_ts(data_train, data_valid, data_test, data_normalize)
tensor_train, tensor_valid, tensor_test = convert_tensor(data_train, data_valid, data_test, batch_size)


#%%
model_name = 'mTAND-full'
with tf.device('/device:GPU:' + str(NUM_GPU)):
    model = models.mTAND_clf(num_ref, dim_time, dim_attn, num_heads, dim_hidden_enc, dim_ffn, dim_latent, dim_clf,
                             dim_hidden_dec=dim_hidden_dec, type_clf=type_clf, name=model_name)

    optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    metrics = {"loss":
                    [Mean(name="loss"),
                     Mean(name="loss_pred"),
                     Mean(name="loss_recon"),
                     Mean(name="loss_nll"),
                     Mean(name="loss_kl"),
                    ],
                "pred":
                   [BinaryAccuracy(threshold=0.5, name="Accuracy"),
                    AUC(curve='ROC', name="AUROC"),
                    AUC(curve='PR', name="AUPRC"),
                    Precision(thresholds=0.5, name="Precision"),
                    Recall(thresholds=0.5, name="Recall"),
                    F1Score(num_classes=1, average='macro', threshold=0.5, name="F1_score"),
                    ],
               "recon":
                   [Mean(name="MSE"),
                    Mean(name="MAE"),
                    Mean(name="MRE"),
                    ],
               }


#%%
filename = (dataset + "_" + model_name + "_" + mode +
            "_e" + str(epoch) + "_b" + str(batch_size) + "_lr" + str(learning_rate) +
            "_lwp" + str(lw_pred) + "_lwr" + str(lw_recon) +
            "_ref" + str(num_ref) + "_t" + str(dim_time) + "_h" + str(num_heads) +
            "_hid-enc" + str(dim_hidden_enc) + "_hid-dec" + str(dim_hidden_dec) +
            "_ffn" + str(dim_ffn) + "_z" + str(dim_latent) + "_clf" + str(dim_clf))

# callbacks
os.makedirs(os.path.join('.', 'results', dataset_full, 'model_tuning'), exist_ok=True)
cp_filepath = os.path.join('.', 'results', dataset_full, 'model_tuning', filename + '.h5')

early_stop = 0
if mode == 'pred':
    best_val = 0
    early_stop_patience = 50
else:
    best_val = np.inf
    early_stop_patience = 100

history_train = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
history_valid = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]


#%%
for ep in range(1, epoch + 1):

    # Train
    time_start = time.time()
    for step, (X, Y) in enumerate(tensor_train):
        Y_pred = Y["pred"]
        Y_recon = Y["recon"]

        with tf.GradientTape() as tape:
            Y_hat = model(X)
            Y_pred_hat = Y_hat["pred"]
            Y_recon_hat = Y_hat["recon"]

            loss_pred = binary_crossentropy(Y_pred, Y_pred_hat, from_logits=False)
            loss_recon, loss_nll, loss_kl = utils.recon_loss(Y_recon, Y_recon_hat, ep, wait_kl=10)

            loss = lw_pred * loss_pred + lw_recon * loss_recon
            loss_mean = tf.reduce_mean(loss)

        gradients = tape.gradient(loss_mean, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        metric_mse = utils.MAE(Y_recon, Y_recon_hat)
        metric_mae = utils.MSE(Y_recon, Y_recon_hat)
        metric_mre = utils.MRE(Y_recon, Y_recon_hat)

        for metric, loss_type in zip(metrics["loss"], [loss, loss_pred, loss_recon, loss_nll, loss_kl]):
            metric.update_state(loss_type)
        for metric in metrics["pred"]:
            metric.update_state(Y_pred, Y_pred_hat)
        for metric, loss_type in zip(metrics["recon"], [metric_mse, metric_mae, metric_mre]):
            metric.update_state(loss_type)

    time_end = time.time()

    ## results
    print()
    print("Epoch {}/{}".format(ep, epoch))
    print("[Train]")
    print("Time {:.4f} | ".format(time_end - time_start) +
          "".join(["{}: {:4.4f} | ".format(metric.name, metric.result().numpy()) for metric in metrics["loss"]]) +
          "".join(["{}: {:.4f} | ".format(metric.name, metric.result().numpy()) for metric in metrics["pred"]]) +
          "".join(["{}: {:4.4f} | ".format(metric.name, metric.result().numpy()) for  metric in metrics["recon"]])
          )

    ## history
    for i, metric in enumerate(metrics["loss"]):
        history_train[i].append(metric.result().numpy())
        metric.reset_states()
    for i, metric in enumerate(metrics["pred"]):
        history_train[i + len(metrics["loss"])].append(metric.result().numpy())
        metric.reset_states()
    for i, metric in enumerate(metrics["recon"]):
        history_train[i + len(metrics["loss"]) + len(metrics["pred"])].append(metric.result().numpy())
        metric.reset_states()

    # Valid
    time_start = time.time()
    for step, (X, Y) in enumerate(tensor_valid):
        Y_pred = Y["pred"]
        Y_recon = Y["recon"]

        Y_hat = model(X)
        Y_pred_hat = Y_hat["pred"]
        Y_recon_hat = Y_hat["recon"]

        loss_pred = binary_crossentropy(Y_pred, Y_pred_hat, from_logits=False)
        loss_recon, loss_nll, loss_kl = utils.recon_loss(Y_recon, Y_recon_hat, ep, wait_kl=10)

        loss = lw_pred * loss_pred + lw_recon * loss_recon

        metric_mse = utils.MAE(Y_recon, Y_recon_hat)
        metric_mae = utils.MSE(Y_recon, Y_recon_hat)
        metric_mre = utils.MRE(Y_recon, Y_recon_hat)

        for metric, loss_type in zip(metrics["loss"], [loss, loss_pred, loss_recon, loss_nll, loss_kl]):
            metric.update_state(loss_type)
        for metric in metrics["pred"]:
            metric.update_state(Y_pred, Y_pred_hat)
        for metric, loss_type in zip(metrics["recon"], [metric_mse, metric_mae, metric_mre]):
            metric.update_state(loss_type)

    time_end = time.time()

    if mode == 'pred':
        if metrics["pred"][1].result().numpy() > best_val:
            model.save_weights(cp_filepath)

            best_val = metrics["pred"][0].result().numpy()
            early_stop = 0
    else:
        if metrics["recon"][1].result().numpy() < best_val:
            model.save_weights(cp_filepath)

            best_val = metrics["recon"][0].result().numpy()
            early_stop = 0

    ## results
    if mode == 'pred':
        print("[Valid] - best val AUROC: {:.4f} - early stop count {}".format(best_val, early_stop))
    else:
        print("[Valid] - best val MSE: {:.4f} - early stop count {}".format(best_val, early_stop))
    print("Time {:.4f} | ".format(time_end - time_start) +
          "".join(["{}: {:4.4f} | ".format(metric.name, metric.result().numpy()) for metric in metrics["loss"]]) +
          "".join(["{}: {:.4f} | ".format(metric.name, metric.result().numpy()) for metric in metrics["pred"]]) +
          "".join(["{}: {:4.4f} | ".format(metric.name, metric.result().numpy()) for metric in metrics["recon"]])
          )

    ## history
    for i, metric in enumerate(metrics["loss"]):
        history_valid[i].append(metric.result().numpy())
        metric.reset_states()
    for i, metric in enumerate(metrics["pred"]):
        history_valid[i + len(metrics["loss"])].append(metric.result().numpy())
        metric.reset_states()
    for i, metric in enumerate(metrics["recon"]):
        history_valid[i + len(metrics["loss"]) + len(metrics["pred"])].append(metric.result().numpy())
        metric.reset_states()

    ## early stop
    early_stop += 1
    if early_stop > early_stop_patience:
        model.load_weights(cp_filepath)
        break


#%%
plt_filepath = os.path.join('.', 'results_learning', dataset_full, 'model_tuning')
utils.plot_learning(history_train, history_valid, figsize=(20, 20), save=True, filepath=plt_filepath, filename=filename)


#%%
model.load_weights(cp_filepath)

_, results_train = utils.print_results(model, tensor_train, metrics, (lw_pred, lw_recon), stage="Train")
_, results_valid = utils.print_results(model, tensor_valid, metrics, (lw_pred, lw_recon), stage="Valid")
_, results_test = utils.print_results(model, tensor_test, metrics, (lw_pred, lw_recon), stage="Test")


#%%
## Generation with reference
def gen_outputs_ref(model, inputs, ref_time):
    latent = model.enc(inputs)

    q_z_mean, q_z_logvar = latent[:, :, :model.dim_latent], latent[:, :, model.dim_latent:]
    epsilon = tf.random.normal(tf.shape(q_z_mean), dtype=tf.float32)
    z = q_z_mean + tf.math.exp(0.5 * q_z_logvar) * epsilon

    outputs_pred_y = model.clf(z)

    inputs_dec = {"latent": z, "outputs_time": ref_time}
    outputs_pred_x = model.dec(inputs_dec)

    outputs_recon = {"outputs_recon": outputs_pred_x, "q_z_mean": q_z_mean, "q_z_logvar": q_z_logvar}
    outputs = {"pred": outputs_pred_y, "recon": outputs_recon}

    return outputs


def gen_recon(model, tensor, ref_time):
    Y_preds, Y_recons, Y_recons_time = [], [], []
    Y_preds_hat, Y_recons_hat, q_z_means, q_z_logvars = [], [], [], []
    for step, (X, Y) in enumerate(tensor):
        Y_pred = Y["pred"]
        Y_recon = Y["recon"]
        Y_recon_time = X["inputs_time"]

        Y_hat = gen_outputs_ref(model, X, ref_time)
        Y_pred_hat = Y_hat["pred"]
        Y_recon_hat = Y_hat["recon"]["outputs_recon"]
        q_z_mean = Y_hat["recon"]["q_z_mean"]
        q_z_logvar = Y_hat["recon"]["q_z_logvar"]

        Y_preds.append(Y_pred.numpy())
        Y_recons.append(Y_recon.numpy())
        Y_recons_time.append(Y_recon_time.numpy())

        Y_preds_hat.append(Y_pred_hat.numpy())
        Y_recons_hat.append(Y_recon_hat.numpy())
        q_z_means.append(q_z_mean.numpy())
        q_z_logvars.append(q_z_logvar.numpy())

    Y_pred = np.concatenate(Y_preds, axis=0)
    Y_recon = np.concatenate(Y_recons, axis=0)
    Y_recon_time = np.concatenate(Y_recons_time, axis=0)

    Y_pred_hat = np.concatenate(Y_preds_hat, axis=0)
    Y_recon_hat = np.concatenate(Y_recons_hat, axis=0)
    q_z_mean = np.concatenate(q_z_means, axis=0)
    q_z_logvar = np.concatenate(q_z_logvars, axis=0)

    return Y_pred, Y_recon, Y_recon_time, Y_pred_hat, Y_recon_hat, q_z_mean, q_z_logvar


ref_time = np.linspace(0.0, 5.0, 12 * 5 * 2)
ref_time_tf = tf.convert_to_tensor(np.expand_dims(ref_time, axis=0))

gen_recon_train = gen_recon(model, tensor_train, ref_time_tf)
gen_recon_valid = gen_recon(model, tensor_valid, ref_time_tf)
gen_recon_test = gen_recon(model, tensor_test, ref_time_tf)

X_train_orig = gen_recon_train[1][:, 0, :, :].copy()
X_valid_orig = gen_recon_valid[1][:, 0, :, :].copy()
X_test_orig = gen_recon_test[1][:, 0, :, :].copy()

X_train_orig[gen_recon_train[1][:, 1, :, :] == 0] = np.nan
X_valid_orig[gen_recon_valid[1][:, 1, :, :] == 0] = np.nan
X_test_orig[gen_recon_test[1][:, 1, :, :] == 0] = np.nan

X_train_orig_time = gen_recon_train[2]
X_valid_orig_time = gen_recon_valid[2]
X_test_orig_time = gen_recon_test[2]

X_train_hat = gen_recon_train[4]
X_valid_hat = gen_recon_valid[4]
X_test_hat = gen_recon_test[4]

y_train_orig = gen_recon_train[0]
y_valid_orig = gen_recon_valid[0]
y_test_orig = gen_recon_test[0]

y_train_hat = gen_recon_train[3]
y_valid_hat = gen_recon_valid[3]
y_test_hat = gen_recon_train[3]


#%%
recon_train_data = (X_train_orig, X_train_hat, X_train_orig_time, ref_time, y_train_orig, y_train_hat)
recon_valid_data = (X_valid_orig, X_valid_hat, X_valid_orig_time, ref_time, y_valid_orig, y_valid_hat)
recon_test_data = (X_test_orig, X_test_hat, X_test_orig_time, ref_time, y_test_orig, y_test_hat)

sample_idx = 0

plt_recon_filepath = os.path.join('.', 'results', dataset_full, 'model_tuning', filename)
utils.plot_recon(recon_train_data, sample_idx, 'train', figsize=(10, 50), ylim=True, plot=False, save=True, filepath=plt_recon_filepath)


#%%
with open(os.path.join(plt_recon_filepath, 'recon_train_data.pickle'), 'wb') as f:
    pickle.dump(recon_train_data, f)
with open(os.path.join(plt_recon_filepath, 'recon_valid_data.pickle'), 'wb') as f:
    pickle.dump(recon_valid_data, f)
with open(os.path.join(plt_recon_filepath, 'recon_test_data.pickle'), 'wb') as f:
    pickle.dump(recon_test_data, f)


#%%
with open(os.path.join(plt_recon_filepath, 'recon_train_data.pickle'), 'rb') as f:
    recon_train_data = pickle.load(f)
with open(os.path.join(plt_recon_filepath, 'recon_valid_data.pickle'), 'rb') as f:
    recon_valid_data = pickle.load(f)
with open(os.path.join(plt_recon_filepath, 'recon_test_data.pickle'), 'rb') as f:
    recon_test_data = pickle.load(f)


#%%
## Generation synthetic
def gen_outputs_sampling(model, gen_num, ref_time):
    p_z_mean = tf.zeros((gen_num, model.num_ref, model.dim_latent), dtype=tf.float32)
    p_z_logvar = tf.zeros((gen_num, model.num_ref, model.dim_latent), dtype=tf.float32)

    epsilon = tf.random.normal(tf.shape(p_z_mean), dtype=tf.float32)
    z = p_z_mean + tf.math.exp(0.5 * p_z_logvar) * epsilon

    outputs_pred_y = model.clf(z)

    inputs_dec = {"latent": z, "outputs_time": ref_time}
    outputs_pred_x = model.dec(inputs_dec)

    outputs_recon = {"outputs_recon": outputs_pred_x, "q_z_mean": p_z_mean, "q_z_logvar": p_z_logvar}
    outputs = {"pred": outputs_pred_y, "recon": outputs_recon}

    return outputs


def gen_synthetic(model, gen_num, gen_batch_size, ref_time):
    Y_preds_hat, Y_recons_hat, q_z_means, q_z_logvars = [], [], [], []
    for _ in range(0, gen_num, gen_batch_size):
        Y_hat = gen_outputs_sampling(model, gen_batch_size, ref_time)

        Y_pred_hat = Y_hat["pred"]
        Y_recon_hat = Y_hat["recon"]["outputs_recon"]
        q_z_mean = Y_hat["recon"]["q_z_mean"]
        q_z_logvar = Y_hat["recon"]["q_z_logvar"]

        Y_preds_hat.append(Y_pred_hat.numpy())
        Y_recons_hat.append(Y_recon_hat.numpy())
        q_z_means.append(q_z_mean.numpy())
        q_z_logvars.append(q_z_logvar.numpy())

    Y_pred_hat = np.concatenate(Y_preds_hat, axis=0)
    Y_recon_hat = np.concatenate(Y_recons_hat, axis=0)
    q_z_mean = np.concatenate(q_z_means, axis=0)
    q_z_logvar = np.concatenate(q_z_logvars, axis=0)

    return Y_pred_hat, Y_recon_hat, q_z_mean, q_z_logvar


gen_num = 3000
gen_batch_size = 100

ref_time = np.linspace(0.0, 5.0, 12 * 5 * 2)
ref_time_tf = tf.convert_to_tensor(np.expand_dims(ref_time, axis=0))

gen_tilde = gen_synthetic(model, gen_num, gen_batch_size, ref_time_tf)

X_tilde = gen_tilde[1]
y_tilde = gen_tilde[0]


#%%
gen_synthetic_data = (X_tilde, ref_time, y_tilde)

sample_idx = 0

plt_gen_filepath = os.path.join('.', 'results', dataset_full, 'model_tuning', filename)
utils.plot_gen(gen_synthetic_data, sample_idx, figsize=(10, 50), ylim=True, plot=False, save=True, filepath=plt_gen_filepath)


#%%
with open(os.path.join(plt_gen_filepath, 'gen_synthetic_data.pickle'), 'wb') as f:
    pickle.dump(gen_synthetic_data, f)


#%%
with open(os.path.join(plt_gen_filepath, 'gen_synthetic_data.pickle'), 'rb') as f:
    gen_synthetic_data = pickle.load(f)


#%%
# Bootstrapping
# https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/
data_idx = np.arange(len(data_test[0]))
sample_size = len(data_test[0])

metric_AUROC = AUC(curve='ROC', name="AUROC")
metric_AUPRC = AUC(curve='PR', name="AUPRC")

sample_AUROC = []
sample_AUPRC = []

sampling = 1000
for i in range(sampling):
    random_idx = np.random.choice(data_idx, size=sample_size, replace=True)
    X_test = {"inputs_t": data_test[0][random_idx, :, :, :],
              "inputs_time": data_test[1][random_idx, :]}
    Y_test = {"pred": data_test[2][random_idx, :],
              "recon": data_test[0][random_idx, :, :, :]}

    tensor_test_bs = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    tensor_test_bs = tensor_test_bs.batch(batch_size)

    for (X, Y) in tensor_test:
        Y_hat = model(X)

        metric_AUROC.update_state(Y["pred"], Y_hat["pred"])
        metric_AUPRC.update_state(Y["pred"], Y_hat["pred"])

    sample_AUROC.append(metric_AUROC.result().numpy())
    sample_AUPRC.append(metric_AUPRC.result().numpy())

    metric_AUROC.reset_states()
    metric_AUPRC.reset_states()

    if i % 100 == 0:
        print("iterations:", i)

sample_AUROC = np.array(sample_AUROC)
sample_AUPRC = np.array(sample_AUPRC)

AUROC_mean = sample_AUROC.mean()
AUPRC_mean = sample_AUPRC.mean()

AUROC_std = sample_AUROC.std(ddof=1)
AUPRC_std = sample_AUPRC.std(ddof=1)

df = sampling - 1
alpha = 0.05
t_value = stats.t.ppf(1 - alpha/2, df)

AUROC_ci = t_value * AUROC_std
AUPRC_ci = t_value * AUPRC_std


#%%
results_param = [model_name, type_clf, mode, epoch, batch_size, learning_rate, lw_pred, lw_recon, num_ref, dim_time,
                 num_heads, dim_attn, dim_hidden_enc, dim_hidden_dec, dim_ffn, dim_latent, dim_clf]
results_stats = [AUROC_mean, AUROC_std, AUROC_ci, AUPRC_mean, AUPRC_std, AUPRC_ci]
csv_write = (results_param + results_train + results_valid + results_test + results_stats)

results_filepath = os.path.join('.', 'results', dataset_full, 'model_tuning', 'mTAND-Full_best.csv')
with open(results_filepath, 'a', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(csv_write)


#%%
os.makedirs(os.path.join('.', 'results', dataset_full, 'model_tuning'), exist_ok=True)
results_filepath = os.path.join('.', 'results', dataset_full, 'model_tuning', 'mTAND-Full_best.csv')

with open(results_filepath, 'a') as f:
    writer = csv.writer(f)
    head = ['model_name',
            'model_type',
            'mode',
            'epoch',
            'batch_size',
            'learning_rate',
            'lw_pred',
            'lw_recon',
            'num_ref',
            'dim_time',
            'num_heads',
            'dim_attn',
            'dim_hidden_enc',
            'dim_hidden_dec',
            'dim_ffn',
            'dim_latent',
            'dim_clf',
            'train_Loss',
            'train_Loss_pred',
            'train_Loss_recon',
            'train_Accuracy',
            'train_AUROC',
            'train_AUPRC',
            'train_Precision',
            'train_Recall',
            'train_F1',
            'valid_Loss',
            'valid_Loss_pred',
            'valid_Loss_recon',
            'valid_Accuracy',
            'valid_AUROC',
            'valid_AUPRC',
            'valid_Precision',
            'valid_Recall',
            'valid_F1',
            'test_Loss',
            'test_Loss_pred',
            'test_Loss_recon',
            'test_Accuracy',
            'test_AUROC',
            'test_AUPRC',
            'test_Precision',
            'test_Recall',
            'test_F1',
            'test_AUROC_mean',
            'test_AUROC_std',
            'test_AUROC_ci',
            'test_AUPRC_mean',
            'test_AUPRC_std',
            'test_AUPRC_ci']
    writer.writerow(head)
