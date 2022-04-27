import sys, os
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
import recon

#%% 변수 설정
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

#%% GPU 설정
NUM_GPU = 0
utils.set_gpu(NUM_GPU)

#%% filepath 및 데이터 전처리
data_filepath = os.path.join(os.getcwd(),'data')
data_train, data_valid, data_test, data_normalize = data_pipeline_v1.clrc_diag(filepath=data_filepath, encoding=encoding, seed=seed)

# pickle
# path = os.getcwd()+'/pickle'
# os.makedirs(path, exist_ok=True)
# 
# # 저장
# with open(os.path.join(os.getcwd(),'pickle', 'data_train.pickle'), 'wb') as f:
#     pickle.dump(data_train, f)
# with open(os.path.join(os.getcwd(),'pickle', 'data_valid.pickle'), 'wb') as f:
#     pickle.dump(data_valid, f)
# with open(os.path.join(os.getcwd(),'pickle', 'data_test.pickle'), 'wb') as f:
#     pickle.dump(data_test, f)
# with open(os.path.join(os.getcwd(),'pickle', 'data_normalize.pickle'), 'wb') as f:
#     pickle.dump(data_normalize, f)

# # 불러오기
# with open(os.path.join(os.getcwd(),'pickle', 'data_train.pickle'), 'rb') as f:
#     data_train = pickle.load(f)
# with open(os.path.join(os.getcwd(),'pickle', 'data_valid.pickle'), 'rb') as f:
#     data_valid = pickle.load(f)
# with open(os.path.join(os.getcwd(),'pickle', 'data_test.pickle'), 'rb') as f:
#     data_test = pickle.load(f)
# with open(os.path.join(os.getcwd(),'pickle', 'data_normalize.pickle'), 'rb') as f:
#     data_normalize = pickle.load(f)

data_train, data_valid, data_test, data_normalize = data_pipeline_v1.combine_general_ts(data_train, data_valid, data_test, data_normalize)

tensor_train, tensor_valid, tensor_test = data_pipeline_v1.convert_tensor(data_train, data_valid, data_test, batch_size)

#%% model
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

filename = (dataset + "_" + model_name + "_" +
            "_ref" + str(num_ref) + "_t" + str(dim_time) + "_h" + str(num_heads) +
            "_hid-enc" + str(dim_hidden_enc) + "_hid-dec" + str(dim_hidden_dec) +
            "_ffn" + str(dim_ffn) + "_z" + str(dim_latent) + "_clf" + str(dim_clf))

os.makedirs(os.path.join(os.getcwd(), 'results', dataset_full, 'model_tuning'), exist_ok=True)
cp_filepath = os.path.join(os.getcwd(), 'results', dataset_full, 'model_tuning', filename + '.h5')

early_stop = 0
if mode == 'pred':
    best_val = 0
    early_stop_patience = 50
else:
    best_val = np.inf
    early_stop_patience = 100

history_train = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
history_valid = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]

#%% training
for ep in range(1):

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

#%% 결과 plot
plt_filepath = os.path.join(os.getcwd(), 'results_learning', dataset_full, 'model_tuning')
utils.plot_learning(history_train, history_valid, figsize=(20, 20), save=True, filepath=plt_filepath, filename=filename)


#%% ref_time에 대한 data reconstruction
model.load_weights(cp_filepath)

_, results_train = utils.print_results(model, tensor_train, metrics, (lw_pred, lw_recon), stage="Train")
_, results_valid = utils.print_results(model, tensor_valid, metrics, (lw_pred, lw_recon), stage="Valid")
_, results_test = utils.print_results(model, tensor_test, metrics, (lw_pred, lw_recon), stage="Test")

ref_time = np.linspace(0.0, 5.0, 12 * 5 * 2)
ref_time_tf = tf.convert_to_tensor(np.expand_dims(ref_time, axis=0))

gen_recon_train = recon.gen_recon(model, tensor_train, ref_time_tf)
gen_recon_valid = recon.gen_recon(model, tensor_valid, ref_time_tf)
gen_recon_test = recon.gen_recon(model, tensor_test, ref_time_tf)

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

q_z_mean = gen_recon_train[5]
q_z_logvar = gen_recon_train[6]

#%% recon data 결과 plot(일부 값 확인하는 용도)
recon_train_data = (X_train_orig, X_train_hat, X_train_orig_time, ref_time, y_train_orig, y_train_hat)
recon_valid_data = (X_valid_orig, X_valid_hat, X_valid_orig_time, ref_time, y_valid_orig, y_valid_hat)
recon_test_data = (X_test_orig, X_test_hat, X_test_orig_time, ref_time, y_test_orig, y_test_hat)

sample_idx = 0

plt_recon_filepath = os.path.join('.', 'results', dataset_full, 'model_tuning', filename)
utils.plot_recon(recon_train_data, sample_idx, 'train', figsize=(10, 50), ylim=True, plot=False, save=True, filepath=plt_recon_filepath)

#%% recon_data에 대해 denormalize(일부 값 확인하는 용도)
real = recon_train_data[0][sample_idx, 0, :]
fake = recon_train_data[1][sample_idx, 0, :]

real_cont = np.array(list(real[:1]) + list(real[6:]))
fake_cont = np.array(list(fake[:1]) + list(fake[6:]))

X_train_min, X_train_max, _ = data_normalize

real_cont_denorm = real_cont * (X_train_max - X_train_min) + X_train_min
fake_cont_denorm = fake_cont * (X_train_max - X_train_min) + X_train_min


#%% recon_data, pickle 모듈 사용하여 저장
with open(os.path.join(plt_recon_filepath, 'recon_train_data.pickle'), 'wb') as f:
    pickle.dump(recon_train_data, f)
with open(os.path.join(plt_recon_filepath, 'recon_valid_data.pickle'), 'wb') as f:
    pickle.dump(recon_valid_data, f)
with open(os.path.join(plt_recon_filepath, 'recon_test_data.pickle'), 'wb') as f:
    pickle.dump(recon_test_data, f)

#%% recon_data, pickle모듈 사용하여 불러오기
with open(os.path.join(plt_recon_filepath, 'recon_train_data.pickle'), 'rb') as f:
    recon_train_data = pickle.load(f)
with open(os.path.join(plt_recon_filepath, 'recon_valid_data.pickle'), 'rb') as f:
    recon_valid_data = pickle.load(f)
with open(os.path.join(plt_recon_filepath, 'recon_test_data.pickle'), 'rb') as f:
    recon_test_data = pickle.load(f)


#%% 학습한 latent의 대표값
q_z_mean_mean = q_z_mean.mean(axis = 0)
q_z_mean_std = q_z_mean.std(axis = 0)

mean = []
for i in range(q_z_mean_mean.shape[0]):
  for j in range(q_z_mean_mean.shape[1]):
    mu = q_z_mean_mean[i,j]
    sigma = q_z_mean_std[i,j]
    mean_sample = tf.random.normal([1],mu,sigma, tf.float32)
    mean.append(mean_sample)

mean = np.array(mean).reshape(q_z_mean_mean.shape[0],q_z_mean_mean.shape[1])

q_z_logvar_mean = q_z_logvar.mean(axis = 0)
q_z_logvar_std = q_z_logvar.std(axis = 0)

logvar = []
for i in range(q_z_logvar_mean.shape[0]):
  for j in range(q_z_logvar_mean.shape[1]):
    mu = q_z_logvar_mean[i,j]
    sigma = q_z_logvar_std[i,j]
    std_sample = tf.random.normal([1],mu,sigma, tf.float32)
    logvar.append(std_sample)

logvar = np.array(logvar).reshape(q_z_logvar_mean.shape[0],q_z_logvar_mean.shape[1])

#%% 데이터 합성
gen_num = 3000
gen_batch_size = 100

ref_time = np.linspace(0.0, 5.0, 12 * 5 * 2)
ref_time_tf = tf.convert_to_tensor(np.expand_dims(ref_time, axis=0))

gen_tilde = recon.gen_synthetic(model, gen_num, gen_batch_size, ref_time_tf, mean, logvar)

X_tilde = gen_tilde[1]
y_tilde = gen_tilde[0]

gen_synthetic_data = (X_tilde, ref_time, y_tilde)

#%% 합성데이터 확인(일부 값)
sample_idx = 0

plt_gen_filepath = os.path.join(os.getcwd(), 'results', dataset_full, 'model_tuning', filename)
utils.plot_gen(gen_synthetic_data, sample_idx, figsize=(10, 50), ylim=True, plot=False, save=True, filepath=plt_gen_filepath)

#%% 합성 데이터 denormalize(일부 값 확인)
fake2 = gen_synthetic_data[0][sample_idx, 0, :]

fake2_cont = np.array(list(fake2[:1]) + list(fake2[6:]))

X_train_min, X_train_max, _ = data_normalize

fake2_cont_denorm = fake_cont * (X_train_max - X_train_min) + X_train_min


#%% 합성 데이터, pickle 모듈 사용하여 저장
with open(os.path.join(plt_gen_filepath, 'gen_synthetic_data.pickle'), 'wb') as f:
    pickle.dump(gen_synthetic_data, f)


#%% 합성 데이터, pickle 모듈 사용하여 불러오기
with open(os.path.join(plt_gen_filepath, 'gen_synthetic_data.pickle'), 'rb') as f:
    gen_synthetic_data = pickle.load(f)

