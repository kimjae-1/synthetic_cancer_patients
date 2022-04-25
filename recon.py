#https://github.com/reml-lab/mTAN

import tensorflow as tf
import numpy as np

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

def sampling_data(data,gen_num):
    list = []
    for i in range(gen_num):
      list.append(data)
    list = tf.constant(list)
    return list

def gen_outputs_sampling(model, gen_num, ref_time, mean, logvar):
    p_z_mean = sampling_data(mean,gen_num)
    p_z_logvar = sampling_data(logvar, gen_num)

    epsilon = tf.random.normal(tf.shape(p_z_mean), dtype=tf.float32)
    z = p_z_mean + tf.math.exp(0.5 * p_z_logvar) * epsilon

    outputs_pred_y = model.clf(z)

    inputs_dec = {"latent": z, "outputs_time": ref_time}
    outputs_pred_x = model.dec(inputs_dec)

    outputs_recon = {"outputs_recon": outputs_pred_x, "q_z_mean": p_z_mean, "q_z_logvar": p_z_logvar}
    outputs = {"pred": outputs_pred_y, "recon": outputs_recon}

    return outputs


def gen_synthetic(model, gen_num, gen_batch_size, ref_time, mean, logvar):
    Y_preds_hat, Y_recons_hat, q_z_means, q_z_logvars = [], [], [], []
    for _ in range(0, gen_num, gen_batch_size):
        Y_hat = gen_outputs_sampling(model, gen_batch_size, ref_time, mean, logvar)

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