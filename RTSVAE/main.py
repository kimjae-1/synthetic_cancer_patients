# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os,sys,time
import random
from sys import maxsize
import pickle
import numpy as np
import torch
from utils.general import init_logger
from aegan import AeGAN
#sys.path.append('./general/')

'''
import pickle
import collections
import logging
import math
import datetime
import torch.nn as nn
'''
# ===-----------------------------------------------------------------------===
# Argument parsing
# ===-----------------------------------------------------------------------===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, dest="dataset", help=".pkl file to use")     
parser.add_argument("--force", default="", dest="force", help="schedule")     
parser.add_argument("--devi", default="0", dest="devi", help="gpu")
parser.add_argument("--epochs", default=800, dest="epochs", type=int,
                    help="Number of full passes through training set for autoencoder")
parser.add_argument("--log-dir", default="./log", dest="log_dir",
                    help="Directory where to write logs / serialized models")
parser.add_argument("--task-name", default=time.strftime("%Y-%m-%d-%H-%M-%S"), dest="task_name",
                    help="Name for this task, use a comprehensive one")
parser.add_argument("--python-seed", dest="python_seed", type=int, default=random.randrange(maxsize),
                    help="Random seed of Python and NumPy")
parser.add_argument("--fix-ae", dest="fix_ae", default=None, help="Test mode")
parser.add_argument("--ae-batch-size", default=128, dest="ae_batch_size", type=int,
                    help="Minibatch size for autoencoder")
parser.add_argument("--gen-num", default=128, dest="gen_num", type=int,
                    help="number of synthetic data")
parser.add_argument("--gen-batch-size", default=128, dest="gen_batch_size", type=int,
                    help="Minibatch size for new sample")

parser.add_argument("--embed-dim", default=512, dest="embed_dim", type=int, help="dim of hidden state")
parser.add_argument("--hidden-dim", default=128, dest="hidden_dim", type=int, help="dim of GRU hidden state")
parser.add_argument("--layers", default=3, dest="layers", type=int, help="layers")
parser.add_argument("--ae-lr", default=1e-3, dest="ae_lr", type=float, help="autoencoder learning rate")
parser.add_argument("--weight-decay", default=0, dest="weight_decay", type=float, help="weight decay")
parser.add_argument("--dropout", default=0.0, dest="dropout", type=float,
                    help="Amount of dropout(not keep rate, but drop rate) to apply to embeddings part of graph")
parser.add_argument("--save", dest="save", default=None, help="save")


options = parser.parse_args()

task_name = options.task_name
root_dir = f"{options.log_dir}/{task_name}"
os.makedirs(root_dir, exist_ok=True)
print(root_dir)

devices = [int(x) for x in options.devi]
device = torch.device(f"cuda:{devices[0]}")  

# ===-----------------------------------------------------------------------===
# Set up logging
# ===-----------------------------------------------------------------------===
logger = init_logger(root_dir)

# ===-----------------------------------------------------------------------===
# Log some stuff about this run
# ===-----------------------------------------------------------------------===
logger.info(' '.join(sys.argv))
logger.info('')
print('Parameters:')
logger.info(options)

random.seed(options.python_seed)
np.random.seed(options.python_seed % (2 ** 32 - 1))
logger.info(f'Python random seed: {options.python_seed}')
print(os.linesep)

# ===-----------------------------------------------------------------------===
# Read in dataset
# ===-----------------------------------------------------------------------===
dataset = pickle.load(open(options.dataset, "rb"))
train_set = dataset["train_set"]
val_set = dataset["val_set"]

dynamic_processor = dataset["dynamic_processor"]
static_processor = dataset["static_processor"]
train_set.set_input("dyn", "mask", "sta", "times", "lag", "seq_len","priv", "nex", "label")
val_set.set_input("dyn", "mask", "sta", "times", "lag", "seq_len","priv", "nex", "label")

# ===-----------------------------------------------------------------------===
# Build model and trainer
# ===-----------------------------------------------------------------------===

params = vars(options)
params["static_processor"] = static_processor
params["dynamic_processor"] = dynamic_processor
params["root_dir"] = root_dir
params["logger"] = logger
params["device"] = device

syn = AeGAN((static_processor, dynamic_processor), params)

if options.fix_ae is not None:
    print('Load trained VAE...')
    syn.load_ae(options.fix_ae)
else:
    print('Training new VAE...')

    val_result = [0]
    best_val_loss = 1e15
    for i in range(options.epochs):
        train_result = syn.train_ae(train_set, i)
        val_result = syn.eval_ae(val_set)
        if i % 10 == 0:
            print(f'Epoch {i+1} : Train Loss = {train_result[0]:.6f}, Val Loss = {val_result[0]:.6f}')
        #print(f'{train_result[1]:.3f}') 70 sec per epoch in a 1070 perhaps
        #if best_val_loss >= val_result[0]:
        #    best_val_loss = val_result[0]
        #    best_model = syn.ae.state_dict()
        #    best_results = val_result
            #tot_loss, time.time() - t1, con_loss, dis_loss, miss_loss1, miss_loss2, KLD, tot
            logger.info(f'Epoch {i+1} (train) tot_loss: {train_result[0]}, time: {train_result[1]}, con_loss: {train_result[2]}, \
              dis_loss: {train_result[3]}, miss_loss1: {train_result[4]}, miss_loss2: {train_result[5]}, KLD: {train_result[6]}')
            logger.info(f'Epoch {i+1} (val) tot_loss: {val_result[0]}, time: {val_result[1]}, con_loss: {val_result[2]}, \
              dis_loss: {val_result[3]}, miss_loss1: {val_result[4]}, miss_loss2: {val_result[5]}, KLD: {val_result[6]}')

        if (options.save and i % 100 == 99):
            torch.save(syn.ae.state_dict(), f'./data/last_model_{options.task_name}_{i}.dat')

            last_model = syn.ae.state_dict()
            #torch.save(best_model, f'./data/best_model_{options.task_name}.dat')
            #if options.save!=None:
            #    torch.save(syn.ae.state_dict(), f'./data/last_model_{options.task_name}.dat')
            #print('VAE Training done.', os.linesep)

            print(f'Synthesizing {options.gen_num} of data...')
            syn.ae.load_state_dict(last_model)
            h = syn.synthesize(options.gen_num, options.gen_batch_size) #Generating n*batch_size synthetic data
            with open(f"./data/synthesized_{options.task_name}_e{i}_gen{options.gen_num}", "wb") as f:
                pickle.dump(h, f)

