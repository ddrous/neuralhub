
#%%[markdown]
# # Bridge Network for generalising the Simple Pendulum

### Summary
# - train the bridge network how to trasform encoder contexts to NODE contexts
# - (optional) fine-tuning the resulting context by the alternating nodes (in a seperate script)

#%%

# import random
import jax

# from jax import config
jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_platform_name", "cpu")

print("\n############# Bridge Network for Simple Pendulum  #############\n")
print("Jax version:", jax.__version__)
print("Available devices:", jax.devices())

import jax.numpy as jnp

import numpy as np
np.set_printoptions(suppress=True)

import equinox as eqx
import diffrax

import matplotlib.pyplot as plt

from neuralhub.utils import *
from neuralhub.integrators import *

import pandas as pd
import seaborn as sns

import optax
from functools import partial

import os
import time

#%%

SEED = 7

## Optimiser hps
init_lr = 3e-2

## Training hps
print_every = 100
nb_epochs = 3000

cutoff = 0.2
context_size = 2

train = True

#%%

# - make a new folder inside 'data' whose name is the currennt time
# data_folder = './data/'+time.strftime("%d%m%Y-%H%M%S")+'/'
data_folder = './data/00_Alternating/'
# os.mkdir(data_folder)

# - save the script in that folder
script_name = os.path.basename(__file__)
# os.system(f"cp {script_name} {data_folder}");

# - save the dataset as well
dataset_path = data_folder
# os.system(f"cp {dataset_path} {data_folder}");



# %%[markdown]
    ### This is the time to train the Alternating NODEs with the contexts that we just got




#%%

enc_contextts = np.load(data_folder+"enc_contexts.npz")
enc_c_in, enc_c_out = enc_contextts['initial'], enc_contextts['final']

node_contexts = np.load(data_folder+"node_contexts.npz")
node_c_in, node_c_out = node_contexts['initial'], node_contexts['final']

# assert jnp.abs(enc_c_out-node_c_in).any() < 1e-10, "Contexts are not the same"

contexts_in, contexts_out = node_c_in, node_c_out

print("Dataset's elements shapes:", contexts_in.shape, contexts_out.shape)

# %%

















# %%
    
## Load the final contexts as ground truth



class Bridge(eqx.Module):
    layers: list

    def __init__(self, context_size, key=None):

        keys = get_new_key(key, num=4)
        self.layers = [eqx.nn.Linear(context_size, 16*4, key=keys[0]), jax.nn.softplus,
                        eqx.nn.Linear(16*4, 8*2, key=keys[1]), jax.nn.softplus,
                        eqx.nn.Linear(8*2, context_size, key=keys[2]) ]

    def __call__(self, xi):
        context = xi
        for layer in self.layers:
            context = layer(context)
        return context


# %%

model_key, training_key, testing_key = get_new_key(SEED, num=3)

model = Bridge(context_size=context_size, 
                key=model_key)

params, static = eqx.partition(model, eqx.is_array)







# %%


def params_norm(params):
    """ norm of the parameters """
    return jnp.array([jnp.linalg.norm(x) for x in jax.tree_util.tree_leaves(params)]).sum()

def make_train_tests_batches(contexts_in, contexts_out, key):

    train_size = int(0.8*contexts_in.shape[0])
    # test_size = context_in.shape[0] - train_size

    # random_idx = jax.random.permutation(key, jnp.arange(contexts_in.shape[0]))
    random_idx = jnp.arange(contexts_in.shape[0])
    train_batch = contexts_in[random_idx[:train_size]], contexts_out[random_idx[:train_size]]
    test_batch = contexts_in[random_idx[train_size:]], contexts_out[random_idx[train_size:]]

    return train_batch, test_batch


# %%


## Main loss function
def loss_fn(params, static, batch):
    # print('\nCompiling function "loss_fn" ...\n')
    context_in, context_out = batch
    print("Shapes of elements in a batch:", context_in.shape, context_out.shape, "\n")

    model = eqx.combine(params, static)

    contexts_hat = jax.vmap(model)(context_in)
    loss_val = jnp.sum((context_out - contexts_hat)**2, axis=-1).mean()

    pen_model = params_norm(params)

    # loss_val = loss_val + 1e-3*pen_model
    loss_val = loss_val

    return loss_val, (pen_model)


@partial(jax.jit, static_argnums=(1))
def train_step(params, static, batch, opt_state):
    print('\nCompiling function "train_step" ...\n')

    (loss, aux_data), grads  = jax.value_and_grad(loss_fn, has_aux=True)(params, static, batch)

    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss, aux_data


if train == True:

    sched = optax.exponential_decay(init_lr, nb_epochs, 0.895)
    opt = optax.adam(sched)
    opt_state = opt.init(params)

    print(f"\n\n=== Beginning training of the bridge network ... ===")

    start_time = time.time()

    losses = []
    pen_params = []
    for epoch in range(nb_epochs):
        train_data, _ = make_train_tests_batches(contexts_in, contexts_out, training_key)
    
        params, opt_state, loss, (pen_model) = train_step(params, static, train_data, opt_state)

        losses.append(loss)
        pen_params.append(pen_model)

        if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
            print(f"    Epoch: {epoch:-5d}      MSELoss: {loss:-.8f}        ParamsLoss: {pen_model:-.8f}", flush=True)

    losses = jnp.array(losses)
    pen_params = jnp.array(pen_params)

    wall_time = time.time() - start_time
    time_in_hmsecs = seconds_to_hours(wall_time)

    print("\nCalibration step training time: %d hours %d mins %d secs" %time_in_hmsecs)


else:
    print("\nNo training, skipping calibration step ...\n")



# %%

if train == True:
    fig, ax = plt.subplot_mosaic('AA;BC', figsize=(6*2, 4.5*2), width_ratios=[1, 1])

    ax['A'].plot(losses, label="MSE", color="brown", linewidth=3, alpha=1.0)
    ax['A'].plot(pen_params, label="Regularisation", color="grey", linewidth=1, alpha=0.5)
    ax['A'].set_xlabel("Epochs")
    ax['A'].set_title("Loss Terms")
    ax['A'].set_yscale('log')
    ax['A'].legend()

    model = eqx.combine(params, static)

    train_data, test_data = make_train_tests_batches(contexts_in, contexts_out, testing_key)
    xis_train = train_data[1]
    xis_test = test_data[1]
    xis_hat_train = jax.vmap(model)(train_data[0])
    xis_hat_test = jax.vmap(model)(test_data[0])

    ## For training
    xis_all = np.vstack([train_data[1], xis_hat_train])
    eps = 0.1
    xmin, xmax = xis_all[:,0].min()-eps, xis_all[:,0].max()+eps
    ymin, ymax = xis_all[:,1].min()-eps, xis_all[:,1].max()+eps
    colors = ['dodgerblue', 'r', 'b', 'g', 'm', 'c', 'y', 'orange', 'purple', 'brown']
    colors = colors*(xis_train.shape[0])

    ax['B'].scatter(xis_train[:,0], xis_train[:,1], s=30, c=colors[:xis_train.shape[0]], marker='X', label="True", alpha=0.5)
    ax['B'].scatter(xis_hat_train[:,0], xis_hat_train[:,1], s=50, c=colors[:xis_hat_train.shape[0]], marker='o', label="Pred")
    for i, (x, y) in enumerate(xis_train[:, :2]):
        ax['B'].annotate(str(i), (x, y), fontsize=8)
    for i, (x, y) in enumerate(xis_hat_train[:, :2]):
        ax['B'].annotate(str(i), (x, y), fontsize=8)
    ax['B'].set_title(r'Training Contexts')
    ax['B'].legend()

    # print("== Training ==")
    # print(" True contexts: \n", xis_train)
    # print(" Pred contexts: \n", xis_hat_train)

    ## For testing
    xis_all = np.vstack([test_data[1], xis_hat_test])
    eps = 0.1
    xmin, xmax = xis_all[:,0].min()-eps, xis_all[:,0].max()+eps
    ymin, ymax = xis_all[:,1].min()-eps, xis_all[:,1].max()+eps
    colors = ['dodgerblue', 'r', 'b', 'g', 'm', 'c', 'y', 'orange', 'purple', 'brown']
    colors = colors*(xis_test.shape[0])

    ax['C'].scatter(xis_test[:,0], xis_test[:,1], s=30, c=colors[:xis_test.shape[0]], marker='X', label="True", alpha=0.5)
    ax['C'].scatter(xis_hat_test[:,0], xis_hat_test[:,1], s=50, c=colors[:xis_hat_test.shape[0]], marker='o', label="Pred")
    for i, (x, y) in enumerate(xis_test[:, :2]):
        ax['C'].annotate(str(i), (x, y), fontsize=8)
    for i, (x, y) in enumerate(xis_hat_test[:, :2]):
        ax['C'].annotate(str(i), (x, y), fontsize=8)
    ax['C'].set_title(r'Testing Contexts')
    ax['C'].legend()

    # print("== Testing ==")
    # print(" True contexts: \n", xis_test)
    # print(" Pred contexts: \n", xis_hat_test)

    plt.tight_layout()
    plt.savefig(data_folder+"/bridge_network.png", dpi=100, bbox_inches='tight')
    plt.show()






























