
#%%[markdown]
# # Encoder (Discriminator) Networks for generalising the Simple Pendulum

### Summary
# - build a family of discriminators, one for each environment
# - get the contexts that they output
# - use those contexts to initialise the Alternatig NODEs (in a seperate script)


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
init_lr = 3e-3

## Training hps
print_every = 100
nb_epochs_cal = 3000

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
os.system(f"cp {script_name} {data_folder}");

# - save the dataset as well
dataset_path = "./data/simple_pendulum_big.npz"
# os.system(f"cp {dataset_path} {data_folder}");

print("Data folder created successfuly:", data_folder)

#%%

dataset = np.load(dataset_path)
data, t_eval = dataset['X'][:, :, :, :], dataset['t']

nb_envs = data.shape[0]
nb_trajs_per_env = data.shape[1]
nb_steps_per_traj = data.shape[2]
data_size = data.shape[3]

batch_size = nb_envs*nb_trajs_per_env
cutoff_length = int(cutoff*nb_steps_per_traj)

print("Dataset's elements shapes:", data.shape, t_eval.shape)

# %%



class Discriminator(eqx.Module):
    layers: list
    proba_layers: list

    def __init__(self, traj_size, context_size, key=None):        ## TODO make this convolutional
        # print("Creating discriminator with traj size:", traj_size, "and context size:", context_size, "\n")

        keys = get_new_key(key, num=4)
        self.layers = [eqx.nn.Linear(traj_size, 100, key=keys[0]), jax.nn.softplus,
                        eqx.nn.Linear(100, 50, key=keys[1]), jax.nn.tanh,
                        eqx.nn.Linear(50, context_size, key=keys[2]) ]
        self.proba_layers = [eqx.nn.Linear(context_size, 1, key=keys[3]), jax.nn.sigmoid]

    def __call__(self, traj):
        # print("Encoder got and input of size:", traj.size)
        input_dat = jnp.reshape(jnp.transpose(traj, axes=(1,0)), (traj.size, ))
        context = input_dat
        for layer in self.layers:
            context = layer(context)

        proba = context
        for layer in self.proba_layers:
            proba = layer(proba)

        return proba, context


class Discriminators(eqx.Module):
    discriminators: list
    traj_size: int

    def __init__(self, data_size, context_size, traj_size, nb_envs, key=None):

        self.discriminators = [Discriminator(traj_size*data_size, context_size, key=key) for _ in range(nb_envs)]       ## All discriminators are identical at the begining

        self.traj_size = traj_size

    # def __call__(self, traj):
    #     probas = []
    #     contexts = []
    #     for discriminator in self.discriminators:
    #         proba, context = discriminator(traj[:self.traj_size, :].ravel())
    #         probas.append(proba)
    #         contexts.append(context)

    #     return jnp.concatenate(probas), jnp.vstack(contexts)


# %%

model_key, training_key, testing_key = get_new_key(SEED, num=3)

model = Discriminators(data_size=2, 
                context_size=context_size, 
                traj_size=cutoff_length,
                nb_envs=nb_envs, 
                key=model_key)

params, static = eqx.partition(model, eqx.is_array)







# %%


def params_norm(params):
    """ norm of the parameters """
    return jnp.array([jnp.linalg.norm(x) for x in jax.tree_util.tree_leaves(params)]).sum()

def l2_norm(X, X_hat):
    total_loss = jnp.mean((X - X_hat)**2, axis=-1)   ## Norm of d-dimensional vectors
    return jnp.sum(total_loss) / (X.shape[-2] * X.shape[-3])

@partial(jax.vmap, in_axes=(0, None, None, None, None))
def meanify_xis(e, es_hat, xis_hat, es_orig, xis_orig):
    # print("All shapes before meaninfying:", es_hat.shape, es_orig.shape, xis_hat.shape, xis_orig.shape, "\n")
    return jax.lax.cond((es_hat==e).sum()>0, 
                        lambda e: jnp.where((es_hat==e), xis_hat, 0.0).sum(axis=0) / (es_hat==e).sum(), 
                        lambda e: jnp.where((es_orig==e), xis_orig, 0.0).sum(axis=0) / (es_orig==e).sum(),      ## TODO, always make sure the batch is representative of all environments
                        e)



def make_training_batch_cal(batch_id, xis, data, cutoff_length, batch_size, key):      ## TODO: benchmark and save these btaches to disk
    """ Make a batch """

    nb_trajs_per_batch_per_env = batch_size//nb_envs
    traj_start, traj_end = batch_id*nb_trajs_per_batch_per_env, (batch_id+1)*nb_trajs_per_batch_per_env

    es_batch = []
    xis_batch = []
    X_batch = []

    for e in range(nb_envs):
        es_batch.append(jnp.ones((nb_trajs_per_batch_per_env,), dtype=int)*e)
        xis_batch.append(jnp.ones((nb_trajs_per_batch_per_env, context_size))*xis[e:e+1, :])
        X_batch.append(data[e, traj_start:traj_end, :cutoff_length, :])

    return (jnp.concatenate(es_batch), jnp.vstack(xis_batch), jnp.vstack(X_batch), t_eval[:cutoff_length]), key



# @partial(jax.jit, static_argnums=(2,3))





# %%

### ==== Calibration of the discriminators with real trajectories ==== ####


## Main loss function
def loss_fn_cal(params, static, batch):
    # print('\nCompiling function "loss_fn" ...\n')
    es, xis, Xs, t_eval = batch
    print("Shapes of elements in a batch:", es.shape, xis.shape, Xs.shape, t_eval.shape, "\n")

    model = eqx.combine(params, static)

    X_input = Xs[:, :model.traj_size, :]
    # X_input = jnp.reshape(jnp.transpose(Xs[:, :cutoff_length, :], axes=(0,2,1)), (Xs.shape[0], -1))

    probas = []
    contexts = []
    for discriminator in model.discriminators:
        proba, context = jax.vmap(discriminator)(X_input)
        probas.append(proba)
        contexts.append(context[:, None, :])

    probas, xis_hat = jnp.concatenate(probas, axis=1), jnp.concatenate(contexts, axis=1)

    cross_ent = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(probas, es))

    es_hat = jnp.argmax(probas, axis=1)

    pen_xis = jnp.mean(xis_hat**2)

    xis_hat = xis_hat[jnp.arange(Xs.shape[0]), es_hat]
    new_xis = meanify_xis(jnp.arange(nb_envs), es_hat[:,None], xis_hat, es[:,None], xis)

    # loss_val = cross_ent
    # loss_val = error_es + 1e-3*pen_params
    # loss_val = cross_ent + 1e-3*pen_params
    loss_val = cross_ent + pen_xis

    return loss_val, (new_xis)


@partial(jax.jit, static_argnums=(1))
def train_step_cal(params, static, batch, opt_state):
    print('\nCompiling function "train_step" ...\n')

    (loss, aux_data), grads  = jax.value_and_grad(loss_fn_cal, has_aux=True)(params, static, batch)

    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss, aux_data

def test_ste_cal(params, static):
    ## Test the discriminator on the training data.

    model = eqx.combine(params, static)

    true = []
    pred = []
    colors = ['dodgerblue', 'r', 'b', 'g', 'm', 'c', 'y', 'orange', 'purple', 'brown']
    colors = colors*(nb_envs)
    cols = []
    for e in range(nb_envs):
        # X_input = jnp.reshape(jnp.transpose(data[e, :, :cutoff_length, :], axes=(0,2,1)), (data.shape[1], -1))
        X_input = data[e, :, :cutoff_length, :]

        probas = []
        for discriminator in model.discriminators:
            proba, _ = jax.vmap(discriminator)(X_input)
            probas.append(proba)
        pred += jnp.argmax(jnp.concatenate(probas, axis=1), axis=1).tolist()

        true += [e]*data.shape[1]
        cols += [colors[e]]*data.shape[1]

    return true, pred, cols




if train == True:

    nb_trajs_per_batch_per_env = batch_size//nb_envs
    nb_train_steps_per_epoch = nb_trajs_per_env//nb_trajs_per_batch_per_env
    assert nb_train_steps_per_epoch > 0, "Batch size is too large"

    total_steps_cal = nb_epochs_cal * nb_train_steps_per_epoch

    # sched = init_lr
    sched = optax.exponential_decay(init_lr, total_steps_cal, 0.895)
    opt = optax.adam(sched)
    opt_state = opt.init(params)

    print(f"\n\n=== Beginning calibration of the discriminators ... ===")
    print(f"    Number of trajectories used in a single batch per environemnts: {nb_trajs_per_batch_per_env}")
    print(f"    Actual size of a batch (number of examples for all envs): {batch_size}")
    print(f"    Number of train steps per epoch: {nb_train_steps_per_epoch}")
    print(f"    Number of calibration epochs: {nb_epochs_cal}")
    print(f"    Total number of calibration steps: {total_steps_cal}")

    start_time = time.time()

    # xis = np.random.normal(size=(nb_envs, 2))
    context_key, batch_key = get_new_key(training_key, num=2)
    # xis = jax.random.normal(context_key, (nb_envs, context_size))   # TODO: initialise them the same !!
    xi = jax.random.normal(context_key, (1, context_size))
    xis = jnp.broadcast_to(xi, (nb_envs, context_size))
    init_xis = xis.copy()

    losses_cal = []
    for epoch in range(nb_epochs_cal):

        nb_batches = 0
        loss_sum = 0

        for i in range(nb_train_steps_per_epoch):

            batch, batch_key = make_training_batch_cal(i, xis, data, cutoff_length, batch_size, batch_key)
        
            params, opt_state, loss, (xis) = train_step_cal(params, static, batch, opt_state)

            loss_sum += loss
            nb_batches += 1

        loss_epoch = loss_sum/nb_batches
        losses_cal.append(loss_epoch)

        if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs_cal-1:
            print(f"    Epoch: {epoch:-5d}      CalibLoss: {loss_epoch:-.8f}", flush=True)

    losses_cal = jnp.vstack(losses_cal)

    wall_time = time.time() - start_time
    time_in_hmsecs = seconds_to_hours(wall_time)

    print("\nCalibration step training time: %d hours %d mins %d secs" %time_in_hmsecs)


else:
    print("\nNo training, skipping calibration step ...\n")



# %%

if train == True:
    fig, ax = plt.subplot_mosaic('AA;CD', figsize=(6*2, 4.5*2), width_ratios=[1, 1])

    ax['A'].plot(losses_cal[:], label="Cross-Entropy", color="brown", linewidth=3, alpha=1.0)
    ax['A'].set_xlabel("Epochs")
    ax['A'].set_title("Calibration Loss")
    ax['A'].set_yscale('log')
    ax['A'].legend()

    true, pred, cols = test_ste_cal(params, static)

    df = pd.DataFrame({'True':true[:], 'Pred':pred[:]})
    df = df.groupby(['True', 'Pred']).size().reset_index(name='Counts')
    df = df.pivot(index='Pred', columns='True', values='Counts')
    sns.heatmap(df, annot=True, ax=ax['C'], cmap='YlOrBr', cbar=False, fmt="n")
    ax['C'].set_title("True vs. Pred for all Envs and Discriminators")
    ax['C'].invert_yaxis()

    ## Calculate accuracy, precision, and recall
    acc = (np.array(true)==np.array(pred)).sum()/len(true)
    print(f"Accuracy: {acc*100:.3f} %")

    xis_all = np.vstack([xis, init_xis])
    eps = 0.1
    xmin, xmax = xis_all[:,0].min()-eps, xis_all[:,0].max()+eps
    ymin, ymax = xis_all[:,1].min()-eps, xis_all[:,1].max()+eps
    colors = ['dodgerblue', 'r', 'b', 'g', 'm', 'c', 'y', 'orange', 'purple', 'brown']
    colors = colors*(nb_envs)

    # ax['D'].scatter(init_xis[:,0], init_xis[:,1], s=30, c=colors[:nb_envs], marker='X', label="Initial", alpha=0.1)
    ax['D'].scatter(xis[:,0], xis[:,1], s=50, c=colors[:nb_envs], marker='o', label="Final")
    for i, (x, y) in enumerate(init_xis[:, :2]):
        ax['D'].annotate(str(i), (x, y), fontsize=8)
    for i, (x, y) in enumerate(xis[:, :2]):
        ax['D'].annotate(str(i), (x, y), fontsize=8)
    ax['D'].set_title(r'Initial and Final Contexts ($\xi^e$)')
    ax['D'].legend()

    np.savez(data_folder+"enc_contexts.npz", initial=init_xis, final=xis)

    plt.tight_layout()
    plt.savefig(data_folder+"/encoder_networks.png", dpi=100, bbox_inches='tight')
    plt.show()



# %%[markdown]
    ### This is the time to train the Alternating NODEs with the contexts that we just got


