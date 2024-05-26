
#%%[markdown]
# # GAN-Neural ODE framework for generalising the Simple Pendulum
# List of ToDOs
# - Use a time series as input to the discriminators, rather than a single point
# - Put (concatenate) the context back in before each layer of the generator (neural ODE)
# - Do I let the optimiser maintain its state from the calibration to the training phase ?

### Summary
# - why not use one discriminator? because it wouldn't be good for adaptation


#%%

# import random
import jax

# from jax import config
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_platform_name", "cpu")

print("\n############# Lotka-Volterra with Generator and Discriminators #############\n")
print("Jax version:", jax.__version__)
print("Available devices:", jax.devices())

import jax.numpy as jnp
# import jax.scipy as jsp
# import jax.scipy.optimize

import numpy as np
np.set_printoptions(suppress=True)
# from scipy.integrate import solve_ivp

import equinox as eqx
import diffrax

import matplotlib.pyplot as plt

from neuralhub.utils import *
from neuralhub.integrators import *

import optax
from functools import partial

import os
import time
# from typing import List, Tuple, Callable


#%%

SEED = 27
# SEED = np.random.randint(0, 1000)

## Integrator hps
integrator = rk4_integrator

## Optimiser hps
init_lr = 3e-2

## Training hps
print_every = 100
nb_epochs_cal = 300
nb_epochs = 2500
batch_size = 2*128*10       ## 2 is the number of environments

cutoff = 0.4
context_size = 20

train = True

#%%

# - make a new folder inside 'data' whose name is the currennt time
data_folder = './data/'+time.strftime("%d%m%Y-%H%M%S")+'/'
os.mkdir(data_folder)

# - save the script in that folder
script_name = os.path.basename(__file__)
os.system(f"cp {script_name} {data_folder}");

# - save the dataset as well
dataset_path = "./data/simple_pendulum_big.npz"
os.system(f"cp {dataset_path} {data_folder}");

print("Data folder created successfuly:", data_folder)

#%%

dataset = np.load(dataset_path)
data, t_eval = dataset['X'][:, :, :, :], dataset['t']

nb_envs = data.shape[0]
nb_trajs_per_env = data.shape[1]
nb_steps_per_traj = data.shape[2]
data_size = data.shape[3]

cutoff_length = int(cutoff*nb_steps_per_traj)

print("Dataset's elements shapes:", data.shape, t_eval.shape)

# %%


class Physics(eqx.Module):
    params: jnp.ndarray

    def __init__(self, key=None):
        keys = get_new_key(key, num=2)
        self.params = jnp.concatenate([jax.random.uniform(keys[0], (1,), minval=0.25, maxval=1.75),
                                       jax.random.uniform(keys[1], (1,), minval=1, maxval=30)])

    def __call__(self, t, x):
        L, g = self.params
        theta, theta_dot = x
        theta_ddot = -(g / L) * jnp.sin(theta)
        return jnp.array([theta_dot, theta_ddot])

class Augmentation(eqx.Module):
    layers: list

    def __init__(self, data_size, width_size, depth, key=None):
        keys = get_new_key(key, num=3)
        self.layers = [eqx.nn.Linear(data_size, width_size, key=keys[0]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, width_size, key=keys[1]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, data_size, key=keys[2]) ]

    def __call__(self, t, x):
        # y = jnp.concatenate([jnp.broadcast_to(t, (1,)), x], axis=0)
        y = x
        for layer in self.layers:
            y = layer(y)
        return y

class SharedProcessor(eqx.Module):
    physics: Physics
    # augmentation: Augmentation

    def __init__(self, data_size, width_size, depth, key=None):
        keys = get_new_key(key, num=2)
        self.physics = Physics(key=keys[0])
        # self.augmentation = Augmentation(data_size, width_size, depth, key=keys[1])

    def __call__(self, t, x):
        # return self.physics(t, x) + self.augmentation(t, x)
        # return self.augmentation(t, x)
        return self.physics(t, x)

# class EnvProcessor(eqx.Module):
#     layers: list

#     def __init__(self, data_size, width_size, depth, context_size, key=None):
#         keys = get_new_key(key, num=3)
#         self.layers = [eqx.nn.Linear(data_size+context_size, width_size, key=keys[0]), jax.nn.softplus,
#         # self.layers = [eqx.nn.Linear(context_size, width_size, key=keys[0]), jax.nn.softplus,
#         # self.layers = [eqx.nn.Linear(data_size+context_size+1, width_size, key=keys[0]), jax.nn.softplus,
#         # self.layers = [eqx.nn.Linear(data_size, width_size, key=keys[0]), jax.nn.softplus,
#                         eqx.nn.Linear(width_size, width_size, key=keys[1]), jax.nn.softplus,
#                         # eqx.nn.Linear(width_size, context_size, key=keys[2])]
#                         eqx.nn.Linear(width_size, data_size, key=keys[2])]
#         # self.layers = [eqx.nn.Linear(data_size+context_size, width_size, key=keys[0]), jax.nn.softplus,
#         #                 eqx.nn.Linear(width_size+context_size, width_size, key=keys[1]), jax.nn.softplus,
#         #                 eqx.nn.Linear(width_size+context_size, data_size, key=keys[2])]
#     def __call__(self, t, x, context):
#         # y = jnp.concatenate([jnp.broadcast_to(t, (1,)), x], axis=0)
#         # y = x

#         # jax.debug.print("\n\n\n\n\n\nx shape {} context {}\n\n\n\n\n\n", x, context)
#         # jax.debug.breakpoint()

#         y = jnp.concatenate([x, context], axis=0)
#         # y = jnp.concatenate([jnp.broadcast_to(t, (1,)), x, context], axis=0)
#         for layer in self.layers:
#             y = layer(y)
#         return y

#         # y = x
#         # for layer in self.layers:
#         #     y = layer(y)
#         # return y@jnp.broadcast_to(context[:, None], (context.shape[0], x.shape[0]))

#         # params = context
#         # for layer in self.layers:
#         #     params = layer(params)
#         # L, g = params
#         # theta, theta_dot = x
#         # theta_ddot = -(g / L) * jnp.sin(theta)
#         # return jnp.array([theta_dot, theta_ddot])


#         # y = x
#         # for i, layer in enumerate(self.layers):
#         #     if i%2==0:
#         #         y = jnp.concatenate([y, context], axis=-1)
#         #     y = layer(y)
#         # return y




# class EnvProcessor(eqx.Module):
#     layers_data: list
#     layers_context: list
#     layers_shared: list

#     def __init__(self, data_size, width_size, depth, context_size, key=None):
#         keys = generate_new_keys(key, num=10)
#         self.layers_data = [eqx.nn.Linear(data_size, width_size, key=keys[0]), jax.nn.softplus,
#                         eqx.nn.Linear(width_size, width_size, key=keys[1]), jax.nn.softplus,
#                         eqx.nn.Linear(width_size, data_size, key=keys[2])]

#         self.layers_context = [eqx.nn.Linear(context_size, width_size, key=keys[3]), jax.nn.softplus,
#                         eqx.nn.Linear(width_size, width_size, key=keys[4]), jax.nn.tanh,
#                         eqx.nn.Linear(width_size, data_size, key=keys[5])]

#         self.layers_shared = [eqx.nn.Linear(data_size*3, width_size, key=keys[6]), jax.nn.softplus,
#         # self.layers_shared = [eqx.nn.Linear(data_size+data_size, width_size, key=keys[6]), jax.nn.softplus,
#                         # eqx.nn.Linear(width_size, width_size, key=keys[7]), jax.nn.softplus,
#                         eqx.nn.Linear(width_size, width_size, key=keys[8]), jax.nn.softplus,
#                         eqx.nn.Linear(width_size, data_size, key=keys[9])]


#     def __call__(self, t, x, context):

#         y = x
#         context = context
#         for i in range(len(self.layers_data)):
#             y = self.layers_data[i](y)
#             context = self.layers_context[i](context)

#         y1 = jnp.concatenate([y, context], axis=0)
#         y2 = y*context
#         # y3 = y+context
#         y =  jnp.concatenate([y1, y2], axis=0)
#         for layer in self.layers_shared:
#             y = layer(y)
#         return y



class EnvProcessor(eqx.Module):
    layers_data: list
    layers_context: list

    def __init__(self, data_size, width_size, depth, context_size, key=None):
        keys = generate_new_keys(key, num=10)
        self.layers_data = [eqx.nn.Linear(data_size, width_size, key=keys[0]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, width_size, key=keys[1]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, width_size, key=keys[6]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, context_size*2, key=keys[2])]

        self.layers_context = [eqx.nn.Linear(context_size, width_size, key=keys[3]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, width_size, key=keys[4]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, width_size, key=keys[7]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, context_size, key=keys[5])]


    def __call__(self, t, x, context):

        y = x
        context = context
        for i in range(len(self.layers_data)):
            y = self.layers_data[i](y)
            context = self.layers_context[i](context)

        y1 = y[:context_size]@context
        y2 = y[context_size:]@context

        return jnp.array([y1, y2])


# def distance(node1, node2):
#     diff = node1 - node2
#     return jnp.sum(diff*diff)      ## Squared distance ! No problem for Gaussain RBF
#     # return jnp.linalg.norm(node1 - node2)       ## Carefull: not differentiable at 0

# def gaussian_rbf(r_squared, shape):
#     return jnp.exp(-r_squared / ( 2 * shape**2))

# def gaussian_rbf_full(node1, node2, shape):
#     return gaussian_rbf(distance(node1, node2), shape)

# class EnvProcessor(eqx.Module):
#     centers: jnp.ndarray
#     shapes: jnp.ndarray     ## Widths for the gaussian RBF network
#     weights: eqx.nn.Linear      ## Weights for the decoder to average its outputs

#     def __init__(self, data_size, width_size, depth, context_size, key=None):
#         in_size = data_size
#         nb_centers = context_size+width_size
#         keys = get_new_key(key, num=3)

#         self.centers = jax.random.uniform(keys[0], (nb_centers, in_size), minval=0., maxval=3.)
#         self.shapes = jax.random.uniform(keys[1], (nb_centers,), minval=0.1, maxval=10.)

#         self.weights = eqx.nn.Linear(nb_centers, in_size, key=keys[2])

#     def define_mat(self):
#         rows_func = jax.vmap(gaussian_rbf_full, in_axes=(0, None, 0), out_axes=(0))
#         mat_func = jax.vmap(rows_func, in_axes=(None, 0, None), out_axes=(0))
#         return mat_func(self.centers, self.centers, self.shapes)

#     def __call__(self, x):  ## Encode
#         zeros = jnp.zeros((self.centers.shape[0]-x.shape[0], ))
#         y = jnp.concatenate([x, zeros], axis=0)
#         ## Solve the linear system for find the weights
#         return jnp.linalg.solve(self.define_mat(), y)

#     def decode(self, lamb):
#         x_full = self.define_mat()@lamb
#         # return self.weights(x_full)
#         return x_full



class Processor(eqx.Module):
    shared: SharedProcessor
    env: EnvProcessor

    def __init__(self, data_size, width_size, depth, context_size, key=None):
        keys = get_new_key(key, num=2)
        self.shared = SharedProcessor(data_size, width_size, depth, key=keys[0])
        self.env = EnvProcessor(data_size, width_size, depth, context_size, key=keys[1])

    def __call__(self, t, x, context):
        return self.shared(t, x) + self.env(t, x, context)



class Generator(eqx.Module):
    processor: Processor

    def __init__(self, data_size, width_size, depth, context_size, key=None):
        self.processor = Processor(data_size, width_size, depth, context_size, key=key)

    def __call__(self, x0, t_eval, context):

        solution = diffrax.diffeqsolve(
                    diffrax.ODETerm(self.processor),
                    diffrax.Tsit5(),
                    args=context,
                    t0=t_eval[0],
                    t1=t_eval[-1],
                    dt0=t_eval[1] - t_eval[0],
                    y0=x0,
                    stepsize_controller=diffrax.PIDController(rtol=1e-1, atol=1e-2),
                    saveat=diffrax.SaveAt(ts=t_eval),
                    max_steps=4096*10,
                )
        return solution.ys, solution.stats["num_steps"]

        # rhs = lambda x, t: self.processor(t, x, context)
        # X_hat = integrator(rhs, x0, t_eval, None, None, None, None, None, None)
        # return X_hat, t_eval.size






class Discriminator(eqx.Module):
    layers: list
    proba_layers: list

    def __init__(self, traj_size, context_size, key=None):        ## TODO make this convolutional
        # super().__init__(**kwargs)
        keys = get_new_key(key, num=4)
        self.layers = [eqx.nn.Linear(traj_size, 100, key=keys[0]), jax.nn.softplus,
                        eqx.nn.Linear(100, 50, key=keys[1]), jax.nn.tanh,
                        eqx.nn.Linear(50, context_size, key=keys[2]) ]
        self.proba_layers = [eqx.nn.Linear(context_size, 1, key=keys[3]), jax.nn.sigmoid]

        # self.layers = [lambda x:x[None,...], eqx.nn.Conv1d(1, 2, kernel_size=5, key=keys[0]), eqx.nn.MaxPool1d(kernel_size=2), jax.nn.relu,
        #                 eqx.nn.Conv1d(2, 1, kernel_size=5, key=keys[1]), eqx.nn.MaxPool1d(kernel_size=2), jax.nn.softplus,
        #                 jnp.ravel, eqx.nn.Linear(90, context_size, key=keys[2]) ]
        # self.proba_layers = [eqx.nn.Linear(context_size, 1, key=keys[3]), jax.nn.sigmoid]


    def __call__(self, traj):
        # print("Encoder got and input of size:", traj.size)
        context = traj
        for layer in self.layers:
            context = layer(context)

        proba = context
        for layer in self.proba_layers:
            proba = layer(proba)

        return proba, context


class GANNODE(eqx.Module):
    generator: Generator
    discriminators: list       ## TODO, rather, an ensemble of discriminators. A list might be better ?
    traj_size: int              ## Based on the above, this shouldn't be needed. TODO: use a time series instead

    def __init__(self, proc_data_size, proc_width_size, proc_depth, context_size, traj_size, nb_envs, key=None):
        keys = get_new_key(key, num=1+nb_envs)

        self.generator = Generator(proc_data_size, proc_width_size, proc_depth, context_size, key=keys[1])        
        self.discriminators = [Discriminator(traj_size*proc_data_size, context_size, key=key) for key in keys[1:]]
        
        self.traj_size = traj_size

    def __call__(self, x0, t_eval, xi):

        traj, nb_steps = self.generator(x0, t_eval, xi)

        probas = []
        contexts = []
        for discriminator in self.discriminators:
            proba, context = discriminator(traj[:self.traj_size, :].ravel())
            probas.append(proba)
            contexts.append(context)

        return traj, nb_steps, jnp.concatenate(probas), jnp.vstack(contexts)         ## TODO: even tho all contexts are returned, only the corresponding ones should be used for the loss


# %%

model_key, training_key, testing_key = get_new_key(SEED, num=3)

model = GANNODE(proc_data_size=2, 
                proc_width_size=16*2, 
                proc_depth=3, 
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
    # return jnp.sum(total_loss) / (X.shape[-2] * X.shape[-3])
    return jnp.sum(total_loss) / (X.shape[-2] * X.shape[-3])


## Gets the mean of the xi's for each environment
@partial(jax.vmap, in_axes=(0, None, None, None, None))
def meanify_xis(e, es_hat, xis_hat, es_orig, xis_orig):      ## TODO: some xi's get updated, others don't
    # print("All shapes before meaninfying:", es_hat.shape, es_orig.shape, xis_hat.shape, xis_orig.shape, "\n")
    return jax.lax.cond((es_hat==e).sum()>0, 
                        lambda e: jnp.where((es_hat==e), xis_hat, 0.0).sum(axis=0) / (es_hat==e).sum(), 
                        lambda e: jnp.where((es_orig==e), xis_orig, 0.0).sum(axis=0) / (es_orig==e).sum(),      ## TODO, always make sure the batch is representative of all environments
                        e)


# @partial(jax.jit, static_argnums=(2,3))
def make_training_batch(batch_id, xis, data, cutoff_length, batch_size, key):      ## TODO: benchmark and save these btaches to disk
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


def make_training_batch_successive_envs(batch_id, epoch, xis, data, cutoff_length, batch_size, key):      ## TODO: benchmark and save these btaches to disk
    """ Make a batch """

    nb_trajs_per_batch_per_env = batch_size//nb_envs
    traj_start, traj_end = batch_id*nb_trajs_per_batch_per_env, (batch_id+1)*nb_trajs_per_batch_per_env

    es_batch = []
    xis_batch = []
    X_batch = []

    nb_epochs_for_one_env = nb_epochs//nb_envs
    e = epoch//nb_epochs_for_one_env

    # print("e:", e, "epoch:", epoch, "nb_epochs_for_one_env:", nb_epochs_for_one_env, "nb_envs:", nb_envs, "\n")

    es_batch.append(jnp.ones((nb_trajs_per_batch_per_env,), dtype=int)*e)
    xis_batch.append(jnp.ones((nb_trajs_per_batch_per_env, context_size))*xis[e:e+1, :])
    X_batch.append(data[e, traj_start:traj_end, :cutoff_length, :])

    return (jnp.concatenate(es_batch), jnp.vstack(xis_batch), jnp.vstack(X_batch), t_eval[:cutoff_length]), key




# %%

### ==== Calibration of the discriminators with real trajectories ==== ####


## Main loss function
def loss_fn_cal(params, static, batch):
    # print('\nCompiling function "loss_fn" ...\n')
    es, xis, Xs, t_eval = batch
    print("Shapes of elements in a batch:", es.shape, xis.shape, Xs.shape, t_eval.shape, "\n")

    model = eqx.combine(params, static)

    X_input = jnp.reshape(jnp.transpose(Xs[:, :cutoff_length, :], axes=(0,2,1)), (Xs.shape[0], -1))

    probas = []
    contexts = []
    for discriminator in model.discriminators:
        proba, context = jax.vmap(discriminator)(X_input)
        probas.append(proba)
        contexts.append(context[:, None, :])

    probas, xis_hat = jnp.concatenate(probas, axis=1), jnp.concatenate(contexts, axis=1)

    cross_ent = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(probas, es))

    es_hat = jnp.argmax(probas, axis=1)
    # error_es = jnp.mean((es_hat - es)**2)
    # error_es = jnp.mean(jnp.abs(es_hat - es))

    # pen_params = params_norm(params.discriminators)

    pen_xis = jnp.mean(xis_hat**2)

    xis_hat = xis_hat[jnp.arange(Xs.shape[0]), es_hat]
    # new_xis = meanify_xis(es_hat, xis_hat, jnp.arange(nb_envs))
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
    cols = []
    for e in range(nb_envs):
        X_input = jnp.reshape(jnp.transpose(data[e, :, :cutoff_length, :], axes=(0,2,1)), (data.shape[1], -1))

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
    sched = optax.exponential_decay(init_lr, total_steps_cal, 0.95)
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
    xis = jax.random.normal(context_key, (nb_envs, context_size))
    init_xis = xis.copy()

    losses_cal = []
    for epoch in range(nb_epochs_cal):

        nb_batches = 0
        loss_sum = 0

        # _, batch_key = get_new_key(batch_key, num=2)
        # batch_keys = get_new_key(batch_key, num=nb_train_steps_per_epoch)

        for i in range(nb_train_steps_per_epoch):   ## Only two trajectories are used for each train_step

            batch, batch_key = make_training_batch(i, xis, data, cutoff_length, batch_size, batch_key)
        
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

import pandas as pd
import seaborn as sns

if train == True:
    fig, ax = plt.subplot_mosaic('AA;CD', figsize=(6*2, 4.5*2), width_ratios=[1, 1])

    ax['A'].plot(losses_cal[:], label="Cross-Entropy", color="brown", linewidth=3, alpha=1.0)
    ax['A'].set_xlabel("Epochs")
    ax['A'].set_title("Calibration Loss")
    ax['A'].set_yscale('log')
    ax['A'].legend()

    true, pred, cols = test_ste_cal(params, static)
    # ax['B'].scatter(true[:plot_size], pred[:plot_size], c=cols[:plot_size])

    df = pd.DataFrame({'True':true[:], 'Pred':pred[:]})
    df = df.groupby(['True', 'Pred']).size().reset_index(name='Counts')
    df = df.pivot(index='Pred', columns='True', values='Counts')
    sns.heatmap(df, annot=True, ax=ax['C'], cmap='YlOrBr', cbar=False, fmt="n")
    ax['C'].set_title("True vs. Pred for all Envs and Discriminators")
    ax['C'].invert_yaxis()

    xis_all = np.vstack([xis, init_xis])
    eps = 0.1
    xmin, xmax = xis_all[:,0].min()-eps, xis_all[:,0].max()+eps
    ymin, ymax = xis_all[:,1].min()-eps, xis_all[:,1].max()+eps
    colors = ['dodgerblue', 'r', 'b', 'g', 'm', 'c', 'y', 'orange', 'purple', 'brown']

    ax['D'].scatter(init_xis[:,0], init_xis[:,1], s=30, c=colors[:nb_envs], marker='X', label="Initial", alpha=0.1)
    ax['D'].scatter(xis[:,0], xis[:,1], s=50, c=colors[:nb_envs], marker='o', label="Final")
    for i, (x, y) in enumerate(init_xis[:, :2]):
        ax['D'].annotate(str(i), (x, y), fontsize=8)
    for i, (x, y) in enumerate(xis[:, :2]):
        ax['D'].annotate(str(i), (x, y), fontsize=8)
    ax['D'].set_title(r'Initial and Final Contexts  ($\xi^e$)')
    ax['D'].legend()

    plt.tight_layout()
    plt.savefig(data_folder+"/list_gan_node_calibration_04.png", dpi=100, bbox_inches='tight')
    plt.show()
















# %%

### ==== Step 2: training both generator and discriminators at once ==== ####


# ## Main loss function
def loss_fn(params, static, batch):
    # print('\nCompiling function "loss_fn" ...\n')
    es, xis, Xs, t_eval = batch
    print("Shapes of elements in a batch:", es.shape, xis.shape, Xs.shape, t_eval.shape, "\n")

    model = eqx.combine(params, static)

    Xs_hat, nb_steps, probas, xis_hat = jax.vmap(model, in_axes=(0, None, 0))(Xs[:, 0, :], t_eval, xis)

    probas_hat = jnp.max(probas, axis=1)        ## TODO: use this for cross-entropy loss

    cross_ent = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(probas, es))

    es_hat = jnp.argmax(probas, axis=1)
    xis_hat = xis_hat[jnp.arange(Xs.shape[0]), es_hat]
    # error_es = jnp.mean((es_hat - e)**2)
    error_es = jnp.mean(jnp.abs(es_hat - es))

    # new_xis = meanify_xis(es_hat, xis_hat, jnp.arange(nb_envs))
    new_xis = meanify_xis(jnp.arange(nb_envs), es_hat[:,None], xis_hat, es[:,None], xis)


    term1 = l2_norm(Xs, Xs_hat)
    # term2 = error_es
    # term2 = error_es+cross_ent
    term2 = cross_ent
    # term3 = params_norm(params.generator.processor.env)
    term3 = params_norm(params.discriminators)
    term4 = jnp.mean(new_xis**2)

    # loss_val = term1 + 1e-2*term2
    loss_val = term1 + term2 + 0e-3*term3 + term4
    # loss_val = term2
    # loss_val = term1 + 1e-2*term2 + term4 + 1e-3*term3

    return loss_val, (new_xis, jnp.sum(nb_steps), term1, term2, term3)
    # return loss_val, (init_xis, jnp.sum(nb_steps), term1, term2, term3)



# def loss_fn(params, static, batch):
#     # print('\nCompiling function "loss_fn" ...\n')
#     es, xis, Xs, t_eval = batch
#     print("Shapes of elements in a batch:", es.shape, xis.shape, Xs.shape, t_eval.shape, "\n")

#     model = eqx.combine(params, static)

#     Xs_hat, nb_steps = jax.vmap(model.generator, in_axes=(0, None, 0))(Xs[:, 0, :], t_eval, xis)

#     term1 = l2_norm(Xs, Xs_hat)
#     term3 = params_norm(params.generator.processor.env)

#     loss_val = term1 + 1e-3*term3

#     return loss_val, (jnp.sum(nb_steps), term1, term3)


@partial(jax.jit, static_argnums=(1))
def train_step(params, static, batch, opt_state):
    print('\nCompiling function "train_step" ...\n')

    (loss, aux_data), grads  = jax.value_and_grad(loss_fn, has_aux=True)(params, static, batch)

    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss, aux_data



if train == True:

    total_steps = nb_epochs * nb_train_steps_per_epoch

    # sched = optax.exponential_decay(init_lr, total_steps, decay_rate)     ## It has to finish at 0.001875

    # sched = optax.linear_schedule(init_lr, 0, total_steps, 0.25)
    sched = optax.piecewise_constant_schedule(init_value=init_lr,
                    # boundaries_and_scales={int(total_steps*0.25):0.5, 
                    #                         int(total_steps*0.5):0.5,
                    #                         int(total_steps*0.75):0.5})
                    boundaries_and_scales={500:0.5, 
                                            1500:0.5,
                                            3000:0.5,
                                            12000:0.5})
                    # boundaries_and_scales={1400:0.5})
    opt = optax.adam(sched)
    opt_state = opt.init(params)


    print(f"\n\n=== Beginning training of generator ... ===")
    print(f"    Number of trajectories used in a single batch per environemnts: {nb_trajs_per_batch_per_env}")
    print(f"    Actual size of a batch (number of examples for all envs): {batch_size}")
    print(f"    Number of train steps per epoch: {nb_train_steps_per_epoch}")
    print(f"    Number of training epochs: {nb_epochs}")
    print(f"    Total number of training steps: {total_steps}")


    start_time = time.time()

    # context_key, batch_key = get_new_key(training_key, num=2)
    # xis = jax.random.normal(context_key, (nb_envs, 2))
    # init_xis = xis.copy()

    losses = []
    nb_steps = []
    for epoch in range(nb_epochs):

        nb_batches = 0
        loss_sum = jnp.zeros(4)
        nb_steps_eph = 0

        # _, batch_key = get_new_key(batch_key, num=2)
        # batch_keys = get_new_key(batch_key, num=nb_train_steps_per_epoch)

        for i in range(nb_train_steps_per_epoch):   ## Only two trajectories are used for each train_step

            batch, batch_key = make_training_batch(i, xis, data, cutoff_length, batch_size, batch_key)
            # batch, batch_key = make_training_batch_successive_envs(i, epoch, xis, data, cutoff_length, batch_size, batch_key)

            params, opt_state, loss, (xis, nb_steps_val, term1, term2, term3) = train_step(params, static, batch, opt_state)
            # params, opt_state, loss, (nb_steps_val, term1, term3) = train_step(params, static, batch, opt_state)
            # term2 = 1.

            loss_sum += jnp.array([loss, term1, term2, term3])
            nb_steps_eph += nb_steps_val
            nb_batches += 1

        loss_epoch = loss_sum/nb_batches
        losses.append(loss_epoch)
        nb_steps.append(nb_steps_eph)

        if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
            print(f"    Epoch: {epoch:-5d}      TotalLoss: {loss_epoch[0]:-.8f}     Traj: {loss_epoch[1]:-.8f}      Discrim: {loss_epoch[2]:-.8f}      Params: {loss_epoch[3]:-.5f}", flush=True)

    losses = jnp.vstack(losses)
    nb_steps = jnp.array(nb_steps)

    wall_time = time.time() - start_time
    time_in_hmsecs = seconds_to_hours(wall_time)
    print("\nTotal GD training time: %d hours %d mins %d secs" %time_in_hmsecs)

    ## Save the results
    np.save("data/losses_04.npy", losses)
    np.save("data/nb_steps_04.npy", nb_steps)
    np.save("data/xis_04.npy", xis)
    np.save("data/init_xis_04.npy", init_xis)

    model = eqx.combine(params, static)
    eqx.tree_serialise_leaves("data/model_04.eqx", model)
    eqx.tree_serialise_leaves(data_folder+"model_04.eqx", model)       ## Make a seperate copy for backup

else:
    print("\nNo training, loading model and results from 'data' folder ...\n")

    losses = np.load("data/losses_04.npy")
    nb_steps = np.load("data/nb_steps_04.npy")
    xis = np.load("data/xis_04.npy")
    init_xis = np.load("data/init_xis_04.npy")

    model = eqx.combine(params, static)
    model = eqx.tree_deserialise_leaves("data/model_04.eqx", model)

















# %%

def test_model(model, batch):
    xi, X0, t_eval = batch

    # model = eqx.combine(params, static)
    X_hat, _, _, _ = model(X0, t_eval, xi)

    return X_hat, _


# e_key, traj_key = get_new_key(testing_key, num=2)
e_key, traj_key = get_new_key(time.time_ns(), num=2)

e = jax.random.randint(e_key, (1,), 0, nb_envs)[0]
# e = 0
traj = jax.random.randint(traj_key, (1,), 0, nb_trajs_per_env)[0]
# traj = 1061

# test_length = cutoff_length
test_length = nb_steps_per_traj
t_test = t_eval[:test_length]
X = data[e, traj, :test_length, :]

print("==  Begining testing ... ==")
print("    Environment id:", e)
print("    Trajectory id:", traj)
print("    Length of the original trajectories:", nb_steps_per_traj)
print("    Length of the training trajectories:", cutoff_length)
print("    Length of the testing trajectories:", test_length)


X_hat, _ = test_model(model, (xis[e], X[0,:], t_test))

fig, ax = plt.subplot_mosaic('AB;CC;DD;EF', figsize=(6*2, 3.5*4))

# mke = np.ceil(losses.shape[0]/100).astype(int)
mks = 2

ax['A'].plot(t_test, X[:, 0], c="deepskyblue", label=r"$\theta$ (GT)")
ax['A'].plot(t_test, X_hat[:, 0], "o", c="royalblue", label=r"$\theta$ (NODE)", markersize=mks)

ax['A'].plot(t_test, X[:, 1], c="violet", label=r"$\dot \theta$ (GT)")
ax['A'].plot(t_test, X_hat[:, 1], "x", c="purple", label=r"$\dot \theta$ (NODE)", markersize=mks)

ax['A'].set_xlabel("Time")
ax['A'].set_ylabel("State")
ax['A'].set_title("Trajectories")
ax['A'].legend()

ax['B'].plot(X[:, 0], X[:, 1], c="turquoise", label="GT")
ax['B'].plot(X_hat[:, 0], X_hat[:, 1], ".", c="teal", label="Neural ODE")
ax['B'].set_xlabel(r"$\theta$")
ax['B'].set_ylabel(r"$\dot \theta$")
ax['B'].set_title("Phase space")
ax['B'].legend()

mke = np.ceil(losses.shape[0]/100).astype(int)

ax['C'].plot(losses[:,0], label="Total", color="grey", linewidth=3, alpha=1.0)
ax['C'].plot(losses[:,1], "x-", markevery=mke, markersize=mks, label="Traj", color="grey", linewidth=1, alpha=0.5)
ax['C'].plot(losses[:,2], "o-", markevery=mke, markersize=mks, label="Discrim", color="grey", linewidth=1, alpha=0.5)
ax['C'].plot(losses[:,3], "^-", markevery=mke, markersize=3, label="Params", color="grey", linewidth=1, alpha=0.5)
ax['C'].set_xlabel("Epochs")
ax['C'].set_title("Loss Terms")
ax['C'].set_yscale('log')
ax['C'].legend()

ax['D'].plot(nb_steps, c="brown")
ax['D'].set_xlabel("Epochs")
ax['D'].set_title("Total Number of Steps Taken per Epoch (Proportional to NFEs)")
ax['D'].set_yscale('log')

xis_all = np.vstack([xis, init_xis])
eps = 0.1
xmin, xmax = xis_all[:,0].min()-eps, xis_all[:,0].max()+eps
ymin, ymax = xis_all[:,1].min()-eps, xis_all[:,1].max()+eps
colors = ['dodgerblue', 'r', 'b', 'g', 'm', 'c', 'y', 'orange', 'purple', 'brown']

ax['E'].scatter(init_xis[:,0], init_xis[:,1], s=30, c=colors[:nb_envs], marker='X')
ax['F'].scatter(xis[:,0], xis[:,1], s=50, c=colors[:nb_envs], marker='o')
for i, (x, y) in enumerate(init_xis[:, :2]):
    ax['E'].annotate(str(i), (x, y), fontsize=8)
for i, (x, y) in enumerate(xis[:, :2]):
    ax['F'].annotate(str(i), (x, y), fontsize=8)
ax['E'].set_title(r'Initial Contexts ($\xi^e$)')
ax['F'].set_title(r'Final Contexts ($\xi^e$)')
# ax['E'].set_xlim(xmin, xmax)
# ax['E'].set_ylim(ymin, ymax)
# ax['F'].set_xlim(xmin, xmax)
# ax['F'].set_ylim(ymin, ymax)

plt.suptitle(f"Results for env={e}, traj={traj}", fontsize=14)

plt.tight_layout()
plt.savefig(data_folder+"list_gan_node_04.png", dpi=100, bbox_inches='tight')
plt.show()

print("Testing finished. Script, data, figures, and models saved in:", data_folder)

# %% [markdown]

# # Preliminary results
# 

# # Conclusion

# %%

