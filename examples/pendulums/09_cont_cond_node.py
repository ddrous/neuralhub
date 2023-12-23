
#%%[markdown]
# # Continuous Conditional framework for generalising the Simple Pendulum

### Summary


#%%

# import random
import jax

# from jax import config
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_platform_name", "cpu")

print("\n############# Continuous Conditional Neural ODEs for Simple Pendulum  #############\n")
print("Jax version:", jax.__version__)
print("Available devices:", jax.devices())

import jax.numpy as jnp

import numpy as np
np.set_printoptions(suppress=True)

import equinox as eqx
import diffrax

import matplotlib.pyplot as plt

from graphpint.utils import *
from graphpint.integrators import *

import pandas as pd
import seaborn as sns

import optax
from functools import partial

import os
import time

#%%

SEED = 27

## Integrator hps
# integrator = rk4_integrator

## Optimiser hps
init_lr = 3e-2

## Training hps
print_every = 100
nb_epochs = 1000
batch_size = 128*8

cutoff = 0.4
context_size = 20

train = True

#%%

# - make a new folder inside 'data' whose name is the current time
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

    def __init__(self, data_size, width_size, depth, context_size, key=None):
        keys = generate_new_keys(key, num=3)
        self.layers = [eqx.nn.Linear(data_size+context_size, width_size, key=keys[0]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, width_size, key=keys[1]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, data_size, key=keys[2])]

    def __call__(self, t, x, context):
        y = jnp.concatenate([x, context], axis=0)
        for layer in self.layers:
            y = layer(y)
        return y


# class Augmentation(eqx.Module):
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
#                         eqx.nn.Linear(width_size, width_size, key=keys[7]), jax.nn.softplus,
#                         eqx.nn.Linear(width_size, width_size, key=keys[8]), jax.nn.softplus,
#                         eqx.nn.Linear(width_size, data_size, key=keys[9])]


#     def __call__(self, t, x, context):

#         y = x
#         context = context
#         for i in range(len(self.layers_data)):
#             y = self.layers_data[i](y)
#             context = self.layers_context[i](context)

#         y = jnp.concatenate([y, context, y*context], axis=0)
#         for layer in self.layers_shared:
#             y = layer(y)
#         return y



class Processor(eqx.Module):
    physics: Physics
    envnet: Augmentation
    context: jnp.ndarray

    def __init__(self, data_size, width_size, depth, context_size, key=None):
        keys = generate_new_keys(key, num=5)
        self.physics = Physics(key=keys[0])
        # self.envnet = Augmentation(data_size, data_size, width_size, depth, context_size, key=keys[1])
        self.envnet = Augmentation(data_size, width_size, depth, context_size, key=keys[1])
        self.context = jax.random.normal(keys[2], (context_size,))

    def __call__(self, t, x, args):
        return self.physics(t, x) + self.envnet(t, x, self.context)
        # return self.physics(t, x) + self.envnet(t, x, jnp.zeros_like(self.context)) 


class NeuralODE(eqx.Module):
    processor: Processor

    def __init__(self, data_size, width_size, depth, context_size, key=None):
        self.processor = Processor(data_size, width_size, depth, context_size, key=key)

    def __call__(self, x0, t_eval):
        solution = diffrax.diffeqsolve(
                    diffrax.ODETerm(self.processor),
                    diffrax.Tsit5(),
                    args=None,
                    t0=t_eval[0],
                    t1=t_eval[-1],
                    dt0=t_eval[1] - t_eval[0],
                    y0=x0,
                    stepsize_controller=diffrax.PIDController(rtol=1e-1, atol=1e-2),
                    saveat=diffrax.SaveAt(ts=t_eval),
                    max_steps=4096*10,
                )
        return solution.ys, solution.stats["num_steps"]

        # rhs = lambda x, t: self.processor(t, x, None)
        # X_hat = integrator(rhs, x0, t_eval, None, None, None, None, None, None)
        # return X_hat, t_eval.size


# %%

model_key, training_key, testing_key = get_new_key(SEED, num=3)

model = NeuralODE(data_size=2, 
                width_size=16*2, 
                depth=3, 
                context_size=context_size, 
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


def make_training_batch(batch_id, env, data, cutoff_length, batch_size, key):
    """ Make a batch """
    traj_start, traj_end = batch_id*batch_size, (batch_id+1)*batch_size
    return (data[env, traj_start:traj_end, :cutoff_length, :], t_eval[:cutoff_length]), key




# ## Main loss function
def loss_fn(params, static, batch):
    # print('\nCompiling function "loss_fn" ...\n')
    Xs, t_eval = batch
    print("Shapes of elements in a batch:", Xs.shape, t_eval.shape, "\n")

    model = eqx.combine(params, static)

    Xs_hat, nb_steps = jax.vmap(model, in_axes=(0, None))(Xs[:, 0, :], t_eval)

    term1 = l2_norm(Xs, Xs_hat)
    term2 = params_norm(params.processor.envnet)
    loss_val = term1 + 1e-2*term2

    return loss_val, (jnp.sum(nb_steps), term1, term2)


@partial(jax.jit, static_argnums=(1))
def train_step(params, static, batch, opt_state):
    print('\nCompiling function "train_step" ...\n')

    (loss, aux_data), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, static, batch)

    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss, aux_data


@partial(jax.jit, static_argnums=(1))
def train_step_frozen_envnet(params, static, batch, opt_state):
    print('\nCompiling function "train_step" ...\n')

    (loss, aux_data), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, static, batch)

    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss, aux_data



if train == True:

    nb_train_steps_per_epoch = nb_trajs_per_env // batch_size
    total_steps = nb_epochs * nb_train_steps_per_epoch

    print(f"\n\n=== Beginning training neural ODE ... ===")
    print(f"    Number of examples in a batch: {batch_size}")
    print(f"    Number of train steps per epoch: {nb_train_steps_per_epoch}")
    print(f"    Number of training epochs: {nb_epochs}")
    print(f"    Total number of training steps: {total_steps}")

    start_time = time.time()

    all_losses = []
    all_nb_steps = []
    contexts = []

    for e in range(nb_envs):
        losses = []
        nb_steps = []

        sched = optax.piecewise_constant_schedule(init_value=init_lr,
                        boundaries_and_scales={int(total_steps*0.25):0.5, 
                                                int(total_steps*0.5):0.5,
                                                int(total_steps*0.75):0.5})

        opt = optax.adam(sched)
        opt_state = opt.init(params)

        _, batch_key = get_new_key(training_key, num=2)

        for epoch in range(nb_epochs):
            nb_batches = 0
            loss_sum = jnp.zeros(3)
            nb_steps_eph = 0

            for i in range(nb_train_steps_per_epoch):
                batch, batch_key = make_training_batch(i, e, data, cutoff_length, batch_size, batch_key)

                if e==0:
                    params, opt_state, loss, (nb_steps_val, term1, term2) = train_step(params, static, batch, opt_state)
                else:
                    params, opt_state, loss, (nb_steps_val, term1, term2) = train_step_frozen_envnet(params, static, batch, opt_state)

                loss_sum += jnp.array([loss, term1, term2])
                nb_steps_eph += nb_steps_val
                nb_batches += 1

            loss_epoch = loss_sum/nb_batches
            losses.append(loss_epoch)
            nb_steps.append(nb_steps_eph)

            if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
                print(f"    Epoch: {epoch:-5d}      TotalLoss: {loss_epoch[0]:-.8f}     Traj: {loss_epoch[1]:-.8f}     Params: {loss_epoch[2]:-.8f}", flush=True)

        losses = jnp.vstack(losses)
        nb_steps = jnp.array(nb_steps)

        all_losses.append(losses[None,...])
        all_nb_steps.append(nb_steps[None,...])
        contexts.append(params.processor.context)

    all_losses = jnp.vstack(all_losses)
    all_nb_steps = jnp.vstack(all_nb_steps)
    contexts = jnp.vstack(contexts)

    wall_time = time.time() - start_time
    time_in_hmsecs = seconds_to_hours(wall_time)
    print("\nTotal gradient descent training time: %d hours %d mins %d secs" %time_in_hmsecs)

    np.save("data/losses_09.npy", losses)
    np.save("data/nb_steps_09.npy", nb_steps)
    np.save("data/contexts_09.npy", contexts)

    model = eqx.combine(params, static)
    eqx.tree_serialise_leaves("data/model_09.eqx", model)
    eqx.tree_serialise_leaves(data_folder+"model_09.eqx", model)       ## Make a seperate copy for backup

else:
    print("\nNo training, loading model and results from 'data' folder ...\n")

    losses = np.load("data/losses_09.npy")
    nb_steps = np.load("data/nb_steps_09.npy")
    contexts = np.load("data/contexts_09.npy")

    model = eqx.combine(params, static)
    model = eqx.tree_deserialise_leaves("data/model_09.eqx", model)

















# %%

def test_model(model, batch, context):
    X, t_eval = batch

    model = eqx.tree_at(lambda m:m.processor.context, model, context)
    X_hat, _ = model(X[0, :], t_eval)

    return X_hat


e_key, traj_key = get_new_key(time.time_ns(), num=2)

e = jax.random.randint(e_key, (1,), 0, nb_envs)[0]
e=1
traj = jax.random.randint(traj_key, (1,), 0, nb_trajs_per_env)[0]

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

X_hat = test_model(model, (X, t_test), contexts[e])

fig, ax = plt.subplot_mosaic('AB;CC;DD;EF', figsize=(6*2, 3.5*4))

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

losses = all_losses[0]      ## TODO - fix this

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

xis = contexts

eps = 0.1
colors = ['dodgerblue', 'r', 'b', 'g', 'm', 'c', 'y', 'orange', 'purple', 'brown']
colors = colors*(nb_envs)

ax['F'].scatter(xis[:,0], xis[:,1], s=50, c=colors[:nb_envs], marker='o')
for i, (x, y) in enumerate(xis[:, :2]):
    ax['F'].annotate(str(i), (x, y), fontsize=8)
ax['F'].set_title(r'Final Contexts ($\xi^e$)')

plt.suptitle(f"Results for env={e}, traj={traj}", fontsize=14)

plt.tight_layout()
plt.savefig(data_folder+"cont_cond_node_09.png", dpi=100, bbox_inches='tight')
plt.show()

print("Testing finished. Script, data, figures, and models saved in:", data_folder)

# %% [markdown]

# # Preliminary results
# 

# # Conclusion
