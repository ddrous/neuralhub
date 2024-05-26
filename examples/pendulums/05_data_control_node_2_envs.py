
#%%[markdown]
# # Data-controlled Neural ODE framework for generalising the Simple Pendulum system
# - https://arxiv.org/abs/2002.08071

### Summary


#%%
import jax

# from jax import config
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_platform_name", "cpu")

print("\n############# Data-Controlled Pendulums with Neural ODE #############\n")
print("Available devices:", jax.devices())

import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.optimize

import diffrax

import numpy as np
np.set_printoptions(suppress=True)
from scipy.integrate import solve_ivp

import equinox as eqx

import matplotlib.pyplot as plt

from neuralhub.utils import *
from neuralhub.integrators import *

import optax
from functools import partial
import time
from typing import List, Tuple, Callable


#%%

SEED = 27
# SEED = np.random.randint(0, 1000)

## Integrator hps
# integrator = rk4_integrator
# integrator = dopri_integrator
integrator = dopri_integrator_diff

## Optimiser hps
init_lr = 3e-3
decay_rate = 0.1

## Training hps
print_every = 100
nb_epochs = 3000
batch_size = 128*1

cutoff = 0.5

train = False           ### Implement this thing !!! It works on Isambard

#%%

dataset = np.load('./data/simple_pendulum_big_2_envs.npz')
data, t_eval = dataset['X'], dataset['t']

cutoff_length = int(cutoff*data.shape[2])

print("data shape (first two are superfluous):", data.shape)

# %%

class Physics(eqx.Module):
    params: jnp.ndarray

    def __init__(self, key=None):
        keys = get_new_key(key, num=2)
        self.params = jnp.concatenate([jax.random.uniform(keys[0], (1,), minval=0.25, maxval=1.75),
                                       jax.random.uniform(keys[1], (1,), minval=2, maxval=10)])

    def __call__(self, t, x):
        L, g = self.params
        theta, theta_dot = x
        theta_ddot = -(g / L) * jnp.sin(theta)
        return jnp.array([theta_dot, theta_ddot])

class Augmentation(eqx.Module):
    layers: list

    def __init__(self, data_size, width_size, depth, context_size, key=None):
        keys = get_new_key(key, num=3)
        self.layers = [eqx.nn.Linear(data_size+context_size, width_size, key=keys[0]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, width_size, key=keys[1]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, data_size, key=keys[2])]
    def __call__(self, t, x, context):
        y = jnp.concatenate([x, context], axis=0)
        for layer in self.layers:
            y = layer(y)
        return y

class Processor(eqx.Module):
    shared: Physics
    env: Augmentation

    def __init__(self, data_size, width_size, depth, context_size, key=None):
        keys = get_new_key(key, num=2)
        self.shared = Physics(key=keys[0])
        self.env = Augmentation(data_size, width_size, depth, context_size, key=keys[1])

    def __call__(self, t, x, context):
        return self.shared(t, x) + self.env(t, x, context)

# %%

model_keys = get_new_key(SEED, num=2)

model = Processor(data_size=2, width_size=16, depth=3, context_size=1, key=model_keys[0])
params, static = eqx.partition(model, eqx.is_array)






# %%


def params_norm(params):
    """ norm of the parameters`"""
    return jnp.array([jnp.linalg.norm(x) for x in jax.tree_util.tree_leaves(params)]).sum()

def l2_norm(X, X_hat):
    total_loss = jnp.mean((X - X_hat)**2, axis=-1)   ## Norm of d-dimensional vectors
    return jnp.sum(total_loss) / (X.shape[-2] * X.shape[-3])

# %%

def loss_fn(params, static, batch):
    print('\nCompiling function "loss_fn" ...\n')
    X, xi, t = batch
    print("Shapes of elements in a batch:", X.shape, xi.shape, t.shape, "\n")

    def solve_ivp(x0, xi):
        sol =  diffrax.diffeqsolve(
            diffrax.ODETerm(eqx.combine(params, static)),
            diffrax.Tsit5(),
            args=xi,
            t0=t[0],
            t1=t[-1],
            dt0=t[1] - t[0],
            y0=x0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=t),
            max_steps=4096*1,
        )
        return sol.ys, sol.stats["num_steps"]

    X_hat, nb_steps = jax.vmap(solve_ivp, in_axes=(0, 0))(X[:, 0, :], xi)

    # t_batched = jnp.broadcast_to(t, (X.shape[0], t.shape[0]))
    # batched_integrator = jax.vmap(integrator, in_axes=(None, None, 0, 0, None, None, None, None, None, None))
    # X_hat = batched_integrator(params, static, X[:, 0, :], t_batched, 1.4e-8, 1.4e-8, jnp.inf, jnp.inf, 50, "checkpointed")

    term1 = l2_norm(X, X_hat)
    term2 = params_norm(params.env)

    # return term1, (term1, term2)
    return term1 + 1e-3*term2, (jnp.sum(nb_steps), term1, term2)


@partial(jax.jit, static_argnums=(1))
def train_step(params, static, batch, opt_state):
    print('\nCompiling function "train_step" ...\n')

    (loss, aux_data), grads  = jax.value_and_grad(loss_fn, has_aux=True)(params, static, batch)

    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss, aux_data


nb_steps_per_epoch = data.shape[1]//batch_size
total_steps = nb_epochs * nb_steps_per_epoch

sched = optax.piecewise_constant_schedule(init_value=init_lr,
                boundaries_and_scales={int(total_steps*0.25):0.5, 
                                        int(total_steps*0.5):0.5,
                                        int(total_steps*0.75):0.5})

opt = optax.adam(sched)
opt_state = opt.init(params)


if train == True:

    print(f"\n\n=== Beginning Training ... ===")

    start_time = time.time()

    nb_steps = []
    losses = []
    for epoch in range(nb_epochs):

        nb_batches = 0
        loss_sum = jnp.zeros(3)
        nb_steps_eph = 0

        for i in range(nb_steps_per_epoch):
            e = i%2
            xi = jnp.ones((batch_size, 1))*(-1) if e==0 else jnp.ones((batch_size, 1))*(1)
            batch = (data[e,i*batch_size:(i+1)*batch_size,:cutoff_length,:], xi, t_eval[:cutoff_length])
        
            params, opt_state, loss, (nb_steps_val, term1, term2) = train_step(params, static, batch, opt_state)

            loss_sum += jnp.array([loss, term1, term2])
            nb_steps_eph += nb_steps_val
            nb_batches += 1

        loss_epoch = loss_sum/nb_batches
        losses.append(loss_epoch)
        nb_steps.append(nb_steps_eph)

        if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
            print(f"    Epoch: {epoch:-5d}      TotalLoss: {loss_epoch[0]:-.8f}      TrajLoss: {loss_epoch[1]:-.8f}      ParamsLoss: {loss_epoch[2]:-.8f}", flush=True)

    losses = jnp.vstack(losses)
    nb_steps = jnp.array(nb_steps)

    wall_time = time.time() - start_time
    time_in_hmsecs = seconds_to_hours(wall_time)
    print("\nTotal GD training time: %d hours %d mins %d secs" %time_in_hmsecs)

    ## Save the results
    np.save("data/losses_05.npy", losses)
    np.save("data/nb_steps_05.npy", nb_steps)
    model = eqx.combine(params, static)
    eqx.tree_serialise_leaves("data/model_05.eqx", model)

else:
    print("\nNo training, loading data from previous run ...\n")

    ## Load the results
    losses = np.load("data/losses_05.npy")
    nb_steps = np.load("data/nb_steps_05.npy")
    model = eqx.combine(params, static)
    model = eqx.tree_deserialise_leaves("data/model_05.eqx", model)


# %%

def test_model(model, batch):
    X0, xi, t = batch

    # X_hat = integrator(params, static, X0, t, 1.4e-8, 1.4e-8, jnp.inf, jnp.inf, 50, "checkpointed")
    # return X_hat

    return diffrax.diffeqsolve(
        diffrax.ODETerm(model),
        diffrax.Tsit5(),
        args=xi,
        t0=t[0],
        t1=t[-1],
        dt0=t[1] - t[0],
        y0=X0,
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        saveat=diffrax.SaveAt(ts=t),
        max_steps=4096*1,
    ).ys

e = np.random.randint(0, 2)
traj = np.random.randint(0, data.shape[1])

# e = 0
# traj = 168

X = data[e, traj, :, :]
t_test = t_eval
xi = jnp.array([-1]) if e==0 else jnp.array([1])

X_hat = test_model(model, (X[0,:], xi, t_test))

fig, ax = plt.subplot_mosaic('AB;CC;DD', figsize=(6*2, 3.5*3))

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
ax['C'].plot(losses[:,2], "o-", markevery=mke, markersize=mks, label="Params", color="grey", linewidth=1, alpha=0.5)
ax['C'].set_xlabel("Epochs")
ax['C'].set_title("Loss Terms")
ax['C'].set_yscale('log')
ax['C'].legend()

ax['D'].plot(nb_steps, c="brown")
ax['D'].set_xlabel("Epochs")
ax['D'].set_title("Total Number of Steps Taken per Epoch (Proportional to NFEs)")
ax['D'].set_yscale('log')

plt.suptitle(f"Results for env={e}, traj={traj}", fontsize=14)

plt.tight_layout()
plt.savefig("data/data_control_node.png", dpi=300, bbox_inches='tight')
plt.show()

print("Testing finished. Results saved in 'data' folder.\n")

