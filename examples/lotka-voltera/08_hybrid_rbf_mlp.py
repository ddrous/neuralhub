
#%%[markdown]
# # RBF Network for generalising the Lotka-Volterra systems

### Summary


#%%
import jax

# from jax import config
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_platform_name", "cpu")

print("\n############# Lotka-Volterra with Neural ODE #############\n")
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

from graphpint.utils import *
from graphpint.integrators import *

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
init_lr = 1e-5
decay_rate = 0.1

## Training hps
print_every = 100
nb_epochs = 5000
# batch_size = 128*10
batch_size = 128

cutoff = 0.1

#%%

# def lotka_volterra(t, state, alpha, beta, delta, gamma):
#     x, y = state
#     dx_dt = alpha * x - beta * x * y
#     dy_dt = delta * x * y - gamma * y
#     return [dx_dt, dy_dt]

# p = {"alpha": 1.5, "beta": 1.0, "delta": 3.0, "gamma": 1.0}
# t_eval = np.linspace(0, 10, 1001)
# initial_state = [1.0, 1.0]

# solution = solve_ivp(lotka_volterra, (0,10), initial_state, args=(p["alpha"], p["beta"], p["delta"], p["gamma"]), t_eval=t_eval)
# # data = solution.y.T[None, None, ...]

dataset = np.load('./data/lotka_volterra_diffrax.npz')
data, t_eval = dataset['ys'][None,...], dataset['ts']

cutoff_length = int(cutoff*data.shape[2])

print("data shape (first one is superfluous):", data.shape)

# %%

class Physics(eqx.Module):
    params: jnp.ndarray

    def __init__(self, in_size=2, out_size=2, key=None):
        self.params = jnp.abs(jax.random.normal(key, (4,)))

    def __call__(self, x, t):
        dx0 = x[0]*self.params[0] - x[0]*x[1]*self.params[1]
        dx1 = x[0]*x[1]*self.params[2] - x[1]*self.params[3]
        return jnp.array([dx0, dx1])

def distance(node1, node2):
    diff = node1 - node2
    return jnp.sum(diff*diff)      ## Squared distance ! No problem for Gaussain RBF
    # return jnp.linalg.norm(node1 - node2)       ## Carefull: not differentiable at 0

def gaussian_rbf(r_squared, shape):
    return jnp.exp(-r_squared / ( 2 * shape**2))

def gaussian_rbf_full(node1, node2, shape):
    return gaussian_rbf(distance(node1, node2), shape)

# class Augmentation(eqx.Module):
#     centers: jnp.ndarray
#     shapes: jnp.ndarray     ## Widths for the gaussian RBF network
#     weights: jnp.ndarray

#     def __init__(self, in_size=2, nb_centers=10, out_size=2, key=None):
#         keys = get_new_key(key, num=3)

#         self.centers = jax.random.uniform(keys[0], (nb_centers, in_size), minval=0., maxval=3.)
#         self.shapes = jax.random.uniform(keys[1], (nb_centers,), minval=0.1, maxval=10.)
#         self.weights = eqx.nn.Linear(nb_centers, out_size, key=keys[2])

#     def __call__(self, x, t):
#         # y = jnp.concatenate([jnp.broadcast_to(t, (1,)), x], axis=0)
#         dists_squared = jax.vmap(distance, in_axes=(0, None))(self.centers, x)
#         activations = jax.vmap(gaussian_rbf, in_axes=(0, 0))(dists_squared, self.shapes)
#         return self.weights(activations)


class Augmentation(eqx.Module):
    layers: list

    def __init__(self, in_size=2, nb_centers=10, out_size=2, key=None):
        keys = get_new_key(key, num=3)
        self.layers = [eqx.nn.Linear(in_size, 16, key=keys[0]), jax.nn.tanh,
                        eqx.nn.Linear(16, 16, key=keys[1]), jax.nn.softplus,
                        eqx.nn.Linear(16, out_size, key=keys[2]) ]

    def __call__(self, x, t):
        # y = jnp.concatenate([jnp.broadcast_to(t, (1,)), x], axis=0)
        y = x
        for layer in self.layers:
            y = layer(y)
        return y



class Encoder(eqx.Module):
    centers: jnp.ndarray
    shapes: jnp.ndarray     ## Widths for the gaussian RBF network
    weights: eqx.nn.Linear      ## Weights for the decoder to average its outputs

    def __init__(self, in_size=2, nb_centers=10, key=None):
        keys = get_new_key(key, num=3)

        self.centers = jax.random.uniform(keys[0], (nb_centers, in_size), minval=0., maxval=3.)
        self.shapes = jax.random.uniform(keys[1], (nb_centers,), minval=0.1, maxval=10.)

        self.weights = eqx.nn.Linear(nb_centers, in_size, key=keys[2])

    def define_mat(self):
        rows_func = jax.vmap(gaussian_rbf_full, in_axes=(0, None, 0), out_axes=(0))
        mat_func = jax.vmap(rows_func, in_axes=(None, 0, None), out_axes=(0))
        return mat_func(self.centers, self.centers, self.shapes)

    def __call__(self, x):  ## Encode
        zeros = jnp.zeros((self.centers.shape[0]-x.shape[0], ))
        y = jnp.concatenate([x, zeros], axis=0)
        ## Solve the linear system for find the weights
        return jnp.linalg.solve(self.define_mat(), y)

    def decode(self, lamb):
        x_full = self.define_mat()@lamb
        # return self.weights(x_full)
        return x_full

class NeuralODE(eqx.Module):
    encoder: Encoder
    physics: Physics
    augmentation: Augmentation

    def __init__(self, in_size=2, nb_centers=10, out_size=2, key=None):
        keys = get_new_key(key, num=2)
        self.encoder = Encoder(in_size, nb_centers, key=keys[0])
        self.physics = Physics(in_size, out_size, key=keys[0])
        self.augmentation = Augmentation(nb_centers, nb_centers, nb_centers, key=keys[1])

    def __call__(self, x, t):
        # return self.physics(x, t) + self.augmentation(x, t)
        
        lamb0 = self.encoder(x)
        
        X_hat = diffrax.diffeqsolve(
                    diffrax.ODETerm(lambda t, y, args: self.augmentation(y, t)),
                    diffrax.Tsit5(),
                    t0=t[0],
                    t1=t[-1],
                    dt0=t[1] - t[0],
                    y0=lamb0,
                    stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
                    saveat=diffrax.SaveAt(ts=t),
                    max_steps=4096*1,
                ).ys
        
        return jax.vmap(self.encoder.decode)(X_hat)[:, :x.shape[0]]     ## TODO the decoder needs to be more complex


# %%

model_keys = get_new_key(SEED, num=2)

model = NeuralODE(in_size=2, nb_centers=5, out_size=2, key=model_keys[0])
params, static = eqx.partition(model, eqx.is_array)






# %%


def p_norm(params):
    """ norm of the parameters`"""
    return jnp.array([jnp.linalg.norm(x) for x in jax.tree_util.tree_leaves(params)]).sum()

def l2_norm(X, X_hat):
    total_loss = jnp.mean((X - X_hat)**2, axis=-1)   ## Norm of d-dimensional vectors
    return jnp.sum(total_loss) / (X.shape[-2] * X.shape[-3])

# %%

### ==== Vanilla Gradient Descent optimisation ==== ####

def loss_fn(params, static, batch):
    print('\nCompiling function "loss_fn" ...\n')
    X, t = batch

    model = eqx.combine(params, static)

    X_hat = jax.vmap(model, in_axes=(0, None))(X[:, 0, :], t)

    print("X shape:", X.shape, "X_hat shape:", X_hat.shape, "t shape:", t.shape)

    term1 = l2_norm(X, X_hat)
    term2 = p_norm(params.augmentation)

    return term1, (term1, term2)
    # return term1 + 1e-3*term2, (term1, term2)


@partial(jax.jit, static_argnums=(1))
def train_step(params, static, batch, opt_state):
    print('\nCompiling function "train_step" ...\n')

    (loss, (term1, term2)), grads  = jax.value_and_grad(loss_fn, has_aux=True)(params, static, batch)

    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss, term1, term2


total_steps = nb_epochs

# sched = optax.exponential_decay(init_lr, total_steps, decay_rate)
# sched = optax.linear_schedule(init_lr, 0, total_steps, 0.25)
sched = optax.piecewise_constant_schedule(init_value=init_lr,
                boundaries_and_scales={int(total_steps*0.25):0.5, 
                                        int(total_steps*0.5):0.15,
                                        int(total_steps*0.75):0.5})
fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))

start_time = time.time()


print(f"\n\n=== Beginning Training ... ===")

opt = optax.adam(sched)
opt_state = opt.init(params)
# cutoff_length = int(cutoff*data.shape[2])

losses = []
for epoch in range(nb_epochs):

    nb_batches = 0
    loss_sum = jnp.zeros(3)
    for i in range(0, 1, batch_size):
        batch = (data[0,i:i+batch_size,:cutoff_length,:], t_eval[:cutoff_length])
    
        params, opt_state, loss, term1, term2 = train_step(params, static, batch, opt_state)

        loss_sum += jnp.array([loss, term1, term2])
        nb_batches += 1

    loss_epoch = loss_sum/nb_batches
    losses.append(loss_epoch)

    if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
        print(f"    Epoch: {epoch:-5d}      TotalLoss: {loss_epoch[0]:-.5f}      TrajLoss: {loss_epoch[1]:-.5f}      ParamsLoss: {loss_epoch[2]:-.5f}", flush=True)

losses = jnp.vstack(losses)
# ax = sbplot(losses, x_label='Epoch', y_label='L2', y_scale="log", title=f'Loss for environment {e}', ax=ax);
ax = sbplot(losses, label=["Total", "Traj", "Params_a"], x_label='Epoch', y_label='L2', y_scale="log", title='Losses', ax=ax);
plt.savefig(f"data/loss_simple.png", dpi=300, bbox_inches='tight')
# plt.show()
plt.legend()
fig.canvas.draw()
fig.canvas.flush_events()

wall_time = time.time() - start_time
time_in_hmsecs = seconds_to_hours(wall_time)
print("\nTotal GD training time: %d hours %d mins %d secs" %time_in_hmsecs)


# %%

def test_model(params, static, batch):
    X0, t = batch

    model = eqx.combine(params, static)
    X_hat = model(X0, t)

    return X_hat


i = np.random.randint(0, 1)

X = data[0, i, :, :]
t = t_eval

X_hat = test_model(params, static, (X[0,:], t))

# print("X_hat values", X_hat)

ax = sbplot(X_hat[:,0], X_hat[:,1], x_label='Preys', y_label='Predators', label=f'Pred', title=f'Phase space, traj {i}')
ax = sbplot(X[:,0], X[:,1], "--", lw=1, label=f'True', ax=ax)

# plt.savefig(f"data/coda_test_env{e}_traj{i}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"data/test_simple.png", dpi=300, bbox_inches='tight')


#%% 

# model = eqx.combine(params, static)

# eqx.tree_serialise_leaves("data/model_02.eqx", model)
# model = eqx.tree_deserialise_leaves("data/model_01.eqx", model)

# %% [markdown]

# # Preliminary results


# # Conclusion
