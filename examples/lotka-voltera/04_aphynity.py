
#%%[markdown]
# # Apnynity framework for generalising the Lotka-Volterra systems

### Summary


#%%
import jax

print("\n############# Lotka-Volterra with Neural ODE #############\n")
print("Available devices:", jax.devices())

from jax import config
# config.update("jax_debug_nans", True)

import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.optimize

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
integrator = rk4_integrator
# integrator = dopri_integrator

## Optimiser hps
init_lr = 1e-4
decay_rate = 0.1

## Training hps
print_every = 100
nb_epochs = 1000
# batch_size = 128*10
batch_size = 1

cutoff = 0.1

#%%

def lotka_volterra(t, state, alpha, beta, delta, gamma):
    # x, y = state
    # dx_dt = alpha * x - beta * x * y
    # dy_dt = delta * x * y - gamma * y
    # return [dx_dt, dy_dt]

    ## A simpler system
    dx = state / (1 + state)
    return np.stack([dx[1], -dx[0]], axis=-1)

p = {"alpha": 1.5, "beta": 1.0, "delta": 3.0, "gamma": 1.0}
t_eval = np.linspace(0, 10, 1001)
initial_state = [1.0, 1.0]

solution = solve_ivp(lotka_volterra, (0,10), initial_state, args=(p["alpha"], p["beta"], p["delta"], p["gamma"]), t_eval=t_eval)

data = solution.y.T[None, None, ...]
cutoff_length = int(cutoff*data.shape[2])

print("data shape (first two are superfluous):", data.shape)

# %%

class Physics(eqx.Module):
    params: jnp.ndarray

    def __init__(self, in_size=2, out_size=2, key=None):
        self.params = jnp.abs(jax.random.normal(key, (4,)))
        # self.params = jax.random.normal(key, (4,))

    def __call__(self, x, t):
        # dx0 = x[0]*self.params[0] - x[0]*x[1]*self.params[1]
        # dx1 = x[0]*x[1]*self.params[2] - x[1]*self.params[3]
        # return jnp.array([dx0, dx1])

        ## A simpler system
        a, b, c, d = self.params
        dx = a*x / (b + c*x)
        return jnp.stack([dx[1], d*dx[0]], axis=-1)


class Augmentation(eqx.Module):
    layers: list

    def __init__(self, in_size=2, out_size=2, key=None):
        keys = get_new_key(key, num=3)
        self.layers = [eqx.nn.Linear(in_size, 32, key=keys[0]), jax.nn.tanh,
                        eqx.nn.Linear(32, 32, key=keys[1]), jax.nn.tanh,
                        eqx.nn.Linear(32, out_size, key=keys[2]) ]

    def __call__(self, x, t):
        # y = jnp.concatenate([jnp.broadcast_to(t, (1,)), x], axis=0)
        y = x
        for layer in self.layers:
            y = layer(y)
        return y

class Processor(eqx.Module):
    physics: Physics
    augmentation: Augmentation

    def __init__(self, in_size=2, out_size=2, key=None):
        keys = get_new_key(key, num=2)
        self.physics = Physics(in_size, out_size, key=keys[0])
        self.augmentation = Augmentation(in_size, out_size, key=keys[1])

    def __call__(self, x, t):
        return self.physics(x, t) + self.augmentation(x, t)

# %%

model_keys = get_new_key(SEED, num=2)

# model_p = Physics(in_size=2, out_size=2, key=model_keys[0])
# model_a = Augmentation(in_size=2, out_size=2, key=model_keys[1])


# params, static = eqx.partition((model_p, model_a), eqx.is_array)


# # params_flat, params_shapes, tree_def = flatten_pytree(params_a)

model = Processor(in_size=2, out_size=2, key=model_keys[0])
params, static = eqx.partition(model, eqx.is_array)






# %%


def p_norm(params):
    """ norm of the parameters`"""
    # params_flat_a, _, _ = flatten_pytree(params)
    # return jnp.linalg.norm(params_flat)

    # return jnp.array([jnp.max(jnp.abs(x)) for x in jax.tree_util.tree_leaves(params)]).max()
    # return jnp.array([jnp.sum(jnp.abs(x)) for x in jax.tree_util.tree_leaves(params)]).sum()
    return jnp.array([jnp.linalg.norm(x) for x in jax.tree_util.tree_leaves(params)]).sum()

def l2_norm(X, X_hat):
    total_loss = jnp.mean((X - X_hat)**2, axis=-1)   ## Norm of d-dimensional vectors
    return jnp.sum(total_loss) / (X.shape[-2])

# %%

# ### ==== Vanilla Gradient Descent optimisation ==== ####

# def loss_fn(params, static, batch):
#     print('\nCompiling function "loss_fn" ...\n')
#     X, t = batch

#     X_hat = integrator(params, static, X[0, 0, :], t, 1.4e-8, 1.4e-8, jnp.inf, jnp.inf, 50, "checkpointed")

#     term1 = l2_norm(X, X_hat)
#     term2 = p_norm(params.augmentation)

#     # return term1, (term1, term2)
#     return term1 + 1e-0*term2, (term1, term2)


# @partial(jax.jit, static_argnums=(1))
# def train_step(params, static, batch, opt_state):
#     print('\nCompiling function "train_step" ...\n')

#     (loss, (term1, term2)), grads  = jax.value_and_grad(loss_fn, has_aux=True)(params, static, batch)

#     updates, opt_state = opt.update(grads, opt_state)
#     params = optax.apply_updates(params, updates)

#     return params, opt_state, loss, term1, term2


# total_steps = nb_epochs

# # sched = optax.exponential_decay(init_lr, total_steps, decay_rate)
# # sched = optax.linear_schedule(init_lr, 0, total_steps, 0.25)
# sched = optax.piecewise_constant_schedule(init_value=init_lr,
#                 boundaries_and_scales={int(total_steps*0.25):0.5, 
#                                         int(total_steps*0.5):0.1,
#                                         int(total_steps*0.75):0.2})
# fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))

# start_time = time.time()


# print(f"\n\n=== Beginning Training ... ===")

# opt = optax.adam(sched)
# opt_state = opt.init(params)
# # cutoff_length = int(cutoff*data.shape[2])

# losses = []
# for epoch in range(nb_epochs):

#     nb_batches = 0
#     loss_sum = jnp.zeros(3)
#     for i in range(0, 1, batch_size):
#         batch = (data[0,i:i+batch_size,:cutoff_length,:], t_eval[:cutoff_length])
    
#         params, opt_state, loss, term1, term2 = train_step(params, static, batch, opt_state)

#         loss_sum += jnp.array([loss, term1, term2])
#         nb_batches += 1

#     loss_epoch = loss_sum/nb_batches
#     losses.append(loss_epoch)

#     if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
#         print(f"    Epoch: {epoch:-5d}      TotalLoss: {loss_epoch[0]:-.5f}      TrajLoss: {loss_epoch[1]:-.5f}      ParamsLoss: {loss_epoch[2]:-.5f}", flush=True)

# losses = jnp.vstack(losses)
# # ax = sbplot(losses, x_label='Epoch', y_label='L2', y_scale="log", title=f'Loss for environment {e}', ax=ax);
# ax = sbplot(losses, label=["Total", "Traj", "Params_a"], x_label='Epoch', y_label='L2', y_scale="log", title='Losses', ax=ax);
# plt.savefig(f"data/loss_simple.png", dpi=300, bbox_inches='tight')
# # plt.show()
# plt.legend()
# fig.canvas.draw()
# fig.canvas.flush_events()

# wall_time = time.time() - start_time
# time_in_hmsecs = seconds_to_hours(wall_time)
# print("\nTotal GD training time: %d hours %d mins %d secs" %time_in_hmsecs)

# params_backup = params

# %%



# ### TODO see the two lines below
# model = Processor(in_size=2, out_size=2, key=model_keys[0])
# params_backup, static = eqx.partition(model, eqx.is_array)

# params_flat, params_shapes, tree_def = flatten_pytree(params_backup)

# @partial(jax.jit, static_argnums=(1))
# def loss_fn_(params_flat, static, batch):
#     print('\nCompiling function "loss_fn" ...\n')
#     X, t = batch
#     params = unflatten_pytree(params_flat, params_shapes, tree_def)

#     X_hat = integrator(params, static, X[0, 0, :], t, 1.4e-8, 1.4e-8, jnp.inf, jnp.inf, 50, "bounded")

#     term1 = l2_norm(X, X_hat)
#     term2 = p_norm(params.augmentation)

#     return term1 + 1e-0*term2


# args=(static, (data[0,:,...], t_eval))

# def newton(x: np.ndarray, f: Callable, gf: Callable, hf: Callable, lr=0.01, lr_decr=0.999, maxiter=10, tol=0.001) -> Tuple[np.ndarray, List[np.ndarray], int]:

#     nit = 0

#     ### Put everything above in a jax fori lopp
#     def body_fn(carry):
#         _, x, lr, errors, nit = carry
#         gradient = gf(x, *args)
#         hessian = hf(x, *args)

#         x_new = x - lr*jnp.linalg.solve(hessian, gradient)
#         # x_new = x - lr * jnp.linalg.inv(hessian)@gradient

#         # jax.debug.print("x_new {}", x)

#         errors = errors.at[nit+1].set(jnp.linalg.norm(x_new - x))
#         return x, x_new, lr*lr_decr, errors, nit+1

#     def cond_fn(carry):
#         _, _, _, errors, nit = carry
#         return (errors[nit] >= tol) & (nit < maxiter)

#     errors = jnp.zeros((maxiter,))
#     errors = errors.at[0].set(2*tol)

#     _, x, lr, errors, nit = jax.lax.while_loop(cond_fn, body_fn, (x, x, lr, errors, nit))

#     return x, None, errors, nit

# gf = jax.jacrev(loss_fn_, argnums=0)
# hf = jax.jacrev(gf, argnums=0)
# params_flat, params_shapes, tree_def = flatten_pytree(params)

# start_time = time.time()

# params_flat, _, errors, nit = newton(params_flat, 
#                                      loss_fn_, 
#                                      gf, 
#                                      hf, 
#                                      tol=1e-10, 
#                                      maxiter=20, 
#                                      lr=1e-0, 
#                                      lr_decr=0.999)

# wall_time = time.time() - start_time
# time_in_hmsecs = seconds_to_hours(wall_time)
# print("\nTotal Newton training time: %d hours %d mins %d secs" %time_in_hmsecs)

# print("\nOptimisation result: nit, loss_val: ", nit, errors[nit-1])

# params = unflatten_pytree(params_flat, params_shapes, tree_def)

# %%

## For the first time, let's see how the method of multiplier performs !

eq_skip = 1

@jax.jit
def l2_norm_single(X):
    total_loss = jnp.mean(X**2, axis=-1)
    return jnp.sum(total_loss) / (X.shape[-2])

@jax.jit
def f(params):
    return p_norm(params.augmentation)

@partial(jax.jit, static_argnums=(1))
def h(params, static, batch):
    X, t = batch

    X_hat = integrator(params, static, X[0, 0, :], t, 1.4e-8, 1.4e-8, jnp.inf, jnp.inf, 50, "checkpointed")
    ret = X - X_hat

    # jax.debug.print("X_hat {}", X_hat[:, :])

    return ret.squeeze()[::eq_skip, :]        ### TODO Take values every 100 to reduce the number of equality constraints

@partial(jax.jit, static_argnums=(1))
def L(params, static, batch, lamb, rho):
    return f(params) + 0.5*rho*jnp.sum((h(params, static, batch) + lamb/rho)**2)

@partial(jax.jit, static_argnums=(1))
def inner_train_step(params, static, batch, lamb, rho, tau1, tau2):
    loss, grads  = jax.value_and_grad(L)(params, static, batch, lamb, rho)
    params = jax.tree_util.tree_map(lambda x, y: x - tau1*y, params, grads)

    return params, loss

nb_constraints = t_eval[:cutoff_length:eq_skip].shape[0]
ones_vec = jnp.ones((nb_constraints, 2))

lamb_min, lamb_max = -10*ones_vec, 10*ones_vec
gamma = 0.95

tau1 = 1e-4 ## Actual learning rate for this method
tau2 = 1e-3 ## Multiplicative factor used as tau in the augmented method

lamb = 5.*ones_vec
rho = 100.      ### Bigin with a huge rho, and it will be reduced naturally.

nb_iter_out = 25
nb_iter_in = 100
tol = 1e-10

### TODO see the two lines below
model = Processor(in_size=2, out_size=2, key=model_keys[0])
params_backup, static = eqx.partition(model, eqx.is_array)

params = params_backup
batch = (data[0,:,:cutoff_length,:], t_eval[:cutoff_length])

metrics = []
iter_count = 0

start_time = time.time()

for k in range(nb_iter_out):

    params_old = params

    for i in range(nb_iter_in):
        params_new, loss = inner_train_step(params, static, batch, lamb, rho, tau1, tau2)
        
        params_diff = jax.tree_util.tree_map(lambda x, y: x - y, params_new, params_old)
        if f(params_diff) < tol:
            break

        iter_count += 1
        params = params_new
        # print("Values:", params.augmentation.layers[0].weight, params.augmentation.layers[0].bias)
        # test_val = h(params, static, batch)
        # print(test_val)
        metrics.append([l2_norm_single(h(params, static, batch)), f(params)])

    lamb = jnp.clip(lamb + rho*h(params_new, static, batch), lamb_min, lamb_max)

    norm_h_old = jnp.linalg.norm(h(params_old, static, batch))
    norm_h = jnp.linalg.norm(h(params_new, static, batch))

    if k==0 or norm_h_old < tau2*norm_h:
        rho = 2*rho
    else:
        rho = gamma*rho

    params_diff = jax.tree_util.tree_map(lambda x, y: x - y, params_new, params_old)
    if f(params_diff) < tol:
        break

    print(f"Iter: {k:-3d}   ParamsLoss={metrics[-1][1]:.8f}   TrajLoss={metrics[-1][0]:.8f}   rho={rho:.6f}    max lambda={lamb.max():.6f}")

print(f"\nTotal number of iterations to achieve a tol of {tol} is: {iter_count}")


wall_time = time.time() - start_time
time_in_hmsecs = seconds_to_hours(wall_time)
print("\nTotal Multipliers training time: %d hours %d mins %d secs" %time_in_hmsecs)

ax = sbplot(metrics, label=["Traj", "Params"], x_label='Epoch', y_label='L2', y_scale="log", title='Losses Aug. Method Multipliers');


# %%

def test_model(params, static, batch):
    X0, t = batch

    X_hat = integrator(params, static, X0, t, 1.4e-8, 1.4e-8, jnp.inf, jnp.inf, 50, "checkpointed")

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

model = eqx.combine(params, static)

eqx.tree_serialise_leaves("data/model_02.eqx", model)
# model = eqx.tree_deserialise_leaves("data/model_01.eqx", model)

# %% [markdown]

# # Preliminary results


# # Conclusion
# 
# 


# %%
