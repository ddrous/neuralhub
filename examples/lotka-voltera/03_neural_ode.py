
#%%[markdown]
# # Neural ODE framework for generalising the Lotka-Volterra systems

### Summary
# - Comapare this with: https://gist.github.com/ChrisRackauckas/a531030dc7ea5c96179c0f5f25de9979


#%%
import jax

print("\n############# Lotka-Volterra with Neural ODE #############\n")
print("Available devices:", jax.devices())

from jax import config

import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.optimize

import numpy as np
from scipy.integrate import solve_ivp

import equinox as eqx

import matplotlib.pyplot as plt

from graphpint.utils import *
from graphpint.integrators import *

import optax
from functools import partial
import time

#%%

SEED = 27

## Integrator hps
integrator = rk4_integrator
# integrator = dopri_integrator

## Optimiser hps
init_lr = 1e-3
decay_rate = 0.9

## Training hps
print_every = 100
nb_epochs = 5000
# batch_size = 128*10
batch_size = 1


#%%

def lotka_volterra(t, state, alpha, beta, delta, gamma):
    x, y = state
    dx_dt = alpha * x - beta * x * y
    dy_dt = delta * x * y - gamma * y
    return [dx_dt, dy_dt]

p = {"alpha": 1.5, "beta": 1.0, "gamma": 1.0, "delta": 3.0}
t_eval = np.linspace(0, 10, 1001)
initial_state = [1.0, 1.0]

solution = solve_ivp(lotka_volterra, (0,10), initial_state, args=(p["alpha"], p["beta"], p["delta"], p["gamma"]), t_eval=t_eval)

#%%

data = solution.y.T[None, None, ...]
print("data shape (first two are superfluous):", data.shape)

# %%

class Processor(eqx.Module):
    layers: list

    def __init__(self, in_size=2, out_size=2, key=None):
        keys = get_new_key(key, num=3)
        self.layers = [eqx.nn.Linear(in_size, 10, key=keys[0]), jax.nn.tanh,
        # self.layers = [eqx.nn.Linear(in_size+1, 50, key=keys[0]), jax.nn.tanh,
                        # eqx.nn.Linear(10, 10, key=keys[1]), jax.nn.tanh,
                        eqx.nn.Linear(10, out_size, key=keys[2]) ]

    def __call__(self, x, t):
        # y = jnp.concatenate([jnp.broadcast_to(t, (1,)), x], axis=0)
        y = x
        for layer in self.layers:
            y = layer(y)
        return y

# %%

model_keys = get_new_key(SEED, num=2)
model = Processor(in_size=2, out_size=2, key=model_keys[0])


params, static = eqx.partition(model, eqx.is_array)
params_flat, params_shapes, tree_def = flatten_pytree(params)









# %%


def params_norm(params):
    return jnp.array([jnp.sum(jnp.abs(x)) for x in jax.tree_util.tree_leaves(params)]).sum()

def l2_norm(X, X_hat):
    total_loss = jnp.mean((X - X_hat)**2, axis=-1)   ## Norm of d-dimensional vectors
    return jnp.sum(total_loss) / (X.shape[-2])

# %%

def loss_fn(params, static, batch):
    print('\nCompiling function "loss_fn" ...\n')
    X, t = batch

    X_hat = integrator(params, static, X[0, 0, :], t, 1.4e-8, 1.4e-8, jnp.inf, jnp.inf, 50, "checkpointed")

    return l2_norm(X, X_hat)
    # return 1e0*params_norm(params_env)
    # return l2_norm(X, X_hat) + 1e-0*params_norm(params_env)


@partial(jax.jit, static_argnums=(1))
def train_step(params, static, batch, opt_state):
    print('\nCompiling function "train_step" ...\n')

    loss, grads = jax.value_and_grad(loss_fn)(params, static, batch)

    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss


total_steps = nb_epochs

# sched = optax.exponential_decay(init_lr, total_steps, decay_rate)
# sched = optax.linear_schedule(init_lr, 0, total_steps, 0.25)
sched = optax.piecewise_constant_schedule(init_value=init_lr,
                                            boundaries_and_scales={int(total_steps*0.25):0.5, 
                                                                    int(total_steps*0.5):0.2,
                                                                    int(total_steps*0.75):0.5})
fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))

start_time = time.time()


print(f"\n\n=== Beginning Training ... ===")

opt = optax.adam(sched)
opt_state = opt.init(params)

losses = []
for epoch in range(nb_epochs):

    nb_batches = 0
    loss_sum = 0.
    for i in range(0, 1, batch_size):
        batch = (data[0,i:i+batch_size,...], t_eval)
    
        params, opt_state, loss = train_step(params, static, batch, opt_state)

        loss_sum += loss
        nb_batches += 1

    loss_epoch = loss_sum/nb_batches
    losses.append(loss_epoch)

    if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
        print(f"    Epoch: {epoch:-5d}      Loss: {loss_epoch:.8f}", flush=True)


# ax = sbplot(losses, x_label='Epoch', y_label='L2', y_scale="log", title=f'Loss for environment {e}', ax=ax);
ax = sbplot(losses, x_label='Epoch', y_label='L2', y_scale="log", title='Losses', ax=ax);
plt.savefig(f"data/loss_simple.png", dpi=300, bbox_inches='tight')
# plt.show()
plt.legend()
fig.canvas.draw()
fig.canvas.flush_events()

wall_time = time.time() - start_time
time_in_hmsecs = seconds_to_hours(wall_time)
print("\nTotal training time: %d hours %d mins %d secs" %time_in_hmsecs)



# %%

@partial(jax.jit, static_argnums=(1))
def loss_fn_(params_flat, static, batch):
    print('\nCompiling function "loss_fn" ...\n')
    X, t = batch
    params = unflatten_pytree(params_flat, params_shapes, tree_def)

    X_hat = integrator(params, static, X[0, 0, :], t, 1.4e-8, 1.4e-8, jnp.inf, jnp.inf, 50, "checkpointed")

    return l2_norm(X, X_hat)
    # return 1e0*params_norm(params_env)
    # return l2_norm(X, X_hat) + 1e-0*params_norm(params_env)

params_flat, params_shapes, tree_def = flatten_pytree(params)


results = jsp.optimize.minimize(loss_fn_, params_flat, args=(static, (data[0,:,...], t_eval)), tol=None, method='BFGS', options={'maxiter': 1000})
# print("Success: ", results.x.shape, results)
print("Optimisation result: ", results.success, results.nit, results.fun)

params_flat = results.x
params = unflatten_pytree(params_flat, params_shapes, tree_def)



# %%

# from typing import List, Tuple, Callable
# args=(static, (data[0,:,...], t_eval))

# def newton(x: np.ndarray, f: Callable, gf: Callable, hf: Callable, lr=0.01, lr_decr=0.999, maxiter=10, tol=0.001) -> Tuple[np.ndarray, List[np.ndarray], int]:

#     # points = [x]
#     nit = 0
#     gradient = gf(x, *args)
#     hessian = hf(x, *args)

#     # while nit < maxiter and np.linalg.norm(gradient) >= tol: 
    
#     #     # x = x - lr * jnp.dot(jnp.linalg.inv(hessian), gradient)  # Matrix multiplication using np.dot(m1, m2)
#     #     x = x - lr*jnp.linalg.solve(hessian, gradient)
#     #     lr *= lr_decr  # Learning rate update: tk+1 = tk * ρ, with ρ being the decay factor.
#     #     # points.append(x)
#     #     nit += 1
#     #     gradient = gf(x, *args)
#     #     hessian = hf(x, *args)

#     #     print(f"    Iter: {nit:-5d}      Loss: {np.linalg.norm(gradient):.8f}", flush=True) 



#     ### Put everything above in a jax fori lopp
#     def body_fn(carry):
#         _, x, lr, errors, nit = carry
#         gradient = gf(x, *args)
#         hessian = hf(x, *args)
#         x_new = x - lr*jnp.linalg.solve(hessian, gradient)

#         errors = errors.at[nit+1].set(jnp.linalg.norm(x_new - x))
#         return x, x_new, lr*lr_decr, errors, nit+1

#     def cond_fn(carry):
#         _, _, _, errors, nit = carry
#         return (errors[nit] >= tol) & (nit < maxiter)

#     errors = jnp.zeros((maxiter,))
#     errors = errors.at[0].set(2*tol)

#     _, x, lr, errors, nit = jax.lax.while_loop(cond_fn, body_fn, (x, x, lr, errors, nit))




#     # return x, points, nit
#     return x, None, nit

# gf = jax.jacrev(loss_fn_, argnums=0)
# hf = jax.jacrev(gf, argnums=0)

# params_flat, _, nit = newton(params_flat, loss_fn_, gf, hf, maxiter=200, lr=0.1, lr_decr=0.999, tol=0.001)


# %%
def test_model(params, static, batch):
    X0, t = batch

    X_hat = integrator(params, static, X0, t, 1.4e-8, 1.4e-8, jnp.inf, jnp.inf, 50, "checkpointed")

    return X_hat


i = np.random.randint(0, 1)

X = data[0, i, :, :]
t = t_eval

X_hat = test_model(params, static, (X[0,:], t))

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
