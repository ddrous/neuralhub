
#%%[markdown]
# # CoDA framework for generalising the Lotka-Volterra systems

### Summary
# - Original paper here: https://arxiv.org/pdf/2202.01889.pdf
# - Compared to the original CoDA, we do not learn the latent vertor $\xi$ like they do.
# - We focus on learning the main network $\theta_c$, then for each new environment $e$, we learn a new network $\theta_c + \delta \theta_e$.

### My remarks
# - Must we learn the environments sequentially ?
# - Can we train all these models in parallel, and only use the avarage accros environments to update the main network?


#%%
import jax

print("\n############# Generalising Lotka-Volterra with CoDA #############\n")
print("Available devices:", jax.devices())

from jax import config

import jax.numpy as jnp
import numpy as np
import equinox as eqx

import matplotlib.pyplot as plt

from neuralhub.utils import *
from neuralhub.integrators import *

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

data = jnp.load('data/lotka_volterra_small.npz')
print("Data shapes for (t, X):", data["t"].shape, data["X"].shape, flush=True)

## Plot one of those Lotka-Volterra systems
# nb_envs = data["X"].shape[0] 
nb_envs = 1        ## Remove this line
nb_trajs_per_env = data["X"].shape[1]
nb_steps_per_traj = data["X"].shape[2]
d = data["X"].shape[3]

e = np.random.randint(0, nb_envs)
i = np.random.randint(0, nb_trajs_per_env)

data_e = data["X"][e, i, :, :]
# ax = sbplot(data_e[:, 0], data_e[:, 1], "-", x_label='Preys', label=f'$e={e},i={i}$', xlim=(0,2), ylim=(0,3), title='Lotka-Volterra')
ax = sbplot(data_e[:, 0], data_e[:, 1], "-", x_label='Preys', label=f'$e={e},i={i}$', title='Lotka-Volterra')

# %%

class Processor(eqx.Module):
    layers: list

    def __init__(self, in_size=2, out_size=2, key=None):
        keys = get_new_key(key, num=3)
        self.layers = [eqx.nn.Linear(in_size, 50, key=keys[0]), jax.nn.tanh,
        # self.layers = [eqx.nn.Linear(in_size+1, 50, key=keys[0]), jax.nn.tanh,
                        # eqx.nn.Linear(100, 100, key=keys[1]), jax.nn.tanh,
                        eqx.nn.Linear(50, out_size, key=keys[2]) ]

    def __call__(self, x, t):
        # y = jnp.concatenate([jnp.broadcast_to(t, (1,)), x], axis=0)
        y = x
        for layer in self.layers:
            y = layer(y)
        return y

# %%

model_keys = get_new_key(SEED, num=nb_envs+1)

model_main = Processor(in_size=d, out_size=d, key=model_keys[0])

models_envs = []
for e in range(nb_envs):
    Theta_e = Processor(in_size=d, out_size=d, key=model_keys[e+1])
    models_envs.append(Theta_e)

params_main, static_main = eqx.partition(model_main, eqx.is_array)
params_envs, static_envs = eqx.partition(models_envs, eqx.is_array)










# %%


def params_norm(params):
    return jnp.array([jnp.sum(jnp.abs(x)) for x in jax.tree_util.tree_leaves(params)]).sum()

def l2_norm(X, X_hat):
    total_loss = jnp.mean((X - X_hat)**2, axis=-1)   ## Norm of d-dimensional vectors
    return jnp.sum(total_loss) / (X.shape[0] * X.shape[1])

def loss_fn(params, static, batch):
    X, t = batch

    params_main, params_env = params
    params_ = eqx.apply_updates(params_env, params_main)

    integrator_batched = jax.vmap(integrator, in_axes=(None, None, 0, None, None, None, None, None, None, None))

    X_hat = integrator_batched(params_, static, X[:, 0, :], t, 1.4e-8, 1.4e-8, jnp.inf, jnp.inf, 50, "checkpointed")

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


# %%

total_steps = nb_epochs*int(np.ceil(nb_trajs_per_env//batch_size))

# sched = optax.exponential_decay(init_lr, total_steps, decay_rate)
# sched = optax.linear_schedule(init_lr, 0, total_steps, 0.25)
sched = optax.piecewise_constant_schedule(init_value=init_lr,
                                            boundaries_and_scales={int(total_steps*0.25):0.5, 
                                                                    int(total_steps*0.5):0.2,
                                                                    int(total_steps*0.75):0.5})
fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))

start_time = time.time()

for e in list(range(nb_envs))*1:        ## TODO just once !!

    print(f"\n\n=== Training environment {e} ... ===")

    opt = optax.adam(sched)

    params_env = params_envs[e]
    params = (params_main, params_env)
    opt_state = opt.init(params)

    losses = []
    for epoch in range(nb_epochs):

        nb_batches = 0
        loss_sum = 0.
        for i in range(0, nb_trajs_per_env, batch_size):
            batch = (data['X'][e,i:i+batch_size,...], data['t'])
        
            params, opt_state, loss = train_step(params, static_main, batch, opt_state)

            loss_sum += loss
            nb_batches += 1

        loss_epoch = loss_sum/nb_batches
        losses.append(loss_epoch)

        if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
            print(f"    Epoch: {epoch:-5d}      Loss: {loss_epoch:.8f}", flush=True)

    ## Update the main network
    params_main = params[0]
    params_envs[e] = params[1]

    # ax = sbplot(losses, x_label='Epoch', y_label='L2', y_scale="log", title=f'Loss for environment {e}', ax=ax);
    ax = sbplot(losses, x_label='Epoch', y_label='L2', y_scale="log", label=f'{e}', title='Losses for each environment', ax=ax);
    plt.savefig(f"data/loss_{e}.png", dpi=300, bbox_inches='tight')
    # plt.show()
    plt.legend()
    fig.canvas.draw()
    fig.canvas.flush_events()
    # ax.clear()

wall_time = time.time() - start_time
time_in_hmsecs = seconds_to_hours(wall_time)
print("\nTotal training time: %d hours %d mins %d secs" %time_in_hmsecs)

# model = eqx.combine(params, static_main)

# %%
# plt.savefig("data/loss.png", dpi=300, bbox_inches='tight')
















# %%
def test_model(params, static, batch):
    X0, t = batch

    params_main, params_env = params
    params_ = eqx.apply_updates(params_env, params_main)

    X_hat = integrator(params_, static, X0, t, 1.4e-8, 1.4e-8, jnp.inf, jnp.inf, 50, "checkpointed")

    return X_hat


e = np.random.randint(0, nb_envs)
i = np.random.randint(0, nb_trajs_per_env)

X = data['X'][e, i, :, :]
t = data['t']

params = (params_main, params_envs[e])
X_hat = test_model(params, static_main, (X[0,:], t))

params_zeros = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params_main)
params = (params_main, params_zeros)
X_hat_main = test_model(params, static_main, (X[0,:], t))

ax = sbplot(X_hat[:,0], X_hat[:,1], x_label='Preys', y_label='Predators', label=f'Pred', title=f'Phase space for env {e}, traj {i}')
ax = sbplot(X_hat_main[:,0], X_hat_main[:,1], "b-", lw=1, label=f'Pred (Main)', ax=ax)
ax = sbplot(X[:,0], X[:,1], "--", lw=1, label=f'True', ax=ax)

# plt.savefig(f"data/coda_test_env{e}_traj{i}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"data/coda_test.png", dpi=300, bbox_inches='tight')


#%% 

model_main = eqx.combine(params_main, static_main)
model_envs = eqx.combine(params_envs, static_envs)
model = (model_main, model_envs)

eqx.tree_serialise_leaves("data/model_01.eqx", model)
# model = eqx.tree_deserialise_leaves("data/model_01.eqx", model)

# %% [markdown]

# # Preliminary results
# - The main networks learns nothing (at least with nb_envs=1)


# # Conclusion
# 
# 



