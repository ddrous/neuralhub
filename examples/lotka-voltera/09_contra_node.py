
#%%[markdown]
# # Contrastive Neural ODE for generalising the Lotka-Volterra systems

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
import jax.lax

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
nb_epochs = 5
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

dataset = np.load('./data/lotka_volterra_big.npz')
data, t_eval = dataset['X'], dataset['t']

cutoff_length = int(cutoff*data.shape[2])

print("data shape:", data.shape, t_eval.shape)

# %%

class Encoder(eqx.Module):
    layers: list

    def __init__(self, traj_size, context_size, key=None):        ## TODO make this convolutional
        # super().__init__(**kwargs)
        keys = get_new_key(key, num=3)
        self.layers = [eqx.nn.Linear(traj_size, 50, key=keys[0]), jax.nn.tanh,
                        eqx.nn.Linear(50, 20, key=keys[1]), jax.nn.softplus,
                        eqx.nn.Linear(20, context_size, key=keys[2]) ]
        # print("Encoder trajectory size input is:", traj_size)

    def __call__(self, traj):
        # print("Encoder got and input of size:", traj.size)
        context = traj
        for layer in self.layers:
            context = layer(context)
        return context

class Hypernetwork(eqx.Module):
    layers: list

    tree_def: jax.tree_util.PyTreeDef
    leave_shapes: list
    static: eqx.Module

    def __init__(self, context_size, processor, key=None):
        keys = get_new_key(key, num=3)

        proc_params, self.static = eqx.partition(processor, eqx.is_array)

        flat, self.leave_shapes, self.tree_def = flatten_pytree(proc_params)
        out_size = flat.shape[0]
        print("Hypernetwork will output", out_size, "parameters")

        self.layers = [eqx.nn.Linear(context_size, 160, key=keys[0]), jax.nn.tanh,
                        eqx.nn.Linear(160, 160*4, key=keys[1]), jax.nn.softplus,
                        eqx.nn.Linear(160*4, out_size, key=keys[2]) ]

    def __call__(self, context):
        weights = context
        for layer in self.layers:
            weights = layer(weights)
        # return weights
        proc_params = unflatten_pytree(weights, self.leave_shapes, self.tree_def)
        return eqx.combine(proc_params, self.static)


class NeuralODE(eqx.Module):
    hypernet: Hypernetwork

    def __init__(self, context_size, processor, key=None):
        self.hypernet = Hypernetwork(context_size, processor, key=key)

    def __call__(self, x0, t_eval, xi):
        processor = self.hypernet(xi)

        xs_hat = diffrax.diffeqsolve(
                    diffrax.ODETerm(lambda t, x, args: processor(x)),
                    diffrax.Tsit5(),
                    t0=t_eval[0],
                    t1=t_eval[-1],
                    dt0=t_eval[1] - t_eval[0],
                    y0=x0,
                    stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
                    saveat=diffrax.SaveAt(ts=t_eval),
                    max_steps=4096*1,
                ).ys
        
        return xs_hat


class ContraNODE(eqx.Module):
    neural_ode: NeuralODE
    encoder: Encoder            ## TODO Important, this needs to accept variable length trajectorirs. A time series, basically ! 
    encoder_input_size: int     ## Based on the above, this shouldn't be needed

    def __init__(self, proc_data_size, proc_width_size, proc_depth, context_size, traj_size, key=None):
        keys = get_new_key(key, num=3)
        processor = eqx.nn.MLP(
            in_size=proc_data_size,
            out_size=proc_data_size,
            width_size=proc_width_size,
            depth=proc_depth,
            activation=jax.nn.softplus,
            key=keys[0],
        )

        self.neural_ode = NeuralODE(context_size, processor, key=keys[1])
        self.encoder = Encoder(traj_size*proc_data_size, context_size, key=keys[2])
        self.encoder_input_size = traj_size*proc_data_size

    def __call__(self, x0, t_eval, xi):
        traj = self.neural_ode(x0, t_eval, xi)
        # print("Trajectory shape:", traj.shape)
        new_context = self.encoder(traj[:self.encoder_input_size, :].ravel())
        return traj, new_context


# %%

model_keys = get_new_key(SEED, num=2)

model = ContraNODE(proc_data_size=2, 
                   proc_width_size=16, 
                   proc_depth=3, 
                   context_size=2, 
                   traj_size=cutoff_length, 
                   key=model_keys[0])

params, static = eqx.partition(model, eqx.is_array)



# %%


def p_norm(params):
    """ norm of the parameters """
    return jnp.array([jnp.linalg.norm(x) for x in jax.tree_util.tree_leaves(params)]).sum()

def l2_norm(X, X_hat):
    total_loss = jnp.mean((X - X_hat)**2, axis=-1)   ## Norm of d-dimensional vectors
    return jnp.sum(total_loss) / (X.shape[-2] * X.shape[-3])

# Cosine similarity function
def cosine_similarity(a, b):
    a_normalized = a / jnp.linalg.norm(a, axis=-1, keepdims=True)
    b_normalized = b / jnp.linalg.norm(b, axis=-1, keepdims=True)
    return jnp.sum(a_normalized * b_normalized, axis=-1)

# Contrastive loss function
def contrastive_loss(xi_a, xi_b, positive, temperature):
    similarity = cosine_similarity(xi_a, xi_b)

    # # Positive pair (similar instances)
    # if positive:
    #     return -jnp.log(jnp.exp(similarity / temperature) / jnp.sum(jnp.exp(similarity / temperature)))

    # # Negative pair (dissimilar instances)
    # else:
    #     return -jnp.log(1.0 - jnp.exp(similarity / temperature))

    ## Make the above JAX-friendly
    return jax.lax.cond(positive, 
                        lambda x: -jnp.log(jnp.exp(x / temperature) / jnp.sum(jnp.exp(x / temperature))),
                        lambda x: -jnp.log(1.0 - jnp.exp(x / temperature)), 
                        operand=similarity)

# %%

### ==== Vanilla Gradient Descent optimisation ==== ####

def loss_fn(params, static, batch):
    print('\nCompiling function "loss_fn" ...\n')
    a, xi_a, Xa, b, xi_b, Xb, t_eval = batch

    model = eqx.combine(params, static)

    X_hat_a, xi_as = jax.vmap(model, in_axes=(0, None, None))(Xa[:, 0, :], t_eval, xi_a)
    X_hat_b, xi_bs = jax.vmap(model, in_axes=(0, None, None))(Xb[:, 0, :], t_eval, xi_b)

    # print("Xa shape:", Xa.shape, "X_hat_a shape:", X_hat_a.shape, "t_eval shape:", t_eval.shape)

    term1 = l2_norm(Xa, X_hat_a) + l2_norm(Xb, X_hat_b)
    # term2 = jax.vmap(contrastive_loss, in_axes=(0, 0, 0, None))(xi_as, xi_bs, a==b, 1.0).mean()
    term2 = contrastive_loss(xi_a, xi_b, a==b, 1.0)
    term3 = p_norm(params)

    new_xi_a = jnp.mean(xi_as, axis=0)
    new_xi_b = jnp.mean(xi_bs, axis=0)
    # new_xi_a, new_xi_b = xi_a, xi_b
    # term4 = MAKE SURE THE xi_as are all constant (i.e. the same) for each a

    loss_val = term1 + term2
    return loss_val, (new_xi_a, new_xi_b, term1, term2, term3)


@partial(jax.jit, static_argnums=(1))
def train_step(params, static, batch, opt_state):
    print('\nCompiling function "train_step" ...\n')

    (loss, aux_data), grads  = jax.value_and_grad(loss_fn, has_aux=True)(params, static, batch)

    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss, aux_data


def make_contrastive_data_batch(batch_id, xis, data, batch_size, cutoff_length):
    """ Make contrastive data from a batch of data """
    nb_envs = data.shape[0]
    nb_examples = data.shape[1]

    # a = np.broadcast_to(np.random.randint(0, nb_envs), (batch_size,))
    # b = np.broadcast_to(np.random.randint(0, nb_envs), (batch_size,))

    a = np.random.randint(0, nb_envs)           ## TODO: increase the chances of a==b
    b = np.random.randint(0, nb_envs)

    xi_a = xis[a]
    xi_b = xis[b]

    batch_id = batch_id % (nb_examples//batch_size)
    start, finish = batch_id*batch_size, (batch_id+1)*batch_size

    Xa = data[a, start:finish, :cutoff_length, :]
    Xb = data[b, start:finish, :cutoff_length, :]

    return batch_id+1, (a, xi_a, Xa, b, xi_b, Xb, t_eval[:cutoff_length])


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

xis = np.random.normal(size=(data.shape[0], 2))

losses = []
for epoch in range(nb_epochs):

    nb_batches = 0
    loss_sum = jnp.zeros(4)
    batch_id = 0
    for i in range(data.shape[1]//batch_size):
        _, batch = make_contrastive_data_batch(i, xis, data, batch_size, cutoff_length)
    
        params, opt_state, loss, (xi_a, xi_b, term1, term2, term3) = train_step(params, static, batch, opt_state)

        a, _, _, b, _, _, _ = batch
        xis[a], xis[b] = xi_a, xi_b

        loss_sum += jnp.array([loss, term1, term2, term3])
        nb_batches += 1

    loss_epoch = loss_sum/nb_batches
    losses.append(loss_epoch)

    if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
        print(f"    Epoch: {epoch:-5d}      TotalLoss: {loss_epoch[0]:-.5f}     TrajLoss: {loss_epoch[1]:-.5f}      ContrastLoss: {loss_epoch[2]:-.5f}      ParamsLoss: {loss_epoch[3]:-.5f}", flush=True)

losses = jnp.vstack(losses)
# ax = sbplot(losses, x_label='Epoch', y_label='L2', y_scale="log", title=f'Loss for environment {e}', ax=ax);
ax = sbplot(losses, label=["Total", "Traj", "Contrast", "Params"], x_label='Epoch', y_scale="log", title='Loss Terms', ax=ax);
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
    xi, X0, t = batch

    model = eqx.combine(params, static)
    X_hat, _ = model(X0, t, xi)

    return X_hat, _

# for e in range(data.shape[0]):

e = np.random.randint(0, data.shape[0])
X = data[e, 0, :, :]

X_hat, _ = test_model(params, static, (xis[e], X[0,:], t_eval))

# print("X_hat values", X_hat)

ax = sbplot(X_hat[:,0], X_hat[:,1], x_label='Preys', y_label='Predators', label=f'Pred', title=f'Phase space, env {e}')
ax = sbplot(X[:,0], X[:,1], "--", lw=1, label=f'True', ax=ax)

# plt.savefig(f"data/contrastnode_test_env{e}_traj{i}.png", dpi=300, bbox_inches='tight')

#%% 
## Plot and annotate all the xis points along with text labels
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

## Named colors matplotlib
colors = ['white', 'r', 'b', 'g', 'm', 'c', 'y', 'orange', 'purple']

ax.scatter(xis[:,0], xis[:,1], s=20, c=colors, marker='x')
for i, (x, y) in enumerate(xis):
    ax.annotate(str(i), (x, y), fontsize=8)

# lim = 0.25
# ax.set_xlim(-lim, lim)
# ax.set_ylim(-lim, lim)
ax.set_title(r'Environment Contexts $\xi_e$')

#%% 

# model = eqx.combine(params, static)

# eqx.tree_serialise_leaves("data/model_02.eqx", model)
# model = eqx.tree_deserialise_leaves("data/model_01.eqx", model)

# %% [markdown]

# # Preliminary results


# # Conclusion

# %%
