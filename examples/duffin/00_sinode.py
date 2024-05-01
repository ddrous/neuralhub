#%%[markdown]
# # Neural ODE mixed with SINDy framework with mixture of experts

### Summary
# - We have a multistable dynamical systems that we want to learn
# - We only learn on a single attrator
# - Does it learn the other attractor ?


#%%
import jax

print("\n############# Neural ODE #############\n")
print("Available devices:", jax.devices())

from jax import config
##  Debug nans
# config.update("jax_debug_nans", True)

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

# SEED = 27
SEED = 2024

## Integrator hps
# integrator = rk4_integrator
integrator = rk4_integrator_args

## Optimiser hps
init_lr = 1e-3
decay_rate = 0.9

## Training hps
print_every = 1000
nb_epochs = 50000
# batch_size = 128*10

skip = 100

#%%
# Define the Duffing system
def duffing(t, state, a, b, c):
    x, y = state
    dxdt = y
    dydt = a*y - x*(b + c*x**2)
    return [dxdt, dydt]

# Parameters
a, b, c = -1/2., -1, 1/10.

t_span = (0, 40)
t_eval = np.arange(t_span[0], t_span[1], 0.01)[::skip]


# plt.figure(figsize=(10, 6))
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

init_conds = np.array([[-0.5, -1], [-0.5, -0.5], [-0.5, 0.5], 
                       [-1.5, 1], 
                    #    [-0.5, 1], 
                       [-1, -1], [-1, -0.5], [-1, 0.5], [-1, 1], 
                       [-2, -1], [-2, -0.5], [-2, 0.5], [-2, 1],
                    #    [0.5, -1], [0.5, -0.5], [0.5, 0.5], [0.5, 1],
                    #    [1, -1], [1, -0.5], [1, 0.5], [1, 1],
                    #    [2, -1], [2, -0.5], [2, 0.5], [2, 1],
                       ])


train_data = []

for state0 in init_conds:
    sol = solve_ivp(duffing, t_span, state0, args=(a, b, c), t_eval=t_eval)
    train_data.append(sol.y.T)

    ## Plot the phase space
    ax = sbplot(sol.y[0], sol.y[1], ".-", ax=ax, dark_background=False)

ax.set_xlabel('Displacement (x)')
ax.set_ylabel('Velocity (y)')
ax.set_title('Phase Space')

# plt.grid(True)
# plt.show()

## Save the training data
data = np.stack(train_data)[None, ...]


# %%

def topk_routing(x, k):
    # sort and print
    values, indices = jax.lax.top_k(x, k)
    ret = jnp.full_like(x, -jnp.inf)
    ret = ret.at[indices].set(values)
    # jax.debug.print("Printing top k values: {} {} {}", ret, x, indices)
    # print("Printing top k values: ", ret, values, indices)
    return ret

class Swish(eqx.Module):
    beta: jnp.ndarray
    def __init__(self, key=None):
        self.beta = jax.random.uniform(key, shape=(1,), minval=0.01, maxval=1.0)
    def __call__(self, x):
        return x * jax.nn.sigmoid(self.beta * x)


def distance(node1, node2):
    diff = node1 - node2
    return jnp.sqrt(diff@diff.T)
    # return jnp.sum(diff*diff)      ## Squared distance ! No problem for Gaussain RBF
    # return jnp.linalg.norm(node1 - node2)       ## Carefull: not differentiable at 0

def gaussian_rbf(r_squared, shape):
    return jnp.exp(-r_squared / ( 2 * shape**2))


def polyharmonic(r, a):
    return r**(2*a+1)

def thin_plate(r, a):
    return jnp.nan_to_num(jnp.log(r) * r**(2*a), neginf=0., posinf=0.)

# def polyharmonic(x, center, a=1):
#     return polyharmonic_func(distance(x, center), a)

# class NeuralNet(eqx.Module):
#     centers: jnp.ndarray
#     shapes: jnp.ndarray     ## Widths for the gaussian RBF network
#     weights: jnp.ndarray

#     def __init__(self, in_size=2, nb_centers=10, out_size=2, key=None):
#         keys = get_new_key(key, num=4)

#         centers_x = jax.random.uniform(keys[0], (nb_centers, 1), minval=-4., maxval=4.)
#         centers_y = jax.random.uniform(keys[3], (nb_centers, 1), minval=-1.5, maxval=1.5)
#         self.centers = jnp.concatenate([centers_x, centers_y], axis=1)

#         self.shapes = jax.random.uniform(keys[1], (nb_centers,), minval=0.1, maxval=10.)
#         self.weights = eqx.nn.Linear(nb_centers, out_size, key=keys[2])

#     def __call__(self, x, t):
#         # y = jnp.concatenate([jnp.broadcast_to(t, (1,)), x], axis=0)
#         dists_squared = jax.vmap(distance, in_axes=(0, None))(self.centers, x)
#         activations = jax.vmap(gaussian_rbf, in_axes=(0, 0))(dists_squared, self.shapes)
#         return self.weights(activations)


# class Processor(eqx.Module):
#     neuralnet: NeuralNet
#     params_router: jnp.ndarray

#     def __init__(self, in_size=2, nb_centers=10, out_size=2, key=None):
#         keys = get_new_key(key, num=2)
#         self.neuralnet = NeuralNet(in_size, nb_centers, out_size, key=keys[1])
#         self.params_router = jax.random.normal(keys[0], (data_size,))

#     def __call__(self, x, t):

#         w_out = self.params_router@x.T
#         gate = jax.lax.cond(w_out>0, lambda x: jnp.array([1., 0.]), lambda x: jnp.array([0., 1.]), x)

#         nb_centers = self.neuralnet.centers.shape[0]
#         ## Sample points to the left of line defined by params_router
#         key = get_new_key(SEED, num=2)
#         left_points_x = jax.random.uniform(key[0], (nb_centers, 1), minval=-4., maxval=4.)


#         ## Sample points to the left of line defined by params_router

#         # return self.physics(x, t) + self.augmentation(x, t)
#         return self.neuralnet(x, t)





class NeuralNet(eqx.Module):
    # centers: jnp.ndarray
    # shapes: jnp.ndarray     ## Widths for the gaussian RBF network
    weights: jnp.ndarray

    def __init__(self, in_size=2, nb_centers=10, out_size=2, key=None):
        keys = get_new_key(key, num=4)

        # centers_x = jax.random.uniform(keys[0], (nb_centers, 1), minval=-4., maxval=4.)
        # centers_y = jax.random.uniform(keys[3], (nb_centers, 1), minval=-1.5, maxval=1.5)
        # self.centers = jnp.concatenate([centers_x, centers_y], axis=1)

        # self.shapes = jax.random.uniform(keys[1], (nb_centers,), minval=0.1, maxval=10.)
        self.weights = eqx.nn.Linear(nb_centers, out_size, key=keys[2])

    def __call__(self, x, t, centers, shapes):
        # y = jnp.concatenate([jnp.broadcast_to(t, (1,)), x], axis=0)
        dists_squared = jax.vmap(distance, in_axes=(0, None))(centers, x)
        activations = jax.vmap(gaussian_rbf, in_axes=(0, 0))(dists_squared, shapes)
        # activations = jax.vmap(thin_plate, in_axes=(0, None))(dists_squared, 1)

        return self.weights(activations)
        # return jnp.nan_to_num(self.weights(activations), neginf=0., posinf=0.)


class Processor(eqx.Module):
    neuralnet: NeuralNet
    params_router: jnp.ndarray
    left_centers: jnp.ndarray
    right_centers: jnp.ndarray

    left_shapes: jnp.ndarray
    right_shapes: jnp.ndarray

    def __init__(self, in_size=2, nb_centers=10, out_size=2, key=None):
        keys = get_new_key(key, num=5)
        self.neuralnet = NeuralNet(in_size, nb_centers, out_size, key=keys[1])
        self.params_router = jax.random.normal(keys[0], (in_size,))

        keys = get_new_key(key, num=2)
        left_centers_x = jax.random.uniform(keys[0], (nb_centers, 1), minval=-4., maxval=1)
        left_centers_y = jax.random.uniform(keys[1], (nb_centers, 1), minval=-1.5, maxval=1.5)
        self.left_centers = jnp.concatenate([left_centers_x, left_centers_y], axis=1)
        self.left_shapes = jax.random.uniform(keys[1], (nb_centers,), minval=0.1, maxval=10.)

        keys = get_new_key(key, num=3)
        right_centers_x = jax.random.uniform(keys[0], (nb_centers, 1), minval=-1., maxval=4.)
        right_centers_y = jax.random.uniform(keys[1], (nb_centers, 1), minval=-1.5, maxval=1.5)
        self.right_centers = jnp.concatenate([right_centers_x, right_centers_y], axis=1)
        self.right_shapes = jax.random.uniform(keys[1], (nb_centers,), minval=0.1, maxval=10.)

    def __call__(self, x, t, key):

        w_out = self.params_router@x.T
        gate = jax.lax.cond(w_out>0, lambda x: jnp.array([1., 0.]), lambda x: jnp.array([0., 1.]), x)

        # nb_centers = self.neuralnet.weights.weight.shape[1]

        ## Sample points to the left of line defined by params_router
        # keys = get_new_key(key, num=2)
        # left_centers_x = jax.random.uniform(keys[0], (nb_centers, 1), minval=-4., maxval=-1)
        # left_centers_y = jax.random.uniform(keys[1], (nb_centers, 1), minval=-1.5, maxval=1.5)
        # left_centers = jnp.concatenate([left_centers_x, left_centers_y], axis=1)

        ## Sample points to the left of line defined by params_router
        # keys = get_new_key(key, num=3)
        # right_centers_x = jax.random.uniform(keys[0], (nb_centers, 1), minval=1., maxval=4.)
        # right_centers_y = jax.random.uniform(keys[1], (nb_centers, 1), minval=-1.5, maxval=1.5)
        # right_centers = jnp.concatenate([right_centers_x, right_centers_y], axis=1)

        # shapes = jax.random.uniform(key, (nb_centers,), minval=0.1, maxval=10.)

        # return self.physics(x, t) + self.augmentation(x, t)
        return gate[0]*self.neuralnet(x, t, self.left_centers, self.left_shapes) + gate[1]*self.neuralnet(x, t, self.right_centers, self.right_shapes)




# %%

model_keys = get_new_key(SEED, num=2)
# model = Processor(data_size=2, int_size=32, context_size=16, key=model_keys[0])
model = Processor(in_size=2, nb_centers=256*32, out_size=2, key=model_keys[0])


params, static = eqx.partition(model, eqx.is_array)
params_flat, params_shapes, tree_def = flatten_pytree(params)









# %%

def params_norm(params):
    return jnp.array([jnp.sum(jnp.abs(x)) for x in jax.tree_util.tree_leaves(params)]).sum()

def l2_norm(X, X_hat):
    total_loss = jnp.mean((X - X_hat)**2, axis=-1)   ## Norm of d-dimensional vectors
    return jnp.sum(total_loss) / (X.shape[-2])

# %%

### ==== Vanilla Gradient Descent optimisation ==== ####

def loss_fn(params, static, batch, key):
    print('\nCompiling function "loss_fn" ...\n')
    X, t = batch

    rhs = eqx.combine(params, static)

    # X_hat = integrator(rhs, X[0, 0, :], t, 1.4e-8, 1.4e-8, jnp.inf, jnp.inf, 50, "checkpointed")
    # X_hat = jax.vmap(integrator, in_axes=(None, 0, None, None, None, None, None, None, None))(rhs, X[:, 0, :], t, 1.4e-8, 1.4e-8, jnp.inf, jnp.inf, 50, "checkpointed")
    X_hat = jax.vmap(integrator, in_axes=(None, 0, None, None, None, None, None, None, None, None))(rhs, X[:, 0, :], t, key, 1.4e-8, 1.4e-8, jnp.inf, jnp.inf, 50, "checkpointed")

    return jnp.mean((X-X_hat)**2)
    # return l2_norm(X, X_hat)
    # return 1e0*params_norm(params_env)
    # return l2_norm(X, X_hat) + 1e-0*params_norm(params_env)


@partial(jax.jit, static_argnums=(1))
def train_step(params, static, batch, opt_state, key):
    print('\nCompiling function "train_step" ...\n')

    loss, grads = jax.value_and_grad(loss_fn)(params, static, batch, key)

    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss


total_steps = nb_epochs

# sched = optax.exponential_decay(init_lr, total_steps, decay_rate)
# sched = optax.linear_schedule(init_lr, 0, total_steps, 0.25)
sched_factor = 0.5
sched = optax.piecewise_constant_schedule(init_value=init_lr,
                                            boundaries_and_scales={int(total_steps*0.25):sched_factor, 
                                                                    int(total_steps*0.5):sched_factor,
                                                                    int(total_steps*0.75):sched_factor})
fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
nb_data_points = data.shape[1]
batch_size = nb_data_points

start_time = time.time()


print(f"\n\n=== Beginning Training ... ===")

opt = optax.adam(sched)
opt_state = opt.init(params)

train_key = get_new_key(SEED*2, num=1)

losses = []
for epoch in range(nb_epochs):

    nb_batches = 0
    loss_sum = 0.
    for i in range(0, nb_data_points, batch_size):
        batch = (data[0,i:i+batch_size,...], t_eval)
    
        train_key, _ = jax.random.split(train_key)
        params, opt_state, loss = train_step(params, static, batch, opt_state, train_key)

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
print("\nTotal GD training time: %d hours %d mins %d secs" %time_in_hmsecs)


# %%

# model = eqx.tree_deserialise_leaves("data/model_06.eqx", model)
# params, static = eqx.partition(model, eqx.is_array)


# %%


# %%
def test_model(params, static, batch, key):
    X0, t = batch
    rhs = eqx.combine(params, static)

    X_hat = jax.vmap(integrator, in_axes=(None, 0, None, None, None, None, None, None, None, None)
                     )(rhs, X0, t, key, 1.4e-8, 1.4e-8, jnp.inf, jnp.inf, 50, "checkpointed")
    # X_hat = integrator(rhs, X0, t, 1.4e-8, 1.4e-8, jnp.inf, jnp.inf, 50, "checkpointed")

    return X_hat


i = np.random.randint(0, 1)

X = data[0, :, :, :]
t = np.linspace(t_span[0], t_span[1], 40)       ## TODO important

X_hat = test_model(params, static, (X[:, 0,:], t), train_key)

# print("L2 error:", jnp.mean((X-X_hat)**2))

fig, ax = plt.subplots(1, 1, figsize=(10, 5))



colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'orange', 'yellow', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
colors = colors*10

for i in range(X.shape[0]):
#     if i==0:
#         ax = sbplot(t, X_hat[i, :, 0], x_label='Time', y_label='Displacement', label=f'Pred', title=f'Trajectory {i}', ax=ax)
#         ax = sbplot(t, X[i, :, 0], "--", lw=1, label=f'True', ax=ax)
#     else:
#         ax = sbplot(t, X_hat[i, :, 0], x_label='Time', y_label='Displacement', title=f'Trajectory {i}', ax=ax)
#         ax = sbplot(t, X[i, :, 0], "--", lw=1, ax=ax)

    if i==0:
        sbplot(X_hat[i, :,0], X_hat[i, :,1], x_label='x', y_label='y', label=f'Pred', title=f'Phase space', ax=ax, alpha=0.5, color=colors[i])
        sbplot(X[i, :,0], X[i, :,1], "o", lw=1, label=f'True', ax=ax, color=colors[i])
    else:
        sbplot(X_hat[i, :,0], X_hat[i, :,1], x_label='x', y_label='y', ax=ax, alpha=0.5, color=colors[i])
        sbplot(X[i, :,0], X[i, :,1], "o", lw=1, ax=ax, color=colors[i])

# plt.savefig(f"data/coda_test_env{e}_traj{i}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"data/test_simple.png", dpi=300, bbox_inches='tight')


#%% 

model = eqx.combine(params, static)
eqx.tree_serialise_leaves("data/model_06.eqx", model)

# %% [markdown]

# # Preliminary results
# - Nothing yet



# # Conclusion
# 
# 


# %%

## Check if the router has identified the correct attractor

# Samples points in the rectangle [-4,4]x[-1.5,1.5]
# import numpy as np
res = 50
x0s = np.array([[x, y] for x in np.linspace(-4, 4, res) for y in np.linspace(-1.5, 1.5, res)])

def classify_bassin(model, x0):

    # w_out = x0
    # for layer in model.layers_rout:
    #     w_out = layer(w_out)
    # return jnp.argmax(w_out)

    w_out = model.params_router@x0.T
    return (w_out>0).astype(int)




y0s = jax.vmap(classify_bassin, in_axes=(None, 0))(model, x0s)

plt.figure(figsize=(10, 5))
plt.scatter(x0s[:, 0], x0s[:, 1], c=y0s, s=5, cmap='coolwarm')
plt.xlabel('Displacement (x)')
plt.ylabel('Velocity (y)')
plt.title('Basins of attraction')
# plt.grid(True)
plt.show()


# %%
## Predict all trajectories using only one expert

def test_expert(model, batch, expert=0):
    X0, t = batch

    # def rhs(x, t):
    #     y = x
    #     layers = model.layers1 if expert==0 else model.layers2
    #     for layer in layers:
    #         y = layer(y)
    #     return y

    ## New model with layers1 equal to -layers2

    # params, static = eqx.partition(model, eqx.is_array)
    # layers = jax.tree_util.tree_map(lambda x:-x, params.layers2)
    # params = eqx.tree_at(lambda m: m.layers1, model, layers)
    # model = eqx.combine(params, static)

    # def rhs(x, t):
    #     ## Top 1 routing function
    #     # w_out = x
    #     # for layer in model.layers_rout:
    #     #     w_out = layer(w_out)
    #     # gate = jax.nn.softmax(topk_routing(w_out, 1))

    #     w_out = model.params_router@x.T
    #     gate = jax.lax.cond(w_out>0, lambda x: jnp.array([1., 0.]), lambda x: jnp.array([0., 1.]), x)

    #     y1 = x
    #     y2 = x
    #     for i in range(len(model.layers1)):
    #         y1 = model.layers1[i](y1)
    #         y2 = model.layers2[i](y2)

    #     # return gate[0]*y1
    #     # return gate[0]*y1 + gate[1]*y2
    #     return gate[0]*y1 if expert==0 else gate[1]*y2
    rhs = model


    # X_hat = jax.vmap(integrator, in_axes=(None, 0, None, None, None, None, None, None, None))(rhs, X0, t, 1.4e-8, 1.4e-8, jnp.inf, jnp.inf, 50, "checkpointed")
    X_hat = jax.vmap(integrator, in_axes=(None, 0, None, None, None, None, None, None, None, None))(model, X0, t, train_key, 1.4e-8, 1.4e-8, jnp.inf, jnp.inf, 50, "checkpointed")

    return X_hat


# X = data[0, :, 0, :]
res = 10
X = np.array([[x, y] for x in np.linspace(-4, 4, res) for y in np.linspace(-1.5, 1.5, res)])
print(X.shape)

expert_id = 0

# X_hat = test_expert(model, (X[:, 0,:], t), expert=expert_id)
t = np.linspace(t_span[0], t_span[1], 400)
X_hat = test_expert(model, (X[:,:], t), expert=expert_id)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

for i in range(X.shape[0]):
    if i==0:
        sbplot(X_hat[i, :,0], X_hat[i, :,1], "+-", x_label='x', y_label='y', label=f'Pred', title=f'Phase space - Expert {expert_id}', ax=ax, alpha=0.5, color=colors[i])
        # sbplot(X[i, :,0], X[i, :,1], "o", lw=1, label=f'True', ax=ax, color=colors[i])
    else:
        sbplot(X_hat[i, :,0], X_hat[i, :,1], "-", x_label='x', y_label='y', ax=ax, alpha=0.5, color=colors[i])
        # sbplot(X[i, :,0], X[i, :,1], "o", lw=1, ax=ax, color=colors[i])

# ax.set_xlim(-4, 4)
# ax.set_ylim(-1.5, 1.5)

# plt.savefig(f"data/coda_test_env{e}_traj{i}.png", dpi=300, bbox_inches='tight')
# plt.savefig(f"data/test_simple.png", dpi=300, bbox_inches='tight')
