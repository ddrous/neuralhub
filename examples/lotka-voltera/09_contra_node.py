
#%%[markdown]
# # Contrastive Neural ODE for generalising the Lotka-Volterra systems
# - Make note of the batch formed by combinatorials. 
# - We randmly sample two trajectories from each environment and form a batch of positive pairs (same environment) and negative pairs (different environments). 
# - To make it easy. All the environments have the same initial conditions for their trajectories. Rather than having a unique init cond for all.
# - So, the batch is of size (nb_envs*(nb_envs-1)//2 + nb_envs) * (nb_trajs_per_batch_per_env//2).


### Summary


#%%
import itertools
import random
import jax

# from jax import config
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_platform_name", "cpu")

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

print("\n############# Generalising Lotka-Volterra with a Contrastive Hypernetwork #############\n")
print("Using JAX, with available devices:", jax.devices())

import jax.numpy as jnp
# import jax.scipy as jsp
import jax.scipy.optimize
import jax.lax

import diffrax

import numpy as np
np.set_printoptions(suppress=True)
from scipy.integrate import solve_ivp

import equinox as eqx

import matplotlib.pyplot as plt
# plt.style.use("bmh")

from graphpint.utils import *
from graphpint.integrators import *

import optax
from functools import partial
import time
from typing import List, Tuple, Callable


#%%

SEED = 22
# SEED = np.random.randint(0, 1000)

## Integrator hps
# integrator = rk4_integrator
# integrator = dopri_integrator
# integrator = dopri_integrator_diff

## Optimiser hps
init_lr = 3e-4
decay_rate = 0.1

## Training hps
print_every = 10
nb_epochs = 1
# batch_size = 2*20              ## Two trajectories per environment are picked for each train_step; this results in 45 contrastive pairs (9 pos, 36 negs). This process is repeated once. 
nb_trajs_per_batch_per_env = 2*128              ## Two trajectories per environment are picked for each train_step; this results in 45 contrastive pairs (9 pos, 36 negs). This process is repeated once. 

cutoff = 0.1

respulsive_dist = .5       ## For the contrastive loss (appplied to the contexts)

train = True

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

nb_envs = data.shape[0]
nb_trajs_per_env = data.shape[1]
nb_steps_per_traj = data.shape[2]
data_size = data.shape[3]

batch_size = (nb_envs*(nb_envs-1)//2 + nb_envs) * (nb_trajs_per_batch_per_env//2)       ## This is actual number of pairs in a batch
cutoff_length = int(cutoff*nb_steps_per_traj)

print("Data and evaluation time shapes:", data.shape, t_eval.shape)

# %%

class Encoder(eqx.Module):
    layers: list

    def __init__(self, traj_size, context_size, key=None):        ## TODO make this convolutional
        # super().__init__(**kwargs)
        keys = get_new_key(key, num=3)
        self.layers = [eqx.nn.Linear(traj_size, 50, key=keys[0]), jax.nn.softplus,
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

    main_output_net: jnp.ndarray

    def __init__(self, context_size, processor, key=None):
        keys = get_new_key(key, num=3)

        proc_params, self.static = eqx.partition(processor, eqx.is_array)

        flat, self.leave_shapes, self.tree_def = flatten_pytree(proc_params)
        out_size = flat.shape[0]

        self.main_output_net = flat

        # print("Hypernetwork will output", out_size, "parameters")

        self.layers = [eqx.nn.Linear(context_size, 160, key=keys[0]), jax.nn.softplus,
                        eqx.nn.Linear(160, 160*4, key=keys[1]), jax.nn.softplus,
                        eqx.nn.Linear(160*4, out_size, key=keys[2]) ]

    def __call__(self, context):
        weights = context
        # print("Hypernetwork got an input of size:", weights.size)
        for layer in self.layers:
            weights = layer(weights)
        weights = self.main_output_net + weights
        proc_params = unflatten_pytree(weights, self.leave_shapes, self.tree_def)
        return eqx.combine(proc_params, self.static)


class Physics(eqx.Module):
    params: jnp.ndarray

    def __init__(self, key=None):
        # self.params = jnp.abs(jax.random.normal(key, (4,)))
        self.params = jax.random.uniform(key, (4,), minval=0., maxval=3.5)

    def __call__(self, t, x):
        dx0 = x[0]*self.params[0] - x[0]*x[1]*self.params[1]
        dx1 = x[0]*x[1]*self.params[2] - x[1]*self.params[3]
        return jnp.array([dx0, dx1])

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

class Processor(eqx.Module):
    physics: Physics
    augmentation: Augmentation

    def __init__(self, data_size, width_size, depth, key=None):
        keys = get_new_key(key, num=2)
        self.physics = Physics(key=keys[0])
        self.augmentation = Augmentation(data_size, width_size, depth, key=keys[1])

    def __call__(self, t, x):
        # return self.physics(t, x) + self.augmentation(t, x)
        return self.augmentation(t, x)


class NeuralODE(eqx.Module):
    hypernet: Hypernetwork

    def __init__(self, context_size, processor, key=None):
        self.hypernet = Hypernetwork(context_size, processor, key=key)

    def __call__(self, x0, t_eval, xi):
        # print("NeuralODE got an input of size:", x0.shape, t_eval.shape, xi.shape)
        processor = self.hypernet(xi)

        solution = diffrax.diffeqsolve(
                    diffrax.ODETerm(lambda t, x, args: processor(t, x)),
                    diffrax.Tsit5(),
                    t0=t_eval[0],
                    t1=t_eval[-1],
                    dt0=t_eval[1] - t_eval[0],
                    y0=x0,
                    stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
                    saveat=diffrax.SaveAt(ts=t_eval),
                    max_steps=4096*10,
                )
        
        return solution.ys, solution.stats["num_steps"]

class ContraNODE(eqx.Module):
    neural_ode: NeuralODE
    encoder: Encoder            ## TODO Important, this needs to accept variable length trajectorirs. A time series, basically ! 
    traj_size: int              ## Based on the above, this shouldn't be needed

    def __init__(self, proc_data_size, proc_width_size, proc_depth, context_size, traj_size, key=None):
        keys = get_new_key(key, num=3)
        # processor = eqx.nn.MLP(
        #     in_size=proc_data_size,
        #     out_size=proc_data_size,
        #     width_size=proc_width_size,
        #     depth=proc_depth,
        #     activation=jax.nn.softplus,
        #     key=keys[0],
        # )
        # processor = Physics(key=keys[0])
        processor = Processor(proc_data_size, proc_width_size, proc_depth, key=keys[0])

        self.neural_ode = NeuralODE(context_size, processor, key=keys[1])
        self.encoder = Encoder(traj_size*proc_data_size, context_size, key=keys[2])
        self.traj_size = traj_size

    def __call__(self, x0, t_eval, xi):
        traj, nb_steps = self.neural_ode(x0, t_eval, xi)
        new_context = self.encoder(traj[:self.traj_size, :].ravel())
        return traj, nb_steps, new_context


# %%

model_key, training_key, testing_key = get_new_key(SEED, num=3)

model = ContraNODE(proc_data_size=2, 
                   proc_width_size=16, 
                   proc_depth=3, 
                   context_size=2, 
                   traj_size=cutoff_length, 
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

# # Cosine similarity function
# def cosine_similarity(a, b):
#     a_normalized = a / jnp.linalg.norm(a, axis=-1, keepdims=True)
#     b_normalized = b / jnp.linalg.norm(b, axis=-1, keepdims=True)
#     return jnp.sum(a_normalized * b_normalized, axis=-1)

# Contrastive loss function
def contrastive_loss(xi_a, xi_b, positive, temperature):
    return jax.lax.cond(positive[0], 
                        lambda xis: jnp.mean((xis[0]-xis[1])**2),
                        lambda xis: jnp.max(jnp.array([0., respulsive_dist-jnp.mean((xis[0]-xis[1])**2)])), 
                        operand=(xi_a, xi_b))

## Gets the mean of the xi's for each environment
@partial(jax.vmap, in_axes=(None, None, 0))
def meanify_xis(es, xis, e):
    # return jnp.mean(xis[es==e], axis=0)
    return jnp.where(es==e, xis, 0.0).sum(axis=0) / (es==e).sum()

## Main loss function
def loss_fn(params, static, batch):
    # print('\nCompiling function "loss_fn" ...\n')
    a, xi_a, Xa, b, xi_b, Xb, t_eval = batch
    print("Shapes of elements in a batch:", a.shape, xi_a.shape, Xa.shape, b.shape, xi_b.shape, Xb.shape, t_eval.shape, "\n")

    model = eqx.combine(params, static)

    X_hat_a, nb_steps_a, xi_a = jax.vmap(model, in_axes=(0, None, 0))(Xa[:, 0, :], t_eval, xi_a)
    X_hat_b, nb_steps_b, xi_b = jax.vmap(model, in_axes=(0, None, 0))(Xb[:, 0, :], t_eval, xi_b)

    # print("Xa shape:", Xa.shape, "X_hat_a shape:", X_hat_a.shape, "t_eval shape:", t_eval.shape)

    term1 = l2_norm(Xa, X_hat_a) + l2_norm(Xb, X_hat_b)
    term2 = jax.vmap(contrastive_loss, in_axes=(0, 0, 0, None))(xi_a, xi_b, a==b, 1.0).mean()
    # term2 = contrastive_loss(xi_a, xi_b, a==b, 1.0)
    term3 = params_norm(params.neural_ode.hypernet.layers)

    es, xis = jnp.concatenate([a, b]), jnp.concatenate([xi_a, xi_b])
    new_xis = meanify_xis(es, xis, jnp.arange(nb_envs))

    # loss_val = term1 + term2
    loss_val = term1 + term2 + 1e-3*term3

    nb_steps = jnp.sum(nb_steps_a + nb_steps_b)
    return loss_val, (new_xis, nb_steps, term1, term2, term3)


@partial(jax.jit, static_argnums=(1))
def train_step(params, static, batch, opt_state):
    print('\nCompiling function "train_step" ...\n')

    (loss, aux_data), grads  = jax.value_and_grad(loss_fn, has_aux=True)(params, static, batch)

    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss, aux_data


@partial(jax.jit, static_argnums=(2,3))
def make_contrastive_batch(xis, data, cutoff_length, nb_trajs_per_batch_per_env, key):      ## TODO: benchmark and save these btaches to disk
    """ Make contrastive data from a batch of data
        Sample 2 trajectories to all the environments environment (same init condition)
        Constitute a set of n_env positive pairs (same environment)
        For the negative pairs, perform a combinatorial product of the environments """

    def batch_from_2_trajs(key):
        # new_key = get_new_key(key, num=1)
        # traj1, traj2 = jax.random.randint(key, (2,), 0, nb_trajs_per_env)
        traj1, traj2 = np.random.randint(0, nb_trajs_per_env, size=(2))

        # jax.debug.print("Traj1: {} Traj2 {}", traj1, traj2 )

        ## Get the positive pairs
        batch_pos = []
        for e in range(nb_envs):
            a = e
            b = e
            xi_a = xis[a]
            xi_b = xis[b]
            Xa = data[a, traj1:traj1+1, :cutoff_length, :]
            Xb = data[b, traj2:traj2+1, :cutoff_length, :]
            batch_pos.append((a, xi_a, Xa, b, xi_b, Xb))

        ## Get the negative pairs (get a and b from combinatorials)
        batch_neg = []
        for (a,b) in itertools.combinations(range(nb_envs), 2):
            xi_a = xis[a]
            xi_b = xis[b]
            Xa = data[a, traj1:traj1+1, :cutoff_length, :]
            Xb = data[b, traj2:traj2+1, :cutoff_length, :]
            batch_neg.append((a, xi_a, Xa, b, xi_b, Xb))

        return batch_pos+batch_neg

    nb_repeats = nb_trajs_per_batch_per_env//2
    multi_batch = []
    for i in range(nb_repeats):
        _, key = get_new_key(key, num=2)
        multi_batch += batch_from_2_trajs(key)

    random.shuffle(multi_batch)

    # as_, xi_as, Xas, bs, xi_bs, Xbs = zip(*batch)     
    list_of_tuples = zip(*multi_batch)        ## List of size 6, each element is a tuple of size batch_size
    list_of_arrays = map(lambda arr: jnp.vstack(arr), list_of_tuples)

    return list_of_arrays+[t_eval[:cutoff_length]]


# %%

### ==== Vanilla Gradient Descent optimisation ==== ####

if train == True:

    nb_train_steps_per_epoch = nb_trajs_per_env//nb_trajs_per_batch_per_env
    total_steps = nb_epochs * nb_train_steps_per_epoch

    # sched = optax.exponential_decay(init_lr, total_steps, decay_rate)
    # sched = optax.linear_schedule(init_lr, 0, total_steps, 0.25)
    sched = optax.piecewise_constant_schedule(init_value=init_lr,
                    boundaries_and_scales={int(total_steps*0.25):0.5, 
                                            int(total_steps*0.5):0.1,
                                            int(total_steps*0.75):0.5})

    start_time = time.time()

    print(f"\n\n=== Beginning Training ... ===")
    print(f"    Number of trajectories used in a single batch per environemnts: {nb_trajs_per_batch_per_env}")
    print(f"    Actual size of a batch (number of contrastive examples): {batch_size}")
    print(f"    Number of train steps per epoch: {nb_train_steps_per_epoch}")
    print(f"    Number of epochs: {nb_epochs}")
    print(f"    Total number of training steps: {total_steps}")

    opt = optax.adam(sched)
    opt_state = opt.init(params)

    # xis = np.random.normal(size=(nb_envs, 2))
    context_key, batch_key = get_new_key(training_key, num=2)
    xis = jax.random.normal(context_key, (nb_envs, 2))
    init_xis = xis.copy()

    losses = []
    nb_steps = []
    aeqb_sum = 0
    for epoch in range(nb_epochs):

        nb_batches = 0
        loss_sum = jnp.zeros(4)
        nb_steps_eph = 0
        batch_id = 0

        _, batch_key = get_new_key(batch_key, num=2)
        batch_keys = get_new_key(batch_key, num=nb_train_steps_per_epoch)

        for i in range(nb_train_steps_per_epoch):   ## Only two trajectories are used for each train_step
            batch = make_contrastive_batch(xis, data, cutoff_length, nb_trajs_per_batch_per_env, batch_keys[i])
        
            params, opt_state, loss, (xis, nb_steps_val, term1, term2, term3) = train_step(params, static, batch, opt_state)

            # a, _, _, b, _, _, _ = batch

            loss_sum += jnp.array([loss, term1, term2, term3])
            nb_steps_eph += nb_steps_val
            nb_batches += 1

        loss_epoch = loss_sum/nb_batches
        losses.append(loss_epoch)
        nb_steps.append(nb_steps_eph)

        if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
            print(f"    Epoch: {epoch:-5d}      TotalLoss: {loss_epoch[0]:-.8f}     Traj: {loss_epoch[1]:-.8f}      Contrast: {loss_epoch[2]:-.8f}      Params: {loss_epoch[3]:-.5f}", flush=True)
            # print(f"\nPercentage of a==b: {np.mean((a==b).astype(int))*100:.2f}%\n")

    losses = jnp.vstack(losses)
    nb_steps = jnp.array(nb_steps)

    wall_time = time.time() - start_time
    time_in_hmsecs = seconds_to_hours(wall_time)
    print("\nTotal GD training time: %d hours %d mins %d secs" %time_in_hmsecs)

    ## Save the results
    np.save("data/losses_09.npy", losses)
    np.save("data/nb_steps_09.npy", nb_steps)
    np.save("data/xis_09.npy", xis)
    np.save("data/init_xis_09.npy", init_xis)

    model = eqx.combine(params, static)
    eqx.tree_serialise_leaves("data/model_09.eqx", model)

else:
    losses = np.load("data/losses_09.npy")
    nb_steps = np.load("data/nb_steps_09.npy")
    xis = np.load("data/xis_09.npy")
    init_xis = np.load("data/init_xis_09.npy")

    model = eqx.combine(params, static)
    model = eqx.tree_deserialise_leaves("data/model_09.eqx", model)


# %%

def test_model(params, static, batch):
    xi, X0, t_eval = batch

    model = eqx.combine(params, static)
    X_hat, _, _ = model(X0, t_eval, xi)

    return X_hat, _


# e_key, traj_key = get_new_key(testing_key, num=2)
e_key, traj_key = get_new_key(time.time_ns(), num=2)

e = jax.random.randint(e_key, (1,), 0, nb_envs)[0]
# e = np.random.randint(0, nb_envs)
# e = 0
traj = jax.random.randint(traj_key, (1,), 0, nb_trajs_per_env)[0]
# traj = np.random.randint(0, nb_trajs_per_env)
# traj = 100

# test_length = cutoff_length
test_length = nb_steps_per_traj
t_test = t_eval[:test_length]
X = data[e, traj, :test_length, :]

X_hat, _ = test_model(params, static, (xis[e], X[0,:], t_test))

fig, ax = plt.subplot_mosaic('AB;CC;DD;EF', figsize=(6*2, 3.5*4))

ax['A'].plot(t_test, X[:, 0], c="dodgerblue", label="Preys (GT)")
ax['A'].plot(t_test, X_hat[:, 0], ".", c="navy", label="Preys (NODE)")

ax['A'].plot(t_test, X[:, 1], c="violet", label="Predators (GT)")
ax['A'].plot(t_test, X_hat[:, 1], ".", c="purple", label="Predators (NODE)")

ax['A'].set_xlabel("Time")
ax['A'].set_title("Trajectories")
ax['A'].legend()

ax['B'].plot(X[:, 0], X[:, 1], c="turquoise", label="GT")
ax['B'].plot(X_hat[:, 0], X_hat[:, 1], ".", c="teal", label="Neural ODE")
ax['B'].set_xlabel("Preys")
ax['B'].set_ylabel("Predators")
ax['B'].set_title("Phase space")
ax['B'].legend()

mke = np.ceil(losses.shape[0]/100).astype(int)
ax['C'].plot(losses[:,0], label="Total", color="grey", linewidth=3, alpha=1.0)
ax['C'].plot(losses[:,1], "x-", markevery=mke, markersize=3, label="Traj", color="grey", linewidth=1, alpha=0.5)
ax['C'].plot(losses[:,2], "o-", markevery=mke, markersize=3, label="Contrast", color="grey", linewidth=1, alpha=0.5)
# ax['C'].plot(losses[:,3], "^-", markevery=mke, markersize=3, label="Params", color="grey", linewidth=1, alpha=0.5)
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
ax['F'].scatter(xis[:,0], xis[:,1], s=30, c=colors[:nb_envs], marker='X')
for i, (x, y) in enumerate(init_xis):
    ax['E'].annotate(str(i), (x, y), fontsize=8)
for i, (x, y) in enumerate(xis):
    ax['F'].annotate(str(i), (x, y), fontsize=8)
ax['E'].set_title(r'Initial Contexts ($\xi_e$)')
ax['F'].set_title(r'Final Contexts')
# ax['E'].set_xlim(xmin, xmax)
# ax['E'].set_ylim(ymin, ymax)
# ax['F'].set_xlim(xmin, xmax)
# ax['F'].set_ylim(ymin, ymax)

plt.suptitle(f"Results for env={e}, traj={traj}", fontsize=14)

plt.tight_layout()
plt.savefig("data/contrast_node.png", dpi=300, bbox_inches='tight')
plt.show()



# %% [markdown]

# # Preliminary results
# - the final contexts are aligned only when the initial conditions are shared accros environments

# # Conclusion

# %%
