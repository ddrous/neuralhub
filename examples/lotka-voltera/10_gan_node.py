
#%%[markdown]
# # GAN-Neural ODE framework for generalising the Lotka-Volterra systems
# List of ToDOs
# - Use a list of discriminators, rather than a vmapped ensemble. This makes it easier to add a new discriminator during adaptation
# - Use a time series as input to the discriminators, rather than a single point
# - Initialise the discriminators on their own trajectories in advance, before training as a whole
# - Put (concatenate) the context back in before each layer of the generator (neural ODE)

### Summary


#%%

# import random
import jax

# from jax import config
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_platform_name", "cpu")

print("\n############# Lotka-Volterra with Generators and Discriminators #############\n")
print("Available devices", jax.devices())

import jax.numpy as jnp
# import jax.scipy as jsp
# import jax.scipy.optimize

import numpy as np
np.set_printoptions(suppress=True)
# from scipy.integrate import solve_ivp

import equinox as eqx
import diffrax

import matplotlib.pyplot as plt

from graphpint.utils import *
from graphpint.integrators import *

import optax
from functools import partial
import time
# from typing import List, Tuple, Callable


#%%

SEED = 23
# SEED = np.random.randint(0, 1000)

## Integrator hps
integrator = rk4_integrator

## Optimiser hps
init_lr = 3e-4

## Training hps
print_every = 10
nb_epochs = 100
batch_size = 9*1       ## 9 is the number of environments

cutoff = 0.1

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

dataset = np.load('./data/lotka_volterra_small.npz')
data, t_eval = dataset['X'], dataset['t']

nb_envs = data.shape[0]
nb_trajs_per_env = data.shape[1]
nb_steps_per_traj = data.shape[2]
data_size = data.shape[3]

cutoff_length = int(cutoff*nb_steps_per_traj)

print("Data and evaluation time shapes:", data.shape, t_eval.shape)

# %%


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

class SharedProcessor(eqx.Module):
    physics: Physics
    augmentation: Augmentation

    def __init__(self, data_size, width_size, depth, key=None):
        keys = get_new_key(key, num=2)
        self.physics = Physics(key=keys[0])
        self.augmentation = Augmentation(data_size, width_size, depth, key=keys[1])

    def __call__(self, t, x):
        # return self.physics(t, x) + self.augmentation(t, x)
        return self.augmentation(t, x)
        # return self.physics(t, x)

class EnvProcessor(eqx.Module):
    layers: list

    def __init__(self, data_size, width_size, depth, context_size, key=None):
        keys = get_new_key(key, num=3)
        self.layers = [eqx.nn.Linear(data_size+context_size, width_size, key=keys[0]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, width_size, key=keys[1]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, data_size, key=keys[2])]

    def __call__(self, t, x, context):
        # y = jnp.concatenate([jnp.broadcast_to(t, (1,)), x], axis=0)
        # y = x

        # jax.debug.print("\n\n\n\n\n\nx shape {} context {}\n\n\n\n\n\n", x, context)
        # jax.debug.breakpoint()

        y = jnp.concatenate([x, context], axis=0)
        for layer in self.layers:
            y = layer(y)
        return y

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
                    stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
                    saveat=diffrax.SaveAt(ts=t_eval),
                    max_steps=4096*10,
                )
        return solution.ys, solution.stats["num_steps"]

        # rhs = lambda x, t: self.processor(t, x, context)
        # X_hat = integrator(rhs, x0, t_eval, None, None, None, None, None, None)
        # return X_hat, t_eval.size






class Discriminator(eqx.Module):
    layers: list
    proba_layers: eqx.nn.Linear

    def __init__(self, traj_size, context_size, key=None):        ## TODO make this convolutional
        # super().__init__(**kwargs)
        keys = get_new_key(key, num=4)
        self.layers = [eqx.nn.Linear(traj_size, 50, key=keys[0]), jax.nn.softplus,
                        eqx.nn.Linear(50, 20, key=keys[1]), jax.nn.softplus,
                        eqx.nn.Linear(20, context_size, key=keys[2]) ]
        self.proba_layers = [eqx.nn.Linear(context_size, 1, key=keys[3]), jax.nn.sigmoid]

    def __call__(self, traj):
        # print("Encoder got and input of size:", traj.size)
        context = traj
        for layer in self.layers:
            context = layer(context)

        proba = context
        for layer in self.proba_layers:
            proba = layer(proba)

        return proba, context

@eqx.filter_vmap(in_axes=(None, None, 0))
def init_discriminator_ensemble(traj_size, context_size, key):
    return Discriminator(traj_size, context_size, key=key)

@eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
def evaluate_discriminator_ensemble(discriminator, traj):
    return discriminator(traj)




class GANNODE(eqx.Module):
    generator: Generator
    discriminators: Discriminator       ## TODO, rather, an ensemble of discriminators. A list might be better ?
    traj_size: int              ## Based on the above, this shouldn't be needed. TODO: use a time series instead

    def __init__(self, proc_data_size, proc_width_size, proc_depth, context_size, traj_size, nb_envs, key=None):
        keys = get_new_key(key, num=1+nb_envs)

        self.generator = Generator(proc_data_size, proc_width_size, proc_depth, context_size, key=keys[1])
        self.discriminators = init_discriminator_ensemble(traj_size*proc_data_size, context_size, keys[1:])
        self.traj_size = traj_size

    def __call__(self, x0, t_eval, xi):
        traj, nb_steps = self.generator(x0, t_eval, xi)
        probas, contexts = evaluate_discriminator_ensemble(self.discriminators, traj[:self.traj_size, :].ravel())
        return traj, nb_steps, probas, contexts         ## TODO: even tho all contexts are returned, only the corresponding ones should be used for the loss


# %%

model_key, training_key, testing_key = get_new_key(SEED, num=3)

model = GANNODE(proc_data_size=2, 
                proc_width_size=16, 
                proc_depth=3, 
                context_size=2, 
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

# ## Gets the mean of the xi's for each environment
# @partial(jax.vmap, in_axes=(None, None, 0))
# def meanify_xis(es, xis, e):
#     jax.debug.breakpoint()
#     return jnp.where(es==e, xis, 0.0).sum(axis=0) / (es==e).sum()


## Gets the mean of the xi's for each environment
@partial(jax.vmap, in_axes=(0, None, None, None))
def meanify_xis(e, es, xis, orig_xis):      ## TODO: some xi's get updated, others don't
    return jax.lax.cond((e==es).sum()>0, 
                        lambda e: jnp.where(es==e, xis, 0.0).sum(axis=0) / (es==e).sum(), 
                        lambda e: orig_xis[e], 
                        e)


## Main loss function
def loss_fn(params, static, batch):
    # print('\nCompiling function "loss_fn" ...\n')
    e, xi, X, t_eval = batch
    print("Shapes of elements in a batch:", e.shape, xi.shape, X.shape, t_eval.shape, "\n")

    model = eqx.combine(params, static)

    X_hat, nb_steps, probas, xis = jax.vmap(model, in_axes=(0, None, 0))(X[:, 0, :], t_eval, xi)

    probas_hat = jnp.max(probas, axis=1)        ## TODO: use this for cross-entropy loss
    es_hat = jnp.argmax(probas, axis=1)
    xis_hat = xis[jnp.arange(X.shape[0]), es_hat.squeeze()]

    # new_xis = meanify_xis(es_hat, xis_hat, jnp.arange(nb_envs))
    new_xis = meanify_xis(jnp.arange(nb_envs), es_hat, xis_hat, xi)

    # print("New xis et al shape is:", new_xis.shape, xis_hat.shape, xis.shape, es_hat.shape, "\n")

    term1 = l2_norm(X, X_hat)
    term2 = jnp.mean((es_hat - e)**2, dtype=jnp.float32)
    term3 = params_norm(params.generator.processor.env)

    loss_val = term1 + term2
    # loss_val = term1 + term2 + 1e-3*term3

    return loss_val, (new_xis, nb_steps, term1, term2, term3)


@partial(jax.jit, static_argnums=(1))
def train_step(params, static, batch, opt_state):
    print('\nCompiling function "train_step" ...\n')

    (loss, aux_data), grads  = jax.value_and_grad(loss_fn, has_aux=True)(params, static, batch)

    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss, aux_data


# @partial(jax.jit, static_argnums=(2,3))
def make_training_batch(batch_id, xis, data, cutoff_length, batch_size, key):      ## TODO: benchmark and save these btaches to disk
    """ Make a batch """

    nb_trajs_per_batch_per_env = batch_size//nb_envs
    traj_start, traj_end = batch_id*nb_trajs_per_batch_per_env, (batch_id+1)*nb_trajs_per_batch_per_env

    es_batch = []
    xis_batch = []
    X_batch = []

    for e in range(nb_envs):
        es_batch.append(jnp.ones(nb_trajs_per_batch_per_env, dtype=int)*e)
        xis_batch.append(jnp.ones((nb_trajs_per_batch_per_env, 2))*xis[e:e+1, :])
        X_batch.append(data[e, traj_start:traj_end, :cutoff_length, :])

    return jnp.concatenate(es_batch), jnp.vstack(xis_batch), jnp.vstack(X_batch), t_eval[:cutoff_length]


# %%

### ==== Vanilla Gradient Descent optimisation ==== ####

if train == True:

    nb_trajs_per_batch_per_env = batch_size//nb_envs
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

        _, batch_key = get_new_key(batch_key, num=2)
        batch_keys = get_new_key(batch_key, num=nb_train_steps_per_epoch)

        for i in range(nb_train_steps_per_epoch):   ## Only two trajectories are used for each train_step
            batch = make_training_batch(i, xis, data, cutoff_length, batch_size, batch_keys[i])
        
            params, opt_state, loss, (xis, nb_steps_val, term1, term2, term3) = train_step(params, static, batch, opt_state)

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
    np.save("data/losses_10.npy", losses)
    np.save("data/nb_steps_10.npy", nb_steps)
    np.save("data/xis_10.npy", xis)
    np.save("data/init_xis_10.npy", init_xis)

    model = eqx.combine(params, static)
    eqx.tree_serialise_leaves("data/model_10.eqx", model)

else:
    losses = np.load("data/losses_10.npy")
    nb_steps = np.load("data/nb_steps_10.npy")
    xis = np.load("data/xis_10.npy")
    init_xis = np.load("data/init_xis_10.npy")

    model = eqx.combine(params, static)
    model = eqx.tree_deserialise_leaves("data/model_10.eqx", model)


# %%

def test_model(params, static, batch):
    xi, X0, t_eval = batch

    model = eqx.combine(params, static)
    X_hat, _, _, _ = model(X0, t_eval, xi)

    return X_hat, _


# e_key, traj_key = get_new_key(testing_key, num=2)
e_key, traj_key = get_new_key(time.time_ns(), num=2)

e = jax.random.randint(e_key, (1,), 0, nb_envs)[0]
# e = 0
traj = jax.random.randint(traj_key, (1,), 0, nb_trajs_per_env)[0]
# traj = 100

test_length = cutoff_length
# test_length = nb_steps_per_traj
t_test = t_eval[:test_length]
X = data[e, traj, :test_length, :]

print("== Testing begining ==")
print("    Environment id:", e)
print("    Trajectory id:", traj)
print("    Length of the original trajectories:", nb_steps_per_traj)
print("    Length of the training trajectories:", cutoff_length)
print("    Length of the testing trajectories:", test_length)


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
ax['C'].plot(losses[:,2], "o-", markevery=mke, markersize=3, label="Discrim", color="grey", linewidth=1, alpha=0.5)
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
plt.savefig("data/gan_node.png", dpi=300, bbox_inches='tight')
plt.show()



# %% [markdown]

# # Preliminary results
# - please Lord !

# # Conclusion

# %%
