
#%%[markdown]
## Alternating Neural ODE and Context Updates for Generalising the Simple Pendulum

### Summary
# - the context is continuous, and it updates at every step after the neural ode
# - we can keep track of a loss per env, which will prove useful

### Findings
# - no need for large context sizes, 2 is enough
# - the network tends to learn the latter environments better
# - don't overcut the learning rate midway
# - the data increases with the number of environments, and the init lr decreases
# - a huge cutoff leads to instability in the training (a jump in the loss), and potential blowup 

### To Do
# - initial the contexts with the discriminators (from a seperate script)
# - learn the mapping from discriminator context to gd context
# - implement the adaptation stage: use the learned mapping, then fine tune
# - look at the loss per env, and dynamically penalyse the total loss (as weighted sum of the env losses). work this out mathematicaly
# - compare everything with LEADS, CoDA, etc.

## Future Work
# - optimal control of the transferable dynamics
# - add in some probabilistic inference (Turing scheme)


#%%

# import random
import jax

# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_platform_name", "cpu")

print("\n############# Alternating Context-Neural ODEs for Simple Pendulum  #############\n")
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

import optax
from functools import partial

import os
import time
# import cProfile

#%%

SEED = 27

## Integrator hps
# integrator = rk4_integrator

## Optimiser hps
init_lr = 3e-3

## Training hps
print_every = 100
nb_epochs = 2000
batch_size = 128*2

cutoff = 0.2
context_size = 2

train = True

#%%

if train == True:
    # - make a new folder inside 'data' whose name is the current time
    # data_folder = './data/'+time.strftime("%d%m%Y-%H%M%S")+'/'
    data_folder = './data/ExperimentTemp/'
    # os.mkdir(data_folder)

    # - save the script in that folder
    script_name = os.path.basename(__file__)
    os.system(f"cp {script_name} {data_folder}");

    # - save the dataset as well
    dataset_path = "./data/simple_pendulum_big.npz"
    # dataset_path = data_folder+"simple_pendulum_big.npz"
    os.system(f"cp {dataset_path} {data_folder}");

    print("Data folder created successfuly:", data_folder)

else:
    # data_folder = "12AlternatingWorks"
    data_folder = "./data/ExperimentTemp/"
    dataset_path = data_folder+"simple_pendulum_big.npz"
    print("No training. Loading data and results from:", data_folder)

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

# class Augmentation(eqx.Module):
#     layers: list

#     def __init__(self, data_size, width_size, depth, context_size, key=None):
#         keys = generate_new_keys(key, num=3)
#         self.layers = [eqx.nn.Linear(data_size+context_size, width_size, key=keys[0]), jax.nn.softplus,
#         # self.layers = [eqx.nn.Linear(data_size, width_size, key=keys[0]), jax.nn.softplus,                  ## TODO- Mark
#                         eqx.nn.Linear(width_size, width_size, key=keys[1]), jax.nn.softplus,
#                         eqx.nn.Linear(width_size, data_size, key=keys[2])]

#     def __call__(self, t, x, context):
#         y = jnp.concatenate([x, context], axis=0)             ## TODO- Mark
#         # y = x
#         for layer in self.layers:
#             y = layer(y)
#         return y


class Augmentation(eqx.Module):
    layers_data: list
    layers_context: list
    layers_shared: list

    def __init__(self, data_size, width_size, depth, context_size, key=None):
        keys = generate_new_keys(key, num=12)
        self.layers_data = [eqx.nn.Linear(data_size, width_size, key=keys[0]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, width_size, key=keys[10]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, width_size, key=keys[1]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, data_size, key=keys[2])]

        self.layers_context = [eqx.nn.Linear(context_size, width_size*2, key=keys[3]), jax.nn.softplus,
                        eqx.nn.Linear(width_size*2, width_size*2, key=keys[11]), jax.nn.softplus,
                        eqx.nn.Linear(width_size*2, width_size, key=keys[4]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, data_size, key=keys[5])]

        self.layers_shared = [eqx.nn.Linear(data_size*2, width_size, key=keys[6]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, width_size, key=keys[7]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, width_size, key=keys[8]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, data_size, key=keys[9])]


    def __call__(self, t, x, context):

        y = x
        context = context
        for i in range(len(self.layers_data)):
            y = self.layers_data[i](y)
            context = self.layers_context[i](context)

        # y = jnp.concatenate([y, context, y*context], axis=0)
        y = jnp.concatenate([y, context], axis=0)
        for layer in self.layers_shared:
            y = layer(y)
        return y



class Processor(eqx.Module):
    physics: Physics
    envnet: Augmentation

    def __init__(self, data_size, width_size, depth, context_size, key=None):
        keys = generate_new_keys(key, num=5)
        self.physics = Physics(key=keys[0])
        self.envnet = Augmentation(data_size, width_size, depth, context_size, key=keys[1])

    def __call__(self, t, x, context):
        return self.physics(t, x) + self.envnet(t, x, context)
        # return self.envnet(t, x, context)


class NeuralODE(eqx.Module):
    processor: Processor
    invariant: eqx.Module

    def __init__(self, data_size, width_size, depth, context_size, key=None):
        self.processor = Processor(data_size, width_size, depth, context_size, key=key)
        self.invariant = eqx.nn.MLP(data_size, 1, width_size, depth, key=key)

    def __call__(self, x0, t_eval, context):
        solution = diffrax.diffeqsolve(
                    diffrax.ODETerm(self.processor),
                    diffrax.Tsit5(),
                    args=context,
                    t0=t_eval[0],
                    t1=t_eval[-1],
                    dt0=t_eval[1] - t_eval[0],
                    y0=x0,
                    stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-4),
                    saveat=diffrax.SaveAt(ts=t_eval),
                    max_steps=4096*10,
                )
        return solution.ys, solution.stats["num_steps"]

        # rhs = lambda x, t: self.processor(t, x, context)
        # X_hat = integrator(rhs, x0, t_eval, None, None, None, None, None, None)
        # return X_hat, t_eval.size

class Context(eqx.Module):
    params: jnp.ndarray

    def __init__(self, nb_envs, context_size, key=None):
        self.params = jax.random.normal(key, (nb_envs, context_size))
        # self.params = jnp.zeros((nb_envs, context_size))

        ## Load from the contexts.npz file
        # self.params = np.load(data_folder+"enc_contexts.npz")['final']
        # assert self.params.shape == (nb_envs, context_size), "Contexts shape mismatch"



# %%

model_key, context_key, training_key, testing_key = get_new_key(SEED, num=4)

model = NeuralODE(data_size=2, 
                width_size=16*2, 
                depth=3, 
                context_size=context_size, 
                key=model_key)

params, static = eqx.partition(model, eqx.is_array)

context = Context(nb_envs, context_size, key=context_key)
init_xis = context.params.copy()















# %%

def params_norm(params):
    """ norm of the parameters """
    return jnp.array([jnp.linalg.norm(x) for x in jax.tree_util.tree_leaves(params)]).sum()

def spectral_norm(params):
    """ spectral norm of the parameters """
    return jnp.array([jnp.linalg.svd(x, compute_uv=False)[0] for x in jax.tree_util.tree_leaves(params) if jnp.ndim(x)==2]).sum()


# def spectral_norm_estimation(params, nb_iters=5, *, key=None):
#     """ estimating the spectral norm with the power iteration: https://arxiv.org/abs/1802.05957 """
#     matrices = [x for x in jax.tree_util.tree_leaves(params) if jnp.ndim(x)==2]
#     keys = generate_new_keys(key, num=len(matrices))
#     us = [jax.random.normal(k, (x.shape[0],)) for k, x in zip(keys, matrices)]
#     vs = [jax.random.normal(k, (x.shape[1],)) for k, x in zip(keys, matrices)]

#     for _ in range(nb_iters):

#         vs = [x.T@u for x, u in zip(matrices, us)]
#         vs = [v / jnp.linalg.norm(v) for v in vs]
#         us = [x@v for x, v in zip(matrices, vs)]
#         us = [u / jnp.linalg.norm(u) for u in us]

#     sigmas = [u.T@x@v for x, u, v in zip(matrices, us, vs)]
#     return jnp.array(sigmas).sum()

def spectral_norm_estimation(params, nb_iters=5, *, key=None):
    """ estimating the spectral norm with the power iteration: https://arxiv.org/abs/1802.05957 """
    matrices = [x for x in jax.tree_util.tree_leaves(params) if jnp.ndim(x)==2]
    nb_matrices = len(matrices)
    keys = generate_new_keys(key, num=nb_matrices)
    us = [jax.random.normal(k, (x.shape[0],)) for k, x in zip(keys, matrices)]
    vs = [jax.random.normal(k, (x.shape[1],)) for k, x in zip(keys, matrices)]

    for _ in range(nb_iters):
        for i in range(nb_matrices):
            vs[i] = matrices[i].T@us[i]
            vs[i] = vs[i] / jnp.linalg.norm(vs[i])
            us[i] = matrices[i]@vs[i]
            us[i] = us[i] / jnp.linalg.norm(us[i])

    sigmas = [u.T@x@v for x, u, v in zip(matrices, us, vs)]
    return jnp.array(sigmas).sum()



def l2_norm(X, X_hat):
    total_loss = jnp.mean((X - X_hat)**2, axis=-1)   ## TODO mean or sum ? Norm of d-dimensional vectors
    return jnp.sum(total_loss) / (X.shape[-2] * X.shape[-3])


def make_training_batch(batch_id, data, cutoff_length, batch_size, key):
    """ Make a batch """
    traj_start, traj_end = batch_id*batch_size, (batch_id+1)*batch_size
    return (data[:, traj_start:traj_end, :cutoff_length, :], t_eval[:cutoff_length]), key




# ## Main loss function
def loss_fn(params, static, context, batch, weights):
    # print('\nCompiling function "loss_fn" ...\n')
    Xs, t_eval = batch
    print("Shapes of elements in a batch:", Xs.shape, t_eval.shape, "\n")

    model = eqx.combine(params, static)

    def loss_for_one_env(Xs_e, context_e):
        Xs_hat_e, nb_steps = jax.vmap(model, in_axes=(0, None, None))(Xs_e[:, 0, :], t_eval, context_e)
        term1 = l2_norm(Xs_e, Xs_hat_e)

        # term2_1 = params_norm(params.processor.envnet)
        # term2_1 = spectral_norm(params.processor.envnet)
        term2_1 = spectral_norm_estimation(params.processor.envnet, key=training_key)


        xs_e_flat = jnp.reshape(Xs_e, (-1, Xs_e.shape[-1]))
        outputs_e = jax.vmap(model.processor.envnet, in_axes=(None, 0, None))(None, xs_e_flat, context_e)
        term2_2 = jnp.mean(jnp.linalg.norm(outputs_e, axis=-1) / jnp.linalg.norm(xs_e_flat, axis=-1))

        term2 = term2_1 + 1e-0 * term2_2

        loss_val = term1 + 1e-3*term2
        # loss_val = term1
        return loss_val, (jnp.sum(nb_steps), term1, term2)

    all_loss, (all_nb_steps, all_term1, all_term2) = jax.vmap(loss_for_one_env, in_axes=(0, 0))(Xs[:, :, :, :], context.params)

    xs_flat = jnp.reshape(Xs, (-1, Xs.shape[-1]))
    grad_inv = jax.vmap(jax.grad(lambda x1, x2: model.invariant(jnp.array([x1,x2]))[0]))(xs_flat[:,0], xs_flat[:,1])   ## TODO = Wrong. The derivative is wrt to time, not the state
    inv_loss = jnp.mean(jnp.linalg.norm(grad_inv, axis=-1))

    # total_loss = jnp.sum(all_loss*weights) + 1e3*inv_loss
    total_loss = jnp.sum(all_loss*weights)

    return total_loss, (jnp.sum(all_nb_steps), all_term1, all_term2)     ## TODO Return non-reduced aux data


@partial(jax.jit, static_argnums=(1))
def train_step_node(params, static, context, batch, weights, opt_state):
    print('\nCompiling function "train_step" for neural ode ...\n')

    (loss, aux_data), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, static, context, batch, weights)

    updates, opt_state = opt_node.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, context, opt_state, loss, aux_data


@partial(jax.jit, static_argnums=(1))
def train_step_cont(params, static, context, batch, weights, opt_state):
    print('\nCompiling function "train_step" for the context ...\n')

    (loss, aux_data), grads = jax.value_and_grad(loss_fn, argnums=2, has_aux=True)(params, static, context, batch, weights)

    updates, opt_state = opt_cont.update(grads, opt_state)
    context = optax.apply_updates(context, updates)

    return params, context, opt_state, loss, aux_data

# pr = cProfile.Profile()
# pr.enable()

# with jax.profiler.trace("./data/jax-trace", create_perfetto_link=True, create_perfetto_trace=True):

if train == True:

    nb_train_steps_per_epoch = nb_trajs_per_env // batch_size
    assert nb_train_steps_per_epoch > 0, "Not enough data for a single epoch"
    total_steps = nb_epochs * nb_train_steps_per_epoch

    sched_node = optax.piecewise_constant_schedule(init_value=init_lr,
                            boundaries_and_scales={200:1.0,
                                                    int(total_steps*0.25):0.25, 
                                                    int(total_steps*0.5):0.25,
                                                    int(total_steps*0.75):0.25})

    opt_node = optax.adabelief(sched_node)
    opt_state_node = opt_node.init(params)

    sched_cont = optax.piecewise_constant_schedule(init_value=init_lr,
                            boundaries_and_scales={200:1.0,
                                                    int(total_steps*0.25):0.25, 
                                                    int(total_steps*0.5):0.25,
                                                    int(total_steps*0.75):0.25})

    opt_cont = optax.adabelief(sched_cont)
    opt_state_cont = opt_cont.init(context)

    print(f"\n\n=== Beginning training neural ODE ... ===")
    print(f"    Number of examples in a batch: {batch_size}")
    print(f"    Number of train steps per epoch: {nb_train_steps_per_epoch}")
    print(f"    Number of training epochs: {nb_epochs}")
    print(f"    Total number of training steps: {total_steps}")

    start_time = time.time()

    losses_node = []
    losses_cont = []
    nb_steps_node = []
    nb_steps_cont = []

    _, batch_key = get_new_key(training_key, num=2)

    weights = jnp.ones(nb_envs) / nb_envs

    for epoch in range(nb_epochs):
        nb_batches = 0
        loss_sum_node = jnp.zeros(1)
        loss_sum_cont = jnp.zeros(1)
        nb_steps_eph_node = 0
        nb_steps_eph_cont = 0

        for i in range(nb_train_steps_per_epoch):
            batch, batch_key = make_training_batch(i, data, cutoff_length, batch_size, batch_key)

            params, context, opt_state_node, loss_node, (nb_steps_val_node, term1, term2) = train_step_node(params, static, context, batch, weights, opt_state_node)


            weights = term1 / jnp.sum(term1)

            # nb_batches += 1

        # for i in range(nb_train_steps_per_epoch):
        #     batch, batch_key = make_training_batch(i, data, cutoff_length, batch_size, batch_key)

            if i%1==0:
                params, context, opt_state_cont, loss_cont, (nb_steps_val_cont, term1, term2) = train_step_cont(params, static, context, batch, weights, opt_state_cont)
                # loss_cont, nb_steps_val_cont = 0., 1.           ## TODO - Mark

            ## construct nb_weights inversely proportional to the losses in term1 (nb_envs, 1)
            term1 = term1 + 1e-8
            weights = term1 / jnp.sum(term1)

            loss_sum_node += jnp.array([loss_node])
            nb_steps_eph_node += nb_steps_val_node

            loss_sum_cont += jnp.array([loss_cont])
            nb_steps_eph_cont += nb_steps_val_cont
            nb_batches += 1

        loss_epoch_node = loss_sum_node/nb_batches
        loss_epoch_cont = loss_sum_cont/nb_batches
        losses_node.append(loss_epoch_node)
        losses_cont.append(loss_epoch_cont)
        nb_steps_node.append(nb_steps_eph_node)
        nb_steps_cont.append(nb_steps_eph_cont)

        if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
            print(f"    Epoch: {epoch:-5d}      LossNeuralODE: {loss_epoch_node[0]:-.8f}     LossContext: {loss_epoch_cont[0]:-.8f}", flush=True)

    losses_node = jnp.vstack(losses_node)
    losses_cont = jnp.vstack(losses_cont)
    nb_steps_node = jnp.array(nb_steps_node)
    nb_steps_cont = jnp.array(nb_steps_cont)


    wall_time = time.time() - start_time
    time_in_hmsecs = seconds_to_hours(wall_time)
    print("\nTotal gradient descent training time: %d hours %d mins %d secs" %time_in_hmsecs)

    np.savez(data_folder+"histories.npz", losses_node=losses_node, losses_cont=losses_cont, nb_steps_node=nb_steps_node, nb_steps_cont=nb_steps_cont)

    model = eqx.combine(params, static)
    eqx.tree_serialise_leaves(data_folder+"model.eqx", model)
    eqx.tree_serialise_leaves(data_folder+"context.eqx", context)

else:
    print("\nNo training, loading model and results from 'data' folder ...\n")

    histories = np.load(data_folder+"histories.npz")
    losses_node = histories['losses_node']
    losses_cont = histories['losses_cont']
    nb_steps_node = histories['nb_steps_node']
    nb_steps_cont = histories['nb_steps_cont']

    model = eqx.combine(params, static)
    model = eqx.tree_deserialise_leaves(data_folder+"model.eqx", model)
    context = eqx.tree_deserialise_leaves(data_folder+"context.eqx", context)


# jax.profiler.stop_trace()

# pr.disable()
# pr.dump_stats("data/program.prof")















# %%

def test_model(model, batch, context):
    X, t_eval = batch

    X_hat, _ = model(X[0, :], t_eval, context)

    return X_hat


e_key, traj_key = get_new_key(time.time_ns(), num=2)

e = jax.random.randint(e_key, (1,), 0, nb_envs)[0]
# e=0
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

X_hat = test_model(model, (X, t_test), context.params[e])

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

nb_steps = nb_steps_node
xis = context.params

mke = np.ceil(losses_node.shape[0]/100).astype(int)

ax['C'].plot(losses_node[:,0], label="NodeLoss", color="grey", linewidth=3, alpha=1.0)
ax['C'].plot(losses_cont[:,0], "x-", markevery=mke, markersize=mks, label="ContextLoss", color="grey", linewidth=1, alpha=0.5)
ax['C'].set_xlabel("Epochs")
ax['C'].set_title("Loss Terms")
ax['C'].set_yscale('log')
ax['C'].legend()

ax['D'].plot(nb_steps, c="brown")
ax['D'].set_xlabel("Epochs")
ax['D'].set_title("Total Number of Steps Taken per Epoch (Proportional to NFEs)")
ax['D'].set_yscale('log')

eps = 0.1
colors = ['dodgerblue', 'r', 'b', 'g', 'm', 'c', 'y', 'orange', 'purple', 'brown']
colors = colors*(nb_envs)

ax['F'].scatter(xis[:,0], xis[:,1], s=50, c=colors[:nb_envs], marker='o')
for i, (x, y) in enumerate(xis[:, :2]):
    ax['F'].annotate(str(i), (x, y), fontsize=8)
ax['F'].set_title(r'Final Contexts ($\xi^e$)')


ax['E'].scatter(init_xis[:,0], init_xis[:,1], s=30, c=colors[:nb_envs], marker='X')
ax['F'].scatter(xis[:,0], xis[:,1], s=50, c=colors[:nb_envs], marker='o')
for i, (x, y) in enumerate(init_xis[:, :2]):
    ax['E'].annotate(str(i), (x, y), fontsize=8)
for i, (x, y) in enumerate(xis[:, :2]):
    ax['F'].annotate(str(i), (x, y), fontsize=8)
ax['E'].set_title(r'Initial Contexts (first 2 dims)')
ax['F'].set_title(r'Final Contexts (first 2 dims)')

np.savez(data_folder+"node_contexts.npz", initial=init_xis, final=xis)

plt.suptitle(f"Results for env={e}, traj={traj}", fontsize=14)

plt.tight_layout()
plt.savefig(data_folder+"alternating_node.png", dpi=100, bbox_inches='tight')
plt.show()

print("Testing finished. Script, data, figures, and models saved in:", data_folder)

#%%

# print(weights)

print(model.processor.physics.params)



#%%

## Level set of the invariant

lim = 5

# x = np.linspace(-lim, lim, 100)
# y = np.linspace(-lim, lim, 100)

e=14

xs_flat = np.reshape(data[e,::100, :, :], (-1, data[e].shape[-1]))
x = xs_flat[:,0]
y = xs_flat[:,1]
X, Y = np.meshgrid(x, y)

Z = jax.vmap(model.invariant)(np.vstack([X.ravel(), Y.ravel()]).T)
Z = np.array(Z).reshape(X.shape)

plt.clf()  # Clear the plot for the next frame
# plt.contourf(X, Y, Z, levels=50, cmap='gray')
plt.contour(X, Y, Z, levels=12, cmap='viridis')
# plt.contourf(X, Y, Z, levels=[0, 1], cmap='gray')
plt.colorbar(label=r'Invariant $I(\theta,\dot\theta)$');

# contour = plt.contour(X, Y, Z, levels=10, cmap='Blues')
# plt.clabel(contour, inline=True, fontsize=8, fmt=r'$I(\theta,\dot\theta)={:1.0f}$'.format(contour.levels[0]))

#%%
## Plot Z against X and Y in 3D

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
