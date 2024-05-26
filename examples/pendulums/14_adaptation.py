
#%%[markdown]
## Alternating Neural ODE and Context Updates for Generalising the Simple Pendulum

### Summary
# - Adaptation

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
from scipy.integrate import solve_ivp
np.set_printoptions(suppress=True)

import equinox as eqx
import diffrax

import matplotlib.pyplot as plt

from neuralhub.utils import *
from neuralhub.integrators import *

import optax
from functools import partial

import os
import time

#%%

SEED = 27

## Integrator hps
# integrator = rk4_integrator

## Optimiser hps
init_lr = 3e-3

## Training hps
print_every = 100
nb_epochs = 1000
# batch_size = 16*8

cutoff = 0.2
context_size = 2

train = True

#%%

if train == True:
    # - make a new folder inside 'data' whose name is the current time
    # data_folder = './data/'
    # dataset_path = "./data/simple_pendulum_big.npz"

    # data_folder = './data/'+time.strftime("%d%m%Y-%H%M%S")+'/'
    # data_folder = './data/00_Alternating/'
    data_folder = './data/08012024-120620/'
    # os.mkdir(data_folder)

    # - save the script in that folder
    script_name = os.path.basename(__file__)
    os.system(f"cp {script_name} {data_folder}");

    # - save the dataset as well
    # dataset_path = "./data/simple_pendulum_big.npz"
    # os.system(f"cp {dataset_path} {data_folder}");

    print("Data folder created successfuly:", data_folder)

else:
    # data_folder = "12AlternatingWorks"
    data_folder = "./data/23122023-121748/"
    print("No training. Loading data and results from:", data_folder)









#%%



def simple_pendulum(t, state, L, g):
    theta, theta_dot = state
    theta_ddot = -(g / L) * np.sin(theta)
    return [theta_dot, theta_ddot]


nb_envs = 1
environments = []

# adapt_key = get_new_key(SEED, num=1)
adapt_key = get_new_key(time.time_ns(), num=1)

# adapt_key = jnp.array([1486095591, 2923071726], dtype=jnp.uint32)

gs = jax.random.uniform(key=adapt_key, minval=15, maxval=25, shape=(nb_envs,))
gs = np.array(gs)

for e in range(nb_envs):
    L = 1
    g = gs[e]
    environments.append({"L": L, "g": g})

n_traj_per_env = 128//4
batch_size = n_traj_per_env
n_steps_per_traj = 201

data = np.zeros((len(environments), n_traj_per_env, n_steps_per_traj, 2))

t_span = (0, 10)  # Shortened time span
t_eval = np.linspace(t_span[0], t_span[-1], n_steps_per_traj)  # Fewer frames

for j in range(n_traj_per_env):
    initial_state = np.concatenate([np.random.uniform(-np.pi/2, np.pi/2, size=(1,)), 
                                    np.random.uniform(-1, 1, size=(1,))])

    for i, selected_params in enumerate(environments):
        solution = solve_ivp(simple_pendulum, t_span, initial_state, args=(selected_params['L'], selected_params['g']), t_eval=t_eval)

        data[i, j, :, :] = solution.y.T

data = data[:, :-4, :, :]
data_test = data[:, -4:, :, :]

nb_trajs_per_env = data.shape[1]
nb_steps_per_traj = data.shape[2]
data_size = data.shape[3]

cutoff_length = int(cutoff*nb_steps_per_traj)


print("Number of adaptation environments:", nb_envs)
print("Parameter in new environment:", gs)

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

model = eqx.tree_deserialise_leaves(data_folder+"model.eqx", model)
params, static = eqx.partition(model, eqx.is_array)

context = Context(nb_envs, context_size, key=context_key)
init_xis = context.params.copy()
# print(context.params)














# %%

def params_norm(params):
    """ norm of the parameters """
    return jnp.array([jnp.linalg.norm(x) for x in jax.tree_util.tree_leaves(params)]).sum()

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
        term2 = params_norm(params.processor.envnet)
        loss_val = term1 + 1e-3*term2
        # loss_val = term1
        return loss_val, (jnp.sum(nb_steps), term1, term2)

    all_loss, (all_nb_steps, all_term1, all_term2) = jax.vmap(loss_for_one_env, in_axes=(0, 0))(Xs[:, :, :, :], context.params)

    return jnp.sum(all_loss*weights), (jnp.sum(all_nb_steps), all_term1, all_term2)     ## TODO Return non-reduced aux data


# @partial(jax.jit, static_argnums=(1))
# def train_step_node(params, static, context, batch, weights, opt_state):
#     print('\nCompiling function "train_step" for neural ode ...\n')

#     (loss, aux_data), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, static, context, batch, weights)

#     updates, opt_state = opt_node.update(grads, opt_state)
#     params = optax.apply_updates(params, updates)

#     return params, context, opt_state, loss, aux_data


@partial(jax.jit, static_argnums=(1))
def train_step_cont(params, static, context, batch, weights, opt_state):
    print('\nCompiling function "train_step" for the context ...\n')

    (loss, aux_data), grads = jax.value_and_grad(loss_fn, argnums=2, has_aux=True)(params, static, context, batch, weights)

    # jax.debug.print("\n\n\n loss {}", loss)

    updates, opt_state = opt_cont.update(grads, opt_state)
    context = optax.apply_updates(context, updates)

    return params, context, opt_state, loss, aux_data



if train == True:

    nb_train_steps_per_epoch = np.ceil(n_traj_per_env/batch_size).astype(int)
    assert nb_train_steps_per_epoch > 0, "Not enough data for a single epoch"
    total_steps = nb_epochs * nb_train_steps_per_epoch

    # sched_node = optax.piecewise_constant_schedule(init_value=init_lr,
    #                         boundaries_and_scales={200:1.0,
    #                                                 int(total_steps*0.25):0.25, 
    #                                                 int(total_steps*0.5):0.25,
    #                                                 int(total_steps*0.75):0.25})

    # opt_node = optax.adam(sched_node)
    # opt_state_node = opt_node.init(params)

    sched_cont = optax.piecewise_constant_schedule(init_value=init_lr,
                            boundaries_and_scales={200:1.0,
                                                    int(total_steps*0.25):0.25, 
                                                    int(total_steps*0.5):0.25,
                                                    int(total_steps*0.75):0.25})

    opt_cont = optax.adam(sched_cont)
    opt_state_cont = opt_cont.init(context)

    print(f"\n\n=== Beginning training neural ODE ... ===")
    print(f"    Number of examples in a batch: {batch_size}")
    print(f"    Number of train steps per epoch: {nb_train_steps_per_epoch}")
    print(f"    Number of training epochs: {nb_epochs}")
    print(f"    Total number of training steps: {total_steps}")

    start_time = time.time()

    # losses_node = []
    losses_cont = []
    # nb_steps_node = []
    nb_steps_cont = []

    _, batch_key = get_new_key(training_key, num=2)

    weights = jnp.ones(nb_envs) / nb_envs

    for epoch in range(nb_epochs):
        nb_batches = 0
        # loss_sum_node = jnp.zeros(1)
        loss_sum_cont = jnp.zeros(1)
        # nb_steps_eph_node = 0
        nb_steps_eph_cont = 0

        for i in range(nb_train_steps_per_epoch):
            batch, batch_key = make_training_batch(i, data, cutoff_length, batch_size, batch_key)

            # params, context, opt_state_node, loss_node, (nb_steps_val_node, term1, term2) = train_step_node(params, static, context, batch, weights, opt_state_node)

            # weights = term1 / jnp.sum(term1)

            # nb_batches += 1

        # for i in range(nb_train_steps_per_epoch):
        #     batch, batch_key = make_training_batch(i, data, cutoff_length, batch_size, batch_key)

            if i%1==0:
                params, context, opt_state_cont, loss_cont, (nb_steps_val_cont, term1, term2) = train_step_cont(params, static, context, batch, weights, opt_state_cont)
                # loss_cont, nb_steps_val_cont = 0., 1.           ## TODO - Mark

            ## construct nb_weights inversely proportional to the losses in term1 (nb_envs, 1)
            term1 = term1 + 1e-8
            weights = term1 / jnp.sum(term1)

            # loss_sum_node += jnp.array([loss_node])
            # nb_steps_eph_node += nb_steps_val_node

            loss_sum_cont += jnp.array([loss_cont])
            nb_steps_eph_cont += nb_steps_val_cont
            nb_batches += 1

        # loss_epoch_node = loss_sum_node/nb_batches
        loss_epoch_cont = loss_sum_cont/nb_batches
        # losses_node.append(loss_epoch_node)
        losses_cont.append(loss_epoch_cont)
        # nb_steps_node.append(nb_steps_eph_node)
        nb_steps_cont.append(nb_steps_eph_cont)

        if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
            print(f"    Epoch: {epoch:-5d}     LossContext: {loss_epoch_cont[0]:-.8f}", flush=True)

    # losses_node = jnp.vstack(losses_node)
    losses_cont = jnp.vstack(losses_cont)
    # nb_steps_node = jnp.array(nb_steps_node)
    nb_steps_cont = jnp.array(nb_steps_cont)


    wall_time = time.time() - start_time
    time_in_hmsecs = seconds_to_hours(wall_time)
    print("\nTotal gradient descent training time: %d hours %d mins %d secs" %time_in_hmsecs)

    # np.savez(data_folder+"histories.npz", losses_node=losses_node, losses_cont=losses_cont, nb_steps_node=nb_steps_node, nb_steps_cont=nb_steps_cont)
    np.savez(data_folder+"histories_adapt.npz", losses_cont=losses_cont, nb_steps_cont=nb_steps_cont)

    model = eqx.combine(params, static)
    # eqx.tree_serialise_leaves(data_folder+"model_adapt.eqx", model)
    eqx.tree_serialise_leaves(data_folder+"context_adapt.eqx", context)

else:
    print("\nNo training, loading model and results from 'data' folder ...\n")

    histories = np.load(data_folder+"histories_adapt.npz")
    losses_cont = histories['losses_cont']
    nb_steps_cont = histories['nb_steps_cont']

    model = eqx.combine(params, static)
    # model = eqx.tree_deserialise_leaves(data_folder+"model_adapt.eqx", model)
    context = eqx.tree_deserialise_leaves(data_folder+"context_adapt.eqx", context)

alter_histories = np.load(data_folder+"histories.npz")
losses_node = alter_histories['losses_node']












# %%

def test_model(model, batch, context):
    X, t_eval = batch

    X_hat, _ = model(X[0, :], t_eval, context)

    return X_hat


e_key, traj_key = get_new_key(time.time_ns(), num=2)

e = jax.random.randint(e_key, (1,), 0, nb_envs)[0]
# e=0
traj = jax.random.randint(traj_key, (1,), 0, data_test.shape[1])[0]

# test_length = cutoff_length
test_length = nb_steps_per_traj
t_test = t_eval[:test_length]
X = data_test[e, traj, :test_length, :]

print("==  Begining testing ... ==")
print("    Environment id:", e)
print("    Trajectory id:", traj)
print("    Length of the original trajectories:", nb_steps_per_traj)
print("    Length of the training trajectories:", cutoff_length)
print("    Length of the testing trajectories:", test_length)

# print(context.params)

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

nb_steps = nb_steps_cont
xis = context.params

mke = np.ceil(losses_cont.shape[0]/100).astype(int)

# ax['C'].plot(losses_node[:,0], label="NodeLoss - During Alternating Updates", color="grey", linewidth=3, alpha=1.0)
ax['C'].plot(losses_cont[:,0], label="NodeLoss - During Context-only Update", color="grey", linewidth=3, alpha=1.0)
# ax['C'].plot(losses_cont[:,0], "x-", markevery=mke, markersize=mks, label="ContextLoss", color="grey", linewidth=1, alpha=0.5)
ax['C'].set_xlabel("Epochs")
ax['C'].set_title("Loss Terms")
ax['C'].set_yscale('log')
ax['C'].legend()

ax['D'].plot(nb_steps, c="brown")
ax['D'].set_xlabel("Epochs")
ax['D'].set_title("Total Number of Steps Taken per Epoch (Proportional to NFEs) - During Context-only Update")
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
print("Initial value of the context:", init_xis[e])
print("Final value of the context:", xis[e])

plt.suptitle(f"Results for env={e}, traj={traj}", fontsize=14)

plt.tight_layout()
plt.savefig(data_folder+"alternating_node.png", dpi=100, bbox_inches='tight')
plt.show()

print("Testing finished. Script, data, figures, and models saved in:", data_folder)

#%%

print(weights)
print(model.processor.physics.params)
