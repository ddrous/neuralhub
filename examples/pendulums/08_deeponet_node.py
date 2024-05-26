
#%%[markdown]
# # DeepO-Neural ODE framework for generalising the Simple Pendulum

### Summary


#%%

# import random
import jax

# from jax import config
jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_platform_name", "cpu")

print("\n############# Deep)Net and Neural ODEs for Simple Pendulum  #############\n")
print("Jax version:", jax.__version__)
print("Available devices:", jax.devices())

import jax.numpy as jnp

import numpy as np
np.set_printoptions(suppress=True)

import equinox as eqx
import diffrax

import matplotlib.pyplot as plt

from neuralhub.utils import *
from neuralhub.integrators import *

import pandas as pd
import seaborn as sns

import optax
from functools import partial

import os
import time

#%%

SEED = 27

## Integrator hps
integrator = rk4_integrator

## Optimiser hps
init_lr = 3e-2

## Training hps
print_every = 100
nb_epochs_cal = 300
nb_epochs = 10
# batch_size = 2*128*10       ## 2 is the number of environments

cutoff = 0.4
context_size = 20
skip_trajs = 8*8*4      ## Skip some trajectories to feed to the DeepO-Net branch

train = True

#%%

# - make a new folder inside 'data' whose name is the currennt time
data_folder = './data/'+time.strftime("%d%m%Y-%H%M%S")+'/'
os.mkdir(data_folder)

# - save the script in that folder
script_name = os.path.basename(__file__)
os.system(f"cp {script_name} {data_folder}");

# - save the dataset as well
dataset_path = "./data/simple_pendulum_envs.npz"
os.system(f"cp {dataset_path} {data_folder}");

print("Data folder created successfuly:", data_folder)

#%%

dataset = np.load(dataset_path)
data, t_eval = dataset['X'][:, :, :, :], dataset['t']

nb_envs = data.shape[0]
nb_trajs_per_env = data.shape[1]
nb_steps_per_traj = data.shape[2]
data_size = data.shape[3]

batch_size = nb_envs*nb_trajs_per_env
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

class MLP(eqx.Module):
    layers: list

    def __init__(self, in_size, out_size, width_size, depth, key=None):
        keys = generate_new_keys(key, num=3)
        self.layers = [eqx.nn.Linear(in_size, width_size, key=keys[0]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, width_size, key=keys[1]), jax.nn.tanh,
                        eqx.nn.Linear(width_size, out_size, key=keys[2])]

    def __call__(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y


class Processor(eqx.Module):
    physics: Physics
    branch: MLP      ## of the DeepO-Net
    trunk: MLP       ## of the DeepO-Net

    def __init__(self, m, p, data_size, width_size, depth, key=None):
        keys = generate_new_keys(key, num=5)
        self.physics = Physics(key=keys[0])
        self.branch = MLP(m, p, width_size, depth, key=keys[1])
        self.trunk = MLP(data_size, p*data_size, width_size, depth, key=keys[2])


    def setup_branch(self, discriminator, trajs):
        batched_discriminator = jax.vmap(discriminator, in_axes=(0))
        trajs = jnp.reshape(trajs, (trajs.shape[0]*trajs.shape[1], trajs.shape[2], -1))
        input_branch, _ = batched_discriminator(trajs)
        return input_branch

    def __call__(self, t, x, input_branch):

    # def __call__(self, t, x, args):
    #     discriminator, trajs = args
    #     batched_discriminator = jax.vmap(discriminator, in_axes=(0))
    #     trajs = jnp.reshape(trajs, (trajs.shape[0]*trajs.shape[1], trajs.shape[2], -1))
    #     input_branch, _ = batched_discriminator(trajs)

        ps = self.physics(t, x)
        bs = self.branch(input_branch.squeeze())
        ts = self.trunk(x).reshape((x.shape[0], -1))

        return ps + ts@bs
        # return ts@bs


class Generator(eqx.Module):
    processor: Processor

    def __init__(self, m, p, data_size, width_size, depth, key=None):
        self.processor = Processor(m, p, data_size, width_size, depth, key=key)

    def __call__(self, x0, t_eval, discriminator, trajs):

        input_branch = self.processor.setup_branch(discriminator, trajs)

        solution = diffrax.diffeqsolve(
                    diffrax.ODETerm(self.processor),
                    diffrax.Tsit5(),
                    args=input_branch,
                    t0=t_eval[0],
                    t1=t_eval[-1],
                    dt0=t_eval[1] - t_eval[0],
                    y0=x0,
                    stepsize_controller=diffrax.PIDController(rtol=1e-1, atol=1e-2),
                    saveat=diffrax.SaveAt(ts=t_eval),
                    max_steps=4096*10,
                )
        return solution.ys, solution.stats["num_steps"]

        # rhs = lambda x, t: self.processor(t, x, input_branch)
        # X_hat = integrator(rhs, x0, t_eval, None, None, None, None, None, None)
        # return X_hat, t_eval.size






class Discriminator(eqx.Module):
    layers: list
    proba_layers: list

    def __init__(self, traj_size, context_size, key=None):        ## TODO make this convolutional
        # print("Creating discriminator with traj size:", traj_size, "and context size:", context_size, "\n")

        keys = get_new_key(key, num=4)
        self.layers = [eqx.nn.Linear(traj_size, 100, key=keys[0]), jax.nn.softplus,
                        eqx.nn.Linear(100, 50, key=keys[1]), jax.nn.tanh,
                        eqx.nn.Linear(50, context_size, key=keys[2]) ]
        self.proba_layers = [eqx.nn.Linear(context_size, 1, key=keys[3]), jax.nn.sigmoid]

    def __call__(self, traj):
        # print("Encoder got and input of size:", traj.size)
        input_dat = jnp.reshape(jnp.transpose(traj, axes=(1,0)), (traj.size, ))
        context = input_dat
        for layer in self.layers:
            context = layer(context)

        proba = context
        for layer in self.proba_layers:
            proba = layer(proba)

        return proba, context


class DeepONODE(eqx.Module):
    generator: Generator
    discriminators: list
    traj_size: int

    def __init__(self, m, p, proc_data_size, proc_width_size, proc_depth, context_size, traj_size, nb_envs, key=None):
        keys = get_new_key(key, num=1+nb_envs)

        self.generator = Generator(m, p, proc_data_size, proc_width_size, proc_depth, key=keys[1])        
        self.discriminators = [Discriminator(traj_size*proc_data_size, context_size, key=key) for key in keys[1:]]
        
        self.traj_size = traj_size

    # def __call__(self, x0, t_eval, discriminator, trajs):

    #     traj, nb_steps = self.generator(x0, t_eval, discriminator, trajs)

    #     probas = []
    #     contexts = []
    #     for discriminator in self.discriminators:
    #         proba, context = discriminator(traj[:self.traj_size, :].ravel())
    #         probas.append(proba)
    #         contexts.append(context)

    #     return traj, nb_steps, jnp.concatenate(probas), jnp.vstack(contexts)


# %%

model_key, training_key, testing_key = get_new_key(SEED, num=3)

model = DeepONODE(m=nb_envs*nb_trajs_per_env//skip_trajs,            ## TODO Only the first trajectories in each env will be used to train the DeepONet
                p=256,
                proc_data_size=2, 
                proc_width_size=16*2, 
                proc_depth=3, 
                context_size=context_size, 
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
    return jnp.sum(total_loss) / (X.shape[-2] * X.shape[-3])

@partial(jax.vmap, in_axes=(0, None, None, None, None))
def meanify_xis(e, es_hat, xis_hat, es_orig, xis_orig):      ## TODO: some xi's get updated, others don't
    # print("All shapes before meaninfying:", es_hat.shape, es_orig.shape, xis_hat.shape, xis_orig.shape, "\n")
    return jax.lax.cond((es_hat==e).sum()>0, 
                        lambda e: jnp.where((es_hat==e), xis_hat, 0.0).sum(axis=0) / (es_hat==e).sum(), 
                        lambda e: jnp.where((es_orig==e), xis_orig, 0.0).sum(axis=0) / (es_orig==e).sum(),      ## TODO, always make sure the batch is representative of all environments
                        e)



def make_training_batch_cal(batch_id, xis, data, cutoff_length, batch_size, key):      ## TODO: benchmark and save these btaches to disk
    """ Make a batch """

    nb_trajs_per_batch_per_env = batch_size//nb_envs
    traj_start, traj_end = batch_id*nb_trajs_per_batch_per_env, (batch_id+1)*nb_trajs_per_batch_per_env

    es_batch = []
    xis_batch = []
    X_batch = []

    for e in range(nb_envs):
        es_batch.append(jnp.ones((nb_trajs_per_batch_per_env,), dtype=int)*e)
        xis_batch.append(jnp.ones((nb_trajs_per_batch_per_env, context_size))*xis[e:e+1, :])
        X_batch.append(data[e, traj_start:traj_end, :cutoff_length, :])

    return (jnp.concatenate(es_batch), jnp.vstack(xis_batch), jnp.vstack(X_batch), t_eval[:cutoff_length]), key



# @partial(jax.jit, static_argnums=(2,3))





# %%

### ==== Calibration of the discriminators with real trajectories ==== ####


## Main loss function
def loss_fn_cal(params, static, batch):
    # print('\nCompiling function "loss_fn" ...\n')
    es, xis, Xs, t_eval = batch
    print("Shapes of elements in a batch:", es.shape, xis.shape, Xs.shape, t_eval.shape, "\n")

    model = eqx.combine(params, static)

    X_input = Xs[:, :cutoff_length, :]
    # X_input = jnp.reshape(jnp.transpose(Xs[:, :cutoff_length, :], axes=(0,2,1)), (Xs.shape[0], -1))

    probas = []
    contexts = []
    for discriminator in model.discriminators:
        proba, context = jax.vmap(discriminator)(X_input)
        probas.append(proba)
        contexts.append(context[:, None, :])

    probas, xis_hat = jnp.concatenate(probas, axis=1), jnp.concatenate(contexts, axis=1)

    cross_ent = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(probas, es))

    es_hat = jnp.argmax(probas, axis=1)

    pen_xis = jnp.mean(xis_hat**2)

    xis_hat = xis_hat[jnp.arange(Xs.shape[0]), es_hat]
    new_xis = meanify_xis(jnp.arange(nb_envs), es_hat[:,None], xis_hat, es[:,None], xis)

    # loss_val = cross_ent
    # loss_val = error_es + 1e-3*pen_params
    # loss_val = cross_ent + 1e-3*pen_params
    loss_val = cross_ent + pen_xis

    return loss_val, (new_xis)


@partial(jax.jit, static_argnums=(1))
def train_step_cal(params, static, batch, opt_state):
    print('\nCompiling function "train_step" ...\n')

    (loss, aux_data), grads  = jax.value_and_grad(loss_fn_cal, has_aux=True)(params, static, batch)

    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss, aux_data

def test_ste_cal(params, static):
    ## Test the discriminator on the training data.

    model = eqx.combine(params, static)

    true = []
    pred = []
    colors = ['dodgerblue', 'r', 'b', 'g', 'm', 'c', 'y', 'orange', 'purple', 'brown']
    colors = colors*(nb_envs)
    cols = []
    for e in range(nb_envs):
        # X_input = jnp.reshape(jnp.transpose(data[e, :, :cutoff_length, :], axes=(0,2,1)), (data.shape[1], -1))
        X_input = data[e, :, :cutoff_length, :]

        probas = []
        for discriminator in model.discriminators:
            proba, _ = jax.vmap(discriminator)(X_input)
            probas.append(proba)
        pred += jnp.argmax(jnp.concatenate(probas, axis=1), axis=1).tolist()

        true += [e]*data.shape[1]
        cols += [colors[e]]*data.shape[1]

    return true, pred, cols




if train == True:

    nb_trajs_per_batch_per_env = batch_size//nb_envs
    nb_train_steps_per_epoch = nb_trajs_per_env//nb_trajs_per_batch_per_env
    assert nb_train_steps_per_epoch > 0, "Batch size is too large"

    total_steps_cal = nb_epochs_cal * nb_train_steps_per_epoch

    # sched = init_lr
    sched = optax.exponential_decay(init_lr, total_steps_cal, 0.95)
    opt = optax.adam(sched)
    opt_state = opt.init(params)

    print(f"\n\n=== Beginning calibration of the discriminators ... ===")
    print(f"    Number of trajectories used in a single batch per environemnts: {nb_trajs_per_batch_per_env}")
    print(f"    Actual size of a batch (number of examples for all envs): {batch_size}")
    print(f"    Number of train steps per epoch: {nb_train_steps_per_epoch}")
    print(f"    Number of calibration epochs: {nb_epochs_cal}")
    print(f"    Total number of calibration steps: {total_steps_cal}")

    start_time = time.time()

    # xis = np.random.normal(size=(nb_envs, 2))
    context_key, batch_key = get_new_key(training_key, num=2)
    xis = jax.random.normal(context_key, (nb_envs, context_size))
    init_xis = xis.copy()

    losses_cal = []
    for epoch in range(nb_epochs_cal):

        nb_batches = 0
        loss_sum = 0

        # _, batch_key = get_new_key(batch_key, num=2)
        # batch_keys = get_new_key(batch_key, num=nb_train_steps_per_epoch)

        for i in range(nb_train_steps_per_epoch):   ## Only two trajectories are used for each train_step

            batch, batch_key = make_training_batch_cal(i, xis, data, cutoff_length, batch_size, batch_key)
        
            params, opt_state, loss, (xis) = train_step_cal(params, static, batch, opt_state)

            loss_sum += loss
            nb_batches += 1

        loss_epoch = loss_sum/nb_batches
        losses_cal.append(loss_epoch)

        if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs_cal-1:
            print(f"    Epoch: {epoch:-5d}      CalibLoss: {loss_epoch:-.8f}", flush=True)

    losses_cal = jnp.vstack(losses_cal)

    wall_time = time.time() - start_time
    time_in_hmsecs = seconds_to_hours(wall_time)

    print("\nCalibration step training time: %d hours %d mins %d secs" %time_in_hmsecs)


else:
    print("\nNo training, skipping calibration step ...\n")



# %%

if train == True:
    fig, ax = plt.subplot_mosaic('AA;CD', figsize=(6*2, 4.5*2), width_ratios=[1, 1])

    ax['A'].plot(losses_cal[:], label="Cross-Entropy", color="brown", linewidth=3, alpha=1.0)
    ax['A'].set_xlabel("Epochs")
    ax['A'].set_title("Calibration Loss")
    ax['A'].set_yscale('log')
    ax['A'].legend()

    true, pred, cols = test_ste_cal(params, static)

    df = pd.DataFrame({'True':true[:], 'Pred':pred[:]})
    df = df.groupby(['True', 'Pred']).size().reset_index(name='Counts')
    df = df.pivot(index='Pred', columns='True', values='Counts')
    sns.heatmap(df, annot=False, ax=ax['C'], cmap='YlOrBr', cbar=False, fmt="n")
    ax['C'].set_title("True vs. Pred for all Envs and Discriminators")
    ax['C'].invert_yaxis()

    xis_all = np.vstack([xis, init_xis])
    eps = 0.1
    xmin, xmax = xis_all[:,0].min()-eps, xis_all[:,0].max()+eps
    ymin, ymax = xis_all[:,1].min()-eps, xis_all[:,1].max()+eps
    colors = ['dodgerblue', 'r', 'b', 'g', 'm', 'c', 'y', 'orange', 'purple', 'brown']
    colors = colors*(nb_envs//(len(colors)-1))

    ax['D'].scatter(init_xis[:,0], init_xis[:,1], s=30, c=colors[:nb_envs], marker='X', label="Initial", alpha=0.1)
    ax['D'].scatter(xis[:,0], xis[:,1], s=50, c=colors[:nb_envs], marker='o', label="Final")
    for i, (x, y) in enumerate(init_xis[:, :2]):
        ax['D'].annotate(str(i), (x, y), fontsize=8)
    for i, (x, y) in enumerate(xis[:, :2]):
        ax['D'].annotate(str(i), (x, y), fontsize=8)
    ax['D'].set_title(r'Initial and Final Contexts  ($\xi^e$)')
    ax['D'].legend()

    plt.tight_layout()
    plt.savefig(data_folder+"/deep_o_node_calibration_08.png", dpi=100, bbox_inches='tight')
    plt.show()
































# %%

### ==== Step 2: training both generator and discriminators at once ==== ####


def make_training_batch(data, discriminators, cutoff_length, key):      ## TODO: benchmark and save these btaches to disk
    """ Make a batch """

    e = jax.random.randint(key, (1,), 0, nb_envs)[0]
    traj = jax.random.randint(key, (1,), 0, nb_trajs_per_env)[0]

    X = data[e, traj, :cutoff_length, :]

    # es_batch = jnp.ones((nb_envs, nb_trajs_per_env), dtype=int)*jnp.arange(nb_envs)[:, None]
    # es_batch.reshape((nb_envs*nb_trajs_per_env,1))

    return (e, discriminators[e], X, t_eval[:cutoff_length], data[:, ::skip_trajs, :cutoff_length, :]), generate_new_keys(key, num=1)[0]



# _, disc_static = eqx.partition(model.discriminators[0], eqx.is_array)

# ## Main loss function
def loss_fn(params, static, batch):
    # print('\nCompiling function "loss_fn" ...\n')
    e, discriminator, X, t_eval, branch_data = batch
    print("Shapes of elements in a batch:", e.shape, X.shape, t_eval.shape, "\n")

    model = eqx.combine(params, static)
    # discriminators = model.discriminators
    discriminator = eqx.combine(discriminator, static.discriminators[0])

    # Xs_hat, nb_steps, probas, xis_hat = jax.vmap(model, in_axes=(0, None, 0))(Xs[:, 0, :], t_eval, xis)
    X_hat, nb_steps = model.generator(X[0, :], t_eval, discriminator, branch_data)

    term1 = jnp.mean((X-X_hat)**2, axis=-1).sum() / X.shape[0]
    # term2 = params_norm(params.discriminators)

    # loss_val = term1 + 1e-2*term2
    loss_val = term1

    return loss_val, (nb_steps, term1)


@partial(jax.jit, static_argnums=(1))
def train_step(params, static, batch, opt_state):
    print('\nCompiling function "train_step" ...\n')

    (loss, aux_data), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, static, batch)

    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss, aux_data


if train == True:

    # nb_train_steps_per_epoch = nb_envs*nb_trajs_per_env
    nb_train_steps_per_epoch = nb_trajs_per_env
    total_steps = nb_epochs * nb_train_steps_per_epoch

    sched = optax.piecewise_constant_schedule(init_value=init_lr,
                    boundaries_and_scales={int(total_steps*0.25):0.5, 
                                            int(total_steps*0.5):0.5,
                                            int(total_steps*0.75):0.5})

    opt = optax.adam(sched)
    opt_state = opt.init(params)

    print(f"\n\n=== Beginning training of generator ... ===")
    print(f"    Actual size of a batch (number of examples for all envs): {1}")
    print(f"    Number of train steps per epoch: {nb_train_steps_per_epoch}")
    print(f"    Number of training epochs: {nb_epochs}")
    print(f"    Total number of training steps: {total_steps}")


    start_time = time.time()

    losses = []
    nb_steps = []
    for epoch in range(nb_epochs):

        nb_batches = 0
        loss_sum = jnp.zeros(2)
        nb_steps_eph = 0

        for i in range(nb_train_steps_per_epoch):   ## Only two trajectories are used for each train_step

            # dists = eqx.combine(params,static).discriminators
            dists = params.discriminators
            batch, batch_key = make_training_batch(data, dists, cutoff_length, batch_key)
            params, opt_state, loss, (nb_steps_val, term1) = train_step(params, static, batch, opt_state)

            loss_sum += jnp.array([loss, term1])
            nb_steps_eph += nb_steps_val
            nb_batches += 1

        loss_epoch = loss_sum/nb_batches
        losses.append(loss_epoch)
        nb_steps.append(nb_steps_eph)

        if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
            print(f"    Epoch: {epoch:-5d}      TotalLoss: {loss_epoch[0]:-.8f}     Traj: {loss_epoch[1]:-.8f}", flush=True)

    losses = jnp.vstack(losses)
    nb_steps = jnp.array(nb_steps)

    wall_time = time.time() - start_time
    time_in_hmsecs = seconds_to_hours(wall_time)
    print("\nTotal GD training time: %d hours %d mins %d secs" %time_in_hmsecs)

    ## Save the results
    np.save("data/losses_08.npy", losses)
    np.save("data/nb_steps_08.npy", nb_steps)
    np.save("data/xis_08.npy", xis)
    np.save("data/init_xis_08.npy", init_xis)

    model = eqx.combine(params, static)
    eqx.tree_serialise_leaves("data/model_08.eqx", model)
    eqx.tree_serialise_leaves(data_folder+"model_08.eqx", model)       ## Make a seperate copy for backup

else:
    print("\nNo training, loading model and results from 'data' folder ...\n")

    losses = np.load("data/losses_08.npy")
    nb_steps = np.load("data/nb_steps_08.npy")
    xis = np.load("data/xis_08.npy")
    init_xis = np.load("data/init_xis_08.npy")

    model = eqx.combine(params, static)
    model = eqx.tree_deserialise_leaves("data/model_08.eqx", model)

















# %%

def test_model(model, batch):
    e, discriminator, X, t_eval, branch_data = batch

    # model = eqx.combine(params, static)
    X_hat, _ = model.generator(X[0, :], t_eval, discriminator, branch_data)

    return X_hat


# e_key, traj_key = get_new_key(testing_key, num=2)
e_key, traj_key = get_new_key(time.time_ns(), num=2)

e = jax.random.randint(e_key, (1,), 0, nb_envs)[0]
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


X_hat = test_model(model, (e, model.discriminators[e], X, t_test, data[:, ::skip_trajs, :cutoff_length, :]))

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

mke = np.ceil(losses.shape[0]/100).astype(int)

ax['C'].plot(losses[:,0], label="Total", color="grey", linewidth=3, alpha=1.0)
ax['C'].plot(losses[:,1], "x-", markevery=mke, markersize=mks, label="Traj", color="grey", linewidth=1, alpha=0.5)
ax['C'].plot(losses[:,2], "o-", markevery=mke, markersize=mks, label="Discrim", color="grey", linewidth=1, alpha=0.5)
ax['C'].plot(losses[:,3], "^-", markevery=mke, markersize=3, label="Params", color="grey", linewidth=1, alpha=0.5)
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
colors = colors*(nb_envs)

ax['E'].scatter(init_xis[:,0], init_xis[:,1], s=30, c=colors[:nb_envs], marker='X')
ax['F'].scatter(xis[:,0], xis[:,1], s=50, c=colors[:nb_envs], marker='o')
for i, (x, y) in enumerate(init_xis[:, :2]):
    ax['E'].annotate(str(i), (x, y), fontsize=8)
for i, (x, y) in enumerate(xis[:, :2]):
    ax['F'].annotate(str(i), (x, y), fontsize=8)
ax['E'].set_title(r'Initial Contexts ($\xi^e$)')
ax['F'].set_title(r'Final Contexts ($\xi^e$)')

plt.suptitle(f"Results for env={e}, traj={traj}", fontsize=14)

plt.tight_layout()
plt.savefig(data_folder+"deep_o_node_08.png", dpi=100, bbox_inches='tight')
plt.show()

print("Testing finished. Script, data, figures, and models saved in:", data_folder)

# %% [markdown]

# # Preliminary results
# 

# # Conclusion

# %%

