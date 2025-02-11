#%%[markdown]

# ## Meta-Learnig via RNNs in weight space
# Use a RNN for map data to a sequence of weights
# - The RNN is a a linear state space model with A and B: theta_{t+1} = A theta_t + B x_t
# - This should work on irregular time series data, since the theta_t is decoded and evaluated between (0,1)
# - The loss function compares the latent space'd decoded output to the ground thruth

## ToDo:
# - [] Add a __reconstuction__ head to the model
# - [x] Remove the B x_t term from the model, see if we keep the same performance. We don't !
# - [] Why is my cros-entropy so bad, and optax so good ?
# - [] Try the epilepsy dataset
# - [] Try the Minist control dataset https://srush.github.io/annotated-s4/#experiments-mnist
# - [] Try the Neural CDE irregular dataset

#%%
import jax

print("Available devices:", jax.devices())

# from jax import config
# config.update("jax_debug_nans", True)

import jax.numpy as jnp

import numpy as np
from scipy.integrate import solve_ivp

import equinox as eqx

# import matplotlib.pyplot as plt
from neuralhub import *
from loaders import TrendsDataset
from selfmod import NumpyLoader, setup_run_folder, torch

import optax
import time

## Set seaborn style to talk
import seaborn as sb
sb.set_context("poster")

# import wandb

#%%

SEED = 2025
main_key = jax.random.PRNGKey(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

## Model hps
mlp_hidden_size = 16
mlp_depth = 2
data_size = 1

## Optimiser hps
init_lr = 1e-3

## Training hps
print_every = 100
nb_epochs = 1000
batch_size = 60
traj_prop_train = 1.0

train = True
data_folder = "./data/trends/" if train else "../../data/trends/"

# run_folder = "./runs/250208-184005-Test/" if train else "./"
run_folder = None if train else "./"

# # Initializing a Weights & Biases Run
# wandb.init(
#     project="weight-space-models",
#     # entity="jax-series",
#     job_type="main-ssm",
#     config={"mlp_hidden_size": mlp_hidden_size,
#             "mlp_depth": mlp_depth,
#             "data_size": data_size,
#             "init_lr": init_lr,
#             "print_every": print_every,
#             "nb_epochs": nb_epochs,
#             "batch_size": batch_size,
#             "traj_prop_train": traj_prop_train}
# )

#%%
### Create and setup the run folder
if run_folder==None:
    run_folder = make_run_folder('./runs/')
else:
    print("Using existing run folder:", run_folder)
_, checkpoints_folder, _ = setup_run_folder(run_folder, os.path.basename(__file__), None)


#%%

train_dataset = TrendsDataset(data_folder, skip_steps=1, adaptation=False, traj_prop_min=1.0, use_full_traj=True)
train_loader = NumpyLoader(train_dataset, batch_size=batch_size, shuffle=True)

batch = next(iter(train_loader))
(data, t_evals), labels = batch
print("Time series data shape:", data.shape)
print("Eval timestamps shape:", t_evals.shape)
print("Labels shape:", labels.shape)

## Plot a few samples, along with their labels as title in a 4x4 grid (chose them at random)
fig, axs = plt.subplots(4, 4, figsize=(40, 20), sharex=True)
colors = ['r', 'g', 'b', 'c', 'm', 'y']
for i in range(4):
    for j in range(4):
        idx = np.random.randint(0, data.shape[0])
        axs[i, j].plot(t_evals[0], data[idx, 0, :, 0], color=colors[labels[idx]])
        axs[i, j].set_title(f"Class: {labels[idx]}")


# %%

class RootMLP(eqx.Module):
    network: eqx.Module
    root_utils: any
    network_size: int     ## The effective/actual size of a root network (flattened neural network)

    def __init__(self, input_dim, output_dim, hidden_size, depth, activation=jax.nn.softplus, key=None):
        key = key if key is not None else jax.random.PRNGKey(0)
        self.network = eqx.nn.MLP(input_dim, output_dim, hidden_size, depth, activation, key=key)
        
        props = (input_dim, output_dim, hidden_size, depth, activation)
        params, static = eqx.partition(self.network, eqx.is_array)
        _, shapes, treedef = flatten_pytree(params)
        self.root_utils = (shapes, treedef, static, props)

        self.network_size = sum(x.size for x in jax.tree_util.tree_leaves(params) if x is not None)

    def __call__(self, x):
        return self.network(x)


# ## Define model and loss function for the learner
class Ses2Seq(eqx.Module):
    """ Sequence to sequence model which takes in an initial latent space """
    A: jnp.ndarray
    B: jnp.ndarray
    theta: jnp.ndarray
    root_utils: list

    def __init__(self, 
                 data_size, 
                 width, 
                 depth, 
                 activation="relu",
                 key=None):

        keys = jax.random.split(key, num=3)
        builtin_fns = {"relu":jax.nn.relu, "tanh":jax.nn.tanh, 'softplus':jax.nn.softplus}
        root = RootMLP(1, 6, width, depth, builtin_fns[activation], key=keys[1])
        self.root_utils = root.root_utils
        root_params, _ = eqx.partition(root.network, eqx.is_array)
        self.theta = flatten_pytree(root_params)[0]

        latent_size = root.network_size
        self.A = jnp.eye(latent_size, latent_size)
        self.B = jnp.zeros((latent_size, data_size))

    def __call__(self, xs, ts):
        """ xs: (batch, time, data_size)
            ts: (batch, time)
            theta: (latent_size)
            """

        def forward(xs_, ts_):
            ## 1. Fine-tune the weights (the latent initilisation)
            def f(thet, x):
                thet_next = self.A@thet + self.B@x
                # thet_next = self.A@thet
                return thet_next, x
            thet_final, _ = jax.lax.scan(f, self.theta, xs_)

            ## 2. Decode the latent space
            shapes, treedef, static, _ = self.root_utils
            params = unflatten_pytree(thet_final, shapes, treedef)
            root_fun = eqx.combine(params, static)

            ## 3. Evaluate the root function
            return eqx.filter_vmap(root_fun)(ts_[:, None])

        ## Batched version of the forward pass (1 ts per env)
        return jax.vmap(forward, in_axes=(0, 0))(xs[:,0,...], ts)


# %%

model_keys = jax.random.split(main_key, num=2)

model = Ses2Seq(data_size=data_size, 
                width=mlp_hidden_size, 
                depth=mlp_depth, 
                activation="relu", 
                key=model_keys[0])

untrained_model = model
## Print the total number of learnable paramters in the model components
print(f"Number of learnable parameters in the root network: {count_params((model.theta,))/1000:3.1f} k")
print(f"Number of learnable parameters in the seqtoseq: {count_params((model.A, model.B))/1000:3.1f} k")
print(f"Number of learnable parameters in the model: {count_params(model)/1000:3.1f} k")

# %%

def loss_fn(model, batch, key):
    (X, t), Y = batch       ## X: (batch, time, data_size) - Y: (batch, num_classes)
    Y_hat = model(X, t)     ## Y_hat: (batch, time, num_classes) 

    pred_logits = Y_hat[:, -1, :]   ## We only care about the last prediction: (batch, num_classes)

    ## Categorical cross-entropy loss with optax
    losses = optax.softmax_cross_entropy_with_integer_labels(pred_logits, Y)
    loss = jnp.mean(losses)

    # ## Manual Categorical cross-entropy loss
    # one_hot = jax.nn.one_hot(Y, 6)
    # pred_probs = jax.nn.softmax(pred_logits) + 1e-8
    # cce_loss = -jnp.sum(one_hot * jnp.log(pred_probs), axis=-1)
    # loss = jnp.mean(cce_loss)

    ## Calculate accuracy
    acc = jnp.mean(jnp.argmax(pred_logits, axis=-1) == Y)

    return loss, (acc,)

@eqx.filter_jit
def train_step(model, batch, opt_state, key):
    print('\nCompiling function "train_step" ...')

    (loss, aux_data), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, batch, key)

    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss, aux_data




#%%

if train:
    sched = optax.exponential_decay(init_value=init_lr, transition_steps=10, decay_rate=0.99)
    opt = optax.adam(sched)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    train_key, _ = jax.random.split(main_key)

    nb_data_points = data.shape[1]
    losses = []

    print(f"\n\n=== Beginning Training ... ===")
    start_time = time.time()

    for epoch in range(nb_epochs):

        nb_batches = 0.
        loss_sum = 0.

        for i, batch in enumerate(train_loader):
            train_key, _ = jax.random.split(train_key)
            model, opt_state, loss, (aux,) = train_step(model, batch, opt_state, train_key)

            loss_sum += loss
            nb_batches += 1

            # # Log Metrics to Weights & Biases
            # wandb.log({
            #     "Train Loss": loss,
            #     "Train Accuracy": aux,
            #     "A matrix": model.A,
            #     "B matrix": model.B,
            #     "Theta": model.theta,
            # }, step=epoch)

        loss_epoch = loss_sum/nb_batches
        losses.append(loss_epoch)

        if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
            print(f"    Epoch: {epoch:-5d}      Loss: {loss_epoch:.12f}      Accuracy: {aux*100:.2f}%", flush=True)
            eqx.tree_serialise_leaves(run_folder+"model.eqx", model)
            eqx.tree_serialise_leaves(checkpoints_folder+f"model_{epoch}.eqx", model)

    wall_time = time.time() - start_time
    time_in_hmsecs = seconds_to_hours(wall_time)
    print("\nTotal GD training time: %d hours %d mins %d secs" %time_in_hmsecs)

    print(f"Training Complete, saving model to folder: {run_folder}")
    eqx.tree_serialise_leaves(run_folder+"model.eqx", model)
    np.save(run_folder+"losses.npy", np.array(losses))

else:
    model = eqx.tree_deserialise_leaves(run_folder+"model.eqx", model)
    try:
        losses = np.load(run_folder+"losses.npy")
    except:
        losses = []

    print("Model loaded from folder")


# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax = sbplot(np.array(losses), x_label='Epoch', y_label='Loss', y_scale="log", label='Cat. Cross-Entropy', ax=ax, dark_background=False);
plt.legend()
plt.draw();
plt.savefig(run_folder+"loss.png", dpi=100, bbox_inches='tight')


# %%

## Let's visualise the distribution of values along the main diagonal of A and theta
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
axs[0].hist(jnp.diag(model.A, k=0), bins=100)
axs[0].set_title("Histogram of diagonal values of A")

axs[1].hist(model.theta, bins=100, label="After Training")
axs[1].hist(untrained_model.theta, bins=100, alpha=0.5, label="Before Training", color='r')
axs[1].set_title(r"Histogram of $\theta$ values")
plt.legend();


# # Close your wandb run
# wandb.finish()