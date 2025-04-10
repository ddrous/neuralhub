#%%[markdown]

# ## Meta-Learnig via RNNs in weight space
# Use a RNN for map data to a sequence of weights
# - The RNN is a a linear state space model with A and B: theta_{t+1} = A theta_t + B x_t
# - This should work on irregular time series data, since the theta_t is decoded and evaluated between (0,1)
# - The loss function compares the latent space'd decoded output to the ground thruth

## ToDo:
# - [] Why is my cros-entropy so bad, and optax so good ?
# - [] Try the Neural CDE irregular dataset
# - [] Add delta_t in front of the A matrix

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
from loaders import TrendsDataset, MNISTDataset
from selfmod import NumpyLoader, setup_run_folder, torch
import torchvision
from torchvision import transforms

import optax
import time

## Set seaborn style to talk
import seaborn as sb
sb.set_context("poster")

#%%

SEED = 2025
main_key = jax.random.PRNGKey(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

## Model hps
mlp_hidden_size = 16
mlp_depth = 2

## Optimiser hps
init_lr = 1e-5

## Training hps
print_every = 1000
nb_epochs = 5000*2
batch_size = 256*1
grounding_length = 45       ## The length of the grounding pixel for the autoregressive digit generation
full_matrix_A = True        ## Whether to use a full matrix A or a diagonal one
classification = False       ## True for classification, False for reconstruction
mini_res_mnist = 1
traj_train_prop = 1.0       ## Proportion of steps to sample to train each time series
nb_recons_loss_steps = 40        ## Number of steps to sample for the reconstruction loss
train_in_inference_mode = True
use_mse_loss = False
run_mnist = False
print("==== Classification Task ====") if classification else print("==== Reconstruction Task ====")

train = True
data_folder = "./data/" if train else "../../data/"

run_folder = "./runs/250208-184005-Test/" if train else "./"
# run_folder = None if train else "./"

#%%
### Create and setup the run folder
if run_folder==None:
    run_folder = make_run_folder('./runs/')
else:
    print("Using existing run folder:", run_folder)
_, checkpoints_folder, _ = setup_run_folder(run_folder, os.path.basename(__file__), None)


#%%

if run_mnist:
    # ### MNIST Classification (From Sacha Rush's Annotated S4)
    print(" #### MNIST Dataset ####")
    trainloader = NumpyLoader(MNISTDataset(data_folder+"data/", data_split="train", mini_res=mini_res_mnist, traj_prop=traj_train_prop), 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=24)
    testloader = NumpyLoader(MNISTDataset(data_folder+"data/", data_split="test", mini_res=mini_res_mnist, traj_prop=1.0),
                                batch_size=batch_size, 
                                shuffle=False, 
                                num_workers=24)
    nb_classes, seq_length, data_size = trainloader.dataset.nb_classes, trainloader.dataset.num_steps, trainloader.dataset.data_size
    print("Training sequence length:", seq_length)
else:
    print(" #### Trends (Synthetic Control) Dataset ####")
    ## ======= below to run the easy Trends dataset instead!
    trainloader = NumpyLoader(TrendsDataset(data_folder+"trends/", skip_steps=1, traj_prop=traj_train_prop), 
                              batch_size=batch_size if batch_size<600 else 600, 
                              shuffle=True)
    testloader = NumpyLoader(TrendsDataset(data_folder+"trends/", skip_steps=1, traj_prop=1.0), 
                             batch_size=batch_size if batch_size<600 else 600,
                             shuffle=False)
    nb_classes, seq_length, data_size = trainloader.dataset.nb_classes, trainloader.dataset.num_steps, trainloader.dataset.data_size

batch = next(iter(testloader))
(images, times), labels = batch
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

## Plot a few samples, along with their labels as title in a 4x4 grid (chose them at random)
fig, axs = plt.subplots(4, 4, figsize=(10, 10), sharex=True)
colors = ['r', 'g', 'b', 'c', 'm', 'y']
res = 28 // mini_res_mnist
for i in range(4):
    for j in range(4):
        idx = np.random.randint(0, images.shape[0])
        if run_mnist:
            axs[i, j].imshow(images[idx].reshape((res, res)), cmap='gray')
        else:
            axs[i, j].plot(images[idx], color=colors[labels[idx]])
        axs[i, j].set_title(f"Class: {labels[idx]}", fontsize=12)
        axs[i, j].axis('off')

# %%

class RootMLP(eqx.Module):
    network: eqx.Module
    props: any      ## Properties of the network

    def __init__(self, input_dim, output_dims, hidden_size, depth, activation=jax.nn.softplus, key=None):
        key = key if key is not None else jax.random.PRNGKey(0)
        keys = jax.random.split(key, num=2)
        self.network = eqx.nn.MLP(input_dim, output_dims, hidden_size, depth, activation, key=keys[0])

        self.props = (input_dim, output_dims, hidden_size, depth, activation)

    def __call__(self, t):
        return self.network(t)


# ## Define model and loss function for the learner
class Ses2Seq(eqx.Module):
    """ Sequence to sequence model which takes in an initial latent space """
    A: jnp.ndarray
    B: jnp.ndarray
    theta: jnp.ndarray
    # alpha: jnp.ndarray

    root_utils: list
    inference_mode: bool
    data_size: int

    def __init__(self, 
                 data_size, 
                 width, 
                 depth, 
                 activation="relu",
                 key=None):

        keys = jax.random.split(key, num=3)
        builtin_fns = {"relu":jax.nn.relu, "tanh":jax.nn.tanh, 'softplus':jax.nn.softplus}
        # out_size = nb_classes if classification else data_size
        if classification:
            out_size = nb_classes
        else:
            if use_mse_loss:
                out_size = data_size
            else:      ## NLL loss
                out_size = 2*data_size
        root = RootMLP(data_size, out_size, width, depth, builtin_fns[activation], key=keys[1])

        params, static = eqx.partition(root, eqx.is_array)
        weights, shapes, treedef = flatten_pytree(params)
        self.root_utils = (shapes, treedef, static, root.props)
        self.theta = weights

        latent_size = weights.shape[0]
        self.A = jnp.eye(latent_size, latent_size) if full_matrix_A else jnp.ones((latent_size,))
        self.B = jnp.zeros((latent_size, data_size))

        # self.B = eqx.nn.Linear(1, latent_size*data_size, use_bias=False, key=keys[2])
        # ## Set the weights of B to zero
        # self.B = eqx.tree_at(lambda m: m.weight, self.B, jnp.zeros_like(self.B.weight))

        self.inference_mode = False     ## Change to True to use the model autoregressively
        self.data_size = data_size
        # self.alpha = jnp.array([0.0])

    def __call__(self, xs, ts):
        """ xs: (batch, time, data_size)
            theta: (latent_size)
            """

        def forward(xs_, ts_):
            ## 1. Fine-tune the latents weights on the sequence we have
            def f(carry, input_signal):
                thet, x_prev, t_prev, x_prev_prev = carry
                x_true, t_curr = input_signal
                delta_t = t_curr - t_prev

                if self.inference_mode:
                    x_t = jnp.where(t_curr<grounding_length/seq_length, x_true, x_prev)
                else:
                    # x_t = x_true
                    alpha = 1.0
                    x_t = alpha*x_true + (1-alpha)*x_prev

                if full_matrix_A:
                    # thet_next = delta_t*self.A@thet + delta_t*self.B@x_t
                    # thet_next = self.A@thet + delta_t*self.B@x_t
                    # thet_next = self.A@thet + self.B@x_t

                    # B = self.B(delta_t).reshape((self.A.shape[0], self.data_size))
                    # thet_next = self.A@thet + B@x_t

                    thet_next = self.A@thet + self.B@(x_t - x_prev_prev)/(1*delta_t + 1e-6)
                    # thet_next = self.A@thet + self.B@(x_t - x_prev_prev)*delta_t
                    # thet_next = self.A@thet + self.B@(x_t)/(1*delta_t + 1e-3)

                else:
                    thet_next = self.A*thet + delta_t*self.B@x_t
                    # thet_next = self.A*thet + self.B@x_t

                ## 2. Decode the latent space
                shapes, treedef, static, _ = self.root_utils
                params = unflatten_pytree(thet_next, shapes, treedef)
                root_fun = eqx.combine(params, static)
                x_next = root_fun(t_curr)
                # x_next_guess = root_fun(t_curr)
                if not use_mse_loss:
                    x_next_mean = x_next[:self.data_size]
                    # x_next_guess_mean = x_next_guess[:self.data_size]
                else:
                    x_next_mean = x_next
                    # x_next_guess_mean = x_next_guess

                return (thet_next, x_next_mean, t_curr, x_prev), (x_next, )

            ## Call the JAX scan
            _, (xs_final, ) = jax.lax.scan(f, (self.theta, xs_[0], -ts_[0:1], xs_[0]), (xs_, ts_[:, None]))

            return xs_final

        ## Batched version of the forward pass
        return eqx.filter_vmap(forward)(xs, ts)


# %%

model_keys = jax.random.split(main_key, num=2)

model = Ses2Seq(data_size=data_size, 
                width=mlp_hidden_size, 
                depth=mlp_depth, 
                activation="relu", 
                key=model_keys[0])

if train_in_inference_mode:
    model = eqx.tree_at(lambda m:m.inference_mode, model, True)
untrained_model = model
## Print the total number of learnable paramters in the model components
print(f"Number of learnable parameters in the root network: {count_params((model.theta,))/1000:3.1f} k")
print(f"Number of learnable parameters in the seqtoseq: {count_params((model.A, model.B))/1000:3.1f} k")
print(f"Number of learnable parameters in the model: {count_params(model)/1000:3.1f} k")

# %%

def loss_fn(model, batch, key):
    (X_true, times), X_labels = batch       ## X: (batch, time, data_size) - Y: (batch, num_classes)

    if classification:
        ## Categorical cross-entropy loss with optax
        X_classes = model(X_true, times)     ## Y_hat: (batch, time, num_classes) 
        pred_logits = X_classes[:, -1, :]   ## We only care about the last prediction: (batch, num_classes)
        losses_c = optax.softmax_cross_entropy_with_integer_labels(pred_logits, X_labels)
        loss = jnp.mean(losses_c)

        ## Calculate accuracy
        acc = jnp.mean(jnp.argmax(pred_logits, axis=-1) == X_labels)
        return loss, (acc,)

    else:
        ## Make a reconstruction loss
        X_recons = model(X_true, times)     ## Y_hat: (batch, time, data_size) 

        ## Randomly sample 2 points in the sequence to compare
        # indices = jax.random.randint(key, (2,), 0, X_true.shape[1])
        # loss_r = optax.l2_loss(X_recons[:, indices], X_true[:, indices])

        batch_size, nb_timesteps = X_true.shape[0], X_true.shape[1]
        indices_0 = jnp.arange(batch_size)
        indices_1 = jax.random.randint(key, (batch_size, nb_recons_loss_steps), 0, nb_timesteps)

        X_recons_ = jnp.stack([X_recons[indices_0, indices_1[:,j]] for j in range(nb_recons_loss_steps)], axis=1)
        X_true_ = jnp.stack([X_true[indices_0, indices_1[:,j]] for j in range(nb_recons_loss_steps)], axis=1)

        if use_mse_loss:
            loss_r = optax.l2_loss(X_recons_, X_true_)
        else: ## Use the negative log likelihood loss
            means = X_recons_[:, :, :data_size]
            stds = jnp.clip(jax.nn.softplus(X_recons_[:, :, data_size:]), 1e-6, 1)
            loss_r = jnp.log(stds) + 0.5*((X_true_ - means)/stds)**2

        loss = jnp.mean(loss_r)
        return loss, (loss,)


@eqx.filter_jit
def train_step(model, batch, opt_state, key):
    print('\nCompiling function "train_step" ...')

    (loss, aux_data), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, batch, key)

    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss, aux_data




#%%

if train:
    # sched = optax.exponential_decay(init_value=init_lr, transition_steps=10, decay_rate=0.99)
    # opt = optax.adam(sched)
    if classification:
        opt = optax.adabelief(init_lr)
    else:
        num_steps = trainloader.num_batches * nb_epochs
        bd_scales = {int(num_steps/3):0.4, int(num_steps*2/3):0.4}
        sched = optax.piecewise_constant_schedule(init_value=init_lr, boundaries_and_scales=bd_scales)
        opt = optax.adabelief(sched)

    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    train_key, _ = jax.random.split(main_key)

    losses = []

    print(f"\n\n=== Beginning Training ... ===")
    start_time = time.time()

    for epoch in range(nb_epochs):

        nb_batches = 0.
        loss_sum = 0.

        for i, batch in enumerate(trainloader):
            train_key, _ = jax.random.split(train_key)
            model, opt_state, loss, aux = train_step(model, batch, opt_state, train_key)

            loss_sum += loss
            nb_batches += 1

        loss_epoch = loss_sum/nb_batches
        losses.append(loss_epoch)

        if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
            if classification:
                acc, = aux
                print(f"    Epoch: {epoch:-5d}      Cross-Ent Loss: {loss_epoch:.6f}      Accuracy: {acc*100:.2f}%", flush=True)
            else:
                aux_loss, = aux
                if use_mse_loss:
                    print(f"    Epoch: {epoch:-5d}      MSELoss: {loss_epoch:.6f}", flush=True)
                else:
                    print(f"    Epoch: {epoch:-5d}      NLL Loss: {loss_epoch:.6f}", flush=True)

            # print("     Current teacher-forcing mixing factor alpha:", model.alpha)

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
ax = sbplot(np.array(losses), x_label='Epoch', y_label='Loss', ax=ax, dark_background=False, y_scale="linear" if not use_mse_loss else "log");
# plt.legend()
# ax.set_ylim(np.min(losses)-2e-2, min(np.max(losses)+2e-2, 1.0))
plt.draw();
plt.savefig(run_folder+"loss.png", dpi=100, bbox_inches='tight')


# %%

## Print the value of alpha
# print("Alpha before training: (no teacher forcing)", 0.)
# print("Alpha after training:", model.alpha)

## Let's visualise the distribution of values along the main diagonal of A and theta
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
if full_matrix_A:
    axs[0].hist(jnp.diag(model.A, k=0), bins=100)
else:
    axs[0].hist(model.A, bins=100)

axs[0].set_title("Histogram of diagonal values of A")

axs[1].hist(model.theta, bins=100, label="After Training")
axs[1].hist(untrained_model.theta, bins=100, alpha=0.5, label="Before Training", color='r')
axs[1].set_title(r"Histogram of $\theta$ values")
plt.legend();
plt.draw();
plt.savefig(run_folder+"A_theta_histograms.png", dpi=100, bbox_inches='tight')

## PLot all values of B in a lineplot (all dimensions)
if not isinstance(model.B, eqx.nn.Linear):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(model.B, label="Values of B")
    ax.set_title("Values of B")
    ax.set_xlabel("Dimension")
    plt.draw();
    plt.savefig(run_folder+"B_values.png", dpi=100, bbox_inches='tight')

if full_matrix_A:
    ## Print the untrained and trained matrices A as imshows with same range
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    # min_val = min(jnp.min(model.A), jnp.min(untrained_model.A))
    # max_val = max(jnp.max(model.A), jnp.max(untrained_model.A))
    min_val = -0.00
    max_val = 0.003

    img = axs[0].imshow(untrained_model.A, cmap='viridis', vmin=min_val, vmax=max_val)
    axs[0].set_title("Untrained A")
    plt.colorbar(img, ax=axs[0], shrink=0.7)

    img = axs[1].imshow(model.A, cmap='viridis', vmin=min_val, vmax=max_val)
    axs[1].set_title("Trained A")
    plt.colorbar(img, ax=axs[1], shrink=0.7)
    plt.draw();
    plt.savefig(run_folder+"A_matrices.png", dpi=100, bbox_inches='tight')



# %%
## Let's evaluate the model on the test set
accs = []
mses = []
for i, batch in enumerate(testloader):
    (X_true, times), X_labels = batch

    if classification:
        X_classes = model(X_true, times)
        pred_logits = X_classes[:, -1, :]   ## We only care about the last prediction: (batch, num_classes)
        acc = jnp.mean(jnp.argmax(pred_logits, axis=-1) == X_labels)
        accs.append(acc)

    else:
        X_recons = model(X_true, times)
        if not use_mse_loss:
            X_recons = X_recons[:, :, :data_size]
        mse = jnp.mean((X_recons - X_true)**2)
        mses.append(mse)

if classification:
    print(f"Mean accuracy on the test set: {np.mean(accs)*100:.2f}%")
else:
    print(f"Mean MSE on the test set: {np.mean(mses):.6f}")



# %%
## Let's visualise the reconstruction of a few samples only based on grounding information. PLot the true and the reconstructed images size by side

if not classification:
    ## Set inference mode to True
    model = eqx.tree_at(lambda m:m.inference_mode, model, True)
    visloader = NumpyLoader(testloader.dataset, batch_size=16, shuffle=True)

    fig, axs = plt.subplots(4, 4*2, figsize=(20*2, 20), sharex=True)

    batch = next(iter(visloader))
    (xs_true, times), labels = batch
    xs_recons = model(xs_true, times)

    if not use_mse_loss:
        xs_recons = xs_recons[:, :, :data_size]

    res = 28 // mini_res_mnist
    for i in range(4):
        for j in range(4):
            x = xs_true[i*4+j]
            x_recons = xs_recons[i*4+j]

            if run_mnist:
                axs[i, 2*j].imshow(x.reshape((res, res)), cmap='gray')
            else:
                axs[i, 2*j].plot(x, color=colors[labels[i*4+j]])
            if i==0:
                axs[i, 2*j].set_title("GT", fontsize=40)
            axs[i, 2*j].axis('off')

            if run_mnist:
                axs[i, 2*j+1].imshow(x_recons.reshape((res, res)), cmap='gray')
            else:
                axs[i, 2*j+1].plot(x_recons, color=colors[labels[i*4+j]])
            if i==0:
                axs[i, 2*j+1].set_title("Recons", fontsize=40)
            axs[i, 2*j+1].axis('off')

    plt.suptitle(f"Reconstruction using {grounding_length} initial pixels", fontsize=65, y=0.97)
    plt.draw();
    plt.savefig(run_folder+"reconstruction.png", dpi=100, bbox_inches='tight')



#%%
## Copy nohup.log to the run folder
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    os.system(f"cp nohup.log {run_folder}")
