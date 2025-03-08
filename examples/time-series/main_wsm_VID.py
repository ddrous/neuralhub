#%%[markdown]

# ## Meta-Learnig via Sequence Models in Weight Space

#%%
# %load_ext autoreload
# %autoreload 2

import jax

print("Available devices:", jax.devices())

from jax import config
# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)
# config.update("jax_enable_x64", True)
# from jax.experimental import checkify

## Ignore all warnings
import warnings
warnings.filterwarnings("ignore")

import jax.numpy as jnp

## Import jax partial
from jax.tree_util import Partial

import numpy as np
from scipy.integrate import solve_ivp

import equinox as eqx

# import matplotlib.pyplot as plt
from neuralhub import *
from loaders import MovingMNISTDataset
from selfmod import NumpyLoader, setup_run_folder, torch

import optax
import time

## Set seaborn style to talk
import seaborn as sb
sb.set_context("poster")


#%%

SEED = 2024
main_key = jax.random.PRNGKey(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

## Model hps
kernel_size = 4

## Optimiser hps
init_lr = 1e-4
lr_decrease_factor = 0.5        ## Reduce on plateau factor

## Training hps
print_every = 1
nb_epochs = 500
batch_size = 32*4
unit_normalise = False
grounding_length = 5          ## The length of the grounding pixel for the autoregressive digit generation
mini_res_mnist = 1
traj_train_prop = 1.0           ## Proportion of steps to sample to train each time series
weights_lim = 5e-1              ## Limit the weights of the root model to this value
nb_recons_loss_steps = -1        ## Number of steps to sample for the reconstruction loss
train_strategy = "flip_coin"     ## "flip_coin", "teacher_forcing", "always_true"
use_mse_loss = False
forcing_prob = 0.15
std_lower_bound = 1e-4              ## Let's optimise the lower bound
grad_clip_norm = 1e-7

train = True
dataset = "mnist_moving"
data_folder = "./data/" if train else "../../data/"

# run_folder = "./runs/250208-184005-Test/" if train else "./"
run_folder = None if train else "./"

#%%
### Create and setup the run folder
if run_folder==None:
    run_folder = make_run_folder('./runs/')
else:
    print("Using existing run folder:", run_folder)
_, checkpoints_folder, _ = setup_run_folder(run_folder, os.path.basename(__file__), None)

## Copy loaders script to the run folder
os.system(f"cp loaders.py {run_folder}");


#%%

print(" #### MNIST Dataset ####")
fashion = dataset=="mnist_fashion"
trainloader = NumpyLoader(MovingMNISTDataset(data_folder, data_split="train", mini_res=mini_res_mnist, unit_normalise=unit_normalise), 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=24)
testloader = NumpyLoader(MovingMNISTDataset(data_folder, data_split="test", mini_res=mini_res_mnist, unit_normalise=unit_normalise),
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=24)
seq_length, img_size = trainloader.dataset.num_steps, trainloader.dataset.data_size
C, H, W = img_size

batch = next(iter(testloader))
(videos, times), labels = batch
print("Videos shape:", videos.shape)
print("Labels shape:", labels.shape)
print("Min and Max in the dataset:", jnp.min(videos), jnp.max(videos))

## Plot a few samples videos (chose them at random)
nb_videos, nb_timesteps = 4, seq_length

fig, axs = plt.subplots(nb_videos, seq_length, figsize=(4*seq_length, 4*nb_videos), sharex=True)
for i in range(nb_videos):
    for j in range(seq_length):
        axs[i, j].imshow(videos[i, j, 0, :, :], cmap='gray')
        axs[i, j].axis('off')

plt.draw();
plt.savefig(run_folder+"sample_videos.png", dpi=100, bbox_inches='tight')


# %%


def enforce_absonerange(x):
    return jax.nn.tanh(x)

def enforce_positivity(x):
    return jax.nn.softplus(x)       ## Will be clipped by the model.

class Upsample2D(eqx.Module):
    """ Upsample 2D image by a factor: https://docs.kidger.site/equinox/examples/unet/ """
    factor: int
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, y):
        C, H, W = y.shape
        y = jnp.reshape(y, [C, H, 1, W, 1])
        y = jnp.tile(y, [1, 1, self.factor, 1, self.factor])
        return jnp.reshape(y, [C, H * self.factor, W * self.factor])

class TimeDecoder(eqx.Module):
    """ An MLP followed by a CNN to produce a frame. This is the root network, should be small.
    Dec: t -> frame_t
    """
    layers: list

    def __init__(self, out_shape, kernel_size, key=None):
        key = key if key is not None else jax.random.PRNGKey(0)

        layer_keys = jax.random.split(key, 5)
        C, H, W = out_shape

        self.layers = [
            eqx.nn.Linear(1, 32, key=layer_keys[0]),
            eqx.nn.PReLU(init_alpha=0.),
            eqx.nn.Linear(32, 4*H*W//(8*8), key=layer_keys[1]),
            eqx.nn.PReLU(init_alpha=0.),
            lambda x: x.reshape((4, H//8, W//8)),
            Upsample2D(factor=2),
            eqx.nn.ConvTranspose2d(4, 12, kernel_size, padding="SAME", key=layer_keys[2]),
            eqx.nn.PReLU(init_alpha=0.),
            Upsample2D(factor=2),
            eqx.nn.ConvTranspose2d(12, 6, kernel_size, padding="SAME", key=layer_keys[3]),
            eqx.nn.PReLU(init_alpha=0.),
            Upsample2D(factor=2),
            eqx.nn.ConvTranspose2d(6, C, kernel_size, padding="SAME", key=layer_keys[4]),
        ]

    def __call__(self, t):
        out = t
        for layer in self.layers:
            out = layer(out)

        if use_mse_loss:
            return jax.nn.tanh(out)
        else:
            recons, stds = jnp.split(out, 2, axis=-1)
            return jnp.concatenate([enforce_absonerange(recons),
                                    enforce_positivity(stds)], axis=-1)

class DataEncoder(eqx.Module):
    """ A CNN followd by an MLP to produce a vector of specific size
    Enc: frame_t -> z_t
    """
    layers: list

    def __init__(self, img_size, kernel_size, out_size, key=None):
        key = key if key is not None else jax.random.PRNGKey(0)

        layer_keys = jax.random.split(key, 5)
        C, H, W = img_size

        self.layers = [
            eqx.nn.Conv2d(C, 6, kernel_size, padding="SAME", key=layer_keys[0]),
            eqx.nn.PReLU(init_alpha=0.),
            eqx.nn.MaxPool2d(2, 2),
            eqx.nn.Conv2d(6, 12, kernel_size, padding="SAME", key=layer_keys[1]),
            eqx.nn.PReLU(init_alpha=0.),
            eqx.nn.MaxPool2d(2, 2),
            eqx.nn.Conv2d(12, 24, kernel_size, padding="SAME", key=layer_keys[2]),
            eqx.nn.PReLU(init_alpha=0.),
            eqx.nn.MaxPool2d(2, 2),
            lambda x: x.flatten(),
            eqx.nn.Linear(H//8*W//8*24, out_size//2, key=layer_keys[3]),
            eqx.nn.PReLU(init_alpha=0.),
            eqx.nn.Linear(out_size//2, out_size, key=layer_keys[4]),
        ]

    def __call__(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


# ## Define model and loss function for the learner
class Ses2Seq(eqx.Module):
    """ Sequence to sequence model which takes in an initial latent space """
    A: jnp.ndarray
    B: eqx.Module
    theta: jnp.ndarray
    std_lb: jnp.ndarray

    root_utils: list
    inference_mode: bool

    def __init__(self, img_size, key=None):
        C, H, W = img_size
        out_chans = C if use_mse_loss else 2*C
        thet_key, A_key, B_key = jax.random.split(key, 3)

        root = TimeDecoder(out_shape=(out_chans, H, W), kernel_size=kernel_size, key=thet_key)
        params, static = eqx.partition(root, eqx.is_array)
        weights, shapes, treedef = flatten_pytree(params)
        self.root_utils = (shapes, treedef, static)
        self.theta = weights

        latent_size = weights.shape[0]
        self.A = jnp.eye(latent_size, latent_size)
        self.B = DataEncoder(img_size, kernel_size, latent_size, key=B_key)     ## TODO: zerout the B weights ?
        self.std_lb = jnp.array([std_lower_bound])

        self.inference_mode = False     ## Change to True to use the model autoregressively

    def __call__(self, xs, ts, key):
        """ xs: (batch, time, c, h, w)     <->      ts: (batch, time)   """   

        def forward(xs_, ts_, k_):

            def f(carry, input_signal):
                """ The weight generation function """

                thet, x_prev, x_prev_prev, t_prev = carry
                x_true, t_curr, key_ = input_signal
                delta_t = t_curr - t_prev

                if self.inference_mode:
                    x_t = jnp.where(t_curr<grounding_length/seq_length, x_true, x_prev)
                else:
                    if train_strategy == "flip_coin":
                        x_t = jnp.where(jax.random.bernoulli(key_, forcing_prob), x_true, x_prev)
                    elif train_strategy == "teacher_forcing":
                        alpha = 0.1
                        x_t = alpha*x_true + (1-alpha)*x_prev
                    else:
                        x_t = x_true

                # thet_next = self.A@thet + self.B(x_t - x_prev_prev)
                thet_next = self.A@thet + self.B(x_t) - self.B(x_prev_prev)   ## TODO: do this?

                ## Decode the latent space
                thet_next = jnp.clip(thet_next, -weights_lim, weights_lim)

                shapes, treedef, static = self.root_utils
                params = unflatten_pytree(thet_next, shapes, treedef)
                root_fun = eqx.combine(params, static)
                y_next = root_fun(t_curr + delta_t)

                x_next_mean = y_next[:x_true.shape[0]]

                return (thet_next, x_next_mean, x_prev, t_curr), (y_next, )

            keys = jax.random.split(k_, xs_.shape[0])
            (_, _, _, _), (ys_, ) = jax.lax.scan(f, 
                                                    (self.theta, xs_[0], xs_[0], -ts_[1:2]), 
                                                    (xs_, ts_[:, None], keys))

            return ys_

        ## Batched version of the forward pass
        ks = jax.random.split(key, xs.shape[0])
        return eqx.filter_vmap(forward)(xs, ts, ks)


# %%

model_key, train_key, test_key = jax.random.split(main_key, num=3)
model = Ses2Seq(img_size, key=model_key)
untrained_model = model

# ## Print the time decoder model
# print("Time Decoder Model:")
# print(TimeDecoder(out_shape=(2*C, H, W), kernel_size=kernel_size))

## Print the total number of learnable paramters in the model components
print(f"Number of learnable parameters in the root network: {count_params((model.theta,))/1000:3.1f} k")
print(f"Number of learnable parameters in the seqtoseq: {count_params((model.A, model.B))/1000:3.1f} k")
print(f"Number of learnable parameters in the model: {count_params(model)/1000:3.1f} k")

# %%

def loss_fn(model, batch, key):
    (X_true, times), _ = batch       ## X: (batch, time, img_size) 

    X_recons = model(X_true, times, key)     ## Y_hat: (batch, time, out_size) 

    if use_mse_loss:
        loss_r = optax.l2_loss(X_recons, X_true)
    else:
        means = X_recons[:, :, :C]
        stds = jnp.maximum(X_recons[:, :, C:], model.std_lb)
        loss_r = jnp.log(stds) + 0.5*((X_true - means)/stds)**2

    loss = jnp.mean(loss_r)
    return loss, (loss,)


@eqx.filter_jit
def train_step(model, batch, opt_state, key):
    # print('\nCompiling function "train_step" ...')

    (loss, aux_data), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, batch, key)

    updates, opt_state = opt.update(grads, opt_state, model, value=loss)        ## For reduce on plateau loss accumulation
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss, aux_data


#%%

if train:
    num_steps = trainloader.num_batches * nb_epochs     ## Total number of train steps

    opt = optax.chain(
        optax.clip(grad_clip_norm),
        optax.adabelief(init_lr),
        optax.contrib.reduce_on_plateau(
            patience=20,
            cooldown=0,
            factor=lr_decrease_factor,
            rtol=1e-4,
            accumulation_size=50,
            min_scale=1e-2,
        ),
    )

    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    train_key, _ = jax.random.split(main_key)

    losses = []
    losses_epoch = []
    lr_scales = []

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

            losses.append(loss)
            lr_scales.append(optax.tree_utils.tree_get(opt_state, "scale"))

        loss_epoch = loss_sum/nb_batches
        losses_epoch.append(loss_epoch)

        if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
            if use_mse_loss:
                print(f"    Epoch: {epoch:-5d}      MSELoss: {loss_epoch:.6f}", flush=True)
            else:
                print(f"    Epoch: {epoch:-5d}      NLL Loss: {loss_epoch:.6f}", flush=True)

            eqx.tree_serialise_leaves(checkpoints_folder+f"model_{epoch}.eqx", model)
            np.save(run_folder+"losses.npy", np.array(losses))
            np.save(run_folder+"lr_scales.npy", np.array(lr_scales))

            ## Save the best model with the lowest loss
            if epoch>0 and loss_epoch<min(losses_epoch[:-1]):
                eqx.tree_serialise_leaves(run_folder+"model.eqx", model)

    wall_time = time.time() - start_time
    time_in_hmsecs = seconds_to_hours(wall_time)
    print("\nTotal GD training time: %d hours %d mins %d secs" %time_in_hmsecs)

    print(f"Training Complete, saving model to folder: {run_folder}")
    if losses[-1]<min(losses_epoch[:-1]):
        eqx.tree_serialise_leaves(run_folder+"model.eqx", model)
    np.save(run_folder+"losses.npy", np.array(losses))
    np.save(run_folder+"lr_scales.npy", np.array(lr_scales))

else:
    model = eqx.tree_deserialise_leaves(run_folder+"model.eqx", model)

    try:
        losses = np.load(run_folder+"losses.npy")
        lr_scales = np.load(run_folder+"lr_scales.npy")
    except:
        losses = []

    print("Model loaded from folder")

## Print the current value of the lower bound
print("== Lower Bound for the Stadard Deviations ==")
print(" - Initial:", std_lower_bound)
print(" - Final  :", model.std_lb)

# %%

if os.path.exists(run_folder+"losses.npy"):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    clean_losses = np.array(losses)
    epochs = np.arange(len(losses))

    ax = sbplot(epochs, clean_losses, label="All losses", x_label='Train Steps', y_label='Loss', ax=ax, dark_background=False, y_scale="linear" if not use_mse_loss else "log");

    clean_losses = np.where(clean_losses<np.percentile(clean_losses, 80), clean_losses, np.nan)
    ## Plot a second plot with the outliers removed
    ax2 = sbplot(epochs, clean_losses, label="96th Percentile", x_label='Train Steps', y_label='Loss', ax=ax2, dark_background=False, y_scale="linear" if not use_mse_loss else "log");

    plt.legend()
    plt.draw();
    plt.savefig(run_folder+"loss.png", dpi=100, bbox_inches='tight')

else: ## Attempt to parse and collect losses from the nohup.log file
    try:
        with open(run_folder+"nohup.log", 'r') as f:
            lines = f.readlines()
        losses = []
        loss_name = "MSELoss" if use_mse_loss else "NLL Loss"
        for line in lines:
            if loss_name in line:
                loss = float(line.split(loss_name+": ")[1].strip())
                losses.append(loss)

        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ## Filter outlier vales and retain the rest
        clean_losses = np.array(losses)
        epochs = np.arange(len(losses))

        ax = sbplot(epochs, clean_losses, label="All losses", x_label='Train Steps', y_label='Loss', ax=ax, dark_background=False, y_scale="linear" if not use_mse_loss else "log");

        clean_losses = np.where(clean_losses<np.percentile(clean_losses, 96), clean_losses, np.nan)
        ## Plot a second plot with the outliers removed
        ax2 = sbplot(epochs, clean_losses, label="96th Percentile", x_label='Train Steps', y_label='Loss', ax=ax2, dark_background=False, y_scale="linear" if not use_mse_loss else "log");

        plt.legend(loc='upper right')
        plt.draw();
        plt.savefig(run_folder+"loss.png", dpi=100, bbox_inches='tight')
    except:
        print("No losses found in the nohup.log file")


if os.path.exists(run_folder+"lr_scales.npy"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    clean_lr_scales = np.array(lr_scales)
    train_steps = np.arange(len(lr_scales))

    ax = sbplot(train_steps, clean_lr_scales, label="Learning Rate Scales", x_label='Train Steps', y_label='LR Scale', ax=ax, dark_background=False, y_scale="log");

    plt.legend()
    plt.draw();
    plt.savefig(run_folder+"lr_scales.png", dpi=100, bbox_inches='tight')

# %%

## Let's visualise the distribution of values along the main diagonal of A and theta
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
axs[0].hist(jnp.diag(model.A, k=0), bins=100)

axs[0].set_title("Histogram of diagonal values of A (first layer)")

axs[1].hist(model.theta, bins=100, label="After Training")
axs[1].hist(untrained_model.theta, bins=100, alpha=0.5, label="Before Training", color='r')
axs[1].set_title(r"Histogram of $\theta_0$ values")
plt.legend();
plt.draw();
plt.savefig(run_folder+"A_theta_histograms.png", dpi=100, bbox_inches='tight')

## Print the untrained and trained matrices A as imshows with same range
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
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

@eqx.filter_jit
def compute_mse(model, data, key):
    key, _ = jax.random.split(key)
    (X_true, times), _ = data

    X_recons = model(X_true, times, key)
    if not use_mse_loss:
        X_recons = X_recons[:, :, :C]
    mse = jnp.mean((X_recons - X_true)**2)

    return mse

accs = []
mses = []
for i, batch in enumerate(testloader):
    mse = compute_mse(model, batch, test_key)
    mses.append(mse)

print(f"Mean MSE on the test set: {np.mean(mses):.6f}")

# %%
## Let's visualise the reconstruction of a few samples only based on grounding information. PLot the true and the reconstructed images size by side

## Set inference mode to True
if isinstance(model, Ses2Seq):
    model = eqx.filter_jit(eqx.tree_at(lambda m:m.inference_mode, model, True))
visloader = NumpyLoader(testloader.dataset, batch_size=1, shuffle=True)

nb_cols = 3 if not use_mse_loss else 2
fig, axs = plt.subplots(nb_cols, seq_length, figsize=(4*seq_length, 4*nb_cols), sharex=True, constrained_layout=True)

batch = next(iter(visloader))
(xs_true, times), labels = batch
xs_recons = model(xs_true, times, test_key)

if not use_mse_loss:
    xs_uncert = xs_recons[:, :, C:]
    xs_recons = xs_recons[:, :, :C]

## Plot the true and the GT images (top), the recons (middle row), and uncertainty (bottom row)
for i in range(seq_length):
    axs[0, i].imshow(xs_true[0, i, 0], cmap='gray')
    axs[0, i].axis('off')

    axs[1, i].imshow(xs_recons[0, i, 0], cmap='gray')
    axs[1, i].axis('off')

    if not use_mse_loss:
        axs[2, i].imshow(xs_uncert[0, i, 0], cmap='gray')
        axs[2, i].axis('off')

plt.suptitle(f"Reconstruction using {grounding_length} initial steps", fontsize=65)
plt.draw();
plt.savefig(run_folder+"reconstruction.png", dpi=100, bbox_inches='tight')


#%%
## Copy nohup.log to the run folder
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    os.system(f"cp nohup.log {run_folder}")
