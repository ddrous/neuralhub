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
from loaders import create_mnist_classification_dataset
from selfmod import NumpyLoader, setup_run_folder, torch
import torchvision
from torchvision import transforms

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
mlp_hidden_size = 16
mlp_depth = 2

## Optimiser hps
init_lr = 1e-4

## Training hps
print_every = 1
nb_epochs = 100
batch_size = 128*1
grounding_length = 300      ## The length of the grounding pixel for the autoregressive digit generation
full_matrix_A = True      ## Whether to use a full matrix A or a diagonal one

train = True
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


#%%


# ### MNIST Classification (From Sacha Rush's Annotated S4)
# **Task**: Predict MNIST class given sequence model over pixels (784 pixels => 10 classes).
def create_mnist_classification_dataset(bsz=128):
    print("[*] Generating MNIST Classification Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 784, 10, 1
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
            transforms.Lambda(lambda x: x.view(IN_DIM, SEQ_LENGTH).t()),
        ]
    )

    train = torchvision.datasets.MNIST(
        data_folder, train=True, download=True, transform=tf
    )
    test = torchvision.datasets.MNIST(
        data_folder, train=False, download=True, transform=tf
    )

    ## Return NumpyLoaders
    trainloader = NumpyLoader(
        train, batch_size=bsz, shuffle=True
    )
    testloader = NumpyLoader(
        test, batch_size=bsz, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


trainloader, testloader, nb_classes, seq_length, data_size = create_mnist_classification_dataset(bsz=batch_size)

batch = next(iter(trainloader))
images, labels = batch
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

## Plot a few samples, along with their labels as title in a 4x4 grid (chose them at random)
fig, axs = plt.subplots(4, 4, figsize=(20, 20), sharex=True)
for i in range(4):
    for j in range(4):
        idx = np.random.randint(0, images.shape[0])
        axs[i, j].imshow(images[idx].reshape((28, 28)), cmap='gray')
        axs[i, j].set_title(f"Class: {labels[idx]}")
        axs[i, j].axis('off')


# %%

class RootMLP(eqx.Module):
    network: eqx.Module
    props: any      ## Properties of the network

    def __init__(self, input_dim, output_dims, hidden_size, depth, activation=jax.nn.softplus, key=None):
        key = key if key is not None else jax.random.PRNGKey(0)
        keys = jax.random.split(key, num=2)
        network_1 = eqx.nn.MLP(input_dim, output_dims[0], hidden_size, depth, activation, key=keys[0])
        network_2 = eqx.nn.MLP(input_dim, output_dims[1], hidden_size, depth, activation, key=keys[1])
        self.network = (network_1, network_2)

        self.props = (input_dim, output_dims, hidden_size, depth, activation)

    def __call__(self, t):
        recons = self.network[0](t)
        classes = self.network[1](t)
        return jnp.concatenate([recons, classes], axis=-1)


# ## Define model and loss function for the learner
class Ses2Seq(eqx.Module):
    """ Sequence to sequence model which takes in an initial latent space """
    A: jnp.ndarray
    B: jnp.ndarray
    theta: jnp.ndarray

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
        root = RootMLP(data_size, (data_size, nb_classes), width, depth, builtin_fns[activation], key=keys[1])

        params, static = eqx.partition(root, eqx.is_array)
        weights, shapes, treedef = flatten_pytree(params)
        self.root_utils = (shapes, treedef, static, root.props)
        self.theta = weights

        latent_size = weights.shape[0]
        self.A = jnp.eye(latent_size, latent_size) if full_matrix_A else jnp.ones((latent_size,))
        self.B = jnp.zeros((latent_size, data_size))

        self.inference_mode = False     ## Change to True to use the model autoregressively
        self.data_size = data_size

    def __call__(self, xs):
        """ xs: (batch, time, data_size)
            theta: (latent_size)
            """

        shapes, treedef, static, _ = self.root_utils

        def forward(xs_):
            ## 1. Fine-tune the latents weights on the sequence we have
            def f(carry, input_signal):
                thet, t_prev = carry
                x_true, t_curr = input_signal
                delta_t = t_curr - t_prev

                if full_matrix_A:
                    # thet_next = self.A@thet + delta_t*self.B@x_true
                    thet_next = self.A@thet + self.B@x_true
                else:
                    thet_next = self.A*thet + delta_t*self.B@x_true

                return (thet_next, t_curr), (x_true, )

            ## Call the JAX scan
            ts_full = jnp.linspace(0, 1, xs_.shape[0])[:, None]
            if self.inference_mode:
                ts_ = ts_[grounding_length:]
                xs_ = xs_[grounding_length:]
            else:
                ts_ = ts_full
                xs_ = xs_
            (thet_final, _), (xs_true, ) = jax.lax.scan(f, (self.theta, xs_[0]), (xs_, ts_))

            ## 2. Decode the latent space
            params = unflatten_pytree(thet_final, shapes, treedef)
            root_fun = eqx.combine(params, static)
            model_out = eqx.filter_vmap(root_fun)(ts_full)
            x_recons = model_out[:, :self.data_size]
            x_classes = model_out[:, self.data_size:]

            return x_recons, x_classes

        ## Batched version of the forward pass
        return jax.vmap(forward)(xs)


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
    X_true, X_labels = batch       ## X: (batch, time, data_size) - Y: (batch, num_classes)
    X_recons, X_classes = model(X_true)     ## Y_hat: (batch, time, num_classes) 

    pred_logits = X_classes[:, -1, :]   ## We only care about the last prediction: (batch, num_classes)

    ## Categorical cross-entropy loss with optax
    losses_c = optax.softmax_cross_entropy_with_integer_labels(pred_logits, X_labels)
    loss_classif = jnp.mean(losses_c)

    # ## Manual Categorical cross-entropy loss
    # one_hot = jax.nn.one_hot(Y, 6)
    # pred_probs = jax.nn.softmax(pred_logits) + 1e-8
    # cce_loss = -jnp.sum(one_hot * jnp.log(pred_probs), axis=-1)
    # loss = jnp.mean(cce_loss)

    ## Calculate accuracy
    acc = jnp.mean(jnp.argmax(pred_logits, axis=-1) == X_labels)

    ## Make a reconstruction loss
    loss_r = optax.l2_loss(X_recons, X_true)
    loss_recons = jnp.mean(loss_r)

    ## Get the aggregate total loss
    loss = 1e-1*loss_classif + 1e-0*loss_recons

    return loss, (acc, loss_classif, loss_recons)

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
            acc, loss_classif, loss_recons = aux
            print(f"    Epoch: {epoch:-5d}      TotalLoss: {loss_epoch:.6f}      ClassifLoss: {loss_classif:.6f}      ReconsLoss: {loss_recons:.6f}      Accuracy: {acc*100:.2f}%", flush=True)
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
ax = sbplot(np.array(losses), x_label='Epoch', y_label='Loss', y_scale="log", label='CatCE + MSE', ax=ax, dark_background=False);
plt.legend()
plt.draw();
plt.savefig(run_folder+"loss.png", dpi=100, bbox_inches='tight')


# %%

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


if full_matrix_A:
    ## Print the untrained and trained matrices A as imshows with same range
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    img = axs[0].imshow(untrained_model.A, cmap='viridis')
    axs[0].set_title("Untrained A")
    plt.colorbar(img, ax=axs[0], shrink=0.7)

    img = axs[1].imshow(model.A, cmap='viridis')
    axs[1].set_title("Trained A")
    plt.colorbar(img, ax=axs[1], shrink=0.7)



# %%
## Let's evaluate the model on the test set
accs = []
mses = []
for i, batch in enumerate(testloader):
    X_true, X_labels = batch
    X_recons, X_classes = model(X_true)

    pred_logits = X_classes[:, -1, :]   ## We only care about the last prediction: (batch, num_classes)

    acc = jnp.mean(jnp.argmax(pred_logits, axis=-1) == X_labels)
    accs.append(acc)

    mse = jnp.mean((X_recons - X_true)**2)
    mses.append(mse)

print(f"Mean accuracy on the test set: {np.mean(accs)*100:.2f}%")
print(f"Mean MSE on the test set: {np.mean(mses):.6f}")




# %%
## Let's visualise the reconstruction of a few samples only based on grounding information. PLot the true and the reconstructed images size by side

## Set inference mode to True
model = eqx.tree_at(lambda m:m.inference_mode, model, True)
visloader = NumpyLoader(trainloader.dataset, batch_size=16, shuffle=True)

fig, axs = plt.subplots(4, 4*2, figsize=(20*2, 20), sharex=True)

batch = next(iter(visloader))
xs_true, labels = batch
xs_recons, _ = model(xs_true)

for i in range(4):
    for j in range(4):
        x = xs_true[i*4+j]
        x_recons = xs_recons[i*4+j]

        axs[i, 2*j].imshow(x.reshape((28, 28)), cmap='gray')
        if i==0:
            axs[i, 2*j].set_title("GT", fontsize=40)
        axs[i, 2*j].axis('off')

        axs[i, 2*j+1].imshow(x_recons.reshape((28, 28)), cmap='gray')
        if i==0:
            axs[i, 2*j+1].set_title("Recons", fontsize=40)
        axs[i, 2*j+1].axis('off')

plt.suptitle(f"Reconstruction using {grounding_length} inital pixels", fontsize=65, y=0.97)


## Copy nohup.log to the run folder
os.system(f"cp nohup.log {run_folder}")