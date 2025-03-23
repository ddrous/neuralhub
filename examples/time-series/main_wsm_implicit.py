#%% Import necessary libraries
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import diffrax
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from typing import Tuple, List, Dict, Any, Optional
from functools import partial
from selfmod import flatten_pytree, unflatten_pytree, make_run_folder, setup_run_folder, count_params
import os
import time

print("JAX version:", jax.__version__)
print("Equinox version:", eqx.__version__)

#%% Set random seed for reproducibility
seed = 2024
np.random.seed(seed)
torch.manual_seed(seed)

batch_size = 64*2
print_every = 10

num_epochs = 200
state_dim = 1  # MNIST digits are 28x28
hidden_dim = 64

learning_rate = 1e-4    ## Outer learning rate
inner_learning_rate = 0.01

forcing_prob = 0.5
latent_dim = 64

grounding_length = 300      ## for inference
num_gradient_steps = 4

train= True
run_folder = None if train else "./"

#%%
### Create and setup the run folder
if run_folder==None:
    run_folder = make_run_folder('./runs/')
else:
    print("Using existing run folder:", run_folder)
_, checkpoints_folder, _ = setup_run_folder(run_folder, os.path.basename(__file__), None)


#%% Define custom collate function to convert PyTorch tensors to JAX arrays
def custom_collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert PyTorch tensors to JAX arrays."""
    images, labels = zip(*batch)
    images = torch.stack(images).numpy()
    labels = torch.tensor(labels).numpy()
    return jnp.array(images), jnp.array(labels)

#%% Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
    transforms.Normalize((0.5,), (0.5,))
])

# Load training data
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

# Load test data
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

#%% Function to preview MNIST images
def preview_mnist_digits(images: jnp.ndarray, labels: jnp.ndarray, num_samples: int = 5) -> None:
    """Display a few MNIST digits from the dataset."""
    plt.figure(figsize=(12, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i][0], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Get a batch of images and labels
for images, labels in train_loader:
    preview_mnist_digits(images, labels)
    break  # Just one batch


#%% Define Neural ODE model using Equinox
class MLPVectorField(eqx.Module):
    """
    Multi-layer perceptron for the theta component of the ODE vector field. theta : t -> x_t
    """
    hidden_dim: int
    state_dim: int
    layers: List[eqx.nn.Linear]
    activation: callable

    def __init__(self, state_dim: int, hidden_dim: int, key: jnp.ndarray):
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.activation = jax.nn.tanh

        # Split the key for each layer
        keys = jax.random.split(key, 3)
        
        # Define the network layers
        self.layers = [
            eqx.nn.Linear(1+state_dim, hidden_dim, key=keys[0]),
            eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[1]),
            eqx.nn.Linear(hidden_dim, state_dim, key=keys[2])
        ]

    def __call__(self, t, y) -> jnp.ndarray:
        """Forward pass through the network."""
        ## If scalar, convert to array
        if jnp.isscalar(t):
            t = jnp.array([t])
        if jnp.isscalar(y):
            y = jnp.array([y])

        # h = t
        h = jnp.concatenate([t, y])
        h = self.activation(self.layers[0](h))
        h = self.activation(self.layers[1](h))
        h = self.layers[2](h)
        return h

class MLPKernel(eqx.Module):
    """
    Multi-layer perceptron for the kernel of ODE vector field. phi : (t, tau) -> state_dim x state_dim
    """
    hidden_dim: int
    state_dim: int
    layers: List[eqx.nn.Linear]
    activation: callable

    def __init__(self, state_dim: int, hidden_dim: int, key: jnp.ndarray):
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.activation = jax.nn.tanh

        # Split the key for each layer
        keys = jax.random.split(key, 3)

        # Define the network layers
        self.layers = [
            eqx.nn.Linear(2, hidden_dim, key=keys[0]),
            eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[1]),
            eqx.nn.Linear(hidden_dim, state_dim*latent_dim, key=keys[2])
        ]

    def __call__(self, t: float, tau: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the network."""
        ## If scalar, convert to array
        if jnp.isscalar(t):
            t = jnp.array([t])
        if jnp.isscalar(tau):
            tau = jnp.array([tau])

        h = jnp.concatenate([t, tau])
        h = self.activation(self.layers[0](h))
        h = self.activation(self.layers[1](h))
        h = self.layers[2](h)
        return h

class NeuralODE(eqx.Module):
    """
    Neural ODE model.
    """
    theta: jnp.ndarray       ## theta_0 initialisation meta-learning
    root_utils: List[Tuple[jnp.ndarray, Any, Any, Any]]

    def __init__(self, state_dim: int, hidden_dim: int, key: jnp.ndarray):
        theta = MLPVectorField(state_dim, hidden_dim, key)
        params, static = eqx.partition(theta, eqx.is_array)
        weights, shapes, treedef = flatten_pytree(params)
        self.root_utils = (shapes, treedef, static, weights.shape[0])
        self.theta = weights

    def __call__(self, t: float, y: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """Solve the ODE from t0 to t1 with initial state y0."""
        shapes, treedef, static, weight_dim = self.root_utils
        params = unflatten_pytree(theta, shapes, treedef)
        theta_fun = eqx.combine(params, static)

        return theta_fun(t, y)






class Model(eqx.Module):
    """
    Sequence to sequence model.
    """
    neuralode: NeuralODE
    inner_loss_fn: callable
    inference: bool
    # phi: eqx.nn.MLP

    def __init__(self, state_dim: int, hidden_dim: int, key: jnp.ndarray):
        self.neuralode = NeuralODE(state_dim, hidden_dim, key)

        weight_dim = self.neuralode.root_utils[-1]
        # self.phi = eqx.nn.MLP(1+weight_dim, weight_dim, weight_dim*2, 3, activation=jax.nn.tanh, key=key)

        def inner_loss_fn(theta, t, x_t, x_prev):
            pred_x_t = self.neuralode(t, x_prev, theta)
            return jnp.mean((pred_x_t-x_t)**2)
        self.inner_loss_fn = inner_loss_fn

        self.inference = False

    def __call__(self, ts, xs, key):

        def forward(ts_, xs_, key_):
            def f(carry, inp):
                thet, t_prev, x_hat, x_prev = carry
                x_true, t_curr, k = inp

                if not self.inference:
                    x_t = jnp.where(jax.random.bernoulli(k, forcing_prob), x_true, x_hat)
                else:
                    x_t = jnp.where(t_curr<grounding_length/784, x_true, x_hat)

                ## The new theta is obtained by one step of gradient descent (TODO: use fixed point)
                for _ in range(num_gradient_steps):
                    grad = eqx.filter_grad(self.inner_loss_fn)(thet, t_curr, x_t, x_prev)
                    thet = thet - inner_learning_rate*grad

                # ## The new theta is a simple transformation of the old theta
                # phi_in = jnp.concatenate([jnp.array([t_curr]), thet])
                # thet = self.phi(phi_in)

                ## Predict the next state
                deltat = t_curr-t_prev
                x_hat = self.neuralode(t_curr+deltat, x_true, thet).squeeze()

                return (thet, t_curr, x_hat, x_true), (x_hat,)

            ## Jax scan
            keys_ = jax.random.split(key_, xs_.shape[0]-1)
            _, (x_hats,) = jax.lax.scan(f, 
                                        (self.neuralode.theta, ts_[0], xs_[0], xs_[0]), 
                                        (xs_[1:], ts_[1:], keys_))
            return jnp.concatenate([xs_[0:1], x_hats], axis=0)

        keys = jax.random.split(key, xs.shape[0])
        return eqx.filter_vmap(forward, in_axes=(None, 0, 0))(ts, xs, keys)


#%% Create time series from MNIST digits
def create_time_series(
    images: jnp.ndarray, 
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create a time series from MNIST digits.
    """

    # Extract and flatten the images
    num_images = images.shape[0]
    time_series = images.reshape(num_images, -1)

    num_steps = time_series.shape[1]
    ts = jnp.linspace(0, 1, num_steps)

    return ts, time_series


#%% Define loss function and training step

@eqx.filter_jit
def loss_fn(model: Model, images, key) -> jnp.ndarray:
    """Compute MSE loss for the overal model."""
    ts, xs = create_time_series(images)

    # Solve ODE from t0 to current time
    pred_state = model(ts, xs, key)

    # Compute MSE loss
    losses = jnp.mean((pred_state - xs) ** 2)

    # Return mean loss across all time points
    return jnp.mean(losses)

@eqx.filter_jit
def train_step(
    model: NeuralODE, 
    batch: jnp.ndarray,
    opt_state: Any, 
    optimizer: optax.GradientTransformation, 
    key: jnp.ndarray
) -> Tuple[NeuralODE, Any, jnp.ndarray]:
    """Perform a single training step using JAX and Equinox."""
    loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model, batch, key)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value

#%% Select two MNIST digits and visualise as a time series
for images, labels in train_loader:
    # Create time series
    ts, xs = create_time_series(images)

    ## Print min and max in the entire bactch
    print("min and max in the batch", np.min(xs), np.max(xs))
    
    ## Select two digits and visualize as a time series
    idx1 = 0
    idx2 = 1
    digit1 = labels[idx1]
    digit2 = labels[idx2]

    plt.figure(figsize=(12, 3))
    plt.plot(ts, xs[idx1], label=f"Digit {digit1}")
    plt.plot(ts, xs[idx2], label=f"Digit {digit2}")
    plt.xlabel('Time')
    plt.ylabel('Pixel Intensity')
    plt.title('MNIST Digit Time Series')
    plt.legend()
    plt.grid(True)
    plt.show()

    break

#%% Initialize Neural ODE model and optimizer

mother_key = jax.random.PRNGKey(seed)
model_key, train_key = jax.random.split(mother_key)
model = Model(state_dim, hidden_dim, model_key)

print(f"Number of parameters in the root: {count_params(model.neuralode)/1000: 0.2f} k")
print(f"Number of parameters in the model: {count_params(model)/1000: 0.2f} k")


# Initialize optimizer
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

#%% Train the model
losses = []

start_time = time.time()
for epoch in range(num_epochs):

    for images, _ in train_loader:
        _, train_key = jax.random.split(train_key)
        model, opt_state, loss = train_step(model, images, opt_state, optimizer, train_key)

        losses.append(loss)

        # break

    eqx.tree_serialise_leaves(checkpoints_folder+f"model_{epoch}.eqx", model)

    if epoch % print_every == 0:
        print(f"Epoch {epoch}/{num_epochs}, LatestLoss: {loss:.6f}")

end_time = time.time()
hours, secs = divmod(end_time-start_time, 3600)
mins, secs = divmod(secs, 60)
print(f"Training time: {hours:.0f}h {mins:.0f}m {secs:.0f}s")

#%% Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training Loss - Neural ODE')
plt.grid(True)
plt.show()


#%% Evaluate the model on test data

#put the model in inference mode
test_key, _ = jax.random.split(train_key)

@eqx.filter_jit
def generate_time_series(model, grounding_images, key):
    new_model = eqx.tree_at(lambda m: m.inference, model, True)
    ts, xs = create_time_series(grounding_images)
    return new_model(ts, xs, key)

test_images, test_labels = next(iter(test_loader))
test_series = generate_time_series(model, test_images, test_key)

## Select a time series at random, and visualize it both as time series and as image
idx = np.random.randint(0, batch_size)

plt.figure(figsize=(12, 3))
plt.plot(ts, test_series[idx], label='Generated')
plt.plot(ts, test_images[idx].reshape(-1), linestyle='--', label='Original')
plt.xlabel('Time')
plt.ylabel('Pixel Intensity')
plt.title('Generated Time Series: Label = {}'.format(test_labels[idx]))
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(5*2, 5))
plt.subplot(1, 2, 1)
plt.imshow(test_images[idx].reshape(28, 28), cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(test_series[idx].reshape(28, 28), cmap='gray')
plt.title('Generated Image')
plt.axis('off')
plt.show()
