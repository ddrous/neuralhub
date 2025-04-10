import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional, Tuple

class MSDSystem:
    """Simulate Mass-Spring-Damper system dynamics."""
    def __init__(self, 
                 mass: float = 1.0, 
                 spring_constant: float = 10.0, 
                 damping: float = 0.5):
        self.mass = mass
        self.k = spring_constant
        self.c = damping
    
    def generate_trajectories(self, 
                               num_samples: int = 1000, 
                               dt: float = 0.01, 
                               total_time: float = 10.0):
        """Generate MSD system trajectories."""
        times = np.linspace(0, total_time, int(total_time/dt))
        trajectories = []
        
        for _ in range(num_samples):
            # Random initial conditions
            x0 = np.random.uniform(-1, 1)
            v0 = np.random.uniform(-1, 1)
            
            # Simulate trajectory
            x = np.zeros_like(times)
            v = np.zeros_like(times)
            x[0], v[0] = x0, v0
            
            for t in range(1, len(times)):
                a = -(self.k/self.mass) * x[t-1] - (self.c/self.mass) * v[t-1]
                v[t] = v[t-1] + a * dt
                x[t] = x[t-1] + v[t] * dt
            
            trajectories.append(x)
        
        return np.array(trajectories)

class LSTMGenerativeModel(eqx.Module):
    """LSTM-based generative model for sequential data."""
    lstm: eqx.nn.LSTMCell
    linear_output: eqx.nn.Linear
    initial_hidden: jnp.ndarray
    initial_cell: jnp.ndarray

    def __init__(self, 
                 input_size: int, 
                 hidden_size: int = 64, 
                 output_size: int = 1, 
                 key: jax.random.PRNGKey = jax.random.PRNGKey(42)):
        lstm_key, hidden_key, linear_key = jax.random.split(key, 3)
        
        self.lstm = eqx.nn.LSTMCell(input_size, hidden_size, key=lstm_key)
        self.linear_output = eqx.nn.Linear(hidden_size, output_size, key=linear_key)
        
        # Initialize hidden and cell states
        self.initial_hidden = jax.random.normal(hidden_key, (hidden_size,))
        self.initial_cell = jax.random.normal(hidden_key, (hidden_size,))

    def __call__(self, 
                 x: jnp.ndarray, 
                 initial_context: Optional[jnp.ndarray] = None,
                 steps: int = 100):
        """Generate sequential data."""
        batch_size = x.shape[0]
        
        # Use initial context or default to zeros
        if initial_context is None:
            initial_context = jnp.zeros((batch_size, x.shape[1]))
        
        # Prepare initial states
        hidden = jnp.repeat(self.initial_hidden[None, :], batch_size, axis=0)
        cell = jnp.repeat(self.initial_cell[None, :], batch_size, axis=0)
        
        # Store generated sequences
        generated_sequence = []
        
        # Initial input is the context
        current_input = initial_context
        
        for _ in range(steps):
            # LSTM step
            hidden, cell = jax.vmap(self.lstm)(current_input, (hidden, cell))
            
            # Output prediction
            output = jax.vmap(self.linear_output)(hidden)
            generated_sequence.append(output)
            
            # Use the output as next input
            current_input = output
        
        return jnp.stack(generated_sequence, axis=1)

def create_dataset(dataset_type: str = 'mnist', 
                   batch_size: int = 64, 
                   num_workers: int = 4):
    """Create dataset loaders for MNIST or MSD."""
    if dataset_type == 'mnist':
        # MNIST Dataset
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
        
        return train_loader
    
    elif dataset_type == 'msd':
        # Mass-Spring-Damper Dataset
        msd_system = MSDSystem()
        trajectories = msd_system.generate_trajectories()
        
        # Convert to torch dataset
        tensor_data = torch.FloatTensor(trajectories)
        dataset = torch.utils.data.TensorDataset(tensor_data)
        
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
    
    else:
        raise ValueError("Dataset type must be 'mnist' or 'msd'")

def loss_fn(model, x, initial_context):
    """Compute MSE loss for generative model."""
    generated = model(x, initial_context)
    return jnp.mean((generated - x) ** 2)

@eqx.filter_jit
def make_step(model, x, initial_context, opt_state, optimizer):
    """Perform a single training step."""
    loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model, x, initial_context)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss_value, model, opt_state

def train(dataset_type: str = 'mnist', 
          epochs: int = 10, 
          batch_size: int = 64, 
          learning_rate: float = 1e-3):
    """Training loop for the generative model."""
    key = jax.random.PRNGKey(42)
    
    # Create dataset
    dataloader = create_dataset(dataset_type, batch_size)
    
    # Determine input and output sizes
    if dataset_type == 'mnist':
        input_size = 28 * 28  # Flattened MNIST image
        steps = 28 * 28  # Generate full image
    else:  # MSD
        input_size = 1  # Single dimension for MSD
        steps = 100  # MSD trajectory length
    
    # Initialize model and optimizer
    model = LSTMGenerativeModel(input_size, key=key)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch in dataloader:
            if dataset_type == 'mnist':
                x = batch[0].numpy().reshape(batch_size, -1)
                initial_context = x[:, :300]  # First 300 pixels as context
            else:
                x = batch[0].numpy()
                initial_context = x[:, :10]  # First 10 time steps as context
            
            x = jnp.asarray(x)
            initial_context = jnp.asarray(initial_context)
            
            loss, model, opt_state = make_step(model, x, initial_context, opt_state, optimizer)
            total_loss += loss
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")
    
    return model

# Example usage
if __name__ == "__main__":
    # Train on MNIST
    mnist_model = train(dataset_type='mnist')
    
    # Train on MSD
    # msd_model = train(dataset_type='msd')