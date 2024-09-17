
#%%[markdown]


#%%
%load_ext autoreload
%autoreload 2

import jax

print("\n############# Neural ODE #############\n")
print("Available devices:", jax.devices())      ## JAX will priotize the GPU if available

import jax.numpy as jnp
import equinox as eqx

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import optax
import time

#%%

SEED = 2024
print(f"Seed for reproducibility: {SEED}")


## Hyperparameters
learning_rate = 1e-4
print_every = 100
nb_epochs = 10000
skip_steps = 100        ## Skip some steps when evaluating the loss

#%%

def oscillator(t, state, *params):
    """ 3-dimensional harmonic oscillator """
    x, v, _ = state
    k, mu, m = params

    du1 = v
    du2 = -k/m*x - mu/m*v
    du3 = -v

    return [du1, du2, du3]

t_eval = np.linspace(0, 1, 2001)[::1]
initial_states = [[1.0, 1.0, 1.0]]

data = []
for y0 in initial_states:
    solution = solve_ivp(oscillator, (t_eval[0], t_eval[-1]), y0, args=(10000., 0.0, 1.0), t_eval=t_eval)

    # print("Max across channels:", np.max(solution.y.T, axis=0))
    chan1 = solution.y.T[:, 0] / np.max(solution.y.T[:, 0])
    chan2 = solution.y.T[:, 1] / np.max(solution.y.T[:, 1])
    chan3 = solution.y.T[:, 2] / np.max(solution.y.T[:, 2])

    traj = np.stack([chan1, chan2, chan3], axis=-1)

    data.append(traj)

plt.plot(t_eval, traj, label=['State 1', 'State 2', 'State 3'])
plt.xlabel('Time')
plt.ylabel('States')
plt.title('Example trajectory')
plt.legend()
plt.show()


#%%

data = jnp.stack(data, axis=-1).transpose(1, 0, 2)
all_inputs, all_outputs = data[:, :-1], data[:, 1:]
t_eval = jnp.asarray(t_eval)
print("Data and t eval shapes:", all_inputs.shape, all_outputs.shape, t_eval.shape)
print("dt for the entire process:", t_eval[1]-t_eval[0])

#%%


class VectorField(eqx.Module):
    layers: list

    def __init__(self, in_size, hidden_size, out_size, key=None):
        keys = jax.random.split(key, 3)
        self.layers = [eqx.nn.Linear(in_size, hidden_size, key=keys[0]), 
                       jax.nn.softplus,
                       eqx.nn.Linear(hidden_size, hidden_size, key=keys[1]), 
                       jax.nn.softplus,
                       eqx.nn.Linear(hidden_size, out_size, key=keys[2]) ]

    def __call__(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y

mother_key = jax.random.PRNGKey(SEED)
model = VectorField(in_size=35, hidden_size=64, out_size=32, key=mother_key)



# %%


def euler_integrator(rhs, y0, inputs, t0, dt):
    """ Integrator with eurler step, scanned over leading time dimension of inputs """

    # y0 is of shape (32,)
    # inputs is of shape (n_steps, 3)

    def euler_step(carry, input_):
        y_prev, t_prev = carry
        nn_input = jnp.concatenate([y_prev, input_], axis=-1)
        y = y_prev + dt * rhs(nn_input)
        return (y, t_prev+dt), y

    _, ys = jax.lax.scan(euler_step, (y0, t0), inputs)
    # ys of shape (n_steps, 32)

    return ys


def loss_fn(model, batch):
    all_inputs, all_outputs = batch
    # all_inputs of shape (3, n_steps, n_samples)

    _, _, n_samples = all_inputs.shape

    batched_integrator = eqx.filter_vmap(euler_integrator, in_axes=(None, 0, 0, None, None))

    # all_inputs is of shape (3, n_steps, n_samples), but we want (n_samples, n_steps, 3)
    all_inputs = jnp.transpose(all_inputs, (2, 1, 0))
    u0s = jnp.zeros((n_samples, 32))

    preds = batched_integrator(model, u0s, all_inputs, t_eval[0], t_eval[1]-t_eval[0])

    ## Preds of shape (n_samples, n_steps, 32), but we want (32, n_steps, n_samples)
    preds = jnp.transpose(preds, (2, 1, 0))

    print("Shapes of predictions and outputs:", preds.shape, all_outputs.shape, "\n")

    # return jnp.mean((preds[:3, ::skip_steps] - all_outputs[:, ::skip_steps])**2)
    return jnp.mean((preds[:3, :30] - all_outputs[:, :30])**2)

opt = optax.adam(learning_rate)
opt_state = opt.init(eqx.filter(model, eqx.is_array))

@eqx.filter_jit
def train_step(model, batch, opt_state):
    print('(Re)Compiling function "train_step" ...\n')

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)

    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss




print(f"\n\n=== Beginning Training ... ===")


losses = []
batch = (all_inputs, all_outputs)

start_time = time.time()

for epoch in range(nb_epochs):

    model, opt_state, loss = train_step(model, batch, opt_state)
    losses.append(loss)

    if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
        print(f"    Epoch: {epoch:-5d}      Loss: {loss:.8f}", flush=True)

wall_time = time.time() - start_time
print("\nTotal GD training time: %d hours %d mins %d secs" % (wall_time//3600, (wall_time%3600)//60, wall_time%60))


# %%

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 3.5*2))

## Plot the loss
ax1.plot(losses, label='Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('L2')
ax1.set_yscale("log")
ax1.set_title('Loss')

## Plot the predictions and the ground truth
preds = euler_integrator(model, jnp.zeros(32), all_inputs[:,:,-1].T, t_eval[0], t_eval[1]-t_eval[0])

print("prediction:", preds.shape)

ax2.plot(t_eval[1:], preds[:, :1], label='Pred1')
ax2.plot(t_eval[1:], all_outputs[:1, :, -1].T, ".", label='GT1')
ax2.set_xlabel('Time')
ax2.set_ylabel('States')
# ax2.set_title('Predictions on last sample')

plt.legend()

