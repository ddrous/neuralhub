#%%
import jax

import jax.numpy as jnp

import numpy as np

import equinox as eqx

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#%%
## Load the data (.npz files in the results folder)

nb_runs = 3

Ts = []
losses = []

for i in range(nb_runs):
    data = np.load(f"results/{9996+i:04d}.npz")
    Ts.append(data['time_horizon'])
    losses.append(data['losses'])

epochs = np.arange(losses[0].shape[0]+1)
losses = jnp.stack(losses)





#%%
## 2D imshow plot with the loss againts the epochs and time horizon T

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
pcm = ax.imshow(losses, aspect='auto', cmap='coolwarm', interpolation='none', origin='lower', norm=mcolors.LogNorm(vmin=losses.min(), vmax=losses.max()))

## Add colorbar
cbar = fig.colorbar(pcm, ax=ax, extend='both', label='MSE')

ax.set_xlabel('Epoch')
ax.set_ylabel('Time horizon T')
ax.set_title('Loss Evolution With Time Horizon T')

## Set x ticks and labels to the epochs
ax.set_xticks(np.arange(0, losses[0].shape[0]+1, 500))
ax.set_xticklabels(epochs[::500])

## Set y ticks and labels to the time horizon
ax.set_yticks(np.arange(len(Ts)))
ax.set_yticklabels(Ts)

plt.savefig(f"results/loss_imshow.png", dpi=300, bbox_inches='tight')
# %%
