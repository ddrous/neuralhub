#%%
import jax

import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#%%
## Load the data (.npz files in the results folder)

nb_runs = 101

Ts = []
losses = []

for i in range(nb_runs):
    data = np.load(f"results/{i:05d}.npz")
    Ts.append(data['time_horizon'])
    losses.append(data['losses'])

epochs = np.arange(losses[0].shape[0]+1)
losses = jnp.stack(losses)





#%%
## 2D imshow plot with the loss againts the epochs and time horizon T

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
pcm = ax.imshow(losses, aspect='auto', cmap='turbo', interpolation='none', origin='lower', norm=mcolors.LogNorm(vmin=losses.min(), vmax=losses.max()))

## Add colorbar
cbar = fig.colorbar(pcm, ax=ax, label='MSE')

ax.set_xlabel('Epoch')
ax.set_ylabel('Time horizon T')
ax.set_title('Loss Evolution With Various Time Horizons')

## Set x ticks and labels to the epochs
ax.set_xticks(np.arange(0, losses[0].shape[0]+1, 500))
ax.set_xticklabels(epochs[::500])

## Set y ticks and labels to the time horizon
ax.set_yticks(np.arange(len(Ts))[::10])
ax.set_yticklabels(Ts[::10])

plt.savefig(f"results/loss_imshow.png", dpi=300, bbox_inches='tight')
# %%

## 1D plot of one loss curve for a specific time horizon T

fig, ax = plt.subplots(1, 1, figsize=(8.5, 3))
ax.plot(losses[50], label=f'T = {Ts[50]:.2f}')
ax.set_xlabel('Epoch')
# ax.set_ylabel('MSE')
ax.set_yscale('log')
ax.set_title(f'Loss Curve for T = {Ts[50]:.1f}')

plt.legend()
plt.savefig(f"results/loss_curve_{Ts[50]}.png", dpi=300, bbox_inches='tight')
