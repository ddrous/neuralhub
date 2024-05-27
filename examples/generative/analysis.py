#%%
import jax

import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#%%
## Load the data (.npz files in the results folder)

savefolder = "reults_1/"
nb_runs = 15

Ts = []
losses = []
for i in range(nb_runs):
    data = np.load(f"{savefolder}{i:05d}.npz")

    Ts.append(data['time_horizon'])
    losses.append(data['values'])

epochs = np.arange(losses[0].shape[0]+1)
losses = jnp.stack(losses)
Ts = np.array(Ts)


#%%
## 2D imshow plot with the loss againts the epochs and time horizon T

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
losses_plot = losses[:,:]
pcm = ax.imshow(losses_plot, aspect='auto', cmap='turbo', interpolation='none', origin='lower', norm=mcolors.LogNorm(vmin=losses_plot.min(), vmax=losses_plot.max()))

## Add colorbar
cbar = fig.colorbar(pcm, ax=ax, label='MSE')

ax.set_xlabel('Epoch')
ax.set_ylabel('Time horizon T')
ax.set_title('Loss Evolution With Various Time Horizons')

## Set x ticks and labels to the epochs
# ax.set_xticks(np.arange(0, losses[0].shape[0]+1, 500))
# ax.set_xticklabels(epochs[::500])

## Set y ticks and labels to the time horizon
Ts_ticks = np.linspace(Ts.min(), Ts.max(), 5)
ax.set_yticks(np.arange(len(Ts))[::len(Ts)//4])
ax.set_yticklabels(Ts_ticks)

plt.savefig(f"{savefolder}loss_imshow.png", dpi=300, bbox_inches='tight')
# %%

## 1D plot of one loss curve for a specific time horizon T
plot_id = nb_runs//2

fig, ax = plt.subplots(1, 1, figsize=(8.5, 3))
ax.plot(losses[50], label=f'T = {Ts[plot_id]:.2f}')
ax.set_xlabel('Epoch')
# ax.set_ylabel('MSE')
ax.set_yscale('log')
ax.set_title(f'Loss Curve for T = {Ts[plot_id]:.1f}')

plt.legend()
plt.savefig(f"{savefolder}loss_curve_{Ts[plot_id]}.png", dpi=300, bbox_inches='tight')




