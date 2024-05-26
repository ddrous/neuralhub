

#%%

import numpy as np
from neuralhub.utils import *
import matplotlib.pyplot as plt


#%%

SEED = 27

## Dataset hps
window_size = 100

## Plotting hps 
plt_hor = 1000


#%%


sp = np.load('data/simple_pendulum.npz')
X1 = np.concatenate([sp['X'].T, sp['t'][:, None]], axis=-1)

ip = np.load('data/inverted_pendulum.npz')
X2 = np.concatenate([ip['X'].T, ip['t'][:, None]], axis=-1)

print("Datasets sizes:", X1.shape, X2.shape)

# %%


sp_to_plot = X1[:plt_hor]
ax = sbplot(sp_to_plot[:, -1], sp_to_plot[:, 0], "--", x_label='Time', label=r'$\theta$', title='Simple Pendulum')
ax = sbplot(sp_to_plot[:, -1], sp_to_plot[:, 1], "--", x_label='Time', label=r'$\dot \theta$', ax=ax)

ip_to_plot = X2[:plt_hor]
ax = sbplot(ip_to_plot[:, -1], ip_to_plot[:, 2], "--", x_label='Time', label=r'$\theta$', title='Inverted Pendulum')
ax = sbplot(ip_to_plot[:, -1], ip_to_plot[:, 3], "--", x_label='Time', label=r'$\dot \theta$', ax=ax)
ax = sbplot(ip_to_plot[:, -1], ip_to_plot[:, 0], "--", x_label='Time', label=r'$x$', ax=ax)
ax = sbplot(ip_to_plot[:, -1], ip_to_plot[:, 1], "--", x_label='Time', label=r'$\dot x$', ax=ax)



#%%
from pydmd import BOPDMD
from pydmd.plotter import plot_summary

# Build a bagging, optimized DMD (BOP-DMD) model.
dmd = BOPDMD(
    svd_rank=15,  # rank of the DMD fit
    compute_A=False,
    num_trials=100,  # number of bagging trials to perform
    trial_size=0.5,  # use 50% of the total number of snapshots per trial
    eig_constraints={"imag", "conjugate_pairs"},  # constrain the eigenvalue structure
    varpro_opts_dict={"tol":0.5, "verbose":True},  # set variable projection parameters
)

# Fit the DMD model.
# X = (n, m) numpy array of time-varying snapshot data
# t = (m,) numpy array of times of data collection
X = X1[:, :-1].T
t = X1[:, -1]
dmd.fit(X, t)

# Display a summary of the DMD results.
plot_summary(dmd)

# %%

vars(dmd)

# %%


# for mode in dmd.modes.T:
#     plt.plot(x, mode.real)
#     plt.title('Modes')
# plt.show()


# plt_hor = 1000

# for dynamic in dmd.dynamics:
#     plt.plot(t[:plt_hor], dynamic.real[:plt_hor])
#     print(dynamic.shape)
#     plt.title('Dynamics')
# plt.show()


# %%

plt_hor = 1000


sp_to_plot_ = dmd.reconstructed_data.T.real[:plt_hor]
sp_to_plot = X.T[:plt_hor]
t_ = t[:plt_hor]

ax = sbplot(t_, sp_to_plot_[:, 0], x_label='Time', label=r'$\hat \theta$', title='Reconstruction')
ax = sbplot(t_, sp_to_plot[:, 0], "g--", lw=1, x_label='Time', label=r'$\theta$',ax=ax)

ax = sbplot(t_, sp_to_plot_[:, 1], x_label='Time', label=r'$\hat \dot \theta$', ax=ax)
ax = sbplot(t_, sp_to_plot[:, 1], "y--", lw=1, x_label='Time', label=r'$\dot \theta$', ax=ax)

# ip_to_plot_ = dmd.reconstructed_data.T.real[:plt_hor]
# ip_to_plot = X.T[:plt_hor]
# t_ = t[:plt_hor]

# ax = sbplot(t_, ip_to_plot_[:, 2], x_label='Time', label=r'$\hat \theta$', title='Inverted Pendulum')
# ax = sbplot(t_, ip_to_plot[:, 2], "g--", lw=1, x_label='Time', label=r'$\theta$', title='Inverted Pendulum', ax=ax)

# ax = sbplot(t_, ip_to_plot_[:, 3], x_label='Time', label=r'$\hat \dot \theta$', ax=ax)
# ax = sbplot(t_, ip_to_plot[:, 3], "y--", lw=1, x_label='Time', label=r'$\dot \theta$', ax=ax)

# ax = sbplot(t_, ip_to_plot_[:, 0], x_label='Time', label=r'$\hat x$', ax=ax)
# ax = sbplot(t_, ip_to_plot[:, 0], "m--", lw=1, x_label='Time', label=r'$x$', ax=ax)

# ax = sbplot(t_, ip_to_plot_[:, 1], x_label='Time', label=r'$\hat \dot x$', ax=ax)
# ax = sbplot(t_, ip_to_plot[:, 1], "r--", lw=1, x_label='Time', label=r'$\dot x$', ax=ax)

# %%
