#%%[markdown]
# OOD Data for the Thesis

#%%
import jax

print("Available devices:", jax.devices())

from jax import config
import jax.numpy as jnp

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
sb.set_theme(context='poster', 
             style='ticks',
             font='sans-serif', 
             font_scale=1.75, 
             color_codes=True, 
             rc={"lines.linewidth": 10})
mpl.rcParams['savefig.facecolor'] = 'w'
plt.rcParams['font.family'] = 'serif'
# plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['savefig.bbox'] = 'tight'

## Reduce the size of the dots in the plots
mpl.rcParams['lines.markersize'] = 4



# Define the Duffing system
def duffing(t, state, a, b, c):
    x, y = state
    dxdt = y
    dydt = a*y - x*(b + c*x**2)
    return [dxdt, dydt]

# Parameters
a, b, c = -1/2., -1, 1/10.
skip = 1  # Skip every 10th point for plotting
arrow_skip = 100  # Skip every 50th point for arrows

t_span = (0, 35)
t_eval = np.arange(t_span[0], t_span[1], 0.005)[::skip]


# plt.figure(figsize=(10, 6))
fig, ax = plt.subplots(1, 1, figsize=(20, 10))


### Traning data
ind = np.array([[-0.5, -1], [-0.5, -0.5], [-0.5, 0.5], 
                       [-1.5, 1], 
                    #    [-0.5, 1], 
                       [-1, -1], [-1, -0.5], [-1, 0.5], [-1, 1], 
                       [-2, -1], [-2, -0.5], [-2, 0.5], [-2, 1],
                       ])



## OOD1 changes the physical paramters a, b, anc c (slightly aroudn the base a, b, and c)
# ood_1 = [[-0.45, -1, 1/10.], [-0.45, -0.5, 1/10.], [-0.45, 0.5, 1/10.],
#          [-0.5, -1.1, 1/10.], [-0.5, -1.1, 1/10.]]
ood_1 = [[-0.45, -1, 1/10.], [-0.55, -1, 1/10.]]

### OOD2: changes the initial conditions
ood_2 = np.array([
    # [-0.5, -1], [-0.5, -0.5], [-0.5, 0.5], 
                    #    [-1.5, 1], 
                       [-0.5, 1], 
                    #    [-1, -1], [-1, -0.5], [-1, 0.5], [-1, 1], 
                    #    [-2, -1], [-2, -0.5], [-2, 0.5], [-2, 1],
                    #    [0.5, -1], [0.5, -0.5], [0.5, 0.5], [0.5, 1],
                    #    [1, -1], [1, -0.5], [1, 0.5], [1, 1],
                    #    [2, -1], 
                    #    [2, -0.5], [2, 0.5], [2, 1],
                       ])


train_data = []

for i, state0 in enumerate(ind):
    sol = solve_ivp(duffing, t_span, state0, args=(a, b, c), t_eval=t_eval)
    train_data.append(sol.y.T)

    ## Plot the phase space
    ax.plot(sol.y[0][::skip], sol.y[1][::skip], "-", label=f"Training" if i==0 else None, alpha=0.4, color="grey")

    ## Add a few arrows to indicate the direction of the flow
    for i in range(0, len(sol.y[0][::skip])-1, arrow_skip):
        ax.arrow(sol.y[0][::skip][i], sol.y[1][::skip][i], 
                 sol.y[0][::skip][i+1] - sol.y[0][::skip][i], 
                 sol.y[1][::skip][i+1] - sol.y[1][::skip][i],
                 head_width=0.04, head_length=0.1, fc='grey', ec='grey', alpha=0.5)




## Simulate and plot OOD1 data (changes the physical parameters)
for i, params in enumerate(ood_1):
    sol = solve_ivp(duffing, t_span, ind[0], args=tuple(params), t_eval=t_eval)
    train_data.append(sol.y.T)

    ## Plot the phase space
    ax.plot(sol.y[0][::skip], sol.y[1][::skip], "-", label=f"OoD Type 1" if i==0 else None, alpha=0.7, color="teal")

    ## Add a few arrows to indicate the direction of the flow
    for i in range(0, len(sol.y[0][::skip])-1, arrow_skip):
        ax.arrow(sol.y[0][::skip][i], sol.y[1][::skip][i], 
                 sol.y[0][::skip][i+1] - sol.y[0][::skip][i], 
                 sol.y[1][::skip][i+1] - sol.y[1][::skip][i],
                 head_width=0.04, head_length=0.1, fc='blue', ec='teal', alpha=0.5)


## Simulate and plot OOD2 data (changes the initial conditions, easy peasy)
for i, state0 in enumerate(ood_2):
    sol = solve_ivp(duffing, t_span, state0, args=(a, b, c), t_eval=t_eval)
    train_data.append(sol.y.T)

    ## Plot the phase space
    ax.plot(sol.y[0][::skip], sol.y[1][::skip], "-", label=f"OoD Type 2" if i==0 else None, alpha=0.7, color="blue")

    ## Add a few arrows to indicate the direction of the flow
    for i in range(0, len(sol.y[0][::skip])-1, arrow_skip):
        ax.arrow(sol.y[0][::skip][i], sol.y[1][::skip][i], 
                 sol.y[0][::skip][i+1] - sol.y[0][::skip][i], 
                 sol.y[1][::skip][i+1] - sol.y[1][::skip][i],
                 head_width=0.04, head_length=0.1, fc='green', ec='blue', alpha=0.5)


# OOD3: Forced Duffing Oscillator to show a different attractor
def forced_duffing(t, state, a, b, c, gamma, omega):
    """Forced Duffing oscillator."""
    x, y = state
    dxdt = y
    dydt = a*y - x*(b + c*x**2) + gamma * np.cos(omega * t)
    return [dxdt, dydt]

# Forcing parameters for OOD3
gamma = 1.0
omega = 1.0

# # Use a longer time span to let the system settle on the attractor
# t_span_ood3 = (0, 300)
# t_eval_ood3 = np.arange(t_span_ood3[0], t_span_ood3[1], 0.02)
# arrow_skip_ood3 = 2000

# Use a the same span as the other cases
t_span_ood3 = t_span
t_eval_ood3 = t_eval
arrow_skip_ood3 = arrow_skip

# OOD3 initial conditions
# ood_3_ics = np.array([
#     [1.0, 1.0],
# ])
ood_3_ics = np.array([
                       [-0.5, 1], 
                    #    [2, -1], 
                       ])

## Simulate and plot OOD3 data (Forced Duffing)
for i, state0 in enumerate(ood_3_ics):
    sol = solve_ivp(forced_duffing, t_span_ood3, state0, args=(a, b, c, gamma, omega), t_eval=t_eval_ood3)

    # Let the transient die out before plotting
    # transient_cutoff = len(t_eval_ood3) // 3
    transient_cutoff = 0
    x_attractor = sol.y[0][transient_cutoff:]
    y_attractor = sol.y[1][transient_cutoff:]

    ## Plot the phase space
    ax.plot(x_attractor, y_attractor, "-", label=f"OoD Type 3" if i==0 else None, alpha=0.7, color="crimson")

    ## Add a few arrows to indicate the direction of the flow
    for j in range(0, len(x_attractor)-1, arrow_skip_ood3):
        ax.arrow(x_attractor[j], y_attractor[j],
                 x_attractor[j+1] - x_attractor[j],
                 y_attractor[j+1] - y_attractor[j],
                 head_width=0.04, head_length=0.1, fc='crimson', ec='crimson', alpha=0.5)


## Remove the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

## Place arrowheads at the ends of the bottom and left spines (for bottom, put it at the right end)
# This plots a ">" marker at the right end of the x-axis (1 in axis coords)
# at the y=0 position (0 in data coords). `clip_on=False` ensures it's visible.
ax.plot(1, -2.2, ">k", transform=ax.get_yaxis_transform(), clip_on=False, markersize=15)
# This plots a "^" marker at the top end of the y-axis (1 in axis coords)
# at the x=0 position (0 in data coords).
ax.plot(-4.35, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False, markersize=15)


ax.set_xlabel(r'Displacement ($x$)', fontsize=36)
ax.set_ylabel(r'Velocity ($\dot x$)', fontsize=36)
# ax.set_title('Phase Space')

# plt.grid(True)
# plt.show()

ax.legend(loc='upper left', fontsize=30, frameon=False, ncol=2)

## Remove the xticks and labels
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_yticks([])
ax.set_yticklabels([])

plt.xlim(-4.35, 4.35)
plt.ylim(-2.2, 2.2)

plt.draw();
# Create the 'data' directory if it doesn't exist to avoid an error
import os
os.makedirs('data', exist_ok=True)
# plt.savefig(f"data/duffing_ood_types_nicer.png", dpi=100, bbox_inches='tight')
plt.savefig(f"data/duffing_ood_types_nicer.pdf", dpi=300, bbox_inches='tight')

plt.show() # Added to display the plot when running the script

