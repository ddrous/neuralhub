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
             rc={"lines.linewidth": 3})
mpl.rcParams['savefig.facecolor'] = 'w'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
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

t_span = (0, 15)
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
                       [2, -1], 
                    #    [2, -0.5], [2, 0.5], [2, 1],
                       ])


train_data = []

for i, state0 in enumerate(ind):
    sol = solve_ivp(duffing, t_span, state0, args=(a, b, c), t_eval=t_eval)
    train_data.append(sol.y.T)

    ## Plot the phase space
    ax.plot(sol.y[0][::skip], sol.y[1][::skip], "-", label=f"Training" if i==0 else None, alpha=0.4, color="grey")

    ## Add a few arrows to indicate the direction of the flow
    for i in range(0, len(sol.y[0][::skip])-1, 500):
        ax.arrow(sol.y[0][::skip][i], sol.y[1][::skip][i], 
                 sol.y[0][::skip][i+1] - sol.y[0][::skip][i], 
                 sol.y[1][::skip][i+1] - sol.y[1][::skip][i],
                 head_width=0.04, head_length=0.1, fc='grey', ec='grey', alpha=0.5)




## Simulate and plot OOD1 data (changes the physical parameters)
for i, params in enumerate(ood_1):
    sol = solve_ivp(duffing, t_span, ind[0], args=tuple(params), t_eval=t_eval)
    train_data.append(sol.y.T)

    ## Plot the phase space
    ax.plot(sol.y[0][::skip], sol.y[1][::skip], "-", label=f"OoD Type 1" if i==0 else None, alpha=0.99, color="blue", lw=5)

    ## Add a few arrows to indicate the direction of the flow
    for i in range(0, len(sol.y[0][::skip])-1, 500):
        ax.arrow(sol.y[0][::skip][i], sol.y[1][::skip][i], 
                 sol.y[0][::skip][i+1] - sol.y[0][::skip][i], 
                 sol.y[1][::skip][i+1] - sol.y[1][::skip][i],
                 head_width=0.09, head_length=0.1, fc='blue', ec='blue', alpha=0.5)


## Simulate and plot OOD2 data (changes the initial conditions, easy peasy)
for i, state0 in enumerate(ood_2):
    sol = solve_ivp(duffing, t_span, state0, args=(a, b, c), t_eval=t_eval)
    train_data.append(sol.y.T)

    ## Plot the phase space
    ax.plot(sol.y[0][::skip], sol.y[1][::skip], "-", label=f"OoD Type 2" if i==0 else None, alpha=0.99, color="green", lw=5)

    ## Add a few arrows to indicate the direction of the flow
    for i in range(0, len(sol.y[0][::skip])-1, 500):
        ax.arrow(sol.y[0][::skip][i], sol.y[1][::skip][i], 
                 sol.y[0][::skip][i+1] - sol.y[0][::skip][i], 
                 sol.y[1][::skip][i+1] - sol.y[1][::skip][i],
                 head_width=0.09, head_length=0.1, fc='green', ec='green', alpha=0.5)


# OOD3 initial conditions - using some from your existing sets
ood_3 = np.array([
    # [-1, 0.25],      # Similar to training data
    # [0.5, 1],     # Similar to training data  
    # [-2, -0.5],    # Similar to training data
    [-0.5, 1], 
    [-1, 1]
])

# Alternative OOD3 option: Duffing-Van der Pol hybrid
def duffing_vdp_hybrid(t, state, a, b, c, mu):
    """
    Hybrid system combining Duffing and other characteristics
    dx/dt = y
    dy/dt = a*y - x*(b + c*x²) + μ(1-x²)y
    """
    x, y = state
    dxdt = y
    # dydt = a*y - x*(b + c*x**2) + mu * (1 - x**2) * y
    dydt = a*y - x*(b + c*x**2) + mu*y**2
    return [dxdt, dydt]

# Hybrid parameters
mu_hybrid = -0.2  # Small 

# Uncomment below to use hybrid instead of pure Van der Pol:
# """
## Simulate and plot OOD3 data (Duffing-Something else)
for i, state0 in enumerate(ood_3):
    sol = solve_ivp(duffing_vdp_hybrid, t_span, state0, args=(a, b, c, mu_hybrid), t_eval=t_eval)
    train_data.append(sol.y.T)

    ## Plot the phase space
    ax.plot(sol.y[0][::skip], sol.y[1][::skip], "-", label=f"OoD Type 3" if i==0 else None, alpha=0.99, color="crimson", lw=5)

    ## Add a few arrows to indicate the direction of the flow
    for j in range(0, len(sol.y[0][::skip])-1, 500):
        ax.arrow(sol.y[0][::skip][j], sol.y[1][::skip][j], 
                 sol.y[0][::skip][j+1] - sol.y[0][::skip][j], 
                 sol.y[1][::skip][j+1] - sol.y[1][::skip][j],
                 head_width=0.09, head_length=0.1, fc='crimson', ec='crimson', alpha=0.5)
# """





ax.set_xlabel(r'Displacement ($x$)')
ax.set_ylabel(r'Velocity ($\dot x$)')
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
# plt.savefig(f"data/duffing_ood_types.png", dpi=300, bbox_inches='tight')
plt.savefig(f"data/duffing_ood_types.pdf", dpi=300, bbox_inches='tight')
