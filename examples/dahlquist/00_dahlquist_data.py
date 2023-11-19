#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from graphpint.utils import *

# Define the Dahlquist ODE
def dahlquist(t, y):
    return -5 * y

# Set the initial conditions
t_span = (0, 1)
y0 = [1.0]
t_eval = np.linspace(0, 1, 10001)

# Solve the ODE using solve_ivp
sol = solve_ivp(dahlquist, t_span, y0, method='RK45', t_eval=t_eval)

# Plot the solution
plt.plot(sol.t, sol.y[0])
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.title('Dahlquist Test Problem')
plt.show()

## Save t_eval and the solutuin to a npz file
# np.savez('data/dahlquist_n5.npz', t=t_eval, X=sol.y)

# %%

## Experiments recovering the solution from the data

def mega_dahlquist(t, y):
    return -13 * y

# Set the initial conditions
t_span = (0, 1)
other_init = 12.0
y0 = [1.0 * other_init]
t_eval = np.linspace(0, 1, 10001)

# Solve the ODE using solve_ivp
mega_sol = solve_ivp(mega_dahlquist, t_span, y0, method='RK45', t_eval=t_eval)

hat_sol = mega_sol.y * np.exp(8 * sol.t) / other_init

# Plot the solution

ax = sbplot(sol.t, sol.y[0], "x--", markevery=100, label='True', x_label='Time', title='Dahlquist with $\\lambda = -5$')
ax = sbplot(sol.t, hat_sol[0], label='Reconstructed', x_label='Time', ax =ax)


# %%
