
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# Define the Lotka-Volterra system
def lotka_volterra(t, state, alpha, beta, delta, gamma):
    x, y = state
    dx_dt = alpha * x - beta * x * y
    dy_dt = delta * x * y - gamma * y
    return [dx_dt, dy_dt]

# Define the set of parameter possibilities
environments = [
    {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 0.5},
    {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.5},
    {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.5},
    {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 0.75},
    {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 1.0},
    {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.75},
    {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 1.0},
    {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.75},
    {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 1.0}
]

n_traj_per_env = 128*10
# n_traj_per_env = 1
n_steps_per_traj = 501

data = np.zeros((len(environments), n_traj_per_env, n_steps_per_traj, 2))

# selected_params = np.random.choice(parameter_possibilities)
# initial_state = [1.0, 0.75]  # Initial concentrations

# Time span for simulation
t_span = (0, 10)  # Shortened time span
t_eval = np.linspace(t_span[0], t_span[1], n_steps_per_traj)  # Fewer frames

for i, selected_params in enumerate(environments):
    for j in range(n_traj_per_env):

        # Initial conditions (prey and predator concentrations)
        initial_state = np.random.uniform(0.25, 1.25, 2)  # Sample initial concentrations

        # Solve the ODEs using SciPy's solve_ivp
        solution = solve_ivp(lotka_volterra, t_span, initial_state, args=(selected_params["alpha"], selected_params["beta"], selected_params["delta"], selected_params["gamma"]), t_eval=t_eval)

        data[i, j, :, :] = solution.y.T

# Extract the solution
prey_concentration, predator_concentration = solution.y

# Create an animation of the Lotka-Volterra system
fig, ax = plt.subplots()
ax.set_xlim(0, 1.8)
ax.set_ylim(0, 2.2)
ax.set_xlabel('Preys')
ax.set_ylabel('Predators')

concentrations, = ax.plot([], [], 'r-', lw=1, label='Concentrations')
time_template = 'Time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

# Add legend
ax.legend()

def animate(i):
    concentrations.set_data(prey_concentration[:i], predator_concentration[:i])
    time_text.set_text(time_template % solution.t[i])
    return concentrations, time_text

ani = FuncAnimation(fig, animate, frames=len(solution.t), interval=5, repeat=False, blit=True)  # Shortened interval
plt.show()

# Save t_eval and the solution to a npz file
np.savez('data/lotka_volterra_big.npz', t=solution.t, X=data)


