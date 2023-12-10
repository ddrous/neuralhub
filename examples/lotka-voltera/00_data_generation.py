
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

import jax
# jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import diffrax

# Define the Lotka-Volterra system
def lotka_volterra(t, state, alpha, beta, delta, gamma):
    x, y = state
    dx_dt = alpha * x - beta * x * y
    dy_dt = delta * x * y - gamma * y
    return [dx_dt, dy_dt]

# @jax.jit
# def lotka_volterra(t, state, alpha, beta, delta, gamma):
#     # jax.debug.breakpoint()
#     # print("State is", state.shape)
#     x, y = state[0], state[1]
#     dx_dt = alpha * x - beta * x * y
#     dy_dt = delta * x * y - gamma * y
#     return jnp.array([dx_dt, dy_dt])


## Test the function
# res = lotka_volterra(0, np.array([1.0, 1.0]), 1.0, 1.0, 1.0, 1.0)
# print("Result is", res)

# Define the set of parameter possibilities
# environments = [
#     {"alpha": 1.5, "beta": 1.0, "gamma": 1.0, "delta": 3.0},    ## TODO remove this!
#     {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 0.5},
#     {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.5},
#     {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.5},
#     {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 0.75},
#     {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 1.0},
#     {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.75},
#     {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 1.0},
#     {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.75},
#     {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 1.0},
# ]


environments = [
    {"alpha": 1.5, "beta": 0.5, "gamma": 1.0, "delta": 2.5},    ## TODO remove this!
    {"alpha": 1.5, "beta": 0.5, "gamma": 1.0, "delta": 3.0},
    {"alpha": 1.5, "beta": 0.5, "gamma": 1.0, "delta": 3.5},
    {"alpha": 1.5, "beta": 1.0, "gamma": 1.0, "delta": 2.5},
    {"alpha": 1.5, "beta": 1.0, "gamma": 1.0, "delta": 3.0},
    {"alpha": 1.5, "beta": 1.0, "gamma": 1.0, "delta": 3.5},
    {"alpha": 1.5, "beta": 1.5, "gamma": 1.0, "delta": 2.5},
    {"alpha": 1.5, "beta": 1.5, "gamma": 1.0, "delta": 3.0},
    {"alpha": 1.5, "beta": 1.5, "gamma": 1.0, "delta": 3.5},
    # {"alpha": 1.5, "beta": 1.25, "gamma": 1.0, "delta": 1.0},
]


n_traj_per_env = 128*100
# n_traj_per_env = 1
n_steps_per_traj = 501

data = np.zeros((len(environments), n_traj_per_env, n_steps_per_traj, 2))

# selected_params = np.random.choice(parameter_possibilities)
# initial_state = [1.0, 0.75]  # Initial concentrations

# Time span for simulation
t_span = (0, 10)  # Shortened time span
t_eval = np.linspace(t_span[0], t_span[-1], n_steps_per_traj)  # Fewer frames

for j in range(n_traj_per_env):
    # Initial conditions (prey and predator concentrations)
    initial_state = np.random.uniform(-0.6, 2, (2,))  # Sample initial concentrations
    initial_state = jnp.abs(initial_state)

    for i, selected_params in enumerate(environments):
        # print("Environment", i)

        # Solve the ODEs using SciPy's solve_ivp
        solution = solve_ivp(lotka_volterra, t_span, initial_state, args=(selected_params["alpha"], selected_params["beta"], selected_params["delta"], selected_params["gamma"]), t_eval=t_eval)

        # Solve the ODEs using Diffrax
        # solution = diffrax.diffeqsolve(
        #             diffrax.ODETerm(lambda t, x, args: lotka_volterra(t, x, *args)),
        #             diffrax.Tsit5(),
        #             args=(selected_params["alpha"], selected_params["beta"], selected_params["delta"], selected_params["gamma"]),
        #             t0=t_eval[0],
        #             t1=t_eval[-1],
        #             dt0=t_eval[1] - t_eval[0],
        #             y0=initial_state,
        #             stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        #             saveat=diffrax.SaveAt(ts=t_eval),
        #             max_steps=4096*1,
        #         )

        data[i, j, :, :] = solution.y.T
        # data[i, j, :, :] = solution.ys

# Extract the solution
prey_concentration, predator_concentration = solution.y
# prey_concentration, predator_concentration = solution.ys

# Create an animation of the Lotka-Volterra system
fig, ax = plt.subplots()
ax.set_xlim(0, np.max(prey_concentration))
ax.set_ylim(0, np.max(predator_concentration))
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



# %%
