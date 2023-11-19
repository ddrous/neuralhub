#%%[markdown]

# # Pendulums


# %%

### Simple pendulum


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# Define the ODE for a simple pendulum
def simple_pendulum(t, state, L, g):
    theta, theta_dot = state
    theta_ddot = -(g / L) * np.sin(theta)
    return [theta_dot, theta_ddot]

# Parameters
L = 1.0  # Length of the pendulum (meters)
g = 9.81  # Acceleration due to gravity (m/s^2)

# Initial conditions (angle and angular velocity)
initial_state = [np.pi / 4, 0.0]

# Time span for simulation
t_span = (0, 100)  # Simulation for 10 seconds
t_eval = np.linspace(t_span[0], t_span[1], 10001)

# Solve the ODEs using SciPy's solve_ivp
solution = solve_ivp(simple_pendulum, t_span, initial_state, args=(L, g), t_eval=t_eval)

# Extract the solution
theta, theta_dot = solution.y

# Create an animation of the pendulum's motion
fig, ax = plt.subplots()
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)

pendulum, = ax.plot([], [], 'ro-', lw=2)
time_template = 'Time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def animate(i):
    x = [0, L * np.sin(theta[i])]
    y = [0, -L * np.cos(theta[i])]
    pendulum.set_data(x, y)
    time_text.set_text(time_template % solution.t[i])
    return pendulum, time_text

ani = FuncAnimation(fig, animate, frames=len(solution.t), interval=10, repeat=False, blit=True)
plt.show()

## Save t_eval and the solutuin to a npz file
np.savez('data/simple_pendulum.npz', t=t_eval, X=solution.y)

# %%

### Inverted pendulum with no control


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp
# from matplotlib.animation import FuncAnimation

# # Constants
# g = 9.81  # Acceleration due to gravity (m/s^2)
# L = 1.0   # Length of the pendulum (m)
# M = 1.0   # Mass of the cart (kg)
# m = 0.1   # Mass of the pendulum bob (kg)

# # Initial conditions
# x0 = 0.0    # Initial horizontal position
# x_dot0 = 0.0  # Initial horizontal velocity
# theta0 = np.pi / 4  # Initial angle
# theta_dot0 = 0.0  # Initial angular velocity

# initial_state = [x0, x_dot0, theta0, theta_dot0]

# # Time span for simulation
# t_span = (0, 10)  # Simulation for 10 seconds

# # Define the ODE for an inverted pendulum
# def inverted_pendulum(t, state):
#     x, x_dot, theta, theta_dot = state
#     x_ddot = (m * np.sin(theta) * (L * theta_dot ** 2 - g * np.cos(theta))) / (M + m * np.sin(theta) ** 2)
#     theta_ddot = (-m * L * theta_dot ** 2 * np.sin(theta) * np.cos(theta) - (M + m) * g * np.sin(theta)) / (L * (M + m * np.sin(theta) ** 2))
#     return [x_dot, x_ddot, theta_dot, theta_ddot]

# # Solve the ODEs using SciPy's solve_ivp
# solution = solve_ivp(inverted_pendulum, t_span, initial_state, t_eval=np.linspace(t_span[0], t_span[1], 1000))

# # Extract the solution
# x, x_dot, theta, theta_dot = solution.y

# # Create an animation of the inverted pendulum's motion
# fig, ax = plt.subplots()
# ax.set_xlim(-2, 2)
# ax.set_ylim(-2, 1)
# ax.set_aspect('equal')

# cart, = ax.plot([], [], lw=4, color='blue')
# pendulum, = ax.plot([], [], lw=2, color='red')
# time_template = 'Time = %.1fs'
# time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

# def animate(i):
#     cart_x = x[i]
#     pendulum_x = cart_x + L * np.sin(theta[i])
#     pendulum_y = -L * np.cos(theta[i])

#     cart.set_data([cart_x - 0.2, cart_x + 0.2], [-0.1, -0.1])
#     pendulum.set_data([cart_x, pendulum_x], [0, pendulum_y])
#     time_text.set_text(time_template % solution.t[i])
#     return cart, pendulum, time_text

# ani = FuncAnimation(fig, animate, frames=len(solution.t), interval=10, repeat=False, blit=True)

# plt.show()


# %%

### Inverted pendulum with bang-bang controls for angle and cart velocity

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# Constants
g = 9.81  # Acceleration due to gravity (m/s^2)
L = 1.0   # Length of the pendulum (m)
M = 1.0   # Mass of the cart (kg)
m = 0.1   # Mass of the pendulum bob (kg)

# Control parameters
angle_range = 0.05  # Desired angle range (-0.1 to 0.1 radians)
max_cart_velocity = 0.01  # Maximum allowed cart velocity
control_force = 10.0  # Magnitude of control force

# Initial conditions
x0 = 0.0    # Initial horizontal position
x_dot0 = 0.0  # Initial horizontal velocity
theta0 = np.pi + np.pi/12  # Initial angle (pointing up)
theta_dot0 = 0.0  # Initial angular velocity

initial_state = [x0, x_dot0, theta0, theta_dot0]

# Time span for simulation
t_span = (0, 100)  # Simulation for 10 seconds
t_eval = np.linspace(t_span[0], t_span[1], 10001)

# Define the ODE for an inverted pendulum with constrained cart velocity
def inverted_pendulum(t, state):
    x, x_dot, theta, theta_dot = state
    x_ddot = (m * np.sin(theta) * (L * theta_dot ** 2 - g * np.cos(theta))) / (M + m * np.sin(theta) ** 2)
    
    # Apply the bang-bang control force to keep the angle within the desired range
    control = 0.0

    # if abs(theta-np.pi) > angle_range:
    #     control = np.sign(np.pi-theta) * control_force

    # # Constrain the cart velocity within the specified range
    # if abs(x_dot) > max_cart_velocity:
    #     x_dot = np.sign(x_dot) * max_cart_velocity
    
    theta_ddot = (-m * L * theta_dot ** 2 * np.sin(theta) * np.cos(theta) - (M + m) * g * np.sin(theta) + control) / (L * (M + m * np.sin(theta) ** 2))
    
    return [x_dot, x_ddot, theta_dot, theta_ddot]

# Solve the ODEs using SciPy's solve_ivp
solution = solve_ivp(inverted_pendulum, t_span, initial_state, t_eval=t_eval)

# Extract the solution
x, x_dot, theta, theta_dot = solution.y

# Create an animation of the inverted pendulum's motion
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-1, 2)
ax.set_aspect('equal')

cart, = ax.plot([], [], lw=4, color='blue')
pendulum, = ax.plot([], [], lw=2, color='red')
time_template = 'Time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def animate(i):
    cart_x = x[i]
    pendulum_x = cart_x + L * np.sin(theta[i])
    pendulum_y = -L * np.cos(theta[i])

    cart.set_data([cart_x - 0.2, cart_x + 0.2], [-0.1, -0.1])
    pendulum.set_data([cart_x, pendulum_x], [0, pendulum_y])
    time_text.set_text(time_template % solution.t[i])
    return cart, pendulum, time_text

# ani = FuncAnimation(fig, animate, frames=len(solution.t), interval=10, repeat=True, blit=True)
# plt.show()

np.savez('data/inverted_pendulum.npz', t=t_eval, X=solution.y)


# %%
