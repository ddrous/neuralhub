#%%

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt

def damped_harmonic_oscillator(x, t, gamma, omega):
    dxdt = x[1], -2 * gamma * x[1] - omega**2 * x[0]
    return dxdt

# Define parameters
gamma = 0.2  # Damping coefficient
omega = 1.0  # Angular frequency

# Initial conditions
x0 = jnp.array([1.0, 0.0])  # Initial position and velocity

# Time points
times = (0, 50, 1001)
t = jnp.linspace(*times)

# Numerically solve the damped harmonic oscillator
x_sol = odeint(damped_harmonic_oscillator, x0, t, gamma, omega)

# Extract position and velocity
position = x_sol[:, 0]
velocity = x_sol[:, 1]

# Create a plot to visualize the damped harmonic oscillator behavior
plt.figure(figsize=(10, 6))
plt.plot(t, position, label="Position (x)")
plt.plot(t, velocity, label="Velocity (dx/dt)")
plt.title("Damped Harmonic Oscillator - JAX Odeint")
plt.xlabel("Time")
plt.legend()
# plt.grid()
plt.show()





#%%
import numpy as np

N = 200
n_s = np.arange(N)
t0, tf = times[0], times[1]
t0_s = t0 + n_s*(tf-t0)/N
tf_s = t0 + (n_s+1)*(tf-t0)/N
t_s = jax.vmap(lambda t0, tf: jnp.linspace(t0, tf, 2), in_axes=(0, 0))(t0_s, tf_s)

# t_s.shape

# x0_s = jax.random.normal(jax.random.PRNGKey(0), (N+1, 2))
x0_s = jnp.broadcast_to(x0, (N+1, 2))
x0_s




#%%


##### We are constructing the initialisation !!!

import equinox as eqx

class Oscillator(eqx.Module):
    gamma: float
    omega: float
    def __init__(self, gamma, omega):
        self.gamma = gamma
        self.omega = omega
    def __call__(self, x, t):
        dxdt = x[1], -2 * self.gamma * x[1] - self.omega**2 * x[0]
        # return dxdt
        return jnp.array(dxdt)

osc_eqx = Oscillator(gamma, omega)
osc_eqx_p, osc_eqx_s = eqx.partition(osc_eqx, eqx.is_array)

# def rk4_integrator(rhs_params, static, y0, t):
#   rhs = eqx.combine(rhs_params, static)
#   def step(state, t):
#     y_prev, t_prev = state
#     h = t - t_prev
#     k1 = h * rhs(y_prev, t_prev)
#     k2 = h * rhs(y_prev + k1/2., t_prev + h/2.)
#     k3 = h * rhs(y_prev + k2/2., t_prev + h/2.)
#     k4 = h * rhs(y_prev + k3, t + h)
#     y = y_prev + 1./6 * (k1 + 2 * k2 + 2 * k3 + k4)
#     return (y, t), y
#   _, ys = jax.lax.scan(step, (y0, t[0]), t[1:])
#   # return ys
#   return jnp.concatenate([y0[jnp.newaxis, :], ys], axis=0)

def euler_integrator(rhs_params, static, y0, t):
  """hmax is never used, but is here for compatibility with other integrators """
  rhs = eqx.combine(rhs_params, static)
  def step(state, t):
    # rhs = eqx.combine(rhs_params, static)
    y_prev, t_prev = state
    dt = t - t_prev
    y = y_prev + dt * rhs(y_prev, t_prev)
    return (y, t), y
  _, ys = jax.lax.scan(step, (y0, t[0]), t[1:])
  # return ys
  return jnp.concatenate([y0[jnp.newaxis, :], ys], axis=0)

# Numerically solve the damped harmonic oscillator
t = jnp.linspace(t0, tf, N+1)
x_sol = euler_integrator(osc_eqx_p, osc_eqx_s, x0, t)

# Extract position and velocity
position = x_sol[:, 0]
velocity = x_sol[:, 1]

# Create a plot to visualize the damped harmonic oscillator behavior
plt.figure(figsize=(10, 6))
plt.plot(t, position, label="Position (x)")
plt.plot(t, velocity, label="Velocity (dx/dt)")
plt.title("Damped Harmonic Oscillator - JAX Odeint")
plt.xlabel("Time")
plt.legend()
# plt.grid()
plt.show()




#%%



import jraph
import jax.numpy as jnp


# Define a three node graph, each node has an integer as its feature.
# factor = x_sol.shape[0] // N
# noise = jax.random.normal(jax.random.PRNGKey(0), (N, 2)) / 100
# node_features = x0_s.at[1:,:].set(x_sol[1::factor,:] + noise)

noise = jax.random.normal(jax.random.PRNGKey(0), (N+1, 2)) / 100
node_features = x_sol[:,:] + noise

# We will construct a graph for which there is a directed edge between each node
# and its successor. We define this with `senders` (source nodes) and `receivers`
# (destination nodes).
senders = n_s[:]
receivers = n_s[:]+1

# You can optionally add edge attributes.
edges = {"times":t_s, "messages":jnp.zeros_like(t_s)}

# We then save the number of nodes and the number of edges.
# This information is used to make running GNNs over multiple graphs
# in a GraphsTuple possible.
n_node = jnp.array([N+1])
n_edge = jnp.array([N])

# Optionally you can add `global` information, such as a graph label.

# global_context = jnp.array([[1.]])
global_context = None

graph = jraph.GraphsTuple(nodes=node_features, senders=senders, receivers=receivers,
edges=edges, n_node=n_node, n_edge=n_edge, globals=global_context)


# node_targets = jnp.array([[True], [False], [True]])
# graph = graph._replace(nodes={'inputs': graph.nodes, 'targets': node_targets})
# graph = graph._replace(nodes={'features': graph.nodes, 'ids': jnp.array([[0], [1], [2]])})

print(graph.edges["messages"])


def my_odeint(params, static, x_init, t_eval):
    osc_eqx = lambda x, t: eqx.combine(params, static)(x,t)
    return odeint(osc_eqx, x_init, t_eval)

osc_vmapped = jax.vmap(my_odeint, in_axes=(None, None, 0, 0))


# As one example, we just pass the edge features straight through.
def update_edge_fn(edge, senders, receivers, globals_):
    t_eval = edges["times"]
    x_init = senders
    # print("Are these the edges?", t_eval.shape, x_init.shape, edge)
    edges["messages"]  = osc_vmapped(osc_eqx_p, osc_eqx_s, x_init, t_eval)[:, -1]
    return edge

def aggregate_edges_for_nodes_fn(edge, sent_or_recv_id, tot_n_nodes):
    """ This function is ran twice, once for senders and once for receivers ids. """
    return edge


def update_node_fn(node, agg_senders, agg_receivers, globals_):
    """agg_senders and agg_receivers are the same, since aggregate_edges_for_nodes_fn doesn't use sent_or_recv_id """
    # print("Just checking: ", agg_senders, agg_receivers, node.shape)
    return agg_senders["messages"]

# def update_global_fn(aggregated_node_features, aggregated_edge_features, globals_):
#     return globals_

net = jraph.GraphNetwork(update_edge_fn=update_edge_fn,
                         update_node_fn=update_node_fn,
                         update_global_fn=None,
                         aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn)


for _ in range(40):
    graph = net(graph)

    ## Reset the initial condition TODO Doesn't wor !!!
    nodes = graph.nodes
    graph = graph._replace(nodes=nodes.at[0,:].set(x0_s[0,:]))

# graph
# print(graph.edges["messages"])
# print(graph.nodes)

position = graph.nodes[:, 0]
velocity = graph.nodes[:, 1]
t = jnp.linspace(t0, tf, N+1)[1:]           ### WTF??? Who is x0 in this whole thing !!!

# Create a plot to visualize the damped harmonic oscillator behavior
plt.figure(figsize=(10, 6))
plt.plot(t, position, label="Position (x)")
plt.plot(t, velocity, label="Velocity (dx/dt)")
plt.title("Damped Harmonic Oscillator - GraphPinT")
plt.xlabel("Time")
plt.legend()
plt.grid()
plt.show()

# %%




# %% [markdown]

## Specifics of this original version of GraphPinT
# - This focuses on time-parallelisation (vmapping along edges)
# - The edge features are the integrators
# - The node features are the initial conditions

# The next version of GraphPinT will focus on generalising dysnmical systems to new systems. We treat a each of those systems as a node, and the edges carry the difference in weight norms between them.

# %%
