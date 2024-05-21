
#%%[markdown]
# # Simple Neural ODE framework for the Mass-Spring-Damper system


#%%
%load_ext autoreload
%autoreload 2

import jax

print("\n############# Lotka-Volterra with Neural ODE #############\n")
print("Available devices:", jax.devices())

from jax import config
##  Debug nans
config.update("jax_debug_nans", True)

import jax.numpy as jnp

import numpy as np
from scipy.integrate import solve_ivp

import equinox as eqx
import diffrax

import matplotlib.pyplot as plt

from graphpint.utils import *
from graphpint.integrators import *

import optax
from functools import partial
import time

#%%

# SEED = 27
SEED = time.time_ns() % 2**15
print(f"Seed: {SEED}")

## Integrator hps
integrator = rk4_integrator
# integrator = dopri_integrator

## Optimiser hps
init_lr = 1e-3
decay_rate = 0.9

## Training hps
print_every = 1000
nb_epochs = 10000
# batch_size = 128*10
batch_size = 1
skip_steps = 10

#%%

def mass_spring_damper(t, state, k, mu, m):
    E = np.array([[0, 1], [-k/m, -mu/m]])
    return E @ state

p = {"k": 1, "mu": 0.25, "m": 1}
t_eval = np.linspace(0, 10, 1001)[::skip_steps]
initial_state = [1.0, 1.0]

solution = solve_ivp(mass_spring_damper, (0,10), initial_state, args=p.values(), t_eval=t_eval)

sbplot(solution.t, solution.y.T, x_label='Time', y_label='State', title='Mass-Spring-Damper System')

#%%

data = solution.y.T[None, None, ::, :]
print("data shape (first two are superfluous):", data.shape)

# %%

class Processor(eqx.Module):
    layers: list
    # physics: jnp.ndarray

    def __init__(self, in_size, out_size, key=None):
        keys = get_new_key(key, num=3)
        # self.layers = [eqx.nn.Linear(in_size, 10, key=keys[0]), jax.nn.tanh,
        self.layers = [eqx.nn.Linear(in_size, 8, key=keys[0]), jax.nn.softplus,
                        eqx.nn.Linear(8, 8, key=keys[1]), jax.nn.softplus,
                        eqx.nn.Linear(8, out_size, key=keys[2]) ]

        # self.physics = jnp.array([1.5, 1.0, 3.0, 1.0])
        # self.physics = jnp.abs(jax.random.normal(keys[1], (4,)))

    def __call__(self, t, x, args):

        ## Physics contribution
        # dx0 = x[0]*self.physics[0] - x[0]*x[1]*self.physics[1]
        # dx1 = x[0]*x[1]*self.physics[2] - x[1]*self.physics[3]

        # ## Neural Net contribution
        # # y = jnp.concatenate([jnp.broadcast_to(t, (1,)), x], axis=0)
        y = x
        for layer in self.layers:
            y = layer(y)
        return y
        # return jnp.array([dx0, dx1])

class NeuralODE(eqx.Module):
    data_size: int
    vector_field: eqx.Module

    def __init__(self, data_size, key=None):
        self.data_size = data_size
        self.vector_field = Processor(data_size, data_size, key=key)

    def __call__(self, x0s, t_eval):

        def integrate(y0):
            sol = diffrax.diffeqsolve(
                    diffrax.ODETerm(self.vector_field),
                    diffrax.Tsit5(),
                    t0=t_eval[0],
                    t1=t_eval[-1],
                    dt0=1e-3,
                    y0=y0,
                    stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
                    saveat=diffrax.SaveAt(ts=t_eval),
                    # adjoint=diffrax.RecursiveCheckpointAdjoint(),
                    max_steps=4096
                )
            return sol.ys, sol.stats["num_steps"]

            # sol = RK4(self.vector_field, 
            #           (t_eval[0], t_eval[-1]), 
            #           y0, 
            #           (coeffs.lambdas, coeffs.gammas), 
            #           t_eval=t_eval, 
            #           subdivisions=4)
            # return sol, len(t_eval)*4

        trajs, nb_fes = eqx.filter_vmap(integrate)(x0s)
        # trajs, nb_fes = integrate(x0s)
        return trajs, jnp.sum(nb_fes)



# %%

model_keys = get_new_key(SEED, num=2)
model = NeuralODE(data_size=2, key=model_keys[0])









# %%


# def params_norm(params):
#     return jnp.array([jnp.sum(jnp.abs(x)) for x in jax.tree_util.tree_leaves(params)]).sum()

# def l2_norm(X, X_hat):
#     total_loss = jnp.mean((X - X_hat)**2, axis=-1)   ## Norm of d-dimensional vectors
#     return jnp.sum(total_loss) / (X.shape[-2])

# %%

### ==== Vanilla Gradient Descent optimisation ==== ####

def loss_fn(model, batch):
    print('\nCompiling function "loss_fn" ...\n')
    X, t = batch

    X_hat, _ = model(X[:, 0, :], t)

    # return jnp.mean((X[...,-1] - X_hat[...,-1])**2)
    return jnp.mean((X - X_hat)**2)


@eqx.filter_jit
def train_step(model, batch, opt_state):
    print('\nCompiling function "train_step" ...\n')

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)

    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


total_steps = nb_epochs

# sched = optax.exponential_decay(init_lr, total_steps, decay_rate)
# sched = optax.linear_schedule(init_lr, 0, total_steps, 0.25)
sched = optax.piecewise_constant_schedule(init_value=init_lr,
                                            boundaries_and_scales={int(total_steps*0.25):0.5, 
                                                                    int(total_steps*0.5):0.2,
                                                                    int(total_steps*0.75):0.5})
fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))

start_time = time.time()


print(f"\n\n=== Beginning Training ... ===")

opt = optax.adam(sched)

# params, static  = eqx.partition(model, eqx.is_array)
# opt_state = opt.init(params)
opt_state = opt.init(eqx.filter(model, eqx.is_array))

losses = []
for epoch in range(nb_epochs):

    nb_batches = 0
    loss_sum = 0.
    for i in range(0, data.shape[1], batch_size):
        batch = (data[0,i:i+batch_size,...], t_eval)
    
        model, opt_state, loss = train_step(model, batch, opt_state)

        loss_sum += loss
        nb_batches += 1

    loss_epoch = loss_sum/nb_batches
    losses.append(loss_epoch)

    if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
        print(f"    Epoch: {epoch:-5d}      Loss: {loss_epoch:.8f}", flush=True)


wall_time = time.time() - start_time
time_in_hmsecs = seconds_to_hours(wall_time)
print("\nTotal GD training time: %d hours %d mins %d secs" %time_in_hmsecs)



# ax = sbplot(losses, x_label='Epoch', y_label='L2', y_scale="log", title=f'Loss for environment {e}', ax=ax);
ax = sbplot(losses, x_label='Epoch', y_label='L2', y_scale="log", title='Losses', ax=ax);
plt.savefig(f"data/nodes/loss_{SEED:05d}.png", dpi=300, bbox_inches='tight')
# plt.show()
plt.legend()
fig.canvas.draw()
fig.canvas.flush_events()



## Save the losses
np.save("data/nodes/losses_{SEED:05d}.npy", np.array(losses))


# %%
def test_model(model, batch):
    X0, t = batch
    X_hat, _ = model(X0, t)
    return X_hat


i = np.random.randint(0, 1)

X = data[0, i:i+1, :, :]
t = t_eval


X_hat = test_model(model, (X[:, 0, :], t))

X= X[i, :, :]
X_hat = X_hat[i, :, :]


# ax = sbplot(X_hat[:,0], X_hat[:,1], x_label='Preys', y_label='Predators', label=f'Pred', title=f'Phase space, traj {i}')
# ax = sbplot(X[:,0], X[:,1], "--", lw=1, label=f'True', ax=ax)


ax = sbplot(t, X_hat[:,:], x_label='Time', label=[f'Preys', f'Predatoes'], title=f'Trajectories, {i}')
ax = sbplot(t, X[:,:], "+", color="grey", x_label='Time', title=f'Trajectories, {i}', ax=ax)

# plt.savefig(f"data/coda_test_env{e}_traj{i}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"data/nodes/trajs_{SEED:05d}.png", dpi=300, bbox_inches='tight')


#%% 

eqx.tree_serialise_leaves(f"data/nodes/model_{SEED:05d}.eqx", model)
# model = eqx.tree_deserialise_leaves("data/model_01.eqx", model)

# %% [markdown]

# # Conclusion
# 
# 


# %%
