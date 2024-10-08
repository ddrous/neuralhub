
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
# config.update("jax_debug_nans", True)

import jax.numpy as jnp

import numpy as np
from scipy.integrate import solve_ivp

import equinox as eqx
import diffrax

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from neuralhub.utils import *
from neuralhub.integrators import *

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
init_lr = 1e-2
decay_rate = 0.9

## Training hps
print_every = 100
nb_epochs = 2000
# batch_size = 128*10
batch_size = 1
skip_steps = 1

T = 25
print("Time horizon:", T)

#%%

def mass_spring_damper(t, state, k, mu, m):
    E = np.array([[0, 1], [-k/m, -mu/m]])
    return E @ state

p = {"k": 1, "mu": 0.25, "m": 1}
t_eval = np.linspace(0, T, 101)[::skip_steps]
initial_state = [1.0, 1.0]

solution = solve_ivp(mass_spring_damper, (0,T), initial_state, args=p.values(), t_eval=t_eval)

sbplot(solution.t, solution.y.T, x_label='Time', y_label='State', title='Mass-Spring-Damper System')

#%%

data = solution.y.T[None, None, ::, :]
print("data shape (first two are superfluous):", data.shape)

# %%

class Processor(eqx.Module):
    # layers: list
    # physics: jnp.ndarray
    matrix: jnp.ndarray
    # k_mu: jnp.ndarray

    def __init__(self, in_size, out_size, key=None):
        keys = get_new_key(key, num=3)
        # self.layers = [eqx.nn.Linear(in_size, 8, key=keys[0]), jax.nn.softplus,
        #                 eqx.nn.Linear(8, 8, key=keys[1]), jax.nn.softplus,
        #                 eqx.nn.Linear(8, out_size, key=keys[2]) ]

        self.matrix = jnp.array([[0., 0.], [0., 0.]])
        # self.k_mu = jnp.array([0., 0.])

        # self.matrix = jnp.zeros((3,))

    def __call__(self, t, x, args):

        # ## Neural Net contribution
        # y = x
        # for layer in self.layers:
        #     y = layer(y)
        # return y

        # k, mu = self.k_mu
        # matrix = jnp.array([[0., 1.], [-k, -mu]])

        return self.matrix @ x


        # matrix = jnp.array([[0., self.matrix[0,1]], [self.matrix[1,0], self.matrix[1,1]]])
        # matrix = self.matrix.at[0,0].set(0)
        # matrix = jnp.concatenate([jnp.array([0.]), self.matrix.flatten()[1:]]).reshape((2,2))

        # matrix = jnp.concatenate([jnp.array([0.]), self.matrix]).reshape((2,2))
        # return matrix @ x


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
    # print('\nCompiling function "loss_fn" ...\n')
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
# sched = optax.piecewise_constant_schedule(init_value=init_lr,
#                                             boundaries_and_scales={int(total_steps*0.25):0.5, 
#                                                                     int(total_steps*0.5):0.2,
#                                                                     int(total_steps*0.75):0.5})
sched = init_lr*1e1

start_time = time.time()


print(f"\n\n=== Beginning Training ... ===")

# opt = optax.adam(sched)
opt = optax.sgd(sched)

# params, static  = eqx.partition(model, eqx.is_array)
# opt_state = opt.init(params)
opt_state = opt.init(eqx.filter(model, eqx.is_array))

losses = []
theta_list = [model.vector_field.matrix]

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
    theta_list.append(model.vector_field.matrix)

    if epoch%print_every==0 or epoch==nb_epochs-1:
        print(f"    Epoch: {epoch:-5d}      Loss: {loss_epoch:.8f}", flush=True)

losses = np.stack(losses)

wall_time = time.time() - start_time
time_in_hmsecs = seconds_to_hours(wall_time)
print("\nTotal GD training time: %d hours %d mins %d secs" %time_in_hmsecs)


# %%


fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))

ax = sbplot(losses, ".", x_label='Epoch', y_label='L2', y_scale="log", label='Loss', ax=ax);

# Make a twin axis for the diff loss
ax2 = ax.twinx()

# Calculate the time(epoch)-derivitative of the loss
diff_lim = 1e-3
diff_losses = np.clip(np.gradient(losses), -diff_lim, diff_lim)
ax2 = sbplot(diff_losses, "r", label='Diff Loss', ax=ax2);

# plt.draw()

plt.savefig(f"data/mass/loss_{SEED:05d}.png", dpi=300, bbox_inches='tight')
# plt.show()
plt.legend()
fig.canvas.draw()
fig.canvas.flush_events()



## Save the losses
np.save("data/mass/losses_{SEED:05d}.npy", np.array(losses))


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


ax = sbplot(t, X_hat[:,:], x_label='Time', label=[f'Displacement', f'Velocity'], title=f'Trajectories, {i}')
ax = sbplot(t, X[:,:], "+", color="grey", x_label='Time', title=f'Trajectories, {i}', ax=ax)

# plt.savefig(f"data/coda_test_env{e}_traj{i}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"data/mass/trajs_{SEED:05d}.png", dpi=300, bbox_inches='tight')


#%% 

eqx.tree_serialise_leaves(f"data/mass/model_{SEED:05d}.eqx", model)
# model = eqx.tree_deserialise_leaves("data/model_01.eqx", model)


## Print original matrix
print("GT matrix:")
k,mu,m = p.values()
E = np.array([[0, 1], [-k/m, -mu/m]])
print(E)

print()
## Print learned matrix
print("Learned matrix:")
print(model.vector_field.matrix)


## Plot the loss landscape
orig_model = eqx.partition(model, eqx.is_array)[0]  ## Just a copy
theta_ids = (1, 2)       ## Indices of the parameters after flatening to plot


thetas = jnp.array([x.flatten()[jnp.array(theta_ids)] for x in theta_list])


## Sample matrices for the contourf plot
t0_min, t0_max = -2, 1.5
t1_min, t1_max = -2, 1.5
theta_0s = jnp.linspace(t0_min, t0_max, 200)
theta_1s = jnp.linspace(t1_min, t1_max, 200)
theta_0s, theta_1s = jnp.meshgrid(theta_0s, theta_1s)
theta_01s = jnp.stack([theta_0s.flatten(), theta_1s.flatten()], axis=-1)

# @eqx.filter_vmap
@eqx.filter_vmap
def eval_loss_fn(theta_01):
    orig_matrix = orig_model.vector_field.matrix.flatten()      ## TODO the original params are non-changing !
    new_matrix = orig_matrix.at[jnp.array(theta_ids)].set(theta_01)

    new_model = eqx.tree_at(lambda m: m.vector_field.matrix, orig_model, new_matrix.reshape((2,2)))

    return loss_fn(new_model, batch)

loss_evals = eval_loss_fn(theta_01s)


## %%
## Plot the loss landscape with contour
fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))

# theta_01s and loss_evals are both of shape (10000, 2)
# loss_evals.shape

## Set the color values to log scale
Z = loss_evals.reshape(theta_1s.shape)

# min_plot = Z.min() if not jnp.isnan(Z.min()) else 1e-6
# max_plot = Z.max() if not jnp.isnan(Z.max()) else 1.0

## Replace all NaNs with 1e-6
Z = jnp.nan_to_num(Z, nan=1e-6)

pcm = ax.pcolormesh(theta_0s, theta_1s, Z, norm=mcolors.LogNorm(vmin=Z.min(), vmax=Z.max()), cmap='nipy_spectral')
fig.colorbar(pcm, ax=ax, extend='both', label='MSE (log scale)')

# # Define logarithmic levels
# log_levels = np.logspace(np.log10(Z.min()), np.log10(Z.max()), num=100)
# contour = ax.contourf(theta_0s, theta_1s, Z, levels=log_levels, norm=mcolors.LogNorm())
# cbar = fig.colorbar(contour, ax=ax, extend='both', label='Loss (log scale)')


ax.set_xlabel(f"Theta {theta_ids[0]}")
ax.set_ylabel(f"Theta {theta_ids[1]}")

## Place thetas with crosses on the plot, every 10th
thetas = np.array(thetas)
thetas[:,0] = np.clip(thetas[:,0], t0_min, t0_max)
thetas[:,1] = np.clip(thetas[:,1], t1_min, t1_max)
skip_iter = 10
ax.scatter(thetas[::skip_iter, 0], thetas[::skip_iter, 1], marker="x", color="white", label="SGD")
ax.legend()


# %% [markdown]

# # Conclusion
# 
# 


# %%

## 2D imshow plot with the loss againts the epochs and time horizon T

epochs = np.arange(nb_epochs+1)
Ts = [T-10, T-5, T]
print(Ts)
# losses_imshow = np.tile(losses, (3,1))
losses_imshow = jnp.stack([losses*10, losses*2, losses], axis=0)

fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))
pcm = ax.imshow(losses_imshow, aspect='auto', cmap='coolwarm', interpolation='none', origin='lower', norm=mcolors.LogNorm(vmin=losses_imshow.min(), vmax=losses_imshow.max()))

## Add colorbar
cbar = fig.colorbar(pcm, ax=ax, extend='both', label='MSE')

ax.set_xlabel('Epoch')
ax.set_ylabel('Time horizon T')
ax.set_title('Loss Evolution With Time Horizon T')

## Set x ticks and labels to the epochs
ax.set_xticks(np.arange(0, nb_epochs+1, 500))
ax.set_xticklabels(epochs[::500])

## Set y ticks and labels to the time horizon
ax.set_yticks(np.arange(3))
ax.set_yticklabels(Ts)

plt.savefig(f"data/mass/loss_imshow_{SEED:05d}.png", dpi=300, bbox_inches='tight')
