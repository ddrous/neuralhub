

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
import equinox as eqx
import diffrax
import optax
import time
import argparse
import numpy as np

from neuralhub import params_norm, sbplot, params_norm_squared

try:
    __IPYTHON__
    _in_ipython_session = True
except NameError:
    _in_ipython_session = False


#%%

if _in_ipython_session:
	# args = argparse.Namespace(time_horizon='39.396473', savepath="results_sgd_8/99999.npz", verbose=1)
	args = argparse.Namespace(time_horizon='40', savepath="./99999.npz", verbose=1)
else:
	parser = argparse.ArgumentParser(description='Description of your program')
	parser.add_argument('--time_horizon', type=str, help='Time Horizon T', default='10.00', required=False)
	parser.add_argument('--savepath', type=str, help='Save the results to', default='results/99999.npz', required=False)
	parser.add_argument('--verbose',type=int, help='Whether to print details or not ?', default=1, required=False)
	args = parser.parse_args()

T = float(args.time_horizon)
savepath = args.savepath
verbose = args.verbose

if verbose:
    print("\n############# Lotka-Volterra with Neural ODE #############\n")
    print("  - Initial time Horizon T: ", T)
    print("  - Savepath: ", savepath)


SEED = 2026

## Training hps
print_every = 10000
nb_epochs = 50000
init_lr_mod = 1e-3
init_lr_T = 1.

traj_length = 11
skip_steps = 1



#%%

train_raw = np.load("./train_data.npz")
train_data, t_eval_full = train_raw["X"], train_raw["t"]

test_raw = np.load("./test_data.npz")
test_data, _ = test_raw["X"], test_raw["t"]

T_ful = t_eval_full[-1]

## Shapes
print("Train data shape: ", train_data.shape)
print("Test data shape: ", test_data.shape)

# %%

class Processor(eqx.Module):

    layers: list
    # physics: jnp.ndarray

    def __init__(self, in_size, out_size, key=None):
        keys = jax.random.split(key, num=3)

        self.layers = [eqx.nn.Linear(in_size, 32, key=keys[0]), jax.nn.softplus,
                        eqx.nn.Linear(32, 32, key=keys[1]), jax.nn.softplus,
                        eqx.nn.Linear(32, out_size, key=keys[2]) ]


    def __call__(self, t, x, args):
        T = args[0]
        y = x
        for layer in self.layers:
            y = layer(y)
        # return T*y
        return y



class NeuralODE(eqx.Module):
    data_size: int
    vector_field: eqx.Module

    def __init__(self, data_size, key=None):
        self.data_size = data_size
        self.vector_field = Processor(data_size, data_size, key=key)

    def __call__(self, x0s, T):
        # t_eval = jnp.linspace(0,1,traj_length)
        t_eval = jnp.linspace(0, T[0], traj_length)

        def integrate(y0):
            sol = diffrax.diffeqsolve(
                    diffrax.ODETerm(self.vector_field),
                    diffrax.Tsit5(),
                    t0=t_eval[0],
                    t1=t_eval[-1],
                    args=(T,),
                    dt0=1e-3,
                    y0=y0,
                    stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
                    saveat=diffrax.SaveAt(ts=t_eval),
                    # adjoint=diffrax.BacksolveAdjoint(),
                    max_steps=4096*10
                )
            return sol.ys, sol.stats["num_steps"]

        trajs, nb_fes = eqx.filter_vmap(integrate)(x0s)
        return trajs, jnp.sum(nb_fes)


def model_real(X_full, t_full, T):
    t_eval = jnp.linspace(0, T, traj_length)

    def interpolate(x):
        x1, x2 = x[:,0], x[:,1]
        new_x1 = jnp.interp(t_eval, t_full, x1)
        new_x2 = jnp.interp(t_eval, t_full, x2)
        return jnp.concatenate([new_x1, new_x2], axis=-1)

    return jax.vmap(interpolate)(X_full), t_eval







class THorizon(eqx.Module):
    T: jnp.ndarray

    def __init__(self, init_T, key=None):
        self.T = jnp.array([init_T])

    def __call__(self):
        return self.T


model = NeuralODE(data_size=2, key=jax.random.PRNGKey(SEED))
T_hrz = THorizon(T)
mega_model = (model, T_hrz)


# %%

def loss_fn(mega_model, batch):
    model, T_hrz = mega_model
    X_full, t_full = batch
    X, _ = model_real(X_full, t_full, T_hrz.T)
    X_hat, _ = model(X_full[...,0,:], T_hrz.T)
    return jnp.mean((X - X_hat)**2)


@eqx.filter_jit
def mega_train_step(mega_model, batch, mega_opt_state):
    loss, mega_grads = eqx.filter_value_and_grad(loss_fn)(mega_model, batch)

    model, T_hrz = mega_model
    grads_mod, grads_T = mega_grads
    opt_state_mod, opt_state_T = mega_opt_state

    updates, opt_state_mod = opt_mod.update(grads_mod, opt_state_mod)
    model = eqx.apply_updates(model, updates)

    # grad_norm = params_norm(grads_mod)
    grad_norm = params_norm_squared(grads_mod)

    # ## Replace grads_T with -grads_T to maximize
    # grads_T = jax.tree.map(lambda x: -x, grads_T)
    # updates, opt_state_T = opt_T.update(grads_T, opt_state_T)
    # T_hrz = eqx.apply_updates(T_hrz, updates)

    return (model, T_hrz), (opt_state_mod, opt_state_T), loss, grad_norm


opt_mod = optax.sgd(init_lr_mod)
opt_state_mod = opt_mod.init(eqx.filter(model, eqx.is_array))

opt_T = optax.sgd(init_lr_T)
opt_state_T = opt_T.init(T_hrz)


if verbose:
    print("\n\n=== Beginning Training ... ===")

start_time = time.time()


losses = []
grad_norms = []
# theta_list = [model.vector_field.matrix]
T_hrz_list = [T_hrz.T]

batch = train_data[0, :, :, :], t_eval_full

for epoch in range(nb_epochs):

    mega_model, opt_states, loss, grad_norm = mega_train_step((model, T_hrz), batch, (opt_state_mod, opt_state_T))

    model, T_hrz = mega_model
    opt_state_mod, opt_state_T = opt_states

    losses.append(loss)
    # theta_list.append(model.vector_field.matrix)
    grad_norms.append(grad_norm)
    T_hrz_list.append(T_hrz.T)

    if verbose and (epoch%print_every==0 or epoch==nb_epochs-1):
        print(f"    Epoch: {epoch:-5d}      Loss: {loss:.12f}", flush=True)

losses = jnp.stack(losses)
# thetas = jnp.stack(theta_list)
grad_norms = jnp.stack(grad_norms)
T_hrz_list = jnp.stack(T_hrz_list)

wall_time = time.time() - start_time

if verbose:
    print("\nTotal GD training time: %d secs\n" %wall_time, flush=True)
    # print("Final model matrix: \n", model.vector_field.matrix)
    print("Final T: ", T_hrz.T)

# %%

## Save results: T, traj_length, losses, thetas, wall_time into a .npz file
np.savez(savepath, time_horizon_init=T, time_horizon_list=T_hrz_list, traj_length=traj_length, losses=losses, wall_time=wall_time, grad_norms=grad_norms)

sbplot(losses, "-", label="MSE", y_scale="log", title="Metrics", x_label="Epochs", y_label="Loss");

# %%

# ## Test 3: Plot to training predictions
# X_full, t_full = batch
# t_eval_small = np.linspace(0, T_hrz.T, traj_length)

# X, _ = model_real(X_full, t_full, T_hrz.T)
# X_hat, _ = model(X_full[...,0,:], T_hrz.T)

# ax = sbplot(t_eval_small, X_hat[0, :, 0], "-", label="Predicted", title="Real vs Predicted", x_label="Time", y_label="Position")
# ax = sbplot(t_eval_small, X[0, :, 0], "x", label="Real", ax=ax)


# %%

## Test the model by computing the MSE on the test data
test_batch = test_data[0, :, :, :], t_eval_full

## Plot a prediucted and true trajectory
X_full, t_full = test_batch
plot_stop = 50
t_eval_small = np.linspace(0, T_hrz.T, traj_length)
X_hat, _ = model(X_full[...,0,:], T_hrz.T)
ax = sbplot(t_full[:plot_stop], X_full[0, :plot_stop, 0], "x", label="True", title="Real vs Predicted", x_label="Time", y_label="Position")
sbplot(t_eval_small, X_hat[0, :, 0], "-", label="Predicted", ax=ax)

# test_loss = loss_fn(mega_model, test_batch)

# if verbose:
#     print("\n\n=== Testing the model on the test data ... ===")
#     print(f"    Test Loss: {test_loss:.12f}\n")

