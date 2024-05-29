

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
import equinox as eqx
import diffrax
import optax
import time
import argparse

from neuralhub import params_norm, sbplot

try:
    __IPYTHON__
    _in_ipython_session = True
except NameError:
    _in_ipython_session = False


#%%

if _in_ipython_session:
	args = argparse.Namespace(time_horizon='5.00', savepath="results_sgd_7/99999.npz", verbose=1)
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

## Training hps
print_every = 300
nb_epochs = 1200
init_lr_mod = 1e-1
init_lr_T = 10.

traj_length = 25
skip_steps = 1

T_full = 100        ### All trajectories are this long, just in case !


#%%

def mass_spring_damper(t, state, k, mu, m):
    E = np.array([[0, 1], [-k/m, -mu/m]])
    return E @ state

p = {"k": 1, "mu": 0.25, "m": 1}
t_eval_full = np.linspace(0, T_full, (traj_length-1)*100+1)[::skip_steps]  ## Biggest t we can find !
initial_state = [1.0, 1.0]

solution = solve_ivp(mass_spring_damper, (0,T_full), initial_state, args=p.values(), t_eval=t_eval_full)
data = solution.y.T[None, None, ::, :]

# ## Test 0: Plot the data
# sbplot(t_eval_full, data[0, 0, :, 0], title="Mass-Spring-Damper: Position", x_label="Time", y_label="Position");

# ## Test 1: Plot the interpolation as well
# t_eval_small = np.linspace(0, T, traj_length)
# sol_int = np.interp(t_eval_small, t_eval_full, data[0, 0, :, 0])
# ax = sbplot(t_eval_small, sol_int, "-", title="Interpolation vs Renormalisation", x_label="Time", y_label="Position", label="Interpolated");

# ## Test 2: Integrate a renomalized vector field
# def mass_spring_damper_renormed(t, state, k, mu, m):
#     E = np.array([[0, 1], [-k/m, -mu/m]])
#     return T * (E @ state)
# t_renormed = np.linspace(0, 1, traj_length)
# sol_renormed = solve_ivp(mass_spring_damper_renormed, (0,1), initial_state, args=p.values(), t_eval=t_renormed)
# ax = sbplot(t_eval_small, sol_renormed.y[0], "x", label="Renormalized", y_label="Position", ax=ax);


# %%

class Processor(eqx.Module):
    matrix: jnp.ndarray

    def __init__(self, in_size, out_size, key=None):
        self.matrix = jnp.array([[0., 0.], [0., 0.]])

    def __call__(self, t, x, args):
        T = args[0]
        return T * (self.matrix @ x)


class NeuralODE(eqx.Module):
    data_size: int
    vector_field: eqx.Module

    def __init__(self, data_size, key=None):
        self.data_size = data_size
        self.vector_field = Processor(data_size, data_size, key=key)

    def __call__(self, x0s, T):
        t_eval = jnp.linspace(0,1,traj_length)

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
                    adjoint=diffrax.BacksolveAdjoint(),
                    max_steps=4096
                )
            return sol.ys, sol.stats["num_steps"]

        trajs, nb_fes = eqx.filter_vmap(integrate)(x0s)
        return trajs, jnp.sum(nb_fes)


def model_real(X_full, t_full, T):
    t_eval = jnp.linspace(0, T, traj_length)

    ## Interpolate X_full to t_eval
    X1 = jnp.interp(t_eval, t_full, X_full[0, ...,0])
    X2 = jnp.interp(t_eval, t_full, X_full[0, ...,1])

    return jnp.concatenate([X1, X2], axis=-1)[None,...], t_eval







class THorizon(eqx.Module):
    T: jnp.ndarray

    def __init__(self, init_T, key=None):
        self.T = jnp.array([init_T])

    def __call__(self):
        return self.T


model = NeuralODE(data_size=2)
T_hrz = THorizon(T)


# model_real(jnp.array([1., 1.])[None, ...], T)
# print(data[0, :, 0].shape)
# model_real(data[:, :, 0], T)


# %%

def loss_fn(model, T_hrz, batch):
    X_full, t_full = batch
    X, _ = model_real(X_full, t_full, T_hrz.T)
    X_hat, _ = model(X_full[...,0,:], T_hrz.T)
    return jnp.mean((X - X_hat)**2)

@eqx.filter_jit
def train_step_min(model, T_hrz, batch, opt_state):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, T_hrz, batch)

    updates, opt_state = opt_mod.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    grad_norm = params_norm(grads)

    return model, opt_state, loss, grad_norm


@eqx.filter_jit
def train_step_max(model, T_hrz, batch, opt_state):

    def new_loss_fn(T_hrz, model, batch):
        return -loss_fn(model, T_hrz, batch)

    loss, grads = eqx.filter_value_and_grad(new_loss_fn)(T_hrz, model, batch)

    updates, opt_state = opt_T.update(grads, opt_state)
    T_hrz = eqx.apply_updates(T_hrz, updates)

    grad_norm = params_norm(grads)

    return T_hrz, opt_state, loss, grad_norm


opt_mod = optax.sgd(init_lr_mod)
opt_state_mod = opt_mod.init(eqx.filter(model, eqx.is_array))

opt_T = optax.sgd(init_lr_T)
opt_state_T = opt_T.init(T_hrz)


if verbose:
    print("\n\n=== Beginning Training ... ===")

start_time = time.time()


losses = []
grad_norms = []
theta_list = [model.vector_field.matrix]
T_hrz_list = [T_hrz.T]

batch = data[0, :, :, :], t_eval_full

for epoch in range(nb_epochs):

    model, opt_state_mod, loss, grad_norm = train_step_min(model, T_hrz, batch, opt_state_mod)

    T_hrz, opt_state_T, loss_T, _ = train_step_max(model, T_hrz, batch, opt_state_T)

    losses.append(loss)
    theta_list.append(model.vector_field.matrix)
    grad_norms.append(grad_norm)
    T_hrz_list.append(T_hrz.T)

    if verbose and (epoch%print_every==0 or epoch==nb_epochs-1):
        print(f"    Epoch: {epoch:-5d}      Loss: {loss:.12f}", flush=True)

losses = jnp.stack(losses)
thetas = jnp.stack(theta_list)
grad_norms = jnp.stack(grad_norms)
T_hrz_list = jnp.stack(T_hrz_list)

wall_time = time.time() - start_time

if verbose:
    print("\nTotal GD training time: %d secs\n" %wall_time, flush=True)
    print("Final model matrix: \n", model.vector_field.matrix)
    print("Final T: ", T_hrz.T)

# %%

## Save results: T, traj_length, losses, thetas, wall_time into a .npz file
np.savez(savepath, time_horizon_init=T, time_horizon_list=T_hrz_list, traj_length=traj_length, losses=losses, thetas=thetas, wall_time=wall_time, grad_norms=grad_norms)


# %%

# ## Test 3: Plot to training predictions
# X_full, t_full = batch
# t_eval_small = np.linspace(0, T_hrz.T, traj_length)

# X, _ = model_real(X_full, t_full, T_hrz.T)
# X_hat, _ = model(X_full[...,0,:], T_hrz.T)

# ax = sbplot(t_eval_small, X_hat[0, :, 0], "-", label="Predicted", title="Real vs Predicted", x_label="Time", y_label="Position")
# ax = sbplot(t_eval_small, X[0, :, 0], "x", label="Real", ax=ax)
