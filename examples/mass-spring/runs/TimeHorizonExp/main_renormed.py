

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
import equinox as eqx
import diffrax
import optax
import time
import argparse

from graphpint import params_norm

try:
    __IPYTHON__
    _in_ipython_session = True
except NameError:
    _in_ipython_session = False


#%%

if _in_ipython_session:
	args = argparse.Namespace(time_horizon='10.00', savepath="results_sgd_5/99999.npz", verbose=1)
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
    print("  - Time Horizon T: ", T)
    print("  - Savepath: ", savepath)

## Training hps
print_every = 100
nb_epochs = 1000
init_lr_mod = 1e-1
init_lr_T = 10.

traj_length = 11
skip_steps = 1

#%%

def mass_spring_damper(t, state, k, mu, m):
    E = np.array([[0, 1], [-k/m, -mu/m]])
    return E @ state

p = {"k": 1, "mu": 0.25, "m": 1}
t_eval = np.linspace(0, T, traj_length)[::skip_steps]
initial_state = [1.0, 1.0]

solution = solve_ivp(mass_spring_damper, (0,T), initial_state, args=p.values(), t_eval=t_eval)
data = solution.y.T[None, None, ::, :]

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


def model_real(x0s, T):
    t_eval = jnp.linspace(0,1,traj_length)

    # jax.debug.print("T: {}", T)
    # print("T: ", T)

    k, mu, m = p.values()

    def integrate(y0):

        def real_vf(t, x, args):
            k, mu, m, T = args
            return T * (jnp.array([[0, 1], [-k/m, -mu/m]]) @ x)

        sol = diffrax.diffeqsolve(
                diffrax.ODETerm(real_vf),
                diffrax.Tsit5(),
                args=(k, mu, m, T),
                t0=t_eval[0],
                t1=t_eval[-1],
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
    # X, t = batch
    X0s = batch
    X, _ = model_real(X0s, T_hrz.T)
    X_hat, _ = model(X0s, T_hrz.T)
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

batch = data[0, :, 0]

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

# from graphpint import sbplot

# sbplot(grad_norms, title="Gradient Norms", y_scale="log") 