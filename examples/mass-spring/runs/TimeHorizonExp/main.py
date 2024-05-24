

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
import equinox as eqx
import diffrax
import optax
import time
import argparse

try:
    __IPYTHON__
    _in_ipython_session = True
except NameError:
    _in_ipython_session = False


#%%

if _in_ipython_session:
	args = argparse.Namespace(time_horizon='10.00', savepath="results/99999.npz", verbose=1)
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
print_every = 500
nb_epochs = 2000
batch_size = 1
init_lr = 1e-1

traj_length = 101
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
        return self.matrix @ x


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

        trajs, nb_fes = eqx.filter_vmap(integrate)(x0s)
        return trajs, jnp.sum(nb_fes)

model = NeuralODE(data_size=2)


# %%

def loss_fn(model, batch):
    X, t = batch
    X_hat, _ = model(X[:, 0, :], t)
    return jnp.mean((X - X_hat)**2)

@eqx.filter_jit
def train_step(model, batch, opt_state):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)

    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


opt = optax.sgd(init_lr)
opt_state = opt.init(eqx.filter(model, eqx.is_array))


if verbose:
    print("\n\n=== Beginning Training ... ===")

start_time = time.time()


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

    if verbose and (epoch%print_every==0 or epoch==nb_epochs-1):
        print(f"    Epoch: {epoch:-5d}      Loss: {loss_epoch:.8f}", flush=True)

losses = jnp.stack(losses)
thetas = jnp.stack(theta_list)

wall_time = time.time() - start_time

if verbose:
    print("\nTotal GD training time: %d secs\n" %wall_time, flush=True)


# %%

## Save results: T, traj_length, losses, thetas, wall_time into a .npz file
np.savez(savepath, time_horizon=T, traj_length=traj_length, losses=losses, thetas=thetas, wall_time=wall_time)
