## Just a bunch of utility functions
import time
import jax
import jax.numpy as jnp
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(context='notebook', style='ticks',
#         font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 2})
# plt.style.use("dark_background")
import os

sns.set_theme(context="notebook", style="ticks",
        font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 2})
# plt.style.use("dark_background")
# Set the following parameters for scientfic plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
mpl.rcParams['savefig.facecolor'] = 'w'


import equinox as eqx

def seconds_to_hours(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return hours, minutes, seconds

## Simply returns a suitable key for all jax operations
def get_new_key(key=None, num=1):
    if key is None:
        print("WARNING: No key provided, using time as seed")
        key = jax.random.PRNGKey(time.time_ns())

    elif isinstance(key, int):
        key = jax.random.PRNGKey(key)

    keys = jax.random.split(key, num=num)

    return keys if num > 1 else keys[0]

def generate_new_keys(key=None, num=1):
    if key is None:
        print("WARNING: No key provided, using time as seed")
        key = jax.random.PRNGKey(time.time_ns())

    elif isinstance(key, int):
        key = jax.random.PRNGKey(key)

    return jax.random.split(key, num=num)

## Wrapper function for matplotlib and seaborn
def sbplot(*args, ax=None, figsize=(6,3.5), x_label=None, y_label=None, title=None, x_scale='linear', y_scale='linear', xlim=None, ylim=None, context="notebook", style="ticks", dark_background=False, **kwargs):
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # sns.set_theme(context=context, style=style,
    #         font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 2})
    # if dark_background:
    #     plt.style.use("dark_background")

    if ax==None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    # sns.despine(ax=ax)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.plot(*args, **kwargs)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    if "label" in kwargs.keys():
        ax.legend()
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    plt.tight_layout()
    return ax

## Alias for sbplot
def plot(*args, ax=None, figsize=(6,3.5), x_label=None, y_label=None, title=None, x_scale='linear', y_scale='linear', xlim=None, ylim=None, **kwargs):
  return sbplot(*args, ax=ax, figsize=figsize, x_label=x_label, y_label=y_scale, title=title, x_scale=x_scale, y_scale=y_scale, xlim=xlim, ylim=ylim, **kwargs)


def pvplot(x, y, show=True, xlabel=None, ylabel=None, title=None, ax=None, **kwargs):
    import pyvista as pv
    # pv.start_xvfb()           ## TODO Only do this on LINUX

    if ax is None:
        ax = pv.Chart2D()

    _ = ax.line(x, y, **kwargs)

    if xlabel is not None:
        ax.x_label = xlabel

    if ylabel is not None:
        ax.y_label = ylabel

    if title is not None:
        ax.title = title

    if show == True:
        ax.show()

    return ax


def flatten_pytree(pytree):
    """ Flatten the leaves of a pytree into a single array. Return the array, the shapes of the leaves and the tree_def. """

    leaves, tree_def = jax.tree_util.tree_flatten(pytree)
    flat = jnp.concatenate([x.flatten() for x in leaves])
    shapes = [x.shape for x in leaves]
    return flat, shapes, tree_def

def unflatten_pytree(flat, shapes, tree_def):
    """ Reconstructs a pytree given its leaves flattened, their shapes, and the treedef. """

    leaves_prod = [0]+[np.prod(x) for x in shapes]

    lpcum = np.cumsum(leaves_prod)
    leaves = [flat[lpcum[i-1]:lpcum[i]].reshape(shapes[i-1]) for i in range(1, len(lpcum))]

    return jax.tree_util.tree_unflatten(tree_def, leaves)



## Function to calculate losses
def params_norm(params):
    """ norm of the parameters """
    return jnp.array([jnp.linalg.norm(x) for x in jax.tree_util.tree_leaves(params)]).sum()

def params_diff_norm(params1, params2):
    """ norm of the parameters difference"""
    params1 = eqx.filter(params1, eqx.is_array, replace=jnp.zeros(1))
    params2 = eqx.filter(params2, eqx.is_array, replace=jnp.zeros(1))

    # diff_tree = jax.tree_util.tree_map(lambda x, y: x-y, params1, params2)
    # return params_norm(diff_tree)

    # return jnp.array([jnp.linalg.norm(x-y) for x, y in zip(jax.tree_util.tree_leaves(params1), jax.tree_util.tree_leaves(params2))]).sum()

    ## Flatten the difference and calculate the norm
    diff_flat, _, _ = flatten_pytree(jax.tree_util.tree_map(lambda x, y: x-y, params1, params2))
    return jnp.linalg.norm(diff_flat)

@eqx.filter_jit
def params_diff_norm_squared(params1, params2):
    """ normalised squared norm of the parameters difference """
    params1 = eqx.filter(params1, eqx.is_array, replace=jnp.zeros(1))
    params2 = eqx.filter(params2, eqx.is_array, replace=jnp.zeros(1))
    diff_flat, _, _ = flatten_pytree(jax.tree_util.tree_map(lambda x, y: x-y, params1, params2))
    return (diff_flat.T@diff_flat) / diff_flat.shape[0]

@eqx.filter_jit
def params_norm_squared(params):
    """ normalised squared norm of the parameter """
    params = eqx.filter(params, eqx.is_array, replace=jnp.zeros(1))
    diff_flat, _, _ = flatten_pytree(params)
    return (diff_flat.T@diff_flat) / diff_flat.shape[0]



def spectral_norm(params):
    """ spectral norm of the parameters """
    return jnp.array([jnp.linalg.svd(x, compute_uv=False)[0] for x in jax.tree_util.tree_leaves(params) if jnp.ndim(x)==2]).sum()

def spectral_norm_estimation(model, nb_iters=5, *, key=None):
    """ estimating the spectral norm with the power iteration: https://arxiv.org/abs/1802.05957 """
    params = eqx.filter(model, eqx.is_array)
    matrices = [x for x in jax.tree_util.tree_leaves(params) if jnp.ndim(x)==2]
    nb_matrices = len(matrices)
    keys = generate_new_keys(key, num=nb_matrices)
    us = [jax.random.normal(k, (x.shape[0],)) for k, x in zip(keys, matrices)]
    vs = [jax.random.normal(k, (x.shape[1],)) for k, x in zip(keys, matrices)]

    for _ in range(nb_iters):
        for i in range(nb_matrices):
            vs[i] = matrices[i].T@us[i]
            vs[i] = vs[i] / jnp.linalg.norm(vs[i])
            us[i] = matrices[i]@vs[i]
            us[i] = us[i] / jnp.linalg.norm(us[i])

    sigmas = [u.T@x@v for x, u, v in zip(matrices, us, vs)]
    return jnp.array(sigmas).sum()

def infinity_norm_estimation(model, xs, ctx):
    xs_flat = jnp.reshape(xs, (-1, xs.shape[-1]))
    ys = jax.vmap(model, in_axes=(None, 0, None))(None, xs_flat, ctx)
    return jnp.mean(jnp.linalg.norm(ys, axis=-1) / jnp.linalg.norm(xs_flat, axis=-1))

def l2_norm_traj(xs, xs_hat):
    total_loss = jnp.mean((xs - xs_hat)**2, axis=-1)   ## TODO mean or sum ? Norm of d-dimensional vectors
    return jnp.sum(total_loss) / (xs.shape[-2] * xs.shape[-3])


def make_run_folder(parent_path='./runs/'):
    """ Create a new folder for the run. """
    if not os.path.exists(parent_path):
        os.mkdir(parent_path)

    # run_folder = parent_path+time.strftime("%y%m%d-%H%M%S")+'/'
    run_folder = os.path.join(parent_path, time.strftime("%y%m%d-%H%M%S")+'/')
    if not os.path.exists(run_folder):
        os.mkdir(run_folder)
        print("Created a new run folder at:", run_folder)

    return run_folder


def setup_run_folder(folder_path, script_name):
    """ Copy the run script, the module files, and create a folder for the adaptation results. """

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        print("Created a new run folder at:", folder_path)

    # Save the run scripts in that folder
    os.system(f"cp {script_name} {folder_path}")

    # Save the selfmod module files as well
    # module_folder = os.path.join(os.path.dirname(__file__), "../")
    module_folder = os.path.join(os.path.dirname(__file__))
    os.system(f"cp -r {module_folder} {folder_path}")
    print("Completed copied scripts ")

    return None