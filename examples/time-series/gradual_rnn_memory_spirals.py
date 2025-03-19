#%%[markdown]
# This is an introductory example. We demonstrate what using Equinox normally looks like day-to-day.

# Here, we'll train an RNN to classify clockwise vs anticlockwise spirals.

# This example is available as a Jupyter notebook here.

#%%
import math

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax  # https://github.com/deepmind/optax
import matplotlib.pyplot as plt

jax.config.update("jax_debug_nans", True)

#%%[markdown]
# We begin by importing the usual libraries, setting up a very simple dataloader, and generating a toy dataset of spirals.


#%%
def dataloader(arrays, batch_size):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = np.arange(dataset_size)
    while True:
        perm = np.random.permutation(indices)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def get_data(dataset_size, seq_len=16, *, key):
    t = jnp.linspace(0, 2 * math.pi, seq_len)
    offset = jrandom.uniform(key, (dataset_size, 1), minval=0, maxval=2 * math.pi)
    x1 = jnp.sin(t + offset) / (1 + t)
    x2 = jnp.cos(t + offset) / (1 + t)
    y = jnp.ones((dataset_size, 1))

    half_dataset_size = dataset_size // 2
    x1 = x1.at[:half_dataset_size].multiply(-1)
    y = y.at[:half_dataset_size].set(0)
    x = jnp.stack([x1, x2], axis=-1)

    return x, y


## A gradual tanh function : y=0.5\cdot\left(1-\tanh\left(\frac{\left(x-n\right)}{0.1}\right)\right)
def gradual_tanh(x, n):
    return 0.5 * (1 - jnp.tanh((x - n) / 0.1))

## x is always jnp.arange(16), n is the step number
for i in range(0, 161, 16):
    plt.plot(gradual_tanh(jnp.arange(160), i), label=f"n={i}")

plt.legend()
plt.show()



#%%[markdown]
# Now for our model.

# Purely by way of example, we handle the final adding on of bias ourselves, rather than letting the linear layer do it. This is just so we can demonstrate how to use custom parameters in models.

class LinearCell(eqx.Module):
    """A simple linear cell to test linear SSMs, and growing memory activation."""
    A: jax.Array
    B: jax.Array
    hidden_size: int

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        key: jax.random.PRNGKey,
    ):
        # self.A = jrandom.normal(key, (hidden_size, hidden_size))
        # self.B = jrandom.normal(key, (hidden_size, input_size))

        self.hidden_size = hidden_size

        ## Initiali A as the identify, and B as zeros
        self.A = jnp.eye(hidden_size)
        self.B = jnp.zeros((hidden_size, input_size))

    # def __call__(self, inp, hidden, *, key=None):
    #     h_next = self.A@hidden + self.B@inp
    #     h_next = jax.nn.tanh(h_next)
    #     return h_next

    # def __call__(self, inp, hidden, old_inp, *, key=None):
    #     h_next = self.A@hidden + self.B@(inp - old_inp)
    #     # h_next = jax.nn.tanh(h_next)
    #     return h_next, inp

    def __call__(self, inp, hidden, n, *, key=None):
        h_next = self.A@hidden + self.B@inp
        # h_next = jax.nn.tanh(h_next)
        sharp_tanh = gradual_tanh(jnp.arange(self.hidden_size), n)
        return h_next * sharp_tanh



class NeuralNetCell(eqx.Module):
    """A simple linear cell to test linear SSMs, and growing memory activation."""
    A: eqx.Module
    B: eqx.Module
    hidden_size: int

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        key: jax.random.PRNGKey,
    ):
        # self.A = jrandom.normal(key, (hidden_size, hidden_size))
        # self.B = jrandom.normal(key, (hidden_size, input_size))

        self.hidden_size = hidden_size

        ## Initiali A and B and MLPs
        a_key, b_key = jrandom.split(key)
        self.A = eqx.nn.MLP(in_size=hidden_size, out_size=hidden_size, width_size=128, depth=3, key=a_key)
        self.B = eqx.nn.MLP(in_size=input_size, out_size=hidden_size, width_size=128, depth=3, key=b_key)

    def __call__(self, inp, hidden, n, *, key=None):
        h_next = self.A(hidden) + self.B(inp)
        h_next = jax.nn.tanh(h_next)
        sharp_tanh = gradual_tanh(jnp.arange(self.hidden_size), n)
        return h_next * sharp_tanh






class RNN(eqx.Module):
    hidden_size: int
    cell: eqx.Module
    linear: eqx.nn.Linear
    bias: jax.Array

    seq_len: int
    # hidden0: jax.Array

    def __init__(self, in_size, out_size, hidden_size, seq_len,*, key):
        ckey, lkey = jrandom.split(key)
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=ckey)
        # self.cell = eqx.nn.LSTMCell(in_size, hidden_size, key=ckey)
        # self.cell = LinearCell(in_size, hidden_size, key=ckey)
        # self.cell = NeuralNetCell(in_size, hidden_size, key=ckey)

        # self.hidden0 = jnp.zeros((self.hidden_size,))

        self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=False, key=lkey)
        self.bias = jnp.zeros(out_size)

    def __call__(self, input):
        hidden = jnp.zeros((self.hidden_size,))

        # ## For GRU and Linear SSM
        # def f(carry, inp):
        #     return self.cell(inp, carry), None
        # out, _ = lax.scan(f, hidden, input)

        # ## For WSM
        # def f(carry, inp):
        #     c, old_x = carry
        #     return self.cell(inp, c, old_x), None
        # (out, _), _ = lax.scan(f, (self.hidden0, input[0]), input)

        ## For LSTM
        # (_, out), _ = lax.scan(f, (hidden, hidden), input)

        # ## For GradualTanh
        # def f(carry, inp):
        #     hidd, step = carry
        #     n = step * self.hidden_size / self.seq_len
        #     return (self.cell(inp, hidd, n), step+1), None
        # (out, _), _ = lax.scan(f, (hidden, 1), input)

        ## For GRU and Gradual Tanh
        def f(carry, inp):
            hidd, step = carry
            n = step * self.hidden_size / self.seq_len
            hidd = hidd * gradual_tanh(jnp.arange(self.hidden_size), n)
            out = self.cell(inp, hidd)
            return (out, step+1), None
        (out, _), _ = lax.scan(f, (hidden, 8), input)

        # sigmoid because we're performing binary classification
        return jax.nn.sigmoid(self.linear(out) + self.bias)
    


#%%[markdown]
# And finally the training loop.



def main(
    dataset_size=10000,
    batch_size=32,
    learning_rate=3e-3,
    steps=200,
    hidden_size=16*4,
    seq_len=100,
    depth=1,
    seed=5678,
    ):
    data_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 2)
    xs, ys = get_data(dataset_size, seq_len, key=data_key)
    iter_data = dataloader((xs, ys), batch_size)

    np.random.seed(seed+1)      ### For reproducibility in the dataloader

    ## Prinrt the shape of the data
    print("== Data Shape ==")
    print(xs.shape)
    print(ys.shape)

    model = RNN(in_size=2, out_size=1, hidden_size=hidden_size, seq_len=xs.shape[1], key=model_key)

    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
        pred_y = jax.vmap(model)(x)
        # Trains with respect to binary cross-entropy
        return -jnp.mean(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y))

    # Important for efficiency whenever you use JAX: wrap everything into a single JIT
    # region.
    @eqx.filter_jit
    def make_step(model, x, y, opt_state):
        loss, grads = compute_loss(model, x, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    optim = optax.adam(learning_rate)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    for step, (x, y) in zip(range(steps), iter_data):
        loss, model, opt_state = make_step(model, x, y, opt_state)
        loss = loss.item()
        if step % 40 == 0:
            print(f"step={step}, loss={loss}")

    pred_ys = jax.vmap(model)(xs)
    num_correct = jnp.sum((pred_ys > 0.5) == ys)
    final_accuracy = (num_correct / dataset_size).item()
    print(f"final_accuracy={final_accuracy}")

    ## Return data ready for visualisation
    return xs, ys, pred_ys


#%%[markdown]
# eqx.filter_value_and_grad will calculate the gradient with respect to all floating-point arrays in the first argument (model). In this case the model parameters will be differentiated, whilst model.hidden_size is an integer and will get None as its gradient.

# Likewise, eqx.filter_jit will look at all the arguments passed to make_step, and automatically JIT-trace every array and JIT-static everything else. In this case the model parameters and the data x and y will be traced, whilst model.hidden_size is an integer and will be static'd instead.



xs, ys, pred_ys = main()  # All right, let's run the code.


#%%

## Visualisation

## For each sequence, plot the spiral, then colour it according to the prediction: in red of green

# for i in range(1000):
#     color = 'red' if pred_ys[i] > 0.5 else 'green'
#     plt.plot(xs[i, :, 0], xs[i, :, 1], color=color)
# plt.show()

## Randomly sample 1000 spirals to plot
sample_indices = jrandom.randint(jrandom.PRNGKey(1234), (100,), 0, 10000)
for i in sample_indices:
    color = 'red' if pred_ys[i] > 0.5 else 'green'
    dotted = 'dotted' if ys[i] > 0.5 else 'solid'
    plt.plot(xs[i, :, 0], xs[i, :, 1], color=color, linestyle=dotted)
plt.show()
