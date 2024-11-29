# %%
import math
import time

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import matplotlib
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax


matplotlib.rcParams.update({"font.size": 30})

class NeuralCDE(eqx.Module):
    initial: eqx.nn.MLP

    def __init__(self, data_size, nb_classes, hidden_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        ikey, fkey, lkey = jr.split(key, 3)
        self.initial = eqx.nn.MLP(data_size, nb_classes, width_size, depth, key=ikey)

    def __call__(self, xs, evolving_out=False):
        # Each sample of data consists of some timestamps `ts`, and some `coeffs`
        # parameterising a control path. These are used to produce a continuous-time
        # input path `control`.
        y0 = self.initial(xs)
        return jax.nn.softmax(y0)

def get_data(split="train", task="condition"):
    ### Davide's Sleep ###
    import numpy as np
    data_folder = "../../data/new_sleep/"
    prefix = "train" if split == "train" else "valid"
    train_horizon = 500

    data = np.load(f'{prefix}_latents_full.npy')
    print(f"Data shape: {data.shape}")
    ys = data[:, -1, :]

    # print("The first 5 samples of the data are:", ys[:5])

    labels = []
    with open(data_folder+f'{prefix}_annotations.csv', 'r') as f:
        for line in f:
            if 'subject' not in line:
                human = int(line.split(',')[0].strip())
                sleep_phase = int(line.split(',')[1].strip())
                labels.append((human, sleep_phase))
    labels = np.array(labels)
    labels = labels[:, 1] if task == "condition" else labels[:, 0]
    labels = labels-1 if task == "condition" else labels
    assert len(data) == len(labels)
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")

    ## Turn the Labels into one-hot encoding
    nb_classes = len(np.unique(labels))
    labels = jax.nn.one_hot(labels, nb_classes)
    print(f"One-hot labels shape: {labels.shape}")
    ######################

    # ts = jnp.broadcast_to(jnp.linspace(0, 1, ys.shape[1]), (ys.shape[0], ys.shape[1]))
    # ys = jnp.concatenate([ts[:, :, None], ys], axis=-1)  # time is a channel

    # coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(ts, ys)

    _, data_size = ys.shape
    return ys, labels, data_size, nb_classes

get_data(split="train", task="condition")

def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


#%%

batch_size,=6925//4,
lr,=3e-4,
steps,=35000,
hidden_size,=16,
width_size,=128,
depth,=1,
seed,=5678,


#####-----------------####################

key = jr.PRNGKey(seed)
train_data_key, test_data_key, model_key, loader_key = jr.split(key, 4)

ys, labels, data_size, nb_classes = get_data(split="train", task="condition")

model = NeuralCDE(data_size, nb_classes, hidden_size, width_size, depth, key=model_key)

# Training loop like normal.

@eqx.filter_jit
def loss(model, xs_i, label_i):
    pred = jax.vmap(model)(xs_i)

    # # Binary cross-entropy
    # bxe = label_i * jnp.log(pred) + (1 - label_i) * jnp.log(1 - pred)
    # bxe = -jnp.mean(bxe)
    # acc = jnp.mean((pred > 0.5) == (label_i == 1))

    print(pred.shape, label_i.shape)

    # Categorical cross-entropy
    bxe = jnp.mean(-jnp.sum(label_i * jnp.log(pred), axis=(-1,)))
    acc = jnp.mean(jnp.argmax(pred, axis=-1) == jnp.argmax(label_i, axis=-1))

    return bxe, acc

grad_loss = eqx.filter_value_and_grad(loss, has_aux=True)

@eqx.filter_jit
def make_step(model, data_i, opt_state):
    xi, label_i = data_i
    (bxe, acc), grads = grad_loss(model, xi, label_i)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return bxe, acc, model, opt_state

scheduler = optax.exponential_decay(init_value=lr, transition_steps=200, decay_rate=0.995)
optim = optax.adam(scheduler)
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
# print("Shapes of ys and labels:", ys.shape, labels.shape)
for step, data_i in zip(range(steps), dataloader((ys, labels), batch_size, key=loader_key)):
    start = time.time()
    bxe, acc, model, opt_state = make_step(model, data_i, opt_state)
    end = time.time()

    if step % (steps//10) == 0:
        print(
            f"Step: {step}, Loss: {bxe}, Accuracy: {acc}, Computation time: "
            f"{end - start}"
        )

#%%
ys, labels, _, _ = get_data(split="train", task="condition")
bxe, acc = loss(model, ys, labels)
print(f"Test loss: {bxe}, Test Accuracy: {acc}")

import numpy as np
test_id = np.random.randint(0, len(ys))
# Plot results
sample_ys = ys[test_id]
pred = model(sample_ys, evolving_out=True)
pred = jnp.argmax(pred, axis=-1)
print(pred.shape)


#%%
# Let's calculate the per-class accuracy
pred = eqx.filter_vmap(lambda x: model(x, evolving_out=False))(ys)
pred = jnp.argmax(pred, axis=-1)
true = jnp.argmax(labels, axis=-1)

# Calculate the per-class accuracy
per_class_acc = []
for i in range(nb_classes):
    mask = true == i
    per_class_acc.append(jnp.mean(pred[mask] == true[mask]))
per_class_acc = jnp.array(per_class_acc)

print("==== Overall accuracy ====")
print(jnp.mean(pred == true))

print("==== Per-class accuracy ====")
print(per_class_acc)



# %%
# Serialise the model
# eqx.tree_serialise_leaves("tmp/kidger_cde.eqx", model)


## Umap cluster the ys using the labels
from neuralhub import *
import umap

reducer = umap.UMAP()
embedding = reducer.fit_transform(ys)
print(embedding.shape, labels.shape, pred.shape)

# %%
# Plot the UMAP
fig = plt.figure(figsize=(16, 8*2))
ax = fig.add_subplot(2, 1, 1)
# sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=np.argmax(labels, axis=-1), ax=ax)

# colors = np.argmax(labels, axis=-1)
# colors = pred

# # ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap="coolwarm")
# sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=colors, ax=ax, palette="coolwarm")

event_id = {'Sleep stage W': 0,
            'Sleep stage 1': 1,
            'Sleep stage 2': 2,
            'Sleep stage 3/4': 3,
            'Sleep stage R': 4}
event_id_inv = {v: k for k, v in event_id.items()}

ax2 = fig.add_subplot(2, 1, 2)
## Plot condition by condition
for cond in range(nb_classes):
    mask = np.argmax(labels, axis=-1) == cond
    ax.scatter(embedding[mask, 0], embedding[mask, 1], label=event_id_inv[cond])

    maxk2 = pred == cond
    ax2.scatter(embedding[maxk2, 0], embedding[maxk2, 1], label=event_id_inv[cond])

ax.set_title("True Labels", fontsize=20)
ax2.set_title("Predicted Labels", fontsize=20)
ax.legend(fontsize=20)
ax2.legend(fontsize=20)
plt.show()

## Save the figure 
fig.savefig("umap_pred.png", dpi=300, bbox_inches="tight")

# %%
