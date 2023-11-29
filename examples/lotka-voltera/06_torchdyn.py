#%%[markdown]
# # Lotka-voltera with TorchDyn
# based on https://github.com/DiffEqML/torchdyn/blob/master/tutorials/module1-neuralde/m1a_neural_ode_cookbook.ipynb
# run this in the conda environment `torch`

#%%
import time
from torchdyn.core import NeuralODE
from torchdyn.nn import DataControl, DepthCat, Augmenter, GalLinear, Fourier
from torchdyn.datasets import *
from torchdyn.utils import *

import numpy as np
import matplotlib.pyplot as plt


#%%

## ts and ys from data/lotka_volterra_diffrax.npz
data = np.load('./data/lotka_volterra_diffrax.npz')
t, X = data['ts'], data['ys']

X.shape, t.shape

#%%

colors = ['orange', 'blue'] 
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
for i in range(1):
    ax.plot(X[i, :, 0], X[i, :, 1], color=colors[i])
# %%

import torch
import torch.utils.data as data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X_train = torch.Tensor(X[:, 0, :]).to(device)
y_train = torch.Tensor(X[:, :, :]).to(device)
train = data.TensorDataset(X_train, y_train)
trainloader = data.DataLoader(train, batch_size=len(X), shuffle=True)

# %%

import torch.nn as nn
import pytorch_lightning as pl

class Learner(pl.LightningModule):
    def __init__(self, t_span:torch.Tensor, model:nn.Module):
        super().__init__()
        self.model, self.t_span = model, t_span
        self.params = torch.abs(torch.randn(4, requires_grad=True))
        self.losses = []
    
    def forward(self, x):
        dx0 = x[0]*self.params[0] - x[0]*x[1]*self.params[1]
        dx1 = x[0]*x[1]*self.params[2] - x[1]*self.params[3]
        physics = torch.Tensor([dx0, dx1])
        # return self.model(x)
        return physics + self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch      
        cutoff = int(0.1 * len(self.t_span))        ## TODO Very important, but not magical !

        t_eval, y_hat = self.model(x, self.t_span[:cutoff])
        y_hat = y_hat.transpose(0, 1)
        # print("shapes:", x.shape, y_hat.shape, y.shape)

        loss_a = 0
        for param in self.model.parameters():
            # losss_p += torch.sum(param ** 2)
            # print(param.shape)
            loss_a += torch.norm(param)

        loss_p = nn.MSELoss()(y_hat, y[:, :cutoff, :])
        loss = loss_p + 1e-5*loss_a
        print(f"TotalLoss: {loss:.8f}   TrajLoss: {loss_p:.8f}  ParamsLoss: {loss_a:.8f}")

        self.losses.append(loss)

        return {'loss': loss}   
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_dataloader(self):
        return trainloader
    
# %%
# vector field parametrized by a NN
f = nn.Sequential(
        nn.Linear(2, 16),
        nn.Tanh(), 
        nn.Linear(16, 16),
        nn.Softplus(),          ## TODO very important !
        nn.Linear(16, 2))

t_span = torch.linspace(0, 10, 100)

# Neural ODE
# `interpolator` here refers to the scheme used together with `solver` to ensure estimates of the solution at all points in `t_span` are returned. 
# During solution of the adjoint, cubic interpolation is used regardless of `interpolator`.
model = NeuralODE(f, sensitivity='adjoint', solver='tsit5', interpolator=None, atol=1e-3, rtol=1e-3).to(device)

# %%

start = time.time()

# train the Neural ODE
learn = Learner(t_span, model)
trainer = pl.Trainer(min_epochs=200, max_epochs=2500)
trainer.fit(learn)

end = time.time()
print("\nTraining time:", time.strftime("%H:%M:%S", time.gmtime(end - start)))

# %%

t_span = torch.linspace(0, 10, 100) ## Forecasting
t_eval, trajectory = model(X_train.cpu(), t_span)

model_y = trajectory.detach().transpose(0, 1).cpu().numpy()
ts = t_span.detach().cpu().numpy()
ys = X

losses = torch.Tensor(learn.losses).detach().cpu().numpy()
# trajectory.shape

# plot_2D_depth_trajectory(t_span, trajectory, X, len(X))
# plot_2D_state_space(trajectory, X, len(X))

# %%

# fig, ax = plt.subplots(2, 2, figsize=(6*2, 3.5*2))
fig, ax = plt.subplot_mosaic('AB;CC', figsize=(6*2, 3.5*2))
# fig, ax = plt.subplot_mosaic('AB', figsize=(6*2, 3.5*1))

ax['A'].plot(ts, ys[0, :, 0], c="dodgerblue", label="Preys (GT)")
ax['A'].plot(ts, model_y[0, :, 0], ".", c="navy", label="Preys (NODE)")

ax['A'].plot(ts, ys[0, :, 1], c="violet", label="Predators (GT)")
ax['A'].plot(ts, model_y[0, :, 1], ".", c="purple", label="Predators (NODE)")

ax['A'].set_xlabel("Time")
ax['A'].legend()
ax['A'].set_title("Trajectories")

ax['B'].plot(ys[0, :, 0], ys[0, :, 1], c="turquoise", label="GT")
ax['B'].plot(model_y[0, :, 0], model_y[0, :, 1], ".", c="teal", label="Neural ODE")
ax['B'].set_xlabel("Preys")
ax['B'].set_ylabel("Predators")
ax['B'].legend()
ax['B'].set_title("Phase space")

ax['C'].plot(losses, c="grey", label="Losses")
ax['C'].set_xlabel("Epochs")
ax['C'].set_title("Loss")
ax['C'].set_yscale('log')

plt.tight_layout()
plt.savefig("data/neural_ode_diffrax.png")
plt.show()


# %% [markdown]

# # Preliminary results
# - Unlike diffrax, this doesn't generalise. Whether we learn on 10% or 100% of the trajectories, the model fails to forecast.
# - On the bright side, the model doesn't return NaNs for long trajectories. Why so ? ## TODO


# %%
