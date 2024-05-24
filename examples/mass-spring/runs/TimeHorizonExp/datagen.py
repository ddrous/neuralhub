# %%

import os
import numpy as np

## Generate multiple time horizons Ts
Ts = np.linspace(5, 25, 101)

## Run the main script to generate the data
for T_id, T in enumerate(Ts):
    os.system(f"python main.py --time_horizon {T} --savepath results/{T_id:05d}.npz --verbose 1")

