# %%

import os
import numpy as np

## Generate multiple time horizons Ts
Ts = np.linspace(2, 24, 51)

## Run the main script to generate the data
for T_id, T in enumerate(Ts):
    # os.system(f"python main.py --time_horizon {T} --savepath results_sgd_4/{T_id:05d}.npz --verbose 1")
    os.system(f"python main_renormed.py --time_horizon {T} --savepath results_sgd_6/{T_id:05d}.npz --verbose 1")
