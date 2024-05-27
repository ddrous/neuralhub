# %%

import os
import numpy as np

## Generate multiple time horizons Ts
Ts = np.logspace(-2, 1, 15)

## Run the main script to generate the data
for T_id, T in enumerate(Ts):
    # os.system(f"python main.py --time_horizon {T} --savepath results_sgd_4/{T_id:05d}.npz --verbose 1")
    os.system(f"python cont_norm_flow.py --time_horizon {T} --savefolder results_1/ --verbose 1")



# %%
# import numpy as np
# np.logspace(-2, 1, 15)