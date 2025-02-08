
import numpy as np

class TimeSeriesDataset:
    """
    For any time series dataset, from which the others will inherit
    """
    def __init__(self, dataset, labels, t_eval, adaptation=False, traj_prop_min=1.0, use_full_traj=True):

        self.dataset = dataset
        n_envs, n_trajs_per_env, n_timesteps, n_dimensions = dataset.shape
        self.t_eval = t_eval
        self.total_envs = n_envs

        self.labels = labels

        if traj_prop_min < 0 or traj_prop_min > 1:
            raise ValueError("The smallest proportion of the trajectory to use must be between 0 and 1")
        self.traj_prop_min = traj_prop_min

        self.num_steps = n_timesteps
        self.data_size = n_dimensions
        self.num_shots = 1
        self.use_full_traj = use_full_traj      ## If True, we use the full trajectory, else we only return the initial condition
        self.adaptation = adaptation

    def __getitem__(self, idx):
        if self.use_full_traj:
            inputs = self.dataset[idx, :self.num_shots, :, :]
            outputs = self.labels[idx]
        else:
            inputs = self.dataset[idx, :self.num_shots, 0, :]
            outputs = self.dataset[idx, :self.num_shots, :, :]
        t_evals = self.t_eval[idx]

        if self.traj_prop_min == 1.0:
            ### STRAIGHFORWARD APPROACH ###
            return (inputs, t_evals), outputs
        else:
            ### SAMPLING APPROACH ###
            ## Sample a start and end time step for the task, and interpolate to produce new timesteps
            ### The minimum distance between the start and finish is min_len
            traj_len = t_evals.shape[0]
            new_traj_len = traj_len ## We always want traj_len samples
            min_len = int(traj_len * self.traj_prop_min)
            start_idx = np.random.randint(0, traj_len - min_len)
            end_idx = np.random.randint(start_idx + min_len, traj_len)

            ts = t_evals[start_idx:end_idx]
            trajs = outputs[:, start_idx:end_idx, :]
            new_ts = np.linspace(ts[0], ts[-1], new_traj_len)
            new_trajs = np.zeros((self.num_shots, new_traj_len, self.data_size))
            for i in range(self.num_shots):
                for j in range(self.data_size):
                    new_trajs[i, :, j] = np.interp(new_ts, ts, trajs[i, :, j])

            if self.use_full_traj:
                return (new_trajs[:,:,:], new_ts), new_trajs
            else:
                return (new_trajs[:,0,:], new_ts), new_trajs

    def __len__(self):
        return self.total_envs


class TrendsDataset(TimeSeriesDataset):
    """
    For the synthetic control dataset from Time Series Classification
    """

    def __init__(self, data_dir, skip_steps=-1, adaptation=False, traj_prop_min=1.0, use_full_traj=True):
        try:
            time_series = []
            with open(data_dir+"synthetic_control.data", 'r') as f:
                for line in f:
                    time_series.append(list(map(float, line.split())))
            raw_data = np.array(time_series, dtype=np.float32)
            raw_data = (raw_data - np.mean(raw_data, axis=0)) / np.std(raw_data, axis=0)        ## normalise the dataset
        except:
            raise ValueError(f"Data not found at {data_dir}")

        dataset = raw_data[:, None, ::skip_steps, None]

        n_envs, n_trajs_per_env, n_timesteps, n_dimensions = dataset.shape

        ## Duplicate t_eval for each environment
        t_eval = np.linspace(0, 1., n_timesteps)
        t_eval = np.repeat(t_eval[None,:], n_envs, axis=0)

        ## We have 600 samples and 6 classes as above. Create the labels
        labels = np.zeros((600,), dtype=int)
        labels[100:200] = 1 
        labels[200:300] = 2
        labels[300:400] = 3
        labels[400:500] = 4
        labels[500:600] = 5

        super().__init__(dataset, labels, t_eval, adaptation, traj_prop_min, use_full_traj)

