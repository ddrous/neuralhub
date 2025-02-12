
import numpy as np

class TimeSeriesDataset:
    """
    For any time series dataset, from which the others will inherit
    """
    def __init__(self, dataset, labels, t_eval, traj_prop=1.0):

        self.dataset = dataset
        n_envs, n_timesteps, n_dimensions = dataset.shape
        self.t_eval = t_eval
        self.total_envs = n_envs

        self.labels = labels

        if traj_prop < 0 or traj_prop > 1:
            raise ValueError("The smallest proportion of the trajectory to use must be between 0 and 1")
        self.traj_prop = traj_prop
        self.traj_len = int(n_timesteps * traj_prop)

        self.num_steps = n_timesteps
        self.data_size = n_dimensions

    def __getitem__(self, idx):
        inputs = self.dataset[idx, :, :]
        outputs = self.labels[idx]
        t_eval = self.t_eval
        traj_len = self.traj_len

        if self.traj_prop == 1.0:
            ### STRAIGHFORWARD APPROACH ###
            return (inputs, t_eval), outputs
        else:
            ### SAMPLING APPROACH ###
            ## Select a random trajectory of length t-2
            # start_idx = np.random.randint(0, self.num_steps - traj_len)
            # end_idx = start_idx + traj_len
            # ts = t_eval[start_idx:end_idx]
            # trajs = inputs[start_idx:end_idx, :]
            # return (trajs, ts), outputs

            ## Select a random subset of traj_len-2 indices, then concatenate the start and end points
            indices = np.sort(np.random.choice(np.arange(1,self.num_steps-1), traj_len-2, replace=False))
            indices = np.concatenate(([0], indices, [self.num_steps-1]))
            ts = t_eval[indices]
            trajs = inputs[indices, :]
            return (trajs, ts), outputs

    def __len__(self):
        return self.total_envs


class TrendsDataset(TimeSeriesDataset):
    """
    For the synthetic control dataset from Time Series Classification
    """

    def __init__(self, data_dir, skip_steps=1, traj_prop=1.0):
        try:
            time_series = []
            with open(data_dir+"synthetic_control.data", 'r') as f:
                for line in f:
                    time_series.append(list(map(float, line.split())))
            raw_data = np.array(time_series, dtype=np.float32)
            raw_data = (raw_data - np.mean(raw_data, axis=0)) / np.std(raw_data, axis=0)        ## normalise the dataset
        except:
            raise ValueError(f"Data not found at {data_dir}")

        dataset = raw_data[:, ::skip_steps, None]

        n_envs, n_timesteps, n_dimensions = dataset.shape

        ## Duplicate t_eval for each environment
        t_eval = np.linspace(0, 1., n_timesteps)

        ## We have 600 samples and 6 classes as above. Create the labels
        labels = np.zeros((600,), dtype=int)
        labels[100:200] = 1 
        labels[200:300] = 2
        labels[300:400] = 3
        labels[400:500] = 4
        labels[500:600] = 5

        self.total_envs = n_envs
        self.nb_classes = 6
        self.num_steps = n_timesteps
        self.data_size = n_dimensions

        super().__init__(dataset, labels, t_eval, traj_prop)




class MNISTDataset(TimeSeriesDataset):
    """
    For the MNIST dataset, where the time series is the pixels of the image, and the example of Trends dataset above
    """

    def __init__(self, data_dir, data_split, mini_res=4, traj_prop=1.0):
        self.nb_classes = 10
        self.num_steps = (28//mini_res)**2
        self.data_size = 1
        self.mini_res = mini_res

        self.traj_prop = traj_prop

        tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5),
                transforms.Lambda(lambda x: x[:, ::mini_res, ::mini_res]) if mini_res>1 else transforms.Lambda(lambda x: x),
                transforms.Lambda(lambda x: x.reshape(self.data_size, self.num_steps).t()),
            ]
        )

        data = torchvision.datasets.MNIST(
            data_dir, train=True if data_split=="train" else False, download=True, transform=tf
        )

        ## Get all the data in one large batch (to apply the transform)
        dataset, labels = next(iter(torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False)))

        t_eval = np.linspace(0., 1., self.num_steps)
        self.total_envs = dataset.shape[0]

        super().__init__(dataset.numpy(), labels.numpy(), t_eval, traj_prop=traj_prop)
































# #%%

# # Copyright 2021 Google LLC

# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at

# #     https://www.apache.org/licenses/LICENSE-2.0

# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """TFDS builder for pathfinder challenge."""

# import os

# import tensorflow as tf
# import tensorflow_datasets as tfds



# class Pathfinder32(tfds.core.BeamBasedBuilder):
#   """Pathfinder TFDS builder (where the resolution is 32).

#   The data for this dataset was generated using the script in
#   https://github.com/drewlinsley/pathfinder with the default parameters, while
#   followings being customized:
#   ```
#     args.paddle_margin_list = [1]
#     args.window_size = [32, 32]
#     args.padding= 1
#     args.paddle_length = 2
#     args.marker_radius = 1.5
#     args.contour_length = 14
#     args.paddle_thickness = 0.5
#     args.antialias_scale = 2
#     args.seed_distance= 7
#     args.continuity = 1.0
#     args.distractor_length = args.contour_length // 3
#     args.num_distractor_snakes = 20 // args.distractor_length
#     args.snake_contrast_list = [2]
#     args.paddle_contrast_list = [0.75]
#   ```
#   """

#   VERSION = tfds.core.Version('1.0.0')

#   def _info(self):
#     return tfds.core.DatasetInfo(
#         builder=self,
#         description=('This is a builder for pathfinder challenge dataset'),
#         features=tfds.features.FeaturesDict({
#             'image': tfds.features.Image(),
#             'label': tfds.features.ClassLabel(num_classes=2)
#         }),
#         supervised_keys=('image', 'label'),
#         homepage='',
#         citation="""@inproceedings{
#                     Kim*2020Disentangling,
#                     title={Disentangling neural mechanisms for perceptual grouping},
#                     author={Junkyung Kim* and Drew Linsley* and Kalpit Thakkar and Thomas Serre},
#                     booktitle={International Conference on Learning Representations},
#                     year={2020},
#                     url={https://openreview.net/forum?id=HJxrVA4FDS}
#                     }""",
#     )

#   def _split_generators(self, dl_manager):
#     """Downloads the data and defines the splits."""

#     return [
#         tfds.core.SplitGenerator(
#             name='easy', gen_kwargs={'file_pattern': 'curv_baseline'}),
#         tfds.core.SplitGenerator(
#             name='intermediate',
#             gen_kwargs={'file_pattern': 'curv_contour_length_9'}),
#         tfds.core.SplitGenerator(
#             name='hard', gen_kwargs={'file_pattern': 'curv_contour_length_14'})
#     ]

#   def _build_pcollection(self, pipeline, file_pattern):
#     """Generate examples as dicts."""
#     beam = tfds.core.lazy_imports.apache_beam

#     def _generate_examples(file_path):
#       """Read the input data out of the source files."""
#       example_id = 0
#       meta_examples = tf.io.read_file(file_path).numpy().decode('utf-8').split(
#           '\n')[:-1]
#       print(meta_examples)
#       for m_example in meta_examples:
#         m_example = m_example.split(' ')
#         image_path = os.path.join(ORIGINAL_DATA_DIR_32, file_pattern,
#                                   m_example[0], m_example[1])
#         example_id += 1
#         yield '_'.join([m_example[0], m_example[1],
#                         str(example_id)]), {
#                             'image': image_path,
#                             'label': int(m_example[3]),
#                         }

#     meta_file_pathes = tf.io.gfile.glob(
#         os.path.join(ORIGINAL_DATA_DIR_32, file_pattern, 'metadata/*.npy'))
#     print(len(meta_file_pathes))
#     return (pipeline
#             | 'Create' >> beam.Create(meta_file_pathes)
#             | 'Generate' >> beam.ParDo(_generate_examples))


# class Pathfinder64(tfds.core.BeamBasedBuilder):
#   """Pathfinder TFDS builder (where the resolution is 64).

#   The data for this dataset was generated using the script in
#   https://github.com/drewlinsley/pathfinder with the default parameters, while
#   followings being customized:
#   ```
#     args.padding = 1
#     args.antialias_scale = 4
#     args.paddle_margin_list = [1]
#     args.seed_distance = 12
#     args.window_size = [64,64]
#     args.marker_radius = 2.5
#     args.contour_length = 14
#     args.paddle_thickness = 1
#     args.antialias_scale = 2
#     args.continuity = 1.8  # from 1.8 to 0.8, with steps of 66%
#     args.distractor_length = args.contour_length / 3
#     args.num_distractor_snakes = 22 / args.distractor_length
#     args.snake_contrast_list = [0.8]
#   ```
#   """

#   VERSION = tfds.core.Version('1.0.0')

#   def _info(self):
#     return tfds.core.DatasetInfo(
#         builder=self,
#         description=('This is a builder for pathfinder challenge dataset'),
#         features=tfds.features.FeaturesDict({
#             'image': tfds.features.Image(),
#             'label': tfds.features.ClassLabel(num_classes=2)
#         }),
#         supervised_keys=('image', 'label'),
#         homepage='',
#         citation="""@inproceedings{
#                     Kim*2020Disentangling,
#                     title={Disentangling neural mechanisms for perceptual grouping},
#                     author={Junkyung Kim* and Drew Linsley* and Kalpit Thakkar and Thomas Serre},
#                     booktitle={International Conference on Learning Representations},
#                     year={2020},
#                     url={https://openreview.net/forum?id=HJxrVA4FDS}
#                     }""",
#     )

#   def _split_generators(self, dl_manager):
#     """Downloads the data and defines the splits."""

#     return [
#         tfds.core.SplitGenerator(
#             name='easy', gen_kwargs={'file_pattern': 'curv_baseline'}),
#         tfds.core.SplitGenerator(
#             name='intermediate',
#             gen_kwargs={'file_pattern': 'curv_contour_length_9'}),
#         tfds.core.SplitGenerator(
#             name='hard', gen_kwargs={'file_pattern': 'curv_contour_length_14'})
#     ]

#   def _build_pcollection(self, pipeline, file_pattern):
#     """Generate examples as dicts."""
#     beam = tfds.core.lazy_imports.apache_beam

#     def _generate_examples(file_path):
#       """Read the input data out of the source files."""
#       example_id = 0
#       meta_examples = tf.io.read_file(file_path).numpy().decode('utf-8').split(
#           '\n')[:-1]
#       print(meta_examples)
#       for m_example in meta_examples:
#         m_example = m_example.split(' ')
#         image_path = os.path.join(ORIGINAL_DATA_DIR_64, file_pattern,
#                                   m_example[0], m_example[1])
#         example_id += 1
#         yield '_'.join([m_example[0], m_example[1],
#                         str(example_id)]), {
#                             'image': image_path,
#                             'label': int(m_example[3]),
#                         }

#     meta_file_pathes = tf.io.gfile.glob(
#         os.path.join(ORIGINAL_DATA_DIR_64, file_pattern, 'metadata/*.npy'))
#     print(len(meta_file_pathes))
#     return (pipeline
#             | 'Create' >> beam.Create(meta_file_pathes)
#             | 'Generate' >> beam.ParDo(_generate_examples))


# class Pathfinder128(tfds.core.BeamBasedBuilder):
#   """Pathfinder TFDS builder (where the resolution is 128).

#   The data for this dataset was generated using the script in
#   https://github.com/drewlinsley/pathfinder with the default parameters, while
#   followings being customized:
#   ```
#     args.padding = 1
#     args.antialias_scale = 4
#     args.paddle_margin_list = [2,3]
#     args.seed_distance = 20
#     args.window_size = [128,128]
#     args.marker_radius = 3
#     args.contour_length = 14
#     args.paddle_thickness = 1.5
#     args.antialias_scale = 2
#     args.continuity = 1.8  # from 1.8 to 0.8, with steps of 66%
#     args.distractor_length = args.contour_length / 3
#     args.num_distractor_snakes = 35 / args.distractor_length
#     args.snake_contrast_list = [0.9]
#   ```
#   """

#   VERSION = tfds.core.Version('1.0.0')

#   def _info(self):
#     return tfds.core.DatasetInfo(
#         builder=self,
#         description=('This is a builder for pathfinder challenge dataset'),
#         features=tfds.features.FeaturesDict({
#             'image': tfds.features.Image(),
#             'label': tfds.features.ClassLabel(num_classes=2)
#         }),
#         supervised_keys=('image', 'label'),
#         homepage='',
#         citation="""@inproceedings{
#                     Kim*2020Disentangling,
#                     title={Disentangling neural mechanisms for perceptual grouping},
#                     author={Junkyung Kim* and Drew Linsley* and Kalpit Thakkar and Thomas Serre},
#                     booktitle={International Conference on Learning Representations},
#                     year={2020},
#                     url={https://openreview.net/forum?id=HJxrVA4FDS}
#                     }""",
#     )

#   def _split_generators(self, dl_manager):
#     """Downloads the data and defines the splits."""

#     return [
#         tfds.core.SplitGenerator(
#             name='easy', gen_kwargs={'file_pattern': 'curv_baseline'}),
#         tfds.core.SplitGenerator(
#             name='intermediate',
#             gen_kwargs={'file_pattern': 'curv_contour_length_9'}),
#         tfds.core.SplitGenerator(
#             name='hard', gen_kwargs={'file_pattern': 'curv_contour_length_14'})
#     ]

#   def _build_pcollection(self, pipeline, file_pattern):
#     """Generate examples as dicts."""
#     beam = tfds.core.lazy_imports.apache_beam
#     def _generate_examples(file_path):
#       """Read the input data out of the source files."""
#       example_id = 0
#       meta_examples = tf.io.read_file(
#           file_path).numpy().decode('utf-8').split('\n')[:-1]
#       print(meta_examples)
#       for m_example in meta_examples:
#         m_example = m_example.split(' ')
#         image_path = os.path.join(ORIGINAL_DATA_DIR_128, file_pattern,
#                                   m_example[0], m_example[1])
#         example_id += 1
#         yield '_'.join([m_example[0], m_example[1], str(example_id)]), {
#             'image': image_path,
#             'label': int(m_example[3]),
#         }

#     meta_file_pathes = tf.io.gfile.glob(
#         os.path.join(ORIGINAL_DATA_DIR_128, file_pattern, 'metadata/*.npy'))
#     print(len(meta_file_pathes))
#     return (
#         pipeline
#         | 'Create' >> beam.Create(meta_file_pathes)
#         | 'Generate' >> beam.ParDo(_generate_examples)
#     )


# class Pathfinder256(tfds.core.BeamBasedBuilder):
#   """Pathfinder TFDS builder (where the resolution is 256).

#   The data for this dataset was generated using the script in
#   https://github.com/drewlinsley/pathfinder with the default parameters, while
#   followings being customized:
#   ```
#     args.antialias_scale = 4
#     args.paddle_margin_list = [3]
#     args.window_size = [256,256]
#     args.marker_radius = 5
#     args.contour_length = 14
#     args.paddle_thickness = 2
#     args.antialias_scale = 2
#     args.continuity = 1.8
#     args.distractor_length = args.contour_length / 3
#     args.num_distractor_snakes = 30 / args.distractor_length
#     args.snake_contrast_list = [1.0]
#   ```
#   """

#   VERSION = tfds.core.Version('1.0.0')

#   def _info(self):
#     return tfds.core.DatasetInfo(
#         builder=self,
#         description=('This is a builder for pathfinder challenge dataset'),
#         features=tfds.features.FeaturesDict({
#             'image': tfds.features.Image(),
#             'label': tfds.features.ClassLabel(num_classes=2)
#         }),
#         supervised_keys=('image', 'label'),
#         homepage='',
#         citation="""@inproceedings{
#                     Kim*2020Disentangling,
#                     title={Disentangling neural mechanisms for perceptual grouping},
#                     author={Junkyung Kim* and Drew Linsley* and Kalpit Thakkar and Thomas Serre},
#                     booktitle={International Conference on Learning Representations},
#                     year={2020},
#                     url={https://openreview.net/forum?id=HJxrVA4FDS}
#                     }""",
#     )

#   def _split_generators(self, dl_manager):
#     """Downloads the data and defines the splits."""

#     return [
#         tfds.core.SplitGenerator(
#             name='easy', gen_kwargs={'file_pattern': 'curv_baseline'}),
#         tfds.core.SplitGenerator(
#             name='intermediate',
#             gen_kwargs={'file_pattern': 'curv_contour_length_9'}),
#         tfds.core.SplitGenerator(
#             name='hard', gen_kwargs={'file_pattern': 'curv_contour_length_14'})
#     ]

#   def _build_pcollection(self, pipeline, file_pattern):
#     """Generate examples as dicts."""
#     beam = tfds.core.lazy_imports.apache_beam

#     def _generate_examples(file_path):
#       """Read the input data out of the source files."""
#       example_id = 0
#       meta_examples = tf.io.read_file(file_path).numpy().decode('utf-8').split(
#           '\n')[:-1]
#       print(meta_examples)
#       for m_example in meta_examples:
#         m_example = m_example.split(' ')
#         image_path = os.path.join(ORIGINAL_DATA_DIR_256, file_pattern,
#                                   m_example[0], m_example[1])
#         example_id += 1
#         yield '_'.join([m_example[0], m_example[1],
#                         str(example_id)]), {
#                             'image': image_path,
#                             'label': int(m_example[3]),
#                         }

#     meta_file_pathes = tf.io.gfile.glob(
#         os.path.join(ORIGINAL_DATA_DIR_256, file_pattern, 'metadata/*.npy'))
#     print(len(meta_file_pathes))
#     return (pipeline
#             | 'Create' >> beam.Create(meta_file_pathes)
#             | 'Generate' >> beam.ParDo(_generate_examples))
  



# ## Create a pthfinder32 object and test how it works
# pathfinder32 = Pathfinder32()
# pathfinder32.download_and_prepare()
# ds = pathfinder32.as_dataset(split='easy')
# print(ds)

# ## Get the first element of the dataset
# for example in ds.take(1):
#     print(example)







































# #%%

# """
# This script downloads and unzips the UEA data from the timeseriesclassification website.
# """

# import os
# import tarfile
# import urllib.request
# import zipfile


# def download_and_unzip(url, save_dir, zipname):
#     """Downloads and unzips a (g)zip file from a url.

#     Args:
#         url (str): The url to download from.
#         save_dir (str): The directory to save the (g)zip file to.
#         zipname (str): The name of the (g)zip file.

#     Returns:
#         None
#     """
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     if len(os.listdir(save_dir)) == 0 or True:
#         urllib.request.urlretrieve(url, zipname)
#         print("Downloaded data to {}".format(zipname))
#         if zipname.split(".")[-1] == "gz":
#             with tarfile.open(zipname, "r:gz") as tar:
#                 tar.extractall(save_dir)
#         else:
#             with zipfile.ZipFile(zipname, "r") as zip:
#                 zip.extractall(save_dir)
#     else:
#         print("Data already exists in {}".format(save_dir))


# if __name__ == "__main__":
#     data_dir = "data_dir"
#     url = (
#         "http://www.timeseriesclassification.com/aeon-toolkit/Archives"
#         "/Multivariate2018_arff.zip"
#     )
#     save_dir = data_dir + "/raw/UEA/"
#     zipname = save_dir + "uea.zip"
#     download_and_unzip(url, save_dir, zipname)


# """
# This module defines the `Dataset` class and functions for generating datasets tailored to different model types.
# A `Dataset` object in this module contains three different dataloaders, each providing a specific version of the data
# required by different models:

# - `raw_dataloaders`: Returns the raw time series data, suitable for recurrent neural networks (RNNs) and structured
#   state space models (SSMs).
# - `coeff_dataloaders`: Provides the coefficients of an interpolation of the data, used by Neural Controlled Differential
#   Equations (NCDEs).
# - `path_dataloaders`: Provides the log-signature of the data over intervals, used by Neural Rough Differential Equations
#   (NRDEs) and Log-NCDEs.

# The module also includes utility functions for processing and generating these datasets, ensuring compatibility with
# different model requirements.
# """

# import os
# import pickle
# from dataclasses import dataclass
# from typing import Dict

# import jax.numpy as jnp
# import jax.random as jr
# import numpy as np

# from data_dir.dataloaders import Dataloader
# from data_dir.generate_coeffs import calc_coeffs
# from data_dir.generate_paths import calc_paths


# @dataclass
# class Dataset:
#     name: str
#     raw_dataloaders: Dict[str, Dataloader]
#     coeff_dataloaders: Dict[str, Dataloader]
#     path_dataloaders: Dict[str, Dataloader]
#     data_dim: int
#     logsig_dim: int
#     intervals: jnp.ndarray
#     label_dim: int


# def batch_calc_paths(data, stepsize, depth, inmemory=True):
#     N = len(data)
#     batchsize = 128
#     num_batches = N // batchsize
#     remainder = N % batchsize
#     path_data = []
#     if inmemory:
#         out_func = lambda x: x
#         in_func = lambda x: x
#     else:
#         out_func = lambda x: np.array(x)
#         in_func = lambda x: jnp.array(x)
#     for i in range(num_batches):
#         path_data.append(
#             out_func(
#                 calc_paths(
#                     in_func(data[i * batchsize : (i + 1) * batchsize]), stepsize, depth
#                 )
#             )
#         )
#     if remainder > 0:
#         path_data.append(
#             out_func(calc_paths(in_func(data[-remainder:]), stepsize, depth))
#         )
#     if inmemory:
#         path_data = jnp.concatenate(path_data)
#     else:
#         path_data = np.concatenate(path_data)
#     return path_data


# def batch_calc_coeffs(data, include_time, T, inmemory=True):
#     N = len(data)
#     batchsize = 128
#     num_batches = N // batchsize
#     remainder = N % batchsize
#     coeffs = []
#     if inmemory:
#         out_func = lambda x: x
#         in_func = lambda x: x
#     else:
#         out_func = lambda x: np.array(x)
#         in_func = lambda x: jnp.array(x)
#     for i in range(num_batches):
#         coeffs.append(
#             out_func(
#                 calc_coeffs(
#                     in_func(data[i * batchsize : (i + 1) * batchsize]), include_time, T
#                 )
#             )
#         )
#     if remainder > 0:
#         coeffs.append(
#             out_func(calc_coeffs(in_func(data[-remainder:]), include_time, T))
#         )
#     if inmemory:
#         coeffs = jnp.concatenate(coeffs)
#     else:
#         coeffs = np.concatenate(coeffs)
#     return coeffs


# def dataset_generator(
#     name,
#     data,
#     labels,
#     stepsize,
#     depth,
#     include_time,
#     T,
#     inmemory=True,
#     idxs=None,
#     use_presplit=False,
#     *,
#     key,
# ):
#     N = len(data)
#     if idxs is None:
#         if use_presplit:
#             train_data, val_data, test_data = data
#             train_labels, val_labels, test_labels = labels
#         else:
#             permkey, key = jr.split(key)
#             bound1 = int(N * 0.7)
#             bound2 = int(N * 0.85)
#             idxs_new = jr.permutation(permkey, N)
#             train_data, train_labels = (
#                 data[idxs_new[:bound1]],
#                 labels[idxs_new[:bound1]],
#             )
#             val_data, val_labels = (
#                 data[idxs_new[bound1:bound2]],
#                 labels[idxs_new[bound1:bound2]],
#             )
#             test_data, test_labels = data[idxs_new[bound2:]], labels[idxs_new[bound2:]]
#     else:
#         train_data, train_labels = data[idxs[0]], labels[idxs[0]]
#         val_data, val_labels = data[idxs[1]], labels[idxs[1]]
#         test_data, test_labels = None, None

#     train_paths = batch_calc_paths(train_data, stepsize, depth)
#     val_paths = batch_calc_paths(val_data, stepsize, depth)
#     test_paths = batch_calc_paths(test_data, stepsize, depth)
#     intervals = jnp.arange(0, train_data.shape[1], stepsize)
#     intervals = jnp.concatenate((intervals, jnp.array([train_data.shape[1]])))
#     intervals = intervals * (T / train_data.shape[1])

#     train_coeffs = calc_coeffs(train_data, include_time, T)
#     val_coeffs = calc_coeffs(val_data, include_time, T)
#     test_coeffs = calc_coeffs(test_data, include_time, T)
#     train_coeff_data = (
#         (T / train_data.shape[1])
#         * jnp.repeat(
#             jnp.arange(train_data.shape[1])[None, :], train_data.shape[0], axis=0
#         ),
#         train_coeffs,
#         train_data[:, 0, :],
#     )
#     val_coeff_data = (
#         (T / val_data.shape[1])
#         * jnp.repeat(jnp.arange(val_data.shape[1])[None, :], val_data.shape[0], axis=0),
#         val_coeffs,
#         val_data[:, 0, :],
#     )
#     if idxs is None:
#         test_coeff_data = (
#             (T / test_data.shape[1])
#             * jnp.repeat(
#                 jnp.arange(test_data.shape[1])[None, :], test_data.shape[0], axis=0
#             ),
#             test_coeffs,
#             test_data[:, 0, :],
#         )

#     train_path_data = (
#         (T / train_data.shape[1])
#         * jnp.repeat(
#             jnp.arange(train_data.shape[1])[None, :], train_data.shape[0], axis=0
#         ),
#         train_paths,
#         train_data[:, 0, :],
#     )
#     val_path_data = (
#         (T / val_data.shape[1])
#         * jnp.repeat(jnp.arange(val_data.shape[1])[None, :], val_data.shape[0], axis=0),
#         val_paths,
#         val_data[:, 0, :],
#     )
#     if idxs is None:
#         test_path_data = (
#             (T / test_data.shape[1])
#             * jnp.repeat(
#                 jnp.arange(test_data.shape[1])[None, :], test_data.shape[0], axis=0
#             ),
#             test_paths,
#             test_data[:, 0, :],
#         )

#     data_dim = train_data.shape[-1]
#     if len(train_labels.shape) == 1 or name == "ppg":
#         label_dim = 1
#     else:
#         label_dim = train_labels.shape[-1]
#     logsig_dim = train_paths.shape[-1]

#     raw_dataloaders = {
#         "train": Dataloader(train_data, train_labels, inmemory),
#         "val": Dataloader(val_data, val_labels, inmemory),
#         "test": Dataloader(test_data, test_labels, inmemory),
#     }
#     coeff_dataloaders = {
#         "train": Dataloader(train_coeff_data, train_labels, inmemory),
#         "val": Dataloader(val_coeff_data, val_labels, inmemory),
#         "test": Dataloader(test_coeff_data, test_labels, inmemory),
#     }

#     path_dataloaders = {
#         "train": Dataloader(train_path_data, train_labels, inmemory),
#         "val": Dataloader(val_path_data, val_labels, inmemory),
#         "test": Dataloader(test_path_data, test_labels, inmemory),
#     }
#     return Dataset(
#         name,
#         raw_dataloaders,
#         coeff_dataloaders,
#         path_dataloaders,
#         data_dim,
#         logsig_dim,
#         intervals,
#         label_dim,
#     )




























#%%

## Import transforms and torchvision
import torch
import torchvision
from torchvision import transforms
from selfmod import NumpyLoader

# ### MNIST Classification
# **Task**: Predict MNIST class given sequence model over pixels (784 pixels => 10 classes).
def create_mnist_classification_dataset(bsz=128):
    print("[*] Generating MNIST Classification Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 784, 10, 1
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
            transforms.Lambda(lambda x: x.view(IN_DIM, SEQ_LENGTH).t()),
        ]
    )

    train = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=tf
    )

    # # Return data loaders, with the provided batch size
    # trainloader = torch.utils.data.DataLoader(
    #     train, batch_size=bsz, shuffle=True
    # )
    # testloader = torch.utils.data.DataLoader(
    #     test, batch_size=bsz, shuffle=False
    # )

    ## Return NumpyLoaders
    trainloader = NumpyLoader(
        train, batch_size=bsz, shuffle=True
    )
    testloader = NumpyLoader(
        test, batch_size=bsz, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM

if __name__ == "__main__":
    ## Test the function
    trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM = create_mnist_classification_dataset()
    print(f"Number of classes: {N_CLASSES}, Sequence length: {SEQ_LENGTH}, Input dimension: {IN_DIM}")
    print(f"Trainloader: {trainloader}, Testloader: {testloader}")

    ## Get a sample, print it, along with its label
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    print(images.shape)
    print(labels.shape)
