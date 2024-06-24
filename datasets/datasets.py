# import matplotlib.pyplot as plt
import os
from operator import itemgetter

import filelock
import sys
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import requests
import scipy.io as sio
import sklearn.datasets
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from pl_bolts.datamodules import CIFAR10DataModule, FashionMNISTDataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, TensorDataset, random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST

# TODO: these will slow down the polyaxon
from tqdm import tqdm


class RandomData(pl.LightningDataModule):
    def __init__(
        self,
        x_dim=101,
        n_samples=1000,
        y_dim=1,
        batch_size=32,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_samples = n_samples
        self.batch_size = batch_size

    def setup(self, stage):
        X_ = torch.tensor(
            np.random.uniform(0, 1, [self.n_samples, self.x_dim]),
            dtype=torch.float,
        )
        Y_ = torch.tensor(
            np.random.uniform(0, 1, [self.n_samples, self.y_dim]),
            dtype=torch.float,
        )
        self.train = TensorDataset(X_, Y_)
        self.valid = TensorDataset(X_, Y_)
        self.teste = TensorDataset(X_, Y_)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.teste, batch_size=1)


class hesma(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = None,
        n_samples: int = 4096 * 16,
        batch_size: int = 256,
        num_workers: int = 4,
        filename: str = "HESMA_Profiles.hdf",
        shuffle_training: bool = True,
        create_if_empty=True,
    ):
        super().__init__()

        if data_dir is None:
            data_dir = "datasets"

        self.filename = os.path.join(data_dir, filename)

        self.data_dir = data_dir
        self.create_if_empty = create_if_empty
        # self.config_data = config_data

        self.n_samples = n_samples

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_training = shuffle_training

        self.transform = transforms.Compose([])

        self.shape_imput = (1, 3)
        self.shape_target = (1, 2)

    def prepare_data(self):
        def data_interp(data_in, data_len=100):
            interpolated_data = []
            for key_ in data_in.keys():
                use_scipy = True
                if use_scipy:
                    from scipy.interpolate import interp1d

                    f_out = interp1d(data_in.velocity, np.log(data_in[key_]))
                    interp_scipy = np.exp(
                        f_out(
                            np.linspace(
                                data_in.velocity.min(),
                                data_in.velocity.max(),
                                data_len,
                            )
                        )
                    )
                    interpolated_data.append(interp_scipy)
                    # interp_np = np.interp(np.linspace(data_in.velocity.min(), data_in.velocity.max(), data_len), data_in.velocity,np.log(data_in[key_]))
                else:
                    interpolated_data.append(
                        np.exp(
                            np.interp(
                                np.linspace(
                                    data_in.velocity.min(),
                                    data_in.velocity.max(),
                                    data_len,
                                ),
                                data_in.velocity,
                                np.log(data_in[key_]),
                            )
                        )
                    )

            # interp_scipy = f_out(np.linspace(data_in.velocity.min(), data_in.velocity.max(), data_len))
            # interp_np = np.interp(np.linspace(data_in.velocity.min(), data_in.velocity.max(), data_len), data_in.velocity,np.log(data_in[key_]))
            # print("-------")
            # print(interp_scipy-interp_np)
            # print(np.interp(np.linspace(data_in.velocity.min(), data_in.velocity.max(), data_len), data_in.velocity,np.log(data_in[key_])))
            # print("---------", data_in[key_].dtype)
            # print(interpolated_data)
            # print(np.array(interpolated_data))
            # print(np.array(interpolated_data)-np.array(interpolated_data_test))
            interpolated_data = np.array(interpolated_data)
            interpolated_data = interpolated_data.swapaxes(1, 0)
            return interpolated_data

        def data_preprocessing(data, data_len=100):
            data_index = list(map(itemgetter(0), data.index.values))
            data_index = list(dict.fromkeys(data_index))

            data_out = []
            for t in data_index:
                data_out.append(data_interp(data.loc[t], data_len=data_len))

            data_out = np.array(data_out)
            # data_out = np.concatenate(data_out, axis=0)
            return data_out

        data = pd.read_hdf(self.filename, key="profiles")

        data_np = data_preprocessing(data)
        parameters_raw = data_np.swapaxes(2, 1).reshape(data_np.shape[0], -1)

        self.parameters_raw_min = parameters_raw.min() - 1e-5

        parameters_n = np.log10(parameters_raw - self.parameters_raw_min)

        # n1, n2, n3 = parameters_n.shape
        # parameters_n = parameters_n.reshape(n1 * n2, n3)
        self.scaler_parameters = preprocessing.StandardScaler().fit(parameters_n)

        n_parameters_n = self.scaler_parameters.transform(parameters_n)
        # .reshape(n1, n2, n3)

        self.data_train = TensorDataset(
            torch.tensor(n_parameters_n, dtype=torch.float),
            torch.tensor(np.zeros([len(n_parameters_n), 1]), dtype=torch.float),
        )
        self.data_validate = self.data_train
        self.data_test = self.data_train

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_training,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_validate,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def _inverse_transform(self, data):
        data_unnorm = 10 ** (
            self.scaler_parameters.inverse_transform(data).astype(float)
            + self.parameters_raw_min
        )

        return data_unnorm


class hesma_label(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = None,
        n_samples: int = 4096 * 16,
        batch_size: int = 256,
        num_workers: int = 4,
        filename: str = "HESMA_Profiles.hdf",
        shuffle_training: bool = True,
        create_if_empty=True,
        use_elemental_mass=False,
        is_sampler_balanced=False,
        mode='train',
        parameters_raw_min=None,
        parameters_scale=None,
        parameters_mean=None
    ):
        super().__init__()

        if mode == 'train':
            self.requires_data = True
            if None not in (parameters_raw_min, parameters_scale, parameters_mean):
                print("Warning: Data Scaling Parameters Specified but not Used!", file=sys.stderr)
        elif mode == 'eval':
            self.requires_data = False
            assert None not in (parameters_raw_min, parameters_scale, parameters_mean)
            self.parameters_raw_min = np.asarray(parameters_raw_min).reshape(100, -1)
            self.parameters_scale = np.asarray(parameters_scale)
            self.parameters_mean = np.asarray(parameters_mean)

            self.scaler_parameters = preprocessing.StandardScaler()
            self.scaler_parameters.mean_ = self.parameters_mean
            self.scaler_parameters.scale_ = self.parameters_scale

        if data_dir is None:
            data_dir = "datasets"

        self.filename = os.path.join(data_dir, filename)

        self.data_dir = data_dir
        self.create_if_empty = create_if_empty
        # self.config_data = config_data
        self.is_sampler_balanced = is_sampler_balanced

        self.n_samples = n_samples

        self.use_elemental_mass = use_elemental_mass
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_training = shuffle_training

        self.transform = transforms.Compose([])

        self.shape_imput = (1, 3)
        self.shape_target = (1, 2)

    def prepare_data(self):

        if not self.requires_data:

            return {"parameters_mean":list(self.parameters_mean), "parameters_scale":list(self.parameters_scale), "parameters_raw_min":list(self.parameters_raw_min.ravel())}

        def data_interp(data_in, data_len=100):
            interpolated_data = []
            for key_ in data_in.keys():
                use_scipy = True
                if use_scipy:
                    from scipy.interpolate import interp1d

                    f_out = interp1d(data_in.velocity, np.log(data_in[key_]))
                    interp_scipy = np.exp(
                        f_out(
                            np.linspace(
                                data_in.velocity.min(),
                                data_in.velocity.max(),
                                data_len,
                            )
                        )
                    )
                    interpolated_data.append(interp_scipy)
                    # interp_np = np.interp(np.linspace(data_in.velocity.min(), data_in.velocity.max(), data_len), data_in.velocity,np.log(data_in[key_]))
                else:
                    interpolated_data.append(
                        np.exp(
                            np.interp(
                                np.linspace(
                                    data_in.velocity.min(),
                                    data_in.velocity.max(),
                                    data_len,
                                ),
                                data_in.velocity,
                                np.log(data_in[key_]),
                            )
                        )
                    )

            interpolated_data = np.array(interpolated_data)
            interpolated_data = interpolated_data.swapaxes(1, 0)
            return interpolated_data

        def data_preprocessing(data, data_len=100):
            # classes = {'ddt':0, 'def':1, 'det':2, 'doubledet':3}
            data_index = list(map(itemgetter(0), data.index.values))
            data_index = list(dict.fromkeys(data_index))

            data_out = []
            labels = []
            for t in data_index:
                if t[:9] in ["doubledet"]:
                    labels.append(t[:9])
                    data_out.append(data_interp(data.loc[t], data_len=data_len))
                elif t[:3] in ["det", "ddt", "def"]:
                    labels.append(t[:3])
                    data_out.append(data_interp(data.loc[t], data_len=data_len))

            data_out = np.array(data_out)
            label_encoder = LabelEncoder()
            labels = label_encoder.fit_transform(labels)
            # data_out = np.concatenate(data_out, axis=0)
            print(label_encoder.classes_)
            print("'ddt':0, 'def':1, 'det':2, 'doubledet':3")
            return data_out, labels

        data = pd.read_hdf(self.filename, key="profiles")
        # These are the profiles that have no zeros in their abundance sum
        data = data[data[data.columns[2:]].sum(axis=1) != 0.0]
        data[data.columns[2:]] += 1e-6
        # data[data.columns[2:]] /= data[data.columns[2:]].sum(axis=1).values[:, None]

        # v0 = dat.velocity.groupby(

        data_np, labels = data_preprocessing(data)

        parameters_raw = data_np.copy()  # .swapaxes(2, 1).reshape(data_np.shape[0], -1)
        idx_density = 0
        idx_velocity = 1
        if not self.use_elemental_mass:
            parameters_raw[:, :, 2:] /= parameters_raw[:, :, 2:].sum(axis=-1)[
                :, :, None
            ]
            self.parameters_raw_min = data_np.min(axis=0) - 1e-5

            parameters_n = parameters_raw.copy()
            # print(parameters_n.shape)
            # import pdb; pdb.set_trace()
            parameters_n[:, :, 0] = np.log10(
                parameters_raw[:, :, 0] - self.parameters_raw_min[:, 0]
            )
            parameters_n[:, :, 1] = (
                parameters_raw[:, :, 1] - self.parameters_raw_min[:, 1]
            )
            parameters_n[:, :, 2:] = parameters_raw[
                :, :, 2:
            ]  # - self.parameters_raw_min[:, 2:])
            assert np.all(
                np.isfinite(parameters_n[:, :, 2:])
            ), "Log of abundances returned non-finite values"

        else:  # This needs to be restructured as the integration is actually nto quite right (as we can start at v0=0)

            v1 = (
                parameters_raw[:, -1, idx_velocity]
                + (
                    parameters_raw[:, -1, idx_velocity]
                    - parameters_raw[:, -2, idx_velocity]
                )
            )[:, None]

            velocity = parameters_raw[:, :, idx_velocity]
            density = parameters_raw[:, :, idx_density]
            centers = (velocity[:, 1:] + velocity[:, :-1]) / 2
            v0 = np.zeros((len(velocity), 1))
            v1 = (2 * velocity[:, -1] - centers[:, -1])[:, None]
            dv = np.diff(np.concatenate((v0, centers, v1), axis=1), axis=1)

            mass = density * velocity ** 2 * dv
            parameters_raw[:, :, 2:] *= mass[:, :, None]

            parameters_raw = parameters_raw[:, :, 1:]
            parameters_n = parameters_raw.copy()

            parameters_n[:, :, :1] = np.concatenate(
                (parameters_raw[:, :1, :1], np.diff(parameters_raw[:, :, :1], axis=1)),
                axis=1,
            )
            self.parameters_raw_min = parameters_n.min(axis=0) - 1e-5

            parameters_n = parameters_n - self.parameters_raw_min
            parameters_n = np.log10(parameters_n)

        # self.parameters_raw_min = parameters_raw.min() - 1e-5

        parameters_n = parameters_n.swapaxes(2, 1).reshape(parameters_raw.shape[0], -1)
        # n1, n2, n3 = parameters_n.shape
        # parameters_n = parameters_n.reshape(n1 * n2, n3)
        # Scale only the velocity and density parameters
        if self.use_elemental_mass:
            self.scaler_parameters = preprocessing.StandardScaler().fit(parameters_n)
            n_parameters_n = self.scaler_parameters.transform(parameters_n)
            # breakpoint()

        else:
            self.scaler_parameters = preprocessing.StandardScaler().fit(
                parameters_n[:, :200]
            )

            n_parameters_n = parameters_n.copy()
            n_parameters_n[:, :200] = self.scaler_parameters.transform(
                parameters_n[:, :200]
            )
            # .reshape(n1, n2, n3)
            assert np.all(
                np.isfinite(n_parameters_n)
            ), "Error in Data Preprocessing, Found bad values"

        index = self.index = np.random.permutation(len(n_parameters_n))
        index_train = self.index_train = index[: 17 * 5]
        index_val = self.index_val = index[17 * 5 :]

        if True:  # self.is_sampler_balanced:
            from torch.utils.data import WeightedRandomSampler

            class_counts = []
            for i in range(4):
                class_counts.append(sum(labels[index_train] == i))
            num_samples = sum(class_counts)

            class_weights = [
                num_samples / class_counts[i] for i in range(len(class_counts))
            ]

            # TODO: remove if. since ddt and det are similar, we can just double the weight of others
            if True:
                weights = [
                    class_weights[labels[index_train[i]]]
                    for i in range(int(num_samples))
                ]
            else:
                weights = []
                for i in range(int(num_samples)):
                    if i in [1, 3]:
                        weights.append(2 * class_weights[labels[index_train[i]]])
                    else:
                        weights.append(class_weights[labels[index_train[i]]])

            self.train_sampler = WeightedRandomSampler(
                torch.DoubleTensor(weights), int(num_samples)
            )

        self.data_train = TensorDataset(
            torch.tensor(n_parameters_n[index_train], dtype=torch.float),
            torch.tensor(labels[index_train], dtype=torch.int64),
        )

        self.data_validate = TensorDataset(
            torch.tensor(n_parameters_n[index_val], dtype=torch.float),
            torch.tensor(labels[index_val], dtype=torch.int64),
        )
        # self.data_validate = self.data_train
        self.data_test = self.data_validate
        print("Data Shape:", len(index), len(self.data_train), len(self.data_validate))

        if self.batch_size == -1:
            self.train_batch_size = len(self.data_train)
            self.val_batch_size = len(self.data_validate)
            self.test_batch_size = len(self.data_test)
        else:
            self.train_batch_size = self.batch_size
            self.val_batch_size = self.batch_size
            self.test_batch_size = self.batch_size

        self.parameters_mean = self.scaler_parameters.mean_
        self.parameters_scale = self.scaler_parameters.scale_ 

        return {"parameters_mean":list(self.parameters_mean), "parameters_scale":list(self.parameters_scale), "parameters_raw_min":list(self.parameters_raw_min.ravel())}


    def train_dataloader(self):
        if self.is_sampler_balanced:
            return DataLoader(
                self.data_train,
                batch_size=self.train_batch_size,
                num_workers=self.num_workers,
                drop_last=True,
                sampler=self.train_sampler,
            )
        else:
            return DataLoader(
                self.data_train,
                batch_size=self.train_batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle_training,
                drop_last=True,
            )

    def val_dataloader(self):
        return DataLoader(
            self.data_validate,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def _inverse_transform(self, data):

        if not self.use_elemental_mass:
            data_unnorm = np.empty_like(data.cpu().numpy()).reshape(-1, 100, 12)
            data_unnorm[:, :, 2:] = data.cpu().numpy()[:, 200:].reshape([-1, 100, 10])
            data_unnorm[:, :, :2] = (
                self.scaler_parameters.inverse_transform(data.cpu().numpy()[:, :200])
                .astype(float)
                .reshape([-1, 2, 100])
                .swapaxes(2, 1)
                + self.parameters_raw_min[:, :2]
            )
            # data_unnorm = self.scaler_parameters.inverse_transform(data).astype(float).reshape([-1, 12, 100]).swapaxes(2,1) + self.parameters_raw_min
            data_unnorm[:, :, 0] = 10 ** data_unnorm[:, :, 0]
            data_unnorm[:, :, 2:] = data_unnorm[:, :, 2:]
        else:
            # breakpoint()
            data_unnorm = (
                self.scaler_parameters.inverse_transform(
                    data.cpu().numpy()
                )  # NOTE: Switch to using data.shape[0] for the first axis and -1 for the middle
                .reshape(-1, 11, 100)
                .swapaxes(1, 2)
            )
            # data_unnorm[:, :, 1:] = 10 ** data_unnorm[:, :, 1:]
            # data_unnorm[:, :, 1:] += self.parameters_raw_min[None, :, 1:]
            data_unnorm = 10 ** data_unnorm
            data_unnorm += self.parameters_raw_min
            data_unnorm[:, :, 0] = np.cumsum(
                np.concatenate((data_unnorm[:, :1, 0], data_unnorm[:, 1:, 0]), axis=1),
                axis=1,
            )

        # data_unnorm = 10 ** (
        #    self.scaler_parameters.inverse_transform(data).astype(float)
        #    + self.parameters_raw_min
        # )

        return data_unnorm


class tardis_spectra(pl.LightningDataModule):
    """contains no labels"""

    def __init__(
        self,
        data_dir: str = None,
        n_samples: int = 4096 * 16,
        batch_size: int = 256,
        num_workers: int = 4,
        filename: str = "VAE_Spectra_train.hdf",
        shuffle_training: bool = True,
        create_if_empty=True,
        with_label: bool = False,
        wave_min: int = 3400,
        wave_max: int = 7600,
        wave_n: int = 500,
        mode: str = 'train',
        param_mean: float = None,
        param_std: float = None,
        spec_mean: float = None,
        spec_std: float = None
    ):
        super().__init__()



        if mode == 'train':
            self.requires_data = True
            if None not in (param_mean, param_std, spec_mean, spec_std):
                print("Warning: Data Scaling Parameters Specified but not Used!", file=sys.stderr)
        elif mode == 'eval':
            self.requires_data = False
            assert None not in (param_mean, param_std, spec_mean, spec_std)
            self.param_mean = np.asarray(param_mean)
            self.param_std = np.asarray(param_std)
            self.spec_mean = np.asarray(spec_mean)
            self.spec_std = np.asarray(spec_std)

        if data_dir is None:
            data_dir = "datasets"

        self.filename = os.path.join(data_dir, filename)
        self.with_label = with_label

        self.data_dir = data_dir
        self.create_if_empty = create_if_empty
        # self.config_data = config_data

        self.n_samples = n_samples

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_training = shuffle_training

        self.transform = transforms.Compose([])

        self.shape_imput = (1, 3)
        self.shape_target = (1, 2)

        self.wave_min, self.wave_max = wave_min, wave_max
        self.wave_n = wave_n

    def prepare_data(self):

        if not self.requires_data:
            self.scaler_parameters = preprocessing.StandardScaler()
            self.scaler_parameters.mean_ = self.param_mean
            self.scaler_parameters.scale_ = self.param_std
            self.scaler_spectra = preprocessing.StandardScaler()
            self.scaler_spectra.mean_ = self.spec_mean
            self.scaler_spectra.scale_ = self.spec_std

            return {"param_mean":list(self.param_mean), "param_std":list(self.param_std), "spec_mean":list(self.spec_mean), "spec_std":list(self.spec_std)}

        def data_interp(data_out, data_len=500, wave_start=3400, wave_end=7600):
            interpolated_data = []

            use_scipy = (data_out.columns.min() < wave_start) or (
                data_out.columns.max() > wave_end
            )
            if use_scipy:
                from scipy.interpolate import interp1d

                for i in range(len(data_out)):
                    f_out = interp1d(
                        np.array(data_out.columns[::-1]),
                        np.log(data_out.iloc[i].values[::-1]),
                        fill_value="extrapolate",
                    )
                    interp_scipy = np.exp(
                        f_out(
                            np.logspace(
                                np.log10(wave_start),
                                np.log10(wave_end),
                                data_len,
                            )
                        )
                    )
                    interpolated_data.append(interp_scipy)
            else:
                interpolated_data = np.exp(
                    np.interp(
                        np.logspace(
                            np.log10(wave_start),
                            np.log10(wave_end),
                            data_len,
                        ),
                        np.array(data_in.columns[::-1]),
                        np.log(data_out.values[:, ::-1]),
                    )
                )

            interpolated_data = np.array(interpolated_data)
            # interpolated_data = interpolated_data.swapaxes(1, 0)
            return interpolated_data

        def data_preprocessing(data, data_len=100):
            # classes = {'ddt':0, 'def':1, 'det':2, 'doubledet':3}
            labels = data.index.get_level_values(0).values
            label_encoder = LabelEncoder()
            labels = label_encoder.fit_transform(labels)

            print(label_encoder.classes_)
            print(dict(enumerate(label_encoder.classes_)))
            print("'ddt':0, 'def':1, 'det':2, 'doubledet':3")
            return data, labels

        with filelock.FileLock(self.filename + ".lock"):  # Prevents memory issues
            data_in = pd.read_hdf(self.filename, key="inputs")
            data_out = pd.read_hdf(self.filename, key="outputs")
            data_in = data_in.loc[data_out.index]
            if self.with_label:
                data_in, labels = data_preprocessing(data_in)
            self.spectra_raw = data_interp(
                data_out, self.wave_n, self.wave_min, self.wave_max
            )
            del data_out

        self.parameters_raw = (
            data_in.copy().values
        )  # .swapaxes(2, 1).reshape(data_np.shape[0], -1)
        if self.with_label:
            self.parameters_raw[:, 0] = np.log10(
                self.parameters_raw[:, 0]
            )  # Log-scale the luminosity
        # import pdb; pdb.set_trace()

        # self.parameters_raw_min = parameters_raw.min(axis=0) - 1e-5
        # parameters_n = parameters_raw - self.parameters_raw_min
        
        self.scaler_parameters = preprocessing.StandardScaler().fit(self.parameters_raw)
        n_parameters_n = self.scaler_parameters.transform(self.parameters_raw)

        self.param_mean = self.scaler_parameters.mean_
        self.param_std = self.scaler_parameters.scale_

        self.spectra_raw_min = self.spectra_raw.min(axis=0) - 1e-5
        spectra_n = np.log10(self.spectra_raw)
        self.scaler_spectra = preprocessing.StandardScaler().fit(spectra_n)
        n_spectra_n = self.scaler_spectra.transform(spectra_n)

        self.spec_mean = self.scaler_spectra.mean_
        self.spec_std = self.scaler_spectra.scale_

        rng = np.random.default_rng(42)  # Same split for all ensemble members
        index = rng.permutation(len(n_parameters_n))
        test_train_split = 0.8
        split_index = int(test_train_split * len(n_parameters_n))
        index_train = index[:split_index]
        index_val = index[split_index:]

        if self.with_label:
            self.data_train = TensorDataset(
                torch.tensor(n_parameters_n[index_train], dtype=torch.float),
                torch.tensor(labels[index_train], dtype=torch.int64),
                torch.tensor(n_spectra_n[index_train], dtype=torch.float),
            )
            self.data_validate = TensorDataset(
                torch.tensor(n_parameters_n[index_val], dtype=torch.float),
                torch.tensor(labels[index_val], dtype=torch.int64),
                torch.tensor(n_spectra_n[index_val], dtype=torch.float),
            )
        else:
            self.data_train = TensorDataset(
                torch.tensor(n_parameters_n[index_train], dtype=torch.float),
                torch.tensor(n_spectra_n[index_train], dtype=torch.float),
            )

            self.data_validate = TensorDataset(
                torch.tensor(n_parameters_n[index_val], dtype=torch.float),
                torch.tensor(n_spectra_n[index_val], dtype=torch.float64),
            )

        self.data_test = self.data_validate  # For now, we'll do test set later
        print("Data Shape:", len(index), len(self.data_train), len(self.data_validate))

        if self.batch_size == -1:
            self.train_batch_size = len(self.data_train)
            self.val_batch_size = len(self.data_validate)
            self.test_batch_size = len(self.data_test)
        else:
            self.train_batch_size = self.batch_size
            self.val_batch_size = self.batch_size
            self.test_batch_size = self.batch_size

        self.labels = labels
        self.index_train = index_train
        self.index_val = index_val

        return {"param_mean":list(self.param_mean), "param_std":list(self.param_std), "spec_mean":list(self.spec_mean), "spec_std":list(self.spec_std)}

    def train_dataloader(self):

        return DataLoader(
            self.data_train,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_training,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_validate,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


    def _inverse_transform(self, y_predict, y_sigma=None):

        y_predict_unnorm = 10 ** self.scaler_spectra.inverse_transform(
            y_predict.astype(np.float64)
        )
        if y_sigma is None:
            return y_predict_unnorm

        y_sigma_unnorm = np.log(10) * y_predict_unnorm * y_sigma.astype(np.float64)
        return y_predict_unnorm, y_sigma_unnorm
