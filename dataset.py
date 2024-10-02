from functools import partial
from os.path import exists
from torch import nn
from pathlib import Path
from torchvision import transforms as T, utils
import h5py
import torch
import pytorch_lightning as pl
from PIL import Image
import os
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image



# Pure LightningDataModule without explicit dataset wrapping
class DatasetXCondDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        image_size,
        batch_size=32,
        num_workers=4,
        augment_horizontal_flip=False,
        convert_image_to=None,
        split=(0.8, 0.1, 0.1),  # Split proportions for train, val, test
        exts=['hdf5']
    ):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment_horizontal_flip = augment_horizontal_flip
        self.convert_image_to = convert_image_to
        self.split = split
        self.exts = exts

        # List all files with the given extensions in the dataset folder
        self.paths = [p for ext in self.exts for p in Path(self.data_dir).glob(f'**/*.{ext}')]

        # Define the transformations
        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.RandomHorizontalFlip() if self.augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(self.image_size),
            T.ToTensor()
        ])

    def prepare_data(self):
        # You can download datasets, etc. here if needed
        pass

    def setup(self, stage=None):
        # Load the data, no dataset class needed, it's handled directly in the dataloaders
        dataset_size = len(self.paths)  # Assuming 512 images per HDF5 file

        # We calculate dataset indices for splitting
        train_size = int(self.split[0] * dataset_size)
        val_size = int(self.split[1] * dataset_size)
        test_size = dataset_size - train_size - val_size

        indices = list(range(dataset_size))
        self.train_indices, self.val_indices, self.test_indices = random_split(indices, [train_size, val_size, test_size])

    def _load_data(self, index):
        # Calculate which file and which image in the file
        i, j = index, index
        path = self.paths[i]

        try:
            h5file = h5py.File(path, 'r')
        except Exception as e:
            os.remove(path)
            print(f"Error reading {path}: {e}")
            raise

        img = Image.fromarray(h5file['full'][512//2, ::].astype('uint8'))
        parameters = h5file['unitPar'][::]
        return self.transform(img), torch.tensor(parameters)

    def _dataset_generator(self, indices):
        # Generator to provide data for dataloaders
        for idx in tqdm(indices):
            yield self._load_data(idx)

    def train_dataloader(self):
        return DataLoader(
            dataset=list(self._dataset_generator(self.train_indices)),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=list(self._dataset_generator(self.val_indices)),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=list(self._dataset_generator(self.test_indices)),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

class center_crop:

    def __init__(self, image_size):

        self.image_size = image_size

    def __call__(self, images):

        c, w, h, l = images.shape
        low = l//2 - self.image_size[0]//2
        high = l//2 + self.image_size[0]//2

        return images[:, low:high, low:high, low:high]
    


# Pure LightningDataModule without explicit dataset wrapping
class DatasetXCond3DDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        image_size,
        batch_size=32,
        num_workers=4,
        augment_horizontal_flip=False,
        convert_image_to=None,
        split=(0.8, 0.1, 0.1),  # Split proportions for train, val, test
        exts=['hdf5']
    ):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment_horizontal_flip = augment_horizontal_flip
        self.convert_image_to = convert_image_to
        self.split = split
        self.exts = exts

        # List all files with the given extensions in the dataset folder
        self.paths = [p for ext in self.exts for p in Path(self.data_dir).glob(f'**/*.{ext}')]
        self.paths = self.paths[:10]


        # Define the transformations
        self.transform = T.Compose([
            center_crop(self.image_size),
        ])

    def prepare_data(self):
        # You can download datasets, etc. here if needed
        pass

    def setup(self, stage=None):
        # Load the data, no dataset class needed, it's handled directly in the dataloaders
        dataset_size = len(self.paths)  # Assuming 512 images per HDF5 file

        # We calculate dataset indices for splitting
        train_size = int(self.split[0] * dataset_size)
        val_size = int(self.split[1] * dataset_size)
        test_size = dataset_size - train_size - val_size

        indices = list(range(dataset_size))
        self.train_indices, self.val_indices, self.test_indices = random_split(indices, [train_size, val_size, test_size])

    def _load_data(self, index):
        # Calculate which file and which image in the file
        i, j = index, index
        path = self.paths[i]

        try:
            h5file = h5py.File(path, 'r')
        except Exception as e:
            os.remove(path)
            print(f"Error reading {path}: {e}")
            raise

        img = h5file['full'][::].astype('float32').swapaxes(3, 0)
        parameters = h5file['unitPar'][::]
        return torch.tensor(self.transform(img)), torch.tensor(parameters)

    def _dataset_generator(self, indices):
        # Generator to provide data for dataloaders
        for idx in tqdm(indices):
            yield self._load_data(idx)

    def train_dataloader(self):
        return DataLoader(
            dataset=list(self._dataset_generator(self.train_indices)),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=list(self._dataset_generator(self.val_indices)),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=list(self._dataset_generator(self.test_indices)),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
# Testing the DatasetXCondDataModule without a model
if __name__ == "__main__":

    for path in os.listdir('test_images'):
        try:
            h5file = h5py.File('test_images/'+path, 'r')
        except Exception as e:
            os.remove('test_images/'+path)
            print(f"Deleted {path}")

    # Parameters
    data_dir = './test_images'  # Update this to the actual directory
    image_size = (32, 32)  # Specify your image size
    batch_size = 32

    # Initialize the data module
    data_module = DatasetXCondDataModule(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        augment_horizontal_flip=True
    )

    data_module.prepare_data()
    data_module.setup()

    # Test the train dataloader
    print("Testing train dataloader...")
    train_loader = data_module.train_dataloader()

    # Iterate through a batch
    for batch_idx, (images, parameters) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Images shape: {images.shape}")  # Expect something like [batch_size, channels, height, width]
        print(f"Parameters shape: {parameters.shape}")  # Expect parameter shape depending on your hdf5 dataset
        break  # Remove this break to test more batches

