from dataset import DatasetXCondDataModule


dataset = DatasetXCondDataModule('./test_images', 32)

for i in range(len(dataset)):
    print(dataset[i])