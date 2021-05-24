from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image


class DeepFashionDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, img_root):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        # Transforms
        self.data_info = pd.read_csv(csv_file, header=None, sep=' ')
        # First column contains the image paths
        self.image_arr = np.asarray(img_root + self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        self.transformations = \
            transforms.Compose([transforms.Resize(300),
                                transforms.CenterCrop(200),
                                transforms.ToTensor()])
        self.data_len = len(self.data_info.index)


    def __len__(self):
        return self.data_len  #len returns the len of the dataset

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)

        # Transform image to tensor
        img_as_tensor = self.transformations(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

