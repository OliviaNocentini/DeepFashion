
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#example taken from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file, sep = '                        ', header=2)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])

        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = torch.from_numpy(np.array([landmarks]).astype('int'))
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            raise Exception("Sorry, but I don't know how to do it")
        return image, landmarks



def collate_fn_padd(batch):
    # Example taken from  https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/2
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


batch_size = 2**5


#loading the train_set

face_dataset = FaceLandmarksDataset(csv_file='/home/oli/datasets/DeepFashion/Category_and_Attribute_Prediction_Benchmark/Anno_coarse/list_category_img.txt',
                                    root_dir='/home/oli/datasets/DeepFashion/Category_and_Attribute_Prediction_Benchmark/img')
                                    
# creating the train_loader

train_loader = torch.utils.data.DataLoader(face_dataset, batch_size=batch_size, collate_fn=collate_fn_padd)

dataiter = iter(train_loader)
print(dataiter)
images,labels =dataiter.next()

#showing the first 8 pictures of the dataset

fig = plt.figure()
for i in range(len(images)):

    ax = plt.subplot(2, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Label {}'.format(labels[i]))
    ax.imshow(images[i])
    ax.axis('off')

    if i == 7:
        plt.show()
        break

