
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



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
        with open(
                '/home/oli/datasets/DeepFashion/Category_and_Attribute_Prediction_Benchmark/Anno_coarse/list_category_img.txt') as fin:
            lines = fin.readlines()[2:]   # returns a list containing each line in the file as a list item.
            #print("lines", lines)
            lines = list(filter(lambda x:len(x) > 0, lines)) #filter filters the lines in which len(x) > 0
            print("lines", lines)
            pairs = list(map(lambda x: x.strip().split(), lines))  # map do the operation of strip.flit to each line
            #print("lines", pairs)
            pairs = np.array(pairs)

        self.landmarks_frame = pairs
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        print("idx",idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame[idx, 0])
        #print(self.root_dir)
        #print(self.landmarks_frame[idx, 0])
        #print("img_name",img_name)

        #print("img_name",img_name)
        image = io.imread(img_name)
        #print("img_name",img_name)

        landmarks = self.landmarks_frame[idx, 1:]
        landmarks = torch.from_numpy(np.array([landmarks]).astype('int'))
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            raise Exception("Sorry, but I don't know how to do it")
        #print("image",image.shape)
        #print("landmarks",landmarks)
        return image, landmarks




def collate_fn_padd(batch):
    # Please check https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/2
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


batch_size = 100

validation_split = .2
shuffle_dataset = True
random_seed= 42

dataset = FaceLandmarksDataset(csv_file='/home/oli/datasets/DeepFashion/Category_and_Attribute_Prediction_Benchmark/Anno_coarse/list_category_img.txt',
                                    root_dir='/home/oli/datasets/DeepFashion/Category_and_Attribute_Prediction_Benchmark/img')
'''

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
print(dataset_size)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

print(len(train_sampler))
'''

lengths = [int(len(dataset)*0.8), int(len(dataset)*0.2)]

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
print(train_dataset)
print(test_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,collate_fn=collate_fn_padd)
validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,collate_fn=collate_fn_padd)

#train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_padd)

print("pippo")

dataiter = iter(train_loader)
#print("dataiter",dataiter)
#print("dataiter.next()", dataiter.next())
images, labels =dataiter.next()


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

