import numpy as np
from torchvision import transforms

import data_loading
import matplotlib.pyplot as plt

if __name__ == '__main__':
    csv_path = '/home/oli/datasets/DeepFashion/Category_and_Attribute_Prediction_Benchmark/Anno_coarse/list_category_img.txt'
    img_root_path =  '/home/oli/datasets/DeepFashion/Category_and_Attribute_Prediction_Benchmark/img/'
    dataset = data_loading.DeepFashionDataset(csv_path,img_root_path)

    fig = plt.figure()
    for i in range(len(dataset)):

        ax = plt.subplot(4, 6, i + 1)
        plt.tight_layout()
        ax.set_title('Image {}'.format(i))
        ax.imshow(dataset[i][0].squeeze().permute(1,2,0))
        ax.axis('off')

        if i == 23:
            plt.show()
            break