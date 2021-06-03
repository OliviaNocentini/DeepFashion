import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import data_loading
import matplotlib.pyplot as plt
from torch.autograd import Variable
from skimage import feature
from skimage.feature import hog

batch_size = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#https://medium.com/ml2vec/intro-to-pytorch-with-image-classification-on-a-fashion-clothes-dataset-e589682df0c5

class CNN(nn.Module):
    def __init__(self, hog_size):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),  #changed here the number of channels
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(20000, 49)  #changed here the number of inputs and ouputs


    def forward(self, x, hog):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        x2 = hog
        x = torch.cat((out, x2), dim=1)
        # print("end of the network")
        # print("x.shape", x.shape)
        return x


class FashionCNN(nn.Module):

    def __init__(self,hog_size):
        super(FashionCNN, self).__init__()

        self.layer1 = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64 * 49 * 49, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

        self.layer3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10 + hog_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 49)
        )

    def forward(self, x, hog):
        out = self.layer1(x)
        #print("out", out.shape)
        out = self.layer2(out)
        #print("out", out.shape)
        out = out.view(out.size(0), -1)
        #print("out", out.shape)
        out = self.fc1(out)

        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        x2= hog
        x = torch.cat((out, x2), dim=1)
        #print("end of the network")
        # print("x.shape", x.shape)
        return self.layer3(x)

if __name__ == '__main__':
    csv_path = '/home/oli/datasets/DeepFashion/Category_and_Attribute_Prediction_Benchmark/Anno_coarse/list_category_img.txt'
    img_root_path =  '/home/oli/datasets/DeepFashion/Category_and_Attribute_Prediction_Benchmark/img/'
    dataset = data_loading.DeepFashionDataset(csv_path,img_root_path)

    lengths = [int(len(dataset) * 0.8), int(len(dataset) * 0.2)]

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # print(train_dataset)
    # print(test_dataset)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,collate_fn=collate_fn_padd)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_padd)

    dataiter = iter(train_loader)
    # print("dataiter",dataiter)
    # print("dataiter.next()", dataiter.next())
    images, labels =dataiter.next()
    #print(images.shape)

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


fd = hog(images[0].numpy().squeeze(), orientations=8, pixels_per_cell=(8, 8),
                  cells_per_block=(1, 1), visualize=False, multichannel=True)


# loading CNN
model = CNN(len(fd))
#model = FashionCNN(len(fd))
model.to(device)

# defining error
error = nn.CrossEntropyLoss()

# defining lr and optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 10
count = 0

# Lists for visualization of loss and accuracy
loss_list = []
iteration_list = []
accuracy_list = []

# Lists for knowing classwise accuracy
predictions_list = []
labels_list = []

print("Back propagation starting")
for epoch in range(num_epochs):
    print("Epoch ", epoch)
    running_loss = 0.0

    for images, labels in train_loader:
        # Transfering images and labels to GPU if available
        images, labels = images.to(device), labels.to(device)

        # visualize = false because nto interested to visualize the hog image

        #print("images.shape", images.shape)
        images= Variable(images)
        #train = Variable(images.view(batch_size, 3, 28, 28))

        # print(data_vector.shape)
        fd = np.array([hog(image.cpu().numpy().squeeze(), orientations=8, pixels_per_cell=(8, 8),
                           cells_per_block=(1, 1), visualize=False, multichannel=True) for image in images])

        fd = fd.astype(np.float32)
        data_vector1 = torch.from_numpy(fd).to(device)

        labels = Variable(labels)

        # Forward pass
        outputs = model(images, data_vector1)

        #print(outputs)
        #print(labels)
        loss = error(outputs, labels)

        # Initializing a gradient as 0 so there is no mixing of gradient among the batches
        optimizer.zero_grad()

        # Propagating the error backward
        loss.backward()

        # Optimizing the parameters
        optimizer.step()

        count += 1
        # Testing the model
        if not (count % 50):  # It's same as "if count % 50 == 0"
            total = 0
            correct = 0

            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)

                test = Variable(images)


                # print(data_vector.shape)
                fd = np.array([hog(image.cpu().numpy().squeeze(), orientations=8, pixels_per_cell=(8, 8),
                                   cells_per_block=(1, 1), visualize=False, multichannel=True) for image in images])

                fd = fd.astype(np.float32)
                data_vector1 = torch.from_numpy(fd).to(device)

                outputs = model(test,data_vector1)

                #print("into test part")

                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()

                total += len(labels)

                accuracy = correct * 100 / total
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)

        if not (count % 500):
            print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))



