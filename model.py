import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from sklearn.neighbors import KDTree

BATCH_SIZE = 100
EPOCHS = 3
NUM_CLASSES = 10
N_NEIGHBOURS = 60
IMAGE_SIZE = (28, 28)


class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 30, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(30, 15, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(375, 128)
        self.fc2 = nn.Linear(128, 50)
        self.fc3 = nn.Linear(50, NUM_CLASSES)

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("Running on the GPU")
        else:
            self.device = torch.device("cpu")
            print("Running on the CPU")

        self.to(self.device)

        self.loss_function = nn.NLLLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        self.load_data()

        self.layer_outputs = [np.array([]), np.array([])]
        self.tree_list = []
        self.label_list = []

    def forward(self, x):
        x = x.to(self.device)
        self.output1 = self.pool1(F.relu(self.conv1(x)))
        self.output2 = self.pool2(F.relu(self.conv2(self.output1)))
        x = self.dropout1(self.output2)

        # print(x[0].shape[0] * x[0].shape[1] * x[0].shape[2])

        x = x.view(-1, 375)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x

    def batch_train(self):
        for epoch in range(EPOCHS):
            for data in self.trainset:
                batch_X, batch_y = data
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.zero_grad()
                outputs = self(batch_X)
                loss = self.loss_function(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
            print(loss)

    def load_data(self):
        train = datasets.MNIST("", train=True, download=True,
                               transform=transforms.Compose([transforms.ToTensor()]))

        test = datasets.MNIST("", train=False, download=True,
                              transform=transforms.Compose([transforms.ToTensor()]))

        self.trainset = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE,
                                                    shuffle=True)
        self.testset = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE,
                                                   shuffle=False)

    def prepare_neighbours(self):
        with torch.no_grad():
            for data in tqdm(self.trainset):
                X, y = data
                X, y = X.to(self.device), y.to(self.device)
                output = self(X)
                self.label_list.append(output)
                self.layer_outputs[0] = np.concatenate((self.layer_outputs[0], self.output1.cpu().detach().numpy().flatten()))
                self.layer_outputs[1] = np.concatenate((self.layer_outputs[1], self.output2.cpu().detach().numpy().flatten()))

            for i in range(2):
                tree = KDTree(self.layer_outputs[i])
                self.tree_list.append(tree)

    def test_data(self):
        correct = 0
        total = 0

        with torch.no_grad():
            for data in self.testset:
                X, y = data
                X, y = X.to(self.device), y.to(self.device)
                output = self(X)

                dis1, nn1 = self.tree_list[0].query([self.output1], k=N_NEIGHBOURS)
                dis2, nn2 = self.tree_list[1].query([self.output2], k=N_NEIGHBOURS)

                for idx, i in enumerate(output):
                    if torch.argmax(i) == y[idx]:
                        correct += 1
                    total += 1
        print("Accuracy", round(correct / total, 3))


if __name__ == "__main__":
    n = TorchModel()
    n.load_data()
    n.batch_train()
    torch.save(n.state_dict(), "params.pt")
    n.test_data()

    # a = np.random.uniform(size=IMAGE_SIZE)
    # a = torch.Tensor(a).view(-1, 1, IMAGE_SIZE[0], IMAGE_SIZE[1])
    # n.forward(a)
