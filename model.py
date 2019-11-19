import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets

BATCH_SIZE = 100
EPOCHS = 3
NUM_CLASSES = 10
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

    def forward(self, x):
        x = x.to(self.device)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout1(x)

        # print(x[0].shape[0] * x[0].shape[1] * x[0].shape[2])

        x = x.view(-1, 375)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x

    def batch_train(self):
        for epoch in range(EPOCHS):
            # for i in tqdm(range(0, len(self.trainset), BATCH_SIZE)):
            for data in self.trainset:
            # for i in range(len(self.trainset)):
                # print(self.trainset.size())
                # break
                # self.train_X, self.train_y = self.trainset[i: i+BATCH_SIZE]
                # batch_X = self.train_X[i:i + BATCH_SIZE].view(-1, 1, IMAGE_SIZE[0], IMAGE_SIZE[1])
                # batch_y = self.train_y[i:i + BATCH_SIZE]
                
                # batch_X, batch_y = self.trainset
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



if __name__ == "__main__":
    n = TorchModel()
    n.load_data()
    n.batch_train()
    torch.save(n.state_dict(), "params.pt")
    # a = np.random.uniform(size=IMAGE_SIZE)
    # a = torch.Tensor(a).view(-1, 1, IMAGE_SIZE[0], IMAGE_SIZE[1])
    # n.forward(a)
