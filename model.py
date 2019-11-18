import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

        print(x[0].shape[0] * x[0].shape[1] * x[0].shape[2])

        x = x.view(-1, 375)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x

    def batch_train(self, train_X, train_y):
        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, IMAGE_SIZE[0], IMAGE_SIZE[1])
                batch_y = train_y[i:i + BATCH_SIZE]

                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.zero_grad()
                outputs = self(batch_X)
                loss = self.loss_function(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
            print(loss)


if __name__ == "__main__":
    n = TorchModel()
    a = np.random.uniform(size=IMAGE_SIZE)
    a = torch.Tensor(a).view(-1, 1, IMAGE_SIZE[0], IMAGE_SIZE[1])
    n.forward(a)
