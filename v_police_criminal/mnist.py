import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class Net(nn.Module):
    def __init__(self):#input 28-by-28
        super().__init__()
        self.fc_x = nn.Linear(4*4*64, 256)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)
        self.conv1 = nn.Conv2d(1, 32, 4, stride=3)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc_x(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # return self.softmax(x)
        return x


if __name__ == '__main__':
    net = Net()
    with open('./mnist/train-labels.idx1-ubyte', 'rb') as p:
        p.read(8)
        y_train = np.fromfile(p, dtype=np.uint8)
    with open('./mnist/train-images.idx3-ubyte', 'rb') as p:
        p.read(16)
        x_train = np.fromfile(p, dtype=np.uint8).reshape(len(y_train), 28, 28)
    with open('./mnist/t10k-labels.idx1-ubyte', 'rb') as p:
        p.read(8)
        y_test = np.fromfile(p, dtype=np.uint8)
    with open('./mnist/t10k-images.idx3-ubyte', 'rb') as p:
        p.read(16)
        x_test = np.fromfile(p, dtype=np.uint8).reshape(len(y_test), 28, 28)
    x_train = torch.from_numpy(x_train).float().unsqueeze(1)
    x_test = torch.from_numpy(x_test).float().unsqueeze(1)
    y_train = torch.from_numpy(y_train).long().unsqueeze(1)
    y_test = torch.from_numpy(y_test).long().unsqueeze(1)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), 1e-3)

    for epoch in range(100):
        optimizer.zero_grad()
        y_pred = net(x_train)
        loss = loss_func(y_pred, y_train.squeeze())
        loss.backward()
        optimizer.step()
        if (epoch+1)%10 == 0:
            print(f'epoch{epoch+1}, loss={loss.item()}')

    with torch.no_grad():
        _, y_predicted_cls = torch.max(net(x_test), 1)
        acc = y_predicted_cls.eq(y_test.squeeze()).sum() / float(y_test.shape[0])
        print(f'accuracy = {acc}')


