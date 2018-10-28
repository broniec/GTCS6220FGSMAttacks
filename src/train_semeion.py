# import torch
# import torch.autograd as ag
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy
# import torchvision.datasets as dset
# from torchvision import datasets, transforms
#
# import misc_functions as mf
#
# criterion = nn.CrossEntropyLoss()
#
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(256, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         # x = self.pool(F.relu(self.conv1(x)))
#         # x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 256)
#         # print(x.data)
#         x = F.relu(self.fc1(x))
#         # print(x.data)
#         x = F.relu(self.fc2(x))
#         # print(x.data)
#         x = self.fc3(x)
#         # print(x.data)
#         return x
#
#
# def train(model, train_loader, optimizer, epoch):
#     for batch_idx, (x, target) in enumerate(train_loader):
#         optimizer.zero_grad()
#         x, target = ag.Variable(x), ag.Variable(target)
#         out = model(x)
#         loss = criterion(out, target)
#         loss.backward()
#         optimizer.step()
#         # if batch_idx % 100 == 0:
#         #     print('Loss:', loss.data[0])
#
#
# def test(model, test_loader):
#     for batch_idx, (x, target) in enumerate(test_loader):
#         out = model(x)
#         _, pred_label = torch.max(out.data, 1)
#         print(target.data[0])
#         print(pred_label.data[0])
#
#
# def main():
#     learning_rate = 0.01
#     num_epochs = 5
#     seed = 1
#
#     torch.manual_seed(seed)
#
#     data, labels = mf.retreive_semeion_data()
#     randomized = numpy.random.permutation(len(labels))
#     train_cube = torch.FloatTensor(data[randomized][:900])
#     valid_cube = torch.FloatTensor(data[randomized][900:1000])
#     train_labels = torch.from_numpy(labels[randomized][:900])
#     valid_labels = torch.from_numpy(labels[randomized][900:1000])
#     train_set = torch.utils.data.TensorDataset(train_cube, train_labels)
#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=1)
#     test_set = torch.utils.data.TensorDataset(valid_cube, valid_labels)
#     test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)
#
#     model = Net()
#     optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
#
#     for epoch in range(num_epochs):
#         train(model, train_loader, optimizer, epoch)
#         test(model, test_loader)
#
#
# if __name__ == '__main__':
#     main()

import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import torchvision.datasets as dset
import cv2
from torchvision import datasets, transforms

import misc_functions as mf

criterion = nn.CrossEntropyLoss()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(256, 2)

    def forward(self, x):
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        return x


def train(model, train_loader, optimizer, epoch):
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # x, target = ag.Variable(x), ag.Variable(target)
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        pass
        # if batch_idx % 10 == 0:
        #     print('Loss:', loss.item())


def test(model, test_loader):
    correct = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        #test
        temp = x.numpy()
        for i in range(16):
            for j in range(16):
                if temp[0][i][j] == 0:
                    print('0', end='')
                else:
                    print('1', end='')
                # print(temp[0][i][j], end='')
            print('\n')
        print('---')
        # print(target)
        #test

        out = model(x)
        _, pred_label = torch.max(out.data, 1)
        if target.data[0].item() == pred_label.data[0].item():
            correct += 1
    print(correct / len(test_loader))


def main():
    learning_rate = 0.01
    num_epochs = 1
    seed = 1
    torch.manual_seed(seed)

    data, labels = mf.retreive_semeion_data()
    # randomized = numpy.random.permutation(len(labels))
    train_cube = torch.FloatTensor(data[:25])
    valid_cube = torch.FloatTensor(data[:25])
    train_labels = torch.from_numpy(labels[:25])
    valid_labels = torch.from_numpy(labels[:25])
    print(valid_labels.data)
    train_set = torch.utils.data.TensorDataset(train_cube, train_labels)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1)
    test_set = torch.utils.data.TensorDataset(valid_cube, valid_labels)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)

    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)


if __name__ == '__main__':
    main()

