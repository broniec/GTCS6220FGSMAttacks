import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import misc_functions_2 as mf
import numpy


class Net(nn.Module):
    def __init__(self, linear_layer=15):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20, linear_layer)
        self.fc2 = nn.Linear(linear_layer, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 20)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 1, 16, 16)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(64, 1, 16, 16)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def run(trogo=64, epochs=300, lr=0.01, linear_layer=15):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=trogo, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 300)')
    parser.add_argument('--epochs', type=int, default=epochs, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=lr, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    numpy.random.seed(args.seed)

    data, labels = mf.retreive_semeion_data()
    data = data + numpy.random.normal(0.0, 1.0, data.shape)
    randomized = numpy.random.permutation(len(labels))
    train_cube = torch.FloatTensor(data[randomized][:1280])
    valid_cube = torch.FloatTensor(data[randomized][1280:1536])
    train_labels = torch.from_numpy(labels[randomized][:1280])
    valid_labels = torch.from_numpy(labels[randomized][1280:1536])
    train_set = torch.utils.data.TensorDataset(train_cube, train_labels)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size)
    test_set = torch.utils.data.TensorDataset(valid_cube, valid_labels)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size)

    model = Net(linear_layer)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch)
        test(model, test_loader)
    return model


if __name__ == '__main__':
    run()
