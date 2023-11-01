import os
import struct
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')
    
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

train_set = load_mnist('./L9_data/MNIST', kind='train')
test_set = load_mnist('./L9_data/MNIST', kind='t10k')

X_train = train_set[0]
y_train = train_set[1]
X_test = test_set[0]
y_test = test_set[1]

X_train = torch.tensor(X_train, dtype=torch.float).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
X_test = torch.tensor(X_test, dtype=torch.float).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

print(train_data[0])

trainloader = DataLoader(train_data, batch_size=256, shuffle=True)
testloader = DataLoader(test_data, batch_size=256, shuffle=False)


# Define the LeNet network structure
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=2)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.sigmoid(self.conv1(x))
        x = self.avg_pool1(x)
        x = torch.sigmoid(self.conv2(x))
        x = self.avg_pool2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        # x = self.softmax(self.fc3(x))
        x = self.fc3(x)
        return x


net = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.2, momentum=0.5)
optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)


train_loss = []
train_acc = []

# Train
for epoch in range(10):
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.view(-1, 1, 28, 28)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_loss.append(running_loss / len(trainloader))
    train_acc.append(100 * correct / total)
    print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))

# Test
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.view(-1, 1, 28, 28)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    test_acc = correct / total
print('test accuracy = ', test_acc)

# Randomly select 10 
indices = torch.randint(0, len(test_data), (10,))
subset = torch.utils.data.Subset(test_data, indices)
dataloader = torch.utils.data.DataLoader(subset, batch_size=10)
# Get a batch of data
images, labels = next(iter(dataloader))
images = images.view(-1, 1, 28, 28)
# Make predictions
outputs = net(images)
_, predicted = torch.max(outputs, 1)

print("Predicted labels:", predicted)
print("True labels:", labels)


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()