import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

def load_iris():
    iris = pd.read_csv('./L9_data/Iris/iris.csv')

    iris = pd.get_dummies(iris, columns=['Species'])

    species = ['Species_setosa', 'Species_versicolor', 'Species_virginica']

    for type in species:
        iris[type]=iris[type].map({False:-1,True:1})

    train_set = pd.concat([iris[iris[species[0]]==1].sample(n=30),
                        iris[iris[species[1]]==1].sample(n=30),
                        iris[iris[species[2]]==1].sample(n=30)])
    test_set = pd.concat([iris[iris[species[0]]==1].drop(train_set.index[:30]).sample(n=20),
                        iris[iris[species[1]]==1].drop(train_set.index[30:60]).sample(n=20),
                        iris[iris[species[2]]==1].drop(train_set.index[60:90]).sample(n=20)])

    return train_set, test_set

train_set, test_set = load_iris()
X_train = train_set.iloc[:, 1:-3]
y_train = train_set.iloc[:, -3:]
X_test = test_set.iloc[:, 1:-3]
y_test = test_set.iloc[:, -3:]

X_train = torch.tensor(X_train.values, dtype=torch.float)
y_train = torch.tensor(y_train.values, dtype=torch.float)
X_test = torch.tensor(X_test.values, dtype=torch.float)
y_test = torch.tensor(y_test.values, dtype=torch.float)

y_train = torch.argmax(y_train, dim=1)
y_test = torch.argmax(y_test, dim=1)

#hidden_layers=1
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        # self.relu = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# #hidden_layers=2
# class Net(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size) 
#         self.fc2 = nn.Linear(hidden_size, hidden_size) 
#         self.relu = nn.ReLU()
#         self.fc3 = nn.Linear(hidden_size, num_classes)  
    
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out) 
#         out = self.relu(out)
#         out = self.fc3(out)
#         return out


def train(model, X_train, y_train, learning_rate, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    losses = []
    
    for epoch in range(num_epochs):
        outputs = model(X_train)

        loss = criterion(outputs, y_train)
        losses.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_train).sum().item() / y_train.size(0)

    return model, losses, accuracy


def plot(train_losses):
    plt.plot(train_losses)
    plt.title('Training loss')
    plt.show()


# Train
model = Net(input_size=4, hidden_size=100, num_classes=3)
model, train_losses, train_acc = train(model, X_train, y_train, learning_rate=0.01, num_epochs=100)
print('Accuracy of the model on the train set: {}%'.format(train_acc * 100))

# Test
outputs = model(X_test)
_, predicted = torch.max(outputs.data, 1)
test_acc = (predicted == y_test).sum().item() / y_test.size(0)
print('Accuracy of the model on the test set: {}%'.format(test_acc * 100))