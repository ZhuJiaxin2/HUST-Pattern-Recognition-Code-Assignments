import torch
from P1 import LR_sgd, sigmoid
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 产生数据集
def generate_data(data_num=200):
    # m1 = torch.tensor([-5., 0.])
    m1 = torch.tensor([1., 0.])
    # m2 = torch.tensor([0., 5.])
    m2 = torch.tensor([0., 1.])
    s1 = torch.eye(2, dtype=torch.float32)
    s2 = torch.eye(2, dtype=torch.float32)
    X1 = torch.distributions.multivariate_normal.MultivariateNormal(m1, s1).sample((data_num,))
    X2 = torch.distributions.multivariate_normal.MultivariateNormal(m2, s2).sample((data_num,))
    Y1 = torch.ones((data_num, 1))
    Y2 = -torch.ones((data_num, 1))
    X = torch.cat((X1, X2), dim=0)
    Y = torch.cat((Y1, Y2), dim=0)
    return X, Y

def train_test_split(X, Y):
    train_size = int(0.8 * X.shape[0])
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_test, Y_test = X[train_size:], Y[train_size:]
    return X_train, Y_train, X_test, Y_test
    
# 计算分类正确率
def accuracy(X, Y, W):
    X0 = torch.ones(X.shape[0], 1)
    X = torch.cat((X0, X), dim=1)
    y_pred = torch.sigmoid(X @ W)
    y_pred = torch.where(y_pred >= 0.5, 1, -1)
    acc = torch.mean((y_pred == Y).float())
    return acc

# 画出数据集和分类面
def plot_data(X, Y, W):
    cmap = ListedColormap(['red', 'blue'])
    plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0], cmap=cmap)
    xrange = torch.linspace(-10, 10, 400)
    yrange = (-W[0] - W[1] * xrange) / W[2]
    plt.plot(xrange, yrange)
    # plt.scatter(5, 0, c='green', s=100)
    plt.show()

# 画出损失函数随epoch增加的变化曲线
def plot_loss(loss_list):
    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def test_me(x, W):
    y_pred = sigmoid(W.T @ x)
    return y_pred

# 产生数据集
X, Y = generate_data()

# 划分训练集和测试集
X_train, Y_train, X_test, Y_test = train_test_split(X, Y)

# 使用梯度下降法求解最小误差平方和最佳解
eta = 0.5
batch_size = 1
epochs = 1000
W, loss_list = LR_sgd(X_train, Y_train, eta, batch_size, epochs)
print("LR最佳解:")
print(W)

# x_test = torch.tensor([[1], [-5], [0]], dtype=torch.float32)
# y_pred = test_me(x_test, W)
# print(y_pred)

train_acc = accuracy(X_train, Y_train, W)
test_acc = accuracy(X_test, Y_test, W)
print("训练集分类正确率：", train_acc.item())
print("测试集分类正确率：", test_acc.item())
plot_data(X, Y, W)
plot_loss(loss_list)
