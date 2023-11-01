import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 产生数据集
def generate_data():
    m1 = torch.tensor([-5., 0.])
    # m1 = torch.tensor([1., 0.])
    m2 = torch.tensor([0., 5.])
    # m2 = torch.tensor([0., 1.])
    s1 = torch.eye(2, dtype=torch.float32)
    s2 = torch.eye(2, dtype=torch.float32)
    X1 = torch.distributions.multivariate_normal.MultivariateNormal(m1, s1).sample((200,))
    X2 = torch.distributions.multivariate_normal.MultivariateNormal(m2, s2).sample((200,))
    Y1 = torch.ones((200, 1))
    Y2 = -torch.ones((200, 1))
    X = torch.cat((X1, X2), dim=0)
    Y = torch.cat((Y1, Y2), dim=0)
    return X, Y

# 使用广义逆求解最小误差平方和最佳解
def generalized_inverse(X, Y):
    W0 = torch.ones(X.shape[0], 1)
    X = torch.cat((W0, X), dim=1)
    X_1 = torch.inverse(torch.matmul(X.T, X))
    X_plus = torch.matmul(X_1, X.T)
    g = torch.matmul(X_plus, Y)
    return g

# 使用梯度下降法求解最小误差平方和最佳解
def gradient_descent(X, Y, eta=0.01, epochs=100, batch_size = 40):#P2 0.01 30
    X0 = torch.ones(X.shape[0], 1)
    X = torch.cat((X0, X), dim=1)
    W = torch.zeros(X.shape[1], 1)
    t = 0
    loss_list = []
    batch_num = X.shape[0]/batch_size
    while 1:
        # sum_grad = 0
        # for i in range(X.shape[0]):
        for n in range(int(batch_num)):
            sum_grad = 0
            min_idx = int(n * batch_size)
            max_idx = int((n+1) * batch_size)
            for i in range(min_idx, max_idx):
                x = torch.unsqueeze(X[i], dim=1)
                Y_pred = torch.matmul(W.T, x)
                gradient = (Y_pred - Y[i]) * x
                sum_grad += gradient
            ave_grad = 2 * sum_grad / X.shape[0]
            t +=1
            if torch.norm(ave_grad) == 0  or t == epochs:
                break
            W -= eta * ave_grad
            loss = torch.norm(torch.matmul(X,W) - Y)**2 / X.shape[0]
            loss_list.append(loss)
        if torch.norm(ave_grad) == 0  or t == epochs:
            break
    return W, loss_list

# 计算分类正确率
def accuracy(X, Y, W):
    X0 = torch.ones(X.shape[0], 1)
    X = torch.cat((X0, X), dim=1)
    Y_pred = torch.matmul(X, W)
    Y_pred[Y_pred > 0] = 1
    Y_pred[Y_pred < 0] = -1
    acc = torch.sum(Y_pred == Y) / X.shape[0]
    return acc

# 画出数据集和分类面
def plot_data(X, Y, W):
    cmap = ListedColormap(['#FF0000', '#0000FF'])
    plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0], cmap=cmap)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, 0.1), torch.arange(y_min, y_max, 0.1))
    Z = torch.matmul(torch.cat((torch.ones((xx.size()[0] * xx.size()[1], 1)), torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)), dim=1), W)
    Z = Z.reshape(xx.size())
    plt.contour(xx, yy, Z, levels=[0], colors='k')
    plt.show()

# 画出损失函数随epoch增加的变化曲线
def plot_loss(loss_list):
    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

# 产生数据集
X, Y = generate_data()

# 划分训练集和测试集
train_size = int(0.8 * X.shape[0])
X_train, Y_train = X[:train_size], Y[:train_size]
X_test, Y_test = X[train_size:], Y[train_size:]

# 使用广义逆求解最小误差平方和最佳解
g = generalized_inverse(X_train, Y_train)
print("使用广义逆求解最小误差平方和最佳解：")
print(g)
train_acc = accuracy(X_train, Y_train, g)
test_acc = accuracy(X_test, Y_test, g)
print("训练集分类正确率：", train_acc.item())
print("测试集分类正确率：", test_acc.item())
plot_data(X, Y, g)

# 使用梯度下降法求解最小误差平方和最佳解
W, loss_list = gradient_descent(X_train, Y_train)
print("使用梯度下降法求解最小误差平方和最佳解：")
print(W)
train_acc = accuracy(X_train, Y_train, W)
test_acc = accuracy(X_test, Y_test, W)
print("训练集分类正确率：", train_acc.item())
print("测试集分类正确率：", test_acc.item())
plot_data(X, Y, W)
plot_loss(loss_list)
