from PLA import pla
from Pocket import pocket

import numpy as np
import time
import matplotlib.pyplot as plt

# 生成数据集
def generate_data(num_data):
    # 生成“+1”类数据
    # mean1 = [-5, 0]
    mean1 = [1,0]
    cov1 = [[1, 0], [0, 1]]
    data1 = np.random.multivariate_normal(mean1, cov1, num_data)
    label1 = np.ones(num_data)

    # 生成“-1”类数据
    # mean2 = [0, 5]
    mean2 = [0,1]
    cov2 = [[1, 0], [0, 1]]
    data2 = np.random.multivariate_normal(mean2, cov2, num_data)
    label2 = -np.ones(num_data)

    # 合并数据
    data = np.concatenate((data1, data2), axis=0)
    label = np.concatenate((label1, label2), axis=0)

    # 打乱数据
    index = np.arange(num_data * 2)
    np.random.shuffle(index)
    data = data[index]#让data按照index的顺序重新排列
    label = label[index]

    return data, label, data1, data2


# 统计分类正确率
def accuracy(x, y, w):
    num_data = np.size(x, 0)
    dim_size = np.size(x, 1) + 1
    D = np.zeros((num_data, dim_size))

    for i in range(num_data):
        D[i] = [1, x[i][0], x[i][1]]

    error = 0
    for n in range(num_data):
        if np.sign(np.dot(w, D[n].T)) != y[n]:
            error = error + 1

    return 1 - error / num_data


# 生成数据集
num_data = 200
data, label, pos_data, neg_data = generate_data(num_data)

# 划分训练集和测试集
num_train = int(num_data * 0.8)
x_train = data[:num_train]
y_train = label[:num_train]
x_test = data[num_train:]
y_test = label[num_train:]

max_iter = 500#在第三问中，为避免分类陷入死循环，加入此行
#若需修改，需在pla中删去变量t
start_time = time.time()
w_pla = pla(x_train, y_train, max_iter)
end_time = time.time()
time_pla = end_time - start_time
print(time_pla)


max_iter = 500
start_time = time.time()
w_pocket = pocket(x_train, y_train, max_iter)
end_time = time.time()
time_pocket = end_time - start_time
print(time_pocket)


# 统计在训练集和测试集上的分类正确率
acc_train_pla = accuracy(x_train, y_train, w_pla)
acc_test_pla = accuracy(x_test, y_test, w_pla)
acc_train_pocket = accuracy(x_train, y_train, w_pocket)
acc_test_pocket = accuracy(x_test, y_test, w_pocket)
print("PLA:")
print('     train_acc:',acc_train_pla)
print('     test_acc:',acc_test_pla)
print("Pocket:")
print('     train_acc:',acc_train_pocket)
print('     train_acc:',acc_test_pocket)

# 画出数据集和分类面
def plot_data_and_boundary(pos, neg, w):
    # 画出数据集
    plt.scatter(pos[:,0], pos[:,1], c='r', marker='o')
    plt.scatter(neg[:,0], neg[:,1], c='b', marker='x')

    # 画出分类面
    x1 = np.linspace(-7, 7, 100)
    x2 = (-w[0] - w[1] * x1) / w[2]
    plt.plot(x1, x2)

    plt.show()

# 画出数据集和分类面
plot_data_and_boundary(pos_data, neg_data, w_pla.reshape(-1))
plot_data_and_boundary(pos_data, neg_data, w_pocket.reshape(-1))


