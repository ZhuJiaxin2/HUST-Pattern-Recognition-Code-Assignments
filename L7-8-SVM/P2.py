from P1 import primal_svm, dual_svm, kernel_svm, phi, phi_gauss

import numpy as np
import matplotlib.pyplot as plt

# 生成数据集
def generate_data(num_data):
    # 生成“+1”类数据
    mean1 = [-5., 0.]
    # mean1 = [3., 0.]
    cov1 = [[1., 0.], [0., 1.]]
    data1 = np.random.multivariate_normal(mean1, cov1, num_data)
    label1 = np.ones((num_data,1))

    # 生成“-1”类数据
    mean2 = [0., 5.]
    # mean2 = [0. ,3. ]
    cov2 = [[1., 0.], [0., 1.]]
    data2 = np.random.multivariate_normal(mean2, cov2, num_data)
    label2 = -np.ones((num_data,1))

    # 合并数据
    data = np.concatenate((data1, data2), axis=0)
    label = np.concatenate((label1, label2), axis=0)

    # 打乱数据
    index = np.arange(num_data * 2)
    np.random.shuffle(index)
    data = data[index]#让data按照index的顺序重新排列
    label = label[index]

    return data, label, data1, data2

def train_test_split(data, label, num_data):
    # 划分训练集和测试集
    num_train = int(num_data * 0.8)
    x_train = data[:num_train]
    y_train = label[:num_train]
    x_test = data[num_train:]
    y_test = label[num_train:]

    return x_train, y_train, x_test, y_test

# 统计分类正确率
def accuracy(X, Y, W, b, x_train_sv=None, y_train_sv=None, alpha=None, kernel=None):
    num_data = np.size(X, 0)

    #kernel svm
    if kernel is not None:
        x_test = X
        y_pred = pred_kernel(x_train_sv, x_test, y_train_sv, alpha, b, kernel)
    else:
        y_pred = np.sign(np.dot(X, W) + b*np.ones((num_data)))
    acc = np.mean(y_pred == Y)
    return acc


def pred_kernel(x_train_sv, x_test, y_train_sv, alpha, b, kernel=None):#x_train_sv和x_test都是矩阵
    n = x_train_sv.shape[0]

    if kernel is not None:
        gsvm = []
        for i in range(x_test.shape[0]):
            temp = 0
            for j in range(x_train_sv.shape[0]):
                temp += alpha.reshape(-1,1)[j] * y_train_sv[j] * kernel(x_train_sv[j], x_test[i])
            gsvm.append(np.sum(temp, axis=0))

    return np.sign(np.array(gsvm).reshape(-1,1) + b)


# 画出数据集和分类面
def plot_data_and_boundary(pos, neg, w, b, x_train_sv=None, y_train_sv=None, alpha=None, kernel=None):
    # 画出数据集
    plt.scatter(pos[:,0], pos[:,1], c='r', marker='o')
    plt.scatter(neg[:,0], neg[:,1], c='b', marker='x')
    
    # dual和kernel用直线分开的代码
    # if sv is not None and len(sv)>0:
    #     #画出支撑向量
    #     plt.scatter(sv[:,0], sv[:,1], c='g', marker='*')

    #     for i in range(len(sv)):
    #         b_sv = -w[1] * sv[i][1] - w[0] * sv[i][0]
    #         x2_sv = (-b_sv - w[0] * x1) / w[1]
    #         plt.plot(x1, x2_sv, c='g', linestyle='--')

    # kernel svm用曲线分开的代码
    if kernel is not None:
        plt.scatter(x_train_sv[:,0], x_train_sv[:,1], c='g', marker='*')

        x1, x2 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
        x_test = np.c_[x1.ravel(), x2.ravel()]
        Z = pred_kernel(x_train_sv, x_test, y_train_sv, alpha, b, kernel)
        Z = Z.reshape(x1.shape)

        # # plt.contourf(x1, x2, Z, alpha=0.5)
        plt.contour(x1, x2, Z, levels=[0], colors='yellow', linewidths=2)

        for i in range(x_train_sv.shape[0]):
            temp = 0
            for j in range(x_train_sv.shape[0]):
                temp += alpha.reshape(-1,1)[j] * y_train_sv[j] * kernel(x_train_sv[j], x_train_sv[i])
            gsvm_i = np.sum(temp, axis=0)
            b_sv = y_train_sv[i] - gsvm_i

            Z0 = pred_kernel(x_train_sv, x_test, y_train_sv, alpha, b_sv, kernel)

            Z0 = Z0.reshape(x1.shape)

            # plt.contourf(x1, x2, Z0, levels=[-1,0], colors='yellow', alpha=0.5)
            # plt.contourf(x1, x2, Z0, levels=[-1,0,1], alpha=0.1)
            plt.contour(x1, x2, Z0, levels=[-1,1], colors='green', linewidths=0.5)

        plt.show()

        return
    
    #dual svm用直线分开代码
    if alpha is not None:
        # 画出分类面
        x1 = np.linspace(-7, 7, 100)
        x2 = (-b - w[0] * x1) / w[1]
        plt.plot(x1, x2)

        #画出支撑向量
        plt.scatter(x_train_sv[:,0], x_train_sv[:,1], c='g', marker='*')

        for i in range(len(x_train_sv)):
            b_sv = -w[1] * x_train_sv[i][1] - w[0] * x_train_sv[i][0]
            x2_sv = (-b_sv - w[0] * x1) / w[1]
            plt.plot(x1, x2_sv, c='g', linestyle='--')

        plt.show()

        return
        

    #primal svm用直线分开代码
    else:
        # 画出分类面
        x1 = np.linspace(-7, 7, 100)
        x2 = (-b - w[0] * x1) / w[1]
        plt.plot(x1, x2)

        plt.show()

        return


# num_data = 200
# data, label, pos_data, neg_data = generate_data(num_data)

# x_train, y_train, x_test, y_test = train_test_split(data, label, num_data)


# W1, b1 = primal_svm(x_train, y_train)

# kernel = phi
# W2, b2, sv_idx_2, sv_alpha_2 = dual_svm(x_train, y_train)

# W3, b3, sv_idx_3, sv_alpha_3 = kernel_svm(x_train, y_train, kernel)


# # 统计在训练集和测试集上的分类正确率
# acc_train_1 = accuracy(x_train, y_train, W1, b1)
# acc_test_1 = accuracy(x_test, y_test, W1, b1)
# print("Primal SVM:")
# print('     分类面:', W1)
# print('     截距:', b1)
# print('     train_acc:',acc_train_1)
# print('     test_acc:',acc_test_1)

# acc_train_2 = accuracy(x_train, y_train, W2, b2)
# acc_test_2 = accuracy(x_test, y_test, W2, b2)
# print("Dual SVM:")
# print('     分类面:', W2)
# print('     截距:', b2)
# print('     train_acc:',acc_train_2)
# print('     test_acc:',acc_test_2)
# print('     SV:', x_train[sv_idx_2])

# acc_train_3 = accuracy(x_train, y_train, W3, b3, x_train[sv_idx_3], y_train[sv_idx_3], sv_alpha_3, kernel)
# acc_test_3 = accuracy(x_test, y_test, W3, b3, x_train[sv_idx_3], y_train[sv_idx_3], sv_alpha_3, kernel)
# print("Kernel SVM:")
# print('     分类面:', W3)
# print('     截距:', b3)
# print('     train_acc:',acc_train_3)
# print('     test_acc:',acc_test_3)
# print('     SV:', x_train[sv_idx_3])

# 画出数据集和分类面
# plot_data_and_boundary(pos_data, neg_data, W1.reshape(-1), b1)
# # # plot_data_and_boundary(pos_data, neg_data, W2.reshape(-1), b2, x_train[sv_idx_2])
# plot_data_and_boundary(pos_data, neg_data, W2.reshape(-1), b2, x_train[sv_idx_2], None, sv_alpha_2)
# plot_data_and_boundary(pos_data, neg_data, W3.reshape(-1), b3, x_train[sv_idx_3], y_train[sv_idx_3], sv_alpha_3, kernel)


