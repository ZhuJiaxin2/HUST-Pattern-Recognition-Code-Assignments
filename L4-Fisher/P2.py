import torch
import matplotlib.pyplot as plt
from math import sqrt

# 数据生成
m1 = torch.tensor([-5., 0.])
m2 = torch.tensor([0., 5.])
s1 = torch.eye(2, dtype=torch.float32)
s2 = torch.eye(2, dtype=torch.float32)
X1 = torch.distributions.multivariate_normal.MultivariateNormal(m1, s1).sample((200,))
X2 = torch.distributions.multivariate_normal.MultivariateNormal(m2, s2).sample((200,))
Y1 = torch.ones((200,1))
Y2 = -torch.ones((200,1))
X_train = torch.cat([X1[:160], X2[:160]])
Y_train = torch.cat([Y1[:160], Y2[:160]])
X_test = torch.cat([X1[160:], X2[160:]])
Y_test = torch.cat([Y1[160:], Y2[160:]])

# Fisher线性判别算法
X_pos = X_train[Y_train.squeeze() == 1]
X_neg = X_train[Y_train.squeeze() == -1]
mean_pos = torch.mean(X_pos, dim=0)
mean_neg = torch.mean(X_neg, dim=0)
S_pos = torch.matmul((X_pos - mean_pos).T, (X_pos - mean_pos))
S_neg = torch.matmul((X_neg - mean_neg).T, (X_neg - mean_neg))
Sw = S_pos + S_neg
w = torch.matmul(torch.inverse(Sw), (mean_pos - mean_neg).reshape(-1, 1))
print(w)
threshold = torch.matmul(w.T, (mean_pos + mean_neg).reshape(-1, 1)) / 2
print(threshold)
Y_pred_train = torch.matmul(X_train, w)
Y_pred_test = torch.matmul(X_test, w)

# 计算分类正确率
Y_pred_train[Y_pred_train >= threshold] = 1
Y_pred_train[Y_pred_train < threshold] = -1
train_accuracy = torch.mean((Y_pred_train == Y_train).float())

Y_pred_test[Y_pred_test >= threshold] = 1
Y_pred_test[Y_pred_test < threshold] = -1
test_accuracy = torch.mean((Y_pred_test == Y_test).float())
print("Train accuracy: {:.2f}%".format(train_accuracy * 100))
print("Test accuracy: {:.2f}%".format(test_accuracy * 100))

# 画出数据集、最佳投影向量和分类阈值
plt.scatter(X1[:, 0], X1[:, 1], label="+1")
plt.scatter(X2[:, 0], X2[:, 1], label="-1")
plt.plot([-10, 10], [(-10 * w[1] / w[0]).item(), (10 * w[1] / w[0]).item()], label="Best projection vector")
x_thres = (mean_pos + mean_neg)/ 2
# plt.plot(x_thres[0], x_thres[1], marker='*', markersize=10, color="red", label="Classification threshold")
plt.plot((w[1]**2 * x_thres[0] - w[0] * w[1] * x_thres[1]) / sqrt(w[0]**2 + w[1]**2), (w[0]**2 * x_thres[1] -w[0] * w[1] * x_thres[0]) / sqrt(w[0]**2 + w[1]**2), marker='*', markersize=10, color="green", label="threshold")
plt.legend()
plt.show()