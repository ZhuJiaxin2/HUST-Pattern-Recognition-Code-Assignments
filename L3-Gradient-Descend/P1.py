import torch

# 使用广义逆求解最小误差平方和最佳解
def generalized_inverse(X, Y):
    X0 = torch.ones(X.shape[0], 1)
    X = torch.cat((X0, X), dim=1)
    X_1 = torch.inverse(torch.matmul(X.T, X))
    X_plus = torch.matmul(X_1, X.T)
    g = torch.matmul(X_plus, Y)
    return g

# 使用梯度下降法求解最小误差平方和最佳解
def gradient_descent(X, Y, eta=0.1, epochs=10000):
    X0 = torch.ones(X.shape[0], 1)
    X = torch.cat((X0, X), dim=1)

    W = torch.zeros(X.shape[1], 1)
    t = 0
    while 1:
        sum_grad = 0
        for i in range(X.shape[0]):
            x = torch.unsqueeze(X[i], dim=1)
            Y_pred = torch.matmul(W.T, x)
            gradient = (Y_pred - Y[i]) * x
            sum_grad += gradient
        ave_grad = 2 * sum_grad / X.shape[0]
        t +=1
        if torch.norm(ave_grad) == 0  or t == epochs:
            break
        W -= eta * ave_grad
    return W

# 测试代码
X = torch.tensor([  [0.2, 0.7], 
                    [0.3, 0.3], 
                    [0.4, 0.5], 
                    [0.6, 0.5], 
                    [0.1, 0.4], 
                    [0.4, 0.6], 
                    [0.6, 0.2], 
                    [0.7, 0.4],
                    [0.8, 0.6],
                    [0.7, 0.5]])
Y = torch.tensor([1,1,1,1,1,-1,-1,-1,-1,-1], dtype=torch.float).reshape(-1,1)

g1 = generalized_inverse(X, Y)
print("使用广义逆求解最小误差平方和最佳解：")
print(g1)

g2 = gradient_descent(X, Y)
print("使用梯度下降法求解最小误差平方和最佳解：")
print(g2)