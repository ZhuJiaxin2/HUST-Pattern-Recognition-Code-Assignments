import torch
import math
import random

def LR_sgd(X, Y, eta=0.1, batch_size=1, max_iter=1000):
    X0 = torch.ones(X.shape[0], 1)
    X = torch.cat((X0, X), dim=1)
    W = torch.zeros(X.shape[1], 1)
    t = 0
    loss_list = []
    batch_num = int(X.shape[0]/batch_size)

    while 1:
        sum_grad = 0
        for n in range(batch_num):
            grad_list = []
            min_idx = int(n * batch_size)
            max_idx = int((n+1) * batch_size)
            for i in range(min_idx, max_idx):
                x = torch.unsqueeze(X[i], dim=1)
                y = Y[i].item()
                gradient = sigmoid(-y * W.T @ x) * (-y * x)
                grad_list.append(gradient)
            rand_idx = random.randint(0, batch_size-1)#此处左右都闭
            sgd_grad = grad_list[rand_idx]
            sum_grad += sgd_grad
        ave_grad = sum_grad / batch_num
        W -= eta * ave_grad
        t+=1
        loss = loss_func(X, Y, W)
        loss_list.append(loss)
        if torch.norm(ave_grad) == 0 or t == max_iter:
            break
    
    return W, loss_list

def sigmoid(x):
     return 1/(1 + math.exp(-x))

def loss_func(X, Y, W):
    sum = 0
    for i in range(X.shape[0]):
        x = torch.unsqueeze(X[i], dim=1)
        y = Y[i].item()
        loss = math.log(1+ math.exp(-y* W.T @ x))
        sum +=loss
    loss = sum / X.shape[0]
    return loss


# # 测试代码
# X = torch.tensor([  [0.2, 0.7], 
#                     [0.3, 0.3], 
#                     [0.4, 0.5], 
#                     [0.6, 0.5], 
#                     [0.1, 0.4], 
#                     [0.4, 0.6], 
#                     [0.6, 0.2], 
#                     [0.7, 0.4],
#                     [0.8, 0.6],
#                     [0.7, 0.5]])
# Y = torch.tensor([1,1,1,1,1,-1,-1,-1,-1,-1], dtype=torch.float).reshape(-1,1)

# W, loss_list = LR_sgd(X, Y)
# print("最佳解：")
# print(W)


