import torch
from torch.utils.data import TensorDataset, DataLoader

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
Y = torch.tensor([1,1,1,1,1,-1,-1,-1,-1,-1], dtype=torch.float).squeeze()

dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=len(dataset))

# Fisher线性判别算法
X_pos = X[Y == 1]
X_neg = X[Y == -1]
mean_pos = torch.mean(X_pos, dim=0)#行向量（只是为了方便和大X进行矩阵计算，实际单样本应该是列向量）
mean_neg = torch.mean(X_neg, dim=0)
S_pos = torch.matmul((X_pos - mean_pos).T, (X_pos - mean_pos))#mean_pos有广播机制，逐行复制
S_neg = torch.matmul((X_neg - mean_neg).T, (X_neg - mean_neg))
Sw = S_pos + S_neg
w = torch.matmul(torch.inverse(Sw), (mean_pos - mean_neg).reshape(-1, 1))
s_pie = torch.matmul(w.T, (mean_pos + mean_neg).reshape(-1, 1)) / 2

# 模型训练和预测
for X_batch, Y_batch in dataloader:
    Y_pred = torch.matmul(X_batch, w)
    Y_pred[Y_pred > s_pie] = 1
    Y_pred[Y_pred <= s_pie] = -1
    accuracy = torch.mean((Y_pred == Y_batch).float())
    print("Accuracy: {:.2f}%".format(accuracy * 100))
