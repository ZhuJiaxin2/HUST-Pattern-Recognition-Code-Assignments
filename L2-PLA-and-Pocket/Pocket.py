import numpy as np
import random
#为了方便键入，我写的输入x是行向量堆叠的形式，因此我的输出w和w_hat也写成了行向量
#但是实际上x和w_hat都应该是列向量

def pocket(x, y, max_iter):
    num_data=np.size(x,0)
    dim_size=np.size(x,1)+1
    D=np.zeros((num_data,dim_size))

    for i in range(num_data):
        D[i]=[1, x[i][0], x[i][1]]

    w, w_hat = init_w(dim_size)

    wrong = [0,num_data]#w,w_hat
    for t in range(max_iter):
        wrong[0]=0
        wr_idx=[]
        for n in range(num_data):
            if sign(np.dot(w,D[n].T))!=y[n]:
                wrong[0] = wrong[0]+1
                wr_idx.append(n)

        # print('w:',wrong[0])
        # print('w_hat:',wrong[1])
        # print(w)
                
        if wrong[0]<=wrong[1]:
            w_hat=w
            wrong[1]=wrong[0]

        try:
            rand_n = random.choice(wr_idx)
        except:
            pass
        
        w = w + y[rand_n]*D[rand_n]
        
    return w_hat


def init_w(dim_size):
    W=np.zeros((1,dim_size))
    W_hat=W
    return W,W_hat


def sign(p):
    if p>0:
        return 1
    if p==0:
        return 0
    if p<0:
        return -1


# x_list=np.array([[0.2,0.7],
#                 [0.3,0.3],
#                 [0.4,0.5],
#                 [0.6,0.5],
#                 [0.1,0.4],
#                 [0.4,0.6],
#                 [0.6,0.2],
#                 [0.7,0.4],
#                 [0.8,0.6],
#                 [0.7,0.5]])
# y_list=np.array([1,1,1,1,1,-1,-1,-1,-1,-1])

# max_iter=20

# W=pocket(x_list,y_list,max_iter)
# print(W.T)




