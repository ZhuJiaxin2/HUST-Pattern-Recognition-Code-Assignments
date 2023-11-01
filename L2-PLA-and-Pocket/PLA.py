import numpy as np
#为了方便键入，我写的输入x是行向量堆叠的形式，因此我的输出w也写成了行向量
#但是实际上x和w都应该是列向量

def pla(x, y, max_iter):#第三问中，为了防止线性不可分数据令pla陷入死循环，加入max_iter
    num_data=np.size(x,0)
    dim_size=np.size(x,1)+1
    D=np.zeros((num_data,dim_size))

    for i in range(num_data):
        D[i]=[1, x[i][0], x[i][1]]

    w = init_w(dim_size)

    wrong = 1
    t = 0
    while wrong:
        wrong = 0
        for n in range(num_data):
            if sign(np.dot(w,D[n].T))!=y[n]:
                w = w + y[n]*D[n]
                wrong = wrong+1
                t +=1
        if t >= max_iter:#避免线性不可分数据进入死循环
            break
        # print('w:',w)
        # print('wrong:', wrong)
        
    return w


def init_w(dim_size):
    W=np.zeros((1,dim_size))
    return W


def sign(p):
    if p>0:
        return 1
    if p==0:
        return 0
    if p<0:
        return -1


# x_list=np.array([[3,3],
#                  [4,3],
#                  [1,1]])
# y_list=np.array([1,1,-1])

# W=pla(x_list,y_list)
# print(W.T)




