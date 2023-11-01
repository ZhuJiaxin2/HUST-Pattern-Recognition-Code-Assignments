from P1 import primal_svm, dual_svm, kernel_svm, phi, phi_gauss
from P2 import pred_kernel, accuracy

import numpy as np


def data(p):
    if p == 1:                        
        x1=[[119.28,26.08],#福州
            [121.31,25.03],#台北
            [121.47,31.23],#上海
            [118.06,24.27],#厦门
            [121.46,39.04],#大连
            [122.10,37.50],#威海
            [124.23,40.07]]#丹东
        
        x2=[[129.87,32.75],#长崎
            [130.33,31.36],#鹿儿岛
            [131.42,31.91],#宫崎
            [130.24,33.35],#福冈
            [133.33,15.43],#鸟取
            [138.38,34.98],#静冈
            [140.47,36.37]]#水户
         
    elif  p == 2:                  
        x1=[[119.28,26.08],#福州
            [121.31,25.03],#台北
            [121.47,31.23],#上海
            [118.06,24.27],#厦门
            [113.53,29.58],#武汉
            [104.06,30.67],#成都
            [116.25,39.54],#北京
            [121.46,39.04],#大连
            [122.10,37.50],#威海
            [124.23,40.07]]#丹东
        
        x2=[[129.87,32.75],#长崎
            [130.33,31.36],#鹿儿岛
            [131.42,31.91],#宫崎
            [130.24,33.35],#福冈
            [136.54,35.10],#名古屋
            [132.27,34.24],#广岛
            [139.46,35.42],#东京
            [133.33,15.43],#鸟取
            [138.38,34.98],#静冈
            [140.47,36.37]]#水户
        
    x = np.concatenate((np.array(x1),np.array(x2)),axis=0) 

    y1 = np.ones((np.array(x1).shape[0],1))
    y2 = -y1
    y = np.concatenate((np.array(y1), np.array(y2)), axis=0)
        
    return x, y


x_train, y_train = data(2)
x_test = np.array([123.28, 25.45])

W1, b1 = primal_svm(x_train, y_train)
acc_train_1 = accuracy(x_train, y_train, W1, b1)
y_pred = np.sign(np.dot(W1.T, x_test.T) + b1)
print("Primal SVM:")
print('     分类面:', W1)
print('     截距:', b1)
print('     train_acc:',acc_train_1)
print('     钓鱼岛的预测值为:',y_pred)


W2, b2, sv_idx_2, sv_alpha_2 = dual_svm(x_train, y_train)
acc_train_2 = accuracy(x_train, y_train, W2, b2)
y_pred = np.sign(np.dot(W2.T, x_test.T) + b2)
print("Dual SVM:")
print('     分类面:', W2)
print('     截距:', b2)
print('     train_acc:',acc_train_2)
print('     SV:', x_train[sv_idx_2])
print('     钓鱼岛的预测值为:',y_pred)


# kernel = phi
# W3, b3, sv_idx_3, sv_alpha_3 = kernel_svm(x_train, y_train, kernel)
# acc_train_3 = accuracy(x_train, y_train, W3, b3, x_train[sv_idx_3], y_train[sv_idx_3], sv_alpha_3, kernel)
# y_pred = pred_kernel(x_train[sv_idx_3], x_test, y_train[sv_idx_3], sv_alpha_3, b3, kernel)
# print("Kernel SVM:")
# print('     分类面:', W3)
# print('     截距:', b3)
# print('     train_acc:',acc_train_3)
# print('     SV:', x_train[sv_idx_3])
# print('     钓鱼岛的预测值为:',y_pred)


