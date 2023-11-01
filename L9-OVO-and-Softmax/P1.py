import pandas as pd
import numpy as np

iris = pd.read_csv('./L9_data/Iris/iris.csv')

iris = pd.get_dummies(iris, columns=['Species'])

species = ['Species_setosa', 'Species_versicolor', 'Species_virginica']

for type in species:
    iris[type]=iris[type].map({False:-1,True:1})

train_set = pd.concat([iris[iris[species[0]]==1].sample(n=30),
                       iris[iris[species[1]]==1].sample(n=30),
                       iris[iris[species[2]]==1].sample(n=30)])
test_set = pd.concat([iris[iris[species[0]]==1].drop(train_set.index[:30]).sample(n=20),
                      iris[iris[species[1]]==1].drop(train_set.index[30:60]).sample(n=20),
                      iris[iris[species[2]]==1].drop(train_set.index[60:90]).sample(n=20)])


iris.to_csv('./iris_final.csv', index=False)


def pla(x, y, max_iter):
    num_data=np.size(x,0)
    dim_size=np.size(x,1)+1
    D=np.concatenate((np.ones((num_data,1)), x), axis=1)

    w = np.zeros((1,dim_size))

    wrong = 1
    t = 0
    while wrong:
        wrong = 0
        for n in range(num_data):
            if np.sign(np.dot(w, D[n].T)) != y[n]:
                w = w + y[n]*D[n]
                wrong = wrong+1
                t +=1
        if t >= max_iter:#避免线性不可分数据进入死循环
            break
        
    return np.expand_dims(w, axis=1)


def ovo_multiclass(train_set, test_set, epochs):
    weights = np.zeros((5,3))

    group = [[0, 1], [0, 2], [1, 2]]
    for i in range(3):
        y_k = train_set.iloc[30*group[i][0]:30+30*group[i][0],-3+group[i][0]]
        y_l = train_set.iloc[30*group[i][1]:30+30*group[i][1],-3+group[i][1]] * -1 ###这里要将另一类标签置为负数！！
        x_k = train_set.iloc[30*group[i][0]:30+30*group[i][0], 1:-3]
        x_l = train_set.iloc[30*group[i][1]:30+30*group[i][1], 1:-3]
        x = np.concatenate((x_k, x_l), axis=0)
        y = np.concatenate((y_k, y_l), axis=0)
        w_kl = pla(x, y, epochs)
        weights[:,i] = w_kl

    X = np.concatenate((np.ones((train_set.shape[0],1)), train_set.iloc[:,1:-3]), axis=1)
    y_pred = np.sign(np.dot(X, weights))
    # print(y_pred)
    y_score = tournament(y_pred)
    # print(y_score)
    y_pred = np.argmax(y_score, axis=1)#返回最大值的索引
    y_true = np.argmax(train_set.iloc[:, -3:].values, axis=1) ###这里.values的输出是012，看来iloc后索引值也变了
    train_acc = np.mean(y_pred == y_true)
    
    # Test
    X_test = np.concatenate((np.ones((test_set.shape[0],1)), test_set.iloc[:,1:-3]), axis=1)
    y_ptest = np.sign(np.dot(X_test, weights))
    test_score = tournament(y_ptest)
    y_ptest = np.argmax(test_score, axis=1)#返回最大值的索引
    y_ttest = np.argmax(test_set.iloc[:, -3:].values, axis=1)
    test_acc = np.mean(y_ptest == y_ttest)
    
    return weights, train_acc, test_acc


def tournament(y):
    #group = [[0, 1], [0, 2], [1, 2]]
    score = np.zeros((y.shape[0], 3))
    for i in range(y.shape[0]):
        if y[i][0] == 1:
            score[i][0] += 1
        else:
            score[i][1] += 1
        if y[i][1] == 1:
            score[i][0] += 1
        else:
            score[i][2] += 1
        if y[i][2] == 1:
            score[i][1] += 1
        else:
            score[i][2] += 1
    return score


def softmax(train_set, test_set, learning_rate, epochs):
    weights = np.zeros((5,3))
    
    x_train = np.concatenate((np.ones((train_set.shape[0],1)), train_set.iloc[:,1:-3]), axis=1)
    y_train = train_set.iloc[:, -3:]
    y_train = np.where(y_train == -1, 0, y_train)
    for epoch in range(epochs):
        y_pred = np.exp(np.dot(x_train, weights)) / np.expand_dims(np.sum(np.exp(np.dot(x_train, weights)), axis=1), axis=1)
        grad = (y_pred - y_train).T @ x_train
        weights -= learning_rate * grad.T
    
    train_pred = np.exp(np.dot(x_train, weights)) / np.expand_dims(np.sum(np.exp(np.dot(x_train, weights)), axis=1), axis=1)
    train_pred = np.argmax(train_pred, axis=1)
    train_true = np.argmax(y_train, axis=1)
    train_acc = np.mean(train_pred == train_true)

    x_test = np.concatenate((np.ones((test_set.shape[0],1)), test_set.iloc[:,1:-3]), axis=1)
    y_test = test_set.iloc[:, -3:]
    y_test = np.where(y_test == -1, 0, y_test)
    test_pred = np.exp(np.dot(x_test, weights)) / np.expand_dims(np.sum(np.exp(np.dot(x_test, weights)), axis=1), axis=1)
    test_pred = np.argmax(test_pred, axis=1)
    test_true = np.argmax(y_test, axis=1)
    test_acc = np.mean(test_pred == test_true)
    
    return weights, train_acc, test_acc


# Test the OVO multi-class classifier algorithm
w_ovo, train_acc_ovo, test_acc_ovo = ovo_multiclass(train_set, test_set, 10000)
print("OVO Train Accuracy:", train_acc_ovo)
print("OVO Test Accuracy:", test_acc_ovo)

# Test the Softmax algorithm
w_sm, train_acc_sm, test_acc_sm = softmax(train_set, test_set, 0.01, 10000)
print("Softmax Train Accuracy:", train_acc_sm)
print("Softmax Test Accuracy:", test_acc_sm)