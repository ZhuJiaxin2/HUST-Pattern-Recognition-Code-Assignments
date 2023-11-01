import numpy as np
import matplotlib.pyplot as plt

# 定义单变量函数
def f(x):
    return x * np.cos(0.25 * np.pi * x)

# 定义梯度函数
def grad(x):
    return np.cos(0.25 * np.pi * x) - 0.25 * np.pi * x * np.sin(0.25 * np.pi * x)

# 定义梯度下降法
def gradient_descent(x, eta=0.4, epochs=10):
    x_list = [x]
    y_list = [f(x)]
    for i in range(epochs):
        x -= eta * grad(x)
        x_list.append(x)
        y_list.append(f(x))
    return x_list, y_list

# 定义随机梯度下降法
def stochastic_gradient_descent(x, eta=0.4, epochs=10):
    x_list = [x]
    y_list = [f(x)]
    for i in range(epochs):
        x -= eta * grad(x + np.random.normal(0, 1))
        x_list.append(x)
        y_list.append(f(x))
    return x_list, y_list

# 定义Adagrad算法
def adagrad(x, eta=0.4, epochs=10):
    x_list = [x]
    y_list = [f(x)]
    G = 0
    for i in range(epochs):
        g = grad(x)
        G += g ** 2
        x -= eta / np.sqrt(G + 1e-6) * g
        x_list.append(x)
        y_list.append(f(x))
    return x_list, y_list

# 定义RMSProp算法
def rmsprop(x, eta=0.4, epochs=10):
    x_list = [x]
    y_list = [f(x)]
    G = 0
    alpha = 0.9
    for i in range(epochs):
        g = grad(x)
        G = alpha * G + (1 - alpha) * g ** 2
        x -= eta / np.sqrt(G + 1e-6) * g
        x_list.append(x)
        y_list.append(f(x))
    return x_list, y_list

# 定义动量法
def momentum(x, eta=0.4, epochs=10):
    x_list = [x]
    y_list = [f(x)]
    v = 0
    lamda = 0.9
    for i in range(epochs):
        g = grad(x)
        v = lamda * v - eta * g
        x += v
        x_list.append(x)
        y_list.append(f(x))
    return x_list, y_list

# 定义Adam算法
def adam(x, eta=0.4, epochs=10):
    x_list = [x]
    y_list = [f(x)]
    m = 0
    v = 0
    beta1 = 0.99
    beta2 = 0.999
    for i in range(epochs):
        g = grad(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** (i + 1))
        v_hat = v / (1 - beta2 ** (i + 1))
        x -= eta / (np.sqrt(v_hat) + 1e-6) * m_hat
        x_list.append(x)
        y_list.append(f(x))
    return x_list, y_list

def draw_plot(x_init, eta, epochs, label):
    if label == 'gd':
        x_list, y_list = gradient_descent(x_init, eta, epochs)
        plt.plot(x_list, y_list, label='gradient descent')
    elif label =='sgd':
        x_list, y_list = stochastic_gradient_descent(x_init, eta, epochs)
        plt.plot(x_list, y_list, label='stochastic gradient descent')
    elif label =='ada':
        x_list, y_list = adagrad(x_init, eta, epochs)
        plt.plot(x_list, y_list, label='Adagrad')
    elif label =='rms':
        x_list, y_list = rmsprop(x_init, eta, epochs)
        plt.plot(x_list, y_list, label='RMSProp')
    elif label =='mom':
        x_list, y_list = momentum(x_init, eta, epochs)
        plt.plot(x_list, y_list, label='momentum')
    elif label =='adam':
        x_list, y_list = adam(x_init, eta, epochs)
        plt.plot(x_list, y_list, label='Adam')
    plt.legend()
    plt.show()


# 画出函数图像和各算法的变化情况
x = np.arange(-4, 4, 0.1)
y = f(x)
plt.plot(x, y, label='function')
x_init = -4
eta = 0.4
epochs = 50
label = 'adam'
draw_plot(x_init, eta, epochs, label)

