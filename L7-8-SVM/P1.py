import numpy as np
import cvxopt

def primal_svm(X, Y):
    n_samples, n_features = X.shape

    # 构造 QP 问题的矩阵
    Q = np.eye(n_features+1)
    Q[0][0] = 0
    Q = cvxopt.matrix(Q)
    
    p = cvxopt.matrix(np.zeros((n_features+1, 1)))

    X_new = np.concatenate((np.ones((n_samples, 1)), X), axis=1)
    A = cvxopt.matrix((-Y * X_new))

    c = cvxopt.matrix(-np.ones((n_samples, 1)))

    # 求解 QP 问题
    solution = cvxopt.solvers.qp(Q, p, A, c)

    # 提取模型参数
    W = np.array(solution['x'][1:n_features+1]).reshape(n_features, 1)
    b = solution['x'][0]

    return W, b



def dual_svm(X, Y):
    n_samples, n_features = X.shape

    # 构造 QP 问题的矩阵
    K = np.dot(X, X.T)
    Q = cvxopt.matrix(np.outer(Y, Y) * K)#这里*是逐项相乘
    
    p = cvxopt.matrix(-np.ones((n_samples,1)))

    A = cvxopt.matrix(-np.eye(n_samples))

    c = cvxopt.matrix(np.zeros((n_samples,1)))

    r = cvxopt.matrix(Y.reshape(1,-1))

    v = cvxopt.matrix([0.0])

    # 求解 QP 问题
    solution = cvxopt.solvers.qp(Q, p, A, c, r, v)

    # 提取拉格朗日乘子
    alpha = np.array(solution['x']).reshape(-1)

    W = np.sum(alpha.reshape(-1,1) * Y * X, axis=0).reshape(-1, 1)

    sv_idx = np.where(alpha > 1e-5)[0]
    sv_alpha = alpha[sv_idx]

    sv_x = X[sv_idx[0]]
    sv_y = Y[sv_idx[0]]
    b = sv_y - np.dot(W.T, sv_x)

    return W, b, sv_idx, sv_alpha



def kernel_svm(X, Y, kernel):
    n_samples, n_features = X.shape

    # 计算核矩阵
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = kernel(X[i], X[j])

    # 构造 QP 问题的矩阵
    Q = cvxopt.matrix(np.outer(Y, Y) * K)

    p = cvxopt.matrix(-np.ones((n_samples,1)))

    A = cvxopt.matrix(-np.eye(n_samples))

    c = cvxopt.matrix(np.zeros((n_samples,1)))

    r = cvxopt.matrix(Y.reshape(1, -1))

    v = cvxopt.matrix([0.0])

    # 求解 QP 问题
    solution = cvxopt.solvers.qp(Q, p, A, c, r, v)

    # 提取拉格朗日乘子
    alpha = np.array(solution['x']).reshape(-1)
    print(alpha)

    W = np.sum(alpha.reshape(-1,1) * Y * X, axis=0).reshape(-1, 1)

    if kernel == phi_gauss:
        sv_idx = np.where(alpha > 1e-1)[0]
    elif kernel == phi:
        sv_idx = np.where(alpha > 1e-5)[0]

    sv_alpha = alpha[sv_idx]

    sv_x = X[sv_idx[0]]
    sv_y = Y[sv_idx[0]]
    b = sv_y - np.dot(W.T, sv_x)

    return W, b, sv_idx, sv_alpha



def phi(x1, x2, eta=1, gamma=1, q=4):
    return (eta + gamma * np.dot(x1, x2.T))**q

def phi_gauss(x1, x2, gamma=1):
    return np.exp(-gamma * np.linalg.norm(x1.T - x2.T)**2)



# X = np.array([[0.2, 0.7], 
#             [0.3, 0.3], 
#             [0.4, 0.5], 
#             [0.6, 0.5], 
#             [0.1, 0.4], 
#             [0.4, 0.6], 
#             [0.6, 0.2], 
#             [0.7, 0.4],
#             [0.8, 0.6],
#             [0.7, 0.5]], dtype=np.double)
# Y = np.array([1.,1.,1.,1.,1.,-1.,-1.,-1.,-1.,-1.], dtype=np.double).reshape(-1,1)

# W1, b1 = primal_svm(X, Y)

# W2, b2, sv2, sv2_y, sv2_alpha = dual_svm(X, Y)

# W3, b3, sv3, sv3_alpha = kernel_svm(X, Y, phi_gauss)