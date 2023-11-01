import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')
    
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

def initialize_weights(dim):
    np.random.seed(42)
    return np.random.normal(0, 0.01, (dim, 10))

def softmax(z):
    e_z = np.exp(z)
    return e_z / np.expand_dims(e_z.sum(axis=1), axis=1)

def compute_loss_and_gradients(X, y, W):
    num_train = X.shape[0]
    num_classes = 10
    scores = np.dot(X, W)
    # print(scores)
    probs_onehot = softmax(scores)
    # print(probs_onehot)
    loss = -np.sum(np.log(probs_onehot[np.arange(num_train), y])) / num_train # probs_onehot[:, y]是错的，这样会返回一个新的二维数组，其中第一行包含了probs_onehot的第y[0]个列
    y_onehot = np.eye(num_classes)[y]
    # print(probs)
    dW = np.dot(X.T, probs_onehot - y_onehot) / num_train
    return loss, dW

def predict(X, W):
    scores = np.dot(X, W)
    return np.argmax(scores, axis=1)

def compute_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def train(X_train, y_train, X_test, y_test, W, batch_size, eta, epochs):
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    num_train = X_train.shape[0]
    num_batches = num_train // batch_size

    for epoch in range(epochs):
        X, y = shuffle(X_train, y_train)
        for i in range(num_batches):
            X_batch = X[i*batch_size:(i+1)*batch_size]
            y_batch = y[i*batch_size:(i+1)*batch_size]
            loss, dW = compute_loss_and_gradients(X_batch, y_batch, W)
            W -= eta * dW
            train_losses.append(loss)

        y_train_pred = predict(X_train, W)
        y_train_true = y_train
        train_accuracy = compute_accuracy(y_train_pred, y_train_true)
        train_accuracies.append(train_accuracy)

        y_test_pred = predict(X_test, W)
        y_test_true = y_test
        test_accuracy = compute_accuracy(y_test_pred, y_test_true)
        test_accuracies.append(test_accuracy)

    return W, train_losses, train_accuracies, test_accuracies

def plot(train_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training loss')
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies)
    plt.title('Training accuracy')
    plt.subplot(1, 3, 3)
    plt.plot(test_accuracies)
    plt.title('Testing accuracy')
    plt.show()

# Load MNIST data
train_set = load_mnist('./L9_data/MNIST', kind='train')
test_set = load_mnist('./L9_data/MNIST', kind='t10k')

# Initialize weights
W = initialize_weights(784)

# Train the model
W, train_losses, train_accuracies, test_accuracies = train(train_set[0]/255.0, train_set[1], test_set[0]/255.0, test_set[1], W, 256, 0.005, 10)

# Plot the results
plot(train_losses, train_accuracies, test_accuracies)

# Predict on 10 random test samples
random_indices = np.random.choice(test_set[0].shape[0], 10, replace=False)
random_test_samples = test_set[0][random_indices]/255.0
random_test_labels = test_set[1][random_indices]
random_test_predictions = predict(random_test_samples, W)

print('Random test samples predictions: ', random_test_predictions)
print('Random test samples true labels: ', random_test_labels)