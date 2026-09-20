import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class PerceptronTorch(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None
        self.errors_ = []

    def sigmoid(self, x):
        return 1.0 / (1.0 + torch.exp(-x))

    def weighted_sum(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        return X @ self.w_[1:] + self.w_[0]

    def predict(self, X):
        scores = self.sigmoid(self.weighted_sum(X))
        return torch.where(scores >= 0.5, torch.tensor(1.0), torch.tensor(0.0))

    def predict_single(self, features):
        x = torch.as_tensor(features, dtype=torch.float32)
        probability = self.sigmoid(x @ self.w_[1:] + self.w_[0])
        return int(torch.where(probability >= 0.5, torch.tensor(1.0), torch.tensor(0.0)).item())

    def fit(self, X, y):
        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)

        self.w_ = torch.zeros(1 + X.shape[1], dtype=torch.float32)
        self.errors_ = []

        print("Weights:", self.w_.detach().numpy())

        for _ in range(self.n_iter):
            error = 0

            for xi, target in zip(X, y):
                y_pred = self.sigmoid(self.weighted_sum(xi))
                update = self.eta * (target - y_pred)

                self.w_[1:] = self.w_[1:] + update * xi
                self.w_[0] = self.w_[0] + update

                pred_label = 1.0 if y_pred >= 0.5 else 0.0
                if pred_label != target:
                    error += 1

                print("Updated Weights:", self.w_[1:].detach().numpy())

            self.errors_.append(error)


# load the iris dataset from UCI repository
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df = shuffle(df)
print(df.head())

X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

print(X[0:5])
print(y[0:5])

train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.25)

train_labels = np.where(train_labels == 'Iris-setosa', 1, 0)
test_labels = np.where(test_labels == 'Iris-setosa', 1, 0)

print('Train Data:', train_data[0:2])
print('Test Data:', test_data[0:2])

print('Train Labels:', train_labels[0:2])
print('Test Labels:', test_labels[0:2])

perceptron = PerceptronTorch(eta=0.01, n_iter=10)
perceptron.fit(train_data, train_labels)

print("Test Predictions:", perceptron.predict(test_data))

y_preds = perceptron.predict(test_data).numpy()
accuracy = accuracy_score(y_preds, test_labels)
print("Accuracy:", round(accuracy, 2) * 100, "%")

print("\nEnter flower measurements: sepal length, sepal width, petal length, petal width")
while True:
    try:
        values = input("Example: 5.1 3.5 1.4 0.2\n> ").split()
        if len(values) != 4:
            raise ValueError

        manual_input = np.array([float(v) for v in values], dtype=float)
        pred = perceptron.predict_single(manual_input)
        print("Prediction (1 = Iris-setosa, 0 = other):", pred)
        break
    except ValueError:
        print("Please enter exactly 4 numeric values separated by spaces.")
