import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml



# load dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# check shape, 784 Xs = 28*28 pixel from each picture
print(X.shape, y.shape)
# check the range
print(np.min(X), np.max(X))
# print the first few values of y
print(y[0:5])


model = MLPClassifier(
    hidden_layer_sizes=(6,),
    max_iter=200, alpha=1e-4,
    solver='sgd', random_state=2)

model.fit(X, y)

# print coeffs
print(model.coefs_)

# hidden and output layers
print(len(model.coefs_))
# 2d array 784(input values)x6(nodes)

# visualize the NN (what each node does)
fig, axes = plt.subplots(2, 3, figsize=(5, 4))
for i, ax in enumerate(axes.ravel()):
    coef = model.coefs_[0][:, i]
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(i + 1)
plt.show()

# check how it predicts the first datapoint
x = X[1]
plt.matshow(x.reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.show()
print(model.predict([x]))

# check overall accuracy
print(model.score(X, y))