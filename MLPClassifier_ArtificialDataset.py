from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
    random_state=3)

plt.scatter(X[y==0][:,0], X[y==0][:,1], s=100, edgecolors='k')
plt.scatter(X[y==1][:,0], X[y==1][:,1], s=100, edgecolors='k', marker='^')

plt.show()

model = MLPClassifier(max_iter=10000, hidden_layer_sizes=(50,100))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=33,
                                                    shuffle=True)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred = model.predict(X_test)

print("Training accuracy:", accuracy_score(y_train, y_train))
print("Testing accuracy:", accuracy_score(y_test, y_pred))


