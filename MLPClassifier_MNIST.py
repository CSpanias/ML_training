from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits



# load dataset
X, y = load_digits(return_X_y=True)

# check shape, 64 Xs = 8*8 pixel from each picture
print(X.shape, y.shape)

# reshape the array for a more intuitive view of the datapoint
print(X[0].reshape(8, 8))

# display an array as a matrix in a new figure window
plt.matshow(X[0].reshape(8, 8), cmap=plt.cm.gray)
# remove x tick marks
plt.xticks(())
# remove y tick marks
plt.yticks(())

plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=44,
                                                    shuffle=True)
model = MLPClassifier()
model.fit(X_train, y_train)

# check how it predicts the first datapoint
x = X_test[1]
plt.matshow(x.reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.show()
print(model.predict([x]))

# check overall accuracy
print(model.score(X_test, y_test))

# check the wrong predictions
y_pred = model.predict(X_test)
incorrect = X_test[y_pred != y_test]
incorrect_true = y_test[y_pred != y_test]
incorrect_pred = y_pred[y_pred != y_test]

# check the 1st image it got wrong
j = 0
plt.matshow(incorrect[j].reshape(8, 8), cmap = plt.cm.gray)
plt.xticks(())
plt.yticks(())

plt.show()

print("true value:", incorrect_true[j])
print("predicted value:", incorrect_pred[j])