import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNN
import pickle as pickle

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

sepal_length_min = np.min(X[:, 0])
sepal_length_max = np.max(X[:, 0])
sepal_length_avg = np.mean(X[:, 0])

sepal_width_min = np.min(X[:, 1])
sepal_width_max = np.max(X[:, 1])
sepal_width_avg = np.mean(X[:, 1])

petal_length_min = np.min(X[:, 2])
petal_length_max = np.max(X[:, 2])
petal_length_avg = np.mean(X[:, 2])

petal_width_min = np.min(X[:, 3])
petal_width_max = np.max(X[:, 3])
petal_width_avg = np.mean(X[:, 3])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

plt.figure()
plt.scatter(X[:,2], X[:,3], c = y, cmap = cmap, edgecolor = 'k', s = 20)
#plt.show()

clf = KNN(k = 5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
input = [[7.0,3.2,4.7,1.4]]
x = clf.predict(input)
print(x)

acc = np.sum(predictions == y_test) / len(y_test)

with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)