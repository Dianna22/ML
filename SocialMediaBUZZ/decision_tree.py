import data_analysis as da
from sklearn import tree
from sklearn.model_selection import train_test_split

data = da.load_data()
# return data[data.columns[:-1]], data[data.columns[-1]]

train, test = train_test_split(data, test_size=0.34)
x_train, y_train = train[data.columns[:-1]], train[data.columns[-1]],
x_test, y_test = test[data.columns[:-1]], test[data.columns[-1]]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)



y_predicted = clf.predict(x_test)
print(zip(y_predicted, y_test))
correct = len(list(filter(lambda x: x[0]==x[1], list(zip(y_predicted, y_test)))))
print(correct)