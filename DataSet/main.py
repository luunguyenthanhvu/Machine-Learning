from  sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
iris_dataset = load_iris()
'''

# in data
print(iris_dataset.data)

# in ra nhãn
print(iris_dataset.target)

# in ra số dữ liệu
print(len(iris_dataset.target))
'''

# Hàm trả về 1 danh sách
x_train, x_test, y_train, y_test = train_test_split(iris_dataset.data, iris_dataset.target, random_state=0)

model = DecisionTreeClassifier()
mymodel = model.fit(x_train, y_train)

X_New = np.array([[6.0 , 3.23, 4.6, 2.5]])
print(mymodel.predict(X_New))
print(mymodel.score(x_test, y_test))
