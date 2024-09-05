import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

# Cách 1: lấy trung bình


data = pd.read_csv('data.csv', header=None)
print("Mảng ban đầu")
print(data)
X = data.values
# mean = trung bình
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

# fit chỉ là cho dữ liệu vào chuyển đổi -> transform
imp.fit(X)
result = imp.transform(X)
print("Mảng sau khi sửa theo trung bình")
print(result)

# Cách 2: lấy tầng số xuất hiện lớn nhất
imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

# fit chỉ là cho dữ liệu vào chuyển đổi -> transform
imp.fit(X)
result = imp.transform(X)
print("Mảng sau khi sửa theo tầng suất ")
print(result)