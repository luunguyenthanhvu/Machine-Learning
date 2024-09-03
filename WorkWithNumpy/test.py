import numpy as np

a = np.array([1, 2, 3, 4, 5, 6])

#In số chiều
print(a.ndim)

# Lấy ra 1 mảng theo cú pháp start -> last -1
b = a[2:4]
print(b)
# Hiển thị số hàng số cột
print(a.shape)

#Hiển thị độ dài của mảng
print(len(a))
