from sklearn import tree

# Bước 1: Thu thập dữ liệu
# Bước 2: Xử lý dữ liệu
# Bước 3: Training model
# Bước 4: Dự đoán kết quả
# Bước 5: Đánh giá
my_tree = tree.DecisionTreeClassifier()
dactrung = [[1,3,3,7],
            [5,2,4,6],
            [1,2,4,6],
            [5,4,4,3],
            [1,4,4,7],
            [3,2,3,7],
            [3,3,3,6],
            [5,2,2,7]]

nhan = [0,1,1,0,0,0,0,1]

result = my_tree.fit(dactrung,nhan)

# Nhẹ Cao Trung bình ít
kq1 = result.predict([[1,4,3,6]])
print("KQ1")
print(kq1)
# Không bị bệnh tim

# Nhẹ Cao trung bình nhiều
kq2 = result.predict([[1,4,3,7]])
print("KQ2")
print(kq2)