from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Tải dữ liệu mẫu
data = load_iris()
X = data.data  # Đặc trưng (features)
y = data.target  # Nhãn (labels)

# Chia dữ liệu thành tập huấn luyện và kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình KNN (với k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# Huấn luyện mô hình
knn.fit(X_train, y_train)

# Dự đoán trên tập kiểm thử
predictions = knn.predict(X_test)

# Đánh giá mô hình
accuracy = knn.score(X_test, y_test)
print(f"Độ chính xác: {accuracy * 100:.2f}%")
