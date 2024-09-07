import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from utils.read_data import get_dataframe_anime_csv

class HybridRecommender:
    """
    Khởi tạo data frame với anime hoặc công việc.
    """

    def __init__(self, anime_csv):
        self.anime = get_dataframe_anime_csv(anime_csv)
        self.tfidf_matrix = None
        self.cosine_similarity_matrix = None
        self.knn_model = None

    def build_model(self):
        self.anime['genre'] = self.anime['genre'].fillna("").astype('str')
        self.tfidf_matrix = tfidf_matrix(self.anime)

        # Tạo mô hình KNN với metric cosine
        self.knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.knn_model.fit(self.tfidf_matrix)

        # Tạo ma trận cosine similarity
        self.cosine_similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def refresh(self):
        self.build_model()

    def fit(self):
        self.refresh()

    def hybrid_recommendations(self, name, top_cosine_x, top_knn_x):
        """
        Kết hợp CB và KNN để đưa ra gợi ý:
        1. Lọc danh sách dựa trên điểm similarity với cosine (CB).
        2. Dùng KNN để tìm những anime/công việc tốt nhất trong danh sách đã lọc.
        """

        # Lọc danh sách bằng Content-Based Filtering (CB)
        names = self.anime["name"]
        indices = pd.Series(self.anime.index, index=self.anime["name"])

        # Lấy chỉ số của anime dựa trên tên
        idx = indices.get(name)
        if idx is None:
            raise ValueError(f"Anime with name '{name}' not found")

        # Lọc bằng cosine similarity
        sim_scores = list(enumerate(self.cosine_similarity_matrix[idx].tolist()))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_cosine_x + 1]  # Bỏ qua chính nó

        # Lấy các chỉ số anime tương đồng
        filtered_indices = [i[0] for i in sim_scores]

        # Áp dụng KNN để tìm các anime tốt nhất từ danh sách lọc
        filtered_matrix = self.tfidf_matrix[filtered_indices]
        distances, knn_indices = self.knn_model.kneighbors(self.tfidf_matrix[idx], n_neighbors=top_knn_x + 1)

        # Bỏ qua chính nó trong KNN kết quả
        knn_indices = knn_indices[0][1:]

        # Trả về các anime sau khi kết hợp hai phương pháp
        return names.iloc[filtered_indices].values, names.iloc[knn_indices].values

# Hàm xử lý TF-IDF
def tfidf_matrix(animes):
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0.01)
    new_tfidf_matrix = tf.fit_transform(animes["genre"])
    return new_tfidf_matrix

# Hàm tính độ tương đồng cosine
def cosine_similarity(matrix):
    return linear_kernel(matrix, matrix)
