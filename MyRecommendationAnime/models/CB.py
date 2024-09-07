import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.read_data import get_dataframe_anime_csv


class CB(object):
  """
  Khởi tạo data frame "animes" với hàm get_dataframe_anime_csv
  """

  def __init__(self, anime_csv):
    self.anime = get_dataframe_anime_csv(anime_csv)
    self.tfidf_matrix = None
    self.knn_model = None

  def build_model(self):
    self.anime['genre'] = self.anime['genre'].fillna("").astype('str')
    self.tfidf_matrix = tfidf_matrix(self.anime)

    # Tạo mô hình KNN với metric cosine
    self.knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    self.knn_model.fit(self.tfidf_matrix)

  def refresh(self):
    self.build_model()

  def fit(self):
    self.refresh()

  def genre_recommendations(self, name, top_x):
    """
    Xây dựng hàm trả về danh sách top anime tương đồng theo tên anime truyền vào:
    + Tham số truyền vào gồm "name" là tên anime và "top_x" là top anime tương đồng cần lấy
    + Tạo ra list "sim_score" là danh sách điểm tương đồng với anime truyền vào
    + Sắp xếp điểm tương đồng từ cao đến thấp
    + Trả về top danh sách tương đồng cao nhất theo giá trị "top_x" truyền vào
    """

    names = self.anime["name"]
    indices = pd.Series(self.anime.index, index=self.anime["name"])

    # Lấy chỉ số của anime dựa trên tên
    idx = indices.get(name)
    if idx is None:
      raise ValueError(f"Anime with name '{name}' not found")

    # Dùng KNN để tìm các anime tương tự
    distances, indices = self.knn_model.kneighbors(self.tfidf_matrix[idx],
                                                   n_neighbors=top_x + 1)

    # Trả về các kết quả: chỉ số anime và khoảng cách tương đồng
    sim_scores = list(zip(indices.flatten(), distances.flatten()))

    # Bỏ qua anime hiện tại (idx) và trả về các anime tương tự nhất
    sim_scores = sim_scores[1:]  # Bỏ qua anime đầu tiên vì đó là chính nó

    # Lấy các tên anime tương đồng
    similar_animes = names.iloc[[i for i, _ in sim_scores]].values

    return sim_scores, similar_animes


# Hàm xử lý TF-IDF
def tfidf_matrix(animes):
  tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0.01)
  new_tfidf_matrix = tf.fit_transform(animes["genre"])
  return new_tfidf_matrix
