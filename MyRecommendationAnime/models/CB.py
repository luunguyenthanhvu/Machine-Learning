import pandas as pd
from dask.array import indices

from .content_base_function import tfidf_matrix, cosine_similarity
from utils.read_data import get_dataframe_anime_csv

class CB(object) :
  """
    Khởi tạo data frame "animes" với hàm get_dataframe_anime_csv
  """

  def __init__(self, anime_csv):
    self.anime = get_dataframe_anime_csv(anime_csv)
    self.tfidf_matrix = None
    self.cosine_similarity = None

  def build_model(self):
    self.anime['genre'] = self.anime['genre'].fillna("").astype('str')
    self.tfidf_matrix = tfidf_matrix(self.anime)
    self.cosine_similarity = cosine_similarity(self.tfidf_matrix)

  def refresh(self):
    self.build_model()

  def fit(self):
    self.refresh()

  def genre_recommendations(self, name, top_x):
    """
    Xây dựng hàm trả về danh sách top anime tương đồng theo tên anime truyền vào:
    + Tham số truyền vào gồm "name" là tên anime và "topX" là top anime tương đồng cần lấy
    + Tạo ra list "sim_score" là danh sách điểm tương đồng với anime truyền vào
    + Sắp xếp điểm tương đồng từ cao đến thấp
    + Trả về top danh sách tương đồng cao nhất theo giá trị "topX" truyền vào
    """

    # Đảm bảo biến `name` không bị thay đổi
    names = self.anime["name"]
    indices = pd.Series(self.anime.index, index=self.anime["name"])

    # Lấy chỉ số của anime dựa trên tên
    idx = indices.get(name)
    if idx is None:
      raise ValueError(f"Anime with name '{name}' not found")

    # Tạo danh sách điểm tương đồng
    sim_scores = list(enumerate(self.cosine_similarity[idx].tolist()))

    # Sắp xếp danh sách dựa trên điểm tương đồng
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Bỏ qua điểm tương đồng với chính anime đó (self)
    sim_scores = sim_scores[1:top_x + 1]

    # Lấy các chỉ số của anime tương đồng
    anime_indices = [i[0] for i in sim_scores]

    # Trả về các kết quả
    return sim_scores, names.iloc[anime_indices].values

