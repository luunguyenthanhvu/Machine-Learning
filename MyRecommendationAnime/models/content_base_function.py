from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas
from pandas import isnull, notnull

"""
  Lọc ra các thể loại anime
  Xây dựng ma trận với số dòng tương ứng với số lượng anime và số cột
"""

# 1. Thiết lập ma trận TF - IDF
def tfidf_matrix(animes) :
  """
    Dùng hàm TfidfVectorizer để chuẩn hóa genre với:
      + analyzer = 'word' : đơn vị trích xuất là word
      + ngram_range = 1, 1 : mỗi lần trích xuất 1 word
      + min_df = 0: tỉ lệ word không đọc được
      --> ma trận trả về số lượng dòng tương ứng với số cột tương ứng với số từ được tách ra từ genre
  :param animes:
  :return:
  """

  tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0.01)
  new_tfidf_matrix = tf.fit_transform(animes["genre"])
  return new_tfidf_matrix

# 2. Tính độ tương đồng giữa các item
def cosine_similarity(matrix) :
  """
        Dùng hàm "linear_kernel" để tạo thành ma trận hình vuông với số hàng và số cột là số lượng anime
        để tính toán điểm tương đồng giữa từng bộ anime với nhau
  """
  new_cosine_sim = linear_kernel(matrix, matrix)
  return new_cosine_sim