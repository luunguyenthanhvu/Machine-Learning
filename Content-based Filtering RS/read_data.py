from idlelib.iomenu import encoding

import pandas as pd
from pandas import read_csv
from spyder_kernels.utils.lazymodules import pandas


def get_dataframe_movies_csv(text) :
  """
    Đọc file csv của movielens, lưu thành dataframe với 3 cột user id, title, genres
  :param text:
  :return:
  """

  movie_cols = ["movieId", "title", "genres"]
  movies = pd.read_csv('./movies2.csv', sep=',', header=0,
                       names=movie_cols, encoding='latin-1')
  return movies

# Thiết lập ma trận TF - IDF
