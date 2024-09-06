import pandas as pd
from pandas import read_csv
from spyder_kernels.utils.lazymodules import pandas

def get_dataframe_anime_csv (path) :
  """
  Đọc file csv của file anime cs, lưu thành data frame với 7 cột
  anime_id
  name
  genre
  type
  episodes
  rating
  members
  :param text:
  :return:
  """

  anime_cols = ["anime_id", "name", "genre", "type", "episodes", "rating", "members"]
  animes = pd.read_csv(path, sep=',', header= 0, names=anime_cols, encoding="utf-8")
  animes["genre"] = animes["genre"].str.split(", ")

  return animes

