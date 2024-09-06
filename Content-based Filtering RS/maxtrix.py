from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas
from pandas import isnull, notnull


def tfidf_matrix(movies):
  tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0.01)
  new_tfidf_matrix = tf.fit_transform(movies['genres'])
  return new_tfidf_matrix


def cosine_sim(matrix):
  new_cosine_sim = linear_kernel(matrix, matrix)
  return new_cosine_sim
