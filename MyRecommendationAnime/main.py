from models.HybridRecommender import HybridRecommender


def main():
  recommender = HybridRecommender('resources/data/anime.csv')

  # Xây dựng mô hình
  recommender.fit()

  while True:
    # Nhập tên anime và số lượng gợi ý
    title = input("Nhập tên anime để tìm kiếm (hoặc 'exit' để thoát): ")
    if title.lower() == 'exit':
      break

    top_knn_x = int(input("Nhập số lượng anime tương tự cần lọc bằng KNN: "))

    # Gợi ý các anime tương tự
    cosine_top_x, knn_top_x = recommender.hybrid_recommendations(title,
                                                                 50,
                                                                 top_knn_x)

    # Hiển thị kết quả
    print(f"\nGợi ý cho anime '{title}':")
    print("Refined by KNN (Top {top_knn_x}):", knn_top_x)


if __name__ == "__main__":
  main()
