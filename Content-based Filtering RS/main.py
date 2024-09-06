from CB import CB

cb = CB('movies2.csv')

cb.fit()

# Test hệ thống gợi ý phim
title = 'Careful (1992)'  # Tên phim bạn muốn tìm phim tương tự
top_x = 10  # Số lượng phim bạn muốn gợi ý

# Gọi hàm gợi ý
sim_scores, recommended_movies = cb.genre_recommendations(title, top_x)

# Hiển thị kết quả
print(f"Phim tương tự '{title}':")
for movie in recommended_movies:
    print(movie)