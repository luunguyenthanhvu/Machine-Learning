from models.CB import CB

cb = CB("resources/data/anime.csv")
cb.fit()

title = "Yahari Ore no Seishun Love Comedy wa Machigatteiru. Zoku"
top_x = 20

# Gọi hàm gợi ý
sim_scores, recommended_anime= cb.genre_recommendations(title, top_x)

# Hiển thị kết quả
print(f"Anime tương tự '{title}':")
for anime in recommended_anime:
    print(anime)