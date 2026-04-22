from sentence_transformers import SentenceTransformer

print("Загрузка модели для эмбеддингов...")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
print("Модель загружена и сохранена в кэш!")