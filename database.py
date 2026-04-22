import chromadb
from sentence_transformers import SentenceTransformer
import time

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

class EmbeddingFunction:
    def __init__(self, model):
        self.model = model
    
    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self.model.encode(input).tolist()
    
    def embed_query(self, input):
        if isinstance(input, str):
            input = [input]
        return self.model.encode(input).tolist()
    
    def embed_documents(self, input):
        if isinstance(input, str):
            input = [input]
        return self.model.encode(input).tolist()
    
    def name(self):
        return "custom_embedding"

embedding_function = EmbeddingFunction(model)

client = chromadb.PersistentClient(path="./chroma_db")

try:
    collection = client.get_or_create_collection(
        name="my_knowledge",
        embedding_function=embedding_function
    )
    print("База данных подключена")
except Exception as e:
    print(f"Ошибка: {e}")
    print("Удаляем старую базу и создаём заново...")
    
    import shutil
    shutil.rmtree("./chroma_db", ignore_errors=True)
    
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.create_collection(
        name="my_knowledge",
        embedding_function=embedding_function
    )
    print("База данных создана заново")

def add_to_db(text):
    doc_id = f"doc_{int(time.time())}"
    collection.add(documents=[text], ids=[doc_id])
    return doc_id

def get_all():
    if collection.count() == 0:
        return []
    
    data = collection.get()
    result = []
    for i in range(len(data['ids'])):
        result.append((data['ids'][i], data['documents'][i]))
    return result

def search(query):
    if collection.count() == 0:
        return []
    
    results = collection.query(query_texts=[query], n_results=3)
    if results['documents'] and results['documents'][0]:
        return results['documents'][0]
    return []