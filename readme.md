# TG RAG Бот

Telegram бот с локальной RAG системой (ChromaDB) и LLM моделью Vikhr 1B.

## Команды

/add текст - добавить в базу данных  
/all - показать все записи  
/generate вопрос - сгенерировать ответ без БД  
/bd вопрос - поиск в БД + генерация ответа

## Быстрый старт

Клонирование и установка:
```
git clone <repo>  
conda create -n tg_bot python=3.10  
conda activate tg_bot  
pip install -r requirements.txt
```

*Создайте файл .env:*  
```BOT_TOKEN=ваш_токен_здесь```

Скачайте модель Vikhr-Llama-3.2-1B-Q4_K_M.gguf (~750 MB) с [Hugging Face](https://huggingface.co/tensorblock/Vikhr-Llama-3.2-1B-Instruct-GGUF):  
Положите файл в папку model/

Запуск:  
```
python bot.py
```
## Структура проекта
```
tg-rag-bot/
├── bot.py
├── database.py
├── model/
│   └── Vikhr-*.gguf
├── chroma_db/
├── .env
├── requirements.txt
└── README.md
```
## Некоторые фрагменты кода
### Класс для эмбеддингов (database.py)
```python
class EmbeddingFunction:
    def __init__(self, model):
        self.model = model
    
    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self.model.encode(input).tolist()
    
    def embed_query(self, input):
        return self.__call__(input)
    
    def embed_documents(self, input):
        return self.__call__(input)
    
    def name(self):
        return "custom_embedding"
```
### Загрузка модели (bot.py)

```python
from llama_cpp import Llama

llm = Llama(
    model_path="model/Vikhr-Llama-3.2-1B-Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=4,
    verbose=False
)
```

### Параметры генерации

```python
response = llm(
    prompt,
    max_tokens=512,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repeat_penalty=1.1,
    stop=["<|eot_id|>", "</s>", "User:"]
)
```
### Генерация ответа (bot.py)

```python
def generate_response(prompt, use_system=False):
    if use_system:
        full_prompt = f"<|start_header_id|>system<|end_header_id|>\nТы - полезный ассистент. Отвечай на русском языке.\n<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n{prompt}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
    else:
        full_prompt = prompt
    
    response = llm(
        full_prompt,
        max_tokens=512,
        temperature=0.7,
        stop=["<|eot_id|>", "</s>"]
    )
    return response['choices'][0]['text'].strip()
```

## Скриншоты

Создайте папку screenshots/ и добавьте файлы:
screenshots/start.png - команда /start
screenshots/add.png - команда /add
screenshots/all.png - команда /all
screenshots/generate.png - команда /generate
screenshots/bd.png - команда /bd

## Ошибки подключения
Включите VPN для доступа к Telegram API и Hugging Face.

## requirements.txt

```
aiogram==3.27.0
chromadb==1.5.7
sentence-transformers==5.4.0
llama-cpp-python==0.3.7
python-dotenv==1.2.2
numpy==2.2.6
```
