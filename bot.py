import os
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from llama_cpp import Llama
import database as db
import asyncio

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

MODEL_PATH = "model/Vikhr-Llama-3.2-1B-Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

bot = Bot(token=TOKEN)
dp = Dispatcher()

async def on_startup():
    await bot.delete_webhook(drop_pending_updates=True)

def generate_response(prompt, use_system=False):
    if use_system:
        full_prompt = f"<|start_header_id|>system<|end_header_id|>\nТы - полезный ассистент. Отвечай на русском языке кратко и по делу.\n<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n{prompt}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
    else:
        full_prompt = prompt
    
    response = llm(
        full_prompt,
        max_tokens=512,
        temperature=0.7,
        stop=["<|eot_id|>", "</s>"]
    )
    return response['choices'][0]['text'].strip()

@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer(
        "Команды бота:\n"
        "/add текст - добавить текст в базу данных\n"
        "/all - показать все записи в базе данных\n"
        "/generate вопрос - сгенерировать ответ без использования базы данных\n"
        "/bd вопрос - найти в базе данных и сгенерировать ответ\n\n"
        "Примеры:\n"
        "/add Python - это язык программирования\n"
        "/generate как создать список в Python\n"
        "/bd что такое Python"
    )

@dp.message(Command("add"))
async def add(message: types.Message):
    text = message.text.replace("/add", "").strip()
    
    if not text:
        await message.answer("Ошибка: напишите текст после команды /add\n\nПример: /add Мой текст для базы данных")
        return
    
    doc_id = db.add_to_db(text)
    await message.answer(f"Добавлено в базу данных!\nID: {doc_id}\nТекст: {text[:100]}...")

@dp.message(Command("all"))
async def show_all(message: types.Message):
    await message.answer("Загрузка записей из базы данных...")
    
    entries = db.get_all()
    
    if not entries:
        await message.answer("База данных пуста. Добавьте текст через команду /add")
        return
    
    answer = f"Всего записей: {len(entries)}\n\n"
    for doc_id, doc_text in entries:
        preview = doc_text[:100].replace('\n', ' ')
        if len(doc_text) > 100:
            preview += "..."
        answer += f"ID: {doc_id}\nТекст: {preview}\n\n---\n\n"
        
        if len(answer) > 4000:
            await message.answer(answer)
            answer = ""
    
    if answer:
        await message.answer(answer)

@dp.message(Command("generate"))
async def generate(message: types.Message):
    question = message.text.replace("/generate", "").strip()
    
    if not question:
        await message.answer("Ошибка: напишите вопрос после команды /generate\n\nПример: /generate как работает цикл for в Python")
        return
    
    await message.answer("Генерирую ответ... ")
    
    response = generate_response(question, use_system=True)
    
    if len(response) > 4096:
        response = response[:4000] + "\n\n...(ответ обрезан)"
    
    await message.answer(f"Ответ:\n\n{response}")

@dp.message(Command("bd"))
async def bd_search(message: types.Message):
    question = message.text.replace("/bd", "").strip()
    
    if not question:
        await message.answer("Ошибка: напишите вопрос после команды /bd\n\nПример: /bd что такое класс в Python")
        return
    
    await message.answer("Поиск в базе данных...")
    
    found_texts = db.search(question)
    
    if not found_texts:
        await message.answer("Ничего не найдено в базе данных. Добавьте информацию через команду /add и попробуйте снова.")
        return
    
    context = "\n\n---\n\n".join(found_texts)
    
    await message.answer("Генерирую ответ на основе найденной информации...")
    
    prompt = f"""Вопрос: {question}

Информация из базы данных:
{context}

Ответь на вопрос используя только информацию выше. Если в информации нет точного ответа, скажи об этом честно.

Ответ:"""

    response = generate_response(prompt, use_system=False)
    
    sources = "\n\n---\nИсточники из базы данных:\n"
    for i, text in enumerate(found_texts, 1):
        preview = text[:150].replace('\n', ' ')
        sources += f"{i}. {preview}...\n"
    
    final = response + sources
    
    if len(final) > 4096:
        final = final[:4000] + "\n\n...(ответ обрезан)"
    
    await message.answer(final)

async def main():
    await on_startup()
    
    print("\n" + "="*50)
    print("Бот запущен!")
    print("Доступные команды:")
    print("  /add <текст> - добавить в БД")
    print("  /all - показать все записи")
    print("  /generate <вопрос> - генерация без БД")
    print("  /bd <вопрос> - поиск в БД + генерация")
    print("="*50 + "\n")
    
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())