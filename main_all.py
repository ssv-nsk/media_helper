from fastapi import FastAPI, Request, Form, File, UploadFile, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import uuid
from typing import Optional, Dict, List
from pydantic import BaseModel
from source.common_questions import *
from source.how_much_questions import *
from source.summarization import *
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

vector_store_dir = "./Vector_store"
folder_path = "./data/"

KNOWLEDGE_BASE_FOLDER = folder_path
VECTOR_STORE_DIR = vector_store_dir

load_and_index_documents(KNOWLEDGE_BASE_FOLDER, VECTOR_STORE_DIR, embeddings)
vector_store = load_vector_store(VECTOR_STORE_DIR, embeddings)

def get_agent(question, model=model, tokenizer=tokenizer):
    try: 
        system_prompt = """Ты большая языковая модель, которая помогает определить, какого агента запустить для получения ответа"""

        user_prompt = f"""Есть вопрос пользователя:
        '{question}'

        Твоя задача в ответе вернуть номер агента, к какому надо обратиться для получения на вопрос.
        Агенты:
        1. Отвечает на общие вопросы, например "расскажи об..", "сколько камер было поставлено в Москве", "Расскажи про электромобиль 'Атом'" и т.п.
        2. Отвечает на вопросы, связанные с подсчетом каких-либо характеристик по сообщениям, напрмер ""сколько раз упоминается корпорация РОСАТОМ в мессенждерах и сми, дай количество по каждому" и т.п.
        3. Делает суммаризацию сообщений.

        Инструкция:
        Верни в ответе номер агента, к которому надо обратиться, чтобы получить ответ на вопрос пользователя.
        Ответ должен быть вида:
        'ageent': номер агента
        Ответ обязательно верни в формате json.

        Следуй строго инструкции, приступай.
        """


        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
            ]
            
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=128001
        ,
            temperature=0.0000000000000001,
            do_sample=True
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        llm_result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return llm_result
    except Exception as e:
        return str(e)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("templates", exist_ok=True)

templates = Jinja2Templates(directory="templates")

class ChatData(BaseModel):
    history: List[Dict] = []

chat_storage = ChatData()

def bot_response(user_input: str) -> str:
    llm_result = get_agent(user_input)
    print("llm_result: ", llm_result)
    if "1" in llm_result:
        final_text, source_files, context_str = generate_response(model, tokenizer, vector_store, user_input)
        return final_text, source_files, context_str
    elif "2" in llm_result:
        answer = final_answer(user_input, model, tokenizer)
        return answer, None, None
    elif "3" in llm_result:
        answer = main_summ(model, tokenizer, user_input)
        return answer, None, None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "history": chat_storage.history})

@app.post("/send")
async def handle_message(
    request: Request,
    message: Optional[str] = Form(None),
    link: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    file_path = None
    text = ''
    if file and file.filename:
        file_ext = os.path.splitext(file.filename)[1]
        # file_name = f"123_{file_ext}"
        file_name = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join("static/uploads", file_name)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        loader = Docx2txtLoader(file_path)
        data = loader.load()
        text = data[0].page_content

    user_message = {
        "type": "user",
        "text": message,
        "link": link,
        "file": file_path.split("/")[-1] if file_path else None
    }
    chat_storage.history.append(user_message)
    
    # Получаем ответ от бота (три компонента)
    final_text, source_files, context_str = bot_response(message or "")


    # Создаем три отдельных сообщения от бота
    if final_text:
        chat_storage.history.append({
            "type": "bot",
            "text": final_text,
            "part": "final_text"
        })
    
    if source_files:
        chat_storage.history.append({
            "type": "bot",
            "text": source_files,
            "part": "source_files"
        })
    
    if context_str:
        chat_storage.history.append({
            "type": "bot",
            "text": context_str,
            "part": "context_str"
        })

    return RedirectResponse(url="/", status_code=303)

@app.post("/clear")
async def clear_chat():
    chat_storage.history.clear()
    return RedirectResponse(url="/", status_code=303)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
