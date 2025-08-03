import os
import psycopg2
import pandas as pd
from tqdm.notebook import tqdm

UID = 'admin'
PWD = '1234'
SERVER = '192.168.32.128'
PORT = '5432'
DBNAME = 'test_rosatom'

conn = psycopg2.connect(dbname=DBNAME, user=UID, password=PWD, host=SERVER, port=PORT)

def get_sql_prompt_2(model, tokenizer, prompt):
    try: 
        system_prompt = """Ты большая языковая модель, которая помогает извлекать сущности из запроса пользователя и на основании ее составлять sql запросы. 
        Ты всегда следуешь инструкции."""

        user_prompt = f'''водные данные: в базе данных PostgreSQL есть таблица с именем test_rosatom,
        в ней есть следующие поля: 
        date - поле в котором записана дата сообщения.
        keywords - поле, в котором хранятся ключевые слова из сообщения, включая организации, людей и тп.
        tone - тональность сообщения, может быть: позитивная, нейтральная и негативная.
        post_type - тип сообщения, может быть: Пост, Репост, Репосты с дополнением, Статья.
        platform - площадка, на которой размещено сообщение (ВКонтакте, Одноклассники и тп.).
        platform_type - тип площадки, на которой размещено сообщение, может быть: Соцсеть, Мессенджер, Блог, Форум, СМИ.
        source - обязательное поле для каждого сообщения, в нем содержится информация о документе, из которого было взято это сообщение.

        Вопрос пользователя: {prompt}
        Извлеки из вопроса пользователя все сущности:
        keywords: основной объект - обязательно,
        date: даты, если есть (например с 20 по 23 сентября 2024 года, или сентябрь 2023),
        platform: конкретная площадка, на которой размещено сообщение (Одноклассники, Вконтакте и пр),
        platform_type: тип площадки, на которой размещено сообщение (Соцсеть, Мессенджер, Блог, Форум, СМИ),
        tone: тональность сообщения,
        post_type: тип сообщения (Пост, Репост, Репосты с дополнением, Статья и т.д.).
        
        И на основании извлеченной информации составь sql запрос, который будет выводить ответ на вопрос пользователя.
        
        **Инструкция**: 
        В sql запросе должны возвращаться все поля (подсказка: используй SELECT *)
        Выделяй всегда у людей только фамилию, у мероприятий - только наименование, у организаций только наименование.
        Обрати внимание, что все основные сущности всегда на русском языке.
        Если в вопросе не указано явно что надо посчитать по конкретным соцсетям, то не используй поле platform.
        Если в вопросе нет запроса о тональности сообщений, то не используй поле tone.
        Если в вопросе не указано что надо посчитать по конкретным типам сообщений, площадкам или типам площадок, то не включай их в запрос.
        Обращай внимание на то, какие типы тональности, типы платформ и типы постов могут быть, не придумывай свои.
        Не придумывай ничего лишнего. Не включай в запрос условия, которые прямо не содержатся в вопросе пользователя.
        Всегда используй оператор ILIKE вместо LIKE.
        В ответе верни запрос обязательно отформатированный в SQL.
        Следуй строго инструкции
'''
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

def get_summary(model, tokenizer, messages, prompt):
    try: 
        system_prompt = """Ты большая языковая модель, которая помогает делать краткую выжимку из сообщений"""

        user_prompt = f"""Есть запрос пользователя: '{prompt}'
         и есть сообщение:
        '{messages}'

        Сделай краткую выжимку из сообщения акцентируя внимание на ключевых словах в его запросе.
        В ответе верни только выжимку и ничего больше, не придумывай ничего и не добавляй от себя.
        Выжимка должна содержать не более 10 предложений. Обязательно следи за этим.
        Не делай нумерацию.
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

def get_summary_final(model, tokenizer, messages, prompt):
    try: 
        system_prompt = """Ты большая языковая модель, которая помогает делать краткую выжимку из сообщений"""

        user_prompt = f"""Есть запрос пользователя: '{prompt}'
         и есть сообщение:
        '{messages}'

        Сделай краткую выжимку из сообщения акцентируя внимание на ключевых словах в его запросе.
        В ответе верни только выжимку и ничего больше, не придумывай ничего и не добавляй от себя.
        Выжимка должна содержать не более 30 предложений. Обязательно следи за этим.
        Не делай нумерацию.
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

        
def main_summ(model, tokenizer, prompt, conn=conn):
    
    answer = get_sql_prompt_2(model, tokenizer, prompt).split("```sql")[-1].split("```")[0]
    result = pd.read_sql(answer, conn)
    print(answer)
    history = ''
    messages = result.drop_duplicates(subset=['message'])['message'].to_list()
    step = 1
    print(len(messages))
    # for i in tqdm(range(0, len(messages), step)):
    for i in tqdm(range(0, 10, step)):
        summ = get_summary(model, tokenizer, messages[i], prompt)
        history = history + summ
        # print(history)

    final_answer = get_summary_final(model, tokenizer, history, prompt)
    return final_answer