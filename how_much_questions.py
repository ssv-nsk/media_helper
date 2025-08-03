import os
import uuid
from typing import Optional, Dict, List
from pydantic import BaseModel

import psycopg2
import pandas as pd


UID = 'admin'
PWD = '1234'
SERVER = '192.168.32.128'
PORT = '5432'
DBNAME = 'test_rosatom'

conn = psycopg2.connect(dbname=DBNAME, user=UID, password=PWD, host=SERVER, port=PORT)

def get_sql_query(prompt, model, tokenizer):
    try: 
        system_prompt = '''Ты специалист по составлению SQL запросов. Есть схема в базе данных PostgreSQL с именем test_rosatom,
        в ней есть следующие поля: 
        date - поле в котором записана дата сообщения.

        keywords - поле, в котором хранятся ключевые слова из сообщения, включая организации, людей и тп.
        tone - тональность сообщения, может быть: позитивная, нейтральная и негативная.
        post_type - тип сообщения, может быть: Пост, Репост, Репосты с дополнением, Статья.
        platform - площадка, на которой размещено сообщение (ВКонтакте, Одноклассники и тп.).
        platform_type - тип площадки, на которой размещено сообщение, бывает: Соцсеть, Мессенджер, Блог, Форум, СМИ.
        source - обязательное поле для каждого сообщения, в нем содержится информация о документе, из которого было взято это сообщение.
        
        Интсрукции: 
        Зная схему составь sql запрос, который будет выводить ответ на вопрос пользователя.
        Выделяй всегда у людей только фамилию, у мероприятий - только наименование, у организаций только наименование.
        Если в вопросе не указано явно что надо посчитать по конкретным соцсетям, то не используй поле platform.
        Если в вопросе нет запроса о тональности сообщений, то не используй поле tone.
        Если в вопросе не указано что надо посчитать по конкретным типам сообщений, площадкам или типам площадок, то не включай их в запрос.
        Не придумывай ничего лишнего. Не включай в запрос условия, которые прямо не содержатся в вопросе пользователя.
        Всегда используй оператор ILIKE вместо LIKE.
        В ответе верни запрос обязательно отформатированный в SQL.
        Следуй строго инструкции'''
        
        # prompt = f"""сколько раз упоминается Лихачев в мессенждерах и соцсетях в третью декаду сентября 2023, дай количество по каждому"""
        prompt_2 = """
         проверь себя обязательно, все ли сделано по интсрукции и нет ли в запросе ошибок"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt + prompt_2}
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

def final_answer(prompt, model, tokenizer, conn=conn):
    try:
        llm_result = get_sql_query(prompt, model, tokenizer)
        print("SQL: ", pd.read_sql(llm_result.split("```sql")[-1].split("```")[0], conn).to_dict())
        system_prompt = f'''Ты специалист по ответам на вопросы по таблицам. Есть схема в базе данных PostgreSQL с именем test_rosatom,
        в ней есть следующие поля: 
        date - поле в котором записана дата сообщения.
        message - поле, в котором хранится текст сообщения.
        keywords - поле, в котором хранятся ключевые слова из сообщения, включая организации, людей и тп.
        tone - тональность сообщения, может быть: позитивная, нейтральная и негативная.
        post_type - тип сообщения, может быть: Пост, Репост, Репосты с дополнением, Статья.
        platform - площадка, на которой размещено сообщение (ВКонтакте, Одноклассники и тп.).
        platform_type - тип площадки, на которой размещено сообщение, бывает: Соцсеть, Мессенджер, Блог, Форум, СМИ.
        
        К базе данных был выполнен следующий запрос: {llm_result}.
        От базы данных вернулся ответ на вопрос пользователя: 
        {pd.read_sql(llm_result.split("```sql")[-1].split("```")[0], conn).to_dict()}
        
        Ответь на вопрос пользователя.
        Старайся ответ дать с цифрами.
        Не включай в ответ sql запрос.
        Не придумывай ничего лишнего.
        Форматировать ответ в стиле markdown не надо. Верни простой текст.
        Следуй строго инструкции'''
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "вопрос пользователя: " + prompt}
            ]
        # print(messages)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        ,
            temperature=0.000000000001,
            do_sample=True
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        llm_result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return llm_result
    except Exception as e:
        return str(e)
