import os
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


emb_model_name = "intfloat/e5-large"
# emb_model_kwargs = {'device': 'cuda'}
emb_model_kwargs = {'device': 'cpu'}
emb_encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=emb_model_name,
    model_kwargs=emb_model_kwargs,
    encode_kwargs=emb_encode_kwargs,
    show_progress=True
)

vector_store_dir = "./Vector_store"
folder_path = "./data/"
KNOWLEDGE_BASE_FOLDER = folder_path
VECTOR_STORE_DIR = vector_store_dir

DOCS_IN_RETRIEVER = 30
RELEVANCE_THRESHOLD_DOCS = 0.1
RELEVANCE_THRESHOLD_PROMPT = 0.6



    
tokenizer_rerank = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side='left', device_map="cpu")
model_rerank = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B").cpu().eval()
token_false_id = tokenizer_rerank.convert_tokens_to_ids("no")
token_true_id = tokenizer_rerank.convert_tokens_to_ids("yes")
max_length = 8192

prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer_rerank.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer_rerank.encode(suffix, add_special_tokens=False)
task = 'Given a web search query, retrieve relevant passages that answer the query'

def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
    return output

def process_inputs(pairs):
    inputs = tokenizer_rerank(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer_rerank.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model_rerank.device)
    return inputs

@torch.no_grad()
def compute_logits(model, inputs, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

def compute_rerank(prompt: str, documents_in: list):
   """
   Функия переанжирования и оценки схожести
   между вектором запроса (prompt) и векторами документов.
   """
   if not documents_in:
       return []


   try:

       relevance_scores = []

       queries = [prompt]
       documents = [i.page_content for i in documents_in]
       pairs = [format_instruction(task, queries, doc) for doc in documents]
       # Tokenize the input texts
       inputs = process_inputs(pairs)
       scores = compute_logits(model_rerank, inputs)
       relevance_scores = [(doc, score) for doc, score in zip(documents_in, scores)]
       return relevance_scores
           
   except Exception as e:
       print(f"Exception in compute_embeddings_similarity: {str(e)}")
       return [(doc, 0.0) for doc in documents]

       

def load_vector_store(vector_store_dir: str, embeddings):
   # Проверяем наличие файла 'index.faiss'
   index_file = os.path.join(vector_store_dir, "index.faiss")
   if not os.path.exists(index_file):
       print(f"Файл {index_file} не найден. Не удалось загрузить vector store.")
       return None
   try:
       vector_store = FAISS.load_local(vector_store_dir, embeddings, allow_dangerous_deserialization=True)
       print(f"Vector store загружен из: {vector_store_dir}")
       return vector_store
   except Exception as e:
       print(f"Ошибка при загрузке vector store: {str(e)}")
       return None

def load_and_index_documents(folder_path: str, vector_store_dir: str, embeddings,
                            chunk_size=1000, chunk_overlap=200) -> bool:
    
   vector_store = load_vector_store(vector_store_dir, embeddings)
   if vector_store:
       print("Существующий vector store успешно загружен.")
       return True
       

def retrieve_documents(
   vector_store,
   user_prompt: str,
   k: int = 15,
   metadata_filters: dict = None
):
   """
   Выполняет поиск по векторному хранилищу FAISS (similarity search).
   Возвращает список кортежей (Document, score).
   """
   if not vector_store:
       print("Vector store не загружен. Сначала загрузите индекс.")
       return []


   try:
       if metadata_filters:
           docs_with_scores = vector_store.similarity_search_with_score(
               user_prompt,
               k=DOCS_IN_RETRIEVER,
               filter=metadata_filters
           )
       else:
           docs_with_scores = vector_store.similarity_search_with_score(user_prompt, k=k)
       return docs_with_scores
   except Exception as e:
       print(f"Ошибка при извлечении документов: {e}")
       return []

def retrieve_documents_2(
   vector_store,
   user_prompt: str,
   k: int = 15,
   metadata_filters: dict = None
):
   """
   Выполняет поиск по векторному хранилищу FAISS (similarity search).
   Возвращает список кортежей (Document, score).
   """
    
   docs_with_scores = vector_store.similarity_search_with_score(user_prompt, k=k)
   return docs_with_scores


def compute_embeddings_similarity(embeddings, prompt: str, documents: list):
   """
   Синхронная функция вычисления косинусной похожести
   между вектором запроса (prompt) и векторами документов.
   """
   if not documents:
       return []


   try:
       # Получаем эмбеддинг для prompt
       prompt_embedding = np.array(embeddings.embed_query(prompt))
       relevance_scores = []


       for doc in documents:
           doc_embedding = np.array(embeddings.embed_query(doc.page_content))
           # Проверяем валидность эмбеддингов
           if (prompt_embedding.size == 0 or doc_embedding.size == 0):
               print(f"Warning: invalid embedding for doc: {doc.metadata.get('source', 'Unknown')}")
               similarity = 0.0
           else:
               dot_product = np.dot(prompt_embedding, doc_embedding)
               norm_prompt = np.linalg.norm(prompt_embedding)
               norm_doc = np.linalg.norm(doc_embedding)
               similarity = 0.0
               # Косинусная похожесть = (A·B) / (|A| * |B|)
               if norm_prompt > 1e-9 and norm_doc > 1e-9:
                   similarity = dot_product / (norm_prompt * norm_doc)
               # «Зажимаем» результат в [-1, 1] для защиты от плавающих погрешностей
               similarity = np.clip(similarity, -1.0, 1.0)


           relevance_scores.append((doc, similarity))


       return relevance_scores


   except Exception as e:
       print(f"Exception in compute_embeddings_similarity: {str(e)}")
       return [(doc, 0.0) for doc in documents]

def is_prompt_relevant_to_documents(relevance_scores, relevance_threshold=RELEVANCE_THRESHOLD_PROMPT):
    """
    Синхронная проверка: считать ли запрос релевантным документам
    по максимальной оценке (similarity) среди всех.
    """
    try:
        if not relevance_scores:
            return False

        max_similarity = max((sim for _, sim in relevance_scores), default=0.0)
        print(f"Debug: max_similarity = {max_similarity:.4f}, "
              f"threshold = {relevance_threshold}, "
              f"is_relevant = {max_similarity >= relevance_threshold}")

        return max_similarity >= relevance_threshold
    except Exception as e:
        print(f"Exception in is_prompt_relevant_to_documents: {str(e)}")
        return False

def postprocess_llm_response(
    model, 
    tokenizer,
    llm_response: str,
    user_prompt: str,
    context_str: str = "",
    references: dict = None,
    is_relevant: bool = False
) -> tuple:
    """
    Упрощённая синхронная постобработка ответа от LLM:
    - улучшает стиль,
    - добавляет/структурирует список ссылок (если нужно).
    Возвращает (final_answer, processed_references).
    """
    if references is None:
        references = {}

    if not is_relevant:
        references = {}
        context_str = ""

    prompt_references = (
        "### Предоставленные данные\n"
        f"LLM raw response:\n{llm_response}\n\n"
        f"User prompt:\n{user_prompt}\n\n"
        f"Context:\n{context_str}\n\n"
        f"References:\n{references}\n\n"
        f"is_relevant: {is_relevant}\n"
        "-------------------------\n"
        "Пожалуйста, перепроверьте ясность и, если есть ссылки, перечислите их в конце.\n"
        "Верните окончательный улучшенный ответ прямо сейчас:\n"
    )
    messages = [
    {"role": "system", "content": "Вы являетесь продвинутой языковой моделью, перед которой стоит задача дать окончательный, хорошо структурированный ответ на основе заданного контента"},
    {"role": "user", "content": prompt_references}
    ]

    # Вызов llm синхронно (упрощённый вариант)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512**2,
        temperature=0.0000000000000001,
        do_sample=True
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    chain_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # response = chat_model.invoke(instructions)
    
    # chain_response = llm(prompt_references)
    final_answer = chain_response.strip()

    return final_answer, references

def generate_response(
    model,
    tokenizer,
    vector_store,
    prompt: str,
    chat_history=None,
    metadata_filters=None,
    context=None
    ):
    
    # 1. Загрузка/создание vector_store
    if not vector_store:
       return "Unable to load Vector Store.", None, None
    
    # 2. Предобрабатываем вопрос
    print("2. Предобрабатываем вопрос")
    if chat_history is None:
       chat_history = []
    # prepared_prompt = preprocess_user_prompt(prompt, chat_history)
    prepared_prompt = prompt
    print("prepared_prompt: ", prepared_prompt)
    
    # 3. Извлекаем документы из FAISS
    print("3. Извлекаем документы из FAISS")
    retrieved_docs_with_scores = retrieve_documents_2(
       vector_store=vector_store,
       user_prompt=prepared_prompt,
       k=DOCS_IN_RETRIEVER,
       metadata_filters=metadata_filters
    )
    retrieved_docs = [doc for doc, _ in retrieved_docs_with_scores]
    
    # 4. Подсчитываем косинусную похожесть
    print("4. Подсчитываем косинусную похожесть")
    # relevance_scores = compute_embeddings_similarity(embeddings, prepared_prompt, retrieved_docs)
    relevance_scores = compute_rerank(prepared_prompt, retrieved_docs)
    
    # 5. Фильтруем документы на основе RELEVANCE_THRESHOLD_DOCS
    print("5. Фильтруем документы на основе RELEVANCE_THRESHOLD_DOCS")
    relevant_docs = [
       doc for (doc, similarity) in relevance_scores
       if similarity >= RELEVANCE_THRESHOLD_DOCS
    ]
    
    # 6. Если ничего не нашлось, выдаём «fallback»-ответ
    if not relevant_docs:
       fallback_answer = "I couldn't find relevant information to answer your question."
       final_answer, _, _ = postprocess_llm_response(
           llm_response=fallback_answer,
           user_prompt=prompt,
           context_str="",
           references=None,
           is_relevant=False
       )
       return final_answer, None
    
    # 7. Формируем «контекст» из релевантных документов
    print("7. Формируем «контекст» из релевантных документов")
    context_str = ""
    for doc in relevant_docs:
       source = doc.metadata.get('source', 'Unknown')
       page = doc.metadata.get('page', 'N/A')
       content = doc.page_content or 'N/A'
       context_str += f"Source: {source}, Page: {page}\nContent:\n{content}\n---\n"
    
    # 8. «Системный» промпт: даём модели контекст
    system_prompt = (
       "Вы эксперт. Дай краткий ответ, основанный на контексте:\n"
       f"{context_str}\n"
       "--- Конец контекста ---\n"
       "Если на вопрос пользователя нет полного ответа в предоставленном контексте, "
       "используйте свои лучшие суждения, оставаясь при этом правдивым.\n"
    )
    prompt_template = ChatPromptTemplate.from_messages([
       ("system", system_prompt),
       MessagesPlaceholder(variable_name="chat_history"),
       ("user", "{input}")
    ])
    print("system_prompt: ", system_prompt)
    # print("prompt_template: ", prompt_template)

    
    # 9. Формируем финальный промпт для LLM
    print("9. Формируем финальный промпт для LLM")
    messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prepared_prompt}
    ]
    
    
    # 10. Вызываем LLM
    print("10. Вызываем LLM")
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512**2,
        temperature=0.0000000000000001,
        do_sample=True
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    llm_result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("llm_result:", llm_result)

    if hasattr(llm_result, "content"):
       answer_text = llm_result.content
    else:
       answer_text = str(llm_result)
    print("answer_text: ", answer_text)
    
    # 11. Оцениваем «глобальную» релевантность (RELEVANCE_THRESHOLD_PROMPT)
    print("11. Оцениваем «глобальную» релевантность (RELEVANCE_THRESHOLD_PROMPT)")
    is_relevant = is_prompt_relevant_to_documents(relevance_scores)
    print("is_relevant: ", is_relevant)
    
    # 12. Готовим список ссылок
    print("12. Готовим список ссылок")
    references = {}
    for doc in relevant_docs:
       filename = doc.metadata.get("source", "Unknown")
       page = doc.metadata.get("page", "N/A")
       references.setdefault(filename, set()).add(page)
    print("references: ", references)
    
    # 13. Пост-обработка ответа
    print("13. Пост-обработка ответа")
    final_answer, processed_refs = postprocess_llm_response(
       model, 
       tokenizer,
       llm_response=answer_text,
       user_prompt=prompt,
       context_str=context_str,
       references=references,
       is_relevant=is_relevant
    )
    print("final_answer: ", final_answer)
    
    # 14. Итоговый форматированный текст
    print("14. Итоговый форматированный текст")
    if is_relevant:
       final_text = final_answer
       source_files = list(processed_refs.keys()) if processed_refs else None
    else:
       final_text = final_answer
       source_files = None
    
    return final_text, source_files, context_str
