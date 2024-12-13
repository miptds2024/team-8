from django.shortcuts import render

import folium
import re
import spacy
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import Chroma
from .forms import MeasurementModelForm
from .utils import get_center_coordinates
from langchain_community.embeddings import SentenceTransformerEmbeddings
from openai import OpenAI
from yaml import safe_load

nlp = spacy.load("ru_core_news_sm")
with open('geo.yml', 'r') as f:
  data = safe_load(f)

COUNT = 3

embedding_function = SentenceTransformerEmbeddings(model_name="deepvk/USER-bge-m3")
cross_model = CrossEncoder('cross-encoder-russian-msmarco', max_length=512)
db1 = Chroma(persist_directory="./db/", embedding_function=embedding_function, collection_name="review_flamp_yandex_v7_exploaded__USER-bge-m3")

def clean_text(text):
  try:
    text = text.lower() # приведение к нижнему регистру
    text = re.sub(r"[^\w\s\n]", " ", text) # удаление лишних символов
    text = re.sub(r'[^а-яА-Яa-zA-Z]', ' ', text) # удаление всех символов, кроме русских и английских букв

    # Применение лемматизации к одному тексту
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    text = " ".join(lemmas)

    # Удаляем стоп-слова
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if not token.is_stop]
    text = " ".join(filtered_tokens)

  except:
      text = ""

  return text

def process_text(user_input, places):
    system_prompt = f""" 
    Ты хочешь помочь клиенту найти наилучшее заведение. Система выбрала 3 ресторана по отзывам, разнообразь
    ответ для клиента.
    Если клиент не написал ничего, что относится к поиску ресторана или еде, то ответь '-'.
    Ниже будут предоставлены отзывы о заведениях и запрос от клиента.
    - Коротко напиши интересный факт или шутку  том, что ищет клиент.
    - Коротко напиши обзор о заведениях на основе отзывов.
    Заведения начинаются с 'Название'.
    Если клиент не написал ничего, что относится к поиску ресторана или еде, то ответь '-'.
    Отзывы: {places}
    """ 

    client = OpenAI(
    api_key=data["PROXY_API_KEY"],
    base_url=data["PROXY_API_URL"],
)

    # Отправляем запрос к модели OpenAI
    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # Убедитесь, что используете правильное название модели
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        max_tokens=300,  # Опционально: ограничение на количество токенов в ответе
        temperature=0.3,  # Опционально: настройка креативности ответа
    )
    response = completion.choices[0].message
    answer = response.content
    return answer

def process_query(request):
    form = MeasurementModelForm(request.POST or None)
    
    # координаты центра Москвы
    # TODO определение координат пользователя
    location_lat, location_lon = (55.751477, 37.619003)
    #инициируем карту    
    m = folium.Map(location=get_center_coordinates(location_lat, location_lon), zoom_start=10)
    output = "Интересный факт"

    if form.is_valid():
        query = form.cleaned_data.get('destination')
        # 1-й шаг. Поиск с использованием биэнкодера
        docs1 = db1.similarity_search_with_relevance_scores(clean_text(query), k=10)
        # 2-й шаг. Реранкинг
        scores = cross_model.predict([(f"{query} - хороший отзыв", doc[0].metadata["review"]) for doc in docs1]) 
        for i in range(len(scores)):
            docs1[i] = (*docs1[i], scores[i])
        # Пересортируем результаты, с учетом данных от реранкера
        docs = sorted(docs1, key=lambda p: p[2], reverse=True)
        k=0
        places = ""
        names = []
        for d in docs:
            if k == COUNT:
                break
            folium.Marker([d[0].metadata["lat"], d[0].metadata["lon"]], tooltip=f'{d[0].metadata["name"]} {d[0].metadata["rating"]}', popup=d[0].page_content, icon=folium.Icon(color='red', icon='cloud')).add_to(m)
            places+=f"Название {d[0].metadata['name']}. Отзыв {d[0].metadata['review']} \n"
            if not d[0].metadata["name"] in names:
                names.append(d[0].metadata["name"])
                k+=1
        output = process_text(query, places)
    
    #рендерим карту в html
    m = m._repr_html_()

    
    # TODO определение расстояния от пользователя до найденных точек
    context = {
        'distance': 0,
        'destination': 0,
        'form': form,
        'map':m,
        'text': output
    }
    
    return render(request, 'measurements/main.html', context)
