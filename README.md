![Alt текст](./media/head.png)
## Проект: рекомендательная система "ГастроМосква"<br><br>

[Открыть UI проекта](http://62.84.125.214:8080//)<br><br>

"ГастроМосква" — интеллектуальная рекомендательная система, которая помогает пользователям находить лучшие заведения общественного питания Москвы. Вводите запрос — от "лучшая баранина" до "самые красивые официанты", и система предложит подходящие варианты, учитывая ваши пожелания и предпочтения.<br><br>
## О проекте
Этот проект был создан в рамках образовательной программы магистратуры МФТИ "Науки о данных". Разработкой занималась команда студентов:<br>
- Ольга Полеткина 
- Дмитрий Зорин  
- Василий Петров 
- Сергей Куриленко
- Дмитрий Поликарпов 
- Валерий Эрнандес <br><br>

Цель проекта — освоение навыков разработки рекомендательных систем и использования современных технологий в области машинного обучения.<br><br>
## Бизнес-ценность
- Проблема:<br>
    - Сложность выбора среди множества заведений Москвы; стандартные платформы часто не учитывают контекстные запросы.<br><br>
- Решение:<br>
    - "ГастроМосква" использует мощь предобученной языковой модели (LLM) и Retrieval-Augmented Generation (RAG) для создания персонализированных рекомендаций. Уникальная система позволяет:
        - Учитывать запросы естественным языком. Создавать маршруты с оптимальной логистикой.
        - Рекомендовать заведения на основе актуальных данных и пользовательских предпочтений.<br><br>
- Целевая аудитория:
    - Туристы, ищущие уникальный гастрономический опыт.
    - Местные жители, открывающие новые места в городе.
    - Владельцы заведений, желающие привлечь аудиторию через улучшенные рекомендации.<br><br>
## Основные возможности проекта
- Поиск по естественному языку.
- Генерация рекомендаций, включающих несколько точек.
- Использование предобученной LLM и технологии RAG для работы с запросами и актуальной информацией.
- Возможность обновления данных из отзывов, рейтингов и других источников.<br><br>
## Архитектура проекта
- Составляющие системы:
    - Frontend: Интерфейс для ввода запросов и отображения рекомендаций.
    - Backend: API для обработки запросов и взаимодействия с LLM и RAG.
    - База данных: Хранилище информации о заведениях (локация, рейтинги, отзывы и др.).
    - ML-компоненты: Предобученная LLM и Retrieval-Augmented Generation для персонализированных рекомендаций.<br><br>
- Технологии:
    - Языковые модели: USER-bge-m3, ru_core_news_sm, cross-encoder-russian-msmarco, GPT-4o
    - RAG: для соединения LLM с базой данных.
    - База данных: ChromaDB — векторная база данных, предназначенная для работы с эмбеддингами.
    - Инструменты: Python (pandas, torch, transformers, django, folium)<br><br>
## Структура репозитория
- [**/preprocessing**](./preprocessing) — сбор и подготовка данных.
- [**/database**](./database) — векторизация данных и формирование коллекции.
- [**/UI**](./UI) — пользовательский интерфейс.<br><br>
## Установка и запуск
- Требования:
    - Python 3.9+
    - Библиотеки, перечисленные в requirements.txt
- Инструкция:
    1. Скопировать каталог UI на сервер с docker compose.
    2. В файле app/geo.yml заполнить данные:
        - PROXY_API_KEY: ключ для API openai
        - PROXY_API_URL: адрес API openai
        - DJANGO_KEY: Django SECRET_KEY
    3. В папку db скопировать базу данных chromadb c отзывами
    4. В папку models загрузить модели:
        - deepvk/USER-bge-m3
        - DiTy/cross-encoder-russian-msmarco
        - spacy/ru_core_news_sm
    5. В папку geoip скачать карты mmdb. Скачать можно [здесь](https://github.com/P3TERX/GeoLite.mmdb)
    6. Запустить сервис командой docker-compose up
    7. Сервис будет доступен по адресу `http://<адрес сервера>:9001`<br><br>
## Дальнейшее развитие
- Добавление функционала с построением маршрутов.
- Интеграция с популярными туристическими сервисами.
- Расширение на другие города.
- Добавление пользовательских аккаунтов и рекомендаций на основе истории запросов.<br><br>
## Лицензия
- Проект распространяется под лицензией MIT.