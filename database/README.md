## Формирование базы данных<br><br>
### Список файлов:
[Векторизация в ChromaDB](chroma.ipynb)<br>
### Ход работы:
1. **Выбор модели для векторизации**
- Для векторизации текста используется модель трансформера, которая может преобразовать текстовые данные в числовые векторы. Мы используем модель USER-bge-m3, которая хорошо подходит для обработки русского языка. Эта модель преобразует отзывы в высококачественные эмбеддинги (векторы), которые можно использовать для дальнейшего анализа.
- Модель загружается через библиотеку Hugging Face Transformers. Она преобразует текст каждого отзыва в вектор, который сохраняется для дальнейшего использования.
2. **Векторизация отзывов**
- Векторизация выполняется с помощью модели. Каждый отзыв передается в модель, которая генерирует эмбеддинг. Полученные эмбеддинги представляют собой числовые векторы, которые содержат все важные семантические особенности текста отзыва.
- Векторизация выполняется пакетами (batch), чтобы ускорить процесс обработки, особенно если у нас много данных. Размер пакета (batch size) выбирается в зависимости от доступных вычислительных мощностей.
3. **Сохранение эмбеддингов в Chroma DB**
- После того как эмбеддинги для всех отзывов получены, они сохраняются в базу данных Chroma. Для этого создается коллекция в базе данных, которая хранит векторные представления отзывов. В Chroma используется алгоритм HNSW (Hierarchical Navigable Small World) для эффективного поиска по векторным данным.
4. **Использование эмбеддингов**
- Теперь, когда эмбеддинги сохранены в базе данных, можно использовать их для различных задач, таких как:
    - Поиск похожих отзывов на основе векторных представлений.
    - Кластеризация отзывов по схожести.
    - Анализ sentiment анализа или построение рекомендаций.
5. **Оптимизация и reranking**
- На этом этапе мы фокусируемся на оптимизации поиска по векторным данным и повышении качества результатов. Одним из способов улучшить точность поиска является использование reranking — процесса повторной сортировки уже найденных результатов на основе дополнительной информации или модели, которая может более точно оценить релевантность.
- Reranking применяется после того, как алгоритм поиска в Chroma вернул набор кандидатов (например, несколько похожих отзывов). Затем, с использованием более сложной модели (например, модели для оценки семантической схожести или модели ранжирования), мы можем заново отсортировать эти результаты, чтобы получить более точные и релевантные результаты для пользователя.
- Для reranking мы можем использовать модели, обученные на задачах ранжирования текста, такие как BERT или другие трансформеры, которые хорошо показывают себя в задачах классификации и ранжирования.
6. **Конечная обработка отзывов и формирования выдачи**
- Используется ChatGPT-4 через API
    - После того как мы нашли и отсортировали релевантные отзывы с помощью Chroma и reranking, мы применяем ChatGPT-4 для финальной обработки и формирования выдачи. Эта модель позволяет на основе найденных отзывов создавать более детализированные и логичные ответы, что помогает улучшить пользовательский опыт.
    - ChatGPT-4 будет использоваться для:
        - Генерации сжато сформулированных резюме отзывов.
        - Обобщения информации из нескольких отзывов в одном сообщении.
        - Преобразования отзывов в более структурированную форму, например, выделение плюсов и минусов.
        - Предоставления рекомендаций или дополнительных комментариев на основе общего контекста.
