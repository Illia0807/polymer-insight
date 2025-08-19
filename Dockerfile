# Используем точную версию Python, которую ты указал.
FROM python:3.13.1-slim

# Устанавливаем рабочую директорию внутри контейнера.
WORKDIR /app

RUN pip install rdkit chromadb --no-cache-dir

# Копируем файл с зависимостями и устанавливаем их.
# Этот шаг кэшируется, что ускоряет последующие сборки.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все остальные файлы проекта, включая app.py.
COPY . .

# Открываем порт 8501, который использует Streamlit.
EXPOSE 8501

# Команда для запуска приложения, когда контейнер стартует.
# Мы используем твой файл app.py и указываем, что Streamlit должен слушать все IP-адреса.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]