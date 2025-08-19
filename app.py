# app.py
import streamlit as st
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import os

# Импортируем новую библиотеку
from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection

# Get the absolute path of the current directory where the app is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Session State Initialization ---
# Инициализируем переменные, если они еще не существуют в состоянии сессии
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Словарь с единицами измерения для каждого свойства
UNITS = {
    'FFV': 'dimensionless',
    'Density': 'g/cm³',
    'Tc': '°C',
    'Rg': 'nm'
}

# Пути к файлам моделей
MODELS_PATH = os.path.join(BASE_DIR, 'models')
CHROMA_DB_PATH = os.path.join(BASE_DIR, 'chroma_db')

# Загрузка обученного импьютера для обработки пропущенных значений
try:
    imputer = joblib.load(os.path.join(MODELS_PATH, 'imputer.pkl'))
except FileNotFoundError:
    st.error("Ошибка: 'imputer.pkl' не найден. Пожалуйста, запустите скрипт для обучения моделей.")
    st.stop()

# Загрузка обученных моделей
TARGETS_TO_PREDICT = ['FFV', 'Density', 'Tc', 'Rg']
models = {}
for target in TARGETS_TO_PREDICT:
    try:
        model_filename = f'lgbm_{target}.pkl'
        models[target] = joblib.load(os.path.join(MODELS_PATH, model_filename))
    except FileNotFoundError:
        st.error(f"Ошибка: '{model_filename}' не найден. Пожалуйста, запустите скрипт для обучения моделей.")
        st.stop()

# --- Инициализация клиента ChromaDB через st.connection ---
try:
    # Передаем конфигурацию для PersistentClient с путем к базе данных
    conn = st.connection("chromadb",
                         type=ChromadbConnection,
                         client="PersistentClient",
                         path=CHROMA_DB_PATH)
    
    # Получаем коллекцию из соединения
    collection = conn.get_collection(name="polymer_facts")

except Exception as e:
    st.error(f"Ошибка подключения к базе данных ChromaDB: {e}.")
    st.stop()


# --- Функции для обработки данных ---
def get_rdkit_descriptors(smiles):
    """Вычисляет набор дескрипторов RDKit для заданной SMILES строки."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [np.nan] * 15  # Возвращаем NaN, если SMILES некорректен
    descriptors = [
        Descriptors.MolWt(mol), Descriptors.TPSA(mol), Descriptors.MolLogP(mol),
        Descriptors.NumRotatableBonds(mol), Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol), Descriptors.HeavyAtomCount(mol),
        Descriptors.RingCount(mol), Descriptors.FractionCSP3(mol),
        Descriptors.NumAliphaticRings(mol), Descriptors.NumAromaticRings(mol),
        Descriptors.BertzCT(mol), Descriptors.SlogP_VSA2(mol),
        Descriptors.EState_VSA1(mol), Descriptors.NumHeteroatoms(mol)
    ]
    return descriptors

def predict_polymer_properties(smiles_string):
    """Предсказывает свойства полимера по его SMILES строке."""
    FEATURES = [f'desc_{i}' for i in range(15)]
    new_data = pd.DataFrame({'SMILES': [smiles_string]})
    
    # Вычисляем дескрипторы
    desc_series = new_data['SMILES'].apply(lambda x: pd.Series(get_rdkit_descriptors(x)))
    desc_series.columns = FEATURES
    new_data = pd.concat([new_data, desc_series], axis=1)

    # Обрабатываем пропуски
    new_data[FEATURES] = imputer.transform(new_data[FEATURES])
    
    # Делаем предсказания
    predictions = {}
    for target in TARGETS_TO_PREDICT:
        prediction = models[target].predict(new_data[FEATURES])[0]
        predictions[target] = prediction
    
    return predictions

# --- Функции чат-бота ---
def retrieve_and_explain(user_question):
    """Извлекает факты из ChromaDB для ответа на вопрос пользователя."""
    try:
        results = collection.query(
            query_texts=[user_question],
            n_results=5  # Извлекаем несколько фактов для лучшего ответа
        )
    except Exception as e:
        return f"Произошла ошибка при запросе к базе данных: {e}"

    if not results or not results.get('documents') or not results['documents'][0]:
        return "К сожалению, информация по вашему запросу не найдена."

    # Фильтруем дубликаты и объединяем факты
    unique_facts = sorted(list(set(results['documents'][0])), key=len, reverse=True)
    response_text = " ".join(unique_facts)

    return response_text

# --- Интерфейс Streamlit ---
st.title("Polymer Insight: предсказание и объяснение свойств")
st.write("Введите SMILES строку полимера, и модель предскажет его физические свойства.")

smiles_input = st.text_area("Введите SMILES строку:", "*C(=O)c1ccc(C*)cc1", key='smiles_input')

if st.button("Предсказать"):
    if smiles_input:
        with st.spinner('Вычисляем свойства...'):
            st.session_state.predictions = predict_polymer_properties(smiles_input)
            st.session_state.messages = [] # Очищаем историю чата при новом предсказании

if st.session_state.predictions:
    st.subheader("Предсказанные свойства:")
    predictions_with_units = {
        'Свойство': list(st.session_state.predictions.keys()),
        'Значение': list(st.session_state.predictions.values()),
        'Единица изм.': [UNITS[prop] for prop in st.session_state.predictions.keys()]
    }
    predictions_df = pd.DataFrame(predictions_with_units)
    st.table(predictions_df.style.format({'Значение': "{:.4f}"}))
    
    st.subheader("Задайте вопрос о свойствах полимера:")

    # Отображаем историю чата
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Поле для ввода нового сообщения
    if prompt := st.chat_input("Например: почему такая плотность?"):
        # Добавляем и отображаем сообщение пользователя
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Получаем и отображаем ответ ассистента
        with st.spinner('Ищу информацию...'):
            response = retrieve_and_explain(prompt)
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
