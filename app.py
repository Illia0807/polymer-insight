# app.py
import streamlit as st
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import chromadb
import os

# Get the absolute path of the current directory where the app is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Session State Initialization ---
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'explanation_property' not in st.session_state:
    st.session_state.explanation_property = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Dictionary with units for each property
UNITS = {
    'FFV': 'dimensionless',
    'Density': 'g/cm³',
    'Tc': '°C',
    'Rg': 'nm'
}

# Paths to model files
MODELS_PATH = os.path.join(BASE_DIR, 'models')
# Path to ChromaDB
CHROMA_DB_PATH = os.path.join(BASE_DIR, 'chroma_db')

# Load the trained imputer
try:
    imputer = joblib.load(os.path.join(MODELS_PATH, 'imputer.pkl'))
except FileNotFoundError:
    st.error("Error: 'imputer.pkl' not found. Please run 'train_and_save.py'.")
    st.stop()

# Load the trained models
TARGETS_TO_PREDICT = ['FFV', 'Density', 'Tc', 'Rg']
models = {}
for target in TARGETS_TO_PREDICT:
    try:
        model_filename = f'lgbm_{target}.pkl'
        models[target] = joblib.load(os.path.join(MODELS_PATH, model_filename))
    except FileNotFoundError:
        st.error(f"Error: '{model_filename}' not found. Please run 'train_and_save.py'.")
        st.stop()

# Initialize ChromaDB client
try:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Check if the collection exists
    try:
        collection = client.get_collection(name="polymer_facts")
    except ValueError:
        st.error("Error: Collection 'polymer_facts' not found. Please run 'create_db.py'.")
        st.stop()

except Exception as e:
    st.error(f"Error connecting to the ChromaDB database: {e}. Make sure you have created the database.")
    st.stop()

# --- Data handling functions ---
def get_rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [np.nan] * 15
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
    FEATURES = [f'desc_{i}' for i in range(15)]
    new_data = pd.DataFrame({'SMILES': [smiles_string]})
    
    new_data[[f'desc_{i}' for i in range(15)]] = new_data['SMILES'].apply(
        lambda x: pd.Series(get_rdkit_descriptors(x))
    )
    
    new_data[FEATURES] = imputer.transform(new_data[FEATURES])
    
    predictions = {}
    for target in TARGETS_TO_PREDICT:
        prediction = models[target].predict(new_data[FEATURES])[0]
        predictions[target] = prediction
    
    return predictions

# --- Chatbot Functions ---
def retrieve_and_explain(user_question):
    # Retrieve facts from ChromaDB
    results = collection.query(
        query_texts=[user_question],
        n_results=5  # Retrieve more facts to filter from
    )

    if not results['documents'] or not results['documents'][0]:
        return "Information related to your request was not found."

    # Filter out duplicates and combine facts
    unique_facts = set(results['documents'][0])
    response_facts = list(unique_facts)

    # Simple logic to improve the response
    words = user_question.lower().split()
    if 'how' in words or 'why' in words:
        # Try to find facts that explain "how" or "why"
        facts = " ".join([fact for fact in response_facts if 'because' in fact or 'since' in fact])
        if not facts:
            facts = " ".join(response_facts)
    else:
        facts = " ".join(response_facts)

    return facts if facts else "Information related to your request was not found."

# --- Streamlit Interface ---
st.title("Polymer Insight: Property Prediction & Explanation")
st.write("Enter a SMILES string, and the model will predict its physical properties.")

smiles_input = st.text_area("Enter SMILES string:", "*C(=O)c1ccc(C*)cc1", key='smiles_input')

if st.button("Predict"):
    if smiles_input:
        st.session_state.predictions = predict_polymer_properties(smiles_input)

if st.session_state.predictions:
    st.subheader("Predicted Properties:")
    predictions_with_units = {
        'Property': list(st.session_state.predictions.keys()),
        'Value': list(st.session_state.predictions.values()),
        'Unit': [UNITS[prop] for prop in st.session_state.predictions.keys()]
    }
    predictions_df = pd.DataFrame(predictions_with_units)
    st.table(predictions_df.style.format({'Value': "{:.4f}"}))
    
    st.subheader("Ask a question about the polymer properties:")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about polymer properties..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        facts = retrieve_and_explain(prompt)
        with st.chat_message("assistant"):
            st.markdown(facts)
        st.session_state.messages.append({"role": "assistant", "content": facts})

#hello world version 1