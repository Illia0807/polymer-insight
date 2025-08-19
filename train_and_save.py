# train_and_save.py
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.impute import KNNImputer
from lightgbm import LGBMRegressor
import joblib
import os

# Get the absolute path of the current directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to data and models
DATA_PATH = os.path.join(BASE_DIR, 'data')
MODELS_PATH = os.path.join(BASE_DIR, 'models')

# Ensure directories exist
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

# Load and combine data
print("Loading and combining data...")
try:
    train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    
    supplementary_df_list = []
    for i in range(1, 5):
        file_path = os.path.join(DATA_PATH, f'dataset{i}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            supplementary_df_list.append(df)
        else:
            print(f"File {file_path} not found. Skipping.")

    if supplementary_df_list:
        supplementary_df = pd.concat(supplementary_df_list, ignore_index=True)
        supplementary_df.rename(columns={'TC_mean': 'Tc'}, inplace=True)
        train_full_df = pd.concat([train_df, supplementary_df], ignore_index=True).drop_duplicates(subset=['SMILES'])
    else:
        train_full_df = train_df.drop_duplicates(subset=['SMILES'])

    train_full_df.reset_index(drop=True, inplace=True)
    print("Data successfully loaded and combined.")
except FileNotFoundError as e:
    print(f"Error: Could not load data files. Make sure they are in the 'data' folder. Details: {e}")
    exit()

# Feature generation
print("Generating RDKit descriptors...")
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

FEATURES = [f'desc_{i}' for i in range(15)]
train_full_df[[f'desc_{i}' for i in range(15)]] = train_full_df['SMILES'].apply(lambda x: pd.Series(get_rdkit_descriptors(x)))

# Impute missing values in features
print("Imputing missing values in features...")
imputer = KNNImputer(n_neighbors=5)
imputer.fit(train_full_df[FEATURES])
train_full_df[FEATURES] = imputer.transform(train_full_df[FEATURES])

# Save imputer
joblib.dump(imputer, os.path.join(MODELS_PATH, 'imputer.pkl'))
print("Imputer successfully saved.")

# Train and save models
TARGETS_TO_PREDICT = ['FFV', 'Density', 'Tc', 'Rg']
for target in TARGETS_TO_PREDICT:
    print(f"--- Training model for {target} ---")
    
    # Drop rows with NaN in the target variable
    temp_df = train_full_df.dropna(subset=[target]).copy()
    X = temp_df[FEATURES]
    y = temp_df[target]
    
    model = LGBMRegressor(random_state=42)
    model.fit(X, y)
    
    # Save model
    model_filename = f'lgbm_{target}.pkl'
    joblib.dump(model, os.path.join(MODELS_PATH, model_filename))
    print(f"Model for {target} successfully trained and saved to {os.path.join(MODELS_PATH, model_filename)}")
    print("--------------------------------------------------")

print("All models have been trained.")