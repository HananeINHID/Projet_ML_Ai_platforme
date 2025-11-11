import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

# Configuration Générale 
OUTPUT_DIR = 'models_ai'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Les modèles seront sauvegardés dans : {OUTPUT_DIR}")

def nettoyer_donnees(df, features, target):
    """
    Nettoie le DataFrame :
    1. Garde uniquement les colonnes utiles.
    2. Supprime TOUTES les lignes  avec des valeurs NaN.
    """
    colonnes_utiles = features + [target]
    df = df[colonnes_utiles]
    
    print(f"Lignes avant nettoyage (données brutes) : {len(df)}")
    df = df.dropna()
    print(f"Lignes après nettoyage (données valides) : {len(df)}")
    
    return df

def train_regression_arbre():
    
    # 1. Définition du Dataset 
    DATASET_FILE = 'Student_Performance.csv'
    
    # 2. Charger les données 
    df = pd.read_csv(DATASET_FILE)
    print(f"Fichier '{DATASET_FILE}' chargé avec succès.")

    # 3. Définir les Caractéristiques (X) et la Cible (y)
    TARGET = 'Performance Index'
    FEATURES = df.drop(columns=[TARGET]).columns.tolist() 
    
    print(f"Cible (y) : {TARGET}")
    print(f"Caractéristiques (X) : {FEATURES}")

    # 4. Nettoyage et Préparation (sans imputation)
    df = nettoyer_donnees(df, FEATURES, TARGET)

    # 5. Encodage automatique des colonnes catégorielles
    print("\nVérification et encodage des colonnes non numériques...")
    for col in FEATURES:
        if df[col].dtype == 'object':
            print(f"→ Encodage de la colonne catégorielle : '{col}'")
            df[col] = df[col].astype('category').cat.codes

    print("\nTypes de données après encodage :")
    print(df.dtypes)

    # 6. Séparation en train/test
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 7. Entraînement du modèle
    print("\nLe modèle Arbre de Décision n'a pas besoin de normalisation (scaler).")
    model = DecisionTreeRegressor(max_depth=10, random_state=42)
    print(f"\nEntraînement du modèle Arbre de Décision...")
    
    model.fit(X_train, y_train)
    
    # 8. Évaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\nRésultats : R² = {r2:.3f}, RMSE = {rmse:.3f}")
    
    # 9. Sauvegarde du modèle
    model_filename = 'DecisionTree_R.pkl'
    joblib.dump(model, os.path.join(OUTPUT_DIR, model_filename))
    print(f"\n Modèle '{model_filename}' sauvegardé dans '{OUTPUT_DIR}'.")


# Exécution du script
print("Début du script d'entraînement pour (Arbre de Décision Régression)...")
train_regression_arbre()
print("\nScript d'entraînement terminé avec succès ")
