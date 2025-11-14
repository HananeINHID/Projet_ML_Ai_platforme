import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, classification_report
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Configuration Générale 
OUTPUT_DIR = 'models_ai'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Les modèles seront sauvegardés dans : {OUTPUT_DIR}")

def nettoyer_donnees(df, features, target):
    """
    Nettoie le DataFrame :
    1. Garde uniquement les colonnes utiles.
    2. Supprime TOUTES les lignes  avec des valeurs NaN.
    3. Gère les valeurs aberrantes (outliers) avec la méthode de l'IQR.
    """
    colonnes_utiles = features + [target]
    df = df[colonnes_utiles]
    
    print(f"Lignes avant nettoyage (données brutes) : {len(df)}")
    df = df.dropna()
    print(f"Lignes après nettoyage (données valides) : {len(df)}")
    
    # Détection et suppression des valeurs aberrantes
    colonnes_numeriques = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\nSuppression des valeurs aberrantes sur :", colonnes_numeriques)

    for col in colonnes_numeriques:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        borne_inf = Q1 - 1.5 * IQR
        borne_sup = Q3 + 1.5 * IQR
        df = df[(df[col] >= borne_inf) & (df[col] <= borne_sup)]

    print(f"Lignes finales après nettoyage complet : {len(df)}")
    
    return df

# Traitement des datasets 

# 1. Dataset 1 : Student performance
student_file = "Student_Performance.csv"
df_student = pd.read_csv(student_file)
print(f"Fichier '{student_file}' chargé.")

target_student = "Performance Index"
features_student = df_student.drop(columns=[target_student]).columns.tolist()
df_student = nettoyer_donnees(df_student, features_student, target_student)

# Encodage des colonnes catégorielles
for col in features_student:
     if df_student[col].dtype == "object":
        print(f"Encodage LabelEncoder pour : {col}")
        le = LabelEncoder()
        df_student[col] = le.fit_transform(df_student[col])

# 2. Dataset 2 : Météo
meteo_file = "dataset_meteo.csv"
df_meteo = pd.read_csv(meteo_file)
print(f"Fichier '{meteo_file}' chargé.")

target_meteo = "Pluie"
features_meteo = df_meteo.drop(columns=[target_meteo]).columns.tolist()
df_meteo = nettoyer_donnees(df_meteo, features_meteo, target_meteo)

# Encodage des colonnes catégorielles
cat_cols = [col for col in features_meteo if df_meteo[col].dtype == "object"]

if cat_cols:
    print("Encodage OneHotEncoder sur :", cat_cols)
    
    ohe = OneHotEncoder(sparse=False, drop="first")
    encoded = ohe.fit_transform(df_meteo[cat_cols])
    
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cat_cols))

    df_meteo = df_meteo.drop(columns=cat_cols).reset_index(drop=True)
    df_meteo = pd.concat([df_meteo, encoded_df], axis=1)

def train_regression_arbre(features_student, target_student ):

    X = df_student[features_student]
    y = df_student[target_student]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 )

    # 1. Entraînement du modèle
    model = DecisionTreeRegressor(max_depth=10)
    model.fit(X_train, y_train)

    # 2. Evaluation 
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Résultats : R² = {r2:.3f}, RMSE = {rmse:.3f}")

    
    
    # 3. Sauvegarde du modèle
    joblib.dump(model, os.path.join(OUTPUT_DIR, "DecisionTreeRegressor.pkl"))
    print("\nModèle DecisionTreeRegressor sauvegardé.\n")

def train_classification_logreg(df_meteo, features_meteo, target_meteo):
    """
    Entraîne une Régression Logistique sur le dataset météo
    déjà NETTOYÉ et déjà ENCODÉ plus haut.
    """
    # 1. Séparation X / y
    X = df_meteo[features_meteo]
    y = df_meteo[target_meteo]
   
    # 2. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Normalisation (StandardScaler)
    print("\nNormalisation (StandardScaler) requise pour la Régression Logistique...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Entraînement du modèle
    model = LogisticRegression(random_state=42)
    print(f"\nEntraînement du modèle Régression Logistique...")
    
    # On entraîne sur les données NORMALISÉES
    model.fit(X_train_scaled, y_train)
    
    # 5. Évaluation
    print("\nRésultats :")
    # On évalue sur les données normalisés
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))
    
    # 6. Sauvegarde des modèles (Modèle ET Scaler)
    joblib.dump(model, os.path.join(OUTPUT_DIR, "LogisticRegression.pkl"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "c_scaler.pkl"))

    print("\n Modèle LogisticRegression sauvegardé.")

# Exécution du script
train_classification_logreg(df_meteo, features_meteo, target_meteo)
train_regression_arbre(features_student, target_student)

