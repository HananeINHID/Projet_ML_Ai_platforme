import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, classification_report
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from xgboost import XGBRegressor
from sklearn.svm import SVC

# Configuration Générale 
OUTPUT_DIR = 'models_ai'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Les modèles seront sauvegardés dans : {OUTPUT_DIR}")

def nettoyer_donnees(df, features, target):
    """
    Nettoie le DataFrame :
    - Supprime les lignes avec des valeurs manquantes.
    - Supprime les outliers basés sur l'IQR pour les colonnes numériques.
    """
    df = df[features + [target]].dropna()

    colonnes_num = df.select_dtypes(include=[np.number]).columns
    for col in colonnes_num:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        borne_inf = Q1 - 1.5 * IQR
        borne_sup = Q3 + 1.5 * IQR
        df = df[(df[col] >= borne_inf) & (df[col] <= borne_sup)]

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
df_meteo = pd.read_csv(meteo_file, sep=';')
print(f"Fichier '{meteo_file}' chargé.")

target_meteo = "Pluie"
features_meteo = df_meteo.drop(columns=[target_meteo]).columns.tolist()

# Encodage de la colonne cible pluie en binaire Oui/Non
le= LabelEncoder()
df_meteo[target_meteo] = le.fit_transform(df_meteo[target_meteo])

df_meteo = nettoyer_donnees(df_meteo, features_meteo, target_meteo)

# Encodage des colonnes catégorielles
cat_cols = [col for col in features_meteo if df_meteo[col].dtype == "object"]

if cat_cols:
    print("Encodage OneHotEncoder sur :", cat_cols)
    
    ohe = OneHotEncoder(sparse_output=False, drop="first")
    encoded = ohe.fit_transform(df_meteo[cat_cols])
    
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cat_cols))

    df_meteo = df_meteo.drop(columns=cat_cols).reset_index(drop=True)
    df_meteo = pd.concat([df_meteo, encoded_df], axis=1, ignore_index=False)
features_meteo = df_meteo.drop(columns=[target_meteo]).columns.tolist()

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

def train_classification_logreg(target_meteo):
    """
    Entraîne une Régression Logistique sur le dataset météo
    déjà NETTOYÉ et déjà ENCODÉ plus haut.
    """
    
    features = ['Temperature_C', 'Humidite_%', 'Vent_kmh', 'Pression_hPa']
    # 1. Séparation X / y
    X = df_meteo[features]
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
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "LogisticRegression_scaler.pkl"))

    print("\n Modèle LogisticRegression sauvegardé.")

def xgboost_regression(features_student, target_student):
    """
    Entraîne un modèle XGBoost pour la régression sur le dataset étudiant.
    """
    
    X = df_student[features_student]
    y = df_student[target_student]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Résultats XGBoost : R² = {r2:.3f}, RMSE = {rmse:.3f}")
    

    joblib.dump(model, os.path.join(OUTPUT_DIR, "XGBoostRegressor.pkl"))
    print("\nModèle XGBoostRegressor sauvegardé.\n")

def svm_classification(target_meteo):
    """
    Entraîne un modèle SVM pour la classification sur le dataset météo.
    """
    features = ['Temperature_C', 'Humidite_%', 'Vent_kmh', 'Pression_hPa']
    X = df_meteo[features]
    y = df_meteo[target_meteo]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print("Résultats SVM :")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, os.path.join(OUTPUT_DIR, "SVM_Classifier.pkl"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "SVM_Classifier_scaler.pkl"))
    print("\nModèle SVM_Classifier sauvegardé.\n")
    
# Exécution du script
if __name__ == "__main__":
    train_classification_logreg(target_meteo)
    train_regression_arbre(features_student, target_student)
    xgboost_regression(features_student, target_student)
    svm_classification(target_meteo)

