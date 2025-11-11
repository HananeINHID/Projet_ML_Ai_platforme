import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import os

# --- 1. Configuration des chemins ---
CSV_FILE = 'dataset_vehicules_classification_100k.csv'
OUTPUT_DIR = 'models_ai' # Dossier de sortie pour les modèles
MODEL_PATH = os.path.join(OUTPUT_DIR, 'logreg_model.pkl')
SCALER_PATH = os.path.join(OUTPUT_DIR, 'scaler.pkl') 

print(f"Fichier CSV source : {CSV_FILE}")
print(f"Dossier de sortie : {OUTPUT_DIR}")

# --- 2. Chargement des données ---
try:
    df = pd.read_csv(CSV_FILE)
    print(f"Fichier '{CSV_FILE}' chargé.")
except FileNotFoundError:
    print(f"ERREUR : Fichier '{CSV_FILE}' introuvable.")
    exit()

# --- 3. Préparation des données ---
# Nettoyer les noms de colonnes (ex: " Hauteur_m " -> "Hauteur_m")
df.columns = df.columns.str.strip()

# Gérer les valeurs manquantes (NaN)
# On remplace les "trous" par la valeur médiane, ce qui est mieux que de supprimer la ligne.
print("Nettoyage des NaN (remplissage par la médiane)...")
df["Hauteur_m"] = df["Hauteur_m"].fillna(df["Hauteur_m"].median())
df["Nombre_de_roues"] = df["Nombre_de_roues"].fillna(df["Nombre_de_roues"].median())

# Convertir les cibles texte en chiffres (le ML ne comprend que les chiffres)
target_map = {'Camion': 0, 'Touristique': 1}
df['type_numeric'] = df['Type_de_vehicule'].map(target_map)

# Définir les colonnes "caractéristiques" (X) et la colonne "cible" (y)
features = ['Hauteur_m', 'Nombre_de_roues']
target = 'type_numeric'

X = df[features] # Données d'entrée (Hauteur, Roues)
y = df[target]   # Ce qu'on veut prédire (0 ou 1)

print("Données prêtes pour l'entraînement.")

# --- 4. Entraînement du "Traducteur" (Scaler) ---
scaler = StandardScaler()

# On entraîne le scaler ET on transforme X en même temps.
# 'scaler' mémorise la moyenne et l'écart-type de 'X'.
# 'X_scaled' est la version "traduite" de X.
X_scaled = scaler.fit_transform(X) 

print(f"Scaler entraîné.")

# --- 5. Entraînement du Modèle ---
model = LogisticRegression()

# On entraîne le modèle sur les données "traduites" (scalées)
model.fit(X_scaled, y) 

print(f"Modèle entraîné. Précision: {model.score(X_scaled, y) * 100:.2f}%")

# --- 6. Sauvegarde des fichiers ---
# S'assurer que le dossier 'models_ai' existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sauvegarder le modèle entraîné 
joblib.dump(model, MODEL_PATH) 
# Sauvegarder le scaler entraîné (TRÈS IMPORTANT pour l'application)
joblib.dump(scaler, SCALER_PATH)

print("\n--- FIN ---")
print(f"Modèle sauvegardé dans : {MODEL_PATH}")
print(f"Scaler sauvegardé dans : {SCALER_PATH}")

