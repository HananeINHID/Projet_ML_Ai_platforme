from django.shortcuts import render , redirect

import joblib  # Utilisé pour charger les modèles .pkl (modèle ML et scaler)
import os      # Utilisé pour la manipulation des chemins de fichiers (os.path, os.path.exists)
import numpy as np # Utilisé pour créer des tableaux (arrays) pour la prédiction
import pandas as pd

import pickle
import os
from django.conf import settings

def load_model(filename):
    """
    Charge un modèle pickle (.pkl) depuis le dossier model_ai et le retourne.
    """
    path = os.path.join(settings.BASE_DIR, 'models_ai', filename)
    if not os.path.exists(path):
        print(f"Erreur : le fichier {filename} n'existe pas à {path}")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)



# --- Vues Statiques ---
# Ces vues ont pour seul rôle de_render_ des pages HTML.

def index(request):
    """
    Affiche la page d'accueil principale du site (index.html).
    """
    return render(request, 'index.html')

def regLog_details(request):
    """
    Affiche la page de détails sur l'algorithme de régression logistique.
    """
    return render(request, 'regLog_details.html')

def regLog_atelier(request):
    """
    Affiche la page de présentation de l'atelier pratique.
    """
    return render(request, 'regLog_atelier.html')

def regLog_tester(request):
    """
    Affiche le formulaire de test (meteo_form.html) pour la prédiction.
    """
    return render(request, 'meteo_form.html')

# --- Fonctions de Support (Helpers) ---

def load_models(name):
    """
    Charge un fichier .pkl (modèle ou scaler) depuis le dossier 'models_ai'.

    Cette fonction détermine dynamiquement le chemin absolu vers le dossier
    'models_ai' pour garantir que le chargement fonctionne quel que soit
    l'endroit d'où le serveur est lancé.

    Args:
        name (str): Le nom du fichier à charger (ex: 'logreg_model.pkl').

    Returns:
        object: Le modèle ou scaler chargé, ou None si le fichier est introuvable.
    """
    # Chemin vers le dossier 'algoAi' (où se trouve ce fichier views.py)
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    # On remonte d'un niveau pour être dans 'my_aiplatform' (la racine du projet)
    app_dir = os.path.dirname(base_dir) 
    # On construit le chemin vers le dossier des modèles
    models_dir = os.path.join(app_dir, 'models_ai')
    
    model_path = os.path.join(models_dir, name)
    
    # Vérification robuste : s'assurer que le fichier existe avant de le charger
    if not os.path.exists(model_path):
        print(f"ERREUR CRITIQUE: Fichier modèle non trouvé : {model_path}")
        return None
        
    # Chargement du fichier .pkl
    ml_model = joblib.load(model_path)
    return ml_model

# --- Vue de Prédiction (Logique Métier) ---

def regLog_prediction(request):
    """
    Gère la logique de prédiction de la MÉTÉO.
    """
    
    # Logique pour une requête POST (l'utilisateur a soumis le formulaire)
    if request.method == 'POST':
        
        # --- Tâche 1 : Récupération des Données Météo ---
        try:
            # Doit correspondre EXACTEMENT aux attributs 'name' de meteo_form.html
            temp = float(request.POST.get('Temperature_C')) 
            hum = float(request.POST.get('Humidite_%'))
            vent = float(request.POST.get('Vent_kmh'))
            pression = float(request.POST.get('Pression_hPa'))
            
            
        except (ValueError, TypeError):
            print("Erreur : Données d'entrée non valides.")
            return render(request, 'erreur_modele.html') # Pensez à créer ce template

        # --- Tâche 2 : Chargement des Modèles Météo ---
        
        # Doit correspondre aux noms de fichiers de 'creer_modele.py'
        model = load_models('LogisticRegression.pkl')
        scaler = load_models('c_scaler.pkl') 
        # (Si vous avez un encodeur, chargez-le aussi)
        # encoder = load_models('meteo_encoder.pkl') 
        
        if model is None or scaler is None:
            print("Erreur : Chargement des fichiers .pkl a échoué.")
            return render(request, 'erreur_modele.html')

        # --- Tâche 3 : Préparation des Données ---
        
        # 1. (Si vous avez un encodeur) : Transformez les données texte en chiffres
        # ...
        
        # 2. Créer un tableau 2D pour le scaler
        # L'ORDRE DOIT ÊTRE EXACTEMENT LE MÊME QUE LORS DE L'ENTRAÎNEMENT
        donnees_brutes = np.array([[temp, hum, vent, pression]]) 
        
        # 3. Appliquer la transformation (normalisation)
        donnees_scalees = scaler.transform(donnees_brutes)

        # --- Tâche 4 : Exécution de la Prédiction ---
        prediction = model.predict(donnees_scalees)
        predicted_class = prediction[0] # ex: 0 ou 1
        
        # --- Tâche 5 : Interprétation des Résultats ---
        # 'non' = 'Non Pluie', 'oui' = 'Pluie'
        type_prediction = {'non':'Non Pluie', 'oui':'Pluie'}
        img_url = {'Non Pluie':'images/soleil.png', 'Pluie':'images/pluie.jpg'} 
        
        pred_texte = type_prediction.get(predicted_class, "Inconnu")
        pred_img = img_url.get(pred_texte)

        # --- Tâche 6 : Préparation du Contexte pour la Réponse ---
        input_data = {
            'Température (°C)': temp,
            'Humidité (%)': hum,
            'Vent (km/h)': vent,
            'Pression (hPa)': pression
        }
        context = {
            'prediction_texte': pred_texte,
            'prediction_image': pred_img,
            'initial_data': input_data 
        }
        
        # On affiche une NOUVELLE page de résultats
        return render(request, 'regLog_results.html', context)
        
    # Logique pour une requête GET (l'utilisateur accède à la page)
    return render(request, 'meteo_form.html')

def ran_forest_details(request):
    return render(request, 'ran_forest_details.html')
def ran_forest_atelier(request):
    return render(request, 'ran_forest_atelier.html')
def ran_forest_tester(request):
    return render(request, 'rain_form.html')
#RF REGRESSION
def ran_forest_reg_details(request):
    return render(request, 'ran_forest_reg_details.html')
def ran_forest_reg_atelier(request):
    return render(request, 'ran_forest_reg_atelier.html')
def ran_forest_reg_tester(request):
    return render(request, 'student_form.html')
from django.shortcuts import render

def rf_prediction(request):
    """
    Gère la prédiction pluie / pas de pluie.
    """

    if request.method == 'POST':

        # --- 1️⃣ Récupération des données du formulaire ---
        try:
            temperature = float(request.POST.get('temperature'))
            humidite = float(request.POST.get('humidite'))
            vent = float(request.POST.get('vent'))
            pression = float(request.POST.get('pression'))
        except (ValueError, TypeError):
            print("Erreur : données invalides.")
            return render(request, 'erreur_modele.html')

        # --- 2️⃣ Charger le modèle Random Forest ---
        model = load_models('rf_classification.pkl')

        if model is None:
            print("Erreur : Impossible de charger random_forest.pkl")
            return render(request, 'erreur_modele.html')

        # --- 3️⃣ Construire la DataFrame pour la prédiction ---
        features = ["Temperature_C", "Humidite_%", "Vent_kmh", "Pression_hPa"]
        entree = pd.DataFrame(
            [[temperature, humidite, vent, pression]],
            columns=features
        )

        # --- 4️⃣ Prédiction ---
        prediction = model.predict(entree)[0]

        # Interprétation
        resultat = "Pluie 🌧️" if prediction == 1 else "Pas de pluie ☀️"
        image = "images/rain.png" if prediction == 1 else "images/sun.png"

        # --- 5️⃣ Contexte pour le template ---
        context = {
            "resultat": resultat,
            "img": image,
            "input_data": {
                "temperature": temperature,
                "humidite": humidite,
                "vent": vent,
                "pression": pression
            }
        }

        return render(request, 'rain_results.html', context)

    # GET -> afficher le formulaire
    return render(request, 'rain_form.html')
#prediction pour regression 

# -------------------------------
# View : prédiction Performance Index
# -------------------------------
def rf_student_prediction(request):
    """
    Gère la prédiction du Performance Index d'un étudiant.
    """

    if request.method == 'POST':
        try:
            hours_studied = float(request.POST.get('hours_studied'))
            previous_scores = float(request.POST.get('previous_scores'))
            extracurricular = request.POST.get('extracurricular')
            sleep_hours = float(request.POST.get('sleep_hours'))
            sample_papers = float(request.POST.get('sample_papers'))

            # Convertir Yes/No en 1/0
            extracurricular = 1 if extracurricular == "Yes" else 0

        except (ValueError, TypeError):
            print("Erreur : données invalides.")
            return render(request, 'erreur_modele.html')

        # Charger le modèle sauvegardé
        model = load_model('rf_reg_model.pkl')
        
        # Construire la DataFrame pour la prédiction
        features = [
            "Hours Studied",
            "Previous Scores",
            "Extracurricular Activities",
            "Sleep Hours",
            "Sample Question Papers Practiced"
        ]
        entree = pd.DataFrame(
            [[hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers]],
            columns=features
        )

        # Prédiction
        prediction = round(model.predict(entree)[0], 2)

        # Interprétation
        if prediction >= 80:
            niveau = "Excellent ⭐⭐⭐"
            img = "images/study.jpg"
        elif prediction >= 60:
            niveau = "Bon 👍"
            img = "images/study.jpg"
        elif prediction >= 40:
            niveau = "Moyen 😕"
            img = "images/student.jpeg"
        else:
            niveau = "Faible ⚠️"
            img = "images/badstudent.jpg"

        # Préparer les données à afficher dans le template
        input_data_display = {
            "hours_studied": hours_studied,
            "previous_scores": previous_scores,
            "extracurricular": "Yes" if extracurricular == 1 else "No",
            "sleep_hours": sleep_hours,
            "sample_papers": sample_papers
        }

        context = {
            "prediction": prediction,  # score réel de la régression
            "resultat": niveau,        # texte interprété
            "img": img,                # image correspondante
            "input_data": input_data_display
        }

        return render(request, 'student_results.html', context)

    # GET -> afficher le formulaire
    return render(request, 'student_form.html')

def XGBoost_details(request):
    return render(request, 'XGBoost_details.html')
def XGBoost_atelier(request):
    return render(request, 'XGBoost_atelier.html')
def XGboost_tester(request):
    return render(request, 'xgrain_form.html')
#XGBOOST PREDICTION LOGIC 
def XGboost_prediction(request):
    """
    Gère la prédiction pluie / pas de pluie.
    """

    if request.method == 'POST':

        # --- 1️⃣ Récupération des données du formulaire ---
        try:
            temperature = float(request.POST.get('temperature'))
            humidite = float(request.POST.get('humidite'))
            vent = float(request.POST.get('vent'))
            pression = float(request.POST.get('pression'))
        except (ValueError, TypeError):
            print("Erreur : données invalides.")
            return render(request, 'erreur_modele.html')

        # --- 2️⃣ Charger le modèle Random Forest ---
        model = load_models('XGboost_class.pkl')

        if model is None:
            print("Erreur : Impossible de charger random_forest.pkl")
            return render(request, 'erreur_modele.html')

        # --- 3️⃣ Construire la DataFrame pour la prédiction ---
        features = ["Temperature_C", "Humidite_%", "Vent_kmh", "Pression_hPa"]
        entree = pd.DataFrame(
            [[temperature, humidite, vent, pression]],
            columns=features
        )

        # --- 4️⃣ Prédiction ---
        prediction = model.predict(entree)[0]

        # Interprétation
        resultat = "Pluie 🌧️" if prediction == 1 else "Pas de pluie ☀️"
        image = "images/rain.png" if prediction == 1 else "images/sun.png"

        # --- 5️⃣ Contexte pour le template ---
        context = {
            "resultat": resultat,
            "img": image,
            "input_data": {
                "temperature": temperature,
                "humidite": humidite,
                "vent": vent,
                "pression": pression
            }
        }

        return render(request, 'xgrain_results.html', context)

    # GET -> afficher le formulaire
    return render(request, 'xgrain_form.html')



# Mapping entre le texte tapé et les URLs
ALGO_URLS = {
    "randomforest": "/randomforest/",
    "xgboost": "/xgboost/",
}


def recherche_algo_view(request):
    algo_tape = request.GET.get('algo', '').lower().replace(" ", "")
    if algo_tape in ALGO_URLS:
        return redirect(ALGO_URLS[algo_tape])
    elif algo_tape:
        message = "Algorithme non trouvé !"
    else:
        message = ""
    return render(request, 'app/index.html', {'message': message})
def randomforest_view(request):
    return render(request, 'app/ran_forest_atelier.html', {
        'algo_nom': 'randomforest'
    })
def about(request):
    return render(request, 'about.html')