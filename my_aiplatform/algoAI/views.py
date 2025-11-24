from django.shortcuts import render
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

# VUE À PROPOS
def about(request):
    """Affiche la page À Propos (Équipe, Version, Contact)."""
    return render(request, 'about.html')

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
        # Doit correspondre EXACTEMENT aux attributs 'name' de meteo_form.html
        temp = float(request.POST.get('Temperature_C')) 
        hum = float(request.POST.get('Humidite_%'))
        vent = float(request.POST.get('Vent_kmh'))
        pression = float(request.POST.get('Pression_hPa'))
            
        # --- Tâche 2 : Chargement des Modèles Météo ---
        
        # Doit correspondre aux noms de fichiers de 'creer_modele.py'
        model = load_models('LogisticRegression.pkl')
        scaler = load_models('LogisticRegression_scaler.pkl') 
        
        if model is None or scaler is None:
            print("Erreur : Chargement des fichiers .pkl a échoué.")
            return 
        # --- Tâche 3 : Préparation des Données ---
        
        # 1. Créer un array numpy avec les données brutes
        donnees_brutes = np.array([[temp, hum, vent, pression]]) 
        
        # 3. Appliquer la transformation (normalisation)
        donnees_scalees = scaler.transform(donnees_brutes)

        # --- Tâche 4 : Exécution de la Prédiction ---
        prediction = model.predict(donnees_scalees)
        predicted_class = prediction[0] 
        
        # --- Tâche 5 : Interprétation des Résultats ---
        type_prediction = {0: 'Non Pluie', 1: 'Pluie'}
        img_path =""
        
        pred_texte = type_prediction.get(predicted_class, "Inconnu")
        
        if pred_texte == "Pluie":
            img_path = 'images/pluie.jpg'
        else:
            if temp <= 5 :
                img_path = 'images/froid.jpeg'
            elif temp <=15 :
                img_path = 'images/temps_cloudy.webp'
            else:
                img_path = 'images/soleil.png'
            
        
        pred_img = img_path

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

#  VUES POUR L'ARBRE DE DÉCISION (REGRESSION)

def tree_r_details(request):
    """Affiche la page de détails sur l'Arbre de Décision (Régression)."""
    return render(request, 'tree_r_details.html')

def tree_r_atelier(request):
    """Affiche la page d'atelier pour l'Arbre de Décision (Régression)."""
    return render(request, 'tree_r_atelier.html')

def tree_r_tester(request):
    """Affiche le formulaire de test pour l'Arbre de Décision (Régression)."""
    return render(request, 'tree_r_form.html')

def tree_r_prediction(request):
    """Gère la prédiction de l'Arbre de Décision (Régression)."""
    
    if request.method == 'POST':
        # Récupération des données
        hours_studied = float(request.POST.get('hours_studied'))
        previous_score = float(request.POST.get('previous_scores'))
        extr_activities = float(request.POST.get('extracurricular'))
        sleep_hours = float(request.POST.get('sleep_hours'))
        sample_questions = float(request.POST.get('sample_papers'))

        # Chargement du modèle Arbre de Décision
        model = load_models('DecisionTreeRegressor.pkl')
        if model is None:
            print("Erreur : Chargement du fichier .pkl a échoué.")
            return render(request, 'tree_r_form.html')

        # Préparation des données
        input_data = np.array([[hours_studied, previous_score, extr_activities, sleep_hours, sample_questions]])
        # Exécution de la prédiction
        prediction = model.predict(input_data)
        predicted_score = prediction[0]  # Valeur prédite
        # Préparation du contexte pour la réponse
        input_features = {
            'Hours_Studied': hours_studied,
            'Previous_Scores': previous_score,
            'Extracurricular_Activities': extr_activities,
            'Sleep_Hours': sleep_hours,
            'Sample_Question_Papers_Practiced': sample_questions
        }
        context = {
            'predicted_score': predicted_score,
            'initial_data': input_features
        }
        # Affichage de la page de résultats
        return render(request, 'tree_r_results.html', context)
    
    return render(request, 'tree_r_form.html')

# VUES POUR SVM (CLASSIFICATION)

def svm_c_details(request):
    """
    Affiche la page de détails sur l'algorithme SVM.
    """
    return render(request, 'svm_c_details.html')

def svm_c_atelier(request):
    """
    Affiche la page de présentation de l'atelier pratique.
    """
    return render(request, 'svm_c_atelier.html')

def svm_c_tester(request):
    """
    Affiche le formulaire de test (svm_c_form.html) pour la prédiction.
    """
    return render(request, 'svm_c_form.html')

def svm_c_prediction(request):
    """
    Gère la logique de prédiction SVM.
    """
    if request.method == 'POST':
        # Récupération des données du formulaire
        temperature = float(request.POST.get('Temperature_C'))
        humidity = float(request.POST.get('Humidite_%'))
        wind_speed = float(request.POST.get('Vent_kmh'))
        pressure = float(request.POST.get('Pression_hPa'))
        
        # Chargement du modèle SVM
        model = load_models('SVM_Classifier.pkl')
        scaler = load_models('SVM_Classifier_scaler.pkl')
        
        if model is None or scaler is None:
            print("Erreur : Chargement des fichiers .pkl a échoué.")
            return render(request, 'svm_c_form.html')
        
        # Préparation des données pour la prédiction
        raw_data = np.array([[temperature, humidity, wind_speed, pressure]])
        scaled_data = scaler.transform(raw_data)
        # Exécution de la prédiction
        prediction = model.predict(scaled_data)
        predicted_class = prediction[0]
        # Interprétation des résultats
        type_prediction = {0: 'Non Pluie', 1: 'Pluie'}
        pred_texte = type_prediction.get(predicted_class, "Inconnu")
        img_path =""
        
        if pred_texte == "Pluie":
            img_path = 'images/pluie.jpg'
        else:
            if temperature <= 5 :
                img_path = 'images/froid.jpeg'
            elif temperature <=15 :
                img_path = 'images/temps_cloudy.webp'
            else:
                img_path = 'images/soleil.png'
            
        # Préparation du contexte pour la réponse
        input_data = {
            'Température (°C)': temperature,
            'Humidité (%)': humidity,
            'Vent (km/h)': wind_speed,
            'Pression (hPa)': pressure
        }
        context = {
            'prediction_texte': pred_texte,
            'prediction_image': img_path,
            'initial_data': input_data
        }
        # Affichage de la page de résultats
        return render(request, 'svm_c_results.html', context)
    
    return render(request, 'svm_c_form.html')
    
# VUES POUR XGBOOST (REGRESSION)
def xgboost_r_details(request):
    """
    Affiche la page de détails sur l'algorithme XGBoost regression.
    """
    return render(request, 'xgboost_r_details.html')

def xgboost_r_atelier(request):
    """
    Affiche la page de présentation de l'atelier pratique.
    """
    return render(request, 'xgboost_r_atelier.html')

def xgboost_r_tester(request):
    """
    Affiche le formulaire de test (xgboost_r_form.html) pour la prédiction.
    """
    return render(request, 'xgboost_r_form.html')

def xgboost_r_prediction(request):
    if request.method == 'POST':
        # Récupération des données
        hours_studied = float(request.POST.get('hours_studied'))
        previous_score = float(request.POST.get('previous_scores'))
        extr_activities = float(request.POST.get('extracurricular'))
        sleep_hours = float(request.POST.get('sleep_hours'))
        sample_questions = float(request.POST.get('sample_papers'))

        # Chargement du modèle XGBoost Regression
        model = load_models('XGBoostRegressor.pkl')
        if model is None:
            print("Erreur : Chargement du fichier .pkl a échoué.")
            return render(request, 'xgboost_r_form.html')

        # Préparation des données
        input_data = np.array([[hours_studied, previous_score, extr_activities, sleep_hours, sample_questions]])
        # Exécution de la prédiction
        prediction = model.predict(input_data)
        predicted_score = prediction[0]  # Valeur prédite
        # Préparation du contexte pour la réponse
        input_features = {
            'Hours_Studied': hours_studied,
            'Previous_Scores': previous_score,
            'Extracurricular_Activities': extr_activities,
            'Sleep_Hours': sleep_hours,
            'Sample_Question_Papers_Practiced': sample_questions
        }
        context = {
            'predicted_score': predicted_score,
            'initial_data': input_features
        }
    
        # Affichage de la page de résultats
        return render(request, 'xgboost_r_results.html', context)
    return render(request, 'xgboost_r_form.html')

# VUES POUR RANDOM FOREST (CLASSIFICATION)
def ran_forest_details(request):
    return render(request, 'ran_forest_details.html')

def ran_forest_atelier(request):
    return render(request, 'ran_forest_atelier.html')

def ran_forest_tester(request):
    return render(request, 'rain_form.html')

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

# --- VUES POUR LA RÉGRESSION LINÉAIRE ---

def linreg_details(request):
    """Affiche la page de détails sur l'algorithme de Régression Linéaire."""
    return render(request, 'linreg_details.html')

def linreg_atelier(request):
    """Affiche la page d'atelier pour la Régression Linéaire."""
    return render(request, 'linreg_atelier.html')

def linreg_tester(request):
    """Affiche le formulaire de test pour la Régression Linéaire (student_form.html)."""
    return render(request, 'student_form.html')

def linreg_prediction(request):
    """Gère la prédiction avec le modèle de Régression Linéaire."""

    if request.method == 'POST':
        try:
            # Récupération des données depuis le formulaire
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

        # Charger le modèle de Régression Linéaire
        model = load_model('linear_regression_model.pkl')
        if model is None:
            print("Erreur : le fichier linear_regression_model.pkl est introuvable.")
            return render(request, 'erreur_modele.html')

        # Préparation des données sous forme de DataFrame
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

        # Interprétation simple
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

        # Préparer le contexte pour le template
        input_data_display = {
            "hours_studied": hours_studied,
            "previous_scores": previous_scores,
            "extracurricular": "Yes" if extracurricular == 1 else "No",
            "sleep_hours": sleep_hours,
            "sample_papers": sample_papers
        }

        context = {
            "prediction": prediction,
            "resultat": niveau,
            "img": img,
            "input_data": input_data_display
        }

        return render(request, 'student_results.html', context)

    # GET -> afficher le formulaire
    return render(request, 'student_form.html')


# VUES POUR RANDOM FOREST (REGRESSION)
def ran_forest_reg_details(request):
    return render(request, 'ran_forest_reg_details.html')

def ran_forest_reg_atelier(request):
    return render(request, 'ran_forest_reg_atelier.html')

def ran_forest_reg_tester(request):
    return render(request, 'student_form.html')

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

# VUES POUR XGBOOST (CLASSIFICATION)
def XGBoost_details(request):
    return render(request, 'XGBoost_details.html')

def XGBoost_atelier(request):
    return render(request, 'XGBoost_atelier.html')