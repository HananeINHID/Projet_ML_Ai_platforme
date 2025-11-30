import os
import joblib
import numpy as np
import pandas as pd
import pickle
from django.conf import settings
from django.shortcuts import render

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

def load_models(name):
    """
    Charge un fichier .pkl (modèle ou scaler) depuis le dossier 'models_ai'.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    app_dir = os.path.dirname(base_dir) 
    models_dir = os.path.join(app_dir, 'models_ai')
    
    model_path = os.path.join(models_dir, name)
    
    if not os.path.exists(model_path):
        print(f"ERREUR CRITIQUE: Fichier modèle non trouvé : {model_path}")
        return None
        
    ml_model = joblib.load(model_path)
    return ml_model

# --- Vues Statiques ---

def index(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

# --- RÉGRESSION LOGISTIQUE ---

def regLog_details(request):
    return render(request, 'regLog_details.html')

def regLog_atelier(request):
    return render(request, 'regLog_atelier.html')

def regLog_tester(request):
    return render(request, 'meteo_form.html')

def regLog_prediction(request):
    if request.method == 'POST':
        temp = float(request.POST.get('Temperature_C')) 
        hum = float(request.POST.get('Humidite_%'))
        vent = float(request.POST.get('Vent_kmh'))
        pression = float(request.POST.get('Pression_hPa'))
            
        model = load_models('LogisticRegression.pkl')
        scaler = load_models('LogisticRegression_scaler.pkl') 
        
        if model is None or scaler is None:
            print("Erreur : Chargement des fichiers .pkl a échoué.")
            return render(request, 'meteo_form.html')
        
        donnees_brutes = np.array([[temp, hum, vent, pression]]) 
        donnees_scalees = scaler.transform(donnees_brutes)
        prediction = model.predict(donnees_scalees)
        predicted_class = prediction[0] 
        
        type_prediction = {0: 'Non Pluie', 1: 'Pluie'}
        pred_texte = type_prediction.get(predicted_class, "Inconnu")
        img_path = ""
        
        if pred_texte == "Pluie":
            img_path = 'images/pluie.jpg'
        else:
            if temp <= 5:
                img_path = 'images/froid.jpeg'
            elif temp <=15:
                img_path = 'images/temps_cloudy.webp'
            else:
                img_path = 'images/soleil.png'
            
        input_data = {
            'Température (°C)': temp,
            'Humidité (%)': hum,
            'Vent (km/h)': vent,
            'Pression (hPa)': pression
        }
        context = {
            'prediction_texte': pred_texte,
            'prediction_image': img_path,
            'initial_data': input_data 
        }
        
        return render(request, 'regLog_results.html', context)
        
    return render(request, 'meteo_form.html')

# --- ARBRE DE DÉCISION (RÉGRESSION) ---

def tree_r_details(request):
    return render(request, 'tree_r_details.html')

def tree_r_atelier(request):
    return render(request, 'tree_r_atelier.html')

def tree_r_tester(request):
    return render(request, 'tree_r_form.html', {'form_action': 'tree_r_prediction'})

def tree_r_prediction(request):
    if request.method == 'POST':
        hours_studied = float(request.POST.get('hours_studied'))
        previous_score = float(request.POST.get('previous_scores'))
        extr_activities = float(request.POST.get('extracurricular'))
        sleep_hours = float(request.POST.get('sleep_hours'))
        sample_questions = float(request.POST.get('sample_papers'))

        model = load_models('DecisionTreeRegressor.pkl')
        if model is None:
            print("Erreur : Chargement du fichier .pkl a échoué.")
            return render(request, 'tree_r_form.html')

        input_data = np.array([[hours_studied, previous_score, extr_activities, sleep_hours, sample_questions]])
        prediction = model.predict(input_data)
        predicted_score = prediction[0]
        
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
        return render(request, 'tree_r_results.html', context)
    
    return render(request, 'tree_r_form.html')

# --- ARBRE DE DÉCISION (CLASSIFICATION) ---

def tree_c_details(request):
    return render(request, 'tree_c_details.html')

def tree_c_atelier(request):
    return render(request, 'tree_c_atelier.html')

def tree_c_tester(request):
    return render(request, 'tree_c_form.html')

def tree_c_prediction(request):
    if request.method == 'POST':
        try:
            temperature = float(request.POST.get('Temperature_C'))
            humidity = float(request.POST.get('Humidite_%'))
            wind_speed = float(request.POST.get('Vent_kmh'))
            pressure = float(request.POST.get('Pression_hPa'))
        except (ValueError, TypeError):
            return render(request, 'tree_c_form.html', {'error': "Veuillez saisir des valeurs valides."})

        model_tree_c = load_models('model_tree_c.pkl')
        if model_tree_c is None:
            return render(request, 'tree_c_form.html', {'error': "Erreur de chargement du modèle."})

        raw_data = np.array([[temperature, humidity, wind_speed, pressure]])
        prediction = model_tree_c.predict(raw_data)
        predicted_class = prediction[0]

        type_prediction = {0: 'Non Pluie', 1: 'Pluie'}
        pred_texte = type_prediction.get(predicted_class, "Inconnu")

        if pred_texte == "Pluie":
            img_path = 'images/pluie.jpg'
        else:
            if temperature <= 5:
                img_path = 'images/froid.jpeg'
            elif temperature <= 15:
                img_path = 'images/temps_cloudy.webp'
            else:
                img_path = 'images/soleil.png'

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

        return render(request, 'tree_c_results.html', context)

    return render(request, 'tree_c_form.html')

# --- SVM (CLASSIFICATION) ---

def svm_c_details(request):
    return render(request, 'svm_c_details.html')

def svm_c_atelier(request):
    return render(request, 'svm_c_atelier.html')

def svm_c_tester(request):
    return render(request, 'svm_c_form.html')

def svm_c_prediction(request):
    if request.method == 'POST':
        temperature = float(request.POST.get('Temperature_C'))
        humidity = float(request.POST.get('Humidite_%'))
        wind_speed = float(request.POST.get('Vent_kmh'))
        pressure = float(request.POST.get('Pression_hPa'))
        
        model = load_models('SVM_Classifier.pkl')
        scaler = load_models('SVM_Classifier_scaler.pkl')
        
        if model is None or scaler is None:
            print("Erreur : Chargement des fichiers .pkl a échoué.")
            return render(request, 'svm_c_form.html')
        
        raw_data = np.array([[temperature, humidity, wind_speed, pressure]])
        scaled_data = scaler.transform(raw_data)
        prediction = model.predict(scaled_data)
        predicted_class = prediction[0]
        
        type_prediction = {0: 'Non Pluie', 1: 'Pluie'}
        pred_texte = type_prediction.get(predicted_class, "Inconnu")
        img_path = ""
        
        if pred_texte == "Pluie":
            img_path = 'images/pluie.jpg'
        else:
            if temperature <= 5:
                img_path = 'images/froid.jpeg'
            elif temperature <=15:
                img_path = 'images/temps_cloudy.webp'
            else:
                img_path = 'images/soleil.png'
            
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
        return render(request, 'svm_c_results.html', context)
    
    return render(request, 'svm_c_form.html')

# --- SVM (RÉGRESSION) ---

# Fonction pour charger le modèle
def load_model(filename):
    try:
        model_path = os.path.join('algoAI', 'models', filename)  # Assure-toi que le chemin est correct
        return joblib.load(model_path)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return None


def svm_r_details(request):
    return render(request, 'svm_r_details.html')


def svm_r_atelier(request):
    return render(request, 'svm_r_atelier.html')


def svm_r_tester(request):
    return render(request, 'svm_r_form.html')


def svm_r_prediction(request):
    if request.method == 'POST':
        try:
            hours_studied = float(request.POST.get('hours_studied'))
            previous_scores = float(request.POST.get('previous_scores'))
            extracurricular = request.POST.get('extracurricular')
            sleep_hours = float(request.POST.get('sleep_hours'))
            sample_papers = float(request.POST.get('sample_papers'))

            # Conversion en 0 ou 1
            extracurricular = 1 if extracurricular == "Yes" else 0

        except (ValueError, TypeError):
            return render(request, 'erreur_modele.html')

        # Chargement du modèle SVM avec joblib
        model_svm_r = load_model('svm_regression_model.joblib')
        if model_svm_r is None:
            return render(request, 'erreur_modele.html')

        # Préparer les données pour la prédiction
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

        # Faire la prédiction
        predicted_score = round(model_svm_r.predict(entree)[0], 2)

        if predicted_score >= 80:
            niveau = "Excellent ⭐⭐⭐"
        elif predicted_score >= 60:
            niveau = "Bon 👍"
        elif predicted_score >= 40:
            niveau = "Moyen 😕"
        else:
            niveau = "Faible ⚠️"

        input_data_display = {
            "Hours Studied": hours_studied,
            "Previous Scores": previous_scores,
            "Extracurricular Activities": "Yes" if extracurricular == 1 else "No",
            "Sleep Hours": sleep_hours,
            "Sample Question Papers Practiced": sample_papers
        }

        context = {
            "predicted_score": predicted_score,
            "niveau": niveau,
            "initial_data": input_data_display
        }

        return render(request, 'svm_r_results.html', context)

    return render(request, 'svm_r_form.html')

# --- XGBOOST (RÉGRESSION) ---

def xgboost_r_details(request):
    return render(request, 'xgboost_r_details.html')

def xgboost_r_atelier(request):
    return render(request, 'xgboost_r_atelier.html')

def xgboost_r_tester(request):
    return render(request, 'xgboost_r_form.html')

def xgboost_r_prediction(request):
    if request.method == 'POST':
        hours_studied = float(request.POST.get('hours_studied'))
        previous_score = float(request.POST.get('previous_scores'))
        extr_activities = float(request.POST.get('extracurricular'))
        sleep_hours = float(request.POST.get('sleep_hours'))
        sample_questions = float(request.POST.get('sample_papers'))

        model = load_models('XGBoostRegressor.pkl')
        if model is None:
            print("Erreur : Chargement du fichier .pkl a échoué.")
            return render(request, 'xgboost_r_form.html')

        input_data = np.array([[hours_studied, previous_score, extr_activities, sleep_hours, sample_questions]])
        prediction = model.predict(input_data)
        predicted_score = prediction[0]
        
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
        return render(request, 'xgboost_r_results.html', context)
    return render(request, 'xgboost_r_form.html')

# --- XGBOOST (CLASSIFICATION) ---

def XGBoost_details(request):
    return render(request, 'XGBoost_details.html')

def XGBoost_atelier(request):
    return render(request, 'XGBoost_atelier.html')

def XGboost_tester(request):
    return render(request, 'xgrain_form.html')

def XGboost_prediction(request):
    if request.method == 'POST':
        try:
            temperature = float(request.POST.get('temperature'))
            humidite = float(request.POST.get('humidite'))
            vent = float(request.POST.get('vent'))
            pression = float(request.POST.get('pression'))
        except (ValueError, TypeError):
            print("Erreur : données invalides.")
            return render(request, 'erreur_modele.html')

        model = load_models('XGboost_class.pkl')
        if model is None:
            print("Erreur : Impossible de charger XGboost_class.pkl")
            return render(request, 'erreur_modele.html')

        features = ["Temperature_C", "Humidite_%", "Vent_kmh", "Pression_hPa"]
        entree = pd.DataFrame(
            [[temperature, humidite, vent, pression]],
            columns=features
        )

        prediction = model.predict(entree)[0]
        resultat = "Pluie 🌧️" if prediction == 1 else "Pas de pluie ☀️"
        image = "images/rain.png" if prediction == 1 else "images/sun.png"

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

    return render(request, 'xgrain_form.html')

# --- RANDOM FOREST (CLASSIFICATION) ---

def ran_forest_details(request):
    return render(request, 'ran_forest_details.html')

def ran_forest_atelier(request):
    return render(request, 'ran_forest_atelier.html')

def ran_forest_tester(request):
    return render(request, 'rain_form.html')

def rf_prediction(request):
    if request.method == 'POST':
        try:
            temperature = float(request.POST.get('temperature'))
            humidite = float(request.POST.get('humidite'))
            vent = float(request.POST.get('vent'))
            pression = float(request.POST.get('pression'))
        except (ValueError, TypeError):
            print("Erreur : données invalides.")
            return render(request, 'erreur_modele.html')

        model = load_models('rf_classification.pkl')
        if model is None:
            print("Erreur : Impossible de charger rf_classification.pkl")
            return render(request, 'erreur_modele.html')

        features = ["Temperature_C", "Humidite_%", "Vent_kmh", "Pression_hPa"]
        entree = pd.DataFrame(
            [[temperature, humidite, vent, pression]],
            columns=features
        )

        prediction = model.predict(entree)[0]
        resultat = "Pluie 🌧️" if prediction == 1 else "Pas de pluie ☀️"
        image = "images/rain.png" if prediction == 1 else "images/sun.png"

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

    return render(request, 'rain_form.html')

# --- RANDOM FOREST (RÉGRESSION) ---

def ran_forest_reg_details(request):
    return render(request, 'ran_forest_reg_details.html')

def ran_forest_reg_atelier(request):
    return render(request, 'ran_forest_reg_atelier.html')

def ran_forest_reg_tester(request):
    return render(request, 'student_form.html')

def rf_student_prediction(request):
    if request.method == 'POST':
        try:
            hours_studied = float(request.POST.get('hours_studied'))
            previous_scores = float(request.POST.get('previous_scores'))
            extracurricular = request.POST.get('extracurricular')
            sleep_hours = float(request.POST.get('sleep_hours'))
            sample_papers = float(request.POST.get('sample_papers'))

            extracurricular = 1 if extracurricular == "Yes" else 0

        except (ValueError, TypeError):
            print("Erreur : données invalides.")
            return render(request, 'erreur_modele.html')

        model = load_model('rf_reg_model.pkl')
        if model is None:
            print("Erreur : Impossible de charger rf_reg_model.pkl")
            return render(request, 'erreur_modele.html')
        
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

        prediction = round(model.predict(entree)[0], 2)

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

    return render(request, 'student_form.html')

# --- RÉGRESSION LINÉAIRE ---

def linreg_details(request):
    return render(request, 'linreg_details.html')

def linreg_atelier(request):
    return render(request, 'linreg_atelier.html')

def linreg_tester(request):
    return render(request, 'linreg_form.html')

def linreg_prediction(request):
    if request.method == 'POST':
        try:
            hours_studied = float(request.POST.get('hours_studied'))
            previous_scores = float(request.POST.get('previous_scores'))
            extracurricular = request.POST.get('extracurricular')
            sleep_hours = float(request.POST.get('sleep_hours'))
            sample_papers = float(request.POST.get('sample_papers'))

            extracurricular = 1 if extracurricular == "Yes" else 0

        except (ValueError, TypeError):
            return render(request, 'linreg_form.html', {"error": "Données invalides."})

        # Construire un chemin relatif pour le modèle
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # dossier my_aiplatform/algoAI
        model_path = os.path.join(base_dir, "models_ai", "linear_regression_model.pkl")

        # Vérifier que le fichier existe
        if not os.path.exists(model_path):
            return render(request, 'linreg_form.html', {"error": "Modèle non trouvé."})

        # Charger le modèle avec joblib
        try:
            model = joblib.load(model_path)
        except Exception as e:
            return render(request, 'linreg_form.html', {"error": f"Erreur lors du chargement du modèle : {e}"})

        # Préparer les données pour la prédiction
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

        # Catégorisation
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

        return render(request, 'linreg_results.html', context)

    return render(request, 'linreg_form.html')