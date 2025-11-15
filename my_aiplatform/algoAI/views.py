from django.shortcuts import render
import joblib  # Utilisé pour charger les modèles .pkl (modèle ML et scaler)
import os      # Utilisé pour la manipulation des chemins de fichiers (os.path, os.path.exists)
import numpy as np # Utilisé pour créer des tableaux (arrays) pour la prédiction

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
        # Doit correspondre EXACTEMENT aux attributs 'name' de meteo_form.html
        temp = float(request.POST.get('Temperature_C')) 
        hum = float(request.POST.get('Humidite_%'))
        vent = float(request.POST.get('Vent_kmh'))
        pression = float(request.POST.get('Pression_hPa'))
            
        # --- Tâche 2 : Chargement des Modèles Météo ---
        
        # Doit correspondre aux noms de fichiers de 'creer_modele.py'
        model = load_models('LogisticRegression.pkl')
        scaler = load_models('c_scaler.pkl') 
        # (Si vous avez un encodeur, chargez-le aussi)
        # encoder = load_models('meteo_encoder.pkl') 
        
        if model is None or scaler is None:
            print("Erreur : Chargement des fichiers .pkl a échoué.")
            return 
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


#  VUES POUR L'ARBRE DE DÉCISION (CLASSIFICATION)

def tree_c_details(request):
    """Affiche la page de détails sur l'Arbre de Décision (Classification)."""
    return render(request, 'tree_c_details.html')

def tree_c_atelier(request):
    """Affiche la page d'atelier pour l'Arbre de Décision (Classification)."""
    return render(request, 'tree_c_atelier.html')

def tree_c_tester(request):
    """Affiche le formulaire de test pour l'Arbre de Décision (Classification)."""
    return render(request, 'tree_c_form.html')

def tree_c_prediction(request):
    """Gère la prédiction de l'Arbre de Décision (Classification)."""
    
    if request.method == 'POST':
        # Récupération des données
        temp = float(request.POST.get('Temperature_C'))
        hum = float(request.POST.get('Humidite_%'))
        vent = float(request.POST.get('Vent_kmh'))
        pression = float(request.POST.get('Pression_hPa'))
            

        # Chargement du modèle Arbre de Décision
        model = load_models('DecisionTreeClassifier.pkl')
        
        if model is None:
            return render(request,{'message': 'Modèle Arbre de Décision non trouvé'})

        # Préparation des données
        donnees_brutes = np.array([[temp, hum, vent, pression]])
        
        # Prédiction
        prediction = model.predict(donnees_brutes)
        predicted_class = prediction[0]
        
        # Interprétation
        type_prediction = {'non': 'Non Pluie', 'oui': 'Pluie'}
        img_url = {'Non Pluie': 'images/soleil.png', 'Pluie': 'images/pluie.jpg'}
        
        pred_texte = type_prediction.get(predicted_class, "Inconnu")
        pred_img = img_url.get(pred_texte)

        # Contexte
        input_data = {
            'Température (°C)': temp,
            'Humidité (%)': hum,
            'Vent (km/h)': vent,
            'Pression (hPa)': pression
        }
        
        context = {
            'prediction_texte': pred_texte,
            'prediction_image': pred_img,
            'initial_data': input_data,
            'modele_utilise': 'Arbre de Décision' 
        }
        
        return render(request, 'regLog_results.html', context)
    
    return render(request, 'tree_c_form.html')

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
    
    return(request, 'svm_c_form.html')
    