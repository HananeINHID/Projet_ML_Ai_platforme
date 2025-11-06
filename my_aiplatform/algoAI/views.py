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
    Affiche le formulaire de test (vehicles_from.html) pour la prédiction.
    """
    return render(request, 'vehicles_from.html')

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
    Gère la logique de prédiction du type de véhicule.

    - Si la méthode est GET : Affiche le formulaire de saisie.
    - Si la méthode est POST : Traite les données soumises, effectue la prédiction
                              et affiche la page de résultats.
    """
    
    # Logique pour une requête POST (l'utilisateur a soumis le formulaire)
    if request.method == 'POST':
        
        # --- Tâche 1 : Récupération et Nettoyage des Données ---
        # On récupère les valeurs du formulaire (attribut 'name' des inputs)
        # On les convertit en 'float' pour les calculs mathématiques.
        try:
            hauteur = float(request.POST.get('hauteur')) 
            nbr_roues = float(request.POST.get('Nombre_de_roues'))
        except (ValueError, TypeError):
            # Gérer le cas où les données ne sont pas des nombres valides
            print("Erreur : Données d'entrée non valides.")
            return render(request, 'erreur_modele.html')

        # --- Tâche 2 : Chargement des Modèles ---
        model = load_models('logreg_model.pkl')
        scaler = load_models('scaler.pkl') 
        
        # Sécurité : Vérifier que les deux modèles sont bien chargés
        if model is None or scaler is None:
            # Si les fichiers .pkl sont introuvables, on affiche une page d'erreur
            print("Erreur : Chargement des fichiers .pkl a échoué.")
            return render(request, 'erreur_modele.html')

        # --- Tâche 3 : Standardisation des Données (Scaling) ---
        # Le modèle a été entraîné sur des données "standardisées" (scalées).
        # Il est OBLIGATOIRE d'appliquer la *même* transformation (le 'scaler')
        # aux nouvelles données avant de faire une prédiction.
        
        # 1. Créer un tableau 2D, car le scaler attend cette structure
        donnees_brutes = np.array([[hauteur, nbr_roues]]) 
        
        # 2. Appliquer la transformation
        donnees_scalees = scaler.transform(donnees_brutes)

        # --- Tâche 4 : Exécution de la Prédiction ---
        # On fournit les données standardisées au modèle
        prediction = model.predict(donnees_scalees)
        predicted_class = prediction[0] # On extrait la prédiction (ex: 0 ou 1)
        
        # --- Tâche 5 : Interprétation des Résultats ---
        # On "traduit" la sortie numérique du modèle (0 ou 1) en
        # une réponse compréhensible par l'utilisateur.
        type_vehicules = {0:'Camion', 1:'Touristique'}
        img_url = {'Camion':'images/camion.jpg', 'Touristique':'images/touristique.jpg'}
        
        # Utiliser .get() est plus sûr que l'accès direct [key]
        pred_vehicule = type_vehicules.get(predicted_class, "Inconnu")
        pred_img = img_url.get(pred_vehicule)

        # --- Tâche 6 : Préparation du Contexte pour la Réponse ---
        # On regroupe toutes les informations à envoyer au template de résultats.
        input_data = {
            'hauteur':hauteur,
            'nbr_roues':nbr_roues
        }
        context = {
            'type_vehicule': pred_vehicule,
            'img_vehicule': pred_img,
            'initial_data': input_data 
        }
        
        # On affiche la page de résultats avec les informations du contexte
        return render(request, 'regLog_results.html', context)
        
    # Logique pour une requête GET (l'utilisateur accède à la page pour la 1ère fois)
    # On affiche simplement le formulaire de saisie.
    return render(request, 'vehicles_from.html')