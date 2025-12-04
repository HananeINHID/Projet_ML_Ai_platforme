# Projet ML AI Plateforme

Bienvenue sur **Projet_ML_Ai_platforme** : une plateforme web interactive conçue pour tester, visualiser et comparer facilement **10 algorithmes de Machine Learning** (classification & régression). Ce projet a été développé dans le cadre du cours d'Intelligence Artificielle (2025/26), encadré par le Pr. Mohammed AMEKSA.

---

## Fonctionnalités principales

- **Interface web moderne et intuitive**
- **Formulaires interactifs** pour la saisie des données utilisateur
- **Prédictions instantanées** avec affichage des résultats en temps réel
- **Visualisation graphique** des résultats et des performances
- **Exportation PDF** du rapport de prédiction
- **Comparaison de multiples algorithmes**

---

## Algorithmes Implémentés

- **Régression Logistique**
- **Régression Linéaire**
- **Arbre de Décision** (classification & régression)
- **SVM** (classification & régression)
- **Random Forest** (classification & régression)
- **XGBoost** (classification & régression)

> *Tous les modèles sont entraînés et sauvegardés dans le dossier dédié, prêts à l'emploi via la plateforme !*

---

## Installation rapide

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/HananeINHID/Projet_ML_Ai_platforme.git
   cd Projet_ML_Ai_platforme
   ```

2. **Créer un environnement virtuel :**
   - Windows :
     ```bash
     python -m venv envML
     envML\Scripts\activate
     ```
   - Mac/Linux :
     ```bash
     python -m venv envML
     source envML/bin/activate
     ```

3. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
   ```

4. **Exécuter les migrations Django :**
   ```bash
   python manage.py migrate
   ```

5. **Lancer le serveur local :**
   ```bash
   python manage.py runserver
   ```

6. **Accéder à l'application :**  
   Ouvrez [http://127.0.0.1:8000/](http://127.0.0.1:8000/) dans votre navigateur.

---

## Structure du projet

```
├── algoAI/             # Vues Django, urls, logique de l’app
├── models_ai/          # Modèles ML entraînés (.pkl)
├── templates/          # Fichiers HTML (UI)
├── static/             # Fichiers statiques : images, PDF, ...
├── requirements.txt    # Dépendances Python
```

---

## Auteurs

- **Inhid Hanane**
- **EL Boudhiri Khadija**
- **El Angui Salma**

_Année Universitaire : 2025/2026_

---

N’hésitez pas à contribuer ou à signaler des issues ! Bon test et exploration des algos ML 🚀🤖