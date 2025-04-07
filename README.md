---
title: Classifieur
emoji: 👁
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 5.23.3
app_file: app.py
pinned: false
short_description: 'Une application de classification de texte utilisant OpenAI '
---

# BrainBox4 - Système de Classification de Texte

Cette application gradio est un système de classification de texte basé sur l'IA, optimisé pour le traitement rapide de grands volumes de données.

## 🚀 Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/simon-dharcourt/brainbox4.git
cd brainbox4
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Configurer la clé API OpenAI (optionnel) :
   - Créer un fichier `.env` à la racine du projet
   - Ajouter votre clé API : `OPENAI_API_KEY=votre_clé_api`

## 💻 Utilisation

1. Lancer l'application :
```bash
python app.py
```

2. Accéder à l'interface web :
   - Ouvrir votre navigateur à l'URL indiquée dans la console.

3. Étapes d'utilisation :
   - Charger votre fichier Excel ou CSV
   - Sélectionner les colonnes à classifier
   - Définir les catégories
   - Lancer la classification

## 🏗 Architecture

```
brainbox4/
├── app.py              # Interface utilisateur
├── classifier.py       # Classification asynchrone
├── config.py          # Configuration
├── prompts.py         # Templates LLM
├── utils.py           # Utilitaires
└── requirements.txt   # Dépendances
```

## 🔧 Optimisations de Performance
- parallélisation des requêtes API par lot de 10 maximum pour accélérer la classification.
- suggestion automatique du modèle.

## 🎨 Optimisations de l'Interface Utilisateur
- Suggestion automatiques de catégories et de colonnes basées sur un échantillon de textes.
- Rapport d'évaluation détaillé après classification : analyse des catégories, détection des incohérences, suggestions d'amélioration.
- Suggestion de reclassification des textes selon les recommandations du rapport.

## ✨ Fonctionnalités Principales
1. **Classification Rapide**
   - Traitement parallèle des textes
   - Support des fichiers Excel/CSV
   - Scores de confiance et justification

2. **Interface Simple**
   - Upload de fichiers
   - Sélection des colonnes
   - Visualisation des résultats

## 🚀 Pistes d'Amélioration

1. **Déploiement Local**
   - Utilisation de modèles locaux via LiteLLM
   - Optimisation des appels aux LLMs pour accélérer la classification 

2. **Interface Avancée**
   - Application web dédiée (React/Vue)
   - Système de comptes utilisateurs
   - Historique des classifications

