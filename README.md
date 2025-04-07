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

### Traitement Parallèle
- Exploitation d'`asyncio` pour effectuer des appels API simultanés.
- Gestion par lots de 20 textes par requête pour optimiser le débit.

### Sélection Intelligente du Modèle
- **GPT-3.5** : Utilisé par défaut pour moins de 100 textes.
- **GPT-3.5-16k** : Adapté pour des volumes de 100 à 500 textes.
- **GPT-4** : Préféré pour plus de 500 textes.
- Intégration future de modèles hébergés localement pour une flexibilité accrue.

## 🎨 Optimisations de l'Interface Utilisateur

### Suggestions Automatiques
- Propositions automatiques de catégories et de colonnes basées sur un échantillon de textes.

### Évaluation et Reclassification
- Rapport d'évaluation détaillé après classification : analyse des catégories, détection des incohérences, suggestions d'amélioration.
- Proposition de reclassification des textes selon les recommandations du rapport, ajustement des catégories et seuils de confiance pour améliorer la précision.


## ✨ Fonctionnalités Principales

1. **Classification Rapide**
   - Traitement parallèle des textes
   - Support des fichiers Excel/CSV
   - Scores de confiance

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

