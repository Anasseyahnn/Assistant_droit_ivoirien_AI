# Assistant Juridique IA - Côte d'Ivoire

Ce projet propose un assistant juridique intelligent spécialisé dans le Code du Travail de Côte d'Ivoire. En s'appuyant sur une architecture RAG (Retrieval-Augmented Generation), l'application permet d'interroger la législation ivoirienne en langage naturel et d'obtenir des analyses juridiques précises et sourcées.

## Fonctionnalités

* **Analyse de la législation** : Interrogation sémantique du Code du Travail ivoirien.
* **Importation de documents** : Possibilité d'ajouter des documents additionnels (contrats, avenants ou lois au format PDF ou TXT) pour une analyse combinée.
* **Architecture RAG Locale & API** : Pipeline utilisant LlamaIndex, combinant des plongements (embeddings) locaux via HuggingFace (`BAAI/bge-small-en-v1.5`) et les modèles de langage de la génération Claude 4 d'Anthropic.
* **Interface Premium** : Interface Streamlit moderne avec un thème clair soigné (Slate-Light) et des visuels vectoriels professionnels.
* **Historique des consultations** : Possibilité d'archiver vos sessions de recherche directement dans l'application.
* **Export Markdown** : Téléchargement direct des notes d'analyse générées en format Markdown.

## Architecture Technique

Le système repose sur un pipeline de type RAG (Retrieval-Augmented Generation) optimisé :
1. **Extraction** : Chargement des textes légaux de base (`data/`) et des fichiers fournis par l'utilisateur.
2. **Indexation sémantique** : Vectorisation locale à l'aide du modèle d'embedding HuggingFace `BAAI/bge-small-en-v1.5` (permettant une exécution rapide et gratuite des embeddings sans clé tierce).
3. **Génération** : Modélisation des réponses par les modèles Claude 4 d'Anthropic, garantissant une compréhension contextuelle avancée et une rédaction juridique rigoureuse.

## Installation

### Prérequis

* Python 3.10 ou supérieur
* Clé API Anthropic (Claude)

### Étapes d'installation

1. Clonez le dépôt du projet :
   ```bash
   git clone https://github.com/Anasseyahnn/Assistant_droit_ivoirien_AI.git
   cd Assistant_droit_ivoirien_AI
   ```

2. Créez un environnement virtuel et activez-le :
   ```bash
   python -m venv venv
   # Sur Windows :
   venv\Scripts\activate
   # Sur macOS/Linux :
   source venv/bin/activate
   ```

3. Installez l'ensemble des dépendances requises :
   ```bash
   pip install -r requirements.txt
   ```

4. Configurez vos variables d'environnement en créant un fichier `.env` à la racine du projet :
   ```env
   ANTHROPIC_API_KEY=votre_cle_api_anthropic
   ```

## Utilisation

Lancez le serveur d'application Streamlit :

```bash
streamlit run main.py
```

L'application s'ouvre automatiquement dans votre navigateur par défaut à l'adresse `http://localhost:8501`.

### Guide d'utilisation

1. **Choix du Modèle** : Sélectionnez le modèle Claude souhaité dans le menu latéral (le modèle recommandé par défaut est `claude-sonnet-4-6`).
2. **Documents supplémentaires (Optionnel)** : Déposez vos fichiers PDF ou TXT (contrats de travail, conventions collectives) dans la zone d'importation pour enrichir le contexte d'analyse.
3. **Requête** : Saisissez votre question juridique en langage naturel (par exemple sur le calcul de préavis, les indemnités de licenciement ou les clauses de CDD/CDI).
4. **Analyse** : Cliquez sur "Analyser la législation". L'assistant produit une réponse structurée contenant les références réglementaires pertinentes.
5. **Export & Sauvegarde** : Téléchargez l'analyse au format Markdown ou enregistrez-la dans l'historique de votre session actuelle.

## Technologies Utilisées

* **Streamlit** : Framework de présentation web réactif et épuré.
* **LlamaIndex** : Framework d'orchestration pour le RAG.
* **Anthropic API** : Accès aux modèles de langage Claude 4 (Claude 4.6 Sonnet).
* **HuggingFace Embeddings** : Représentation vectorielle locale avec BGE-Small.

## Avertissement Légal

Les réponses fournies par cet assistant sont générées de manière automatisée à des fins informatives et pédagogiques. Elles ne constituent pas des conseils juridiques formels et ne remplacent en aucun cas la consultation d'un avocat ou d'un professionnel du droit agréé.
