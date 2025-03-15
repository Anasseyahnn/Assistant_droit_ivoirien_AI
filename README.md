# Assistant Juridique IA ⚖️

Une application Streamlit qui fournit des conseils juridiques automatisés sur le droit ivoirien en utilisant des modèles d'intelligence artificielle (Llama et GPT).
![image](https://github.com/user-attachments/assets/157c034d-033f-4cfe-b700-9a2dd39002f2)

## 📝 Description

L'Assistant Juridique IA est une application qui permet aux utilisateurs d'obtenir des informations juridiques sur le droit ivoirien à partir de questions formulées en langage naturel. L'application utilise des modèles d'IA avancés pour fournir des réponses détaillées avec références aux textes de loi pertinents.

### Fonctionnalités principales

- 💬 Interface conversationnelle simple pour poser des questions juridiques
- 🔄 Support de différents modèles d'IA (Llama 3.2 via Ollama, GPT via OpenAI)
- 📋 Historique des questions et réponses
- 💾 Export des réponses au format Markdown
- 📱 Interface responsive adaptée à tous les appareils

## 🚀 Installation

### Prérequis

- Python 3.8+
- [Ollama](https://ollama.ai/) installé localement (pour les modèles Llama)
- Clé API OpenAI (pour les modèles GPT)

### Étapes d'installation

1. Clonez ce dépôt :
```bash
git clone https://github.com/Anasseyahnn/Assistant_droit_ivoirien_AI.git
cd Assistant_droit_ivoirien_AI
```

2. Créez un environnement virtuel et activez-le :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

4. Créez un fichier `.env` avec vos clés API :
```
OPENAI_API_KEY=votre_clé_api_openai
```

## 💻 Utilisation

Lancez l'application avec Streamlit :

```bash
streamlit run app.py
```

L'application sera accessible dans votre navigateur à l'adresse `http://localhost:8501`.

### Guide d'utilisation

1. Saisissez votre question juridique dans la zone de texte
2. Choisissez le modèle d'IA à utiliser (Ollama/Llama ou OpenAI/GPT)
3. Cliquez sur "Obtenir une réponse"
4. Utilisez le bouton "Télécharger la réponse" pour sauvegarder la réponse au format Markdown
5. Ajoutez des questions à l'historique pour les consulter ultérieurement

## 🧰 Technologies utilisées

- [Streamlit](https://streamlit.io/) - Framework pour la création d'applications web en Python
- [Ollama](https://ollama.ai/) - Exécution locale de modèles d'IA avancés comme Llama 3.2
- [OpenAI API](https://openai.com/) - Accès aux modèles GPT
- [Python-dotenv](https://github.com/theskumar/python-dotenv) - Gestion des variables d'environnement
- [Markdown](https://python-markdown.github.io/) - Formatage des réponses

## 📝 Exemple de prompts utilisés

L'application utilise des prompts spécialisés pour optimiser les réponses :

```python
system_prompt = "Tu es un assistant juridique spécialisé en droit ivoirien. Tu fournis des informations précises et à jour sur la législation, la jurisprudence et les procédures juridiques en Côte d'Ivoire. Tu cites tes sources quand c'est possible."
```

## ⚠️ Avertissement

Les informations fournies par cette application ne remplacent pas les conseils d'un avocat professionnel. Elles sont fournies à titre indicatif uniquement et peuvent ne pas refléter les dernières évolutions législatives ou jurisprudentielles.

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 📞 Contact

Pour toute question ou suggestion, n'hésitez pas à me contacter à [anasseyahanan@gmail.com](mailto:anasseyahanan@gmail.com).

---

⭐ Si vous trouvez ce projet utile, n'hésitez pas à lui donner une étoile sur GitHub !
