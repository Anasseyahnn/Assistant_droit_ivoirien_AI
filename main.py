import streamlit as st
import os
from dotenv import load_dotenv
import ollama
from openai import OpenAI
import markdown

# Configuration de la page
st.set_page_config(
    page_title="Assistant Juridique IA",
    page_icon="⚖️",
    layout="wide"
)

# Chargement des variables d'environnement
load_dotenv()


# Initialisation des clients API
@st.cache_resource
def get_openai_client():
    return OpenAI()


# Fonction pour obtenir une réponse d'Ollama
def get_ollama_response(question, model="llama3.2"):
    system_prompt = "Tu es un assistant juridique spécialisé en droit ivoirien. Tu fournis des informations précises et à jour sur la législation, la jurisprudence et les procédures juridiques en Côte d'Ivoire. Tu cites tes sources quand c'est possible."
    user_prompt = f"S'il te plait donne moi les détails et les explications: {question}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = ollama.chat(model=model, messages=messages)
        return response['message']['content']
    except Exception as e:
        return f"Erreur lors de l'appel à Ollama: {str(e)}"


# Fonction pour obtenir une réponse d'OpenAI
def get_openai_response(question, model="gpt-4o"):
    client = get_openai_client()
    system_prompt = "Tu es un assistant juridique spécialisé en droit ivoirien. Tu fournis des informations précises et à jour sur la législation, la jurisprudence et les procédures juridiques en Côte d'Ivoire. Tu cites tes sources quand c'est possible."
    user_prompt = f"S'il te plait donne moi les détails et les explications: {question}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur lors de l'appel à OpenAI: {str(e)}"


# Interface utilisateur
st.title("Assistant Juridique IA")
st.subheader("Posez vos questions sur le droit ivoirien")

# Sidebar pour les paramètres
with st.sidebar:
    st.header("Paramètres")

    model_option = st.selectbox(
        "Choisissez un modèle:",
        ["Ollama (Llama3.2)", "OpenAI (GPT)"]
    )

    if model_option == "Ollama (Llama3.2)":
        ollama_model = st.selectbox(
            "Modèle Ollama:",
            ["llama3.2", "llama3", "mistral"]
        )
    else:
        openai_model = st.selectbox(
            "Modèle OpenAI:",
            ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
        )

    st.markdown("---")
    st.markdown("### À propos")
    st.markdown("Cette application utilise l'IA pour fournir des informations juridiques sur le droit ivoirien.")
    st.markdown("⚠️ **Note**: Les informations fournies ne remplacent pas les conseils d'un avocat professionnel.")

# Zone principale pour la question et la réponse
question = st.text_area("Votre question juridique:",
                        "Je suis victime d'un licenciement abusif. Quels sont mes droits selon le code du travail ivoirien?",
                        height=100)

if st.button("Obtenir une réponse"):
    with st.spinner("Recherche en cours..."):
        if model_option == "Ollama (Llama3.2)":
            response = get_ollama_response(question, ollama_model)
        else:
            response = get_openai_response(question, openai_model)

        # Afficher la réponse formatée
        st.markdown("## Réponse:")
        st.markdown(response)

        # Option pour télécharger la réponse
        st.download_button(
            label="Télécharger la réponse",
            data=response,
            file_name="reponse_juridique.md",
            mime="text/markdown"
        )

# Historique des questions (optionnel, utilise la session state de Streamlit)
if 'history' not in st.session_state:
    st.session_state.history = []

if st.button("Ajouter à l'historique"):
    if question not in [q for q, _ in st.session_state.history]:
        if model_option == "Ollama (Llama3.2)":
            response = get_ollama_response(question, ollama_model)
        else:
            response = get_openai_response(question, openai_model)
        st.session_state.history.append((question, response))
    st.success("Question ajoutée à l'historique!")

if st.session_state.history:
    st.markdown("## Historique des questions")
    for i, (q, r) in enumerate(st.session_state.history):
        with st.expander(f"Question {i + 1}: {q[:50]}..."):
            st.markdown(r)

# Footer
st.markdown("---")
st.markdown("Développé avec Streamlit | Utilise Ollama et OpenAI pour les réponses")