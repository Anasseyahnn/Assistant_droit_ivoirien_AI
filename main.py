import streamlit as st
import os
import shutil
from dotenv import load_dotenv
import ollama
from openai import OpenAI
import markdown

# LlamaIndex Imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic as LlamaAnthropic
from llama_index.llms.ollama import Ollama as LlamaOllama
from llama_index.embeddings.ollama import OllamaEmbedding

# Configuration de la page
st.set_page_config(
    page_title="Assistant Juridique IA - Droit Ivoirien",
    page_icon="⚖️",
    layout="wide"
)

# Application d'un style CSS premium (Thème Clair Moderne)
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    /* Supprimer les en-têtes et pieds de page par défaut de Streamlit */
    header {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    
    /* Styles généraux - Thème Clair Moderne */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f8fafc !important;
        background-image: radial-gradient(at 0% 0%, rgba(37, 99, 235, 0.05) 0px, transparent 50%),
                          radial-gradient(at 100% 0%, rgba(99, 102, 241, 0.03) 0px, transparent 50%) !important;
        color: #334155 !important;
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif !important;
    }
    
    /* Style de la barre latérale */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    
    /* Titres globaux */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Boutons principaux */
    .stButton>button {
        width: 100% !important;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.25) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }
    .stButton>button:active {
        transform: translateY(0) !important;
    }
    
    /* Boutons secondaires (Historique, etc.) */
    .stButton>button[key*="save_btn"] {
        background-color: #ffffff !important;
        color: #475569 !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }
    .stButton>button[key*="save_btn"]:hover {
        background-color: #f8fafc !important;
        border-color: #94a3b8 !important;
        color: #0f172a !important;
    }
    
    /* Bouton Téléchargement */
    div[data-testid="stDownloadButton"] > button {
        background-color: rgba(16, 185, 129, 0.06) !important;
        color: #059669 !important;
        border: 1px solid rgba(16, 185, 129, 0.2) !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background-color: rgba(16, 185, 129, 0.12) !important;
        border-color: rgba(16, 185, 129, 0.3) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1) !important;
    }
    
    /* Zone de texte de saisie */
    [data-testid="stTextArea"] textarea {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        color: #1e293b !important;
        font-size: 1rem !important;
        padding: 14px !important;
        line-height: 1.6 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
    }
    
    /* Sélecteurs déroulants (Selectboxes) */
    [data-testid="stSelectbox"] > div {
        background-color: #ffffff !important;
    }
    [data-testid="stSelectbox"] div[role="button"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        color: #1e293b !important;
        padding: 4px 12px !important;
    }
    
    /* Zone de dépôt des fichiers */
    [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border: 1.5px dashed #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 14px !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: transparent !important;
        border: none !important;
    }
    
    /* Conteneur de réponse RAG (avec bordure) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 16px !important;
        padding: 24px !important;
        margin-top: 15px !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02) !important;
    }
    
    /* Accordéons / Expanders de l'historique */
    div[data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.01) !important;
    }
    div[data-testid="stExpander"] details {
        border: none !important;
    }
    div[data-testid="stExpander"] summary {
        background-color: #ffffff !important;
        color: #334155 !important;
        font-weight: 600 !important;
        padding: 14px 18px !important;
    }
    div[data-testid="stExpander"] summary:hover {
        color: #2563eb !important;
    }
    
    /* Case à cocher */
    [data-testid="stCheckbox"] label span {
        color: #334155 !important;
    }
</style>
""", unsafe_allow_html=True)

# Chargement des variables d'environnement
load_dotenv()

# Dossiers pour la base de connaissances et les uploads temporaires
DATA_DIR = "data"
UPLOAD_DIR = "temp_uploads"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialisation des clients API
@st.cache_resource
def get_openai_client():
    return OpenAI()

# Fonction pour obtenir une réponse directe d'Ollama (sans RAG)
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

# Fonction pour obtenir une réponse directe d'OpenAI (sans RAG)
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

# Fonction pour obtenir une réponse directe d'Anthropic Claude (sans RAG)
def get_anthropic_response(question, model="claude-sonnet-4-6"):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "Erreur : La clé ANTHROPIC_API_KEY n'est pas définie dans votre fichier .env."
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        system_prompt = "Tu es un assistant juridique spécialisé en droit ivoirien. Tu fournis des informations précises et à jour sur la législation, la jurisprudence et les procédures juridiques en Côte d'Ivoire. Tu cites tes sources quand c'est possible."
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system_prompt,
            messages=[
                {"role": "user", "content": f"S'il te plait donne moi les détails et les explications: {question}"}
            ]
        )
        return response.content[0].text
    except Exception as e:
        return f"Erreur lors de l'appel à Anthropic (Claude) : {str(e)}"

# Fonction pour configurer LlamaIndex (RAG)
def configure_settings(model_option, sub_model):
    if model_option == "OpenAI (GPT)":
        Settings.llm = LlamaOpenAI(model=sub_model)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    elif model_option == "Anthropic (Claude)":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("La clé ANTHROPIC_API_KEY n'est pas configurée dans le fichier .env.")
        Settings.llm = LlamaAnthropic(model=sub_model, api_key=api_key)
        
        # Configuration de l'Embedding pour Claude
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=openai_key)
        else:
            # Fallback sur un modèle d'embedding HuggingFace local et gratuit si aucune clé OpenAI n'est présente
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    else:
        Settings.llm = LlamaOllama(model=sub_model, request_timeout=120.0)
        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Fonction RAG avec LlamaIndex
def get_rag_response(question, model_option, sub_model, uploaded_files=None):
    try:
        configure_settings(model_option, sub_model)
    except Exception as e:
        return f"Erreur lors de la configuration des modèles LlamaIndex : {str(e)}"

    # Nettoyer et stocker temporairement les nouveaux uploads
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

    try:
        readers = []
        if os.path.exists(DATA_DIR) and os.listdir(DATA_DIR):
            readers.append(SimpleDirectoryReader(DATA_DIR))
        if os.path.exists(UPLOAD_DIR) and os.listdir(UPLOAD_DIR):
            readers.append(SimpleDirectoryReader(UPLOAD_DIR))

        documents = []
        for reader in readers:
            documents.extend(reader.load_data())

        if not documents:
            return "Aucun document trouvé dans la base de connaissances. Veuillez ajouter des fichiers textuels dans le dossier 'data' ou charger un document ci-dessous."

        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query(question)
        return str(response)

    except Exception as e:
        # Fallback intelligent si nomic-embed-text ou Ollama Embedding n'est pas installé localement
        if model_option == "Ollama (Llama3.2)":
            return f"Note : Pour utiliser le RAG avec Ollama local, assurez-vous d'avoir installé le modèle d'embedding avec la commande : 'ollama pull nomic-embed-text'.\n\nErreur détaillée : {str(e)}"
        return f"Erreur lors du traitement LlamaIndex : {str(e)}"

# Interface utilisateur principale
st.markdown("""
<div style="display: flex; align-items: center; gap: 16px; margin-top: -30px; margin-bottom: 5px;">
    <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="filter: drop-shadow(0 0 8px rgba(37, 99, 235, 0.15));">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
    </svg>
    <div>
        <h1 style="margin: 0; padding: 0; font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #0f172a 40%, #1e3a8a 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.03em;">
            Assistant Juridique
        </h1>
        <p style="margin: 4px 0 0 0; padding: 0; color: #475569; font-size: 1rem; font-weight: 500;">
            Côte d'Ivoire — Modélisation IA & Analyse Législative
        </p>
    </div>
</div>
<hr style="margin: 15px 0 25px 0; border: 0; border-top: 1px solid #e2e8f0;">
""", unsafe_allow_html=True)

# Configuration de la Sidebar
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
        <span style="font-size: 1.15rem; font-weight: 700; color: #1e293b;">Configuration</span>
    </div>
    """, unsafe_allow_html=True)

    model_option = st.selectbox(
        "Fournisseur d'IA :",
        ["Anthropic (Claude)", "OpenAI (GPT)", "Ollama (Llama3.2)"]
    )

    # Indicateurs d'état des API en temps réel (Style Thème Clair)
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    
    status_claude = "🟢 Actif" if has_anthropic else "🔴 Non configuré"
    status_gpt = "🟢 Actif" if has_openai else "🔴 Non configuré"
    status_ollama = "🔴 Non configuré" # Ollama non disponible sur le système
    
    st.markdown(f"""
    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; margin-top: 10px; font-size: 0.85rem;">
        <div style="display:flex; justify-content:space-between; margin-bottom: 6px;">
            <span style="color:#64748b;">Claude API :</span>
            <span style="font-weight:600; color:{'#059669' if has_anthropic else '#dc2626'};">{status_claude}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom: 6px;">
            <span style="color:#64748b;">OpenAI API :</span>
            <span style="font-weight:600; color:{'#059669' if has_openai else '#dc2626'};">{status_gpt}</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span style="color:#64748b;">Ollama Local :</span>
            <span style="font-weight:600; color:#dc2626;">{status_ollama}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if model_option == "Ollama (Llama3.2)":
        sub_model = st.selectbox(
            "Modèle Ollama :",
            ["llama3.2", "llama3", "mistral"]
        )
    elif model_option == "OpenAI (GPT)":
        sub_model = st.selectbox(
            "Modèle OpenAI :",
            ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
        )
    else:
        sub_model = st.selectbox(
            "Modèle Anthropic :",
            ["claude-sonnet-4-6", "claude-sonnet-4-20250514", "claude-opus-4-8", "claude-haiku-4-5-20251001"]
        )

    st.markdown("---")
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path></svg>
        <span style="font-size: 1.1rem; font-weight: 700; color: #1e293b;">Base de données</span>
    </div>
    """, unsafe_allow_html=True)
    
    use_rag = st.checkbox("Activer le RAG (LlamaIndex)", value=True, help="Si activé, l'IA cherchera les réponses directement dans les extraits officiels de loi et vos documents importés.")

    if use_rag:
        st.markdown("""
        <div style="background-color: rgba(37,99,235,0.04); border: 1px solid rgba(37,99,235,0.12); border-radius: 8px; padding: 10px; font-size: 0.8rem; color: #1e3a8a; margin-top: 10px; line-height: 1.4;">
            <strong>Index sémantique actif :</strong><br>
            • Code du Travail de Côte d'Ivoire
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#d97706" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
        <span style="font-size: 0.85rem; font-weight: 700; color: #d97706;">Information Légale</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.8rem; color: #64748b; line-height: 1.4; margin: 0;'>Les réponses fournies sont issues d'une analyse automatisée et ont un but informatif. Elles ne remplacent pas l'avis officiel d'un avocat.</p>", unsafe_allow_html=True)

# Zone d'import de documents utilisateur
st.markdown("""
<div style="display: flex; align-items: center; gap: 10px; margin-top: 5px; margin-bottom: 12px;">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
    <span style="font-size: 1.2rem; font-weight: 700; color: #1e293b;">Documents additionnels (Optionnel)</span>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Importez vos contrats ou fichiers de lois au format PDF ou TXT pour les analyser sémantiquement :",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# Formulaire de question
st.markdown("""
<div style="display: flex; align-items: center; gap: 10px; margin-top: 15px; margin-bottom: 12px;">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
    <span style="font-size: 1.2rem; font-weight: 700; color: #1e293b;">Votre requête juridique</span>
</div>
""", unsafe_allow_html=True)

question = st.text_area(
    "Saisissez votre question juridique :",
    value="Je suis victime d'un licenciement abusif. Quels sont mes droits en cas de préavis et de dommages-intérêts selon le code du travail ivoirien ?",
    height=110,
    label_visibility="collapsed"
)

# Logique de traitement
col_submit, col_save = st.columns([2, 1])

with col_submit:
    submit_clicked = st.button("Analyser la législation", key="submit_btn")

# Traitement de la question principale
if submit_clicked:
    if not question.strip():
        st.warning("Veuillez saisir une question valide.")
    else:
        with st.spinner("Recherche et analyse juridique en cours..."):
            if use_rag:
                response = get_rag_response(question, model_option, sub_model, uploaded_files)
            else:
                if model_option == "Ollama (Llama3.2)":
                    response = get_ollama_response(question, sub_model)
                elif model_option == "OpenAI (GPT)":
                    response = get_openai_response(question, sub_model)
                else:
                    response = get_anthropic_response(question, sub_model)

            # Affichage de la réponse dans un cadre stylé
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 10px; margin-top: 25px; margin-bottom: 12px;">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="filter: drop-shadow(0 0 5px rgba(5, 150, 105, 0.15));"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                <span style="font-size: 1.3rem; font-weight: 700; color: #1e293b;">Analyse Juridique Délivrée</span>
            </div>
            """, unsafe_allow_html=True)
            
            with st.container(border=True):
                st.markdown(response)

            # Boutons d'action pour la réponse
            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            st.download_button(
                label="Télécharger la note d'analyse (Markdown)",
                data=response,
                file_name="note_juridique_ivoirienne.md",
                mime="text/markdown",
                key="download_btn"
            )

# Gestion de l'historique des requêtes
if 'history' not in st.session_state:
    st.session_state.history = []

with col_save:
    save_clicked = st.button("Enregistrer", key="save_btn")

if save_clicked:
    if question.strip():
        with st.spinner("Génération de l'archive..."):
            if use_rag:
                response = get_rag_response(question, model_option, sub_model, uploaded_files)
            else:
                if model_option == "Ollama (Llama3.2)":
                    response = get_ollama_response(question, sub_model)
                elif model_option == "OpenAI (GPT)":
                    response = get_openai_response(question, sub_model)
                else:
                    response = get_anthropic_response(question, sub_model)
        st.session_state.history.append((question, response))
        st.success("Recherche ajoutée à l'historique.")

if st.session_state.history:
    st.markdown("---")
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-top: 10px; margin-bottom: 20px;">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#4f46e5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8v4l3 3"></path><circle cx="12" cy="12" r="9"></circle></svg>
        <span style="font-size: 1.2rem; font-weight: 700; color: #1e293b;">Historique de vos consultations</span>
    </div>
    """, unsafe_allow_html=True)
    for i, (q, r) in enumerate(st.session_state.history):
        with st.expander(f"Consultation #{i + 1} : {q[:80]}..."):
            st.markdown(r)

# Pied de page
st.markdown("---")
st.markdown("<center style='font-size: 0.85rem; color: #64748b;'>Développé avec <b>Streamlit</b> & <b>LlamaIndex</b> | MLOps Data Science & RAG Pipeline</center>", unsafe_allow_html=True)