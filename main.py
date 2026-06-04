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

# Application d'un style CSS premium (Thème Dark SaaS Minimaliste)
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
    /* Supprimer les en-têtes et pieds de page par défaut de Streamlit */
    header {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    
    /* Styles généraux - Thème Studio Dark */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #050505 !important;
        color: #ffffff !important;
        font-family: 'Space Grotesk', -apple-system, sans-serif !important;
    }
    
    /* Style de la barre latérale */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 1px solid #1f1f1f !important;
    }
    
    /* Titres globaux */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Boutons principaux */
    .stButton>button {
        width: 100% !important;
        background: #ffffff !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        text-transform: uppercase !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.05em !important;
    }
    .stButton>button:hover {
        background: #e5e5e5 !important;
        transform: translateY(-1px) !important;
    }
    
    /* Boutons secondaires (Historique, etc.) */
    .stButton>button[key*="save_btn"] {
        background-color: transparent !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    .stButton>button[key*="save_btn"]:hover {
        background-color: #111111 !important;
        border-color: #555555 !important;
    }
    
    /* Bouton Téléchargement */
    div[data-testid="stDownloadButton"] > button {
        background-color: transparent !important;
        color: #ffffff !important;
        border: 1px dashed #555555 !important;
        border-radius: 4px !important;
        width: 100% !important;
        text-transform: uppercase !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.05em !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background-color: #111111 !important;
        border-color: #ffffff !important;
    }
    
    /* Zone de texte de saisie */
    [data-testid="stTextArea"] textarea {
        background-color: #0a0a0a !important;
        border: 1px solid #333333 !important;
        border-radius: 4px !important;
        color: #ffffff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
        padding: 14px !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: #ffffff !important;
        box-shadow: none !important;
    }
    
    /* Sélecteurs déroulants (Selectboxes) */
    [data-testid="stSelectbox"] > div {
        background-color: transparent !important;
    }
    [data-testid="stSelectbox"] div[role="button"] {
        background-color: #0a0a0a !important;
        border: 1px solid #333333 !important;
        border-radius: 4px !important;
        color: #ffffff !important;
    }
    
    /* Zone de dépôt des fichiers */
    [data-testid="stFileUploader"] {
        background-color: #0a0a0a !important;
        border: 1px dashed #333333 !important;
        border-radius: 4px !important;
        padding: 14px !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: transparent !important;
        border: none !important;
        color: #ffffff !important;
    }
    
    /* Conteneur de réponse RAG */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #0a0a0a !important;
        border: 1px solid #333333 !important;
        border-radius: 4px !important;
        padding: 24px !important;
        margin-top: 15px !important;
    }
    
    /* Accordéons / Expanders de l'historique */
    div[data-testid="stExpander"] {
        background-color: #0a0a0a !important;
        border: 1px solid #333333 !important;
        border-radius: 4px !important;
    }
    div[data-testid="stExpander"] summary {
        background-color: #0a0a0a !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    div[data-testid="stExpander"] summary:hover {
        color: #a1a1aa !important;
    }
    
    /* Text input pour API keys */
    [data-testid="stTextInput"] input {
        background-color: #0a0a0a !important;
        border: 1px solid #333333 !important;
        border-radius: 4px !important;
        color: #ffffff !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #ffffff !important;
        box-shadow: none !important;
    }
    
    /* Case à cocher */
    [data-testid="stCheckbox"] label span {
        color: #a1a1aa !important;
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
<div style="margin-top: -30px; margin-bottom: 25px;">
    <h1 style="margin: 0; padding: 0; font-size: 2.5rem; font-weight: 700; color: #ffffff; text-transform: uppercase; letter-spacing: -0.02em;">
        Assistant Juridique
    </h1>
    <p style="margin: 4px 0 0 0; padding: 0; color: #a1a1aa; font-size: 1rem; font-family: 'JetBrains Mono', monospace;">
        Côte d'Ivoire — Modélisation IA & Analyse Législative
    </p>
</div>
<hr style="margin: 15px 0 25px 0; border: 0; border-top: 1px solid #333333;">
""", unsafe_allow_html=True)

# Configuration de la Sidebar
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 15px;">
        <span style="font-size: 0.9rem; font-weight: 700; color: #ffffff; text-transform: uppercase; letter-spacing: 0.05em;">Configuration</span>
    </div>
    """, unsafe_allow_html=True)

    model_option = st.selectbox(
        "Fournisseur d'IA :",
        ["Anthropic (Claude)", "OpenAI (GPT)", "Ollama (Llama3.2)"]
    )

    user_openai_key = st.text_input("Clé API OpenAI (optionnel)", type="password")
    user_anthropic_key = st.text_input("Clé API Anthropic (optionnel)", type="password")

    if user_openai_key:
        os.environ["OPENAI_API_KEY"] = user_openai_key
    if user_anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = user_anthropic_key

    # Indicateurs d'état des API en temps réel (Style Thème Clair)
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    
    status_claude = "ACTIF" if has_anthropic else "NON CONFIGURÉ"
    status_gpt = "ACTIF" if has_openai else "NON CONFIGURÉ"
    status_ollama = "NON CONFIGURÉ" # Ollama non disponible sur le système
    
    st.markdown(f"""
    <div style="background-color: #0a0a0a; border: 1px solid #333333; border-radius: 4px; padding: 12px; margin-top: 10px; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
        <div style="display:flex; justify-content:space-between; margin-bottom: 6px;">
            <span style="color:#71717a;">Claude API :</span>
            <span style="color:{'#ffffff' if has_anthropic else '#52525b'};">{status_claude}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom: 6px;">
            <span style="color:#71717a;">OpenAI API :</span>
            <span style="color:{'#ffffff' if has_openai else '#52525b'};">{status_gpt}</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span style="color:#71717a;">Ollama Local :</span>
            <span style="color:#52525b;">{status_ollama}</span>
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
    <div style="margin-bottom: 12px;">
        <span style="font-size: 0.9rem; font-weight: 700; color: #ffffff; text-transform: uppercase; letter-spacing: 0.05em;">Base de données</span>
    </div>
    """, unsafe_allow_html=True)
    
    use_rag = st.checkbox("Activer le RAG (LlamaIndex)", value=True, help="Si activé, l'IA cherchera les réponses directement dans les extraits officiels de loi et vos documents importés.")

    if use_rag:
        st.markdown("""
        <div style="background-color: #111111; border: 1px dashed #333333; border-radius: 4px; padding: 10px; font-size: 0.8rem; color: #a1a1aa; margin-top: 10px; font-family: 'JetBrains Mono', monospace;">
            <strong>Index sémantique actif :</strong><br>
            > Code du Travail de Côte d'Ivoire
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <span style="font-size: 0.75rem; font-weight: 700; color: #a1a1aa; text-transform: uppercase; letter-spacing: 0.05em;">Information Légale</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.75rem; color: #52525b; line-height: 1.4; margin: 0;'>Les réponses fournies sont issues d'une analyse automatisée et ont un but informatif. Elles ne remplacent pas l'avis officiel d'un avocat.</p>", unsafe_allow_html=True)

# Zone d'import de documents utilisateur
st.markdown("""
<div style="margin-top: 5px; margin-bottom: 12px;">
    <span style="font-size: 1rem; font-weight: 700; color: #ffffff; text-transform: uppercase;">Documents additionnels (Optionnel)</span>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Importez vos contrats ou fichiers de lois au format PDF ou TXT pour les analyser sémantiquement :",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# Formulaire de question
st.markdown("""
<div style="margin-top: 15px; margin-bottom: 12px;">
    <span style="font-size: 1rem; font-weight: 700; color: #ffffff; text-transform: uppercase;">Votre requête juridique</span>
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
            <div style="margin-top: 25px; margin-bottom: 12px;">
                <span style="font-size: 1.1rem; font-weight: 700; color: #ffffff; text-transform: uppercase; border-bottom: 2px solid #ffffff; padding-bottom: 4px;">Analyse Juridique Délivrée</span>
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
    <div style="margin-top: 10px; margin-bottom: 20px;">
        <span style="font-size: 1rem; font-weight: 700; color: #ffffff; text-transform: uppercase;">Historique de vos consultations</span>
    </div>
    """, unsafe_allow_html=True)
    for i, (q, r) in enumerate(st.session_state.history):
        with st.expander(f"Consultation #{i + 1} : {q[:80]}..."):
            st.markdown(r)

st.markdown("---")
st.markdown("<center style='font-size: 0.75rem; color: #52525b; font-family: \"JetBrains Mono\", monospace;'>DÉVELOPPÉ AVEC STREAMLIT & LLAMAINDEX | LLM ENGINEERING & RAG PIPELINE</center>", unsafe_allow_html=True)