import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth # OAuth para poder criar playlists
from spotipy.exceptions import SpotifyException
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
import pandas as pd
import json
import os
import hashlib

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Notify | O teu DJ Emocional", page_icon="🎧", layout="centered")


def aplicar_tema_global(emocao):
    cor_base = cores_emocao.get(emocao, "#000000")
    # Usar preto sem clarear: gradiente para preto sem ir para branco
    cor_secundaria = 'rgba(0,0,0,0.6)'
    st.markdown(
        f"""
        <style>
            /* Fundo de toda a aplicação com gradiente suave entre duas tonalidades */
            .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
                background-color: {cor_base} !important;
                background-image: linear-gradient(135deg, {cor_base}, {cor_secundaria}) !important;
                background-repeat: no-repeat !important;
                background-attachment: fixed !important;
            }}
            /* Tornar a zona de chat e o rodapé em baixo transparentes */
            [data-testid="stBottom"], [data-testid="stBottom"] > div {{
                background-color: transparent !important;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )


def _shade_hex(hex_color, amount=0.25):
    """Return a hex color lightened by amount (0..1) or darkened if amount negative."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except Exception:
        return '#' + hex_color

    def clamp(v):
        return max(0, min(255, int(v)))

    if amount >= 0:
        r2 = clamp(r + (255 - r) * amount)
        g2 = clamp(g + (255 - g) * amount)
        b2 = clamp(b + (255 - b) * amount)
    else:
        r2 = clamp(r * (1 + amount))
        g2 = clamp(g * (1 + amount))
        b2 = clamp(b * (1 + amount))

    return '#{0:02x}{1:02x}{2:02x}'.format(r2, g2, b2)

cores_emocao = {
    "Joy": "#C2C221", "Sadness": "#214C6C", "Anger": "#680F0F",
    "Fear": "#3A3A3A", "Trust": "#276B29", "Disgust": "#684224",
    "Surprise": "#257BA3", "Anticipation": "#4E297F", "Neutral": "#2A2A2A"
}

# --- Gestão de Utilizadores ---
DB_FILE = "utilizadores.json"
TARGET_TRACKS = 20

def carregar_utilizadores():
    if not os.path.exists(DB_FILE): return {}
    with open(DB_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def guardar_utilizadores(dados):
    with open(DB_FILE, "w") as f: json.dump(dados, f, indent=4)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def normalizar_registo_utilizador(raw):
    if isinstance(raw, str):
        return {"password_hash": raw, "last_emotion": "Neutral"}
    if isinstance(raw, dict):
        return {
            "password_hash": raw.get("password_hash", ""),
            "last_emotion": raw.get("last_emotion", "Neutral")
        }
    return {"password_hash": "", "last_emotion": "Neutral"}


def aplicar_fundo_emocional(emocao):
    aplicar_tema_global(emocao)

# --- CARREGAR MODELO ---
@st.cache_resource
def carregar_ia():
    caminho = "./modelo"
    tokenizer = AutoTokenizer.from_pretrained(caminho)
    modelo = AutoModelForSequenceClassification.from_pretrained(caminho)
    return tokenizer, modelo

try:
    tokenizer, modelo = carregar_ia()
except:
    st.error("Erro ao carregar o modelo de IA. Verifica a pasta './modelo'")

# --- CONFIGURAÇÃO AUTENTICAÇÃO DO SPOTIFY ---
CLIENT_ID = "4ddae513ee4c4982b7f8615380c219ce"
CLIENT_SECRET = "c617602e6a4342fa99933a4f3a3e45e1"
REDIRECT_URI = "http://127.0.0.1:8080"
# Pedimos permissões para criar/editar playlists públicas e privadas.
SCOPE = "playlist-modify-public playlist-modify-private playlist-read-private"
CACHE_PATH = ".spotify_cache"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE,
    cache_path=CACHE_PATH,
    show_dialog=True
))

emocoes = ['Joy', 'Sadness', 'Anger', 'Fear', 'Trust', 'Disgust', 'Surprise', 'Anticipation']

# Limites para evitar músicas muito fracas e favorecer variedade com qualidade.
POPULARITY_MINIMA = 30
POPULARITY_PESO_EXPONENTE = 1.4
DISTANCE_PESO_EXPONENTE = 1.8

# Palavras-chave/estilos associadas a cada emoção (usadas como "modificador" para busca/descrição)
plutchik_modifiers = {
    "Joy": "happy upbeat energetic",
    "Sadness": "sad acoustic melancholic slow",
    "Anger": "aggressive dark intense heavy",
    "Trust": "relaxing calm ambient peaceful",
    "Fear": "suspense dark tension eerie",
    "Anticipation": "focus driving energetic motivation",
    "Disgust": "gritty alternative raw",
    "Surprise": "dynamic explosive unexpected",
    "Neutral": "chill background smooth",
}

# Map Plutchik emotions to approximate target valence/energy (0..1)
EMOTION_TO_VALENCE_ENERGY = {
    "Joy": (0.95, 0.8),  # 0.95, 0.65
    "Sadness": (0.05, 0.3),  # -0.9, -0.4
    "Anger": (0.2, 0.8),  # -0.65, 0.7
    "Fear": (0.1, 0.8),   # -0.85, -0.65
    "Trust": (0.85, 0.4),  # 0.75, 0.1
    "Disgust": (0.1, 0.7),   # -0.9, 0.55
    "Surprise": (0.8, 0.85),   # 0.75, 0.75
    "Anticipation": (0.6, 0.7),
    "Neutral": (0.5, 0.5)
}

# Target emotion for gradual progression (calm + happy)
TARGET_EMOTION_MOOD = ("Joy", (0.8, 0.4))  # High valence, moderate-low energy = calm and happy

GENEROS_SUGERIDOS = [
    "pop",
    "hip-hop",
    "rock",
    "r-n-b",
    "latin",
    "edm",
    "reggaeton",
    "k-pop",
    "country",
    "indie",
    "alternative",
    "metal",
    "reggae",
    "classical",
    "jazz",
]


def gerar_progressao_mood(
    emocao_inicial,
    valence_inicial,
    energy_inicial,
    num_passos=TARGET_TRACKS,
    passos_iniciais_fixos=5,
):
    _, (target_val, target_eng) = TARGET_EMOTION_MOOD

    progressoes = []
    passos_iniciais_fixos = max(0, min(passos_iniciais_fixos, num_passos))

    # Mantém as primeiras N músicas no mood inicial.
    for _ in range(passos_iniciais_fixos):
        progressoes.append((valence_inicial, energy_inicial))

    # A partir daí, faz a transição gradual até ao alvo.
    restantes = num_passos - passos_iniciais_fixos
    if restantes > 0:
        for i in range(restantes):
            t = i / (restantes - 1) if restantes > 1 else 1.0  # [0..1]
            val = valence_inicial + t * (target_val - valence_inicial)
            eng = energy_inicial + t * (target_eng - energy_inicial)
            progressoes.append((val, eng))

    return progressoes[:num_passos]

@st.cache_resource
def carregar_dataset():
    import time
    start = time.time()
    path = "C:\\Users\\diogo\\Desktop\\UNI\\CA\\Trabalho\\Projeto_CA\\dataset\\spotify_musics_limpo.csv"
    if not path:
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
        elapsed = time.time() - start
        print(f'Dataset carregado em {elapsed:.2f}s ({len(df)} linhas)')
        return df
    except Exception as e:
        print('Erro a ler dataset:', e)
        return None


@st.cache_resource
def carregar_generos_disponiveis():
    """Lê os géneros únicos do dataset."""
    df_local = carregar_dataset()
    if df_local is None or len(df_local) == 0:
        return []

    genre_col = next((c for c in df_local.columns if 'genre' in c.lower()), None)
    if not genre_col:
        return []

    generos = (
        df_local[genre_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )
    generos.sort()

    sep_sugeridos = "────────────────────── GÉNEROS SUGERIDOS ───────────────────────"
    sep_restantes = "────────────────────── RESTANTES GÉNEROS ───────────────────────"

    generos_sugeridos = [g for g in GENEROS_SUGERIDOS if g in generos]
    restantes = [g for g in generos if g not in generos_sugeridos]
    return [sep_sugeridos] + generos_sugeridos + [sep_restantes] + restantes


def formatar_dropdown(opcao):
    if opcao in ["────────────────────── GÉNEROS SUGERIDOS ───────────────────────", "────────────────────── RESTANTES GÉNEROS ───────────────────────"]:
        return opcao
    return opcao.title()


def select_tracks_from_dataset(genres, target_val, target_eng, tol=0.25, max_results=TARGET_TRACKS):
    import time
    start = time.time()
    
    df_local = carregar_dataset()
    if df_local is None or len(df_local) == 0:
        return []

    val_col = next((c for c in df_local.columns if 'valence' in c.lower()), None)
    eng_col = next((c for c in df_local.columns if 'energy' in c.lower()), None)
    pop_col = next((c for c in df_local.columns if 'popularity' in c.lower()), None)
    uri_col = next((c for c in df_local.columns if 'uri' in c.lower() or 'track_uri' in c.lower()), None)
    id_col = next((c for c in df_local.columns if c.lower() in ('track_id','id','spotify_id')), None)
    genre_col = next((c for c in df_local.columns if 'genre' in c.lower()), None)

    if not val_col or not eng_col:
        return []

    mask_val_eng = pd.notna(df_local[val_col]) & pd.notna(df_local[eng_col])
    df_filtered = df_local[mask_val_eng].copy()

    # Filtra músicas muito pouco populares para evitar faixas fracas no resultado final.
    if pop_col:
        pop_values = pd.to_numeric(df_filtered[pop_col], errors='coerce').fillna(0)
        df_filtered = df_filtered[pop_values >= POPULARITY_MINIMA].copy()
    
    # Filtro de género musical
    wanted = [g.lower() for g in genres]
    if genre_col and wanted:
        genre_values = df_filtered[genre_col].fillna('').astype(str).str.strip().str.lower()
        genre_mask = genre_values.isin(wanted)
        df_filtered = df_filtered[genre_mask]
    
    if len(df_filtered) == 0:
        return []
    
    # Distancia euclidiana valence-energy
    distances = ((df_filtered[val_col].astype(float) - target_val) ** 2 + 
                 (df_filtered[eng_col].astype(float) - target_eng) ** 2) ** 0.5
    
    # Filtrar por tolerância e ordenar por distância
    mask_tol = distances <= tol
    df_candidates = df_filtered[mask_tol].copy()
    df_candidates['_dist'] = distances[mask_tol]
    
    if len(df_candidates) == 0:
        return []

    if pop_col:
        df_candidates['_pop'] = pd.to_numeric(df_candidates[pop_col], errors='coerce').fillna(0)
        dist_weights = 1 / (1 + df_candidates['_dist'].clip(lower=0)) ** DISTANCE_PESO_EXPONENTE
        pop_values = df_candidates['_pop'].clip(lower=POPULARITY_MINIMA)
        pop_weights = (pop_values / pop_values.max()) ** POPULARITY_PESO_EXPONENTE if pop_values.max() > 0 else 1.0
        weights = dist_weights * pop_weights
        weights = weights.where(weights > 0, 1.0)
    else:
        weights = 1 / (1 + df_candidates['_dist'].clip(lower=0)) ** DISTANCE_PESO_EXPONENTE

    # Amostragem ponderada: mantém o mood, mas varia entre execuções.
    if len(df_candidates) > max_results:
        df_candidates = df_candidates.sample(n=max_results, weights=weights, replace=False)
    else:
        df_candidates = df_candidates.sample(frac=1, weights=weights)

    df_candidates = df_candidates.sort_values(['_dist', pop_col] if pop_col else ['_dist'])

    uris = []
    for _, row in df_candidates.iterrows():
        if len(uris) >= max_results:
            break
        uri = None
        if uri_col and pd.notna(row.get(uri_col)):
            uri = str(row[uri_col])
        elif id_col and pd.notna(row.get(id_col)):
            uri = f"spotify:track:{row[id_col]}"
        if uri and uri not in uris:
            uris.append(uri)
    
    elapsed = time.time() - start
    print(f'  [tol={tol:.2f}] {target_val:.2f},{target_eng:.2f}: {len(uris)}/{max_results} em {elapsed:.2f}s')
    return uris


def select_tracks_progressive(genres, progressao_mood, max_results=TARGET_TRACKS):
    """
    Seleciona faixas usando a progressão de (valence, energy) ao longo da playlist.
    """
    if not progressao_mood:
        return []
    
    all_uris = []
    uris_per_step = max_results // len(progressao_mood)
    remainder = max_results % len(progressao_mood)
    
    for i, (val, eng) in enumerate(progressao_mood):
        # Adiciona 1 extra aos primeiros 'remainder' passos para distribuir os que faltam
        n = uris_per_step + (1 if i < remainder else 0)
        step_uris = select_tracks_from_dataset(genres, val, eng, tol=0.25, max_results=n)
        all_uris.extend(step_uris)
    
    seen = set()
    final = []
    for u in all_uris:
        if u not in seen:
            seen.add(u)
            final.append(u)
        if len(final) >= max_results:
            break

    if len(final) < max_results:
        faltam = max_results - len(final)
        for tol_extra in (0.30, 0.35, 0.45, 0.55):
            if faltam <= 0:
                break
            extras = select_tracks_from_dataset(
                genres,
                TARGET_EMOTION_MOOD[1][0],
                TARGET_EMOTION_MOOD[1][1],
                tol=tol_extra,
                max_results=faltam,
            )
            for uri in extras:
                if uri not in seen:
                    seen.add(uri)
                    final.append(uri)
                    faltam -= 1
                    if len(final) >= max_results:
                        break

    # Se ainda faltar, tenta sem filtro de género para preencher o resto da playlist
    if len(final) < max_results:
        faltam = max_results - len(final)
        for tol_extra in (0.60, 0.80, 1.0):
            if faltam <= 0:
                break
            extras = select_tracks_from_dataset(
                [],  # empty genres => no genre filter
                TARGET_EMOTION_MOOD[1][0],
                TARGET_EMOTION_MOOD[1][1],
                tol=tol_extra,
                max_results=faltam,
            )
            for uri in extras:
                if uri not in seen:
                    seen.add(uri)
                    final.append(uri)
                    faltam -= 1
                    if faltam <= 0:
                        break
    
    return final[:max_results]

@st.cache_data(ttl=20)
def obter_modelos_ollama():
    try:
        resp = requests.get('http://localhost:11434/api/tags', timeout=3)
        if resp.status_code != 200:
            return []
        modelos = resp.json().get('models', [])
        return [m.get('name') for m in modelos if m.get('name')]
    except Exception:
        return []

# Função auxiliar para extrair o ID da playlist de uma URL
def extract_playlist_id(url_or_id: str) -> str:
    if not url_or_id:
        return ""
    if re.fullmatch(r"[A-Za-z0-9_-]{22,}", url_or_id):
        return url_or_id
    m = re.search(r"/playlist/([A-Za-z0-9_-]+)", url_or_id)
    if m:
        return m.group(1)
    return url_or_id

# --- OLLAMA FEW-SHOT ---
def gerar_resposta_empatica(texto_utilizador, emocao_detetada, modelo_ollama):
    prompt = f"""
    You are an empathetic AI DJ assistant named Notify. Your goal is to validate the user's emotions in a comforting, natural, and very concise way (maximum 2 short sentences).
    
    Strict Rules:
    1. NEVER give medical, psychological, or life advice.
    2. ALWAYS finish by saying you have prepared a personalized playlist.
    3. Respond strictly in English.
    
    Examples of Interaction:
    User: 'I finally got that promotion at work!' | Emotion: Joy
    Response: That is excellent news, congratulations on your hard-earned achievement! I've prepared an upbeat playlist for you to celebrate.
    
    User: 'My dog passed away today and I feel so empty.' | Emotion: Sadness
    Response: I am so sorry for your loss, I can only imagine the pain you're going through. I've prepared a calming playlist to keep you company during this difficult time.
    
    User: 'I am so sick and tired of my roommate leaving their trash everywhere!' | Emotion: Anger
    Response: It is completely understandable that you are feeling frustrated with that situation. I've prepared an intense playlist to help you release some of that tension.
    
    Now it is your turn:
    User: '{texto_utilizador}' | Emotion: {emocao_detetada}
    Response:
    """
    try:
        resposta = requests.post(
            'http://localhost:11434/api/generate',
            json={"model": modelo_ollama, "prompt": prompt, "stream": False},
            timeout=20,
        )
        if resposta.status_code == 200:
            body = resposta.json()
            if body.get('response'):
                return body['response'], True
        return "I'm here for you. I'll continue to help you with the right music.", False
    except Exception:
        return "I'm here for you. I'll continue to help you with the right music.", False

# --- GESTÃO DE SESSÃO E HISTÓRICO DE CHAT ---
if "autenticado" not in st.session_state: st.session_state.autenticado = False
if "username" not in st.session_state: st.session_state.username = ""
# Criar a memória do chat para as caixinhas de texto aparecerem em sequência
if "historico_chat" not in st.session_state: st.session_state.historico_chat = []

# --- ECRÃ DE AUTENTICAÇÃO ---
if not st.session_state.autenticado:
    st.title("🔐 Notify - Autenticação")
    aba_login, aba_registo = st.tabs(["Entrar na Conta", "Criar Nova Conta"])
    utilizadores = carregar_utilizadores()
    
    with aba_login:
        user_login = st.text_input("Nome de Utilizador", key="login_user")
        pass_login = st.text_input("Palavra-passe", type="password", key="login_pass")
        if st.button("Iniciar Sessão"):
            registo = normalizar_registo_utilizador(utilizadores.get(user_login))
            if user_login in utilizadores and registo["password_hash"] == hash_password(pass_login):
                st.session_state.autenticado = True
                st.session_state.username = user_login
                st.rerun()
            else: st.error("Utilizador ou palavra-passe incorretos.")
                
    with aba_registo:
        user_registo = st.text_input("Escolhe um Nome de Utilizador", key="reg_user")
        pass_registo = st.text_input("Escolhe uma Palavra-passe", type="password", key="reg_pass")
        if st.button("Criar Conta"):
            if user_registo in utilizadores: st.error("Utilizador já existe.")
            else:
                utilizadores[user_registo] = {
                    "password_hash": hash_password(pass_registo),
                    "last_emotion": "Neutral"
                }
                guardar_utilizadores(utilizadores)
                st.success("Conta criada! Faz login ao lado.")

# --- ECRÃ PRINCIPAL---
else:
    utilizadores = carregar_utilizadores()
    registo = normalizar_registo_utilizador(utilizadores.get(st.session_state.username, {}))
    emocao_guardada = registo.get("last_emotion", "Neutral")
    aplicar_fundo_emocional(emocao_guardada)

    st.sidebar.write(f"Sessão: **{st.session_state.username}**")
    if st.sidebar.button("Reautenticar Spotify"):
        if os.path.exists(CACHE_PATH):
            os.remove(CACHE_PATH)
        st.sidebar.success("Cache OAuth limpa. No próximo pedido vais autorizar o Spotify novamente.")

    playlist_url_input = st.sidebar.text_input(
        "Playlist URL to update (optional)",
        value="https://open.spotify.com/playlist/7oLFMmr10Cq5QpfDjzodOP?si=d49e3db453e64354",
        help="If set, the app will replace this playlist's tracks instead of creating a new playlist."
    )
    modelos_ollama = obter_modelos_ollama()
    default_model = "llama3"
    if modelos_ollama:
        if default_model not in modelos_ollama:
            default_model = modelos_ollama[0]
        modelo_ollama = st.sidebar.selectbox("Modelo Ollama", options=modelos_ollama, index=modelos_ollama.index(default_model))
        st.sidebar.success("Ollama detectado.")
    else:
        modelo_ollama = "llama3"
        st.sidebar.warning("Ollama não respondeu em http://localhost:11434. Confirma com: ollama list")

    if st.sidebar.button("Terminar Sessão"):
        st.session_state.autenticado = False
        st.session_state.username = ""
        st.session_state.historico_chat = []
        st.rerun()

    st.title("🎧 Notify - Chat")
    st.markdown("Select your preferred genres and share your feelings in the input below. I'll create a personalized playlist for you on Spotify!")
    
    # Seleção de géneros
    generos_disponiveis = carregar_generos_disponiveis()
    if not generos_disponiveis:
        st.warning("Não foi possível carregar os géneros do dataset. Verifica a coluna 'track_genre'.")
        generos_disponiveis = ['acoustic']
    # 6. A Barra de Pesquisa
    escolhas_brutas = st.multiselect(
        "Available Genres:",
        options=generos_disponiveis,
        format_func=formatar_dropdown,
        default=[]
    )

    # 7. Remover os separadores caso o utilizador os selecione por engano
    generos_preferidos = [g for g in escolhas_brutas if g not in ["────────────────────── GÉNEROS SUGERIDOS ───────────────────────", "────────────────────── RESTANTES GÉNEROS ───────────────────────"]]

    # Mostrar o histórico do chat
    for msg in st.session_state.historico_chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "playlist_url" in msg and msg["playlist_url"] != "#":
                st.markdown(f"-> **Emotion detected:** {msg['emocao']}")
                # Apresenta apenas o link inline de forma limpa, sem forçar abertura de pop-ups
                st.markdown(f"🟢 [**Click here to open your Playlist with {TARGET_TRACKS} songs on Spotify**]({msg['playlist_url']})")
            if "analise" in msg:
                with st.expander("View complete emotional analysis"):
                    st.json(msg["analise"])

    # CAIXA DE INPUT ESTILO CHATBOT
    if desabafo := st.chat_input("How are you feeling today? Share your thoughts with me..."):
        if not generos_preferidos:
            st.warning("Choose at least one genre above before sending!")
        else:
            # Mostrar imediatamente o texto do utilizador no ecrã
            st.session_state.historico_chat.append({"role": "user", "content": desabafo})
            with st.chat_message("user"):
                st.write(desabafo)
                
            with st.spinner('A processar e a sintonizar o Spotify...'):
                # Analisar a emoção usando o modelo local
                inputs = tokenizer(desabafo, return_tensors="pt", truncation=True, max_length=64)
                with torch.no_grad(): outputs = modelo(**inputs)
                probabilidades = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                id_vencedor = torch.argmax(probabilidades).item()
                percentagem = probabilidades[id_vencedor].item()
                emocao_final = "Neutral" if percentagem < 0.50 else emocoes[id_vencedor]
                analise_emocoes = {
                    emocao: round(float(probabilidades[i].item()) * 100, 2)
                    for i, emocao in enumerate(emocoes)
                }
                analise_emocoes["escolhida"] = emocao_final
                analise_emocoes["confianca_escolhida_%"] = round(percentagem * 100, 2)

                aplicar_fundo_emocional(emocao_final)
                registo["last_emotion"] = emocao_final
                utilizadores[st.session_state.username] = registo
                guardar_utilizadores(utilizadores)

                # Chamar o Chatbot Ollama
                resposta_bot, ollama_ok = gerar_resposta_empatica(desabafo, emocao_final, modelo_ollama)
                if not ollama_ok:
                    st.info("Ollama indisponível/modelo inválido. A usar resposta local de fallback.")
                
                # usar dataset local por valence/energy + géneros
                modificador = plutchik_modifiers[emocao_final]
                uris_musicas = []

                target_val, target_eng = EMOTION_TO_VALENCE_ENERGY.get(emocao_final, (0.5, 0.5))

                progressao = gerar_progressao_mood(
                    emocao_final,
                    target_val,
                    target_eng,
                    num_passos=TARGET_TRACKS,
                    passos_iniciais_fixos=10,
                )

                try:
                    uris_musicas = select_tracks_progressive(generos_preferidos, progressao, max_results=TARGET_TRACKS)
                except Exception as e:
                    print(f"Erro ao selecionar do dataset local com progressão: {e}")
                
                # Fallback seguro caso não consigamos 20 músicas
                if len(uris_musicas) < TARGET_TRACKS:
                    print(f"Fallback ativado! Obtivemos {len(uris_musicas)} músicas, esperávamos {TARGET_TRACKS}")

                # Verificar quantas das músicas selecionadas correspondem efetivamente aos géneros preferidos (pode haver músicas fora do género para preencher a playlist)
                num_genre_matched = 0
                try:
                    df_local = carregar_dataset()
                    if df_local is not None and len(uris_musicas) > 0:
                        genre_col = next((c for c in df_local.columns if 'genre' in c.lower()), None)
                        id_col = next((c for c in df_local.columns if c.lower() in ('track_id','id','spotify_id')), None)
                        uri_col = next((c for c in df_local.columns if 'uri' in c.lower() or 'track_uri' in c.lower()), None)
                        # Criar um mapping de ID/URI para género para verificar os géneros das músicas selecionadas
                        mapping = {}
                        if id_col and genre_col:
                            tmp = df_local[[id_col, genre_col]].dropna()
                            for _, r in tmp.iterrows():
                                mapping[str(r[id_col])] = str(r[genre_col]).strip().lower()
                        elif uri_col and genre_col:
                            tmp = df_local[[uri_col, genre_col]].dropna()
                            for _, r in tmp.iterrows():
                                mapping[str(r[uri_col])] = str(r[genre_col]).strip().lower()

                        wanted_lower = {g.lower() for g in generos_preferidos}
                        for u in uris_musicas:
                            uid = None
                            if u.startswith('spotify:track:'):
                                uid = u.split(':')[-1]
                                gen = mapping.get(uid) or mapping.get(u)
                            else:
                                gen = mapping.get(u)
                            if gen and gen in wanted_lower:
                                num_genre_matched += 1
                except Exception:
                    num_genre_matched = 0

                # CRIAR OU ATUALIZAR A PLAYLIST REAL NA CONTA DO SPOTIFY
                url_playlist = "#"
                playlist_ok = False
                if uris_musicas:
                    try:
                        # Verificar o ID do utilizador autenticado para criar a playlist
                        try:
                            current_user = sp.current_user()
                            user_id = current_user.get('id')
                        except Exception as e:
                            user_id = None
                            print("Spotify auth/current_user failed:", e)

                        used_existing = False
                        # Se tiver uma URL de playlist, tentamos atualizar essa playlist (substituir as faixas)
                        if playlist_url_input and playlist_url_input.strip():
                            pid = extract_playlist_id(playlist_url_input.strip())
                            try:
                                sp.playlist_replace_items(playlist_id=pid, items=uris_musicas)
                                used_existing = True
                                playlist_ok = True
                                url_playlist = playlist_url_input.strip()
                            except Exception as e:
                                print(f"Erro ao atualizar playlist {pid}: {e}")

                        # Se não tem uma playlist existente, ou se falhou a atualização, tenta criar uma nova playlist
                        if not used_existing:
                            if user_id:
                                try:
                                    nova_playlist = sp.user_playlist_create(
                                        user=user_id,
                                        name=f"Notify: Mood {emocao_final}",
                                        public=True,
                                        description=f"Playlist de 20 músicas gerada por IA. Géneros: {', '.join(generos_preferidos)}."
                                    )
                                    sp.playlist_add_items(playlist_id=nova_playlist['id'], items=uris_musicas)
                                    url_playlist = nova_playlist['external_urls']['spotify']
                                    playlist_ok = True
                                except Exception as e:
                                    print(f"ERRO DO SPOTIFY NA CRIAÇÃO: {e}")
                                    resposta_bot += " (Nota: Não consegui criar a playlist física no Spotify)."
                            else:
                                print("Não autenticado no Spotify: não é possível criar playlist nova.")
                                resposta_bot += " (Nota: Não estou autenticado no Spotify; não consegui criar ou atualizar a playlist)."

                    except Exception as e:
                        print(f"Erro geral do Spotify ao manipular playlist: {e}")
                        resposta_bot += f" (Erro do Spotify: {e})"

                # Guardar no histórico e atualizar o ecrã
                if playlist_ok:
                    estado_playlist = f"I updated the playlist with {len(uris_musicas)} tracks combining: {', '.join(generos_preferidos)}."
                    if num_genre_matched < len(uris_musicas):
                        estado_playlist += f" (Note: {len(uris_musicas)-num_genre_matched} tracks were added outside the selected genres to complete {TARGET_TRACKS} tracks)."
                else:
                    estado_playlist = "I couldn't create or update the Spotify playlist. Click in the reauthenticate button in the sidebar and try again, or check the console for errors."

                conteudo_bot = f"{resposta_bot}\n\n{estado_playlist}"
                st.session_state.historico_chat.append({
                    "role": "assistant", 
                    "content": conteudo_bot,
                    "emocao": emocao_final,
                    "playlist_url": url_playlist,
                    "analise": analise_emocoes
                })
                st.rerun()