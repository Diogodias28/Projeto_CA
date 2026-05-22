import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("A ligar aos servidores do Spotify...")

caminho_do_modelo = "./modelo"
tokenizer = AutoTokenizer.from_pretrained(caminho_do_modelo)
modelo = AutoModelForSequenceClassification.from_pretrained(caminho_do_modelo)

# --- CONFIGURAÇÃO DO SPOTIFY ---
CLIENT_ID = "4ddae513ee4c4982b7f8615380c219ce"
CLIENT_SECRET = "c617602e6a4342fa99933a4f3a3e45e1"

credenciais = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=credenciais)

print("Ligação estabelecida com sucesso!")

# A nossa lista de emoções na ordem exata em que o modelo as aprendeu
emocoes = ['Joy', 'Sadness', 'Anger', 'Fear', 'Trust', 'Disgust', 'Surprise', 'Anticipation']

# --- MAPA HÍBRIDO (PLUTCHIK -> DIMENSIONAL) ---
plutchik_to_spotify = {
    "Joy": {"valence": 0.8, "energy": 0.8, "genres": ["pop", "dance", "happy"]},
    "Sadness": {"valence": 0.2, "energy": 0.2, "genres": ["acoustic", "piano", "sad"]},
    "Anger": {"valence": 0.2, "energy": 0.9, "genres": ["metal", "hard-rock", "punk"]},
    "Trust": {"valence": 0.7, "energy": 0.3, "genres": ["chill", "ambient", "indie"]},
    "Fear": {"valence": 0.3, "energy": 0.7, "genres": ["electronic", "industrial", "dark"]},
    "Anticipation": {"valence": 0.5, "energy": 0.7, "genres": ["synthwave", "techno", "trance"]},
    "Disgust": {"valence": 0.3, "energy": 0.5, "genres": ["grunge", "alternative", "blues"]},
    "Surprise": {"valence": 0.5, "energy": 0.8, "genres": ["edm", "electro"]},
    "Neutral": {"valence": 0.5, "energy": 0.5, "genres": ["pop", "indie-pop", "chill"]}
}

def analisar_desabafo(texto_do_utilizador):
    print(f"\nA analisar: '{texto_do_utilizador}' ")
    
    # Transformar o texto em tokens
    inputs = tokenizer(texto_do_utilizador, return_tensors="pt", truncation=True, max_length=64)
    
    with torch.no_grad():
        outputs = modelo(**inputs)
    
    probabilidades = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    
    # Descobrir qual é a emoção vencedora e a sua percentagem
    id_vencedor = torch.argmax(probabilidades).item()
    percentagem_confianca = probabilidades[id_vencedor].item()
    emocao_prevista = emocoes[id_vencedor]
    
    # Neutro quando a confiança for baixa (menos de 50%)
    if percentagem_confianca < 0.50:
        print(f"IA com {percentagem_confianca*100:.1f}% de certeza). Assumimos NEUTRO.")
        emocao_final = "Neutral"
    else:
        emocao_final = emocao_prevista
        print(f"Emoção detetada: {emocao_final} (Confiança: {percentagem_confianca*100:.1f}%)")

    # Traduzir para o Spotify
    spotify_coords = plutchik_to_spotify[emocao_final]
    print(f"Instruções para o Spotify: Valência {spotify_coords['valence']} | Energia {spotify_coords['energy']}")

    print("\nA gerar a playlist...")
    resultados = sp.recommendations(
        seed_genres=spotify_coords['genres'],
        target_valence=spotify_coords['valence'],
        target_energy=spotify_coords['energy'],
        limit=15
    )

    for i, track in enumerate(resultados['tracks']):
        nome_musica = track['name']
        artista = track['artists'][0]['name']
        link = track['external_urls']['spotify']
        print(f"{i+1}. {nome_musica} - {artista} | Ouvir: {link}")
    
    print("-" * 50)
    return emocao_final, spotify_coords